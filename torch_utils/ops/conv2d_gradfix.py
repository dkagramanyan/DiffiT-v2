# Copyright (c) 2024, DiffiT authors.
# Custom replacements for a few low-level PyTorch ops.

from __future__ import annotations

import contextlib
import warnings

import torch

enabled = True
weight_gradients_disabled = False


@contextlib.contextmanager
def no_weight_gradients(disable: bool = True):
    """Context manager to temporarily disable weight gradients."""
    global weight_gradients_disabled
    old = weight_gradients_disabled
    if disable:
        weight_gradients_disabled = True
    yield
    weight_gradients_disabled = old


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Wrapper for torch.nn.functional.conv2d with custom gradient handling."""
    if _should_use_custom_op(input):
        return _conv2d_gradfix(
            transpose=False,
            weight_shape=weight.shape,
            stride=stride,
            padding=padding,
            output_padding=0,
            dilation=dilation,
            groups=groups,
        ).apply(input, weight, bias)
    return torch.nn.functional.conv2d(
        input=input, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups
    )


def conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    """Wrapper for torch.nn.functional.conv_transpose2d with custom gradient handling."""
    if _should_use_custom_op(input):
        return _conv2d_gradfix(
            transpose=True,
            weight_shape=weight.shape,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
        ).apply(input, weight, bias)
    return torch.nn.functional.conv_transpose2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    )


def _should_use_custom_op(input):
    """Check if custom op should be used."""
    return enabled and input.device.type == "cuda"


_conv2d_gradfix_cache = dict()


def _conv2d_gradfix(transpose, weight_shape, stride, padding, output_padding, dilation, groups):
    """Get cached custom gradient function for conv2d."""
    ndim = 2
    weight_shape = tuple(weight_shape)
    stride = (stride,) * ndim if isinstance(stride, int) else tuple(stride)
    padding = (padding,) * ndim if isinstance(padding, int) else tuple(padding)
    output_padding = (output_padding,) * ndim if isinstance(output_padding, int) else tuple(output_padding)
    dilation = (dilation,) * ndim if isinstance(dilation, int) else tuple(dilation)

    key = (transpose, weight_shape, stride, padding, output_padding, dilation, groups)
    if key in _conv2d_gradfix_cache:
        return _conv2d_gradfix_cache[key]

    common_kwargs = dict(stride=stride, padding=padding, dilation=dilation, groups=groups)

    def calc_output_padding(input_shape, output_shape):
        expected = [
            (i - 1) * s - 2 * p + d * (w - 1) + op + 1
            for i, s, p, d, w, op in zip(input_shape, stride, padding, dilation, weight_shape[2:], output_padding)
        ]
        assert all(e == o for e, o in zip(expected, output_shape))
        return [0] * ndim

    class Conv2d(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias):
            assert weight.shape == weight_shape
            ctx.save_for_backward(
                input if weight.requires_grad else torch.tensor([]),
                weight if input.requires_grad else torch.tensor([]),
            )
            ctx.input_shape = input.shape
            ctx.weight_shape = weight_shape
            ctx.bias_size = bias.shape[0] if bias is not None else 0

            if transpose:
                return torch.nn.functional.conv_transpose2d(
                    input=input, weight=weight, bias=bias, output_padding=output_padding, **common_kwargs
                )
            else:
                return torch.nn.functional.conv2d(input=input, weight=weight, bias=bias, **common_kwargs)

        @staticmethod
        def backward(ctx, grad_output):
            input, weight = ctx.saved_tensors
            input_shape = ctx.input_shape
            weight_shape = ctx.weight_shape
            grad_input = None
            grad_weight = None
            grad_bias = None

            if ctx.needs_input_grad[0]:
                p = calc_output_padding(input_shape=input_shape[2:], output_shape=grad_output.shape[2:])
                op = calc_output_padding(input_shape=grad_output.shape[2:], output_shape=input_shape[2:])
                if transpose:
                    grad_input = torch.nn.functional.conv2d(input=grad_output, weight=weight, bias=None, **common_kwargs)
                else:
                    grad_input = torch.nn.functional.conv_transpose2d(
                        input=grad_output, weight=weight, bias=None, output_padding=op, **common_kwargs
                    )

            if ctx.needs_input_grad[1] and not weight_gradients_disabled:
                grad_weight = Conv2dGradWeight.apply(grad_output, input)

            if ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum([0, 2, 3])

            return grad_input, grad_weight, grad_bias

    class Conv2dGradWeight(torch.autograd.Function):
        @staticmethod
        def forward(ctx, grad_output, input):
            ctx.save_for_backward(
                grad_output if input.requires_grad else torch.tensor([]),
                input if grad_output.requires_grad else torch.tensor([]),
            )
            ctx.grad_output_shape = grad_output.shape
            ctx.input_shape = input.shape

            op = calc_output_padding(input_shape=input.shape[2:], output_shape=grad_output.shape[2:])
            if transpose:
                grad_weight = torch.nn.functional.conv2d(
                    input=input.transpose(0, 1),
                    weight=grad_output.transpose(0, 1),
                    bias=None,
                    stride=dilation,
                    padding=padding,
                    dilation=stride,
                    groups=groups,
                )
                grad_weight = grad_weight.transpose(0, 1)
            else:
                grad_weight = torch.nn.functional.conv2d(
                    input=grad_output.transpose(0, 1),
                    weight=input.transpose(0, 1),
                    bias=None,
                    stride=dilation,
                    padding=padding,
                    dilation=stride,
                    groups=groups,
                )
                grad_weight = grad_weight.transpose(0, 1)
            return grad_weight

        @staticmethod
        def backward(ctx, grad2_grad_weight):
            grad_output, input = ctx.saved_tensors
            grad_output_shape = ctx.grad_output_shape
            input_shape = ctx.input_shape
            grad2_grad_output = None
            grad2_input = None

            if ctx.needs_input_grad[0]:
                if transpose:
                    grad2_grad_output = torch.nn.functional.conv2d(
                        input=input, weight=grad2_grad_weight, bias=None, **common_kwargs
                    )
                else:
                    p = calc_output_padding(input_shape=input_shape[2:], output_shape=grad_output_shape[2:])
                    grad2_grad_output = torch.nn.functional.conv_transpose2d(
                        input=input, weight=grad2_grad_weight, bias=None, output_padding=p, **common_kwargs
                    )

            if ctx.needs_input_grad[1]:
                p = calc_output_padding(input_shape=input_shape[2:], output_shape=grad_output_shape[2:])
                if transpose:
                    grad2_input = torch.nn.functional.conv_transpose2d(
                        input=grad_output, weight=grad2_grad_weight, bias=None, output_padding=p, **common_kwargs
                    )
                else:
                    grad2_input = torch.nn.functional.conv2d(
                        input=grad_output, weight=grad2_grad_weight, bias=None, **common_kwargs
                    )

            return grad2_grad_output, grad2_input

    _conv2d_gradfix_cache[key] = Conv2d
    return Conv2d
