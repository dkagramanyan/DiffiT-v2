# Copyright (c) 2024, DiffiT authors.
# Facilities for pickling Python code alongside other data.

from __future__ import annotations

import copy
import inspect
import io
import pickle
import sys
import types
import uuid

import dnnlib

_version = 6
_decorators = set()
_import_hooks = []
_module_to_src_dict = dict()
_src_to_module_dict = dict()


def persistent_class(orig_class):
    """Class decorator that extends a given class to save its source code when pickled.

    Example:
        from torch_utils import persistence

        @persistence.persistent_class
        class MyNetwork(torch.nn.Module):
            def __init__(self, num_inputs, num_outputs):
                super().__init__()
                self.fc = MyLayer(num_inputs, num_outputs)
    """
    assert isinstance(orig_class, type)
    if is_persistent(orig_class):
        return orig_class

    assert orig_class.__module__ in sys.modules
    orig_module = sys.modules[orig_class.__module__]
    orig_module_src = _module_to_src(orig_module)

    class Decorator(orig_class):
        _orig_module_src = orig_module_src
        _orig_class_name = orig_class.__name__

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._init_args = copy.deepcopy(args)
            self._init_kwargs = copy.deepcopy(kwargs)
            assert orig_class.__name__ in orig_module.__dict__
            _check_pickleable(self.__reduce__())

        @property
        def init_args(self):
            return copy.deepcopy(self._init_args)

        @property
        def init_kwargs(self):
            return dnnlib.EasyDict(copy.deepcopy(self._init_kwargs))

        def __reduce__(self):
            fields = list(super().__reduce__())
            fields += [None] * max(3 - len(fields), 0)
            if fields[0] is not _reconstruct_persistent_obj:
                meta = dict(
                    type="class",
                    version=_version,
                    module_src=self._orig_module_src,
                    class_name=self._orig_class_name,
                    state=fields[2],
                )
                fields[0] = _reconstruct_persistent_obj
                fields[1] = (meta,)
                fields[2] = None
            return tuple(fields)

    Decorator.__name__ = orig_class.__name__
    _decorators.add(Decorator)
    return Decorator


def is_persistent(obj):
    """Test whether the given object or class is persistent."""
    try:
        if obj in _decorators:
            return True
    except TypeError:
        pass
    return type(obj) in _decorators


def import_hook(hook):
    """Register an import hook that is called whenever a persistent object is being unpickled."""
    assert callable(hook)
    _import_hooks.append(hook)


def _reconstruct_persistent_obj(meta):
    """Hook called by pickle to unpickle a persistent object."""
    meta = dnnlib.EasyDict(meta)
    meta.state = dnnlib.EasyDict(meta.state)
    for hook in _import_hooks:
        meta = hook(meta)
        assert meta is not None

    assert meta.version == _version
    module = _src_to_module(meta.module_src)

    assert meta.type == "class"
    orig_class = module.__dict__[meta.class_name]
    decorator_class = persistent_class(orig_class)
    obj = decorator_class.__new__(decorator_class)

    setstate = getattr(obj, "__setstate__", None)
    if callable(setstate):
        setstate(meta.state)
    else:
        obj.__dict__.update(meta.state)
    return obj


def _module_to_src(module):
    """Query the source code of a given Python module."""
    src = _module_to_src_dict.get(module, None)
    if src is None:
        src = inspect.getsource(module)
        _module_to_src_dict[module] = src
        _src_to_module_dict[src] = module
    return src


def _src_to_module(src):
    """Get or create a Python module for the given source code."""
    module = _src_to_module_dict.get(src, None)
    if module is None:
        module_name = "_imported_module_" + uuid.uuid4().hex
        module = types.ModuleType(module_name)
        sys.modules[module_name] = module
        _module_to_src_dict[module] = src
        _src_to_module_dict[src] = module
        exec(src, module.__dict__)
    return module


def _check_pickleable(obj):
    """Check that the given object is pickleable."""
    def recurse(obj):
        if isinstance(obj, (list, tuple, set)):
            return [recurse(x) for x in obj]
        if isinstance(obj, dict):
            return [[recurse(x), recurse(y)] for x, y in obj.items()]
        if isinstance(obj, (str, int, float, bool, bytes, bytearray)):
            return None
        if f"{type(obj).__module__}.{type(obj).__name__}" in [
            "numpy.ndarray",
            "torch.Tensor",
            "torch.nn.parameter.Parameter",
        ]:
            return None
        if is_persistent(obj):
            return None
        return obj

    with io.BytesIO() as f:
        pickle.dump(recurse(obj), f)
