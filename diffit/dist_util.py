"""
Helpers for distributed training using PyTorch DDP.
"""

import os
import socket
import datetime

import builtins
import torch as th
import torch.distributed as dist
from safetensors.torch import load_file as safe_load_file


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def synchronize():
    if not is_dist_avail_and_initialized():
        return
    if dist.get_world_size() == 1:
        return
    dist.barrier()


def setup_for_distributed(is_master):
    """Disable printing when not in master process."""
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print(f"[{now}] ", end="")
            builtin_print(*args, **kwargs)

    builtins.print = print


def setup_dist():
    """
    Initialize a distributed process group using environment variables
    (MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK, LOCAL_RANK).
    Works with torchrun, srun, or manual env-var setup.
    """
    if dist.is_initialized():
        return

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    th.cuda.set_device(local_rank)

    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        setup_for_distributed(rank == 0)
        synchronize()

    print(
        f"[dist] rank={rank}, local_rank={local_rank}, "
        f"world_size={world_size}, device=cuda:{local_rank}",
        force=True,
    )


def dev():
    """Get the device for the current process."""
    if th.cuda.is_available():
        return th.device("cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load either a PyTorch checkpoint (.pt/.pth) or a SafeTensors file.
    """
    _, ext = os.path.splitext(path)
    ext = ext.lower()

    if ext == ".safetensors":
        map_location = kwargs.pop("map_location", "cpu")
        return safe_load_file(path, device=str(map_location))

    return th.load(path, **kwargs)


def sync_params(params):
    """Synchronize a sequence of Tensors across ranks from rank 0."""
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
