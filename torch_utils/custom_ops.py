# Copyright (c) 2024, DiffiT authors.
# Custom CUDA operations for DiffiT.

from __future__ import annotations

import glob
import hashlib
import importlib
import os
import re
import shutil
import uuid

import torch
import torch.utils.cpp_extension

# Global options
verbosity = "brief"  # Verbosity level: 'none', 'brief', 'full'


def _get_cuda_arch_flags() -> list[str]:
    """Return CUDA architecture flags targeting common GPUs."""
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        major, minor = 8, 0  # Default to Ampere

    # Include common architectures
    arch_flags = [
        "-gencode=arch=compute_80,code=sm_80",  # A100 (Ampere)
        "-gencode=arch=compute_90,code=sm_90",  # H100/H200 (Hopper)
        "-gencode=arch=compute_90,code=compute_90",  # PTX for forward compatibility
    ]

    # Add current device's arch explicitly if different
    current_arch = f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"
    if current_arch not in arch_flags:
        arch_flags.insert(0, current_arch)

    return arch_flags


def _find_compiler_bindir() -> str | None:
    """Find MSVC compiler binary directory on Windows."""
    patterns = [
        "C:/Program Files (x86)/Microsoft Visual Studio/*/Professional/VC/Tools/MSVC/*/bin/Hostx64/x64",
        "C:/Program Files (x86)/Microsoft Visual Studio/*/BuildTools/VC/Tools/MSVC/*/bin/Hostx64/x64",
        "C:/Program Files (x86)/Microsoft Visual Studio/*/Community/VC/Tools/MSVC/*/bin/Hostx64/x64",
        "C:/Program Files (x86)/Microsoft Visual Studio */vc/bin",
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if len(matches):
            return matches[-1]
    return None


def _get_mangled_gpu_name() -> str:
    """Get a safe string representation of GPU name for caching."""
    name = torch.cuda.get_device_name().lower()
    out = []
    for c in name:
        if re.match("[a-z0-9_-]+", c):
            out.append(c)
        else:
            out.append("-")
    return "".join(out)


# Plugin cache
_cached_plugins = dict()


def get_plugin(
    module_name: str,
    sources: list[str],
    headers: list[str] | None = None,
    source_dir: str | None = None,
    **build_kwargs,
):
    """Compile and load a C++/CUDA plugin.

    Args:
        module_name: Name of the module to compile.
        sources: List of source file names.
        headers: List of header file names (optional).
        source_dir: Directory containing the source files (optional).
        **build_kwargs: Additional arguments passed to torch.utils.cpp_extension.load().

    Returns:
        The compiled and loaded module.
    """
    import sys

    assert verbosity in ["none", "brief", "full"]
    if headers is None:
        headers = []
    if source_dir is not None:
        sources = [os.path.join(source_dir, fname) for fname in sources]
        headers = [os.path.join(source_dir, fname) for fname in headers]

    if module_name in _cached_plugins:
        return _cached_plugins[module_name]

    if verbosity == "full":
        print(f'Setting up PyTorch plugin "{module_name}"...')
    elif verbosity == "brief":
        print(f'Setting up PyTorch plugin "{module_name}"... ', end="", flush=True)
    verbose_build = verbosity == "full"

    build_dir = None

    # Inject architecture flags
    cuda_arch_flags = _get_cuda_arch_flags()
    if "extra_cuda_cflags" in build_kwargs:
        original_flags = build_kwargs["extra_cuda_cflags"]
        build_kwargs["extra_cuda_cflags"] = cuda_arch_flags + list(original_flags)
    else:
        build_kwargs["extra_cuda_cflags"] = cuda_arch_flags

    try:
        if os.name == "nt" and os.system("where cl.exe >nul 2>nul") != 0:
            compiler_bindir = _find_compiler_bindir()
            if compiler_bindir is None:
                raise RuntimeError(f'Could not find MSVC installation. Check _find_compiler_bindir() in "{__file__}".')
            os.environ["PATH"] += ";" + compiler_bindir

        all_source_files = sorted(sources + headers)
        all_source_dirs = set(os.path.dirname(fname) for fname in all_source_files)

        if len(all_source_dirs) == 1:
            hash_md5 = hashlib.md5()
            for src in all_source_files:
                with open(src, "rb") as f:
                    hash_md5.update(f.read())

            # Include architecture flags in hash
            arch_flag_str = "|".join(cuda_arch_flags)
            hash_md5.update(arch_flag_str.encode("utf-8"))

            source_digest = hash_md5.hexdigest()
            build_top_dir = torch.utils.cpp_extension._get_build_directory(module_name, verbose=verbose_build)
            cached_build_dir = os.path.join(build_top_dir, f"{source_digest}-{_get_mangled_gpu_name()}")
            build_dir = cached_build_dir

            if not os.path.isdir(cached_build_dir):
                tmpdir = f"{build_top_dir}/srctmp-{uuid.uuid4().hex}"
                os.makedirs(tmpdir)
                for src in all_source_files:
                    shutil.copyfile(src, os.path.join(tmpdir, os.path.basename(src)))
                try:
                    os.replace(tmpdir, cached_build_dir)
                except OSError:
                    shutil.rmtree(tmpdir)
                    if not os.path.isdir(cached_build_dir):
                        raise

            cached_sources = [os.path.join(cached_build_dir, os.path.basename(fname)) for fname in sources]
            torch.utils.cpp_extension.load(
                name=module_name,
                build_directory=cached_build_dir,
                verbose=verbose_build,
                sources=cached_sources,
                **build_kwargs,
            )
        else:
            build_dir = torch.utils.cpp_extension._get_build_directory(module_name, verbose=verbose_build)
            torch.utils.cpp_extension.load(
                name=module_name,
                verbose=verbose_build,
                sources=sources,
                **build_kwargs,
            )

        # Ensure the directory containing *.so is importable
        if build_dir and os.path.isdir(build_dir) and build_dir not in sys.path:
            sys.path.insert(0, build_dir)

        module = importlib.import_module(module_name)

    except Exception:
        if verbosity == "brief":
            print("Failed!")
        raise

    if verbosity == "full":
        print(f'Done setting up PyTorch plugin "{module_name}".')
    elif verbosity == "brief":
        print("Done.")

    _cached_plugins[module_name] = module
    return module
