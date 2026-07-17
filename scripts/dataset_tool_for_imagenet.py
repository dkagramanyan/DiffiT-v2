"""
Dataset builder for DiffiT training (``diffit-prepare-data``).

Reads an ImageNet-style directory structure (or any ``train/<class>/`` layout)
and writes a StyleGAN-style ZIP archive of resized/cropped RGB PNGs plus a
``dataset.json`` carrying both the integer ``labels`` and the index-aligned
``class_names`` (§5 label contract). Grayscale sources are converted to RGB
here, at build time — the dataset loader asserts 3 channels instead of
converting silently.

``diffit-prepare-data`` is a click *group*; today it exposes a single
``convert`` subcommand (mirroring the EDM2-v2 shape), sharing the transform set
``center-crop`` / ``center-crop-wide`` / ``center-crop-dhariwal``.

Usage:
    diffit-prepare-data convert \
        --source /path/to/ILSVRC \
        --dest ./datasets/imagenet_256x256.zip \
        --resolution 256x256 \
        --transform center-crop
"""

import functools
import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import click
import numpy as np
import PIL.Image
from tqdm import tqdm


def error(msg):
    print("Error: " + msg)
    sys.exit(1)


def parse_tuple(s: str) -> Tuple[int, int]:
    """Parse a 'M,N' or 'MxN' integer tuple.

    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    """
    m = re.match(r"^(\d+)[x,](\d+)$", s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f"cannot parse tuple {s}")


def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a


def file_ext(name: Union[str, Path]) -> str:
    return str(name).split(".")[-1]


def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f".{ext}" in PIL.Image.EXTENSION


def open_imagenet(source_dir, *, max_images: Optional[int]):
    """Open an ImageNet-style directory and yield RGB images with labels.

    Returns ``(max_idx, class_names, iterator)``. The integer label is the index
    of the class folder in alphabetical (``sorted``) order (§5 Rule 1), and
    ``class_names`` is that same sorted folder-name list, index-aligned. Every
    image is decoded and converted to 3-channel RGB here so the archive is RGB
    end to end.
    """
    # Look for standard ImageNet layout: Data/CLS-LOC/train/nXXXXXXXX/
    train_dirs = sorted(Path(source_dir).rglob("Data/CLS-LOC/train/*"))
    if not train_dirs:
        # Fallback: look for direct subdirectories (e.g., train/n01440764/)
        train_root = Path(source_dir)
        if (train_root / "train").exists():
            train_root = train_root / "train"
        train_dirs = sorted([d for d in train_root.iterdir() if d.is_dir()])

    class_names = [d.name for d in train_dirs]

    images = []
    labels = []
    for idx, input_dir in enumerate(train_dirs):
        input_images = sorted([
            str(f) for f in Path(input_dir).rglob("*")
            if is_image_ext(f) and f.is_file()
        ])
        images.extend(input_images)
        labels.extend([idx] * len(input_images))

    max_idx = maybe_min(len(images), max_images)

    def iterate_images():
        for idx, fname in enumerate(images):
            if idx >= max_idx:
                break
            # Build-time grayscale/CMYK → RGB conversion (§5): the archive is
            # 3-channel and the loader never converts.
            img = PIL.Image.open(fname).convert("RGB")
            yield dict(img=np.array(img), label=labels[idx])

    return max_idx, class_names, iterate_images()


def make_transform(
    transform: Optional[str],
    output_width: Optional[int],
    output_height: Optional[int],
) -> Callable[[np.ndarray], Optional[np.ndarray]]:

    def scale(width, height, img):
        w = img.shape[1]
        h = img.shape[0]
        if width == w and height == h:
            return img
        img = PIL.Image.fromarray(img)
        ww = width if width is not None else w
        hh = height if height is not None else h
        img = img.resize((ww, hh), PIL.Image.LANCZOS)
        return np.array(img)

    def center_crop(width, height, img):
        crop = np.min(img.shape[:2])
        img = img[
            (img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2,
            (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2,
        ]
        img = PIL.Image.fromarray(img, "RGB")
        img = img.resize((width, height), PIL.Image.LANCZOS)
        return np.array(img)

    def center_crop_wide(width, height, img):
        ch = int(np.round(width * img.shape[0] / img.shape[1]))
        if img.shape[1] < width or ch < height:
            return None
        img = img[(img.shape[0] - ch) // 2 : (img.shape[0] + ch) // 2]
        img = PIL.Image.fromarray(img, "RGB")
        img = img.resize((width, height), PIL.Image.LANCZOS)
        img = np.array(img)
        canvas = np.zeros([width, width, 3], dtype=np.uint8)
        canvas[(width - height) // 2 : (width + height) // 2, :] = img
        return canvas

    def center_crop_dhariwal(width, height, img):
        """ADM/Dhariwal center crop: iterative BOX halving then a single BICUBIC
        resize of the short side, then a square center crop. Requires a square
        output (width == height)."""
        if width != height:
            error("center-crop-dhariwal requires a square --resolution (W == H)")
        size = width
        pil = PIL.Image.fromarray(img, "RGB")
        while min(*pil.size) >= 2 * size:
            pil = pil.resize(tuple(x // 2 for x in pil.size), resample=PIL.Image.BOX)
        scale_f = size / min(*pil.size)
        pil = pil.resize(
            tuple(round(x * scale_f) for x in pil.size), resample=PIL.Image.BICUBIC
        )
        arr = np.array(pil)
        crop_y = (arr.shape[0] - size) // 2
        crop_x = (arr.shape[1] - size) // 2
        return arr[crop_y : crop_y + size, crop_x : crop_x + size]

    if transform is None:
        return functools.partial(scale, output_width, output_height)
    crops = {
        "center-crop": center_crop,
        "center-crop-wide": center_crop_wide,
        "center-crop-dhariwal": center_crop_dhariwal,
    }
    if transform in crops:
        if output_width is None or output_height is None:
            error("must specify --resolution=WxH when using " + transform + " transform")
        return functools.partial(crops[transform], output_width, output_height)
    assert False, "unknown transform"


def open_dest(dest: str):
    dest_ext = file_ext(dest)

    if dest_ext == "zip":
        if os.path.dirname(dest) != "":
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        zf = zipfile.ZipFile(file=dest, mode="w", compression=zipfile.ZIP_STORED)

        def zip_write_bytes(fname: str, data):
            zf.writestr(fname, data)

        return "", zip_write_bytes, zf.close
    else:
        if os.path.isdir(dest) and len(os.listdir(dest)) != 0:
            error("--dest folder must be empty")
        os.makedirs(dest, exist_ok=True)

        def folder_write_bytes(fname: str, data):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, "wb") as fout:
                if isinstance(data, str):
                    data = data.encode("utf8")
                fout.write(data)

        return dest, folder_write_bytes, lambda: None


@click.group()
def prepare_data():
    """DiffiT dataset preparation tools."""


@prepare_data.command(name="convert")
@click.pass_context
@click.option("--source", required=True, type=str, help="Directory for input dataset", metavar="PATH")
@click.option("--dest", required=True, type=str, help="Output directory or archive name", metavar="PATH")
@click.option("--max-images", type=int, default=None, help="Output only up to `max-images` images")
@click.option("--transform", type=click.Choice(["center-crop", "center-crop-wide", "center-crop-dhariwal"]), help="Input crop/resize mode")
@click.option("--resolution", type=parse_tuple, help="Output resolution (e.g., '256x256')", metavar="WxH")
def convert_dataset(
    ctx,
    source: str,
    dest: str,
    max_images: Optional[int],
    transform: Optional[str],
    resolution: Optional[Tuple[int, int]],
):
    """Convert an ImageNet-style dataset into a ZIP archive usable with DiffiT.

    Class labels and names are stored in a ``dataset.json`` at the dataset root:

    \b
    {
        "labels": [["00000/img00000000.png", 1], ...],
        "class_names": ["Ultra_Co11", "Ultra_Co25", "Ultra_Co6_2"]
    }

    ``class_names`` is index-aligned with the integer labels, so grain-class
    identity travels with the archive (§5 Rule 2). Images are stored as
    uncompressed RGB PNG.
    """
    PIL.Image.init()

    if dest == "":
        ctx.fail("--dest output filename or directory must not be an empty string")

    num_files, class_names, input_iter = open_imagenet(source, max_images=max_images)
    archive_root_dir, save_bytes, close_dest_fn = open_dest(dest)

    if resolution is None:
        resolution = (None, None)
    transform_image = make_transform(transform, *resolution)

    dataset_attrs = None
    labels: List[Optional[list]] = []

    for idx, image in tqdm(enumerate(input_iter), total=num_files):
        idx_str = f"{idx:08d}"
        archive_fname = f"{idx_str[:5]}/img{idx_str}.png"

        img = transform_image(image["img"])
        if img is None:
            continue

        channels = img.shape[2] if img.ndim == 3 else 1
        cur_image_attrs = {
            "width": img.shape[1],
            "height": img.shape[0],
            "channels": channels,
        }
        if dataset_attrs is None:
            dataset_attrs = cur_image_attrs
            width = dataset_attrs["width"]
            height = dataset_attrs["height"]
            if width != height:
                error(f"Image dimensions after scale and crop are required to be square. Got {width}x{height}")
            if dataset_attrs["channels"] != 3:
                error("Input images must be stored as RGB (grayscale is converted to RGB at build time)")
            if width != 2 ** int(np.floor(np.log2(width))):
                error("Image width/height after scale and crop are required to be power-of-two")
        elif dataset_attrs != cur_image_attrs:
            err = [
                f"  dataset {k}/cur image {k}: {dataset_attrs[k]}/{cur_image_attrs[k]}"
                for k in dataset_attrs.keys()
            ]
            error(f"Image {archive_fname} attributes must be equal across all images of the dataset.  Got:\n" + "\n".join(err))

        img = PIL.Image.fromarray(img, "RGB")
        image_bits = io.BytesIO()
        img.save(image_bits, format="png", compress_level=0, optimize=False)
        save_bytes(os.path.join(archive_root_dir, archive_fname), image_bits.getbuffer())
        labels.append([archive_fname, image["label"]] if image["label"] is not None else None)

    # Every image must carry a label — a single unlabeled image is an error, not
    # a silent drop to unconditional (§5 label contract).
    missing = [i for i, x in enumerate(labels) if x is None]
    if missing:
        error(f"{len(missing)} image(s) have no class label; refusing to write a partially-labeled dataset")

    metadata = {"labels": labels, "class_names": class_names}
    save_bytes(os.path.join(archive_root_dir, "dataset.json"), json.dumps(metadata))
    close_dest_fn()

    print(f"Done. Processed {len(labels)} images across {len(class_names)} classes.")


if __name__ == "__main__":
    prepare_data()
