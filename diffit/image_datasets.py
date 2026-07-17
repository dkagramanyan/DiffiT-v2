"""
Image dataset utilities for DiffiT training and evaluation.
Uses standard PyTorch data loading with DistributedSampler support.

Dataset item contract (v2 convention, §5): every ``__getitem__`` yields a
``uint8`` CHW image at the zip's resolution and, when class-conditional, a
one-hot ``float32`` label. All normalization (uint8 → ``[-1, 1]``) and the
one-hot → index reduction happen in the training loop, never inside the dataset
class. Images are asserted to be 3-channel RGB — grayscale→RGB conversion is a
build-time step (``diffit-prepare-data``), not a silent per-item convert.
"""

import io
import json
import math
import os
import random
import zipfile

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler


def read_class_meta(data_dir):
    """Return ``(class_names, num_classes)`` for a dataset directory or .zip.

    ``class_names`` is the index-aligned list of grain-class names (§5 Rule 2)
    when the source records them, else ``None``. For zips this is the
    ``dataset.json`` ``class_names`` field; a legacy zip without it falls back to
    ``max(label) + 1`` for the count and ``None`` names. For directory datasets
    the classes are the sorted (alphabetical) set of immediate parent folder
    names (§5 Rule 1).
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    if data_dir.endswith(".zip"):
        with zipfile.ZipFile(data_dir, "r") as zf:
            names = set(zf.namelist())
            if "dataset.json" not in names:
                return None, 1
            with zf.open("dataset.json") as f:
                meta = json.loads(f.read().decode("utf-8"))
        class_names = meta.get("class_names")
        labels = meta.get("labels")
        if class_names is not None:
            return list(class_names), len(class_names)
        if labels:
            num = max(int(lbl) for _, lbl in labels) + 1
            return None, num
        return None, 1

    # Directory dataset: class = immediate parent folder name, alphabetical.
    all_files = _list_image_files_recursively(data_dir)
    class_names = sorted({os.path.basename(os.path.dirname(p)) for p in all_files})
    return class_names, max(len(class_names), 1)


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    num_classes,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    mirror=False,
    num_workers=4,
    distributed=False,
    cache_in_ram=False,
    drop_last=True,
    seed=0,
):
    """
    Create a generator over (images, kwargs) pairs.

    Each ``images`` is an NCHW ``uint8`` tensor; the kwargs dict contains a
    ``"y"`` key mapping to a batched one-hot ``float32`` label tensor when
    ``class_cond`` is set. Normalization and the one-hot → index reduction are
    the caller's responsibility (done in the training loop).

    :param data_dir: a dataset directory or zip file.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param num_classes: label width for the one-hot encoding.
    :param class_cond: if True, include a "y" key for class labels.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param mirror: if True, apply a stochastic per-item horizontal flip
        (the §2 loader-level augmentation). Eval/reference loaders pass False.
    :param num_workers: number of DataLoader workers.
    :param distributed: if True, use DistributedSampler for multi-GPU.
    :param seed: base seed for the DistributedSampler shuffle (§2).
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    if data_dir.endswith(".zip"):
        dataset = ZipImageDataset(
            data_dir,
            image_size,
            num_classes=num_classes,
            class_cond=class_cond,
            random_crop=random_crop,
            mirror=mirror,
            cache_in_ram=cache_in_ram,
        )
    else:
        all_files = _list_image_files_recursively(data_dir)
        classes = None
        if class_cond:
            # Class identity is the immediate parent folder name, indexed by the
            # alphabetical sort of the folder-name set (§5 Rule 1).
            class_names = [os.path.basename(os.path.dirname(path)) for path in all_files]
            sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
            classes = [sorted_classes[x] for x in class_names]
        dataset = ImageDataset(
            image_size,
            all_files,
            num_classes=num_classes,
            classes=classes,
            random_crop=random_crop,
            mirror=mirror,
            cache_in_ram=cache_in_ram,
        )

    sampler = None
    shuffle = not deterministic
    if distributed:
        # Seed the sampler from --seed so multi-GPU data order is reproducible
        # and controlled by the run seed (§2).
        sampler = DistributedSampler(dataset, shuffle=not deterministic, seed=seed)
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )
    # Advance the DistributedSampler epoch each pass so every rank gets a fresh
    # shuffle. The sampler's per-epoch generator is seeded with (seed + epoch),
    # so --seed fully determines the multi-GPU data order.
    epoch = 0
    while True:
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif os.path.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


def count_data(data_dir):
    """Return the number of images in a dataset directory or .zip archive."""
    if not data_dir:
        raise ValueError("unspecified data directory")
    if data_dir.endswith(".zip"):
        with zipfile.ZipFile(data_dir, "r") as zf:
            return sum(
                1 for name in zf.namelist()
                if name.split(".")[-1].lower() in ("png", "jpg", "jpeg", "gif")
            )
    return len(_list_image_files_recursively(data_dir))


def _prepare_arr(pil_image, resolution, random_crop, mirror):
    """Decode a PIL image to a ``uint8`` CHW array at ``resolution``.

    Asserts 3-channel RGB rather than silently converting: the pipeline is
    3-channel end to end and grayscale→RGB conversion happens once at dataset
    build time (§5).
    """
    if pil_image.mode != "RGB":
        raise ValueError(
            f"expected a 3-channel RGB image, got mode {pil_image.mode!r}; "
            "convert grayscale sources to RGB at dataset build time "
            "(diffit-prepare-data), not at load time"
        )
    if random_crop:
        arr = random_crop_arr(pil_image, resolution)
    else:
        arr = center_crop_arr(pil_image, resolution)
    if mirror and random.random() < 0.5:
        arr = arr[:, ::-1]
    assert arr.ndim == 3 and arr.shape[2] == 3, f"expected HWC RGB, got {arr.shape}"
    # uint8 CHW; normalization is done in the training loop.
    return np.ascontiguousarray(np.transpose(arr, [2, 0, 1])).astype(np.uint8)


def _onehot(idx, num_classes):
    v = np.zeros(num_classes, dtype=np.float32)
    v[int(idx)] = 1.0
    return v


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        num_classes,
        classes=None,
        random_crop=False,
        mirror=False,
        cache_in_ram=False,
    ):
        super().__init__()
        self.resolution = resolution
        self.image_paths = image_paths
        self.num_classes = num_classes
        self.classes = classes
        self.random_crop = random_crop
        self.mirror = mirror
        self._cache = None

        if cache_in_ram:
            print(f"Caching {len(image_paths)} images in RAM...")
            self._cache = []
            for i, path in enumerate(image_paths):
                with open(path, "rb") as f:
                    self._cache.append(f.read())
                if (i + 1) % 10000 == 0:
                    print(f"  cached {i + 1}/{len(image_paths)} images")
            print(f"All {len(image_paths)} images cached in RAM.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self._cache is not None:
            pil_image = Image.open(io.BytesIO(self._cache[idx]))
            pil_image.load()
        else:
            path = self.image_paths[idx]
            with open(path, "rb") as f:
                pil_image = Image.open(f)
                pil_image.load()

        img = _prepare_arr(pil_image, self.resolution, self.random_crop, self.mirror)

        out_dict = {}
        if self.classes is not None:
            out_dict["y"] = _onehot(self.classes[idx], self.num_classes)
        return img, out_dict


class ZipImageDataset(Dataset):
    """Dataset that reads images from a zip archive (StyleGAN-XL format)."""

    def __init__(
        self,
        zip_path,
        resolution,
        num_classes,
        class_cond=False,
        random_crop=False,
        mirror=False,
        cache_in_ram=False,
    ):
        super().__init__()
        self.zip_path = zip_path
        self.resolution = resolution
        self.num_classes = num_classes
        self.class_cond = class_cond
        self.random_crop = random_crop
        self.mirror = mirror

        self._zipfile = None
        self._image_fnames = []
        self._labels = {}
        self._cache = None

        with zipfile.ZipFile(zip_path, "r") as zf:
            self._all_names = set(zf.namelist())
            for name in sorted(self._all_names):
                ext = name.split(".")[-1].lower()
                if ext in ("png", "jpg", "jpeg", "gif"):
                    self._image_fnames.append(name)

            if "dataset.json" in self._all_names:
                with zf.open("dataset.json") as f:
                    meta = json.loads(f.read().decode("utf-8"))
                if meta.get("labels") is not None:
                    self._labels = {
                        fname: label for fname, label in meta["labels"]
                    }

            if cache_in_ram:
                print(f"Caching {len(self._image_fnames)} images in RAM from {zip_path}...")
                self._cache = {}
                for i, fname in enumerate(self._image_fnames):
                    with zf.open(fname) as f:
                        self._cache[fname] = f.read()
                    if (i + 1) % 10000 == 0:
                        print(f"  cached {i + 1}/{len(self._image_fnames)} images")
                print(f"All {len(self._image_fnames)} images cached in RAM.")

    def _open_zip(self):
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self.zip_path, "r")
        return self._zipfile

    def __len__(self):
        return len(self._image_fnames)

    def __getitem__(self, idx):
        fname = self._image_fnames[idx]
        if self._cache is not None:
            pil_image = Image.open(io.BytesIO(self._cache[fname]))
            pil_image.load()
        else:
            zf = self._open_zip()
            with zf.open(fname) as f:
                pil_image = Image.open(f)
                pil_image.load()

        img = _prepare_arr(pil_image, self.resolution, self.random_crop, self.mirror)

        out_dict = {}
        if self.class_cond and fname in self._labels:
            out_dict["y"] = _onehot(self._labels[fname], self.num_classes)
        return img, out_dict


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
