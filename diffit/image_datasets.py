"""
Image dataset utilities for DiffiT training and evaluation.
Uses standard PyTorch data loading with DistributedSampler support.
"""

import math
import os
import random
import zipfile
import json

import io

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    num_workers=4,
    distributed=False,
    cache_in_ram=False,
):
    """
    Create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.

    :param data_dir: a dataset directory or zip file.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key for class labels.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    :param num_workers: number of DataLoader workers.
    :param distributed: if True, use DistributedSampler for multi-GPU.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    if data_dir.endswith(".zip"):
        dataset = ZipImageDataset(
            data_dir,
            image_size,
            class_cond=class_cond,
            random_crop=random_crop,
            random_flip=random_flip,
            cache_in_ram=cache_in_ram,
        )
    else:
        all_files = _list_image_files_recursively(data_dir)
        classes = None
        if class_cond:
            class_names = [os.path.basename(path).split("_")[0] for path in all_files]
            sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
            classes = [sorted_classes[x] for x in class_names]
        dataset = ImageDataset(
            image_size,
            all_files,
            classes=classes,
            random_crop=random_crop,
            random_flip=random_flip,
            cache_in_ram=cache_in_ram,
        )

    sampler = None
    shuffle = not deterministic
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=not deterministic)
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    while True:
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(getattr(loader, "_epoch", 0))
        yield from loader


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


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        random_crop=False,
        random_flip=True,
        cache_in_ram=False,
    ):
        super().__init__()
        self.resolution = resolution
        self.image_paths = image_paths
        self.classes = classes
        self.random_crop = random_crop
        self.random_flip = random_flip
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
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.classes is not None:
            out_dict["y"] = np.array(self.classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


class ZipImageDataset(Dataset):
    """Dataset that reads images from a zip archive (StyleGAN-XL format)."""

    def __init__(
        self,
        zip_path,
        resolution,
        class_cond=False,
        random_crop=False,
        random_flip=True,
        cache_in_ram=False,
    ):
        super().__init__()
        self.zip_path = zip_path
        self.resolution = resolution
        self.class_cond = class_cond
        self.random_crop = random_crop
        self.random_flip = random_flip

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
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.class_cond and fname in self._labels:
            out_dict["y"] = np.array(self._labels[fname], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


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
