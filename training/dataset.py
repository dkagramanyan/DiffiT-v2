# Copyright (c) 2024, DiffiT authors.
# Streaming images and labels from datasets.
#
# Supports datasets created with dataset_tool.py:
# - Folder or ZIP archive with PNG images
# - Images in subdirectories (00000/img00000000.png)
# - Optional dataset.json with class labels
#
# Label format in dataset.json:
# {"labels": [["00000/img00000000.png", class_id], ...]}

from __future__ import annotations

import copy
import json
import os
import zipfile

import numpy as np
import PIL.Image
import torch

import dnnlib

# #region agent log
import os as _debug_os
def _debug_log(location, message, data, hypothesis_id):
    import json, time
    log_path = "/home/dgkagramanyan/.cursor/debug.log"
    entry = {"location": location, "message": message, "data": data, "hypothesisId": hypothesis_id, "timestamp": time.time(), "pid": _debug_os.getpid()}
    with open(log_path, "a") as f: f.write(json.dumps(entry) + "\n")
# #endregion

try:
    import pyspng
except ImportError:
    pyspng = None


class Dataset(torch.utils.data.Dataset):
    """Base class for datasets."""

    def __init__(
        self,
        name: str,
        raw_shape: list[int],
        max_size: int | None = None,
        use_labels: bool = False,
        xflip: bool = False,
        random_seed: int = 1,
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        self._base_raw_idx = copy.deepcopy(self._raw_idx)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self):
        """Close the dataset. Override in subclass if needed."""
        pass

    def _load_raw_image(self, raw_idx: int) -> np.ndarray:
        """Load a raw image. Override in subclass."""
        raise NotImplementedError

    def _load_raw_labels(self) -> np.ndarray | None:
        """Load raw labels. Override in subclass."""
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx: int):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3  # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx: int):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx: int):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = int(self._xflip[idx]) != 0
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3  # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3  # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)


class ImageFolderDataset(Dataset):
    """Dataset from a folder of images."""

    def __init__(
        self,
        path: str,
        resolution: int | None = None,
        **super_kwargs,
    ):
        self._path = path
        self._zipfile = None
        self._zipfile_pid = None  # Track which process created the zipfile
        # #region agent log
        _debug_log("dataset.py:__init__", "dataset_init_start", {"path": path}, "D")
        # #endregion

        if os.path.isdir(self._path):
            self._type = "dir"
            self._all_fnames = {
                os.path.relpath(os.path.join(root, fname), start=self._path)
                for root, _dirs, files in os.walk(self._path)
                for fname in files
            }
        elif self._file_ext(self._path) == ".zip":
            self._type = "zip"
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError("Path must point to a directory or zip")

        PIL.Image.init()
        self._image_fnames = sorted(
            fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION
        )
        if len(self._image_fnames) == 0:
            raise IOError("No image files found in the specified path")

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError("Image files do not match the specified resolution")
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname: str) -> str:
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == "zip"
        current_pid = os.getpid()
        # #region agent log
        _debug_log("dataset.py:_get_zipfile", "zipfile_check", {"zipfile_is_none": self._zipfile is None, "zipfile_id": id(self._zipfile) if self._zipfile else None, "zipfile_pid": self._zipfile_pid, "current_pid": current_pid}, "A")
        # #endregion
        # Re-create zipfile if accessed from a different process (forked workers)
        if self._zipfile is None or self._zipfile_pid != current_pid:
            if self._zipfile is not None:
                # #region agent log
                _debug_log("dataset.py:_get_zipfile", "reopening_for_new_process", {"old_pid": self._zipfile_pid, "new_pid": current_pid}, "A")
                # #endregion
                try:
                    self._zipfile.close()
                except Exception:
                    pass
            # Disable strict timestamp checking to avoid false "zip bomb" warnings
            # in Python 3.12+ with certain legitimate archive structures
            self._zipfile = zipfile.ZipFile(self._path, strict_timestamps=False)
            self._zipfile_pid = current_pid
            # #region agent log
            _debug_log("dataset.py:_get_zipfile", "created_new_zipfile", {"new_zipfile_id": id(self._zipfile), "pid": current_pid}, "B")
            # #endregion
        return self._zipfile

    def _open_file(self, fname: str):
        if self._type == "dir":
            return open(os.path.join(self._path, fname), "rb")
        if self._type == "zip":
            # #region agent log
            _debug_log("dataset.py:_open_file", "opening_zip_entry", {"fname": fname, "zipfile_id": id(self._zipfile) if self._zipfile else None}, "C")
            # #endregion
            # Use read() instead of open() to avoid Python 3.12's overly strict
            # overlap detection which can incorrectly flag legitimate archives
            import io
            try:
                data = self._get_zipfile().read(fname)
            except Exception as e:
                # #region agent log
                _debug_log("dataset.py:_open_file", "read_error", {"fname": fname, "error": str(e), "error_type": type(e).__name__}, "C")
                # #endregion
                raise
            return io.BytesIO(data)
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        # #region agent log
        _debug_log("dataset.py:__getstate__", "pickling_dataset", {"zipfile_id_before": id(self._zipfile) if self._zipfile else None}, "E")
        # #endregion
        return dict(super().__getstate__(), _zipfile=None, _zipfile_pid=None)

    def _load_raw_image(self, raw_idx: int) -> np.ndarray:
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == ".png":
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
        image = image.transpose(2, 0, 1)  # HWC => CHW
        return image

    def _load_raw_labels(self) -> np.ndarray | None:
        """Load class labels from dataset.json.
        
        Expected format (from dataset_tool.py):
        {"labels": [["00000/img00000000.png", 6], ["00000/img00000001.png", 3], ...]}
        
        Labels can be:
        - Integer class IDs (e.g., 0-999 for ImageNet)
        - Float vectors (for embeddings)
        """
        fname = "dataset.json"
        if fname not in self._all_fnames:
            return None
        
        with self._open_file(fname) as f:
            data = json.load(f)
        
        labels_list = data.get("labels", None)
        if labels_list is None:
            return None
        
        # Convert list of [filename, label] to dict
        labels_dict = {item[0].replace("\\", "/"): item[1] for item in labels_list}
        
        # Get labels in the same order as image files
        ordered_labels = []
        for img_fname in self._image_fnames:
            # Normalize path separators for matching
            normalized_fname = img_fname.replace("\\", "/")
            if normalized_fname in labels_dict:
                ordered_labels.append(labels_dict[normalized_fname])
            else:
                # Try without leading directory
                alt_fname = normalized_fname.lstrip("./")
                if alt_fname in labels_dict:
                    ordered_labels.append(labels_dict[alt_fname])
                else:
                    # Label not found, return None to indicate incomplete labels
                    return None
        
        labels = np.array(ordered_labels)
        
        # Convert to appropriate dtype
        if labels.ndim == 1:
            labels = labels.astype(np.int64)  # Integer class labels
        else:
            labels = labels.astype(np.float32)  # Float embeddings
        
        return labels
