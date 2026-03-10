# -*- coding: utf-8 -*-
"""
Monkey-patch Ultralytics BaseDataset.load_image,
read images from LMDB instead of file IO.
"""

import os
import cv2
import math
import numpy as np
from pathlib import Path
from functools import lru_cache

_lmdb_envs = {}


def _get_lmdb_env(lmdb_dir: str):
    if lmdb_dir not in _lmdb_envs:
        import lmdb
        env = lmdb.open(
            lmdb_dir,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            max_readers=128,
        )
        _lmdb_envs[lmdb_dir] = env
    return _lmdb_envs[lmdb_dir]


def _infer_lmdb_dir(im_file: str) -> str:
    p = Path(im_file)
    split = p.parent.name  # e.g. "train2017"
    data_root = p.parent.parent.parent  # data/images/train2017 → data
    lmdb_dir = data_root / "lmdb" / split
    return str(lmdb_dir)


def _load_from_lmdb(lmdb_dir: str, filename: str, cv2_flag: int) -> np.ndarray:
    env = _get_lmdb_env(lmdb_dir)
    key = filename.encode("utf-8")
    with env.begin(write=False) as txn:
        buf = txn.get(key)
    if buf is None:
        return None
    img_array = np.frombuffer(buf, dtype=np.uint8)
    im = cv2.imdecode(img_array, cv2_flag)
    return im


def _patched_load_image(self, i, rect_mode=True):
    im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
    if im is None:
        lmdb_dir = _infer_lmdb_dir(f)
        lmdb_ok = Path(lmdb_dir).exists()

        if lmdb_ok:
            filename = Path(f).name
            im = _load_from_lmdb(lmdb_dir, filename, self.cv2_flag)

        if im is None:
            if fn.exists():
                try:
                    im = np.load(fn)
                except Exception:
                    from ultralytics.utils import imread
                    im = imread(f, flags=self.cv2_flag)
            else:
                from ultralytics.utils import imread
                im = imread(f, flags=self.cv2_flag)

        if im is None:
            raise FileNotFoundError(f"Image Not Found {f}")

        h0, w0 = im.shape[:2]
        if rect_mode:
            r = self.imgsz / max(h0, w0)
            if r != 1:
                w, h = (min(math.ceil(w0 * r), self.imgsz),
                        min(math.ceil(h0 * r), self.imgsz))
                im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        elif not (h0 == w0 == self.imgsz):
            im = cv2.resize(im, (self.imgsz, self.imgsz),
                            interpolation=cv2.INTER_LINEAR)
        if im.ndim == 2:
            im = im[..., None]

        if self.augment:
            self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]
            self.buffer.append(i)
            if 1 < len(self.buffer) >= self.max_buffer_length:
                j = self.buffer.pop(0)
                if self.cache != "ram":
                    self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

        return im, (h0, w0), im.shape[:2]

    return self.ims[i], self.im_hw0[i], self.im_hw[i]


def patch_lmdb_loader():
    from ultralytics.data.base import BaseDataset
    BaseDataset.load_image = _patched_load_image
    print("LMDB image loading patch enabled")


def close_all():
    for env in _lmdb_envs.values():
        env.close()
    _lmdb_envs.clear()
