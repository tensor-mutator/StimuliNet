import cv2
import numpy as np
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from glob import glob
import os
import re
import json

path = os.path.join(os.path.split(__file__)[0], "MPISintel")
train_path = os.path.join(path, "training")
test_path = os.path.join(path, "test")
flow_path = os.path.join(train_path, "flow")

def _get_X(resolution: Tuple[int, int], rendering: str, normalize: bool) -> np.ndarray:
    X_path = os.path.join(train_path, rendering)
    dirs = sorted(glob("{}/*".format(X_path)))
    X = np.zeros(shape=[0, 2] + list(resolution) + [3], dtype=np.float32)
    for dir in dirs:
        imgs = sorted(glob("{}/*".format(dir)), key=lambda x: int(re.search(r'[0-9]{1,}', x).group()))
        src_imgs = imgs[:-1]
        dest_imgs = imgs[1:]
        for src_img , dest_img in zip(src_imgs, dest_imgs):
            src_img_arr, dest_img_arr = cv2.resize(cv2.imread(src_img), resolution), cv2.resize(cv2.imread(dest_img), resolution)
            img_packed = np.stack([src_img_arr, dest_img_arr], axis=0)
            X = np.concatenate([X, np.expand_dims(img_packed, axis=0)])
    return X/255.0 if normalize else X

def _read_flow(filename: str) -> np.ndarray:
    with open(filename, 'rb') as f:
         magic = np.fromfile(f, np.float32, count=1)
         data2d = None
         if 202021.25 != magic:
            print("Magic number incorrect. Invalid .flo file")
         else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            data2d = np.fromfile(f, np.float32, count=2 * w[0] * h[0])
            data2d = np.resize(data2d, (h[0], w[0], 2))
    return data2d

def _get_y(resolution: Tuple[int, int]) -> np.ndarray:
    dirs = sorted(glob("{}/*".format(flow_path)))
    y = np.zeros(shape=[0] + list(resolution) + [2], dtype=np.float32)
    for dir in dirs:
        flows = sorted(glob("{}/*".format(dir)), key=lambda x: int(re.search(r'[0-9]{1,}', x).group()))
        for flow in flows:
            flow_arr = np.resize(_read_flow(flow), resolution + (2,)) if resolution else _read_flow(flow)
            y = np.concatenate([y, np.expand_dims(flow_arr, axis=0)])
    return y

def load(img_resolution: Tuple[int, int], flow_resolution: Tuple[int, int] = None, train_size: float = 0.8, test_size: float = 0.2,
         rendering: str = "clean", normalize: bool = True) -> List:
    y = _get_y(flow_resolution)
    X = _get_X(img_resolution, rendering, normalize)
    return train_test_split(X[:, 0, :, :, :], X[:, 1, :, :, :], y, train_size=train_size, test_size=test_size)
