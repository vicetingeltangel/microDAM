from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List
import json
import time
import uuid
import warnings
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

import cv2
from .config import Config
from .utils import safe_imread, to_gray_uint8

def load_image(cfg: Config) -> Dict[str, Any]:
    img = safe_imread(cfg.image_path)
    original = img.copy()
    if all(v is not None for v in [cfg.crop_x, cfg.crop_y, cfg.crop_w, cfg.crop_h]):
        x, y, w, h = cfg.crop_x, cfg.crop_y, cfg.crop_w, cfg.crop_h
        img = img[y:y + h, x:x + w].copy()
    gray = to_gray_uint8(img)
    return {"original": original, "working": img, "gray": gray}


def preprocess_image(gray: np.ndarray, cfg: Config) -> Dict[str, np.ndarray]:
    proc = gray.copy()
    if cfg.denoise_method == "median":
        k = max(3, cfg.median_ksize if cfg.median_ksize % 2 == 1 else cfg.median_ksize + 1)
        proc = cv2.medianBlur(proc, k)
    elif cfg.denoise_method == "gaussian":
        proc = cv2.GaussianBlur(proc, (0, 0), cfg.gaussian_sigma)
    elif cfg.denoise_method == "nlmeans":
        proc = cv2.fastNlMeansDenoising(proc, None, h=10, templateWindowSize=7, searchWindowSize=21)
    else:
        raise ValueError(f"Unbekannte denoise_method: {cfg.denoise_method}")

    if cfg.do_background_correction:
        bg = cv2.GaussianBlur(proc, (0, 0), cfg.bg_blur_sigma)
        proc = cv2.subtract(proc, bg)
        proc = cv2.normalize(proc, None, 0, 255, cv2.NORM_MINMAX)

    tile_grid = cfg.clahe_tile_grid_size
    if isinstance(tile_grid, int):
        tile_grid = (tile_grid, tile_grid)
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip_limit, tileGridSize=tile_grid)
    proc_clahe = clahe.apply(proc.astype(np.uint8))
    return {"denoised": proc, "preprocessed": proc_clahe}

