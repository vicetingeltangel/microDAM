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

def timestamped_output_dir(root: str) -> Path:
    p = Path(root)
    p.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    out = p / f"run_{run_id}"
    out.mkdir(parents=True, exist_ok=False)
    for sub in ["images", "geometry", "material", "statistics", "config"]:
        (out / sub).mkdir(exist_ok=True)
    return out

def ensure_odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1

def safe_imread(path: str) -> np.ndarray:
    if not Path(path).exists():
        raise FileNotFoundError(f"Bilddatei nicht gefunden: {path}")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Bild konnte nicht gelesen werden: {path}")
    return img

def to_gray_uint8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Nicht unterstützte Bilddimension: {img.shape}")
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray

def add_scale_bar(ax, um_per_px: Optional[float], img_shape: Tuple[int, int], frac: float = 0.2):
    if um_per_px is None:
        return
    h, w = img_shape
    bar_px = int(w * frac)
    bar_um = bar_px * um_per_px
    x0 = int(w * 0.05)
    y0 = int(h * 0.93)
    ax.plot([x0, x0 + bar_px], [y0, y0], color='white', lw=3)
    ax.text(x0, y0 - 8, f"{bar_um:.1f} µm", color='white', fontsize=8, va='bottom')

def spacing_from_cfg(cfg: Config) -> tuple:
    """
    Voxel-Kantenlänge in Metern (SI).

    Wenn downsample_factor > 1 verwendet wird, wird die
    ursprüngliche Pixelgröße entsprechend skaliert.

    Beispiel:
        um_per_pixel = 0.35
        downsample_factor = 4

        -> 1.40 µm / Voxel
        -> 1.40e-6 m / Voxel
    """

    factor = max(1, int(cfg.downsample_factor))

    if cfg.voxel_size_um is not None:
        sx, sy, sz = cfg.voxel_size_um
        return (
            sx * 1e-6,
            sy * 1e-6,
            sz * 1e-6
        )

    elif cfg.um_per_pixel is not None:
        px = cfg.um_per_pixel * factor

        return (
            px * 1e-6,
            px * 1e-6,
            px * 1e-6
        )

    return (1.0, 1.0, 1.0)

def effective_um_per_pixel(cfg: Config) -> Optional[float]:
    """
    Effektive räumliche Auflösung nach dem Downsampling.
    """

    if cfg.um_per_pixel is None:
        return None

    return cfg.um_per_pixel * max(1, int(cfg.downsample_factor))

