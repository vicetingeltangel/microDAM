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

from scipy import ndimage as ndi
from skimage import segmentation, feature
from .config import Config

def select_rve(phase_map: np.ndarray, cfg: Config) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    H, W = phase_map.shape
    rw, rh = cfg.rve_w, cfg.rve_h
    if rw <= 0 or rh <= 0:
        raise ValueError("RVE-Größe muss >0 sein.")
    if rw > W or rh > H:
        raise ValueError(f"RVE ({rw}x{rh}) größer als Bild ({W}x{H}).")
    x = cfg.rve_x if cfg.rve_x is not None else (W - rw) // 2
    y = cfg.rve_y if cfg.rve_y is not None else (H - rh) // 2
    if x < 0 or y < 0 or x + rw > W or y + rh > H:
        raise ValueError("RVE-Fenster liegt außerhalb des segmentierten Bildes.")
    rve = phase_map[y:y + rh, x:x + rw].copy()
    return rve, (x, y, rw, rh)


def downsample_phase_map(
    phase_map: np.ndarray,
    factor: int
) -> np.ndarray:
    """
    Reduziert eine binäre Phasenkarte.

    factor = 4:
        4x4 Originalpixel -> 1 neues Pixel/Voxel

    Für jeden Block wird die dominante Phase verwendet:
        >= 50 % helle Phase -> helle Phase (1)
        <  50 % helle Phase -> dunkle Phase (0)

    Parameters
    ----------
    phase_map : np.ndarray
        2D Phasenkarte mit:
            0 = dunkle Phase
            1 = helle Phase

    factor : int
        Downsampling-Faktor.

    Returns
    -------
    np.ndarray
        Reduzierte Phasenkarte.
    """

    if factor < 1:
        raise ValueError("downsample_factor muss >= 1 sein.")

    if factor == 1:
        return phase_map.astype(np.uint8, copy=True)

    ny, nx = phase_map.shape

    # Nur vollständige Blöcke verwenden
    new_ny = ny // factor
    new_nx = nx // factor

    if new_ny < 1 or new_nx < 1:
        raise ValueError(
            f"Bild ({nx}x{ny}) ist kleiner als "
            f"der Downsampling-Faktor {factor}."
        )

    cropped = phase_map[
        :new_ny * factor,
        :new_nx * factor
    ]

    # Aufteilung in:
    # (neue_y, factor, neue_x, factor)
    blocks = cropped.reshape(
        new_ny,
        factor,
        new_nx,
        factor
    )

    # Mittelwert innerhalb jedes 4x4-Blocks
    # >= 0.5 => helle Phase (1)
    light_fraction = blocks.mean(axis=(1, 3))

    reduced = (light_fraction >= 0.5).astype(np.uint8)

    return reduced


def optional_dendrite_mode(preprocessed: np.ndarray, phase_map: np.ndarray, cfg: Config) -> Optional[np.ndarray]:
    if not cfg.enable_dendrite_mode:
        return None
    target = (phase_map == cfg.dendrite_phase_id)
    distance = ndi.distance_transform_edt(target)
    coords = feature.peak_local_max(distance, min_distance=cfg.dendrite_min_distance, labels=target.astype(np.uint8))
    markers = np.zeros_like(distance, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i
    labels = segmentation.watershed(-distance, markers, mask=target)
    return labels.astype(np.int32)


def build_voxel_grid_3d_from_grain_map(grain_map: np.ndarray, nz: int) -> np.ndarray:
    if nz < 1:
        raise ValueError("nz_layers muss >=1 sein.")
    ny, nx = grain_map.shape
    grid3d = np.repeat(grain_map[np.newaxis, :, :], repeats=nz, axis=0)
    assert grid3d.shape == (nz, ny, nx)
    return grid3d.astype(np.int32)

