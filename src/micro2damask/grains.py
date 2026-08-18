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

from scipy.spatial import cKDTree
from skimage import measure, morphology
from .config import Config

def identify_grains(phase_map: np.ndarray, cfg: Config) -> Tuple[np.ndarray, int, Dict[int, int]]:
    if cfg.connectivity not in (4, 8):
        raise ValueError("connectivity muss 4 oder 8 sein.")
    conn = 1 if cfg.connectivity == 4 else 2
    ny, nx = phase_map.shape
    grain_map = np.full((ny, nx), fill_value=-1, dtype=np.int32)
    next_id = 0
    grain_to_phase: Dict[int, int] = {}
    for phase_val in [0, 1]:
        mask = (phase_map == phase_val)
        if mask.sum() == 0:
            continue
        lbl = measure.label(mask, connectivity=conn)
        n_labels = lbl.max()
        if n_labels == 0:
            continue
        for lab in range(1, n_labels + 1):
            pos = (lbl == lab)
            grain_map[pos] = next_id
            grain_to_phase[next_id] = int(phase_val)
            next_id += 1

    if (grain_map < 0).any():
        missing = np.where(grain_map < 0)
        if len(missing[0]) > 0:
            coords = np.column_stack(np.nonzero(grain_map >= 0))
            labels = grain_map[grain_map >= 0]
            if len(coords) > 0:
                tree = cKDTree(coords)
                for r, c in zip(*missing):
                    _, idx = tree.query([r, c], k=1)
                    grain_map[r, c] = labels[idx]

    n_grains = next_id
    if cfg.small_grain_mode in ("remove", "merge") and cfg.min_grain_size > 0:
        grain_map = handle_small_grains(grain_map, phase_map, cfg)

    uniq = np.unique(grain_map)
    mapping = {old: new for new, old in enumerate(uniq)}
    grain_map = np.vectorize(mapping.get)(grain_map)
    new_gtp = {}
    for old, new in mapping.items():
        new_gtp[new] = grain_to_phase.get(int(old), 0)
    grain_to_phase = new_gtp
    n_grains = len(uniq)

    return grain_map.astype(np.int32), int(n_grains), grain_to_phase


def handle_small_grains(grain_map: np.ndarray, phase_map: np.ndarray, cfg: Config) -> np.ndarray:
    if cfg.min_grain_size <= 0:
        return grain_map
    out = grain_map.copy()
    labels, counts = np.unique(grain_map, return_counts=True)
    area_dict = {int(l): int(c) for l, c in zip(labels, counts)}
    small_labels = [int(l) for l, c in zip(labels, counts) if c < cfg.min_grain_size]
    if not small_labels:
        return out
    for small in small_labels:
        mask_small = (out == small)
        if mask_small.sum() == 0:
            continue
        dilated = morphology.dilation(mask_small, morphology.disk(3))
        neighbor_mask = dilated & (~mask_small)
        neighbor_labels = np.unique(out[neighbor_mask])
        phase_small = int(phase_map[mask_small][0])
        neighbor_labels_same_phase = [int(l) for l in neighbor_labels
                                       if l >= 0 and np.any(phase_map[out == l] == phase_small)]
        if not neighbor_labels_same_phase:
            ys, xs = np.nonzero(mask_small)
            coords_small = np.column_stack((ys, xs))
            coords_other = np.column_stack(np.nonzero(out >= 0))
            labels_other = out[out >= 0]
            if coords_other.size == 0:
                continue
            tree = cKDTree(coords_other)
            for r, c in coords_small:
                _, idx = tree.query([r, c], k=1)
                out[r, c] = int(labels_other[idx])
            continue
        best = max(neighbor_labels_same_phase, key=lambda x: area_dict.get(int(x), 0))
        out[mask_small] = best
        area_dict[int(best)] = area_dict.get(int(best), 0) + mask_small.sum()
        area_dict.pop(int(small), None)
    return out

