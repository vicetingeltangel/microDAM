from __future__ import annotations
from typing import Dict
import warnings

import cv2
import numpy as np
from skimage import morphology

from .config import Config
from .utils import ensure_odd


def segment_microstructure(preprocessed: np.ndarray, cfg: Config) -> Dict[str, np.ndarray]:
    """Segment a grayscale image into a dark phase (0) and a light phase (1)."""
    img = preprocessed

    if cfg.threshold_method == "otsu":
        t, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr_value = int(t)
        phase_map_raw = (img >= thr_value).astype(np.uint8)

    elif cfg.threshold_method == "adaptive":
        bs = ensure_odd(max(3, cfg.adaptive_block_size))
        bw = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            bs,
            cfg.adaptive_C,
        )
        phase_map_raw = (bw > 0).astype(np.uint8)

    elif cfg.threshold_method == "manual":
        if cfg.manual_threshold is None:
            raise ValueError("manual_threshold muss bei threshold_method='manual' gesetzt sein.")
        thr_value = int(np.clip(cfg.manual_threshold, 0, 255))
        _, bw = cv2.threshold(img, thr_value, 255, cv2.THRESH_BINARY)
        phase_map_raw = (img >= thr_value).astype(np.uint8)

    else:
        raise ValueError(f"Unbekannte threshold_method: {cfg.threshold_method}")

    if cfg.invert_binary_after_threshold:
        phase_map_raw = 1 - phase_map_raw

    dark_fraction = float(np.mean(phase_map_raw == 0))
    light_fraction = float(np.mean(phase_map_raw == 1))
    if min(dark_fraction, light_fraction) < 0.005:
        warnings.warn(
            "Segmentierung liefert fast nur eine Phase "
            f"({cfg.dark_phase_name}={dark_fraction:.4f}, "
            f"{cfg.light_phase_name}={light_fraction:.4f}). Parameter prüfen."
        )

    return {
        "raw_binary": (phase_map_raw * 255).astype(np.uint8),
        "phase_map_raw": phase_map_raw.astype(np.uint8),
    }


def clean_segmentation(phase_map_raw: np.ndarray, cfg: Config) -> np.ndarray:
    """Apply binary morphology to the configured target phase."""
    target_name = cfg.morphology_target_phase.lower()
    if target_name == "dark":
        target_id = 0
    elif target_name == "light":
        target_id = 1
    else:
        raise ValueError("morphology_target_phase muss 'dark' oder 'light' sein.")

    target = phase_map_raw == target_id
    target = morphology.remove_small_objects(
        target,
        max_size=cfg.remove_small_objects_min_size,
    )
    target = morphology.remove_small_holes(
        target,
        max_size=cfg.remove_small_holes_area_threshold,
    )

    if cfg.do_opening:
        target = morphology.opening(target, morphology.disk(cfg.morph_disk_radius))
    if cfg.do_closing:
        target = morphology.closing(target, morphology.disk(cfg.morph_disk_radius))

    other_id = 1 - target_id
    cleaned = np.full(phase_map_raw.shape, other_id, dtype=np.uint8)
    cleaned[target] = target_id
    return cleaned
