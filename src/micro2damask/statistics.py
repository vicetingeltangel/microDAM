from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from skimage import measure

from .config import Config
from .utils import effective_um_per_pixel


def connected_region_stats(binary_phase: np.ndarray, um_per_px: Optional[float]) -> Dict[str, Any]:
    lbl = measure.label(binary_phase, connectivity=2)
    props = measure.regionprops(lbl)
    areas_px = np.array([p.area for p in props], dtype=float) if props else np.array([], dtype=float)
    n = len(areas_px)
    mean_area_px = float(np.mean(areas_px)) if n > 0 else 0.0

    if um_per_px is not None:
        areas_um2 = areas_px * (um_per_px ** 2)
        mean_area_um2 = float(np.mean(areas_um2)) if n > 0 else 0.0
    else:
        areas_um2 = None
        mean_area_um2 = None

    return {
        "n_regions": n,
        "areas_px": areas_px,
        "mean_area_px": mean_area_px,
        "areas_um2": areas_um2,
        "mean_area_um2": mean_area_um2,
    }


def nearest_neighbor_distance(binary_phase: np.ndarray, um_per_px: Optional[float]) -> Optional[float]:
    lbl = measure.label(binary_phase, connectivity=2)
    props = measure.regionprops(lbl)
    if len(props) < 2:
        return None

    centroids = np.array([p.centroid for p in props])[:, ::-1]
    tree = cKDTree(centroids)
    dists, _ = tree.query(centroids, k=2)
    value = float(np.mean(dists[:, 1]))

    if um_per_px is not None:
        value *= um_per_px
    return value


def calculate_statistics(
    rve_phase_map: np.ndarray,
    um_per_px: Optional[float],
    cfg: Config,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    masks = {
        "dark": rve_phase_map == 0,
        "light": rve_phase_map == 1,
    }

    region_stats = {
        key: connected_region_stats(mask, um_per_px)
        for key, mask in masks.items()
    }

    nn = {
        key: nearest_neighbor_distance(mask, um_per_px)
        for key, mask in masks.items()
    }

    total = rve_phase_map.size
    out = {
        "dark_phase_name": cfg.dark_phase_name,
        "light_phase_name": cfg.light_phase_name,
        "dark_area_fraction": float(masks["dark"].sum() / total),
        "light_area_fraction": float(masks["light"].sum() / total),
        "dark_n_regions": region_stats["dark"]["n_regions"],
        "light_n_regions": region_stats["light"]["n_regions"],
        "dark_mean_region_area_px": region_stats["dark"]["mean_area_px"],
        "light_mean_region_area_px": region_stats["light"]["mean_area_px"],
        "dark_mean_region_area_um2": region_stats["dark"]["mean_area_um2"],
        "light_mean_region_area_um2": region_stats["light"]["mean_area_um2"],
        "dark_mean_nn_distance": nn["dark"],
        "light_mean_nn_distance": nn["light"],
        "nn_distance_unit": "um" if um_per_px is not None else "px",
    }
    return out, region_stats


def check_periodicity(rve_phase_map: np.ndarray, tolerance: float = 0.10) -> Dict[str, Any]:
    left = rve_phase_map[:, 0]
    right = rve_phase_map[:, -1]
    top = rve_phase_map[0, :]
    bottom = rve_phase_map[-1, :]
    lr_mismatch = float(np.mean(left != right))
    tb_mismatch = float(np.mean(top != bottom))
    return {
        "lr_mismatch": lr_mismatch,
        "tb_mismatch": tb_mismatch,
        "tolerance": tolerance,
        "warning_left_right": lr_mismatch > tolerance,
        "warning_top_bottom": tb_mismatch > tolerance,
    }


def calculate_grain_statistics(
    grain_map: np.ndarray,
    grain_to_phase: Dict[int, int],
    cfg: Config,
) -> pd.DataFrame:
    labeled_for_props = grain_map + 1
    props = measure.regionprops(labeled_for_props)
    rows = []
    effective_pixel_size = effective_um_per_pixel(cfg)
    um2_factor = effective_pixel_size ** 2 if effective_pixel_size is not None else None

    for p in props:
        gid = int(p.label - 1)
        phase_id = int(grain_to_phase.get(gid, -1))
        area_px = int(p.area)
        area_um2 = float(area_px * um2_factor) if um2_factor is not None else None
        cy, cx = p.centroid
        minr, minc, maxr, maxc = p.bbox
        perim = float(p.perimeter) if hasattr(p, "perimeter") else None
        equiv_d = (float(p.equivalent_diameter_area) if hasattr(p, "equivalent_diameter_area")else None
        )

        rows.append(
            {
                "grain_id": gid,
                "phase_id": phase_id,
                "phase_name": cfg.phase_name(phase_id) if phase_id in (0, 1) else "undefined",
                "area_px": area_px,
                "area_um2": area_um2,
                "centroid_x_px": float(cx),
                "centroid_y_px": float(cy),
                "bbox_min_row": int(minr),
                "bbox_min_col": int(minc),
                "bbox_max_row": int(maxr),
                "bbox_max_col": int(maxc),
                "perimeter_px": perim,
                "equiv_diameter_px": equiv_d,
            }
        )

    return pd.DataFrame(rows).sort_values("grain_id").reset_index(drop=True)


def calculate_rve_statistics_from_grains(
    df_grains: pd.DataFrame,
    phase_map: np.ndarray,
    cfg: Config,
) -> pd.DataFrame:
    areas = df_grains["area_px"].values
    out = {
        "dark_phase_name": cfg.dark_phase_name,
        "light_phase_name": cfg.light_phase_name,
        "total_grains": int(len(df_grains)),
        "n_dark_grains": int((df_grains["phase_id"] == 0).sum()),
        "n_light_grains": int((df_grains["phase_id"] == 1).sum()),
        "dark_area_fraction": float((phase_map == 0).sum() / phase_map.size),
        "light_area_fraction": float((phase_map == 1).sum() / phase_map.size),
        "mean_grain_area_px": float(areas.mean()) if len(areas) > 0 else None,
        "median_grain_area_px": float(np.median(areas)) if len(areas) > 0 else None,
        "min_grain_area_px": int(areas.min()) if len(areas) > 0 else None,
        "max_grain_area_px": int(areas.max()) if len(areas) > 0 else None,
    }
    return pd.DataFrame([out])
