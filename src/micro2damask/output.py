from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pandas as pd
import yaml

from .config import Config


def save_results(
    cleaned_phase_map: np.ndarray,
    rve_phase_map: np.ndarray,
    stats: Dict[str, Any],
    phase_region_stats: Dict[str, Dict[str, Any]],
    periodic_info: Dict[str, Any],
    damask_info: Dict[str, Any],
    orientation_info: pd.DataFrame,
    cfg: Config,
    out_dir: Path,
    grain_map: Optional[np.ndarray] = None,
    grain_stats_df: Optional[pd.DataFrame] = None,
    rve_stats_df: Optional[pd.DataFrame] = None,
):
    cv2.imwrite(str(out_dir / "images" / "segmentation_cleaned.png"), (cleaned_phase_map * 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / "images" / "rve_phase_map.png"), (rve_phase_map * 255).astype(np.uint8))

    pd.DataFrame([{**stats, **periodic_info, **damask_info}]).to_csv(
        out_dir / "statistics" / "statistics_summary.csv",
        index=False,
    )

    for key in ("dark", "light"):
        reg = phase_region_stats[key]
        areas_px = reg["areas_px"]
        if areas_px is None or len(areas_px) == 0:
            continue
        df_areas = pd.DataFrame({f"{key}_phase_region_area_px": areas_px})
        if reg["areas_um2"] is not None:
            df_areas[f"{key}_phase_region_area_um2"] = reg["areas_um2"]
        df_areas.to_csv(
            out_dir / "statistics" / f"{key}_phase_region_area_distribution.csv",
            index=False,
        )

    if grain_stats_df is not None:
        grain_stats_df.to_csv(out_dir / "statistics" / "grain_statistics.csv", index=False)
    if rve_stats_df is not None:
        rve_stats_df.to_csv(out_dir / "statistics" / "rve_statistics.csv", index=False)

    orientation_info.to_csv(out_dir / "material" / "grain_orientations.csv", index=False)

    if grain_map is not None:
        np.save(out_dir / "geometry" / "grain_map_2d.npy", grain_map.astype(np.int32))

    config_dict = asdict(cfg)
    with open(out_dir / "config" / "parameters.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)
    with open(out_dir / "config" / "parameters.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config_dict, f, sort_keys=False)
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)


def list_geometry_files(geom_dir: Path) -> None:
    geom_dir = Path(geom_dir)
    if not geom_dir.exists():
        print("Geometry-Ordner nicht gefunden:", geom_dir)
        return
    print("Dateien in", geom_dir)
    for p in sorted(geom_dir.iterdir()):
        print(" -", p.name)


def list_vti_arrays(vti_path: Path) -> List[str]:
    vti_path = Path(vti_path)
    if not vti_path.exists():
        print("VTI nicht gefunden:", vti_path)
        return []
    try:
        tree = ET.parse(str(vti_path))
        root = tree.getroot()
        arrays = []
        for dd in root.findall(".//CellData/DataArray") + root.findall(".//PointData/DataArray"):
            arrays.append(dd.get("Name") or dd.get("name"))
        print(f"Arrays in {vti_path.name}: {arrays}")
        return arrays
    except Exception as e:
        print("VTI-XML konnte nicht geparst werden:", e)
        return []


def print_summary(stats, periodic_info, damask_info, cfg: Config, out_dir: Path):
    print("\n===== Auswertung =====")
    print(f"{cfg.dark_phase_name} (dunkel) Flächenanteil: {stats['dark_area_fraction']:.4f}")
    print(f"{cfg.light_phase_name} (hell) Flächenanteil:  {stats['light_area_fraction']:.4f}")
    print(f"{cfg.dark_phase_name} Regionen: {stats['dark_n_regions']}")
    print(f"{cfg.light_phase_name} Regionen: {stats['light_n_regions']}")

    if stats["dark_mean_nn_distance"] is not None:
        print(
            f"Mittlerer {cfg.dark_phase_name}-{cfg.dark_phase_name} Abstand "
            f"[{stats['nn_distance_unit']}]: {stats['dark_mean_nn_distance']:.3f}"
        )
    if stats["light_mean_nn_distance"] is not None:
        print(
            f"Mittlerer {cfg.light_phase_name}-{cfg.light_phase_name} Abstand "
            f"[{stats['nn_distance_unit']}]: {stats['light_mean_nn_distance']:.3f}"
        )

    print("\n===== Periodizität =====")
    print(f"Left-Right mismatch: {periodic_info['lr_mismatch']:.3f}")
    print(f"Top-Bottom mismatch: {periodic_info['tb_mismatch']:.3f}")
    if periodic_info["warning_left_right"] or periodic_info["warning_top_bottom"]:
        print("WARNUNG: Starke Randdiskontinuität erkannt. RVE-Lage/Größe prüfen.")
    else:
        print("Periodizitäts-Check unauffällig (innerhalb Toleranz).")

    print("\n===== DAMASK =====")
    print(damask_info.get("damask_message", ""))
    print(f"Geometrie-VTI für Simulation: {damask_info.get('vti_path')}")
    print(f"Ausgabeordner: {out_dir.resolve()}")
