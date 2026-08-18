from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import numpy as np

from .config import Config
from .utils import timestamped_output_dir, effective_um_per_pixel
from .preprocessing import load_image, preprocess_image
from .segmentation import segment_microstructure, clean_segmentation
from .rve import select_rve, downsample_phase_map, build_voxel_grid_3d_from_grain_map
from .statistics import calculate_statistics, check_periodicity, calculate_grain_statistics, calculate_rve_statistics_from_grains
from .grains import identify_grains
from .orientations import generate_orientations_per_grain
from .geometry import create_damask_geometry_from_grains
from .material import create_damask_material_file
from .validation import validate_damask_model
from .visualization import plot_results
from .output import save_results, print_summary

def run_pipeline(cfg: Config) -> Dict[str, Any]:
    out_dir = timestamped_output_dir(cfg.output_root)
    data = load_image(cfg)
    gray = data["gray"]
    prep = preprocess_image(gray, cfg)
    preprocessed = prep["preprocessed"]
    seg = segment_microstructure(preprocessed, cfg)
    phase_raw = seg["phase_map_raw"]
    phase_clean = clean_segmentation(phase_raw, cfg)
    rve_original, rve_rect = select_rve(phase_clean, cfg)
    rve_phase_map = downsample_phase_map(rve_original, max(1, int(cfg.downsample_factor)))
    stats, phase_region_stats = calculate_statistics(rve_phase_map, effective_um_per_pixel(cfg), cfg)
    periodic_info = check_periodicity(rve_phase_map, cfg.periodicity_tolerance)
    grain_map, n_grains, grain_to_phase = identify_grains(rve_phase_map, cfg)
    grain_stats_df = calculate_grain_statistics(grain_map, grain_to_phase, cfg)
    rve_stats_df = calculate_rve_statistics_from_grains(grain_stats_df, rve_phase_map, cfg)
    grain_orientations = generate_orientations_per_grain(n_grains, grain_to_phase, cfg)
    damask_info = create_damask_geometry_from_grains(grain_map, rve_phase_map, cfg, out_dir)
    material_info = create_damask_material_file(grain_orientations, grain_to_phase, cfg, out_dir / "material")
    voxel3d = build_voxel_grid_3d_from_grain_map(grain_map, cfg.nz_layers)
    validation = validate_damask_model(
        voxel3d=voxel3d,
        grain_map_2d=grain_map,
        grain_orientations=grain_orientations,
        grain_to_phase=grain_to_phase,
        cfg=cfg,
    )
    if cfg.save_debug_plots:
        plot_results(data["working"], gray, preprocessed, seg["raw_binary"], phase_clean, rve_phase_map, rve_rect, periodic_info, cfg, out_dir, grain_map, grain_stats_df)
    save_results(
        cleaned_phase_map=phase_clean,
        rve_phase_map=rve_phase_map,
        stats=stats,
        phase_region_stats=phase_region_stats,
        periodic_info=periodic_info,
        damask_info=damask_info,
        orientation_info=grain_orientations,
        cfg=cfg,
        out_dir=out_dir,
        grain_map=grain_map,
        grain_stats_df=grain_stats_df,
        rve_stats_df=rve_stats_df,
    )
    print_summary(stats, periodic_info, damask_info, cfg, out_dir)
    return {"output_dir": out_dir, "statistics": stats, "periodicity": periodic_info, "grain_map": grain_map, "grain_to_phase": grain_to_phase, "grain_orientations": grain_orientations, "damask": damask_info, "material": material_info, "validation": validation}
