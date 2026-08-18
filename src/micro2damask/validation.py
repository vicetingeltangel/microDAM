from __future__ import annotations
from typing import Any, Dict

import numpy as np
import pandas as pd

from .config import Config


def validate_damask_model(
    voxel3d: np.ndarray,
    grain_map_2d: np.ndarray,
    grain_orientations: pd.DataFrame,
    grain_to_phase: Dict[int, int],
    cfg: Config,
) -> Dict[str, Any]:
    out = {"ok": True, "errors": [], "warnings": []}

    if voxel3d.ndim != 3:
        return {
            "ok": False,
            "errors": [f"voxel3d muss 3-dimensional sein, ist aber {voxel3d.ndim}D."],
            "warnings": [],
        }

    nz, ny, nx = voxel3d.shape
    unique_ids = set(int(x) for x in np.unique(voxel3d))

    negative_ids = sorted(gid for gid in unique_ids if gid < 0)
    if negative_ids:
        out["ok"] = False
        out["errors"].append(
            f"Negative Material/Grain-IDs im Voxelgitter gefunden: {negative_ids}"
        )

    voxel_ids = {gid for gid in unique_ids if gid >= 0}

    if "grain_id" not in grain_orientations.columns:
        out["ok"] = False
        out["errors"].append("grain_orientations enthält keine Spalte 'grain_id'.")
        return out

    orientation_ids = set(int(x) for x in grain_orientations["grain_id"].values)
    phase_ids = set(int(x) for x in grain_to_phase.keys())
    expected_ids = orientation_ids | phase_ids

    undefined_voxel_ids = voxel_ids - expected_ids
    if undefined_voxel_ids:
        out["ok"] = False
        out["errors"].append(
            f"Voxel enthält inkonsistente IDs: {sorted(undefined_voxel_ids)}"
        )

    for gid in sorted(voxel_ids):
        if gid not in grain_to_phase:
            out["ok"] = False
            out["errors"].append(f"Grain {gid} hat keine Phase in grain_to_phase.")
        if gid not in orientation_ids:
            out["ok"] = False
            out["errors"].append(f"Grain {gid} hat keine Orientierung eingetragen.")

    invalid_phase_ids = sorted(
        {int(pid) for pid in grain_to_phase.values() if int(pid) not in (0, 1)}
    )
    if invalid_phase_ids:
        out["ok"] = False
        out["errors"].append(
            "Ungültige phase_id gefunden: "
            f"{invalid_phase_ids}. Zulässig sind 0=dunkel und 1=hell."
        )

    if grain_map_2d.shape != (ny, nx):
        out["ok"] = False
        out["errors"].append(
            "2D grain_map shape stimmt nicht mit voxel3d[0] überein."
        )
    elif not np.array_equal(grain_map_2d, voxel3d[0]):
        out["ok"] = False
        out["errors"].append(
            "2D grain_map stimmt inhaltlich nicht mit voxel3d[0] überein."
        )

    if grain_orientations["grain_id"].duplicated().any():
        dup = sorted(
            int(x)
            for x in grain_orientations.loc[
                grain_orientations["grain_id"].duplicated(keep=False), "grain_id"
            ].unique()
        )
        out["ok"] = False
        out["errors"].append(f"Doppelte grain_id in grain_orientations: {dup}")

    if "phase_id" in grain_orientations.columns:
        orientation_phase = {
            int(row.grain_id): int(row.phase_id)
            for row in grain_orientations[["grain_id", "phase_id"]].itertuples(index=False)
        }
        for gid in sorted(voxel_ids & orientation_ids & set(grain_to_phase.keys())):
            if orientation_phase.get(gid) != int(grain_to_phase[gid]):
                out["ok"] = False
                out["errors"].append(
                    f"Grain {gid}: phase_id in grain_orientations "
                    f"({orientation_phase.get(gid)}) stimmt nicht mit grain_to_phase "
                    f"({grain_to_phase[gid]}) überein."
                )

    if cfg.dark_phase_name == cfg.light_phase_name:
        out["ok"] = False
        out["errors"].append(
            "dark_phase_name und light_phase_name müssen unterschiedlich sein."
        )

    return out
