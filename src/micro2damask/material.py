from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

from .config import Config


def create_damask_material_file(
    grain_orientations: pd.DataFrame,
    grain_to_phase: Dict[int, int],
    cfg: Config,
    out_dir: Path,
) -> Dict[str, Any]:
    """
    Write a DAMASK material.yaml for two generic image phases.

    Convention:
        phase_id 0 = dark phase
        phase_id 1 = light phase

    The phase names and complete DAMASK phase definitions come from Config.
    No Al/Si material law is silently assumed.
    """
    try:
        from scipy.spatial.transform import Rotation as R  # type: ignore
        has_rotation = True
    except Exception:
        has_rotation = False

    if cfg.dark_phase_name == cfg.light_phase_name:
        raise ValueError("dark_phase_name und light_phase_name müssen unterschiedlich sein.")

    homogenization = {
        "SX": {"N_constituents": 1, "mechanical": {"type": "pass"}}
    }

    phase = {
        cfg.dark_phase_name: cfg.phase_material(0),
        cfg.light_phase_name: cfg.phase_material(1),
    }

    materials_list = []
    sorted_grains = grain_orientations.sort_values("grain_id").reset_index(drop=True)

    for _, row in sorted_grains.iterrows():
        gid = int(row["grain_id"])
        if gid not in grain_to_phase:
            raise ValueError(f"Grain {gid} besitzt keinen Eintrag in grain_to_phase.")

        phase_id = int(grain_to_phase[gid])
        phase_name = cfg.phase_name(phase_id)

        phi1 = row.get("phi1_deg", float("nan"))
        Phi = row.get("Phi_deg", float("nan"))
        phi2 = row.get("phi2_deg", float("nan"))

        if has_rotation and not (np.isnan(phi1) or np.isnan(Phi) or np.isnan(phi2)):
            rot = R.from_euler("ZXZ", [phi1, Phi, phi2], degrees=True)
            q_xyzw = rot.as_quat()
            orientation = [
                float(q_xyzw[3]),
                float(q_xyzw[0]),
                float(q_xyzw[1]),
                float(q_xyzw[2]),
            ]
        else:
            orientation = [1.0, 0.0, 0.0, 0.0]

        materials_list.append(
            {
                "constituents": [
                    {"phase": phase_name, "O": orientation, "v": 1.0}
                ],
                "homogenization": "SX",
            }
        )

    material_file_obj = {
        "homogenization": homogenization,
        "phase": phase,
        "material": materials_list,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    mat_path = out_dir / cfg.material_filename
    with open(mat_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(material_file_obj, f, sort_keys=False)

    sorted_grains.to_csv(out_dir / "grain_orientations.csv", index=False)

    return {
        "material_file": str(mat_path),
        "n_grains": int(len(materials_list)),
        "phase_names": {
            0: cfg.dark_phase_name,
            1: cfg.light_phase_name,
        },
    }
