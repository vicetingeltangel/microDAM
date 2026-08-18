from __future__ import annotations
from typing import Dict

import numpy as np
import pandas as pd

from .config import Config


def generate_orientations_per_grain(
    n_grains: int,
    grain_to_phase: Dict[int, int],
    cfg: Config,
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_seed)
    rows = []

    for gid in range(n_grains):
        phase = int(grain_to_phase.get(gid, 0))
        mode = cfg.phase_orientation_mode(phase)

        if mode == "random":
            phi1 = float(rng.uniform(0, 360))
            Phi = float(rng.uniform(0, 180))
            phi2 = float(rng.uniform(0, 360))
        elif mode == "ebsd":
            phi1 = float("nan")
            Phi = float("nan")
            phi2 = float("nan")
        else:
            raise ValueError(
                f"Unbekannter orientation mode '{mode}' für Phase '{cfg.phase_name(phase)}'."
            )

        rows.append(
            {
                "grain_id": gid,
                "phase_id": phase,
                "phase_name": cfg.phase_name(phase),
                "phi1_deg": phi1,
                "Phi_deg": Phi,
                "phi2_deg": phi2,
                "representation": "Bunge_deg",
            }
        )

    return pd.DataFrame(rows)
