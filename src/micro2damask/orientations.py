from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import Config


@dataclass(frozen=True)
class MTEXODF:
    """Binned ODF exported by MTEX on a Bunge-Euler grid."""

    euler_deg: np.ndarray
    values: np.ndarray
    crystal_symmetry: Optional[str] = None
    specimen_symmetry: Optional[str] = None
    source: Optional[str] = None


def _header_value(text: str, key: str) -> Optional[str]:
    match = re.search(
        rf'^\s*%\s*{re.escape(key)}\s*:\s*["\']?([^"\'\n]+)["\']?\s*$',
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return match.group(1).strip() if match else None


def load_mtex_odf(path: str | Path) -> MTEXODF:
    """
    Load a generic MTEX ASCII ODF export.

    Expected numeric columns are ``phi1 Phi phi2 value`` with Bunge Euler
    angles in degrees, as produced by MTEX ``export(odf, ..., 'Bunge')``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"MTEX ODF nicht gefunden: {path}")

    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        raw = np.loadtxt(path, comments="%", dtype=float)
    except ValueError as exc:
        raise ValueError(f"MTEX ODF konnte nicht gelesen werden: {path}") from exc

    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.ndim != 2 or raw.shape[1] != 4:
        raise ValueError(
            "MTEX ODF muss genau vier numerische Spalten enthalten: "
            "phi1 Phi phi2 value."
        )

    euler = raw[:, :3]
    values = raw[:, 3]

    if not np.all(np.isfinite(euler)) or not np.all(np.isfinite(values)):
        raise ValueError("MTEX ODF enthält NaN oder unendliche Werte.")
    if np.any(euler < 0.0):
        raise ValueError("MTEX ODF enthält negative Eulerwinkel.")
    if np.any(euler[:, 0] >= 360.0) or np.any(euler[:, 1] > 180.0) or np.any(euler[:, 2] >= 360.0):
        raise ValueError(
            "Eulerwinkel außerhalb der erwarteten Bunge-Bereiche "
            "phi1=[0,360), Phi=[0,180], phi2=[0,360)."
        )

    # The probability conversion below assumes a regular Cartesian Bunge grid,
    # which is the default generic MTEX ODF export used by this importer.
    axes = [np.unique(euler[:, i]) for i in range(3)]
    if int(np.prod([len(a) for a in axes])) != len(euler):
        raise ValueError(
            "MTEX ODF ist kein vollständiges reguläres Bunge-Euler-Raster. "
            "Exportiere die ODF als generic ASCII grid."
        )
    for axis in axes:
        if len(axis) > 2 and not np.allclose(np.diff(axis), np.diff(axis)[0]):
            raise ValueError("MTEX ODF besitzt kein gleichmäßig aufgelöstes Euler-Raster.")

    # A physical ODF is non-negative. Small negative reconstruction artefacts
    # are treated in the same spirit as DAMASK Rotation.from_ODF: they do not
    # contribute to the sampled volume fraction.
    values = np.maximum(values, 0.0)
    if not np.any(values > 0.0):
        raise ValueError("MTEX ODF enthält keine positive Intensität.")

    return MTEXODF(
        euler_deg=euler,
        values=values,
        crystal_symmetry=_header_value(text, "crystal symmetry"),
        specimen_symmetry=_header_value(text, "specimen symmetry"),
        source=str(path),
    )


def mtex_odf_sampling_probabilities(odf: MTEXODF) -> np.ndarray:
    """
    Convert MTEX ODF intensity values to sampling probabilities.

    MTEX exports ODF values as probability densities (m.r.d.), not as discrete
    volume fractions. For a regular Bunge-Euler grid, the Euler-space volume
    element is proportional to ``sin(Phi)``. Constant grid-spacing factors
    cancel during normalization.

    This is equivalent to the density correction used by
    ``damask.Rotation.from_ODF(..., fractions=False)`` for a regular grid.
    """
    Phi_rad = np.deg2rad(odf.euler_deg[:, 1])
    mass = odf.values * np.sin(Phi_rad)
    total = float(np.sum(mass))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(
            "ODF besitzt nach Euler-Raum-Gewichtung keine positive Wahrscheinlichkeitsmasse."
        )
    return mass / total


def sample_mtex_odf(
    odf: MTEXODF,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample ``n`` Bunge-Euler orientations from an MTEX ODF."""
    if n < 0:
        raise ValueError("n muss >= 0 sein.")
    if n == 0:
        return pd.DataFrame(
            columns=["phi1_deg", "Phi_deg", "phi2_deg", "odf_value_mrd"]
        )

    probabilities = mtex_odf_sampling_probabilities(odf)
    indices = rng.choice(len(probabilities), size=n, replace=True, p=probabilities)
    sampled = odf.euler_deg[indices]

    return pd.DataFrame(
        {
            "phi1_deg": sampled[:, 0],
            "Phi_deg": sampled[:, 1],
            "phi2_deg": sampled[:, 2],
            "odf_value_mrd": odf.values[indices],
        }
    )


def generate_orientations_per_grain(
    n_grains: int,
    grain_to_phase: Dict[int, int],
    cfg: Config,
) -> pd.DataFrame:
    """Generate one crystallographic orientation for each grain."""
    rng = np.random.default_rng(cfg.random_seed)
    rows = []

    # Cache ODF files because all grains of one phase use the same source.
    odf_cache: Dict[int, MTEXODF] = {}
    sampled_by_gid: Dict[int, dict] = {}

    for phase in (0, 1):
        gids = [gid for gid in range(n_grains) if int(grain_to_phase.get(gid, 0)) == phase]
        if not gids:
            continue

        mode = cfg.phase_orientation_mode(phase)
        if mode != "mtex_odf":
            continue

        odf_path = cfg.phase_odf_path(phase)
        if not odf_path:
            raise ValueError(
                f"orientation mode 'mtex_odf' für Phase '{cfg.phase_name(phase)}' "
                "benötigt einen ODF-Pfad."
            )

        odf = load_mtex_odf(odf_path)
        odf_cache[phase] = odf
        sampled = sample_mtex_odf(odf, len(gids), rng)

        for gid, sample in zip(gids, sampled.to_dict(orient="records")):
            sampled_by_gid[gid] = sample

    for gid in range(n_grains):
        phase = int(grain_to_phase.get(gid, 0))
        mode = cfg.phase_orientation_mode(phase)

        odf_value = float("nan")
        orientation_source = mode

        if mode == "random":
            # Kept for backwards compatibility. A separate orientation patch can
            # replace this with uniform SO(3) sampling without changing ODF I/O.
            phi1 = float(rng.uniform(0, 360))
            Phi = float(rng.uniform(0, 180))
            phi2 = float(rng.uniform(0, 360))
        elif mode == "mtex_odf":
            sample = sampled_by_gid[gid]
            phi1 = float(sample["phi1_deg"])
            Phi = float(sample["Phi_deg"])
            phi2 = float(sample["phi2_deg"])
            odf_value = float(sample["odf_value_mrd"])
            orientation_source = str(odf_cache[phase].source)
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
                "orientation_mode": mode,
                "orientation_source": orientation_source,
                "odf_value_mrd": odf_value,
            }
        )

    return pd.DataFrame(rows)
