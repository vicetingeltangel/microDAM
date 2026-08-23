from pathlib import Path

import numpy as np
import pandas as pd

from micro2damask.config import Config
from micro2damask.visualization import (
    _cubic_ipf_rgb_from_crystal_direction,
    build_cubic_ipf_map,
    save_ipf_orientation_maps,
)


def _phase(lattice="cI"):
    return {
        "lattice": lattice,
        "mechanical": {
            "output": ["F", "P"],
            "elastic": {
                "type": "Hooke",
                "C_11": 200e9,
                "C_12": 120e9,
                "C_44": 80e9,
            },
            "plastic": {"type": "none"},
        },
    }


def test_cubic_ipf_standard_triangle_corner_colors():
    directions = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    rgb = _cubic_ipf_rgb_from_crystal_direction(directions)
    np.testing.assert_allclose(rgb[0], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(rgb[1], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(rgb[2], [0.0, 0.0, 1.0], atol=1e-12)


def test_cube_orientation_is_red_for_nd():
    grain_map = np.array([[0, 0], [0, 0]], dtype=int)
    orientations = pd.DataFrame(
        {
            "grain_id": [0],
            "phi1_deg": [0.0],
            "Phi_deg": [0.0],
            "phi2_deg": [0.0],
        }
    )
    rgb = build_cubic_ipf_map(grain_map, orientations, np.array([0.0, 0.0, 1.0]))
    expected = np.broadcast_to(np.array([1.0, 0.0, 0.0]), rgb.shape)
    np.testing.assert_allclose(rgb, expected, atol=1e-12)


def test_save_ipf_orientation_maps_writes_three_axes_and_overview(tmp_path: Path):
    grain_map = np.array([[0, 0], [1, 1]], dtype=int)
    orientations = pd.DataFrame(
        {
            "grain_id": [0, 1],
            "phi1_deg": [0.0, 45.0],
            "Phi_deg": [0.0, 35.0],
            "phi2_deg": [0.0, 20.0],
        }
    )
    cfg = Config(
        dark_phase_material=_phase("cI"),
        light_phase_material=_phase("cI"),
    )

    result = save_ipf_orientation_maps(grain_map, orientations, cfg, tmp_path)

    assert set(result) == {"RD_x", "TD_y", "ND_z", "overview"}
    for path in result.values():
        assert Path(path).is_file()
        assert Path(path).stat().st_size > 0
