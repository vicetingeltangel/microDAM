from pathlib import Path

import pandas as pd
import pytest
import yaml

from micro2damask.config import Config
from micro2damask.material import create_damask_material_file


def _phase(c11):
    return {
        "lattice": "cF",
        "mechanical": {
            "output": ["F", "P"],
            "elastic": {
                "type": "Hooke",
                "C_11": c11,
                "C_12": 0.5 * c11,
                "C_44": 0.25 * c11,
            },
            "plastic": {"type": "none"},
        },
    }


def test_custom_phase_names_are_written_to_material_yaml(tmp_path: Path):
    orientations = pd.DataFrame(
        {
            "grain_id": [0, 1],
            "phase_id": [0, 1],
            "phi1_deg": [0.0, 0.0],
            "Phi_deg": [0.0, 0.0],
            "phi2_deg": [0.0, 0.0],
        }
    )
    cfg = Config(
        dark_phase_name="Ferrit",
        light_phase_name="Martensit",
        dark_phase_material=_phase(200e9),
        light_phase_material=_phase(220e9),
    )

    result = create_damask_material_file(
        orientations,
        {0: 0, 1: 1},
        cfg,
        tmp_path,
    )

    with open(result["material_file"], "r", encoding="utf-8") as f:
        material = yaml.safe_load(f)

    assert set(material["phase"]) == {"Ferrit", "Martensit"}
    assert material["material"][0]["constituents"][0]["phase"] == "Ferrit"
    assert material["material"][1]["constituents"][0]["phase"] == "Martensit"


def test_missing_material_definition_is_rejected(tmp_path: Path):
    orientations = pd.DataFrame(
        {
            "grain_id": [0],
            "phase_id": [0],
            "phi1_deg": [0.0],
            "Phi_deg": [0.0],
            "phi2_deg": [0.0],
        }
    )

    with pytest.raises(ValueError, match="keine DAMASK-Materialdefinition"):
        create_damask_material_file(
            orientations,
            {0: 0},
            Config(dark_phase_name="Ferrit"),
            tmp_path,
        )
