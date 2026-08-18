from pathlib import Path

import cv2
import numpy as np
import yaml

from micro2damask import Config, run_pipeline


def _elastic_cubic_phase(c11: float, c12: float, c44: float):
    """Minimal material definition used only for the integration test."""
    return {
        "lattice": "cF",
        "mechanical": {
            "output": ["F", "P"],
            "elastic": {
                "type": "Hooke",
                "C_11": c11,
                "C_12": c12,
                "C_44": c44,
            },
            "plastic": {"type": "none"},
        },
    }


def test_full_pipeline_with_custom_phase_names(tmp_path: Path):
    image = np.full((64, 64), 220, dtype=np.uint8)
    cv2.circle(image, (16, 16), 6, 30, -1)
    cv2.circle(image, (46, 20), 5, 40, -1)
    cv2.rectangle(image, (25, 40), (35, 50), 35, -1)

    image_path = tmp_path / "synthetic_microstructure.png"
    assert cv2.imwrite(str(image_path), image)

    cfg = Config(
        image_path=str(image_path),
        um_per_pixel=0.5,
        dark_phase_name="Matrix",
        light_phase_name="SecondPhase",
        dark_phase_material=_elastic_cubic_phase(200e9, 120e9, 80e9),
        light_phase_material=_elastic_cubic_phase(150e9, 90e9, 60e9),
        rve_w=64,
        rve_h=64,
        downsample_factor=2,
        nz_layers=3,
        output_root=str(tmp_path / "output"),
        save_debug_plots=False,
    )

    result = run_pipeline(cfg)

    assert result["validation"]["ok"] is True
    assert result["grain_map"].shape == (32, 32)
    assert Path(result["material"]["material_file"]).exists()
    assert (result["output_dir"] / "geometry" / "grain_map_3d_nz_ny_nx.npy").exists()
    assert (result["output_dir"] / "statistics" / "statistics_summary.csv").exists()

    with open(result["material"]["material_file"], "r", encoding="utf-8") as f:
        material_yaml = yaml.safe_load(f)

    assert set(material_yaml["phase"].keys()) == {"Matrix", "SecondPhase"}
    assert set(result["grain_orientations"]["phase_name"]) == {"Matrix", "SecondPhase"}
    assert result["statistics"]["dark_phase_name"] == "Matrix"
    assert result["statistics"]["light_phase_name"] == "SecondPhase"
