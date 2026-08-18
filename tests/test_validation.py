import numpy as np
import pandas as pd

from micro2damask.config import Config
from micro2damask.validation import validate_damask_model


def test_valid_damask_model():
    """
    Prüft einen vollständig konsistenten kleinen DAMASK-Datensatz.
    """

    grain_map = np.array(
        [
            [0, 0],
            [1, 1],
        ],
        dtype=np.int32,
    )

    voxel3d = grain_map[np.newaxis, :, :]

    grain_to_phase = {
        0: 0,
        1: 1,
    }

    grain_orientations = pd.DataFrame(
        {
            "grain_id": [0, 1],
        }
    )

    cfg = Config()

    result = validate_damask_model(
        voxel3d=voxel3d,
        grain_map_2d=grain_map,
        grain_orientations=grain_orientations,
        grain_to_phase=grain_to_phase,
        cfg=cfg,
    )

    assert result["ok"] is True
    assert result["errors"] == []


def test_missing_orientation_is_detected():
    """
    Grain 1 besitzt keine Orientierung und muss deshalb
    als inkonsistent erkannt werden.
    """

    grain_map = np.array(
        [
            [0, 0],
            [1, 1],
        ],
        dtype=np.int32,
    )

    voxel3d = grain_map[np.newaxis, :, :]

    grain_to_phase = {
        0: 0,
        1: 1,
    }

    grain_orientations = pd.DataFrame(
        {
            "grain_id": [0],
        }
    )

    result = validate_damask_model(
        voxel3d=voxel3d,
        grain_map_2d=grain_map,
        grain_orientations=grain_orientations,
        grain_to_phase=grain_to_phase,
        cfg=Config(),
    )

    assert result["ok"] is False

    assert any(
        "Grain 1 hat keine Orientierung" in error
        for error in result["errors"]
    )


def test_missing_phase_mapping_is_detected():
    """
    Grain 1 besitzt keinen Eintrag in grain_to_phase.
    """

    grain_map = np.array(
        [
            [0, 0],
            [1, 1],
        ],
        dtype=np.int32,
    )

    voxel3d = grain_map[np.newaxis, :, :]

    grain_to_phase = {
        0: 0,
    }

    grain_orientations = pd.DataFrame(
        {
            "grain_id": [0, 1],
        }
    )

    result = validate_damask_model(
        voxel3d=voxel3d,
        grain_map_2d=grain_map,
        grain_orientations=grain_orientations,
        grain_to_phase=grain_to_phase,
        cfg=Config(),
    )

    assert result["ok"] is False

    assert any(
        "Grain 1 hat keine Phase" in error
        for error in result["errors"]
    )


def test_invalid_voxel_id_is_detected():
    """
    Die Voxel-ID 2 ist weder über eine Orientierung
    noch über eine Phase definiert.
    """

    grain_map = np.array(
        [
            [0, 0],
            [1, 1],
        ],
        dtype=np.int32,
    )

    voxel3d = np.array(
        [
            [
                [0, 0],
                [1, 2],
            ]
        ],
        dtype=np.int32,
    )

    grain_to_phase = {
        0: 0,
        1: 1,
    }

    grain_orientations = pd.DataFrame(
        {
            "grain_id": [0, 1],
        }
    )

    result = validate_damask_model(
        voxel3d=voxel3d,
        grain_map_2d=grain_map,
        grain_orientations=grain_orientations,
        grain_to_phase=grain_to_phase,
        cfg=Config(),
    )

    assert result["ok"] is False

    assert any(
        "Voxel enthält inkonsistente IDs" in error
        for error in result["errors"]
    )


def test_negative_voxel_id_is_detected():
    """
    Negative Grain-/Material-IDs dürfen nicht vorkommen.
    """

    grain_map = np.array(
        [
            [0, 0],
            [1, 1],
        ],
        dtype=np.int32,
    )

    voxel3d = np.array(
        [
            [
                [0, -1],
                [1, 1],
            ]
        ],
        dtype=np.int32,
    )

    grain_to_phase = {
        0: 0,
        1: 1,
    }

    grain_orientations = pd.DataFrame(
        {
            "grain_id": [0, 1],
        }
    )

    result = validate_damask_model(
        voxel3d=voxel3d,
        grain_map_2d=grain_map,
        grain_orientations=grain_orientations,
        grain_to_phase=grain_to_phase,
        cfg=Config(),
    )

    assert result["ok"] is False

    assert any(
        "Negative Material/Grain-IDs" in error
        for error in result["errors"]
    )


def test_incompatible_grain_map_shape_is_detected():
    """
    Die 2D-Grain-Map muss dieselbe y/x-Geometrie
    wie das Voxelgitter besitzen.
    """

    grain_map = np.array(
        [
            [0, 1],
        ],
        dtype=np.int32,
    )

    voxel3d = np.array(
        [
            [
                [0, 0],
                [1, 1],
            ]
        ],
        dtype=np.int32,
    )

    grain_to_phase = {
        0: 0,
        1: 1,
    }

    grain_orientations = pd.DataFrame(
        {
            "grain_id": [0, 1],
        }
    )

    result = validate_damask_model(
        voxel3d=voxel3d,
        grain_map_2d=grain_map,
        grain_orientations=grain_orientations,
        grain_to_phase=grain_to_phase,
        cfg=Config(),
    )

    assert result["ok"] is False

    assert any(
        "grain_map shape" in error
        for error in result["errors"]
    )

def test_non_contiguous_grain_ids_are_valid():
    """
    Grain-IDs müssen nicht zwingend fortlaufend sein.
    Entscheidend ist, dass Phase und Orientierung existieren.
    """

    grain_map = np.array(
        [
            [10, 10],
            [42, 42],
        ],
        dtype=np.int32,
    )

    voxel3d = grain_map[np.newaxis, :, :]

    grain_to_phase = {
        10: 0,
        42: 1,
    }

    grain_orientations = pd.DataFrame(
        {
            "grain_id": [10, 42],
        }
    )

    result = validate_damask_model(
        voxel3d=voxel3d,
        grain_map_2d=grain_map,
        grain_orientations=grain_orientations,
        grain_to_phase=grain_to_phase,
        cfg=Config(),
    )

    assert result["ok"] is True
    assert result["errors"] == []