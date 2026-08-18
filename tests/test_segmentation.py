import numpy as np

from micro2damask.config import Config
from micro2damask.segmentation import clean_segmentation


def test_clean_segmentation_returns_binary_array():
    phase = np.array(
        [
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    cfg = Config(
        remove_small_objects_min_size=0,
        remove_small_holes_area_threshold=0,
        do_opening=False,
        do_closing=False,
    )

    result = clean_segmentation(phase, cfg)

    assert result.shape == phase.shape

    assert set(np.unique(result)).issubset({0, 1})


def test_remove_small_light_phase_object():
    phase = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
        ],
        dtype=np.uint8,
    )

    cfg = Config(
        remove_small_objects_min_size=2,
        remove_small_holes_area_threshold=0,
        do_opening=False,
        do_closing=False,
    )

    result = clean_segmentation(phase, cfg)

    # Einzelpixel muss entfernt werden
    assert result[1, 1] == 0

    # 2x2-Bereich bleibt bestehen
    assert result[3, 3] == 1

def test_intensity_mapping_is_dark_zero_light_one():
    from micro2damask.segmentation import segment_microstructure

    image = np.array(
        [
            [10, 10, 240, 240],
            [10, 10, 240, 240],
        ],
        dtype=np.uint8,
    )

    result = segment_microstructure(
        image,
        Config(threshold_method="manual", manual_threshold=128),
    )["phase_map_raw"]

    assert np.all(result[:, :2] == 0)
    assert np.all(result[:, 2:] == 1)
