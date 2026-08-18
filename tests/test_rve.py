import numpy as np
import pytest

from micro2damask.rve import downsample_phase_map


def test_downsample_factor_one_returns_copy():
    phase = np.array(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=np.uint8,
    )

    result = downsample_phase_map(phase, factor=1)

    np.testing.assert_array_equal(result, phase)

    # Sicherstellen, dass es wirklich eine Kopie ist
    assert result is not phase


def test_downsample_majority_rule():
    phase = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 0],
        ],
        dtype=np.uint8,
    )

    result = downsample_phase_map(phase, factor=2)

    expected = np.array(
        [
            [1, 0],
            [0, 1],
        ],
        dtype=np.uint8,
    )

    np.testing.assert_array_equal(result, expected)


def test_downsample_exactly_fifty_percent_is_light_phase():
    """
    Laut Definition:
    >= 50 % helle Phase -> helle Phase
    """

    phase = np.array(
        [
            [1, 1],
            [0, 0],
        ],
        dtype=np.uint8,
    )

    result = downsample_phase_map(phase, factor=2)

    expected = np.array([[1]], dtype=np.uint8)

    np.testing.assert_array_equal(result, expected)


def test_downsample_invalid_factor_raises():
    phase = np.zeros((4, 4), dtype=np.uint8)

    with pytest.raises(ValueError):
        downsample_phase_map(phase, factor=0)


def test_downsample_non_divisible_dimensions():
    """
    Dieser Test muss an das tatsächliche Verhalten deiner Funktion angepasst
    werden.

    Falls Restpixel abgeschnitten werden:
    """

    phase = np.zeros((5, 5), dtype=np.uint8)

    result = downsample_phase_map(phase, factor=2)

    assert result.shape == (2, 2)