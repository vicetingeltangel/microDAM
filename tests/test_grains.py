import numpy as np

from micro2damask.config import Config
from micro2damask.grains import identify_grains


def test_connected_components_create_distinct_grains():
    """
    Zwei nicht zusammenhängende Bereiche der hellen Phase müssen unterschiedliche
    Grain-IDs erhalten.
    """

    phase = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 0, 0],
        ],
        dtype=np.uint8,
    )

    grain_map, n_grains, grain_to_phase = identify_grains(
        phase,
        Config(connectivity=4),
    )

    assert n_grains >= 2

    top_right_id = grain_map[0, 2]
    bottom_left_id = grain_map[2, 0]

    assert top_right_id != bottom_left_id

    assert grain_to_phase[top_right_id] == 1
    assert grain_to_phase[bottom_left_id] == 1


def test_single_connected_region_is_one_grain():
    phase = np.ones((4, 4), dtype=np.uint8)

    grain_map, n_grains, grain_to_phase = identify_grains(
        phase,
        Config(connectivity=4),
    )

    assert n_grains == 1

    assert len(np.unique(grain_map)) == 1

    grain_id = grain_map[0, 0]

    assert grain_to_phase[grain_id] == 1


def test_two_phases_are_both_represented():
    phase = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.uint8,
    )

    grain_map, n_grains, grain_to_phase = identify_grains(
        phase,
        Config(connectivity=4),
    )

    assert n_grains == 2

    unique_phases = set(grain_to_phase.values())

    assert unique_phases == {0, 1}

def test_diagonal_connectivity_changes_result():
    """
    Zwei diagonal berührende Pixel:

    connectivity=4:
        zwei getrennte Körner

    connectivity=8:
        ein Korn
    """

    phase = np.array(
        [
            [1, 0],
            [0, 1],
        ],
        dtype=np.uint8,
    )

    _, n_grains_4, _ = identify_grains(
        phase,
        Config(connectivity=4),
    )

    _, n_grains_8, _ = identify_grains(
        phase,
        Config(connectivity=8),
    )

    assert n_grains_4 > n_grains_8