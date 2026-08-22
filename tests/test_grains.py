import numpy as np

from micro2damask.config import Config
from micro2damask.grains import handle_small_grains, identify_grains


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

def test_small_grain_fallback_merges_into_nearest_other_same_phase_grain():
    """Regression test: the KD-tree fallback must not select the small grain itself."""
    grain_map = np.zeros((11, 11), dtype=np.int32)
    phase_map = np.ones((11, 11), dtype=np.uint8)

    # A sufficiently large phase-0 grain far enough away that the radius-3
    # dilation around the tiny grain does not see it.
    grain_map[:, :4] = 1
    phase_map[:, :4] = 0

    # Tiny isolated phase-0 grain. Under the old implementation its pixels
    # were part of the KD-tree candidate set and therefore selected themselves.
    grain_map[5, 8] = 2
    phase_map[5, 8] = 0

    cfg = Config(min_grain_size=2, small_grain_mode="merge")
    merged = handle_small_grains(grain_map, phase_map, cfg)

    assert merged[5, 8] == 1
    assert 2 not in np.unique(merged)


def test_small_grain_merge_never_changes_phase_when_no_same_phase_target_exists():
    """A lone grain of a phase is kept instead of being merged across phases."""
    grain_map = np.zeros((7, 7), dtype=np.int32)
    phase_map = np.ones((7, 7), dtype=np.uint8)

    grain_map[3, 3] = 1
    phase_map[3, 3] = 0

    cfg = Config(min_grain_size=2, small_grain_mode="merge")
    merged = handle_small_grains(grain_map, phase_map, cfg)

    assert merged[3, 3] == 1
