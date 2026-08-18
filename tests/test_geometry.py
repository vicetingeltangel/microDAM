import numpy as np

from micro2damask.geometry import build_voxel_grid_3d_from_grain_map


def test_build_voxel_grid_shape():

    grain_map = np.array(
        [
            [0, 1],
            [2, 3],
        ],
        dtype=np.int32,
    )

    nz_layers = 3

    voxel3d = build_voxel_grid_3d_from_grain_map(
        grain_map,
        nz_layers,
    )

    assert voxel3d.shape == (3, 2, 2)


def test_build_voxel_grid_repeats_grain_map():

    grain_map = np.array(
        [
            [0, 1],
            [2, 3],
        ],
        dtype=np.int32,
    )

    voxel3d = build_voxel_grid_3d_from_grain_map(
        grain_map,
        nz= 2,
    )

    assert voxel3d.shape == (2, 2, 2)

    np.testing.assert_array_equal(
        voxel3d[0],
        grain_map,
    )

    np.testing.assert_array_equal(
        voxel3d[1],
        grain_map,
    )