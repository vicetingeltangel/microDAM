import numpy as np
import pytest

from micro2damask.geometry import (
    build_voxel_grid_3d_from_grain_map,
    _zyx_to_damask_xyz,
    _flatten_zyx_for_damask,
    write_vti_vtk_compressed,
)


def test_build_voxel_grid_shape():
    grain_map = np.array([[0, 1], [2, 3]], dtype=np.int32)
    voxel3d = build_voxel_grid_3d_from_grain_map(grain_map, 3)
    assert voxel3d.shape == (3, 2, 2)


def test_build_voxel_grid_repeats_grain_map():
    grain_map = np.array([[0, 1], [2, 3]], dtype=np.int32)
    voxel3d = build_voxel_grid_3d_from_grain_map(grain_map, nz=2)
    assert voxel3d.shape == (2, 2, 2)
    np.testing.assert_array_equal(voxel3d[0], grain_map)
    np.testing.assert_array_equal(voxel3d[1], grain_map)


def test_zyx_to_damask_xyz_is_explicit_axis_swap():
    a_zyx = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    a_xyz = _zyx_to_damask_xyz(a_zyx)

    assert a_xyz.shape == (4, 3, 2)
    for z in range(2):
        for y in range(3):
            for x in range(4):
                assert a_xyz[x, y, z] == a_zyx[z, y, x]


def test_flatten_matches_damask_geomgrid_convention():
    a_zyx = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    expected = a_zyx.transpose(2, 1, 0).flatten(order="F")
    np.testing.assert_array_equal(_flatten_zyx_for_damask(a_zyx), expected)
    # Useful invariant for an internal [z,y,x] C-order array.
    np.testing.assert_array_equal(_flatten_zyx_for_damask(a_zyx), a_zyx.flatten(order="C"))


def test_vti_fallback_roundtrip_preserves_axis_order(tmp_path):
    vtk = pytest.importorskip("vtk")
    from vtk.util.numpy_support import vtk_to_numpy

    # Deliberately non-cubic and asymmetric, so every axis swap is detectable.
    material_zyx = np.array(
        [
            [[0, 1, 2], [3, 4, 5]],
            [[6, 7, 8], [9, 10, 11]],
        ],
        dtype=np.int32,
    )  # (nz, ny, nx) = (2, 2, 3)

    out = tmp_path / "axis_roundtrip.vti"
    ok, msg = write_vti_vtk_compressed(
        material_zyx,
        None,
        out,
        spacing=(1.0e-6, 2.0e-6, 3.0e-6),
    )
    assert ok, msg

    raw = out.read_bytes()
    assert b"<AppendedData" not in raw
    assert b'format="binary"' in raw

    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(str(out))
    reader.Update()
    image = reader.GetOutput()

    # nx+1, ny+1, nz+1 VTK points -> nx,ny,nz cells.
    assert image.GetDimensions() == (4, 3, 3)
    assert image.GetNumberOfCells() == material_zyx.size
    assert image.GetCellData().GetArray("material") is not None

    flat = vtk_to_numpy(image.GetCellData().GetArray("material"))
    material_xyz = flat.reshape((3, 2, 2), order="F")
    roundtrip_zyx = material_xyz.transpose(2, 1, 0)

    np.testing.assert_array_equal(roundtrip_zyx, material_zyx)
    np.testing.assert_allclose(image.GetSpacing(), (1.0e-6, 2.0e-6, 3.0e-6))
