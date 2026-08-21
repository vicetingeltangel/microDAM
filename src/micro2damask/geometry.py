from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import numpy as np

from .config import Config
from .utils import spacing_from_cfg
from .rve import build_voxel_grid_3d_from_grain_map


# Internal convention used throughout micro2damask:
#   array.shape == (nz, ny, nx)  -> indexing [z, y, x]
# DAMASK GeomGrid convention:
#   material.shape == (nx, ny, nz) -> indexing [x, y, z]
#
# Keep the conversion at the I/O boundary only. This avoids implicit axis swaps
# in the rest of the image-processing pipeline.
def _validate_zyx_grid(array: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim != 3:
        raise ValueError(f"{name} muss 3D sein und die Form (nz, ny, nx) haben; erhalten: {arr.shape}")
    if any(n <= 0 for n in arr.shape):
        raise ValueError(f"{name} darf keine leere Achse enthalten; erhalten: {arr.shape}")
    return arr


def _zyx_to_damask_xyz(array_zyx: np.ndarray) -> np.ndarray:
    """Convert internal [z,y,x] storage to DAMASK/VTK [x,y,z] storage."""
    arr = _validate_zyx_grid(array_zyx, "array_zyx")
    return np.transpose(arr, (2, 1, 0))


def _flatten_zyx_for_damask(array_zyx: np.ndarray) -> np.ndarray:
    """
    Flatten internal [z,y,x] data exactly as damask.GeomGrid.save() does.

    DAMASK stores a GeomGrid with shape [x,y,z] and writes
    ``material.flatten(order='F')``. Therefore, for our internal [z,y,x]
    representation, the equivalent operation is:

        array_zyx.transpose(2,1,0).flatten(order='F')

    (which is also equal to array_zyx.flatten(order='C')).
    """
    return _zyx_to_damask_xyz(array_zyx).flatten(order="F")


def write_vti_vtk_compressed(
    material_3d: np.ndarray,
    phase_3d: Optional[np.ndarray],
    out_vti_path: Path,
    spacing=(1.0, 1.0, 1.0),
    origin=(0.0, 0.0, 0.0),
) -> Tuple[bool, str]:
    """
    Write a DAMASK-style VTK XML ImageData file using VTK.

    micro2damask accepts arrays in internal ``(nz, ny, nx)`` order. At this
    I/O boundary they are converted to DAMASK's ``(nx, ny, nz)`` convention.

    The writer intentionally uses *inline binary* data (optionally zlib
    compressed), not VTK ``AppendedData``. Material IDs are attached as
    ``CellData`` to mirror the current implementation of
    ``damask.GeomGrid.save()``.

    ``phase_3d`` is optional and only intended as an informational/debug field;
    phase assignment for DAMASK is defined in ``material.yaml``.
    """
    out_vti_path = Path(out_vti_path)
    out_vti_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        material_zyx = _validate_zyx_grid(material_3d, "material_3d")
        phase_zyx = None if phase_3d is None else _validate_zyx_grid(phase_3d, "phase_3d")
    except ValueError as e:
        return False, str(e)

    if phase_zyx is not None and material_zyx.shape != phase_zyx.shape:
        return False, f"Shape mismatch: material {material_zyx.shape} vs phase {phase_zyx.shape}"

    if len(spacing) != 3 or any(float(v) <= 0.0 for v in spacing):
        return False, f"Ungültiges spacing={spacing}; erwartet werden drei positive Werte (sx, sy, sz)."
    if len(origin) != 3:
        return False, f"Ungültiger origin={origin}; erwartet werden drei Werte (x, y, z)."

    try:
        import vtk  # type: ignore
        from vtk.util.numpy_support import numpy_to_vtk  # type: ignore
    except Exception as e:
        return False, f"vtk nicht verfügbar: {e}"

    # DAMASK material shape is [x,y,z]. Our internal shape is [z,y,x].
    nx, ny, nz = _zyx_to_damask_xyz(material_zyx).shape

    img = vtk.vtkImageData()
    # nx * ny * nz cells require (nx+1) * (ny+1) * (nz+1) grid points.
    img.SetDimensions(int(nx) + 1, int(ny) + 1, int(nz) + 1)
    img.SetOrigin(float(origin[0]), float(origin[1]), float(origin[2]))
    img.SetSpacing(float(spacing[0]), float(spacing[1]), float(spacing[2]))

    flat_mat = np.ascontiguousarray(_flatten_zyx_for_damask(material_zyx), dtype=np.int64)

    vtk_type = None
    for candidate in ("VTK_LONG_LONG", "VTK_INT64", "VTK_LONG"):
        if hasattr(vtk, candidate):
            vtk_type = getattr(vtk, candidate)
            break
    if vtk_type is None:
        vtk_type = vtk.VTK_INT

    vtk_mat = numpy_to_vtk(num_array=flat_mat, deep=True, array_type=vtk_type)
    vtk_mat.SetName("material")

    # Match current damask.GeomGrid.save(): material is cell-centered data.
    cell_data = img.GetCellData()
    cell_data.AddArray(vtk_mat)
    cell_data.SetScalars(vtk_mat)

    if phase_zyx is not None:
        flat_phase = np.ascontiguousarray(_flatten_zyx_for_damask(phase_zyx), dtype=np.int64)
        vtk_phase = numpy_to_vtk(num_array=flat_phase, deep=True, array_type=vtk_type)
        vtk_phase.SetName("phase")
        cell_data.AddArray(vtk_phase)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(str(out_vti_path))

    # DAMASK's grid solver does not support VTK AppendedData. Binary mode writes
    # inline base64 binary data; zlib compression is enabled when supported.
    try:
        writer.SetDataModeToBinary()
    except Exception:
        pass
    try:
        if hasattr(writer, "SetCompressorTypeToZLib"):
            writer.SetCompressorTypeToZLib()
    except Exception:
        pass
    try:
        if hasattr(writer, "SetHeaderTypeToUInt32"):
            writer.SetHeaderTypeToUInt32()
    except Exception:
        pass

    try:
        try:
            writer.SetInputData(img)
        except AttributeError:
            writer.SetInput(img)
        ok = writer.Write()
        if not ok:
            return False, "vtk writer meldete Fehler (Write() == 0)"

        # Structural guard: an accidentally changed writer mode must not
        # silently create an AppendedData file that DAMASK cannot read. Search
        # in chunks so large VTI files are not loaded fully into memory.
        try:
            token = b"<AppendedData"
            overlap = b""
            with out_vti_path.open("rb") as fh:
                while chunk := fh.read(1024 * 1024):
                    data = overlap + chunk
                    if token in data:
                        return False, (
                            "VTI enthält unerwartet <AppendedData>; Datei ist nicht "
                            "als DAMASK-Fallback freigegeben."
                        )
                    overlap = data[-(len(token) - 1):]
        except OSError:
            pass

        return True, f"VTI geschrieben (vtk, inline binary, DAMASK axis order): {out_vti_path}"
    except Exception as e:
        return False, f"vtk writer exception: {e}"


def write_vti_with_material_and_phase(
    material_3d: np.ndarray,
    phase_3d: Optional[np.ndarray],
    out_vti_path: Path,
    spacing: tuple = (1.0, 1.0, 1.0),
    origin: tuple = (0.0, 0.0, 0.0),
) -> Tuple[bool, str]:
    """
    Write a DAMASK-compatible fallback VTI using VTK only.

    PyEVTK is deliberately *not* used here: its ``imageToVTK`` writer stores
    arrays in an ``AppendedData`` section, while DAMASK's grid solver explicitly
    does not support appended VTK XML data.
    """
    return write_vti_vtk_compressed(
        material_3d,
        phase_3d,
        out_vti_path,
        spacing=spacing,
        origin=origin,
    )


def try_export_with_damask_api(voxel3d: np.ndarray, out_dir: Path, cfg: Config) -> Tuple[bool, str]:
    """
    Write ``geometry/damask_geom.vti`` through the official DAMASK GeomGrid API.

    ``voxel3d`` uses micro2damask's internal ``(nz, ny, nx)`` order and is
    converted exactly once to DAMASK's ``(nx, ny, nz)`` order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    voxel_zyx = _validate_zyx_grid(voxel3d, "voxel3d")
    np.save(out_dir / "grain_id_grid_nz_ny_nx.npy", voxel_zyx.astype(np.int32))

    try:
        import damask  # type: ignore
    except Exception:
        return False, "DAMASK Python-Modul nicht installiert; NPY-Fallback erzeugt."

    if not hasattr(damask, "GeomGrid"):
        return False, "DAMASK installiert, aber GeomGrid-Klasse fehlt; NPY-Fallback gespeichert."

    try:
        material_xyz = _zyx_to_damask_xyz(voxel_zyx)

        # spacing_from_cfg returns (sx, sy, sz) in metres.
        spacing = np.asarray(spacing_from_cfg(cfg), dtype=float)
        if spacing.shape != (3,) or np.any(spacing <= 0.0):
            raise ValueError(f"invalid spacing {spacing}")
        size = np.asarray(material_xyz.shape, dtype=float) * spacing

        # Do not retry with the internal z,y,x array: that would silently swap
        # physical axes for non-cubic grids.
        g = damask.GeomGrid(material=material_xyz, size=size)

        target = out_dir / "damask_geom.vti"
        if hasattr(g, "save"):
            g.save(str(target))
        elif hasattr(g, "to_vtk"):
            g.to_vtk(str(target))
        else:
            return False, "DAMASK GeomGrid besitzt weder save() noch to_vtk(); NPY-Fallback gespeichert."

        if not target.exists():
            return False, f"DAMASK meldete keinen Fehler, aber {target.name} wurde nicht erzeugt."

        # If available, verify a roundtrip through DAMASK itself. This catches
        # axis regressions immediately and is stronger than merely checking that
        # a file exists.
        if hasattr(damask.GeomGrid, "load"):
            loaded = damask.GeomGrid.load(str(target))
            loaded_material = np.asarray(loaded.material)
            if loaded_material.shape != material_xyz.shape or not np.array_equal(loaded_material, material_xyz):
                return False, (
                    "DAMASK GeomGrid roundtrip fehlgeschlagen: exportierte Materialanordnung "
                    f"{loaded_material.shape} stimmt nicht mit {material_xyz.shape} überein."
                )

        return True, "DAMASK GeomGrid erfolgreich exportiert und Achsenkonvention geprüft."
    except Exception as e:
        return False, f"DAMASK-Export fehlgeschlagen: {e}. NPY-Fallback gespeichert."


def create_damask_geometry_from_grains(
    grain_map_2d: np.ndarray,
    phase_map_2d: np.ndarray,
    cfg: Config,
    out_dir: Path,
) -> Dict[str, Any]:
    voxel3d_material = build_voxel_grid_3d_from_grain_map(grain_map_2d, cfg.nz_layers)
    phase3d = np.repeat(phase_map_2d[np.newaxis, :, :], repeats=cfg.nz_layers, axis=0).astype(np.int32)

    geometry_dir = out_dir / "geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    np.save(geometry_dir / "grain_map_3d_nz_ny_nx.npy", voxel3d_material.astype(np.int32))
    np.save(geometry_dir / "phase_map_3d_nz_ny_nx.npy", phase3d.astype(np.int32))

    # 1. Preferred path: official DAMASK API.
    ok_api, msg_api = try_export_with_damask_api(voxel3d_material, geometry_dir, cfg)
    damask_native_path = geometry_dir / "damask_geom.vti"

    result: Dict[str, Any] = {
        "voxel_shape_nz_ny_nx": tuple(voxel3d_material.shape),
        "n_materials": int(voxel3d_material.max() + 1),
    }

    if ok_api and damask_native_path.exists():
        result["damask_export_ok"] = True
        result["damask_message"] = msg_api
        result["vti_path"] = str(damask_native_path)
    else:
        # 2. Fallback: VTK inline-binary writer that mirrors DAMASK's current
        #    GeomGrid data layout and axis order.
        fallback_path = geometry_dir / "rve_material_only.vti"
        ok_manual, msg_manual = write_vti_with_material_and_phase(
            voxel3d_material,
            None,
            fallback_path,
            spacing=spacing_from_cfg(cfg),
        )
        result["damask_export_ok"] = bool(ok_manual)
        result["damask_message"] = f"{msg_api} | Fallback: {msg_manual}"
        result["vti_path"] = str(fallback_path) if ok_manual else None

    # 3. Additional debug VTI for ParaView. It uses the same axis convention,
    #    but includes the phase array as well.
    debug_path = geometry_dir / "debug_material_and_phase.vti"
    ok_debug, msg_debug = write_vti_with_material_and_phase(
        voxel3d_material,
        phase3d,
        debug_path,
        spacing=spacing_from_cfg(cfg),
    )
    result["debug_vti_path"] = str(debug_path) if ok_debug else None
    result["debug_vti_message"] = msg_debug

    return result
