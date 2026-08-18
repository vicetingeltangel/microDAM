from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List
import json
import time
import uuid
import warnings
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

from .config import Config
from .utils import spacing_from_cfg
from .rve import build_voxel_grid_3d_from_grain_map

def write_vti_vtk_compressed(material_3d: np.ndarray,
                              phase_3d: Optional[np.ndarray],
                              out_vti_path: Path,
                              spacing=(1.0, 1.0, 1.0),
                              origin=(0.0, 0.0, 0.0)) -> Tuple[bool, str]:
    """
    Schreibt eine .vti im DAMASK-kompatiblen Inline-Binary-Format (ZLib-komprimiert,
    Base64-inline, KEIN <AppendedData>-Block -- das entspricht dem nativen
    DAMASK-Referenzformat). 'phase_3d' ist optional und rein informativ (wird von
    DAMASK selbst nicht benötigt -- die Phasenzuordnung steht in material.yaml).
    """
    out_vti_path = Path(out_vti_path)
    out_vti_path.parent.mkdir(parents=True, exist_ok=True)

    if phase_3d is not None and material_3d.shape != phase_3d.shape:
        return False, f"Shape mismatch: material {material_3d.shape} vs phase {phase_3d.shape}"

    try:
        import vtk  # type: ignore
        from vtk.util.numpy_support import numpy_to_vtk  # type: ignore
    except Exception as e:
        return False, f"vtk nicht verfügbar: {e}"

    nz, ny, nx = material_3d.shape

    img = vtk.vtkImageData()
    img.SetExtent(0, int(nx), 0, int(ny), 0, int(nz))
    img.SetOrigin(float(origin[0]), float(origin[1]), float(origin[2]))
    img.SetSpacing(float(spacing[0]), float(spacing[1]), float(spacing[2]))

    flat_mat = np.ascontiguousarray(material_3d.ravel(order='F')).astype(np.int64)

    vtk_type = None
    for candidate in ("VTK_LONG_LONG", "VTK_INT64", "VTK_LONG"):
        if hasattr(vtk, candidate):
            vtk_type = getattr(vtk, candidate)
            break
    if vtk_type is None:
        vtk_type = vtk.VTK_INT

    vtk_mat = numpy_to_vtk(num_array=flat_mat, deep=True, array_type=vtk_type)
    vtk_mat.SetName("material")

    cd = img.GetCellData()
    cd.AddArray(vtk_mat)
    cd.SetScalars(vtk_mat)

    if phase_3d is not None:
        flat_phase = np.ascontiguousarray(phase_3d.ravel(order='F')).astype(np.int64)
        vtk_phase = numpy_to_vtk(num_array=flat_phase, deep=True, array_type=vtk_type)
        vtk_phase.SetName("phase")
        cd.AddArray(vtk_phase)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(str(out_vti_path))

    # WICHTIG: Inline-Binary (nicht Appended!) -- entspricht dem Format,
    # das DAMASKs Reader erwartet (kein <AppendedData>-Block).
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
        if ok:
            return True, f"VTI geschrieben (vtk, inline binary): {out_vti_path}"
        return False, "vtk writer meldete Fehler (Write() == 0)"
    except Exception as e:
        return False, f"vtk writer exception: {e}"


def write_vti_with_material_and_phase(material_3d: np.ndarray,
                                       phase_3d: Optional[np.ndarray],
                                       out_vti_path: Path,
                                       spacing: tuple = (1.0, 1.0, 1.0),
                                       origin: tuple = (0.0, 0.0, 0.0)) -> Tuple[bool, str]:
    """Versucht pyevtk (schreibt standardmäßig Inline-Binary), sonst vtk-Fallback."""
    out_vti_path = Path(out_vti_path)
    out_vti_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from pyevtk.hl import imageToVTK  # type: ignore
        mat_p = np.transpose(material_3d, (2, 1, 0)).astype(np.int32)
        cell_data = {"material": mat_p}
        if phase_3d is not None:
            cell_data["phase"] = np.transpose(phase_3d, (2, 1, 0)).astype(np.int32)
        stem = str(out_vti_path.with_suffix('').absolute())
        imageToVTK(stem, origin=origin, spacing=spacing, cellData=cell_data)
        return True, f"VTI geschrieben (pyevtk): {out_vti_path}"
    except Exception as e_pye:
        warnings.warn(f"pyevtk nicht verfügbar/fehlgeschlagen ({e_pye}), versuche vtk...")

    return write_vti_vtk_compressed(material_3d, phase_3d, out_vti_path, spacing=spacing, origin=origin)


def try_export_with_damask_api(voxel3d: np.ndarray, out_dir: Path, cfg: Config) -> Tuple[bool, str]:
    """
    Schreibt geometry/damask_geom.vti über die offizielle damask.GeomGrid-API.
    Das garantiert ein Format, das DAMASK_grid direkt lesen kann.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import damask  # type: ignore
    except Exception:
        np.save(out_dir / "grain_id_grid_nz_ny_nx.npy", voxel3d.astype(np.int32))
        return False, "DAMASK Python-Modul nicht installiert; NPY-Fallback erzeugt."

    try:
        np.save(out_dir / "grain_id_grid_nz_ny_nx.npy", voxel3d.astype(np.int32))
        if not hasattr(damask, "GeomGrid"):
            return True, "DAMASK installiert, aber GeomGrid-Klasse fehlt; NPY-Fallback gespeichert."

        ny, nx = voxel3d.shape[1], voxel3d.shape[2]
        nz = voxel3d.shape[0]

        # WICHTIG: DAMASK erwartet 'size' in Metern (SI) -- µm/px werden hier konvertiert!
        sx, sy, sz = spacing_from_cfg(cfg)
        size = np.array([nx * sx, ny * sy, nz * sz], dtype=float)

        try:
            g = damask.GeomGrid(material=np.transpose(voxel3d, (2, 1, 0)), size=size)
        except Exception:
            g = damask.GeomGrid(material=voxel3d, size=size)

        if hasattr(g, "save"):
            g.save(str(out_dir / "damask_geom.vti"))
            return True, "DAMASK GeomGrid erfolgreich exportiert (save)."
        elif hasattr(g, "to_vtk"):
            g.to_vtk(str(out_dir / "damask_geom.vti"))
            return True, "DAMASK GeomGrid erfolgreich exportiert (to_vtk)."

        return True, "DAMASK installiert, jedoch GeomGrid-Export nicht möglich; NPY-Fallback gespeichert."
    except Exception as e:
        np.save(out_dir / "grain_id_grid_nz_ny_nx.npy", voxel3d.astype(np.int32))
        return False, f"DAMASK-Export fehlgeschlagen: {e}. NPY-Fallback gespeichert."


def create_damask_geometry_from_grains(grain_map_2d: np.ndarray, phase_map_2d: np.ndarray,
                                        cfg: Config, out_dir: Path) -> Dict[str, Any]:
    voxel3d_material = build_voxel_grid_3d_from_grain_map(grain_map_2d, cfg.nz_layers)
    phase3d = np.repeat(phase_map_2d[np.newaxis, :, :], repeats=cfg.nz_layers, axis=0).astype(np.int32)

    np.save(out_dir / "geometry" / "grain_map_3d_nz_ny_nx.npy", voxel3d_material.astype(np.int32))
    np.save(out_dir / "geometry" / "phase_map_3d_nz_ny_nx.npy", phase3d.astype(np.int32))

    # 1. Bevorzugt: offizielle DAMASK-API -> garantiert kompatibles Format.
    ok_api, msg_api = try_export_with_damask_api(voxel3d_material, out_dir / "geometry", cfg)
    damask_native_path = out_dir / "geometry" / "damask_geom.vti"

    result: Dict[str, Any] = {
        "voxel_shape_nz_ny_nx": tuple(voxel3d_material.shape),
        "n_materials": int(voxel3d_material.max() + 1),
    }
    if ok_api and damask_native_path.exists():
        result["damask_export_ok"] = True
        result["damask_message"] = msg_api
        result["vti_path"] = str(damask_native_path)  # <- diese Datei für DAMASK_grid verwenden!
    else:
        # 2. Fallback: manueller Writer (inline binary, s.o.)
        fallback_path = out_dir / "geometry" / "rve_material_only.vti"
        ok_manual, msg_manual = write_vti_with_material_and_phase(
            voxel3d_material, None, fallback_path, spacing=spacing_from_cfg(cfg))
        result["damask_export_ok"] = bool(ok_manual)
        result["damask_message"] = f"{msg_api} | Fallback: {msg_manual}"
        result["vti_path"] = str(fallback_path) if ok_manual else None

    # 3. Zusätzliches Debug-VTI NUR fürs Visualisieren in ParaView (material + phase).
    #    Wird NICHT an die Simulation übergeben.
    debug_path = out_dir / "geometry" / "debug_material_and_phase.vti"
    ok_debug, msg_debug = write_vti_with_material_and_phase(
        voxel3d_material, phase3d, debug_path, spacing=spacing_from_cfg(cfg))
    result["debug_vti_path"] = str(debug_path) if ok_debug else None
    result["debug_vti_message"] = msg_debug

    return result

