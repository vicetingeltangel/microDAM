from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import color

from .config import Config
from .utils import add_scale_bar


def plot_results(
    original_working: np.ndarray,
    gray: np.ndarray,
    preprocessed: np.ndarray,
    raw_binary: np.ndarray,
    cleaned_phase_map: np.ndarray,
    rve_phase_map: np.ndarray,
    rve_rect: Tuple[int, int, int, int],
    periodic_info: Dict[str, Any],
    cfg: Config,
    out_dir: Path,
    grain_map: Optional[np.ndarray] = None,
    grain_stats_df: Optional[pd.DataFrame] = None,
):
    x, y, w, h = rve_rect
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.ravel()

    if original_working.ndim == 3:
        axs[0].imshow(cv2.cvtColor(original_working, cv2.COLOR_BGR2RGB))
    else:
        axs[0].imshow(original_working, cmap="gray")
    axs[0].set_title("Original and RVE area")
    axs[0].add_patch(plt.Rectangle((x, y), w, h, edgecolor="yellow", facecolor="none", lw=1.5))
    add_scale_bar(axs[0], cfg.um_per_pixel, gray.shape)

    axs[1].imshow(gray, cmap="gray")
    axs[1].set_title("Gray Scale Image")
    add_scale_bar(axs[1], cfg.um_per_pixel, gray.shape)

    axs[2].imshow(preprocessed, cmap="gray")
    axs[2].set_title("Denoise + CLAHE")
    add_scale_bar(axs[2], cfg.um_per_pixel, gray.shape)

    axs[3].imshow(raw_binary, cmap="gray")
    axs[3].set_title(
        f"RAW\n0={cfg.dark_phase_name} (dunkel), 1={cfg.light_phase_name} (hell)"
    )

    axs[4].imshow(cleaned_phase_map, cmap="gray", vmin=0, vmax=1)
    axs[4].set_title("Filtered Segmentation")

    cmap_phase = plt.cm.get_cmap("tab10", 2)
    im = axs[5].imshow(rve_phase_map, cmap=cmap_phase, vmin=0, vmax=1)
    axs[5].set_title(
        f"Final 2D-RVE\nLR mismatch={periodic_info['lr_mismatch']:.3f}, "
        f"TB mismatch={periodic_info['tb_mismatch']:.3f}"
    )

    for ax in axs:
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px]")

    cbar = fig.colorbar(im, ax=axs[5], fraction=0.046, pad=0.04, ticks=[0, 1])
    cbar.ax.set_yticklabels([
        f"{cfg.dark_phase_name}",
        f"{cfg.light_phase_name}",
    ])

    fig.tight_layout()
    fig.savefig(out_dir / "images" / "pipeline_overview.png", dpi=cfg.figure_dpi)
    plt.close(fig)

    #if grain_map is not None:
        # dummy_image = np.zeros(grain_map.shape, dtype=np.uint8)
        # rgb = color.label2rgb(
        #     grain_map,
        #     image=dummy_image,
        #     bg_label=-1,
        #     bg_color=(0, 0, 0),
        #     kind="avg",
        # )
        # fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
        # ax2.imshow(rgb)
        # ax2.set_title("Grain map")
        # ax2.set_xlabel("x [px]")
        # ax2.set_ylabel("y [px]")
        #
        # if cfg.show_grain_labels and grain_stats_df is not None:
        #     for _, row in grain_stats_df.iterrows():
        #         ax2.text(
        #             row["centroid_x_px"],
        #             row["centroid_y_px"],
        #             str(int(row["grain_id"])),
        #             color="white",
        #             fontsize=6,
        #             ha="center",
        #             va="center",
        #         )
        #
        # fig2.tight_layout()
        # fig2.savefig(out_dir / "images" / "grain_map.png", dpi=cfg.figure_dpi)
        # plt.close(fig2)



# DAMASK/MTEX-style cubic inverse-pole-figure colors.
# The fundamental triangle is [001] -> red, [101] -> green, [111] -> blue.
_CUBIC_IPF_BASIS = np.array(
    [
        [-1.0, 0.0, 1.0],
        [np.sqrt(2.0), -np.sqrt(2.0), 0.0],
        [0.0, np.sqrt(3.0), 0.0],
    ],
    dtype=float,
)


def _cubic_ipf_rgb_from_crystal_direction(crystal_direction: np.ndarray) -> np.ndarray:
    """Map crystal-frame directions to DAMASK-style cubic IPF RGB colors."""
    v = np.asarray(crystal_direction, dtype=float)
    if v.shape[-1] != 3:
        raise ValueError("crystal_direction muss auf der letzten Achse Länge 3 haben.")

    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(norm == 0.0):
        raise ValueError("IPF-Richtung darf kein Nullvektor sein.")
    v = v / norm

    # Cubic symmetry can map every pole to the standard stereographic triangle.
    # For DAMASK's improper triangle the canonical component order is y <= x <= z.
    ordered = np.sort(np.abs(v), axis=-1)
    sst = np.stack([ordered[..., 1], ordered[..., 0], ordered[..., 2]], axis=-1)

    components = np.einsum("ij,...j->...i", _CUBIC_IPF_BASIS, sst)
    components = np.clip(components, 0.0, None)

    with np.errstate(invalid="ignore", divide="ignore"):
        rgb = (components / np.linalg.norm(components, axis=-1, keepdims=True)) ** (1.0 / 3.0)
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb /= np.max(rgb, axis=-1, keepdims=True)

    return np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)


def cubic_ipf_colors_from_bunge(
    euler_deg: np.ndarray,
    lab_direction: np.ndarray,
) -> np.ndarray:
    """
    Calculate cubic IPF colors from Bunge Euler angles.

    ``lab_direction`` is a specimen/lab-frame direction, e.g. RD=[1,0,0],
    TD=[0,1,0], or ND=[0,0,1].  SciPy rotations are active; applying the
    inverse gives the same passive frame conversion used by DAMASK.
    """
    from scipy.spatial.transform import Rotation as R

    euler = np.asarray(euler_deg, dtype=float)
    if euler.ndim == 1:
        euler = euler.reshape(1, 3)
    if euler.shape[-1] != 3:
        raise ValueError("euler_deg muss die Form (..., 3) besitzen.")
    if not np.all(np.isfinite(euler)):
        raise ValueError("IPF-Darstellung benötigt endliche Eulerwinkel.")

    direction = np.asarray(lab_direction, dtype=float)
    if direction.shape != (3,) or np.linalg.norm(direction) == 0.0:
        raise ValueError("lab_direction muss ein dreikomponentiger Nicht-Nullvektor sein.")
    direction = direction / np.linalg.norm(direction)

    rotations = R.from_euler("ZXZ", euler, degrees=True)
    lab = np.broadcast_to(direction, (len(euler), 3)).copy()
    crystal = rotations.apply(lab, inverse=True)
    return _cubic_ipf_rgb_from_crystal_direction(crystal)


def build_cubic_ipf_map(
    grain_map: np.ndarray,
    grain_orientations: pd.DataFrame,
    lab_direction: np.ndarray,
) -> np.ndarray:
    """Create an RGB IPF map by assigning one color to every grain ID."""
    required = {"grain_id", "phi1_deg", "Phi_deg", "phi2_deg"}
    missing = required.difference(grain_orientations.columns)
    if missing:
        raise ValueError(f"grain_orientations fehlen Spalten: {sorted(missing)}")

    gm = np.asarray(grain_map)
    rgb_map = np.zeros(gm.shape + (3,), dtype=float)

    table = grain_orientations.sort_values("grain_id")
    euler = table[["phi1_deg", "Phi_deg", "phi2_deg"]].to_numpy(dtype=float)
    colors = cubic_ipf_colors_from_bunge(euler, lab_direction)

    for gid, color_rgb in zip(table["grain_id"].astype(int).to_numpy(), colors):
        rgb_map[gm == gid] = color_rgb

    return rgb_map


def save_ipf_orientation_maps(
    grain_map: np.ndarray,
    grain_orientations: pd.DataFrame,
    cfg: Config,
    out_dir: Path,
) -> Dict[str, str]:
    """
    Save cubic inverse-pole-figure maps for rolling/sample axes.

    Output directions follow the image/RVE coordinate convention:
      RD / X = [1,0,0]
      TD / Y = [0,1,0]
      ND / Z = [0,0,1]

    The coloring follows DAMASK's cubic standard triangle:
    [001] red, [101] green, [111] blue.
    """
    # This implementation is intentionally restricted to cubic phases because
    # the MTEX ODF importer currently targets m-3m data. Never silently apply a
    # cubic color key to a non-cubic phase.
    for phase_id, material in ((0, cfg.dark_phase_material), (1, cfg.light_phase_material)):
        if material is None:
            continue
        lattice = material.get("lattice")
        if lattice not in {"cP", "cI", "cF"}:
            raise ValueError(
                f"IPF-Farbkarte unterstützt derzeit nur kubische Phasen; "
                f"Phase '{cfg.phase_name(phase_id)}' verwendet lattice={lattice!r}."
            )

    directions = {
        "RD_x": np.array([1.0, 0.0, 0.0]),
        "TD_y": np.array([0.0, 1.0, 0.0]),
        "ND_z": np.array([0.0, 0.0, 1.0]),
    }
    maps = {
        name: build_cubic_ipf_map(grain_map, grain_orientations, direction)
        for name, direction in directions.items()
    }

    image_dir = Path(out_dir) / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    for name, rgb in maps.items():
        path = image_dir / f"ipf_{name}.png"
        plt.imsave(path, rgb)
        paths[name] = str(path)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    labels = {
        "RD_x": "IPF || RD / X",
        "TD_y": "IPF || TD / Y",
        "ND_z": "IPF || ND / Z",
    }
    for ax, name in zip(axs, directions):
        ax.imshow(maps[name])
        ax.set_title(labels[name])
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px]")
        ax.set_aspect("equal")

    overview = image_dir / "ipf_orientation_maps.png"
    fig.savefig(overview, dpi=cfg.figure_dpi)
    plt.close(fig)
    paths["overview"] = str(overview)
    return paths
