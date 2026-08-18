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
    axs[0].set_title("Original (Arbeitsbereich)")
    axs[0].add_patch(plt.Rectangle((x, y), w, h, edgecolor="yellow", facecolor="none", lw=1.5))
    add_scale_bar(axs[0], cfg.um_per_pixel, gray.shape)

    axs[1].imshow(gray, cmap="gray")
    axs[1].set_title("Graustufenbild")
    add_scale_bar(axs[1], cfg.um_per_pixel, gray.shape)

    axs[2].imshow(preprocessed, cmap="gray")
    axs[2].set_title("Vorverarbeitet (Denoise + CLAHE)")
    add_scale_bar(axs[2], cfg.um_per_pixel, gray.shape)

    axs[3].imshow(raw_binary, cmap="gray")
    axs[3].set_title(
        f"Rohsegmentierung\n0={cfg.dark_phase_name} (dunkel), 1={cfg.light_phase_name} (hell)"
    )

    axs[4].imshow(cleaned_phase_map, cmap="gray", vmin=0, vmax=1)
    axs[4].set_title("Bereinigte Segmentierung")

    cmap_phase = plt.cm.get_cmap("tab10", 2)
    im = axs[5].imshow(rve_phase_map, cmap=cmap_phase, vmin=0, vmax=1)
    axs[5].set_title(
        f"Finales RVE\nLR mismatch={periodic_info['lr_mismatch']:.3f}, "
        f"TB mismatch={periodic_info['tb_mismatch']:.3f}"
    )

    for ax in axs:
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px]")

    cbar = fig.colorbar(im, ax=axs[5], fraction=0.046, pad=0.04, ticks=[0, 1])
    cbar.ax.set_yticklabels([
        f"{cfg.dark_phase_name} (0, dunkel)",
        f"{cfg.light_phase_name} (1, hell)",
    ])

    fig.tight_layout()
    fig.savefig(out_dir / "images" / "pipeline_overview.png", dpi=cfg.figure_dpi)
    plt.close(fig)

    if grain_map is not None:
        dummy_image = np.zeros(grain_map.shape, dtype=np.uint8)
        rgb = color.label2rgb(
            grain_map,
            image=dummy_image,
            bg_label=-1,
            bg_color=(0, 0, 0),
            kind="avg",
        )
        fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
        ax2.imshow(rgb)
        ax2.set_title("Grain map (jedes Korn eigene Farbe)")
        ax2.set_xlabel("x [px]")
        ax2.set_ylabel("y [px]")

        if cfg.show_grain_labels and grain_stats_df is not None:
            for _, row in grain_stats_df.iterrows():
                ax2.text(
                    row["centroid_x_px"],
                    row["centroid_y_px"],
                    str(int(row["grain_id"])),
                    color="white",
                    fontsize=6,
                    ha="center",
                    va="center",
                )

        fig2.tight_layout()
        fig2.savefig(out_dir / "images" / "grain_map.png", dpi=cfg.figure_dpi)
        plt.close(fig2)
