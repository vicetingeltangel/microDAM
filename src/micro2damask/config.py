from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Config:
    # ---------------------------
    # Input
    # ---------------------------
    image_path: Optional[str] = None
    um_per_pixel: float = 1.0

    # Optional crop
    crop_x: Optional[int] = None
    crop_y: Optional[int] = None
    crop_w: Optional[int] = None
    crop_h: Optional[int] = None

    # ---------------------------
    # Phase convention
    # ---------------------------
    # Canonical representation used throughout the project:
    #   phase_id = 0 -> dark phase in the image
    #   phase_id = 1 -> light phase in the image
    dark_phase_name: str = "dark_phase"
    light_phase_name: str = "light_phase"

    # Full DAMASK phase definitions. They are intentionally not guessed:
    # arbitrary phase names must not silently inherit Al/Si material laws.
    dark_phase_material: Optional[Dict[str, Any]] = None
    light_phase_material: Optional[Dict[str, Any]] = None

    # ---------------------------
    # Preprocessing
    # ---------------------------
    denoise_method: str = "median"
    median_ksize: int = 3
    gaussian_sigma: float = 1.0

    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    do_background_correction: bool = False
    bg_blur_sigma: float = 25.0

    # ---------------------------
    # Segmentation
    # ---------------------------
    threshold_method: str = "otsu"
    manual_threshold: Optional[int] = None
    adaptive_block_size: int = 51
    adaptive_C: float = 2.0

    # Normally 0=dark and 1=light. Set True only if the phase assignment
    # shall deliberately be swapped after thresholding.
    invert_binary_after_threshold: bool = False

    remove_small_objects_min_size: int = 0
    remove_small_holes_area_threshold: int = 0

    # Morphology is applied to one selected phase only.
    morphology_target_phase: str = "light"  # "dark" or "light"
    do_opening: bool = False
    do_closing: bool = False
    morph_disk_radius: int = 1

    # ---------------------------
    # RVE
    # ---------------------------
    rve_w: int = 512
    rve_h: int = 512
    rve_x: Optional[int] = None
    rve_y: Optional[int] = None

    periodicity_tolerance: float = 0.10
    enable_dendrite_mode: bool = False
    dendrite_min_distance: int = 5
    dendrite_phase_id: int = 0

    # ---------------------------
    # Voxelization
    # ---------------------------
    downsample_factor: int = 1
    nz_layers: int = 1
    voxel_size_um: Optional[float] = None

    # ---------------------------
    # Grain identification
    # ---------------------------
    connectivity: int = 4
    min_grain_size: int = 5
    small_grain_mode: str = "keep"

    # ---------------------------
    # Orientation
    # ---------------------------
    dark_phase_orientation_mode: str = "random"
    light_phase_orientation_mode: str = "random"
    random_seed: Optional[int] = 420

    # ---------------------------
    # Output
    # ---------------------------
    output_root: str = "output"
    save_debug_plots: bool = True
    figure_dpi: int = 150
    show_grain_labels: bool = False
    material_filename: str = "material.yaml"

    def phase_name(self, phase_id: int) -> str:
        if phase_id == 0:
            return self.dark_phase_name
        if phase_id == 1:
            return self.light_phase_name
        raise ValueError(f"Unbekannte phase_id: {phase_id}. Erwartet sind 0 oder 1.")

    def phase_orientation_mode(self, phase_id: int) -> str:
        if phase_id == 0:
            return self.dark_phase_orientation_mode
        if phase_id == 1:
            return self.light_phase_orientation_mode
        raise ValueError(f"Unbekannte phase_id: {phase_id}. Erwartet sind 0 oder 1.")

    def phase_material(self, phase_id: int) -> Dict[str, Any]:
        material = self.dark_phase_material if phase_id == 0 else self.light_phase_material if phase_id == 1 else None
        if material is None:
            name = self.phase_name(phase_id)
            raise ValueError(
                f"Für Phase '{name}' (phase_id={phase_id}) ist keine DAMASK-Materialdefinition hinterlegt. "
                f"Setze {'dark_phase_material' if phase_id == 0 else 'light_phase_material'} in Config."
            )
        return material
