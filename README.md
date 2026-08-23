# micro2damask

`micro2damask` is a Python workflow for converting a two-phase microstructure image into an image-based representative volume element (RVE) for DAMASK.

The current implementation is designed for grayscale micrographs containing one **dark** and one **light** phase. The phase names are fully configurable: the code does not assume Al/Si or any other fixed material system.

> **Research software:** Material models, segmentation settings, spatial resolution, grain reconstruction, and orientation assumptions must be validated for the specific material and experiment before using generated models for quantitative simulations.

## Features

- grayscale image import and preprocessing
- Otsu, adaptive, or manual threshold segmentation
- generic two-phase convention
  - `phase_id = 0` → dark phase
  - `phase_id = 1` → light phase
- freely configurable phase names and DAMASK material definitions
- optional morphological cleanup of either phase
- automatic or fixed RVE selection
- configurable x/y downsampling
- grain identification within each phase
- grain-to-phase mapping
- per-grain orientation generation
- 2D-to-3D extrusion of the grain map
- DAMASK geometry export
- DAMASK `material.yaml` generation
- phase, grain, RVE, and periodicity statistics
- validation of the generated grain/material model
- debug plots and VTI/NumPy output for inspection
- unit and integration tests with `pytest`

## Phase convention

The image representation is intentionally independent of the actual material names:

```text
phase_id 0 = dark pixels
phase_id 1 = light pixels
```

For example:

```yaml
dark:
  name: Si_eutectic
  material: ...

light:
  name: Al_matrix
  material: ...
```

or:

```yaml
dark:
  name: Martensite
  material: ...

light:
  name: Ferrite
  material: ...
```

The phase names are propagated into the generated statistics, orientation table, and DAMASK material configuration.

## Workflow

```text
Microstructure image
        |
        v
Image preprocessing
        |
        v
Two-phase segmentation
(dark = 0, light = 1)
        |
        v
RVE selection
        |
        v
Downsampling
        |
        v
Grain identification
        |
        +--------------------+
        |                    |
        v                    v
Grain orientations      Grain-to-phase map
        |                    |
        +---------+----------+
                  |
                  v
          3D voxel extrusion
                  |
          +-------+-------+
          |               |
          v               v
   DAMASK geometry    material.yaml
          |               |
          +-------+-------+
                  |
                  v
              Validation
```

## Repository structure

```text
micro2damask/
├── pyproject.toml
├── README.md
├── run_micro2damask.py
├── phase_config_example.yaml
├── src/
│   └── micro2damask/
│       ├── config.py
│       ├── preprocessing.py
│       ├── segmentation.py
│       ├── rve.py
│       ├── grains.py
│       ├── orientations.py
│       ├── geometry.py
│       ├── material.py
│       ├── statistics.py
│       ├── validation.py
│       ├── visualization.py
│       ├── output.py
│       └── pipeline.py
└── tests/
```

## Requirements

- Python >= 3.10
- NumPy
- pandas
- OpenCV
- matplotlib
- SciPy
- scikit-image
- PyYAML

Optional:

- `damask` for geometry export through the official `damask.GeomGrid` API
- `vtk` for DAMASK-compatible inline-binary VTI fallbacks and debug VTI files

## Installation

Clone the repository and install it in editable mode:

```bash
git clone <repository-url>
cd micro2damask
python -m pip install -e .
```

For the optional VTK exporters:

```bash
python -m pip install -e ".[vtk]"
```

If the DAMASK Python package is available in your environment:

```bash
python -m pip install -e ".[damask]"
```

The exact DAMASK installation procedure may depend on the DAMASK version and computing environment used for the simulation.

## Quick start in PyCharm

For development, `run_micro2damask.py` can be used as a direct entry point.

Place it in the repository root, next to `src/`, edit the settings at the top of the script, and press **Run** in PyCharm.

Typical settings are:

```python
IMAGE_PATH = Path("/path/to/microstructure.tif")
PHASE_CONFIG_PATH = PROJECT_DIR / "phase_config_example.yaml"

UM_PER_PIXEL = 0.35

RVE_WIDTH = 512
RVE_HEIGHT = 512
RVE_X = None
RVE_Y = None

DOWNSAMPLE_FACTOR = 4
NZ_LAYERS = 10

MORPHOLOGY_TARGET_PHASE = "light"

OUTPUT_DIR = PROJECT_DIR / "output"
SAVE_DEBUG_PLOTS = True
```

With

```python
RVE_WIDTH = 512
RVE_HEIGHT = 512
DOWNSAMPLE_FACTOR = 4
```

the resulting x/y voxel grid has approximately

```text
128 x 128 voxels
```

before extrusion in the z direction.

The effective x/y voxel size is

```text
original pixel size x downsampling factor
```

For example:

```text
0.35 µm/pixel x 4 = 1.4 µm/voxel
```

## Phase and material configuration

The names and DAMASK definitions of the dark and light phases are stored in a YAML file.

Example:

```yaml
dark:
  name: Matrix
  material:
    lattice: cF
    mechanical:
      output: [F, P]
      elastic:
        type: Hooke
        C_11: 200000000000.0
        C_12: 120000000000.0
        C_44: 80000000000.0
      plastic:
        type: none

light:
  name: SecondPhase
  material:
    lattice: cF
    mechanical:
      output: [F, P]
      elastic:
        type: Hooke
        C_11: 150000000000.0
        C_12: 90000000000.0
        C_44: 60000000000.0
      plastic:
        type: none
```

The numerical values in `phase_config_example.yaml` are **example/test values only**. Replace them with physically validated DAMASK material definitions for the material system being studied.

The program deliberately raises an error if a phase has no DAMASK material definition. Arbitrary phase names must not silently inherit material parameters from another material system.

## Important configuration options

The central configuration class is `micro2damask.config.Config`.

### Preprocessing

```python
denoise_method = "median"       # median, gaussian, nlmeans
median_ksize = 3
gaussian_sigma = 1.0
clahe_clip_limit = 2.0
clahe_tile_grid_size = 8
do_background_correction = False
```

### Segmentation

```python
threshold_method = "otsu"       # otsu, adaptive, manual
manual_threshold = None
adaptive_block_size = 51
adaptive_C = 2.0
invert_binary_after_threshold = False
```

Normally the segmentation convention is:

```text
dark pixels  -> 0
light pixels -> 1
```

Use `invert_binary_after_threshold=True` only when this assignment should deliberately be reversed.

### Morphology

```python
morphology_target_phase = "light"   # dark or light
remove_small_objects_min_size = 0
remove_small_holes_area_threshold = 0
do_opening = False
do_closing = False
morph_disk_radius = 1
```

### RVE and voxelization

```python
rve_w = 512
rve_h = 512
rve_x = None
rve_y = None

downsample_factor = 4
nz_layers = 10
```

### Grain reconstruction

```python
connectivity = 4
min_grain_size = 5
small_grain_mode = "keep"
```

### Orientations

```python
dark_phase_orientation_mode = "random"
light_phase_orientation_mode = "random"
random_seed = 420
```

The current `random` mode generates random Bunge Euler angles per grain. These orientations are synthetic and are not a substitute for experimentally measured texture data.

## Output

Each run creates a timestamped output directory below `output_root`, for example:

```text
output/
└── run_YYYYMMDD_HHMMSS_<id>/
    ├── config/
    ├── geometry/
    ├── images/
    ├── material/
    └── statistics/
```

Typical files include:

```text
geometry/
├── damask_geom.vti
├── debug_material_and_phase.vti
├── grain_map_2d.npy
├── grain_map_3d_nz_ny_nx.npy
└── phase_map_3d_nz_ny_nx.npy

material/
├── material.yaml
└── grain_orientations.csv

statistics/
├── statistics_summary.csv
├── grain_statistics.csv
├── rve_statistics.csv
├── dark_phase_region_area_distribution.csv
└── light_phase_region_area_distribution.csv
```

The exact set of files depends on the available export backend and the selected options.

## DAMASK geometry vs. debug VTI

This distinction is important.

### Simulation geometry

Use the VTI path returned by the pipeline as:

```python
result["damask"]["vti_path"]
```

If the DAMASK Python API is available, this will normally be:

```text
geometry/damask_geom.vti
```

This file contains the material/grain IDs required by the DAMASK grid geometry.

If the DAMASK API is unavailable, the code attempts to create a material-only VTI fallback.

### Debug VTI

The file

```text
geometry/debug_material_and_phase.vti
```

contains both:

```text
material
phase
```

and is intended for inspection in tools such as ParaView.

**Do not assume that the debug VTI is the file that should be passed to a DAMASK simulation.** Use the geometry path reported by `result["damask"]["vti_path"]` or printed by `run_micro2damask.py`.

## `material` and `phase` fields

The two fields have different meanings:

- `phase` describes the segmented image phase (`0 = dark`, `1 = light`).
- `material` identifies the reconstructed grain/material entry used by the DAMASK model.

Multiple grains can therefore belong to the same phase while having different `material` IDs.

The mapping is stored through the grain-to-phase information and represented in the generated `material.yaml` and `grain_orientations.csv`.

## MTEX ODF orientations

A phase can sample one Bunge-Euler orientation per grain from a generic MTEX
ASCII ODF export (`phi1 Phi phi2 value`). The ODF values are interpreted as
probability densities (m.r.d.); sampling includes the Euler-space `sin(Phi)`
Jacobian.

```python
cfg.dark_phase_orientation_mode = "mtex_odf"
cfg.dark_phase_odf_path = "odf_ferrite.txt"
```

The other phase can independently use another ODF or another orientation mode.
The random seed controls reproducible ODF sampling.

### IPF orientation maps

By default, the pipeline also writes inverse-pole-figure (IPF) false-color maps
from the realized grain orientations. The current implementation is restricted
to cubic phases (`cP`, `cI`, `cF`), matching MTEX ODFs with `m-3m` symmetry.

Generated files:

```text
images/ipf_RD_x.png
images/ipf_TD_y.png
images/ipf_ND_z.png
images/ipf_orientation_maps.png
```

The sample-frame directions are interpreted as RD/X = `[1,0,0]`, TD/Y =
`[0,1,0]`, and ND/Z = `[0,0,1]`. The cubic IPF key follows the DAMASK
standard stereographic triangle: `[001]` red, `[101]` green, `[111]` blue.
Set `save_ipf_maps = False` to disable these files.


## Validation

Before finishing a run, `validate_damask_model()` checks the consistency of the reconstructed model, including:

- negative material/grain IDs
- voxel IDs without a corresponding definition
- grains without a phase assignment
- grains without an orientation
- consistency between the 2D grain-map geometry and the 3D voxel grid dimensions

The validation result is returned as:

```python
result["validation"]
```

with the structure:

```python
{
    "ok": True,
    "errors": [],
    "warnings": [],
}
```

A successful software validation means that the generated data are internally consistent. It does **not** by itself validate the physical material model or the representativeness of the RVE.

## Testing

Install the package in editable mode and run:

```bash
pytest -q
```

The test suite contains unit tests for individual processing steps as well as an end-to-end integration test using freely configurable phase names.

During development without an editable installation, the src-layout can also be exposed explicitly:

```bash
PYTHONPATH=src pytest -q
```

## Troubleshooting

### `ValueError: ... keine DAMASK-Materialdefinition hinterlegt`

Both phases require a DAMASK material definition. Check that the phase YAML contains both:

```yaml
dark:
  name: ...
  material: ...

light:
  name: ...
  material: ...
```

and that `run_micro2damask.py` actually loads this YAML into `Config`.

### DAMASK: `error 100 - could not open file`

This normally means DAMASK cannot access the geometry path itself. Check that:

1. the reported VTI file exists,
2. the path is correct relative to the directory from which DAMASK is started, and
3. preferably an absolute geometry path is used when diagnosing the problem.

This error occurs before DAMASK can validate the internal VTI contents.

### VTI fallback requires `vtk`

The DAMASK fallback intentionally uses VTK's inline-binary writer. `pyevtk` is not used for solver input because it writes VTK XML `AppendedData`, which is not supported by the DAMASK grid solver. Install the optional VTK dependency if required:

```bash
python -m pip install -e ".[vtk]"
```

### TIFF `Unknown field with tag ...`

Some microscopy TIFF files contain vendor-specific metadata tags that OpenCV does not recognize. These warnings do not necessarily mean that image loading failed.

### scikit-image deprecation warnings

Depending on the installed scikit-image version, warnings may be emitted for morphology or region-property APIs. These warnings should be addressed when updating the code, but they are not necessarily fatal for the current run.

## Current limitations

- The segmentation model currently supports exactly **two image phases**.
- Phase identification is intensity-based: dark and light image regions must be separable by the selected thresholding approach.
- The generated 3D RVE is currently based on extrusion of a 2D grain map; it is not a reconstructed true 3D microstructure.
- The periodicity check is diagnostic; it does not automatically make the RVE periodic.
- DAMASK material parameters are supplied by the user and must be validated independently.
- Segmentation and grain reconstruction parameters strongly affect the generated model and should be verified visually and quantitatively.

## Recommended workflow for a new material system

1. Verify the input image scale in µm/pixel.
2. Decide which physical material corresponds to the dark and light image phase.
3. Define both phase names and validated DAMASK material models.
4. Tune preprocessing and segmentation on representative images.
5. Inspect the cleaned phase map and `debug_material_and_phase.vti`.
6. Verify RVE size, phase fractions, grain statistics, and boundary mismatch metrics.
7. Check the generated `grain_orientations.csv` and `material.yaml`.
8. Confirm that the pipeline validation reports `ok = True`.
9. Use only the reported DAMASK simulation geometry for the solver.
10. Run a small DAMASK test case before scaling to production simulations.

## Citation

Not published yet - ownership: Lorenz Maier, Technical University of Munich

## License

MIT License