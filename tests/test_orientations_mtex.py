from pathlib import Path

import numpy as np
import pytest

from micro2damask.config import Config
from micro2damask.orientations import (
    generate_orientations_per_grain,
    load_mtex_odf,
    mtex_odf_sampling_probabilities,
    sample_mtex_odf,
)


def _write_odf(path: Path, rows: str) -> Path:
    path.write_text(
        '% MTEX ODF\n'
        '% crystal symmetry: "m-3m"\n'
        '% specimen symmetry: "1"\n'
        '% phi1    Phi     phi2    value\n'
        + rows,
        encoding='utf-8',
    )
    return path


def test_load_mtex_odf_reads_metadata_and_grid(tmp_path):
    path = _write_odf(
        tmp_path / 'odf.txt',
        '0 0 0 0.5\n0 90 0 2.0\n',
    )
    odf = load_mtex_odf(path)

    assert odf.crystal_symmetry == 'm-3m'
    assert odf.specimen_symmetry == '1'
    assert odf.euler_deg.shape == (2, 3)
    np.testing.assert_allclose(odf.values, [0.5, 2.0])


def test_odf_probability_includes_bunge_sin_Phi_jacobian(tmp_path):
    path = _write_odf(
        tmp_path / 'odf.txt',
        '0 0 0 100.0\n0 90 0 1.0\n',
    )
    p = mtex_odf_sampling_probabilities(load_mtex_odf(path))

    # A grid point exactly at Phi=0 has zero Euler-space volume for a density ODF.
    np.testing.assert_allclose(p, [0.0, 1.0], atol=1e-15)


def test_sample_mtex_odf_is_reproducible(tmp_path):
    path = _write_odf(
        tmp_path / 'odf.txt',
        '0 45 0 1.0\n90 45 0 3.0\n',
    )
    odf = load_mtex_odf(path)

    a = sample_mtex_odf(odf, 20, np.random.default_rng(42))
    b = sample_mtex_odf(odf, 20, np.random.default_rng(42))
    assert a.equals(b)


def test_generate_orientations_maps_mtex_odf_only_to_configured_phase(tmp_path):
    path = _write_odf(
        tmp_path / 'odf.txt',
        '15 90 25 1.0\n',
    )
    cfg = Config(
        dark_phase_orientation_mode='mtex_odf',
        dark_phase_odf_path=str(path),
        light_phase_orientation_mode='random',
        random_seed=7,
    )
    grain_to_phase = {0: 0, 1: 1, 2: 0}

    result = generate_orientations_per_grain(3, grain_to_phase, cfg)
    dark = result[result['phase_id'] == 0]
    light = result[result['phase_id'] == 1]

    np.testing.assert_allclose(dark['phi1_deg'], 15.0)
    np.testing.assert_allclose(dark['Phi_deg'], 90.0)
    np.testing.assert_allclose(dark['phi2_deg'], 25.0)
    assert set(dark['orientation_mode']) == {'mtex_odf'}
    assert set(light['orientation_mode']) == {'random'}


def test_mtex_odf_mode_requires_path():
    cfg = Config(dark_phase_orientation_mode='mtex_odf')
    with pytest.raises(ValueError, match='ODF-Pfad'):
        generate_orientations_per_grain(1, {0: 0}, cfg)
