from micro2damask.config import Config


def test_default_config():
    cfg = Config()

    assert cfg.connectivity == 4
    assert cfg.downsample_factor >= 1
    assert cfg.nz_layers >= 1


def test_config_accepts_custom_values():
    cfg = Config(
        connectivity=8,
        downsample_factor=4,
        nz_layers=3,
    )

    assert cfg.connectivity == 8
    assert cfg.downsample_factor == 4
    assert cfg.nz_layers == 3