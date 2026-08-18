import numpy as np

from micro2damask.config import Config
from micro2damask.rve import select_rve


def test_select_centered_rve():

    phase = np.arange(100).reshape(10, 10)

    cfg = Config(
        rve_w=4,
        rve_h=4,
        rve_x=None,
        rve_y=None,
    )

    rve, rect = select_rve(phase, cfg)

    assert rve.shape == (4, 4)

    x, y, w, h = rect

    assert w == 4
    assert h == 4


def test_select_explicit_rve():

    phase = np.arange(100).reshape(10, 10)

    cfg = Config(
        rve_w=3,
        rve_h=2,
        rve_x=2,
        rve_y=4,
    )

    rve, rect = select_rve(phase, cfg)

    expected = phase[4:6, 2:5]

    np.testing.assert_array_equal(rve, expected)

    assert rect == (2, 4, 3, 2)