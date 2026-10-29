"""Tests for horizon detection on synthetic sky masks."""

import numpy as np

from smoke_localisation.sky import find_horizon_line


def test_horizon_ignores_detached_sky_patches():
    mask = np.zeros((5, 3))
    mask[:2, 0] = 1  # sky rows 0-1
    mask[[0, 3], 1] = 1  # row 3 is detached from the sky at the top
    # column 2: no sky
    assert find_horizon_line(mask).tolist() == [1, 0, -1]


def test_cloud_hole_does_not_become_the_horizon():
    mask = np.zeros((6, 5))
    mask[:4] = 1
    mask[1, 2] = 0
    assert find_horizon_line(mask).tolist() == [3] * 5
