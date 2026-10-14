"""Tests for horizon detection on synthetic sky masks."""

import numpy as np

from smoke_localisation.sky import find_horizon_line


def test_horizon_is_last_sky_row_per_column():
    mask = np.zeros((5, 3))
    mask[:2, 0] = 1       # sky rows 0-1
    mask[[0, 3], 1] = 1   # sky rows 0 and 3 -> last is 3
    # column 2: no sky
    assert find_horizon_line(mask).tolist() == [1, 3, -1]
