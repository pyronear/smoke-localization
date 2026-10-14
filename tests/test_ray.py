"""Tests for ray-terrain intersection."""

import math

import numpy as np

from smoke_localisation.ray import EFFECTIVE_EARTH_RADIUS_M, ray_ground_intersection


def test_earth_curvature_pushes_the_hit_further_on_flat_ground():
    lats = np.linspace(45.5, 44.5, 101)
    lons = np.linspace(4.5, 5.5, 101)
    elev = np.zeros((101, 101))
    h, el = 100.0, -0.5
    hit = ray_ground_intersection(45.0, 5.0, h, 0.0, el, elev, lats, lons, max_distance_m=30_000)

    # Solve h - d*tan(a) + d^2 / (2R) = 0 for the horizontal distance d
    r_tan = EFFECTIVE_EARTH_RADIUS_M * math.tan(math.radians(-el))
    expected = r_tan - math.sqrt(r_tan**2 - 2 * EFFECTIVE_EARTH_RADIUS_M * h)  # ~12.7 km, vs 11.5 km on a flat Earth
    assert hit is not None
    assert abs(hit[3] * math.cos(math.radians(el)) - expected) < 5
