"""Tests for ray-terrain intersection."""

import math

import numpy as np

from smoke_localisation.ray import ray_ground_intersection


def test_earth_curvature_pushes_the_hit_further_on_flat_ground():
    lats = np.linspace(45.5, 44.5, 101)
    lons = np.linspace(4.5, 5.5, 101)
    elev = np.zeros((101, 101))
    h, el = 100.0, -0.5
    hit = ray_ground_intersection(45.0, 5.0, h, 0.0, el, elev, lats, lons, max_distance_m=30_000)

    # Solve h - d*tan(a) + d^2 / (2R) = 0 for the horizontal distance d
    effective_radius = 6_371_000 / (1 - 0.13)
    r_tan = effective_radius * math.tan(math.radians(-el))
    expected = r_tan - math.sqrt(r_tan**2 - 2 * effective_radius * h)
    assert hit is not None
    assert abs(hit[3] * math.cos(math.radians(el)) - expected) < 5


def test_refraction_coefficient_applies_to_ray_intersection():
    lats = np.linspace(45.5, 44.5, 101)
    lons = np.linspace(4.5, 5.5, 101)
    elev = np.zeros((101, 101))
    args = (45.0, 5.0, 100.0, 0.0, -0.5, elev, lats, lons)

    no_refraction = ray_ground_intersection(*args, refraction_coefficient=0.0)
    refraction_0_2 = ray_ground_intersection(*args, refraction_coefficient=0.2)

    assert no_refraction is not None and refraction_0_2 is not None
    assert no_refraction[3] > refraction_0_2[3]
