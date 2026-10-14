"""Tests for the pinhole camera model."""

import math

from smoke_localisation.camera import pixel_to_ray


def test_center_pixel_gives_camera_azimuth():
    az, el = pixel_to_ray(0.5, 0.5, azimuth_deg=180.0, hfov_deg=60.0, tilt_deg=0.0)
    assert az == 180.0
    assert abs(el) < 1e-10


def test_horizontal_offset():
    az, el = pixel_to_ray(1.0, 0.5, azimuth_deg=0.0, hfov_deg=60.0, tilt_deg=0.0)
    assert abs(az - 30.0) < 1e-10  # half of 60 deg FOV


def test_tilt_shifts_elevation():
    _, el_no_tilt = pixel_to_ray(0.5, 0.5, azimuth_deg=0.0, hfov_deg=60.0, tilt_deg=0.0)
    _, el_tilted = pixel_to_ray(0.5, 0.5, azimuth_deg=0.0, hfov_deg=60.0, tilt_deg=5.0)
    assert abs(el_no_tilt) < 1e-10
    assert abs(el_tilted - (-5.0)) < 1e-10


def test_top_pixel_looks_up():
    _, el = pixel_to_ray(0.5, 0.0, azimuth_deg=0.0, hfov_deg=60.0, tilt_deg=0.0)
    assert el > 0  # top of image = looking up


def test_azimuth_wraps():
    az, _ = pixel_to_ray(0.0, 0.5, azimuth_deg=10.0, hfov_deg=60.0, tilt_deg=0.0)
    assert az == 340.0  # 10 - 30 = -20 -> 340


def test_roll_tilts_elevation_across_the_image():
    _, el_center = pixel_to_ray(0.5, 0.5, azimuth_deg=0.0, hfov_deg=60.0, roll_deg=2.0)
    _, el_right = pixel_to_ray(1.0, 0.5, azimuth_deg=0.0, hfov_deg=60.0, roll_deg=2.0)
    assert abs(el_center) < 1e-10
    assert abs(el_right + math.sin(math.radians(2.0)) * 30.0) < 1e-10


def test_roll_also_shifts_azimuth_of_pixels_above_centre():
    az, _ = pixel_to_ray(0.5, 0.0, azimuth_deg=10.0, hfov_deg=60.0, roll_deg=2.0)
    _, y_top = pixel_to_ray(0.5, 0.0, azimuth_deg=10.0, hfov_deg=60.0)
    assert abs(az - (10.0 + y_top * math.sin(math.radians(2.0)))) < 1e-10
