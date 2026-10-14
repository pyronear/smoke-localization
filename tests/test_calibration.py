"""Calibration on a synthetic DSM: recover the orientation used to draw the sky mask."""

import math

import numpy as np

from smoke_localisation.calibration import calibrate, dem_skyline
from smoke_localisation.camera import pixel_angles, vfov_deg
from smoke_localisation.dem import dsm_interpolator


def test_calibrate_recovers_tilt_roll_and_azimuth():
    lats = np.linspace(45.3, 44.7, 241)
    lons = np.linspace(4.6, 5.4, 321)
    lat_g, lon_g = np.meshgrid(lats, lons, indexing="ij")
    elev = np.zeros_like(lat_g)
    # A few hills north of the camera, at different bearings and distances
    for lat0, lon0, height in [(45.06, 4.93, 500), (45.10, 5.02, 800), (45.05, 5.08, 400), (45.12, 4.98, 650)]:
        elev += height * np.exp(-((lat_g - lat0) ** 2 + (lon_g - lon0) ** 2) / 0.012**2)

    w, h, hfov, cam_height = 640, 360, 60.0, 15.0
    true_az, true_tilt, true_roll = 3.0, 1.2, 0.8

    interp = dsm_interpolator(elev, lats, lons)
    cam_z = float(interp((45.0, 5.0))) + cam_height
    x, _ = pixel_angles((np.arange(w) + 0.5) / w, 0.5, hfov, (w, h))
    e_img = dem_skyline(45.0, 5.0, cam_z, true_az + x, interp) + true_tilt + math.tan(math.radians(true_roll)) * x
    boundary_rows = (0.5 - e_img / vfov_deg(hfov, (w, h))) * h
    sky_mask = (np.arange(h)[:, None] < np.round(boundary_rows)[None, :]).astype(np.float32)

    cal = calibrate(sky_mask, 45.0, 5.0, cam_height, azimuth_deg=0.0, hfov_deg=hfov,
                    elev=elev, lats=lats, lons=lons)

    assert abs(cal.azimuth_deg - true_az) <= 0.15
    assert abs(cal.tilt_deg - true_tilt) < 0.1
    assert abs(cal.roll_deg - true_roll) < 0.1
    assert cal.residual_deg < 0.1


def test_flat_skyline_keeps_the_given_azimuth():
    lats = np.linspace(45.3, 44.7, 61)
    lons = np.linspace(4.6, 5.4, 81)
    elev = np.zeros((61, 81))
    sky_mask = np.zeros((360, 640), dtype=np.float32)
    sky_mask[:170] = 1  # straight horizon: no shape to pin the azimuth

    cal = calibrate(sky_mask, 45.0, 5.0, 15.0, azimuth_deg=12.0, hfov_deg=60.0,
                    elev=elev, lats=lats, lons=lons)

    assert cal.azimuth_deg == 12.0
