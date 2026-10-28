"""Calibration on a synthetic DSM: recover the orientation used to draw the sky mask."""

import math

import numpy as np
import pytest
from scipy.optimize import least_squares

import smoke_localisation.calibration as calibration_module
from smoke_localisation.calibration import calibrate, dem_skyline
from smoke_localisation.camera import pixel_angles, vfov_deg
from smoke_localisation.dem import dsm_interpolator


def test_batched_fit_matches_soft_l1_reference_with_outliers(monkeypatch):
    w, h, hfov = 320, 180, 60
    x, _ = pixel_angles((np.arange(w) + 0.5) / w, 0.5, hfov, (w, h))
    rows = np.round((0.5 - (0.5 + 0.014 * x) / vfov_deg(hfov, (w, h))) * h)
    rows[-30:] += 8
    mask = (np.arange(h)[:, None] < rows).astype(float)
    monkeypatch.setattr(
        calibration_module,
        "dem_skyline",
        lambda a, b, c, az, *args, **kwargs: np.zeros_like(az),
    )
    dsm = (np.zeros((11, 11)), np.linspace(45.1, 44.9, 11), np.linspace(4.9, 5.1, 11))
    actual = calibrate(mask, 45, 5, 15, 0, hfov, *dsm, max_azimuth_offset_deg=0)
    _, observed = pixel_angles((np.arange(w) + 0.5) / w, rows / h, hfov, (w, h))
    expected = least_squares(
        lambda p: p[0] + p[1] * x - observed, [0.5, 0.014], loss="soft_l1", f_scale=0.1
    )
    assert abs(actual.tilt_deg - expected.x[0]) < 1e-5
    assert abs(math.sin(math.radians(actual.roll_deg)) - expected.x[1]) < 1e-5


@pytest.mark.parametrize("true_az", [3.0, 12.0])
def test_calibrate_recovers_tilt_roll_and_azimuth(true_az):
    lats = np.linspace(45.3, 44.7, 241)
    lons = np.linspace(4.6, 5.4, 321)
    lat_g, lon_g = np.meshgrid(lats, lons, indexing="ij")
    elev = np.zeros_like(lat_g)
    # A few hills north of the camera, at different bearings and distances
    for lat0, lon0, height in [
        (45.06, 4.93, 500),
        (45.10, 5.02, 800),
        (45.05, 5.08, 400),
        (45.12, 4.98, 650),
    ]:
        elev += height * np.exp(-((lat_g - lat0) ** 2 + (lon_g - lon0) ** 2) / 0.012**2)

    w, h, hfov, cam_height = 640, 360, 60.0, 15.0
    true_tilt, true_roll = 1.2, 0.8

    interp = dsm_interpolator(elev, lats, lons)
    cam_z = float(interp((45.0, 5.0))) + cam_height
    x, _ = pixel_angles((np.arange(w) + 0.5) / w, 0.5, hfov, (w, h))
    e_img = (
        dem_skyline(45.0, 5.0, cam_z, true_az + x, interp)
        + true_tilt
        + math.tan(math.radians(true_roll)) * x
    )
    boundary_rows = (0.5 - e_img / vfov_deg(hfov, (w, h))) * h
    sky_mask = (np.arange(h)[:, None] < np.round(boundary_rows)[None, :]).astype(
        np.float32
    )

    cal = calibrate(
        sky_mask,
        45.0,
        5.0,
        cam_height,
        azimuth_deg=0.0,
        hfov_deg=hfov,
        elev=elev,
        lats=lats,
        lons=lons,
    )

    assert abs(cal.azimuth_deg - true_az) <= 0.15
    assert abs(cal.tilt_deg - true_tilt) < 0.1
    assert abs(cal.roll_deg - true_roll) < 0.1
    assert cal.residual_deg < 0.1
    assert np.isfinite(cal.tilt_uncertainty_deg) and cal.tilt_uncertainty_deg > 0


def test_flat_skyline_keeps_the_given_azimuth():
    lats = np.linspace(45.3, 44.7, 61)
    lons = np.linspace(4.6, 5.4, 81)
    elev = np.zeros((61, 81))
    sky_mask = np.zeros((360, 640), dtype=np.float32)
    sky_mask[:170] = 1  # straight horizon: no shape to pin the azimuth

    cal = calibrate(
        sky_mask,
        45.0,
        5.0,
        15.0,
        azimuth_deg=12.0,
        hfov_deg=60.0,
        elev=elev,
        lats=lats,
        lons=lons,
    )

    assert cal.azimuth_deg == 12.0
    assert cal.azimuth_uncertainty_deg >= 19.9
    assert np.isfinite(cal.tilt_uncertainty_deg)


@pytest.mark.parametrize("sky", [0.0, 1.0])
def test_uniform_masks_are_rejected(sky):
    with pytest.raises(ValueError, match="interior skyline"):
        calibrate(
            np.full((360, 640), sky),
            45,
            5,
            15,
            0,
            60,
            np.zeros((61, 81)),
            np.linspace(45.3, 44.7, 61),
            np.linspace(4.6, 5.4, 81),
        )


def test_upsampling_does_not_shrink_uncertainty():
    mask = np.zeros((480, 640))
    mask[:200] = 1
    terrain = (
        np.zeros((61, 81)),
        np.linspace(45.3, 44.7, 61),
        np.linspace(4.6, 5.4, 81),
    )
    fit = calibrate(mask, 45, 5, 15, 0, 60, *terrain)
    upsampled = calibrate(
        np.repeat(np.repeat(mask, 2, axis=0), 2, axis=1), 45, 5, 15, 0, 60, *terrain
    )
    assert math.isclose(
        fit.tilt_uncertainty_deg, upsampled.tilt_uncertainty_deg, abs_tol=1e-6
    )


def test_bad_skyline_fit_is_rejected():
    boundary = 180 + 80 * np.sin(np.arange(640) / 30)
    mask = (np.arange(360)[:, None] < boundary).astype(float)
    with pytest.raises(ValueError, match="residual"):
        calibrate(
            mask,
            45,
            5,
            15,
            0,
            60,
            np.zeros((61, 81)),
            np.linspace(45.3, 44.7, 61),
            np.linspace(4.6, 5.4, 81),
        )
