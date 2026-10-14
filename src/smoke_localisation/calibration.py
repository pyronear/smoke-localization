"""Camera orientation from the skyline: match the detected sky boundary with the DSM skyline."""

from typing import NamedTuple

import numpy as np
import pyproj
from scipy.optimize import least_squares

from .camera import pixel_angles, vfov_deg
from .dem import dsm_interpolator
from .ray import curvature_drop_m
from .sky import find_horizon_line


class Calibration(NamedTuple):
    tilt_deg: float          # positive = looking down
    roll_deg: float          # positive = true horizon higher on the right of the image
    azimuth_deg: float       # corrected camera azimuth
    residual_deg: float      # median |detected - predicted| skyline elevation
    skyline_rows: np.ndarray  # predicted DSM skyline row (float) for each image column


def dem_skyline(
    cam_lat: float,
    cam_lon: float,
    cam_z: float,
    azimuths_deg: np.ndarray,
    interp,
    min_distance_m: float = 300,
    max_distance_m: float = 25_000,
    step_m: float = 30,
) -> np.ndarray:
    """Elevation angle (deg) of the highest DSM point seen from the camera, for each azimuth.

    Points closer than min_distance_m are skipped so that trees around the mast
    do not hide the real skyline. Points outside the DSM are ignored.
    """
    az, d = np.meshgrid(azimuths_deg, np.arange(min_distance_m, max_distance_m, step_m), indexing="ij")
    lon, lat, _ = pyproj.Geod(ellps="WGS84").fwd(
        np.full(az.shape, cam_lon), np.full(az.shape, cam_lat), az, d
    )
    z = interp(np.stack([lat, lon], axis=-1)) - curvature_drop_m(d)
    return np.degrees(np.nanmax(np.arctan2(z - cam_z, d), axis=1))


def calibrate(
    sky_mask: np.ndarray,
    cam_lat: float,
    cam_lon: float,
    cam_height_m: float,
    azimuth_deg: float,
    hfov_deg: float,
    elev: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    max_azimuth_offset_deg: float = 5.0,
    threshold: float = 0.5,
) -> Calibration:
    """Estimate tilt, roll and azimuth by fitting the detected skyline to the DSM skyline.

    For each azimuth offset on a 0.1 deg grid, the difference between detected and
    predicted skyline elevations is fitted as `tilt + sin(roll) * x` (x = horizontal
    angle) with a robust loss, so that haze or a tree only affects a few columns.
    The offset with the lowest robust cost wins, unless the given azimuth fits almost
    as well (a skyline without distinctive shape cannot pin the azimuth).
    """
    h, w = sky_mask.shape
    horizon = find_horizon_line(sky_mask, threshold)
    cols = np.flatnonzero(horizon >= 0)
    if cols.size < 0.1 * w:
        raise ValueError("Too few columns with detected sky to calibrate.")

    # The sky/ground boundary sits below the last sky pixel
    x, e_img = pixel_angles((cols + 0.5) / w, (horizon[cols] + 1) / h, hfov_deg, (w, h))

    interp = dsm_interpolator(elev, lats, lons)
    cam_z = float(interp((cam_lat, cam_lon))) + cam_height_m
    half = hfov_deg / 2 + max_azimuth_offset_deg + 0.1
    az_grid = np.arange(azimuth_deg - half, azimuth_deg + half, 0.05)
    sky_grid = dem_skyline(cam_lat, cam_lon, cam_z, az_grid, interp)

    fits = []
    for offset in np.round(np.arange(-max_azimuth_offset_deg, max_azimuth_offset_deg + 1e-9, 0.1), 1):
        diff = e_img - np.interp(azimuth_deg + offset + x, az_grid, sky_grid)
        fit = least_squares(
            lambda p: p[0] + p[1] * x - diff,
            x0=[np.median(diff), 0.0], loss="soft_l1", f_scale=0.1,
        )
        fits.append((fit.cost, offset, fit.x, fit.fun))

    best = min(fits, key=lambda f: f[0])
    at_given_azimuth = min(fits, key=lambda f: abs(f[1]))
    # ponytail: fixed 10 % margin; use a proper identifiability test (cost curvature vs noise) if it misfires
    chosen = at_given_azimuth if at_given_azimuth[0] <= best[0] * 1.1 else best
    _, offset, (tilt, roll_slope), residuals = chosen
    azimuth = azimuth_deg + offset

    # Predicted skyline row for every column, for display
    x_all, _ = pixel_angles((np.arange(w) + 0.5) / w, 0.5, hfov_deg, (w, h))
    e_pred = np.interp(azimuth + x_all, az_grid, sky_grid) + tilt + roll_slope * x_all
    skyline_rows = (0.5 - e_pred / vfov_deg(hfov_deg, (w, h))) * h

    return Calibration(
        tilt_deg=float(tilt),
        roll_deg=float(np.degrees(np.arcsin(np.clip(roll_slope, -1, 1)))),
        azimuth_deg=float(azimuth % 360),
        residual_deg=float(np.median(np.abs(residuals))),
        skyline_rows=skyline_rows,
    )
