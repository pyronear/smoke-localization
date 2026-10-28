"""Camera orientation from the skyline: match the detected sky boundary with the DSM skyline."""

from typing import NamedTuple

import numpy as np
import pyproj

from .camera import pixel_angles, vfov_deg
from .dem import dsm_interpolator
from .ray import curvature_drop_m
from .sky import SKY_INPUT_SIZE, find_horizon_line


class Calibration(NamedTuple):
    tilt_deg: float  # positive = looking down
    roll_deg: float  # positive = true horizon higher on the right of the image
    azimuth_deg: float  # corrected camera azimuth
    residual_deg: float  # median |detected - predicted| skyline elevation
    skyline_rows: np.ndarray  # predicted DSM skyline row (float) for each image column
    tilt_uncertainty_deg: float = float(
        "nan"
    )  # conditional diagnostic radius, not a confidence level
    azimuth_uncertainty_deg: float = float("nan")


def dem_skyline(
    cam_lat: float,
    cam_lon: float,
    cam_z: float,
    azimuths_deg: np.ndarray,
    interp,
    min_distance_m: float = 300,
    max_distance_m: float = 25_000,
    step_m: float = 30,
    refraction_coefficient: float = 0.13,
) -> np.ndarray:
    """Elevation angle (deg) of the highest DSM point seen from the camera, for each azimuth.

    Points closer than min_distance_m are skipped so that trees around the mast
    do not hide the real skyline. Points outside the DSM are ignored.
    """
    az, d = np.meshgrid(
        azimuths_deg, np.arange(min_distance_m, max_distance_m, step_m), indexing="ij"
    )
    lon, lat, _ = pyproj.Geod(ellps="WGS84").fwd(
        np.full(az.shape, cam_lon), np.full(az.shape, cam_lat), az, d
    )
    z = interp(np.stack([lat, lon], axis=-1)) - curvature_drop_m(
        d, refraction_coefficient
    )
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
    max_azimuth_offset_deg: float = 20.0,
    threshold: float = 0.5,
    max_residual_deg: float = 0.5,
    max_skyline_distance_m: float = 25_000,
    refraction_coefficient: float = 0.13,
) -> Calibration:
    """Estimate tilt, roll and azimuth by fitting the detected skyline to the DSM skyline.

    For each azimuth offset on a 0.1 deg grid, the difference between detected and
    predicted skyline elevations is fitted as `tilt + sin(roll) * x` (x = horizontal
    angle) with a robust loss, so that haze or a tree only affects a few columns.
    The offset with the lowest robust cost wins, unless the given azimuth fits almost
    as well (a skyline without distinctive shape cannot pin the azimuth).

    The tilt uncertainty is a conditional diagnostic radius: spatial block
    disagreement, near-optimal azimuth fits and half a native mask pixel. It has
    no confidence level and excludes FOV, mast height, DSM and absolute bias.
    """
    if (
        sky_mask.ndim != 2
        or min(sky_mask.shape) < 4
        or not np.isfinite(sky_mask).all()
        or np.any((sky_mask < 0) | (sky_mask > 1))
    ):
        raise ValueError("Supply a finite, two-dimensional sky-probability mask.")
    if (
        not all(
            np.isfinite(v)
            for v in (
                cam_lat,
                cam_lon,
                cam_height_m,
                azimuth_deg,
                hfov_deg,
                max_azimuth_offset_deg,
                threshold,
                max_residual_deg,
                max_skyline_distance_m,
                refraction_coefficient,
            )
        )
        or not -89 < cam_lat < 89
        or not -180 <= cam_lon <= 180
        or not 0 < cam_height_m < 1000
        or not 1 < hfov_deg < 179
        or not 0 <= max_azimuth_offset_deg <= 90
        or not 0 < threshold < 1
        or max_residual_deg <= 0
        or not 300 < max_skyline_distance_m <= 100_000
        or not 0 <= refraction_coefficient < 1
    ):
        raise ValueError("Invalid camera, search, threshold or residual limit.")
    h, w = sky_mask.shape
    horizon = find_horizon_line(sky_mask, threshold)
    cols = np.flatnonzero((horizon >= 0) & (horizon < h - 1))
    if cols.size < max(16, 0.1 * w) or np.ptp(cols) < 0.5 * w:
        raise ValueError("Too few interior skyline columns across the image.")
    # Upsampling the 384 px segmentation mask does not create new observations.
    cols = cols[
        np.linspace(0, cols.size - 1, min(cols.size, SKY_INPUT_SIZE), dtype=int)
    ]

    # The sky/ground boundary sits below the last sky pixel
    x, e_img = pixel_angles((cols + 0.5) / w, (horizon[cols] + 1) / h, hfov_deg, (w, h))

    interp = dsm_interpolator(elev, lats, lons)
    cam_z = float(interp((cam_lat, cam_lon))) + cam_height_m
    half = hfov_deg / 2 + max_azimuth_offset_deg + 0.1
    az_grid = np.arange(azimuth_deg - half, azimuth_deg + half, 0.05)
    sky_grid = dem_skyline(
        cam_lat,
        cam_lon,
        cam_z,
        az_grid,
        interp,
        max_distance_m=max_skyline_distance_m,
        refraction_coefficient=refraction_coefficient,
    )

    if not np.isfinite(sky_grid).all():
        raise ValueError("DSM does not cover the skyline search.")
    offsets = np.arange(-max_azimuth_offset_deg, max_azimuth_offset_deg + 0.05, 0.1)
    offsets = np.unique(np.r_[offsets, 0.0])
    diff = e_img - np.interp(azimuth_deg + offsets[:, None] + x, az_grid, sky_grid)
    tilts = np.median(diff, axis=1)
    slopes = np.zeros_like(tilts)
    # Batched iteratively reweighted least squares for the existing soft-L1 loss.
    for _ in range(64):
        residuals = tilts[:, None] + slopes[:, None] * x - diff
        weights = 1 / np.sqrt(1 + (residuals / 0.1) ** 2)
        sw, sx, sxx = (
            weights.sum(axis=1),
            (weights * x).sum(axis=1),
            (weights * x * x).sum(axis=1),
        )
        sy, sxy = (weights * diff).sum(axis=1), (weights * x * diff).sum(axis=1)
        new_slopes = (sw * sxy - sx * sy) / (sw * sxx - sx * sx)
        new_tilts = (sy - new_slopes * sx) / sw
        change = max(
            np.max(np.abs(new_tilts - tilts)), np.max(np.abs(new_slopes - slopes))
        )
        tilts, slopes = new_tilts, new_slopes
        if change < 1e-7:
            break
    residual_grid = tilts[:, None] + slopes[:, None] * x - diff
    costs = 0.01 * (np.sqrt(1 + (residual_grid / 0.1) ** 2) - 1).sum(axis=1)
    best, given = int(np.argmin(costs)), int(np.argmin(np.abs(offsets)))
    chosen = given if costs[given] <= costs[best] * 1.1 else best
    offset, tilt, roll_slope = offsets[chosen], tilts[chosen], slopes[chosen]
    residuals = residual_grid[chosen]
    residual = float(np.median(np.abs(residuals)))
    if residual > max_residual_deg or abs(roll_slope) > np.sin(np.radians(5)):
        raise ValueError("Skyline fit residual or image roll exceeds the model limits.")
    if max_azimuth_offset_deg and abs(offset) >= max_azimuth_offset_deg - 0.05:
        raise ValueError("Azimuth fit reaches the search boundary; widen the search.")
    blocks = np.array_split(residuals, min(12, len(residuals) // 8))
    block_offsets = np.array([np.median(block) for block in blocks])
    spatial_spread = 1.4826 * np.median(
        np.abs(block_offsets - np.median(block_offsets))
    )
    native_half_pixel = vfov_deg(hfov_deg, (w, h)) / min(h, SKY_INPUT_SIZE) / 2
    noise = max(spatial_spread, native_half_pixel)
    near = costs <= costs[best] + noise**2 * len(cols) / len(blocks)
    # shortcut: conditional radius only; DSM, height, FOV and absolute bias need references.
    tilt_uncertainty = max(noise, np.max(np.abs(tilts[near] - tilt)))
    azimuth_uncertainty = np.max(np.abs(offsets[near] - offset))
    azimuth = azimuth_deg + offset

    # Predicted skyline row for every column, for display
    x_all, _ = pixel_angles((np.arange(w) + 0.5) / w, 0.5, hfov_deg, (w, h))
    e_pred = np.interp(azimuth + x_all, az_grid, sky_grid) + tilt + roll_slope * x_all
    skyline_rows = (0.5 - e_pred / vfov_deg(hfov_deg, (w, h))) * h

    return Calibration(
        tilt_deg=float(tilt),
        roll_deg=float(np.degrees(np.arcsin(np.clip(roll_slope, -1, 1)))),
        azimuth_deg=float(azimuth % 360),
        residual_deg=residual,
        skyline_rows=skyline_rows,
        tilt_uncertainty_deg=float(tilt_uncertainty),
        azimuth_uncertainty_deg=float(azimuth_uncertainty),
    )
