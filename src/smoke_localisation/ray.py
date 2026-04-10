"""Ray-terrain intersection by marching on a DSM grid."""

import math

import numpy as np
import pyproj
from scipy.interpolate import RegularGridInterpolator


def ray_ground_intersection(
    cam_lat: float,
    cam_lon: float,
    cam_height_m: float,
    ray_az_deg: float,
    ray_el_deg: float,
    elev: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    max_distance_m: float = 30_000,
    step_m: float = 10,
    verbose: bool = True,
) -> tuple[float, float, float, float] | None:
    """March a ray from the camera and find where it hits the DSM.

    Returns (hit_lat, hit_lon, hit_elev, distance_m) or None if no hit.
    """
    # Interpolator expects ascending lat
    if lats[0] > lats[-1]:
        elev_flip = elev[::-1, :]
        lats_flip = lats[::-1]
    else:
        elev_flip = elev
        lats_flip = lats

    interp = RegularGridInterpolator(
        (lats_flip, lons), elev_flip,
        method="linear", bounds_error=False, fill_value=np.nan,
    )

    cam_ground_elev = float(interp((cam_lat, cam_lon)))
    cam_z = cam_ground_elev + cam_height_m
    if verbose:
        print(f"Camera ground elevation: {cam_ground_elev:.1f} m  ->  camera Z = {cam_z:.1f} m")

    geod = pyproj.Geod(ellps="WGS84")

    el_rad = math.radians(ray_el_deg)
    cos_el = math.cos(el_rad)
    sin_el = math.sin(el_rad)

    n_steps = int(max_distance_m / step_m)

    for i in range(1, n_steps + 1):
        d = i * step_m
        horiz_d = d * cos_el
        dz = d * sin_el

        ray_z = cam_z + dz
        target_lon, target_lat, _ = geod.fwd(cam_lon, cam_lat, ray_az_deg, horiz_d)

        terrain_z = interp((target_lat, target_lon))
        if terrain_z is None or np.isnan(terrain_z):
            continue

        terrain_z = float(terrain_z)

        if ray_z <= terrain_z:
            # Refine with bisection
            d_lo = (i - 1) * step_m
            d_hi = d
            for _ in range(20):
                d_mid = (d_lo + d_hi) / 2
                hz = d_mid * cos_el
                rz = cam_z + d_mid * sin_el
                tlon, tlat, _ = geod.fwd(cam_lon, cam_lat, ray_az_deg, hz)
                tz = float(interp((tlat, tlon)))
                if rz <= tz:
                    d_hi = d_mid
                else:
                    d_lo = d_mid

            d_final = (d_lo + d_hi) / 2
            hz = d_final * cos_el
            hit_lon, hit_lat, _ = geod.fwd(cam_lon, cam_lat, ray_az_deg, hz)
            hit_elev = float(interp((hit_lat, hit_lon)))
            return hit_lat, hit_lon, hit_elev, d_final

    return None
