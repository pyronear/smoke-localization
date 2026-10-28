"""Ray-terrain intersection by marching on a DSM grid."""

import math

import numpy as np
import pyproj

from .dem import dsm_interpolator

# Earth radius inflated for standard atmospheric refraction (k = 0.13)
EFFECTIVE_EARTH_RADIUS_M = 6_371_000 / (1 - 0.13)


def curvature_drop_m(distance_m, refraction_coefficient=0.13):
    """How far the terrain falls below the camera's local horizontal plane at a given distance."""
    return np.square(distance_m) * (1 - refraction_coefficient) / (2 * 6_371_000)


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
) -> tuple[float, float, float, float] | None:
    """March a ray from the camera and find where it hits the DSM.

    Returns (hit_lat, hit_lon, hit_elev, distance_m) or None if no hit.
    """
    interp = dsm_interpolator(elev, lats, lons)
    cam_z = float(interp((cam_lat, cam_lon))) + cam_height_m
    geod = pyproj.Geod(ellps="WGS84")
    el_rad = math.radians(ray_el_deg)
    cos_el, sin_el = math.cos(el_rad), math.sin(el_rad)

    def probe(d: float) -> tuple[float, float, float, float]:
        """(ray height above terrain, lat, lon, terrain elevation) at distance d along the ray."""
        lon, lat, _ = geod.fwd(cam_lon, cam_lat, ray_az_deg, d * cos_el)
        terrain_z = float(interp((lat, lon)))
        drop = curvature_drop_m(d * cos_el)
        return cam_z + d * sin_el - (terrain_z - drop), lat, lon, terrain_z

    for i in range(1, int(max_distance_m / step_m) + 1):
        if not probe(i * step_m)[0] <= 0:  # NaN (outside DSM) is not a hit
            continue
        # Refine with bisection
        d_lo, d_hi = (i - 1) * step_m, i * step_m
        for _ in range(20):
            d_mid = (d_lo + d_hi) / 2
            if probe(d_mid)[0] <= 0:
                d_hi = d_mid
            else:
                d_lo = d_mid
        d = (d_lo + d_hi) / 2
        _, lat, lon, terrain_z = probe(d)
        return lat, lon, terrain_z, d

    return None
