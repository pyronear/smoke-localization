"""Pinhole camera model: convert normalised pixel coordinates to ray directions."""

import math


def pixel_to_ray(
    u: float,
    v: float,
    azimuth_deg: float,
    hfov_deg: float,
    aspect_ratio: tuple[int, int] = (16, 9),
    tilt_deg: float = 0.0,
) -> tuple[float, float]:
    """Convert normalised pixel (u, v) to a ray direction.

    Parameters
    ----------
    u, v : float in [0, 1]
        Normalised pixel. (0, 0) = top-left, (1, 1) = bottom-right.
    azimuth_deg : float
        Camera centre azimuth (degrees, clockwise from north).
    hfov_deg : float
        Horizontal field of view in degrees.
    aspect_ratio : (w, h)
        Sensor aspect ratio (default 16:9).
    tilt_deg : float
        Camera pitch -- positive = looking *down* from horizon.

    Returns
    -------
    ray_azimuth_deg, ray_elevation_deg : float
        Azimuth (deg, clockwise from N) and elevation (deg from horizontal,
        negative = downward) of the ray.
    """
    hfov = math.radians(hfov_deg)
    vfov = 2 * math.atan(math.tan(hfov / 2) * aspect_ratio[1] / aspect_ratio[0])

    delta_az = (u - 0.5) * hfov_deg
    delta_el = (0.5 - v) * math.degrees(vfov)

    ray_az = azimuth_deg + delta_az
    ray_el = delta_el - tilt_deg

    return ray_az % 360, ray_el
