"""Pinhole camera model: convert normalised pixel coordinates to ray directions."""

import math


def vfov_deg(hfov_deg: float, aspect_ratio: tuple[int, int] = (16, 9)) -> float:
    """Vertical field of view from the horizontal one and the (w, h) aspect ratio."""
    hfov = math.radians(hfov_deg)
    return math.degrees(2 * math.atan(math.tan(hfov / 2) * aspect_ratio[1] / aspect_ratio[0]))


def pixel_angles(u, v, hfov_deg: float, aspect_ratio: tuple[int, int] = (16, 9)):
    """Horizontal and vertical angle (deg) of normalised pixel (u, v) from the optical axis.

    Works on scalars or numpy arrays. Positive = right / up.
    """
    return (u - 0.5) * hfov_deg, (0.5 - v) * vfov_deg(hfov_deg, aspect_ratio)


def pixel_to_ray(
    u: float,
    v: float,
    azimuth_deg: float,
    hfov_deg: float,
    aspect_ratio: tuple[int, int] = (16, 9),
    tilt_deg: float = 0.0,
    roll_deg: float = 0.0,
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
    roll_deg : float
        Camera roll -- positive = the true horizon appears higher on the right
        side of the image. Small-angle model.

    Returns
    -------
    ray_azimuth_deg, ray_elevation_deg : float
        Azimuth (deg, clockwise from N) and elevation (deg from horizontal,
        negative = downward) of the ray.
    """
    x, y = pixel_angles(u, v, hfov_deg, aspect_ratio)
    ray_el = y - tilt_deg - math.tan(math.radians(roll_deg)) * x
    return (azimuth_deg + x) % 360, ray_el
