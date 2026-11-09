"""Sky segmentation (ncnn) and horizon-based tilt estimation."""

import functools
from pathlib import Path

import cv2
import ncnn
import numpy as np

from .camera import vfov_deg

_MODEL_DIR = Path(__file__).resolve().parent / "models" / "skyseg"
_PARAM_PATH = _MODEL_DIR / "skysegsmall_sim-opt-fp16.param"
_BIN_PATH = _MODEL_DIR / "skysegsmall_sim-opt-fp16.bin"

SKY_INPUT_SIZE = 384


@functools.cache
def _load_net() -> ncnn.Net:
    net = ncnn.Net()
    net.load_param(str(_PARAM_PATH))
    net.load_model(str(_BIN_PATH))
    return net


def detect_sky(image_bgr: np.ndarray) -> np.ndarray:
    """Run the sky segmentation model on a BGR image.

    Returns a float32 mask (H, W) in [0, 1] at the original image resolution,
    where 1 = sky and 0 = ground.
    """
    h, w = image_bgr.shape[:2]

    mat_in = ncnn.Mat.from_pixels_resize(
        image_bgr,
        ncnn.Mat.PixelType.PIXEL_BGR2RGB,
        w,
        h,
        SKY_INPUT_SIZE,
        SKY_INPUT_SIZE,
    )
    mat_in.substract_mean_normalize([0.0, 0.0, 0.0], [1 / 255.0, 1 / 255.0, 1 / 255.0])

    net = _load_net()
    ex = net.create_extractor()
    ex.input("input.1", mat_in)
    _, mat_out = ex.extract("1959")

    mask_small = np.array(mat_out).squeeze()
    mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_LINEAR)
    return mask


def find_horizon_line(sky_mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Find the horizon row for each column of the sky mask.

    Keep sky components connected to the top edge, including sky around cloud holes.
    Detached bright patches below the skyline are not sky-boundary observations.
    Returns an array of shape (W,) with the horizon row index per column.
    Values of -1 indicate no sky found in that column.
    """
    _, labels = cv2.connectedComponents((sky_mask > threshold).astype(np.uint8))
    top_labels = np.unique(labels[0])
    binary = np.isin(labels, top_labels[top_labels != 0])
    last_sky_row = binary.shape[0] - 1 - np.argmax(binary[::-1], axis=0)
    return np.where(binary.any(axis=0), last_sky_row, -1).astype(np.int32)


def estimate_tilt(
    sky_mask: np.ndarray,
    hfov_deg: float,
    aspect_ratio: tuple[int, int] = (16, 9),
    threshold: float = 0.5,
) -> tuple[float, np.ndarray]:
    """Estimate camera tilt from a line fitted to the sky mask horizon.

    Returns (tilt_deg, horizon_rows) where:
    - tilt_deg: estimated tilt in degrees (positive = down)
    - horizon_rows: fitted horizon row (float) for each image column
    """
    h, w = sky_mask.shape
    horizon = find_horizon_line(sky_mask, threshold)

    valid = horizon >= 0
    if valid.sum() < w * 0.1:
        raise ValueError("Too few columns with detected sky to estimate horizon.")

    coeffs = np.polyfit(np.where(valid)[0], horizon[valid], deg=1)
    horizon_rows = np.polyval(coeffs, np.arange(w))

    horizon_v_center = np.polyval(coeffs, w / 2) / h
    tilt_deg = (0.5 - horizon_v_center) * vfov_deg(hfov_deg, aspect_ratio)
    return float(tilt_deg), horizon_rows
