"""Sky segmentation (ncnn) and horizon-based tilt estimation."""

import math
from pathlib import Path

import cv2
import ncnn
import numpy as np

_MODEL_DIR = Path(__file__).resolve().parent / "models" / "skyseg"
_PARAM_PATH = _MODEL_DIR / "skysegsmall_sim-opt-fp16.param"
_BIN_PATH = _MODEL_DIR / "skysegsmall_sim-opt-fp16.bin"

_INPUT_SIZE = 384


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
        image_bgr, ncnn.Mat.PixelType.PIXEL_BGR2RGB, w, h, _INPUT_SIZE, _INPUT_SIZE
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

    For each column, the horizon is the last row (from top) where sky > threshold.
    Returns an array of shape (W,) with the horizon row index per column.
    Values of -1 indicate no sky found in that column.
    """
    binary = (sky_mask > threshold).astype(np.uint8)
    w = binary.shape[1]
    horizon = np.full(w, -1, dtype=np.int32)
    for col in range(w):
        sky_rows = np.where(binary[:, col] == 1)[0]
        if len(sky_rows) > 0:
            horizon[col] = sky_rows.max()
    return horizon


def estimate_tilt(
    sky_mask: np.ndarray,
    hfov_deg: float,
    aspect_ratio: tuple[int, int] = (16, 9),
    threshold: float = 0.5,
) -> tuple[float, float, float]:
    """Estimate camera tilt from the sky mask horizon line.

    Returns (tilt_deg, horizon_v_mean, horizon_v_center) where:
    - tilt_deg: estimated tilt in degrees (positive = down)
    - horizon_v_mean: mean normalised v-position of horizon across all columns
    - horizon_v_center: normalised v-position of horizon at the image centre column
    """
    h, w = sky_mask.shape
    horizon = find_horizon_line(sky_mask, threshold)

    valid = horizon >= 0
    if valid.sum() < w * 0.1:
        raise ValueError("Too few columns with detected sky to estimate horizon.")

    cols_valid = np.where(valid)[0]
    rows_valid = horizon[valid]
    coeffs = np.polyfit(cols_valid, rows_valid, deg=1)
    a, b = coeffs

    center_col = w / 2
    horizon_row_center = a * center_col + b
    horizon_v_center = horizon_row_center / h

    horizon_v_mean = rows_valid.mean() / h

    vfov_deg = math.degrees(
        2 * math.atan(math.tan(math.radians(hfov_deg) / 2) * aspect_ratio[1] / aspect_ratio[0])
    )
    tilt_deg = (0.5 - horizon_v_center) * vfov_deg

    return tilt_deg, horizon_v_mean, horizon_v_center
