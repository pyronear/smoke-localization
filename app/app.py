"""Streamlit app — click on a camera image to project the pixel onto a map."""

import csv
import math
import os

import cv2
import numpy as np
import streamlit as st
import folium
from PIL import Image
from streamlit_folium import st_folium
from streamlit_image_coordinates import streamlit_image_coordinates

from smoke_localisation import (
    detect_sky,
    estimate_tilt,
    find_horizon_line,
    load_dsm,
    pixel_to_ray,
    ray_ground_intersection,
)

st.set_page_config(page_title="Smoke Localisation", layout="wide")
st.title("Smoke Localisation")

# ── Load demo cameras from CSV ─────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")
CAMERAS_CSV = os.path.join(DATA_DIR, "cameras.csv")


@st.cache_data
def load_cameras():
    cameras = {}
    with open(CAMERAS_CSV) as f:
        for row in csv.DictReader(f):
            cameras[row["name"]] = {
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "fov_h": float(row["fov_h"]),
                "height": float(row["height"]),
            }
    return cameras


@st.cache_data
def list_demo_images():
    """Return {camera_name: [filename, ...]} for images in data/samples/."""
    images = {}
    if not os.path.isdir(SAMPLES_DIR):
        return images
    for fname in sorted(os.listdir(SAMPLES_DIR)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        for cam_name in load_cameras():
            if cam_name in fname:
                images.setdefault(cam_name, []).append(fname)
                break
    return images


def parse_azimuth_from_filename(fname):
    """Extract azimuth from filename pattern: source_site_azimuth_time."""
    stem = os.path.splitext(fname)[0]
    parts = stem.split("_")
    for part in reversed(parts):
        if part.isdigit() and 0 <= int(part) <= 360:
            return float(part)
    return None


cameras = load_cameras()
demo_images = list_demo_images()

# ── Sidebar: demo selector + camera parameters ────────────────────────
st.sidebar.header("Demo examples")
demo_options = ["(none — upload your own)"]
for cam_name, fnames in demo_images.items():
    for fname in fnames:
        demo_options.append(f"{cam_name}: {fname}")

selected_demo = st.sidebar.selectbox("Load a demo image", demo_options)

# Determine defaults from demo selection
if selected_demo != demo_options[0]:
    demo_cam_name = selected_demo.split(":")[0].strip()
    demo_fname = selected_demo.split(":", 1)[1].strip()
    cam_defaults = cameras[demo_cam_name]
    demo_azimuth = parse_azimuth_from_filename(demo_fname)
else:
    demo_cam_name = None
    demo_fname = None
    demo_azimuth = None
    cam_defaults = {"lat": 44.545, "lon": 4.216, "fov_h": 87.0, "height": 15.0}

st.sidebar.header("Camera parameters")
cam_location = st.sidebar.text_input(
    "Location (lat, lon)",
    value=f"{cam_defaults['lat']}, {cam_defaults['lon']}",
)
try:
    cam_lat, cam_lon = (float(x.strip()) for x in cam_location.split(","))
except ValueError:
    st.sidebar.error("Enter location as: lat, lon")
    st.stop()
cam_azimuth = st.sidebar.number_input(
    "Azimuth (deg, clockwise from N)",
    value=demo_azimuth if demo_azimuth is not None else 184.0,
    format="%.1f",
)
cam_hfov = st.sidebar.number_input("Horizontal FOV (deg)", value=cam_defaults["fov_h"], format="%.1f")
cam_height = st.sidebar.number_input("Height above ground (m)", value=cam_defaults["height"], format="%.1f")

auto_tilt = st.sidebar.checkbox("Auto-calibrate tilt from sky detection", value=True)
manual_tilt = st.sidebar.number_input(
    "Manual tilt (deg, positive = down)", value=3.0, format="%.1f",
    disabled=auto_tilt,
)

# ── Cached helpers ─────────────────────────────────────────────────────


@st.cache_data(show_spinner="Downloading DSM tiles...")
def cached_load_dsm(lat, lon, margin):
    return load_dsm(lat, lon, margin_deg=margin)


@st.cache_data(show_spinner="Running sky detection...")
def cached_detect_sky(image_bytes):
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return detect_sky(bgr)


# ── Image: demo or upload ──────────────────────────────────────────────
if demo_fname is not None:
    demo_path = os.path.join(SAMPLES_DIR, demo_fname)
    image = Image.open(demo_path)
    with open(demo_path, "rb") as f:
        image_bytes = f.read()
else:
    uploaded = st.file_uploader("Upload a camera image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload an image or select a demo example from the sidebar.")
        st.stop()
    image = Image.open(uploaded)
    uploaded.seek(0)
    image_bytes = uploaded.read()

img_w, img_h = image.size
aspect_w, aspect_h = img_w, img_h

# ── Sky detection & tilt calibration ────────────────────────────────────
if auto_tilt:
    sky_mask = cached_detect_sky(image_bytes)
    tilt_deg, horizon_v_mean, horizon_v_center = estimate_tilt(
        sky_mask, cam_hfov, aspect_ratio=(aspect_w, aspect_h)
    )
    cam_tilt = tilt_deg

    # Build overlay: tint sky blue + draw horizon line
    sky_overlay = np.array(image).copy()
    sky_overlay[sky_mask > 0.5, 2] = np.clip(
        sky_overlay[sky_mask > 0.5, 2].astype(np.int16) + 80, 0, 255
    ).astype(np.uint8)
    horizon = find_horizon_line(sky_mask)
    valid = horizon >= 0
    cols_valid = np.where(valid)[0]
    rows_valid = horizon[valid]
    if len(cols_valid) > 1:
        coeffs = np.polyfit(cols_valid, rows_valid, deg=1)
        for c in range(img_w):
            r = int(coeffs[0] * c + coeffs[1])
            if 0 <= r < img_h:
                sky_overlay[max(0, r-1):min(img_h, r+2), c] = [255, 0, 0]
    overlay_image = Image.fromarray(sky_overlay)
else:
    cam_tilt = manual_tilt
    overlay_image = image

# ── Layout ──────────────────────────────────────────────────────────────
col_img, col_map = st.columns(2)

with col_img:
    st.subheader("Click on the image")
    if auto_tilt:
        st.caption(
            f"Sky-calibrated tilt: **{cam_tilt:.2f}°** "
            f"(horizon at v={horizon_v_center:.3f})"
        )
    else:
        st.caption(f"Manual tilt: **{cam_tilt:.1f}°**")

    display_width = 700
    coords = streamlit_image_coordinates(overlay_image, width=display_width, key="click")

if coords is None:
    with col_map:
        st.info("Click on the image to project a ray.")
    st.stop()

# Convert display-pixel click -> normalised [0,1] coordinates
scale = img_w / display_width
click_x = coords["x"] * scale
click_y = coords["y"] * scale
u = click_x / img_w
v = click_y / img_h

with col_img:
    st.markdown(f"**Pixel**: ({click_x:.0f}, {click_y:.0f}) / ({img_w}x{img_h})  ->  "
                f"**normalised**: ({u:.4f}, {v:.4f})")

# ── Check if pixel is in sky ────────────────────────────────────────────
if auto_tilt:
    pixel_row = min(int(click_y), img_h - 1)
    pixel_col = min(int(click_x), img_w - 1)
    if sky_mask[pixel_row, pixel_col] > 0.5:
        with col_img:
            st.warning("Pixel is in the sky -- no ground projection possible.")
        with col_map:
            st.subheader("Map")
            m = folium.Map(location=[cam_lat, cam_lon], zoom_start=13)
            folium.Marker(
                [cam_lat, cam_lon], tooltip="Camera",
                icon=folium.Icon(color="blue", icon="video", prefix="fa"),
            ).add_to(m)
            st_folium(m, use_container_width=True, height=500)
        st.stop()

# ── Compute ray ─────────────────────────────────────────────────────────
ray_az, ray_el = pixel_to_ray(
    u, v, cam_azimuth, cam_hfov,
    aspect_ratio=(aspect_w, aspect_h),
    tilt_deg=cam_tilt,
)

with col_img:
    st.markdown(f"**Ray**: azimuth {ray_az:.2f}°, elevation {ray_el:.2f}°")

# ── DSM + intersection ──────────────────────────────────────────────────
elev, lats, lons = cached_load_dsm(cam_lat, cam_lon, margin=0.3)

result = ray_ground_intersection(
    cam_lat, cam_lon, cam_height,
    ray_az, ray_el,
    elev, lats, lons,
    max_distance_m=30_000,
    step_m=5,
    verbose=False,
)

# ── Map ─────────────────────────────────────────────────────────────────
with col_map:
    st.subheader("Map")
    if result is None:
        st.warning("Ray did not hit the terrain (pixel may be above the horizon).")
        m = folium.Map(location=[cam_lat, cam_lon], zoom_start=13)
        folium.Marker(
            [cam_lat, cam_lon],
            tooltip="Camera",
            icon=folium.Icon(color="blue", icon="video", prefix="fa"),
        ).add_to(m)
    else:
        hit_lat, hit_lon, hit_elev, dist = result
        st.success(
            f"**Hit**: {hit_lat:.6f}, {hit_lon:.6f}  \n"
            f"**Elevation**: {hit_elev:.1f} m -- **Distance**: {dist:.0f} m"
        )

        center_lat = (cam_lat + hit_lat) / 2
        center_lon = (cam_lon + hit_lon) / 2
        zoom = max(10, min(17, int(17 - math.log2(max(dist, 100) / 100))))
        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom)

        folium.Marker(
            [cam_lat, cam_lon],
            tooltip="Camera",
            icon=folium.Icon(color="blue", icon="video", prefix="fa"),
        ).add_to(m)

        folium.Marker(
            [hit_lat, hit_lon],
            tooltip=f"Hit: {hit_lat:.6f}, {hit_lon:.6f} ({dist:.0f} m)",
            icon=folium.Icon(color="red", icon="fire", prefix="fa"),
        ).add_to(m)

        folium.PolyLine(
            [[cam_lat, cam_lon], [hit_lat, hit_lon]],
            color="red", weight=2, dash_array="5",
        ).add_to(m)

    st_folium(m, use_container_width=True, height=500)
