"""Streamlit app — click on a camera image to project the pixel onto a map."""

import csv
import os

import cv2
import folium
import numpy as np
import pyproj
import streamlit as st
from PIL import Image
from streamlit_folium import st_folium
from streamlit_image_coordinates import streamlit_image_coordinates

from smoke_localisation import (
    detect_sky,
    estimate_tilt,
    load_dsm,
    pixel_to_ray,
    ray_ground_intersection,
)

st.set_page_config(page_title="Smoke Localisation", layout="wide")
st.title("Smoke Localisation")
st.caption(
    "Click on the base of a smoke plume to project it onto the terrain. "
    "Camera parameters are in the sidebar."
)

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

# ── Sky detection & tilt calibration ────────────────────────────────────
sky_mask = cached_detect_sky(image_bytes) if auto_tilt else None
tilt_fit = None
if auto_tilt:
    try:
        tilt_fit = estimate_tilt(sky_mask, cam_hfov, aspect_ratio=(img_w, img_h))
    except ValueError as e:
        st.sidebar.warning(f"{e} Set the tilt manually below.")

if tilt_fit is None:
    cam_tilt = st.sidebar.number_input("Manual tilt (deg, positive = down)", value=3.0, format="%.1f")
    tilt_caption = f"Tilt {cam_tilt:.1f}° (manual)"
    overlay_image = image
else:
    cam_tilt, horizon_rows = tilt_fit
    tilt_caption = f"Tilt {cam_tilt:.2f}° (from sky detection, horizon line in red)"
    # Light blue tint on the sky + horizon line
    rgb = np.array(image.convert("RGB"), dtype=np.float32)
    sky = sky_mask > 0.5
    rgb[sky] = 0.7 * rgb[sky] + 0.3 * np.array([120, 180, 255])
    rows = horizon_rows.round().astype(int)
    cols = np.arange(img_w)
    inside = (rows >= 0) & (rows < img_h - 1)
    for dr in (0, 1):
        rgb[rows[inside] + dr, cols[inside]] = [255, 60, 60]
    overlay_image = Image.fromarray(rgb.astype(np.uint8))

# ── Layout ──────────────────────────────────────────────────────────────
col_img, col_map = st.columns(2, gap="large")

with col_img:
    coords = streamlit_image_coordinates(
        overlay_image, use_column_width="always", cursor="crosshair", key="click"
    )
    st.caption(tilt_caption)

# ── Click -> ray -> terrain hit ─────────────────────────────────────────
result = None
if coords is not None:
    u = coords["x"] / coords["width"]
    v = coords["y"] / coords["height"]
    pixel_row = min(int(v * img_h), img_h - 1)
    pixel_col = min(int(u * img_w), img_w - 1)
    details = [f"Pixel ({pixel_col}, {pixel_row}) of {img_w}x{img_h}, normalised ({u:.4f}, {v:.4f})"]

    with col_img:
        if sky_mask is not None and sky_mask[pixel_row, pixel_col] > 0.5:
            st.warning("This pixel is in the sky: no ground projection possible.")
        else:
            ray_az, ray_el = pixel_to_ray(
                u, v, cam_azimuth, cam_hfov,
                aspect_ratio=(img_w, img_h),
                tilt_deg=cam_tilt,
            )
            details.append(f"Ray azimuth {ray_az:.2f}°, elevation {ray_el:.2f}°")
            elev, lats, lons = cached_load_dsm(cam_lat, cam_lon, margin=0.3)
            result = ray_ground_intersection(
                cam_lat, cam_lon, cam_height,
                ray_az, ray_el,
                elev, lats, lons,
                max_distance_m=30_000,
                step_m=5,
            )
            if result is None:
                st.warning("The ray does not hit the terrain within 30 km.")
            else:
                hit_lat, hit_lon, hit_elev, dist = result
                c1, c2 = st.columns(2)
                c1.metric("Distance", f"{dist / 1000:.2f} km")
                c2.metric("Elevation", f"{hit_elev:.0f} m")
                st.code(f"{hit_lat:.6f}, {hit_lon:.6f}", language=None)
        with st.expander("Details"):
            st.markdown("  \n".join(details))

# ── Map: camera field of view, plus the hit once a pixel is clicked ────
with col_map:
    geod = pyproj.Geod(ellps="WGS84")
    camera = [cam_lat, cam_lon]
    fov_edges = []
    for side in (-1, 1):
        lon, lat, _ = geod.fwd(cam_lon, cam_lat, cam_azimuth + side * cam_hfov / 2, 10_000)
        fov_edges.append([lat, lon])

    m = folium.Map(location=camera, zoom_start=12)
    folium.Polygon(
        [camera, *fov_edges], color="#3388ff", weight=1, fill=True, fill_opacity=0.1,
        tooltip="Camera field of view (10 km)",
    ).add_to(m)
    folium.Marker(
        camera, tooltip="Camera", icon=folium.Icon(color="blue", icon="video", prefix="fa"),
    ).add_to(m)

    if result is None:
        m.fit_bounds([camera, *fov_edges])
    else:
        hit = [hit_lat, hit_lon]
        folium.PolyLine([camera, hit], color="red", weight=2, dash_array="5").add_to(m)
        folium.Marker(
            hit, tooltip=f"Hit: {hit_lat:.6f}, {hit_lon:.6f} ({dist:.0f} m)",
            icon=folium.Icon(color="red", icon="fire", prefix="fa"),
        ).add_to(m)
        m.fit_bounds([camera, hit], padding=(40, 40))

    st_folium(m, use_container_width=True, height=500, returned_objects=[])
