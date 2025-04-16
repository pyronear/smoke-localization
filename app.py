import streamlit as st
import numpy as np
import math
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(layout="wide")

# === Configuration ===
canvas_width = 800  # Display width
fov_h_deg = 87      # Horizontal FOV in degrees
fov_v_deg = 44      # Vertical FOV in degrees

dem_path = "dem.jpg"
img_path = "/Users/mateo/pyronear/vision/datasets/pyro-sdis/images/val/sdis-07_brison-200_2024-02-02T12-06-40.jpg"

# === Helpers ===
def load_and_resize(image_path, target_width):
    img = Image.open(image_path)
    w, h = img.size
    scale = target_width / w
    resized = img.resize((target_width, int(h * scale)))
    return img, resized, scale

# === Load and resize images ===
dem_orig, dem_resized, scale_dem = load_and_resize(dem_path, canvas_width)
img_orig, img_resized, scale_img = load_and_resize(img_path, canvas_width)

st.title("🧭 Estimate Rotation from DEM → Camera Image")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Step 1: Click on the DEM image")
    dem_canvas = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=1,
        background_image=dem_resized,
        update_streamlit=True,
        height=dem_resized.size[1],
        width=dem_resized.size[0],
        drawing_mode="point",
        key="canvas_dem"
    )

with col2:
    st.subheader("Step 2: Click on the Target image")
    img_canvas = st_canvas(
        fill_color="rgba(0, 0, 255, 0.3)",
        stroke_width=1,
        background_image=img_resized,
        update_streamlit=True,
        height=img_resized.size[1],
        width=img_resized.size[0],
        drawing_mode="point",
        key="canvas_img"
    )

# === Store clicks ===
if "clicks" not in st.session_state:
    st.session_state.clicks = {"dem": None, "target": None}

# Capture DEM click
if dem_canvas.json_data and len(dem_canvas.json_data["objects"]) > 0:
    pt = dem_canvas.json_data["objects"][-1]
    x = pt["left"]
    y = pt["top"]
    st.session_state.clicks["dem"] = (x / scale_dem, y / scale_dem)

# Capture Target click
if img_canvas.json_data and len(img_canvas.json_data["objects"]) > 0:
    pt = img_canvas.json_data["objects"][-1]
    x = pt["left"]
    y = pt["top"]
    st.session_state.clicks["target"] = (x / scale_img, y / scale_img)

# === Rotation computation ===
if st.session_state.clicks["dem"] and st.session_state.clicks["target"]:
    x1, y1 = st.session_state.clicks["dem"]
    x2, y2 = st.session_state.clicks["target"]

    dx = x2 - x1
    dy = y2 - y1

    # Image dimensions
    width = dem_orig.size[0]
    height = dem_orig.size[1]

    # Compute focal lengths from FOV
    fov_h_rad = math.radians(fov_h_deg)
    fov_v_rad = math.radians(fov_v_deg)

    f_x = width / (2 * math.tan(fov_h_rad / 2))
    f_y = height / (2 * math.tan(fov_v_rad / 2))

    # Compute rotations
    angle_horizontal = math.degrees(math.atan2(dx, f_x))
    angle_vertical = math.degrees(math.atan2(dy, f_y))

    st.markdown("### 📐 Estimated Rotation to Align DEM → Target")
    st.write(f"**Horizontal rotation**: {angle_horizontal:.2f}°")
    st.write(f"**Vertical rotation**: {angle_vertical:.2f}°")

    st.markdown("---")
    st.write(f"📌 **DEM Click**: ({x1:.1f}, {y1:.1f})")
    st.write(f"📌 **Target Click**: ({x2:.1f}, {y2:.1f})")


    import matplotlib.pyplot as plt

    st.markdown("### 🖼️ DEM Overlay on Target Image")

    # Convert images to numpy arrays (original size, not resized)
    target_np = np.array(img_orig.convert("RGB"))
    dem_np = np.array(dem_orig.convert("L"))  # Use grayscale for DEM

    # Normalize DEM for colormap display
    dem_norm = (dem_np - dem_np.min()) / (dem_np.max() - dem_np.min())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(target_np)
    ax.imshow(dem_norm, cmap="plasma", alpha=0.4)
    ax.set_title("DEM Overlay on Camera Image")
    ax.axis("off")

    st.pyplot(fig)

