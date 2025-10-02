# align_app/callbacks.py
import os
import base64
from dash import Input, Output, State, ctx, no_update
import numpy as np
import pandas as pd

def encode_image(path):
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

def register_callbacks(app):
    @app.callback(
        Output("camera-index", "data"),
        Input("camera-prev", "n_clicks"),
        Input("camera-next", "n_clicks"),
        State("camera-index", "data"),
    )
    def update_camera_index(prev, next_, index):
        paths = app.camera_paths
        index = index or 0
        if ctx.triggered_id == "camera-prev":
            return (index - 1) % len(paths)
        if ctx.triggered_id == "camera-next":
            return (index + 1) % len(paths)
        return index

    @app.callback(
        Output("dem-index", "data"),
        Input("dem-prev", "n_clicks"),
        Input("dem-next", "n_clicks"),
        State("dem-index", "data"),
    )
    def update_dem_index(prev, next_, index):
        paths = app.dem_paths
        index = index or 0
        if ctx.triggered_id == "dem-prev":
            return (index - 1) % len(paths)
        if ctx.triggered_id == "dem-next":
            return (index + 1) % len(paths)
        return index

    @app.callback(
        Output("camera-image", "src"),
        Input("camera-index", "data"),
    )
    def update_camera_image(index):
        paths = app.camera_paths
        if index is None or index >= len(paths):
            return no_update
        return encode_image(paths[index])

    @app.callback(
        Output("dem-image", "src"),
        Input("dem-index", "data"),
    )
    def update_dem_image(index):
        paths = app.dem_paths
        if index is None or index >= len(paths):
            return no_update
        return encode_image(paths[index])
    
    @app.callback(
        Output("camera-filename", "children"),
        Input("camera-index", "data"),
    )
    def update_camera_filename(index):
        paths = app.camera_paths
        if not paths or index is None or index >= len(paths):
            return ""
        return os.path.basename(paths[index])

    @app.callback(
        Output("dem-filename", "children"),
        Input("dem-index", "data"),
    )
    def update_dem_filename(index):
        paths = app.dem_paths
        if not paths or index is None or index >= len(paths):
            return ""
        return os.path.basename(paths[index])

    # ─── New: update filename displays ────────────────────────────────────────

    app.clientside_callback(
        """
        function(n_clicks) {
          // only fire when n_clicks > 0
          if (!n_clicks) {
            return window.dash_clientside.no_update;
          }
          // grab the global window.lastClick object
          return window.lastClick || null;
        }
        """,
        Output("camera-coords-store", "data"),
        Input("camera-overlay", "n_clicks"),
        prevent_initial_call=True
    )
    app.clientside_callback(
        """
        function(n_clicks) {
          if (!n_clicks) {
            return window.dash_clientside.no_update;
          }
          return window.lastClick || null;
        }
        """,
        Output("dem-coords-store", "data"),
        Input("dem-overlay", "n_clicks"),
        prevent_initial_call=True
    )

    @app.callback(
        Output("camera-coords", "children"),
        Input("camera-coords-store", "data"),
    )
    def show_camera_coords(data):
        if not data:
            return ""
        return f"Clicked at x={data['offsetX']}, y={data['offsetY']} on {data['width']}×{data['height']}"

    @app.callback(
        Output("dem-coords", "children"),
        Input("dem-coords-store", "data"),
    )
    def show_dem_coords(data):
        if not data:
            return ""
        return f"Clicked at x={data['offsetX']}, y={data['offsetY']} on {data['width']}×{data['height']}"
    
        
    @app.callback(
        Output("azimuth-result", "children"),
        Input("compute-btn",        "n_clicks"),
        State("camera-coords-store","data"),
        State("dem-coords-store",   "data"),
        State("camera-index",       "data"),
        State("dem-index",          "data"),
        State("csv-path-store",     "data")
    )
    def compute_azimuth(n_clicks, cam_click, dem_click, cam_idx, dem_idx, csv_path):
        if n_clicks == 0:
            return no_update

        if not cam_click or not dem_click:
            return "Please click on both images first"

        # Get filenames
        cam_file = os.path.basename(app.camera_paths[cam_idx])
        dem_file = os.path.basename(app.dem_paths[dem_idx])

        # Extract true azimuth from DEM filename
        true_az = float(os.path.splitext(dem_file)[0])

        # Compute delta and centre azimuth
        dx      = dem_click["offsetX"] - cam_click["offsetX"]
        W       = dem_click["width"]
        fov_rad = np.radians(54.2)
        delta   = np.degrees(2 * np.arctan((dx * np.tan(fov_rad/2)) / W))
        centre  = (true_az + delta) % 360

        # Create a DataFrame for the new row
        new_row = pd.DataFrame([{
            "img_name": cam_file,
            "dem_image": dem_file,
            "dx": dx,
            "real_azimuth": true_az,
            "centre": round(centre, 2)
        }])

        # Ensure directory exists
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        # Append to CSV
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df = new_row

        df.to_csv(csv_path, index=False)

        return f"centre azimuth: {centre:.2f}°  ·  saved to {os.path.basename(csv_path)}"
