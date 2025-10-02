import os
import base64
from dash import Dash, html, dcc
from align_app import callbacks 


# === Global Config ===

name = "videlles"
cam_id = "192_168_1_12"

folders = {
    "camera_images": f"/Users/mateo/pyronear/vision/smoke-localization/site_data/{name}/captured_poses/{cam_id}",
    "dem_generated": f"/Users/mateo/pyronear/vision/smoke-localization/site_data/{name}/dem/max",
}

print(folders)

# Read paths once at app init
camera_paths = sorted([
    os.path.join(folders["camera_images"], f)
    for f in os.listdir(folders["camera_images"])
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])
dem_paths = sorted([
    os.path.join(folders["dem_generated"], f)
    for f in os.listdir(folders["dem_generated"])
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

# === Reusable Styles ===

panel_style = {
    "display": "flex",
    "flexDirection": "column",
    "padding": "10px",
    "width": "15%",
}

image_container_style = {
    "width": "85%",
}

overlay_style = {
    "position": "absolute",
    "top": 0,
    "left": 0,
    "width": "700px",
    "height": "100%",
    "zIndex": 2,
    "backgroundColor": "rgba(255, 255, 255, 0.01)",
    "cursor": "crosshair",
}

image_style = {
    "width": "700px",
    "height": "auto",
    "cursor": "crosshair",
    "border": "1px solid #ccc",
    "objectFit": "contain",
}

def encode_image(path):
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()


# === App Definition ===

app = Dash(__name__)
app.title = "Dual Image Viewer"

app.index_string = """
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>Dual Image Viewer</title>
    {%favicon%}
    {%css%}
    <script>
      document.addEventListener("DOMContentLoaded", function () {
        document.body.addEventListener("click", function (e) {
          const rect = e.target.getBoundingClientRect();
          window.lastClick = {
            offsetX: Math.round(e.clientX - rect.left),
            offsetY: Math.round(e.clientY - rect.top),
            width:   Math.round(rect.width),
            height:  Math.round(rect.height)
          };
        });
      });
    </script>
  </head>
  <body>
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
    </footer>
  </body>
</html>
"""

# === Layout ===

app.layout = html.Div(
    [
        # ── LEFT COLUMN: CAMERA + DEM VIEWERS ─────────────────────────
        html.Div(
            [
                # CAMERA BLOCK
                html.Div(
                    [
                        html.Div(
                            [
                                html.Button("Previous", id="camera-prev", n_clicks=0, style={"marginBottom":"10px"}),
                                html.Button("Next",     id="camera-next", n_clicks=0),
                                html.Div( id="camera-coords",   style={"marginTop":"20px","fontWeight":"bold"}),
                                html.Div( id="camera-filename", style={"marginTop":"10px","fontSize":"14px"}),
                            ],
                            style=panel_style,
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Img( id="camera-image", style=image_style),
                                        html.Div( id="camera-overlay", n_clicks=0, style=overlay_style),
                                    ],
                                    style={"position":"relative"},
                                )
                            ],
                            style=image_container_style,
                        ),
                    ],
                    style={"display":"flex","marginBottom":"30px"},
                ),

                # DEM BLOCK
                html.Div(
                    [
                        html.Div(
                            [
                                html.Button("Previous", id="dem-prev", n_clicks=0, style={"marginBottom":"10px"}),
                                html.Button("Next",     id="dem-next", n_clicks=0),
                                html.Div( id="dem-coords",   style={"marginTop":"20px","fontWeight":"bold"}),
                                html.Div( id="dem-filename", style={"marginTop":"10px","fontSize":"14px"}),
                            ],
                            style=panel_style,
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Img( id="dem-image",   style=image_style),
                                        html.Div( id="dem-overlay", n_clicks=0, style=overlay_style),
                                    ],
                                    style={"position":"relative"},
                                )
                            ],
                            style=image_container_style,
                        ),
                    ],
                    style={"display":"flex","marginBottom":"30px"},
                ),
            ],
            style={"width":"70%","display":"flex","flexDirection":"column","padding":"20px"},
        ),

        # ── RIGHT COLUMN: COMPUTE PANEL ────────────────────────────────
        html.Div(
            [
                html.Button(
                    "Compute Azimuth",
                    id="compute-btn",
                    n_clicks=0,
                    style={"marginBottom":"20px","width":"100%"}
                ),
                html.Div(
                    id="azimuth-result",
                    style={"fontWeight":"bold","textAlign":"center"}
                ),
            ],
            style={
                "width":"30%",
                "display":"flex",
                "flexDirection":"column",
                "justifyContent":"center",
                "alignItems":"center",
                "padding":"20px",
                "borderLeft":"1px solid #ddd",
            },
        ),

        # ── STORES ─────────────────────────────────────────────────────
        dcc.Store(id="camera-index",       data=0),
        dcc.Store(id="camera-coords-store"),
        dcc.Store(id="dem-index",          data=0),
        dcc.Store(id="dem-coords-store"),
        dcc.Store(id="csv-path-store",    data=f"/Users/mateo/pyronear/vision/smoke-localization/site_data/{name}/df_ref_images_{cam_id}.csv")
    ],

    # root flex container
    style={"display":"flex","height":"100vh","boxSizing":"border-box"}
)



import align_app.callbacks as callbacks
# Share data with callback module
app.camera_paths = camera_paths
app.dem_paths = dem_paths

# register all callbacks on our one app instance
callbacks.register_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True)

