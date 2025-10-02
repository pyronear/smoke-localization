import os
import base64
import io
import json
from PIL import Image
import dash
from dash import html, dcc, Input, Output, State, ctx
import dash_bootstrap_components as dbc

# ==== CONFIGURATION ====

folder_path = "/Users/mateo/pyronear/vision/smoke-localization/site_data/laluque/captured_poses/192_168_1_12"
output_json = folder_path + "_ref_azimuth.json"
FOV = 54.2  # degrees

image_paths = sorted([
    os.path.join(folder_path, f)
    for f in os.listdir(folder_path)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

# ==== APP SETUP ====

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

app.title = "Camera Azimuth Estimator"

app.index_string = """
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>Camera Azimuth Estimator</title>
    {%favicon%}
    {%css%}
    <script>
      document.addEventListener("DOMContentLoaded", function () {
        document.body.addEventListener("click", function (e) {
          const rect = e.target.getBoundingClientRect();
          window.lastClick = {
            offsetX: Math.round(e.clientX - rect.left),
            offsetY: Math.round(e.clientY - rect.top),
            width: Math.round(rect.width),
            height: Math.round(rect.height)
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

app.layout = dbc.Container([
    html.H3("Estimate Camera Azimuth from Reference Point"),
    
    dbc.Row([
        dbc.Col([
            dbc.Button("Previous", id="prev-btn", n_clicks=0, color="secondary"),
            dbc.Button("Next", id="next-btn", n_clicks=0, color="secondary", className="ms-2"),
        ], width=6),
    ], className="my-2"),

    html.Div(id='image-container', style={"textAlign": "center"}),

    html.Div(id='click-coords-display', style={"textAlign": "center", "marginTop": "10px"}),

    dcc.Store(id='image-index', data=0),
    dcc.Store(id='click-coords'),
    dcc.Store(id='image-width'),

    dbc.Row([
        dbc.Col([
            dbc.Input(id="azimuth-input", type="number", placeholder="Enter reference azimuth (°)", min=0, max=359),
        ], width=6),
        dbc.Col([
            dbc.Button("Compute Center Azimuth", id="compute-btn", color="primary"),
        ], width=6),
    ], className="my-3"),

    html.Div(id="output", style={"fontWeight": "bold", "textAlign": "center", "marginBottom": "20px"}),
])

# ==== CALLBACKS ====

@app.callback(
    Output('image-container', 'children'),
    Output('image-width', 'data'),
    Output('image-index', 'data'),
    Input('prev-btn', 'n_clicks'),
    Input('next-btn', 'n_clicks'),
    State('image-index', 'data'),
)
def update_image(prev_clicks, next_clicks, current_index):
    triggered_id = ctx.triggered_id
    total_images = len(image_paths)

    if triggered_id == 'prev-btn':
        new_index = (current_index - 1) % total_images
    elif triggered_id == 'next-btn':
        new_index = (current_index + 1) % total_images
    else:
        new_index = current_index

    path = image_paths[new_index]
    with open(path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode()
    img_src = f"data:image/jpeg;base64,{encoded}"
    
    with Image.open(path) as img:
        width = img.width

    return html.Img(id="image", src=img_src, style={"width": "100%", "cursor": "crosshair"}, n_clicks=0), width, new_index

# Capture click coordinates (client-side)
app.clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) return window.dash_clientside.no_update;
        return window.lastClick || null;
    }
    """,
    Output("click-coords", "data"),
    Input("image", "n_clicks"),
    prevent_initial_call=True
)

@app.callback(
    Output("click-coords-display", "children"),
    Input("click-coords", "data")
)
def show_click(click_data):
    if click_data and click_data["offsetX"] is not None:
        return f"Clicked at: x = {click_data['offsetX']}, y = {click_data['offsetY']}"
    return "Click anywhere on the image to set a reference point."

@app.callback(
    Output("output", "children"),
    Input("compute-btn", "n_clicks"),
    State("click-coords", "data"),
    State("azimuth-input", "value"),
    State("image-width", "data"),
    State("image-index", "data"),
)
def compute_center(n, click_data, ref_azimuth, img_width, index):
    if not n or not click_data or ref_azimuth is None or img_width is None:
        return ""

    x = click_data["offsetX"]
    display_width = click_data["width"]

    if not display_width or display_width == 0:
        return "Invalid image display width."

    # Rescale click to real image width
    x_real = x * (img_width / display_width)
    x_norm = x_real / img_width
    offset_deg = (x_norm - 0.5) * FOV
    center_azimuth = (ref_azimuth - offset_deg) % 360
    center_azimuth = round(center_azimuth, 2)

    img_name = os.path.basename(image_paths[index])

    # Save to JSON (overwrite if exists)
    try:
        if os.path.exists(output_json):
            with open(output_json, 'r') as f:
                data = json.load(f)
        else:
            data = {}

        data[img_name] = {
            "ref_click": {"x": round(x_real, 1)},
            "ref_azimuth": ref_azimuth,
            "computed_center_azimuth": center_azimuth
        }

        with open(output_json, 'w') as f:
            json.dump(data, f, indent=2)

    except Exception as e:
        return f"Error saving result: {e}"

    return f"Estimated center azimuth: {center_azimuth}° (saved for {img_name})"

# ==== RUN APP ====

if __name__ == "__main__":
    app.run(debug=True)
