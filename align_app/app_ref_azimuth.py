import dash
from dash import html, dcc, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import base64
import io
from PIL import Image

FOV = 54.2

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Camera Azimuth Estimator"

# Custom index with click tracking
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

app.layout = dbc.Container([
    html.H2("Upload an image, click a reference point, and enter its azimuth"),
    dcc.Upload(
        id='upload-image',
        children=html.Div(['Drag and drop or click to upload an image']),
        style={'border': '2px dashed #888', 'padding': '20px', 'textAlign': 'center'},
        multiple=False
    ),
    html.Div(id='image-container'),
    dcc.Store(id="click-coords"),
    dcc.Store(id="image-width"),
    dbc.Row([
        dbc.Col([
            dbc.Input(id="azimuth-input", type="number", placeholder="Enter clicked point azimuth (°)", min=0, max=359),
        ], width=6),
        dbc.Col([
            dbc.Button("Compute Center Azimuth", id="compute-btn", color="primary", className="ms-2"),
        ], width=6),
    ], className="my-3"),
    html.Div(id="output")
])

# Decode and display uploaded image
@app.callback(
    Output("image-container", "children"),
    Output("image-width", "data"),
    Input("upload-image", "contents"),
)
def update_image(content):
    if content is None:
        return html.Div(), None

    # Decode base64 image
    content_type, encoded = content.split(',')
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    width = image.width

    return html.Img(id="image", src=content, style={"width": "100%", "cursor": "crosshair"}, n_clicks=0), width

# Client-side callback to capture click
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

# Compute center azimuth
@app.callback(
    Output("output", "children"),
    Input("compute-btn", "n_clicks"),
    State("click-coords", "data"),
    State("azimuth-input", "value"),
    State("image-width", "data"),
)
def compute_center(n, click_data, ref_azimuth, img_width):
    if not n or not click_data or ref_azimuth is None or img_width is None:
        return ""

    x = click_data["offsetX"]
    display_width = click_data["width"]

    # Scale x from display width to true image width
    x_real = x * (img_width / display_width)
    x_norm = x_real / img_width

    offset_deg = (x_norm - 0.5) * FOV
    center_azimuth = (ref_azimuth - offset_deg) % 360

    return f"Estimated camera center azimuth: {round(center_azimuth, 2)}°"

if __name__ == "__main__":
    app.run_server(debug=True)
