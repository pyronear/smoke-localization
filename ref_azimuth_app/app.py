import dash
from dash import html, dcc, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import base64
import io
from PIL import Image
import dash_svg as svg

FOV = 54.2

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

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
    html.Div(id='image-wrapper', style={'position': 'relative', 'display': 'inline-block'}),
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
    html.Div(id="click-info", className="mb-2"),      # shows x, y
    html.Div(id="output", className="mb-2"),          # shows center azimuth

])

# Decode and display uploaded image
@app.callback(
    Output("image-wrapper", "children"),
    Output("image-width", "data"),
    Input("upload-image", "contents"),
)
def update_image(content):
    if content is None:
        return html.Div(), None

    # Decode image to get width
    content_type, encoded = content.split(',')
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    width = image.width

    # Container with image and empty SVG overlay
    return html.Div([
        html.Img(id="image", src=content,
                 style={"maxWidth": "1000px", "width": "100%", "cursor": "crosshair"},
                 n_clicks=0),
        html.Div(id="overlay", style={
            "position": "absolute", "top": "0", "left": "0", "pointerEvents": "none",
            "width": "100%", "height": "100%"
        }),
        html.Div(id="degree-scale", style={"textAlign": "center", "marginTop": "10px"})
    ], style={"position": "relative", "display": "inline-block"}), width

import dash_svg as svg  # Make sure this is at the top of your script

@app.callback(
    Output("overlay", "children"),
    Output("degree-scale", "children"),
    Input("click-coords", "data"),
    State("image-width", "data"),
)
def draw_overlay(click_data, img_width):
    if not click_data or img_width is None:
        return "", ""

    x = click_data["offsetX"]
    y = click_data["offsetY"]
    display_width = click_data["width"]
    display_height = click_data["height"]

    # Center of image in display coordinates
    center_x = display_width // 2
    center_y = y

    # Draw SVG overlay
    svg_overlay = svg.Svg([
        svg.Line(x1=center_x, y1=center_y, x2=x, y2=y, stroke="red", strokeWidth=2),
        svg.Circle(cx=str(x), cy=str(y), r="5", fill="red")
    ], width=display_width, height=display_height,
       style={"position": "absolute", "top": "0", "left": "0"})

    # Draw degree scale
    num_ticks = 11
    step_deg = FOV / (num_ticks - 1)
    tick_labels = []
    for i in range(num_ticks):
        pos = int(display_width * i / (num_ticks - 1))
        deg = round(-FOV / 2 + step_deg * i)
        tick_labels.append(
            html.Div(f"{deg:+}°", style={
                "position": "absolute",
                "left": f"{pos}px",
                "transform": "translateX(-50%)"
            })
        )
    grad = html.Div(
        tick_labels,
        style={"position": "relative", "height": "20px", "width": f"{display_width}px", "margin": "auto"}
    )

    return svg_overlay, grad



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

@app.callback(
    Output("click-info", "children"),
    Input("click-coords", "data"),
    prevent_initial_call=True
)
def show_click(data):
    if not data:
        return "Click on the image."
    return f"Clicked at x={data['offsetX']}, y={data['offsetY']} on image width {data['width']} px"


@app.callback(
    Output("center-azimuth", "children"),
    Input("compute-btn", "n_clicks"),
    State("click-coords", "data"),
    State("azimuth-input", "value"),
    State("image-width", "data"),
    prevent_initial_call=True
)
def compute_center_azimuth(n_clicks, click_data, ref_azimuth, img_width):
    if not n_clicks or not click_data or ref_azimuth is None or img_width is None:
        return "Missing input."

    x = click_data["offsetX"]
    display_width = click_data["width"]

    # Adjust x to actual image width
    x_real = x * (img_width / display_width)
    x_norm = x_real / img_width

    offset_deg = (x_norm - 0.5) * FOV
    center_azimuth = (ref_azimuth - offset_deg) % 360

    return f"Estimated camera center azimuth: {round(center_azimuth, 2)}°"



if __name__ == "__main__":
    app.run(debug=True)
