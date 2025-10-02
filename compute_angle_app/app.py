import os
import base64
import json
from dash import Dash, html, dcc, Output, Input, State
import dash

# ==== CONFIGURATION ====

folder_path = "/Users/mateo/pyronear/vision/smoke-localization/site_data/laluque/captured_poses/192_168_1_12"
output_json = folder_path + "_angle_shift.json"
FOV_X = 54.2  # degrees

image_paths = sorted([
    os.path.join(folder_path, f)
    for f in os.listdir(folder_path)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

def encode_image(image_path):
    with open(image_path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode()
    return f'data:image/jpeg;base64,{encoded}'

# ==== APP SETUP ====

app = Dash(__name__)
app.title = "Image Pair Viewer"

app.index_string = """
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>Image Pair Viewer</title>
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

app.layout = html.Div([
    html.Div([
        html.Button("Previous", id="prev-btn", n_clicks=0),
        html.Button("Next", id="next-btn", n_clicks=0),
    ], style={"textAlign": "center", "margin": "20px"}),

    html.Div([
        html.Div([
            html.Img(id='image-1', n_clicks=0, style={"width": "1000px", "border": "1px solid black", "cursor": "crosshair"}),
            html.Div(id="coords-1", style={"textAlign": "center", "margin": "10px"})
        ], style={"marginBottom": "40px"}),

        html.Div([
            html.Img(id='image-2', n_clicks=0, style={"width": "1000px", "border": "1px solid black", "cursor": "crosshair"}),
            html.Div(id="coords-2", style={"textAlign": "center", "margin": "10px"})
        ]),
    ], style={"textAlign": "center"}),

    html.Button("Compute Overlap", id="compute-btn", n_clicks=0, style={"margin": "20px"}),

    html.Div(id="angle-result", style={"textAlign": "center", "fontWeight": "bold", "margin": "10px"}),

    dcc.Store(id='image-index', data=0),
    dcc.Store(id='click1-store'),
    dcc.Store(id='click2-store'),
])

# ==== CALLBACKS ====
@app.callback(
    Output('image-1', 'src'),
    Output('image-2', 'src'),
    Output('image-index', 'data'),
    Input('prev-btn', 'n_clicks'),
    Input('next-btn', 'n_clicks'),
    State('image-index', 'data'),
)
def update_images(prev_clicks, next_clicks, current_index):
    triggered_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]

    total_pairs = len(image_paths) - 1  # total number of image pairs

    if triggered_id == 'prev-btn':
        new_index = (current_index - 1) % total_pairs
    elif triggered_id == 'next-btn':
        new_index = (current_index + 1) % total_pairs
    else:
        new_index = current_index

    img1 = encode_image(image_paths[new_index])
    img2 = encode_image(image_paths[new_index + 1])

    return img1, img2, new_index



app.clientside_callback(
    """
    function(n_clicks) {
        const click = window.lastClick || {offsetX: null, offsetY: null};
        return [click.offsetX, click.offsetY, click.width];
    }
    """,
    Output('click1-store', 'data'),
    Input('image-1', 'n_clicks')
)

app.clientside_callback(
    """
    function(n_clicks) {
        const click = window.lastClick || {offsetX: null, offsetY: null};
        return [click.offsetX, click.offsetY, click.width];
    }
    """,
    Output('click2-store', 'data'),
    Input('image-2', 'n_clicks')
)

@app.callback(
    Output('coords-1', 'children'),
    Input('click1-store', 'data')
)
def update_coords1(coords):
    if coords and coords[0] is not None:
        return f"Image 1 Click: x={coords[0]}, y={coords[1]}"
    return "No click yet"

@app.callback(
    Output('coords-2', 'children'),
    Input('click2-store', 'data')
)
def update_coords2(coords):
    if coords and coords[0] is not None:
        return f"Image 2 Click: x={coords[0]}, y={coords[1]}"
    return "No click yet"
@app.callback(
    Output('angle-result', 'children'),
    Input('compute-btn', 'n_clicks'),
    State('click1-store', 'data'),
    State('click2-store', 'data'),
    State('image-index', 'data'),
)
def compute_angle(n_clicks, click1, click2, index):
    if not click1 or not click2 or click1[0] is None or click2[0] is None:
        return "Please click both images first."

    x1, y1, width1 = click1
    x2, y2, width2 = click2

    if width1 != width2 or width1 is None:
        return "Image widths are inconsistent."

    delta_x = x1 - x2  # FIXED SIGN HERE
    angle = (delta_x / width1) * FOV_X
    angle = round(angle, 2)
    overlap = FOV_X - angle

    img1_name = os.path.basename(image_paths[index])
    img2_name = os.path.basename(image_paths[index + 1])
    key = f"{img1_name}__{img2_name}"

    result = {
        "image_1": img1_name,
        "image_2": img2_name,
        "click_1": {"x": x1, "y": y1},
        "click_2": {"x": x2, "y": y2},
        "angle_deg": angle,
        "overlap" : overlap
    }

    # Save to JSON (overwrite key if exists)
    try:
        if os.path.exists(output_json):
            with open(output_json, 'r') as f:
                data = json.load(f)
        else:
            data = {}

        data[key] = result  # overwrite or add

        with open(output_json, 'w') as f:
            json.dump(data, f, indent=2)

    except Exception as e:
        return f"Error saving result: {e}"

    return f"Angle: {angle}° (saved as key: {key})"


# ==== RUN APP ====

if __name__ == '__main__':
    app.run(debug=True)
