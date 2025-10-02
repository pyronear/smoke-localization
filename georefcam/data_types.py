import pandas as pd
from nptyping import Int, Float, NDArray, Shape

# Typages numpy
Coord1DFloatGrid = NDArray[Shape["* n_rows, * n_cols, [z]"], Float]
Coord2DIntPoints = NDArray[Shape["* n_points, [x, y]"], Int]
Coord3DFloatGrid = NDArray[Shape["* n_rows, * n_cols, [x, y, z]"], Float]
Coord3DFloatPoints = NDArray[Shape["* n_points, [x, y, z]"], Float]
ImageArrayGray = NDArray[Shape["* height, * width, 1 grayscale"], Int]
ImageArrayRGB = NDArray[Shape["* height, * width, 3 rgb"], Int]
RayCoord3DFloatPoints = NDArray[
    Shape["* n_points, [x, y, z], [origin, destination]"], Float
]

# Typage alias pour le DataFrame des rayons
DfRayInstance = pd.DataFrame

# Validation manuelle de la structure du DataFrame
expected_columns = [
    "ori_x",
    "ori_y",
    "ori_z",
    "dest_x",
    "dest_y",
    "dest_z",
    "inter_x",
    "inter_y",
    "inter_z",
    "n_tri",
    "dist_o",
]


def validate_df_ray(df: pd.DataFrame):
    missing = set(expected_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in df_ray: {missing}")

    # Optionnel : vérifie aussi les types (float sauf n_tri)
    float_cols = [col for col in expected_columns if col != "n_tri"]
    wrong_types = [
        col for col in float_cols if not pd.api.types.is_float_dtype(df[col])
    ]
    if "n_tri" in df.columns and not pd.api.types.is_integer_dtype(df["n_tri"]):
        wrong_types.append("n_tri")
    if wrong_types:
        raise TypeError(f"Incorrect column types in df_ray: {wrong_types}")
