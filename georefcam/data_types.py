import pandera as pa
from pandera import Column, DataFrameSchema, Index

from nptyping import Int, Float, NDArray, Shape

Coord1DFloatGrid = NDArray[Shape["* n_rows, * n_cols, [z]"], Float]
Coord2DIntPoints = NDArray[Shape["* n_points, [x, y]"], Int]
Coord3DFloatGrid = NDArray[Shape["* n_rows, * n_cols, [x, y, z]"], Float]
Coord3DFloatPoints = NDArray[Shape["* n_points, [x, y, z]"], Float]
ImageArrayGray = NDArray[Shape["* height, * width, 1 grayscale"], Int]
ImageArrayRGB = NDArray[Shape["* height, * width, 3 rgb"], Int]
RayCoord3DFloatPoints = NDArray[Shape["* n_points, [x, y, z], [origin, destination]"], Float]

# Schema Pandera sans SchemaModel
DfRaySchema = DataFrameSchema(
    columns={
        "ori_x": Column(float),
        "ori_y": Column(float),
        "ori_z": Column(float),
        "dest_x": Column(float),
        "dest_y": Column(float),
        "dest_z": Column(float),
        "inter_x": Column(float),
        "inter_y": Column(float),
        "inter_z": Column(float),
        "n_tri": Column(int, nullable=True),
        "dist_o": Column(float),
    },
    index=Index(int, name="n_ray"),
)

DfRayInstance = pa.typing.DataFrame[DfRaySchema]
