from nptyping import NDArray, Shape, Float


Coord3DFloatSinglePoint = NDArray[Shape["[x, y, z]"], Float]
Coord3DFloatPoints = NDArray[Shape["* n_points, [x, y, z]"], Float]
Coord3DFloatTile = NDArray[Shape["[tl, tr, bl, br], [x, y, z]"], Float]
Coord1DFloatMesh = NDArray[Shape["* n_rows, * n_cols, [z]"], Float]
Coord3DFloatMesh = NDArray[Shape["* n_rows, * n_cols, [x, y, z]"], Float]
