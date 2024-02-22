import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm

from data_types import *
from Geometry3D import Point, Line, ConvexPolygon, intersection
from pandera.typing import Series
from pathlib import Path


def intersect_ray_dem_bruteforce(
        ray: Coord3DFloatPoints,
        header: Series[float],
        unsampled_alts: Coord1DFloatMesh,
) -> (Coord3DFloatSinglePoint | None, Coord3DFloatTile | None):
    """Finds the intersection point between a ray and a Digital Elevation Model.

    The intersection is found by sequentially searching for an intersection
    between every elementary triangle of the DEM and the ray. If multiple
    intersections are found, only the one that is closest to the ray's origin
    is kept.

    Parameters
    ----------
    ray : Coord3DFloatPoints
        The ray to intersect, defined as a half line. It is  described by two
        points: the ray's origin and another point belonging to the ray.
    header : Series[float]
        The header of the .asc file describing the DEM.
    unsampled_alts : Coord1DFloatMesh
        The file's altimetry table describing the DEM.

    Returns
    -------
    closest_inter_point : Coord3DFloatSinglePoint | None
        The intersection point found.
    closest_inter_tile : Coord3DFloatTile | None
        The elementary tile (generated from the DEM's point cloud) that contains
        the intersection point.
    """

    grid = generate_dem_meshgrid(header, unsampled_alts, 1)
    intersecting_tiles, closest_inter_tile = [], None
    intersection_points, closest_inter_point = [], None
    ray_line = Line(Point(ray[0]), Point(ray[1]))

    # Intersection search loop
    pbar = tqdm.tqdm(total=grid[:, :, 0].size)
    for i in range(grid.shape[0] - 1):
        for j in range(grid.shape[1] - 1):

            # Generate the two triangles forming the tile
            try:
                dem_tri1 = ConvexPolygon((Point(grid[i, j, :]), Point(grid[i+1, j, :]), Point(grid[i, j+1, :])))
                dem_tri2 = ConvexPolygon((Point(grid[i+1, j, :]), Point(grid[i, j+1, :]), Point(grid[i+1, j+1, :])))
            except ValueError as e:
                faulty_points = np.vstack((grid[i, j, :], grid[i+1, j, :], grid[i, j+1, :]))
                plot_faulty_points(faulty_points)
                raise e

            for dem_tri in [dem_tri1, dem_tri2]:
                inter = intersection(ray_line, dem_tri)
                if inter is not None:
                    intersection_points.append(np.array([inter.x, inter.y, inter.z]))
                    intersecting_tiles.append(grid[i: i+2, j: j+2, :])
                    # break
            pbar.update(1)
        #     if inter is not None:
        #         break
        # if inter is not None:
        #     break

    if len(intersection_points) > 0:
        closest_inter_point_idx = np.argmin(np.linalg.norm(np.vstack(intersection_points) - ray[0]))
        closest_inter_point = intersection_points[closest_inter_point_idx]
        closest_inter_tile = intersecting_tiles[closest_inter_point_idx]
    return closest_inter_point, closest_inter_tile


def generate_dem_meshgrid(header: Series[float], alts: Coord1DFloatMesh, sample_step: int) -> Coord3DFloatMesh:
    """Generates the 3D-coordinate meshgrid corresponding to a .asc file.

    Parameters
    ----------
    header : Series[float]
        The file's header information.
    alts : Coord1DFloatMesh
        The file's altimetry table.
    sample_step : int
        The sampling step, used to subsample the mapping before creating the
        mesh. For example, sample_step = 10 will sample 1 in 10 points.

    Returns
    -------
    grid : Coord3DFloatMesh
        The resulting meshgrid.
    """

    x_mesh = (header.xllcorner + np.arange(0, header.ncols) * header.cellsize)[::sample_step]
    y_mesh = np.flip(header.yllcorner + np.arange(0, header.nrows) * header.cellsize)[::sample_step]  # flip nécessaire pour avoir y croissants ascendants
    xx_mesh, yy_mesh = np.meshgrid(x_mesh, y_mesh)
    grid = np.dstack((xx_mesh, yy_mesh, alts[::sample_step, ::sample_step]))
    return grid


def generate_header_and_alts(tile_file: Path | str) -> (Series[float], Coord1DFloatMesh):
    """Reads the header and altimetry table from an IGN tile file (.asc).

    Parameters
    ----------
    tile_file : Path | str
        The path to the tile file to read.

    Returns
    -------
    header : Series[float]
        The file's header containing the following values: ncols, nrows,
        xllcorner, yllcorner, cellsize, NODATA_value.
    alts : Coord1DFloatMesh
        The altimetry table for the tile.
    """

    n_lines = 0
    header_lines = []
    with open(tile_file, "r") as f:
        while n_lines < 6:
            header_lines.append(f.readline().split())
            n_lines += 1
    header = pd.DataFrame(header_lines).set_index(0).squeeze().astype(float)
    alts = pd.read_csv(tile_file, sep=" ", skiprows=6, header=None).drop(0, axis=1).replace(header.NODATA_value, np.nan).values
    return header, alts


def plot_faulty_points(points: Coord3DFloatPoints) -> None:
    """Plots a triangular 3D surface based on input faulty points.

    Used to visualize a collection of points that caused a ValueError when
    trying to create a ConvexPolygon from them using Geometry3D.

    Parameters
    ----------
    points : Coord3DFloatPoints
        The collection of faulty points.

    Returns
    -------
    out : None
    """
    fig_points = plt.figure()
    ax_points = fig_points.add_subplot(111, projection="3d")
    ax_points.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax_points.plot_trisurf(points[:, 0], points[:, 1], points[:, 2])
    ax_points.set_title("Faulty points for convex polygon creation")
    plt.show()
