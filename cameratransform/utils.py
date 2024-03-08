import numpy as np
import pandas as pd

from data_types import *
from pandera.typing import Series
from pathlib import Path


def generate_dem_meshgrid(header: Series[float], alts: Coord1DFloatGrid, sample_step: int) -> Coord3DFloatGrid:
    """Generates the 3D-coordinate meshgrid corresponding to a .asc file.

    Parameters
    ----------
    header : Series[float]
        The file's header information.
    alts : Coord1DFloatGrid
        The file's altimetry table.
    sample_step : int
        The sampling step, used to subsample the mapping before creating the
        mesh. For example, sample_step = 10 will sample 1 in 10 points.

    Returns
    -------
    grid : Coord3DFloatGrid
        The resulting meshgrid.
    """

    x_mesh = (header.xllcorner + np.arange(0, header.ncols) * header.cellsize)[::sample_step]
    y_mesh = np.flip(header.yllcorner + np.arange(0, header.nrows) * header.cellsize)[::sample_step]  # flip nécessaire pour avoir y croissants ascendants
    xx_mesh, yy_mesh = np.meshgrid(x_mesh, y_mesh)
    grid = np.dstack((xx_mesh, yy_mesh, alts[::sample_step, ::sample_step]))
    return grid


def generate_header_and_alts(tile_file: Path | str) -> (Series[float], Coord1DFloatGrid):
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
    alts : Coord1DFloatGrid
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
