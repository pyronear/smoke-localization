import cameratransform as ct
import numpy as np
import pandas as pd
import pyvista as pv

from logger import logger
from data_types import *
from pandera.typing import Series
from pathlib import Path
from pyproj import CRS, Transformer


class GeoRefCam:

    def __init__(self, camera_model, dem):

        self.camera_model = camera_model
        self.dem = dem

    def project_points_from_cam_to_dem_crs(self, points: Coord3DFloatPoints) -> Coord3DFloatPoints:

        proj_points = project_points_to_crs(points, self.camera_model.crs, self.dem.crs)
        return proj_points

    def project_rays_from_cam_to_dem_crs(self, rays: RayCoord3DFloatPoints) -> RayCoord3DFloatPoints:

        origins, destinations = rays[:, :, 0], rays[:, :, 1]
        proj_raypoints = self.project_points_from_cam_to_dem_crs(np.vstack((origins, destinations)))
        proj_rays = np.dstack((proj_raypoints[:len(origins)], proj_raypoints[len(origins):]))
        return proj_rays

    def cast_rays(
            self,
            rays: RayCoord3DFloatPoints,
            check_crs: bool = True,
            min_d_intersection_m: float = 0,
            return_df_ray: bool = False,
    ) -> (Coord3DFloatPoints, NDArray[Shape["* n_rays"], Int], DfRayInstance | None):

        assert_isinstance(rays, RayCoord3DFloatPoints)
        if check_crs and self.camera_model.crs != self.dem.crs:
            rays = self.project_rays_from_cam_to_dem_crs(rays)

        origins, destinations = rays[:, :, 0], rays[:, :, 1]

        df_ray = pd.DataFrame(
            data=np.hstack((origins, destinations)),
            columns=["ori_x", "ori_y", "ori_z", "dest_x", "dest_y", "dest_z"]
        )
        df_ray["n_ray"] = np.arange(len(df_ray))

        logger.debug(f"df_ray:\n{df_ray}")

        inter_points, parent_rays, inter_triangles = self.dem.mesh.multi_ray_trace(
            df_ray[["ori_x", "ori_y", "ori_z"]],
            df_ray[["dest_x", "dest_y", "dest_z"]],
            retry=False,
            first_point=False
        )

        df_inter = pd.DataFrame(
            data=np.hstack((inter_points, parent_rays[:, np.newaxis], inter_triangles[:, np.newaxis])),
            columns=["inter_x", "inter_y", "inter_z", "n_ray", "n_tri"]
        )

        df_inter = df_inter.astype({"n_ray": int, "n_tri": "Int64"})  # the type assigned to 'n_tri' works as a nullable integer type
        logger.debug(f"df_inter pre-filtering:\n{df_inter}")
        df_inter[["ori_x", "ori_y", "ori_z"]] = df_ray.loc[df_inter.n_ray, ["ori_x", "ori_y", "ori_z"]].values
        df_inter["dist_o"] = np.linalg.norm(df_inter[["inter_x", "inter_y", "inter_z"]].values - df_inter[["ori_x", "ori_y", "ori_z"]].values, axis=1)
        df_inter = df_inter[df_inter["dist_o"] > min_d_intersection_m]
        logger.debug(f"df_inter post min_dist filtering:\n{df_inter}")

        gp_ray = df_inter.groupby("n_ray")
        df_inter = df_inter.loc[gp_ray.dist_o.idxmin()]
        logger.debug(f"df_inter post closest_point filtering:\n{df_inter}")
        df_ray = df_ray.merge(df_inter[["inter_x", "inter_y", "inter_z", "n_ray", "n_tri", "dist_o"]], on="n_ray", how="left")
        df_ray = df_ray.sort_values("n_ray").set_index("n_ray")
        logger.debug(f"df_ray final:\n{df_ray}")

        filtered_inter_points = df_ray[["inter_x", "inter_y", "inter_z"]].values
        filtered_inter_triangles = df_ray.n_tri.values
        if return_df_ray:
            return filtered_inter_points, filtered_inter_triangles, df_ray
        else:
            return filtered_inter_points, filtered_inter_triangles, None


class CTCameraModel:

    crs = "WGS84"

    def __init__(self, device_params: dict, ypr_orientation_params: dict, location_params: dict):

        self.device_params, self.ypr_orientation_params, self.location_params = device_params, ypr_orientation_params, location_params
        self.cam_model = ct.Camera(
            ct.RectilinearProjection(**device_params),
            ct.SpatialOrientationYawPitchRoll(**ypr_orientation_params)
        )
        self.cam_model.setGPSpos(location_params["lat"], location_params["lon"], location_params["alt"])

    def project_pixel_points_to_world_rays(self, pixels: Coord2DIntPoints) -> RayCoord3DFloatPoints:

        proj_pixels = self.cam_model.gpsFromImage(pixels)
        campoint_world = np.array([self.location_params["lat"], self.location_params["lon"], self.location_params["alt"]])
        campoint_world_broadcasted = np.broadcast_to(campoint_world, proj_pixels.shape)
        proj_rays = np.dstack((campoint_world_broadcasted, proj_pixels))

        return proj_rays


class ASCIIGridDEM:
    """

    Attributes
    ----------
    filepath : Path | str
    crs : str | None
    header : Series[float]
    alts : Coord1DFloatGrid
    pcd : Coord3DFloatGrid | None
    mesh : pv.PolyData | None
    """

    def __init__(self, filepath: Path | str, crs: str | CRS | None = None):

        self.filepath, self.crs = filepath, crs
        self.header, self.alts = self._build_header_and_alts_table()
        self.pcd, self.mesh = None, None

    def build_pcd(self, sample_step: int) -> None:
        """Builds the 3D-point could corresponding to an .asc file.

        Parameters
        ----------
        sample_step : int
            The sampling step, used to subsample the mapping before creating the
            grid. For example, sample_step = 10 will sample 1 in 10 points.
        """

        x_grid = (self.header.xllcorner + np.arange(0, self.header.ncols) * self.header.cellsize)[::sample_step]
        y_grid = np.flip(self.header.yllcorner + np.arange(0, self.header.nrows) * self.header.cellsize)[::sample_step]  # flip nécessaire pour avoir y croissants ascendants
        xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)
        self.pcd = np.dstack((xx_grid, yy_grid, self.alts[::sample_step, ::sample_step]))

    def build_mesh(self) -> None:

        pv_pcd = pv.StructuredGrid(*[self.pcd[:, :, i] for i in range(3)])
        self.mesh = pv_pcd.cast_to_poly_points().delaunay_2d()

    def _build_header_and_alts_table(self) -> (Series[float], Coord1DFloatGrid):
        """Reads the header and altimetry table from an IGN tile file (.asc).

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
        with open(self.filepath, "r") as f:
            while n_lines < 6:
                header_lines.append(f.readline().split())
                n_lines += 1
        header = pd.DataFrame(header_lines).set_index(0).squeeze().astype(float)
        alts = pd.read_csv(self.filepath, sep=" ", skiprows=6, header=None).drop(0, axis=1).replace(header.NODATA_value, np.nan).values
        return header, alts


def project_points_to_crs(points: Coord3DFloatPoints, from_crs: str, to_crs: str) -> Coord3DFloatPoints:

    transformer = Transformer.from_crs(from_crs, to_crs)
    if points.ndim == 2:
        xx, yy, zz = [points[:, i] for i in range(3)]
    else:
        xx, yy, zz = points
    pr_xx, pr_yy, pr_zz = transformer.transform(xx, yy, zz)
    proj_points = np.vstack((pr_xx, pr_yy, pr_zz)).T

    return proj_points
