from georefcam.camera_model import CTCameraModel
import pyvista as pv
from georefcam.georefcam import GeoRefCam, get_direction_vector
import numpy as np


def get_dem_proj(dem, test_img, cam_info, azimuth, fov_x, fov_y):
    plotter = pv.Plotter(off_screen=True, window_size=test_img.size)

    # Create the mesh once
    plot_pv_meshgrid = pv.StructuredGrid(*[dem.pcd[:, :, i] for i in range(3)])
    plot_pv_meshgrid["alt"] = dem.pcd[:, :, 2].ravel(order="F")
    plotter.add_mesh(plot_pv_meshgrid, lighting=False)
    plotter.remove_scalar_bar()
    cam_model = CTCameraModel(
        test_img,
        fov_x,
        fov_y,
        azimuth,
        cam_info.pitch,
        cam_info.roll,
        cam_info.lat,
        cam_info.lon,
        cam_info.elevation,
        crs="EPSG:2154",
    )

    geocam = GeoRefCam(cam_model, dem)
    cam_dirvec = get_direction_vector(
        geocam.camera_model.yaw_deg, geocam.camera_model.pitch_deg
    )

    if geocam.camera_model.cam_loc[3] != geocam.dem.crs:
        cam_loc = geocam.project_points_from_cam_to_dem_crs(
            np.array([geocam.camera_model.cam_loc[:3]])
        )[0]
    else:
        cam_loc = geocam.camera_model.cam_loc[:3]

    # Update camera
    pv_camera = pv.Camera()
    pv_camera.position = cam_loc
    pv_camera.focal_point = cam_loc + cam_dirvec
    pv_camera.view_angle = geocam.camera_model.view_y_deg
    pv_camera.clipping_range = (30, 1e5)
    pv_camera.up = (0, 0, 1)
    plotter.camera = pv_camera

    # Render & get depth
    plotter.screenshot()
    dem_depth = -1 * plotter.get_image_depth()
    return np.nan_to_num(dem_depth, nan=1e6)
