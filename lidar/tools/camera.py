import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui

def get_focal_length_px(f, width, height, diag):
    '''Compute focal lengths in pixels from camera sensor specifications

    Args:
        f (float): focal length in mm
        width (int): image width in px
        height (int): image height in px
        diag (float): sensor diagonal, for example 1/2.49

    Returns:
        (float, float): focal lengths for x and y, in pixels
    '''
    diag *= 16 # in mm
    a = np.arctan(height/width)
    sensorWidth = diag * np.cos(a)
    sensorHeight = diag * np.sin(a)
    fx = f*width/sensorWidth
    fy = f*height/sensorHeight
    return fx, fy

def callbacks(parameters, view_point):
    '''Set camera actions and return callbacks dict

    Args:
        parameters (open3d.camera.PinholeCameraParameters): virtual camera parameters
        view_point (np.array): view point coordinates [x,y,z]

    Returns:
        dict: dict of keys to callback functions
    '''
    def rotate_view_right(vis):
        '''Camera rotation around local up axis, clockwise
        Args:
            vis (o3d.visualization.Visualizer): visualizer
        Returns:
            bool: False
        '''
        ctr = vis.get_view_control()
        ctr.camera_local_rotate(4.0, 0.0)
        return False

    def rotate_view_left(vis):
        '''Camera rotation around local up axis, counterclockwise
        Args:
            vis (o3d.visualization.Visualizer): visualizer
        Returns:
            bool: False
        '''
        ctr = vis.get_view_control()
        ctr.camera_local_rotate(-4.0, 0.0)
        return False

    def translate_view(vis):
        '''Camera translation towards local front axis
        Args:
            vis (o3d.visualization.Visualizer): visualizer
        Returns:
            bool: False
        '''
        ctr = vis.get_view_control()
        ctr.camera_local_translate(10,0,0)
        return False

    def correct_up(vis):
        '''Set the local up axis to be colinear to global z axis
        Args:
            vis (o3d.visualization.Visualizer): visualizer
        Returns:
            bool: False
        '''
        ctr = vis.get_view_control()
        ctr.set_up(np.array([0,0,1]))
        return False

    def set_view(vis):
        '''Set the camera position to view point
        Args:
            vis (o3d.visualization.Visualizer): visualizer
        Returns:
            bool: False
        '''
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(parameters, True)
        return False

    def look_at_view_point(vis):
        '''Set the local front axis to look at viewpoint
        Args:
            vis (o3d.visualization.Visualizer): visualizer
        Returns:
            bool: False
        '''
        ctr = vis.get_view_control()
        ctr.set_lookat(view_point)
        return False

    key_to_callback = {}
    key_to_callback[gui.KeyName.RIGHT] = rotate_view_right
    key_to_callback[gui.KeyName.LEFT] = rotate_view_left
    key_to_callback[ord("T")] = translate_view
    key_to_callback[ord("U")] = correct_up
    key_to_callback[ord("V")] = set_view
    key_to_callback[ord("L")] = look_at_view_point

    return key_to_callback

def save_skyline_with_terrain(pc, skyline_points, image_path):
    '''Render an orthographic projection (no perspective) of the terrain from above, and add the skyline. Save it as image

    Args:
        pc (open3d.geometry.PointCloud): the terrain point cloud
        skyline_points (np.array (360,3)): skyline cartesian coordinates
        image_path (str): path and name where to save the image 
    '''
    # connect each point with the next one (except last with first)
    line_indices = [[i, i + 1] for i in range(0, 360)]
    line_indices[-1][1] = 0
    # create open3d lines from skyline points
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(skyline_points)
    lines.lines = o3d.utility.Vector2iVector(line_indices)
    #colors = [[1.0, 0.2, 0.7] for _ in range(0, 360)]
    #lines.colors = o3d.utility.Vector3dVector(colors)

    # Generate a render
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=500, height=500, visible=False)
    vis.add_geometry(pc)
    vis.add_geometry(lines)
    # fov < 5 is rendered as orthographic projection
    vis.get_view_control().change_field_of_view(-60)
    vis.get_view_control().set_zoom(0.55)
    vis.poll_events()
    vis.update_renderer()
    # save the image
    vis.capture_screen_image(image_path)
    print('Image saved at',image_path)
    vis.destroy_window()