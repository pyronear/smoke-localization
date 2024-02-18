import numpy as np
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
