import numpy as np
from pyproj import Transformer
import matplotlib.pyplot as plt

# project a point from latitude longitude to Lambert 93 coordinates
def to_lambert93(lat, lon, altitude):
    '''project a point from latitude longitude (ESPG:4326) to Lambert 93 coordinates (ESPG:2154)

    Args:
        lat (float): latitude
        lon (float): longitude
        altitude (float): altitude

    Returns:
        np.array: [x,y,z]
    '''
    projector = Transformer.from_crs("EPSG:4326", "EPSG:2154")
    point = np.array(projector.transform(lat,lon)+(altitude,))
    return point

def array_cartesian_to_spherical(points, view_point):
    '''Convert each cartesian points [x,y,z] of an array to spherical coordinates [r,theta,phi] (physics convention).

    Args:
        points (np.array (n,3)): list of points
        view_point (list or np.array (1,3)): viewpoint to consider as origin

    Return:
        np.array: array of spherical coordinates
    '''
    def cartesian_to_spherical(point):
        '''Convert a point [x,y,z] to spherical coordinates [r,theta,phi] (physics convention).
        The origin is the view_point.

        Args:
            point (list or np.array (1,3)): [x,y,z]

        Returns:
            list: [r,theta,phi]
        '''
        x0, y0, z0 = view_point
        x, y, z = point
        ydif = y-y0
        zdif = z-z0
        rho2 = np.square(x-x0)+np.square(ydif)
        r = np.sqrt(rho2+np.square(zdif))
        theta = np.arcsin(zdif/r)
        if x==x0 and y==y0:
            phi = 0
        elif x>=x0:
            phi = np.arcsin(ydif/np.sqrt(rho2))
        else: #x<x0
            phi = -np.arcsin(ydif/np.sqrt(rho2))+np.pi
        return [r, theta, phi]
    spherical = np.apply_along_axis(cartesian_to_spherical, 1, np.asarray(points))
    return spherical

def spherical_to_cartesian(point, replaceZ=False):
    '''Convert a point from spherical coordinates [r,theta,phi] to cartesian [x,y,z]

    Args:
        point (list or np.array (1,3)): the point in spherical coordinates (angles in radian)
        getZ (bool or int, optional): whether or not to compute the z dimension. Defaults to False.

    Returns:
        list: [x,y] or [x,y,z]
    '''
    r, theta, phi = point
    rho = r*np.cos(theta)
    x = rho*np.cos(phi)
    y = rho*np.sin(phi)
    if replaceZ:
        return [x,y,replaceZ]
    else:
        z = r*np.sin(theta)
        return [x, y, z]

def get_deg_angles(spherical, decimals=0):
    '''Get angles in degree from spherical coordinates. Rounds only phi.

    Args:
        spherical (np.array (3,n)): array of spherical coordinates [[r,theta,phi],...]
        decimals (int, optional): decimals to keep when rounding phi. Defaults to 0.

    Returns:
        np.array (2,n): array of angles in degree [[theta, phi],...]
    '''
    # convert angles to degree (drop r)
    deg = np.rad2deg(spherical[:,1:])
    # round phi
    deg[:,1] = np.round(deg[:,1], decimals)
    return deg

def delete_max(array):
    '''Get the maximum value of an array and remove it

    Args:
        array (np.array): the array to compute max from

    Returns:
        (int, np.array): max value, array without max
    '''
    if len(array)==1:
        return array[0], array
    imax = np.argmax(array)
    m = array[imax]
    new_array = np.delete(array, imax)
    return m, new_array

def get_skyline(angles, threshold=1e-4):
    '''Extract skyline from spherical coordinates points

    Args:
        angles (np.array (3,n)): array of spherical coordinates angles in degree [[theta,phi],...]
        threshold (float, optional): outliers distants by more than this value are removed. Defaults to 1e-4.

    Returns:
        np.array (360,): skyline, i.e. max elevation angle for each azimuth angle
    '''
    # sort 
    angles = angles[angles[:, 1].argsort()]
    phi_values = np.unique(angles[:, 1], return_index=True)
    # group by phi 
    grouped_by_phi = np.split(angles[:,0], phi_values[1][1:])

    # get max theta for each phi
    skyline = np.empty(shape=(360,))
    for i, thetas in enumerate(grouped_by_phi):
        # remove outliers (max values too far away from second max)
        m, t = delete_max(thetas)
        while m-np.max(t) > threshold:
            thetas = t
            m, t = delete_max(thetas)
        skyline[i%360] = np.max(thetas)
    return skyline

def plot_skyline(skyline, title):
    '''Plot the maximum elevation angle according to the azimuth angle

    Args:
        skyline (np.array (360,)): the skyline data (max theta for each phi)
        title (str): the plot title
    '''
    plt.style.use('default')
    plt.figure(figsize=(20,5))
    plt.plot(skyline, linewidth=3, color='sienna')
    font_params = {'size':15}
    plt.title(title, fontdict=font_params)
    plt.xlabel("Azimuth angle (°)", fontdict=font_params)
    plt.ylabel("Max elevation angle (°)", fontdict=font_params)
    plt.show()

def skyline_to_cartesian(spherical, angles, skyline, view_point, max_z):
    '''Transform the skyline to cartesian coordinates, so it can be diplayed with the terrain

    Args:
        spherical (np.array (n,3)): spherical coordinates of the terrain points
        angles (np.array (n,2)): angles only in degree of the terrain points
        skyline (np.array (360,)): max elevation angle for each azimuth
        view_point (list or np.array (1,3)): viewpoint considered as origin for spherical coordinates
        max_z (float): maximum altitude in the point cloud

    Returns:
        np.array (n,3): array of coordinates [x,y,z] for each point of the skyline
    '''
    skyline_points = np.empty(shape=(360,3))
    # for each angle pair
    for phi, theta in enumerate(skyline):
        # get the index corresponding to that angle combination (spherical and angles have the same order)
        i = np.argwhere((angles[:,0]==theta) & ((angles[:,1]+90)%360==phi))[0][0]
        # convert the point to cartesian coordinates
        cart_point = spherical_to_cartesian(spherical[i], replaceZ=max_z)
        # translate cartesian conversion to viewpoint
        skyline_points[phi] = np.add(cart_point, view_point)
    return skyline_points

