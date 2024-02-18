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

def array_to_spherical(pc, view_point):
    def point_to_spherical(point):
        '''Convert a point [x,y,z] to spherical coordinates [r,theta,phi] (physics convention).
        The origin is the view_point.

        Args:
            point (np.array): [x,y,z]

        Returns:
            np.array: [r,theta,phi]
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
    spherical = np.apply_along_axis(point_to_spherical, 1, np.asarray(pc.points))
    return spherical

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

def get_skyline(spherical, threshold=1e-4):
    '''Extract skyline from spherical coordinates points

    Args:
        spherical (np.array (3,n)): array of spherical coordinates [[r,theta,phi],...]
        threshold (float, optional): outliers distants by more than this value are removed. Defaults to 1e-4.

    Returns:
        np.array (360,): skyline, i.e. max elevation angle for each azimuth angle
    '''
    spherical = get_deg_angles(spherical)
    # sort 
    spherical = spherical[spherical[:, 1].argsort()]
    phi_values = np.unique(spherical[:, 1], return_index=True)
    # group by phi 
    grouped_by_phi = np.split(spherical[:,0], phi_values[1][1:])

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

def plot_skyline(skyline):
    plt.style.use('default')
    plt.figure(figsize=(20,5))
    plt.plot(skyline, linewidth=3, color='sienna')
    font_params = {'size':15}
    plt.xlabel("Azimuth angle (°)", fontdict=font_params)
    plt.ylabel("Max elevation angle (°)", fontdict=font_params)
    plt.show()