import numpy as np
from scipy import spatial


def slope_and_intercept(x1, y1, x2, y2):
    """
    Returns the slope and intercept of the line.
    Parameters:
      x1: the x coordinate of the first point
      y1: the y coordinate of the first point
      x2: the x coordinate of the second point
      y2: the y coordinate of the second point
    """
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m, b


def eqn_of_profile_plane(x1, y1, x2, y2, z):
    """
    Returns the equation of the plane that contains the profile.
    Parameters:
      x1: the x coordinate of the first point
      y1: the y coordinate of the first point
      x2: the x coordinate of the second point
      y2: the y coordinate of the second point
      z: the z data
    """
    m, b = slope_and_intercept(x1, y1, x2, y2)
    zp = np.linspace(min(z), max(z), 50)
    xp = np.linspace(x1, x2, 50)
    xp, zp = np.meshgrid(xp, zp)
    yp = m * xp + b
    return xp, yp, zp


def change_dist_to_km(d, R=6371):
    """
    Returns the distance in km.
    Parameters:
      d: the distance to be converted
      R: the radius of the earth
    """
    alpha = (d / (2 * R)) * np.pi / 180.0
    gamma = 2 * np.arcsin(alpha)
    d_converted = 2 * R * np.sin(gamma / 2)
    return d_converted


def convert_xy_to_dist(x, y):
    """
    Converts xy data to spatial distances.
    Parameters:
      x: the x data
      y: the y data    
    """
    xy = np.array([x, y]).T
    d = spatial.distance.cdist(xy, xy, 'euclidean')


def haversine(lat1, long1, lat2, long2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    Parameters:
      lat1: the latitude of the first point
      long1: the longitude of the first point
      lat2: the latitude of the second point
      long2: the longitude of the second point
    """
    rad = np.pi / 180
    R = 6371.0
    dlat = (lat2 - lat1) * rad
    dlon = (long1 - long1) * rad
    a = (np.sin(dlat / 2))**2 + np.cos(lat1 * rad) * \
        np.cos(lat2 * rad) * np.sin((dlon / 2))**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c
    return d
