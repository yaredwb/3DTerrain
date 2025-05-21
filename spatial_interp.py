import numpy as np
from scipy import spatial
from helper_functions import *


def get_grid_size(grid_size):
    """
    Returns the numerical grid size based on the grid size description.
    Parameters:
      grid_size: the grid size description
    """
    if grid_size == 'Very coarse':
        N = 10
    elif grid_size == 'Coarse':
        N = 25
    elif grid_size == 'Medium':
        N = 50
    elif grid_size == 'Fine':
        N = 100
    else:
        N = 150

    return N


def spatial_KDTree(x, y, z):
    """
    Returns the spatial KDTree of the xyz data.
    Parameters:
      x: the x data
      y: the y data
      z: the z data
    """
    known_xy = np.c_[x, y]
    known_z = np.c_[z]
    tree_known_xy = spatial.KDTree(known_xy)

    return tree_known_xy, known_xy, known_z


def spatial_interp_NN(x, y, z, grid_size):
    """
    Returns the spatial interpolation of the xyz data using the nearest neighbor (NN) method.
    Parameters:
      x: the x data
      y: the y data
      z: the z data
      grid_size: the grid size description
    """
    N = get_grid_size(grid_size)
    tree_known_xy, known_xy, known_z = spatial_KDTree(x, y, z)
    # Create unknown grid of interpolation points
    x_min_grid = np.floor(min(x))
    y_min_grid = np.floor(min(y))
    x_max_grid = np.ceil(max(x))
    y_max_grid = np.ceil(max(y))

    x_lin = np.linspace(x_min_grid, x_max_grid, N)
    y_lin = np.linspace(y_min_grid, y_max_grid, N)

    x_terrain, y_terrain = np.meshgrid(x_lin, y_lin)
    unknown_xy = np.c_[x_terrain.ravel(), y_terrain.ravel()]
    unknown_z = []
    for xy in unknown_xy:
        _, idx = tree_known_xy.query(xy, 1)
        unknown_z.append(known_z[idx])

    z_terrain = np.array(unknown_z)
    z_terrain = z_terrain.reshape((N, N))

    return x_terrain, y_terrain, z_terrain, N


def spatial_interp_IDW(x, y, z, grid_size, n_sample=5, p=2):
    """
    Returns the spatial interpolation of the xyz data using the inverse distance weighting (IDW) method.
    Parameters:
      x: the x data
      y: the y data
      z: the z data
      grid_size: the grid size description
      n_sample: the number of samples to use for the interpolation
      p: the power of the inverse distance weighting
    """
    N = get_grid_size(grid_size)
    tree_known_xy, known_xy, known_z = spatial_KDTree(x, y, z)

    # Create unknown grid of interpolation points
    x_min_grid = np.floor(min(x))
    y_min_grid = np.floor(min(y))
    x_max_grid = np.ceil(max(x))
    y_max_grid = np.ceil(max(y))

    x_lin = np.linspace(x_min_grid, x_max_grid, N)
    y_lin = np.linspace(y_min_grid, y_max_grid, N)

    x_terrain, y_terrain = np.meshgrid(x_lin, y_lin)
    unknown_xy = np.c_[x_terrain.ravel(), y_terrain.ravel()]
    unknown_z = []
    for xy in unknown_xy:
        ds, ids = tree_known_xy.query(xy, n_sample)
        A = np.sum(known_z[ids].T / ds**p)
        B = np.sum(1 / ds**p)
        unknown_z.append(A / B)

    z_terrain = np.array(unknown_z)
    z_terrain = z_terrain.reshape((N, N))

    return x_terrain, y_terrain, z_terrain, N


def spatial_interp_TIN(x, y, z, grid_size):
    """
    Returns the spatial interpolation of the xyz data using the triangulated ittegular network (TIN) method.
    Parameters:
      x: the x data
      y: the y data
      z: the z data
      grid_size: the grid size description
    """
    N = get_grid_size(grid_size)
    known_xy = np.c_[x, y]
    known_z = np.c_[z]

    # Create unknown grid of interpolation points
    x_min_grid = np.floor(min(x))
    y_min_grid = np.floor(min(y))
    x_max_grid = np.ceil(max(x))
    y_max_grid = np.ceil(max(y))

    x_lin = np.linspace(x_min_grid, x_max_grid, N)
    y_lin = np.linspace(y_min_grid, y_max_grid, N)

    x_terrain, y_terrain = np.meshgrid(x_lin, y_lin)
    unknown_xy = np.c_[x_terrain.ravel(), y_terrain.ravel()]

    # Triangulate the known data points
    tri = spatial.Delaunay(known_xy)
    # Find which triangle each unknown point belongs to
    tri_loc_xy = tri.find_simplex(unknown_xy)
    # Indices of vertices of triangle for each unknown point
    indices = tri.simplices[tri_loc_xy]

    # Barycentric coordinates of each point within its triangle
    # ----------------------------------------------------------
    # Affine transformation for triangle containing each unknown point
    X = tri.transform[tri_loc_xy, :2]
    # Offset of each unknown point from the origin of its containing triangle
    Y = unknown_xy - tri.transform[tri_loc_xy, 2]
    b = np.einsum('ijk,ik->ij', X, Y)
    bcoords = np.c_[b, 1 - b.sum(axis=1)]
    # Elevations at the vertices of each triangle
    tri_zs = known_z[indices]

    # Interpolate elevations at unknown points
    unknown_z = []
    for i in range(bcoords.shape[0]):
        # Use np.nan values for points outside triangulation
        if tri_loc_xy[i] == -1:
            unknown_z.append(np.nan)
        else:
            zc = np.dot(bcoords[i], tri_zs[i])
            unknown_z.append(zc[0])

    z_terrain = np.array(unknown_z)
    z_terrain = z_terrain.reshape((N, N))

    return x_terrain, y_terrain, z_terrain, N


def get_elevation_profile(x1, y1, x2, y2, x, y, z, N, method, n_sample=5, p=2):
    """
    Returns the elevation profile between two points.
    Parameters:
      x1: the x coordinate of the start point
      y1: the y coordinate of the start point
      x2: the x coordinate of the end point
      y2: the y coordinate of the end point
      x: the x data
      y: the y data
      z: the z data
      N: the grid size description
      method: the method to use for the interpolation
      n_sample: the number of samples to use for the interpolation
      p: the power of the inverse distance weighting
    """
    m, b = slope_and_intercept(x1, y1, x2, y2)
    tree_known_xy, _, known_z = spatial_KDTree(x, y, z)
    profile_x = np.linspace(x1, x2, N)
    profile_y = m * profile_x + b
    profile_xy = np.c_[profile_x.ravel(), profile_y.ravel()]
    profile_z = []
    for xy in profile_xy:
        if method == 'Nearest Neighbor (NN)':
            _, idx = tree_known_xy.query(xy, 1)
            profile_z.append(known_z[idx])
        elif method == 'Inverse Distance Weighting (IDW)':
            ds, ids = tree_known_xy.query(xy, n_sample)
            A = np.sum(known_z[ids].T / ds**p)
            B = np.sum(1 / ds**p)
            profile_z.append(A / B)
        elif method == 'Triangulated Irregular Network (TIN)':
            # Create Delaunay triangulation
            tri = spatial.Delaunay(np.c_[x, y])
            # Find which triangle each profile point belongs to
            tri_loc_prof = tri.find_simplex(xy)
            if tri_loc_prof == -1:
                profile_z.append(np.nan)
            else:
                # Barycentric coordinates
                X_prof = tri.transform[tri_loc_prof, :2]
                Y_prof = xy - tri.transform[tri_loc_prof, 2]
                b_prof = np.einsum('ij,i->j', X_prof, Y_prof)
                bcoords_prof = np.r_[b_prof, 1 - b_prof.sum()]
                # Elevations at the vertices of the triangle
                tri_zs_prof = known_z[tri.simplices[tri_loc_prof]]
                # Interpolate elevation
                profile_z.append(np.dot(bcoords_prof, tri_zs_prof)[0])

    profile_d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    profile_x = np.linspace(0, profile_d, N)
    profile_z = np.array(profile_z).reshape((N,))

    return profile_x, profile_z
