import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import spatial
from haversine import haversine_vector, Unit
import urllib.request
import json

@st.cache(suppress_st_warning=True)
def readSampleData(sample_file):
  # f = open(sample_file, 'r')
  # lines = f.readlines()
  # f.close()

  # x, y, z = [], [], []  
  # for line in lines:
  #   data = line.split(',')    
  #   x.append(float(data[0]))
  #   y.append(float(data[1]))
  #   z.append(float(data[2]))
  
  df = pd.read_csv('survey_data.csv', delimiter=',', header=None)  
  x = [float(i) for i in df[0].values.tolist()]
  y = [float(i) for i in df[1].values.tolist()]
  z = [float(i) for i in df[2].values.tolist()]

  return x, y, z

@st.cache
def readXYZFile(input_file, skip_rows, delimiter, decimal):
  if delimiter == 'Space':
    delim = None
  elif delimiter == 'Comma':
    delim = ','
  elif delimiter == 'Semicolon':
    delim = ';'
  else:
    delim = '\t'

  if input_file == None:
    f = open('survey_data.csv', 'r')
    delim = ','
  else:
    f = open(input_file.name, 'r')  
  lines = f.readlines()[skip_rows:]
  f.close()

  x, y, z = [], [], []  
  for line in lines:
    data = line.split(delim)
    if decimal == 'Comma':
      data = [x.replace(',','.') for x in data]
    x.append(float(data[0]))
    y.append(float(data[1]))
    z.append(float(data[2]))

  return x, y, z

@st.cache(suppress_st_warning=True)
def requestDataFromOpenElevation(lat1, long1, lat2, long2):
  x = np.random.uniform(long1, long2, 100)
  y = np.random.uniform(lat1, lat2, 100)

  xy = [{}] * len(x)
  for i in range(len(x)):
    xy[i] = {
      'latitude': y[i],
      'longitude': x[i]
    }

  locations = {'locations': xy}
  json_locs = json.dumps(locations, skipkeys=int).encode('utf8')

  number_of_tries = 3
  count = 0
  while count <= number_of_tries:
    try:
      url = "https://api.open-elevation.com/api/v1/lookup"
      response = urllib.request.Request(url, json_locs, headers={'Content-Type': 'application/json'})
      response_file = urllib.request.urlopen(response)
    except (OSError, urllib.error.HTTPError):
      count += 1
      continue
    break

  # Process response
  data = response_file.read()
  data_decoded = data.decode('utf8')
  json_data = json.loads(data_decoded)
  response_file.close()

  z = []
  results = json_data['results']
  for i in range(len(results)):
    z.append(float(results[i]['elevation']))

  x = x.tolist()
  y = y.tolist()
  
  return x, y, z

@st.cache(suppress_st_warning=True)
def changeDistTokm(d, R=6371):
  alpha = (d / (2 * R)) * np.pi / 180.0
  gamma = 2 * np.arcsin(alpha)
  d_converted = 2 * R * np.sin(gamma / 2)
  return d_converted

#@st.cache(suppress_st_warning=True)
def convertXYToDistance(x, y):
  xy = np.array([x, y]).T
  st.write(xy.shape)
  d = spatial.distance.cdist(xy, xy, 'euclidean')
  st.write(d)


@st.cache(suppress_st_warning=True)
def getGridSize(grid_size):
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

@st.cache(suppress_st_warning=True)
def spatialKDTree(x, y, z):
  known_xy = np.c_[x, y]
  known_z  = np.c_[z]
  tree_known_xy = spatial.KDTree(known_xy)

  return tree_known_xy, known_xy, known_z 

def haversine(lat1, long1, lat2, long2):
  rad = np.pi / 180
  R = 6371.0
  dlat = (lat2 - lat1) * rad
  dlon = (long1 - long1) * rad
  a = (np.sin(dlat / 2))**2 + np.cos(lat1 * rad) * np.cos(lat2 *rad) * np.sin((dlon / 2))**2
  c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
  d = R * c
  return d

@st.cache
def spatialInterpolationNN(x, y, z, grid_size):
  N = getGridSize(grid_size)
  tree_known_xy, known_xy, known_z = spatialKDTree(x, y, z)
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
  


@st.cache(suppress_st_warning=True)
def spatialInterpolationIDW(x, y, z, grid_size, n_sample=5, p=2):   
  N = getGridSize(grid_size)  
  tree_known_xy, known_xy, known_z = spatialKDTree(x, y, z)

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
    #ds = haversine_vector(xy, known_xy[ids], comb=True)
    # ds = []
    # for id in ids:
    #   d = haversine(xy[1], xy[0], known_xy[id][1], known_xy[id][0])
    #   ds.append(d)
    # ds = np.array(ds)        
    A = np.sum(known_z[ids].T / ds**p)  
    B = np.sum(1 / ds**p)
    unknown_z.append(A / B)

  z_terrain = np.array(unknown_z)
  z_terrain = z_terrain.reshape((N, N))

  return x_terrain, y_terrain, z_terrain, N

@st.cache(suppress_st_warning=True)
def spatialInterpolationTIN(x, y, z, grid_size):
  N = getGridSize(grid_size)
  known_xy = np.c_[x, y]
  known_z  = np.c_[z]

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
  #----------------------------------------------------------
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


@st.cache(suppress_st_warning=True)
def slopeAndYInterceptOfLine(x0, y0, x1, y1):
  m = (y1 - y0) / (x1 - x0)
  b = y0 - m * x0 
  return m, b


@st.cache(suppress_st_warning=True)
def eqnOfProfilePlane(x0, y0, x1, y1, x, y, z):
  m, b = slopeAndYInterceptOfLine(x0, y0, x1, y1)
  zp = np.linspace(min(z), max(z), 50)
  xp = np.linspace(x0, x1, 50)
  xp, zp = np.meshgrid(xp, zp)
  yp = m * xp + b
  return xp, yp, zp


@st.cache(suppress_st_warning=True)
def getElevationProfile(x0, y0, x1, y1, x, y, z, N, interp, n_sample=5, p=2): 
  m, b = slopeAndYInterceptOfLine(x0, y0, x1, y1)
  tree_known_xy, _, known_z = spatialKDTree(x, y, z)
  profile_x = np.linspace(x0, x1, N)
  profile_y = m * profile_x + b
  profile_xy = np.c_[profile_x.ravel(), profile_y.ravel()]
  profile_z = []
  for xy in profile_xy:        
    if interp == 'Nearest Neighbor (NN)':
      _, idx = tree_known_xy.query(xy, 1)
      profile_z.append(known_z[idx])
    elif interp == 'Inverse Distance Weighting (IDW)':
      ds, ids = tree_known_xy.query(xy, n_sample)
      A = np.sum(known_z[ids].T / ds**p)  
      B = np.sum(1 / ds**p)
      profile_z.append(A / B)
    elif interp == 'Triangulated Irregular Network (TIN)':
      _, idx = tree_known_xy.query(xy, 1)
      profile_z.append(known_z[idx])

  profile_d = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
  profile_x = np.linspace(0, profile_d, N)
  profile_z = np.array(profile_z).reshape((N,))

  return profile_x, profile_z

st.set_page_config(
  page_title='3D Terrain Generator',
  page_icon='🏔',
  layout='wide',
  initial_sidebar_state='auto'
)

st.title('3D Terrain Generator')


with st.sidebar:
  # Exapnder for generating terrain from XZY data file
  exp1 = st.beta_expander('Elevation Data Source', expanded=True)  
  with exp1:
    data_option = st.selectbox(
      label='Select terrain data source',
      options=[
        'Sample data',
        'Raw XYZ data file',
        'Latitude and longitude bounds'
      ],
      index=0
    )
    if data_option == 'Raw XYZ data file':
      input_file = st.file_uploader('Upload file with XYZ data (*.txt, *.csv, *.xyz etc)')         
      skip_rows = st.number_input(
        label='Number of rows to skip',
        value=0,
        min_value=0,
        max_value=20,
        step=1
      )
      delimiter = st.selectbox(
        label='Data delimiter',
        options=['Space', 'Comma', 'Semicolon', 'Tab']
      )
      decimal = st.selectbox(
        label='Decimal separator',
        options=['Dot', 'Comma']
      )
      xy_lat_long = st.checkbox(
        label='X and Y are geographic coordinates',
        value=False
      )
      if xy_lat_long:
        xy_coordds = st.selectbox(
          label='(X, Y) corresponds to',
          options=[
            '(Longitude, Latitude)',
            '(Latitude, Longitude)'            
          ]
        )
      st.markdown('Example XYZ data source [here.](https://topex.ucsd.edu/cgi-bin/get_data.cgi)')
    elif data_option == 'Latitude and longitude bounds':      
      lat1 = st.number_input(
        label='Latitude 1',
        value=8.905401,
        min_value=-85.000000,
        max_value=85.000000,
        step=0.01,
        format='%.6f'
      )
      lat2 = st.number_input(
        label='Latitude 2',
        value=9.005401,
        min_value=-85.000000,
        max_value=85.000000,
        step=0.01,
        format='%.6f'
      )
      long1 = st.number_input(
        label='Longitude 1',
        value=38.663611,
        min_value=-180.0,
        max_value=180.0,
        step=0.01,
        format='%.6f'
      )
      long2 = st.number_input(
        label='Longitude 2',
        value=38.763611,
        min_value=-180.000000,
        max_value=180.000000,
        step=0.01,
        format='%.6f'
      )       
      if lat2 - lat1 > 1.0 or long2 - long1 > 1.0:
        st.markdown('**The area requested is too large!**')
        lat_long_button = False
      else:
        lat_long_button = st.button(
          label='Get Data and Generate Terrain'
        )
      st.markdown('''
        *Note: Elevation data request from [Open Elevation](https://open-elevation.com/) may take too long, may not always succeed or may not return good resolution data depending on the location.*
      ''')
      
      
    
  # Expander for generating terrain from latitude and longitude bounds
  exp2 = st.beta_expander('Spatial Interpolation Settings', expanded=True)
  with exp2:
    interp = st.selectbox(
      label='Spatial interpolation method',
      options=[
        'Nearest Neighbor (NN)',
        'Inverse Distance Weighting (IDW)',
        'Triangulated Irregular Network (TIN)'
      ],
      index=1
    )
    grid_size = st.selectbox(
      label='Interpolation grid size',
      options=[
        'Very coarse',
        'Coarse',
        'Medium',
        'Fine',
        'Very fine'
      ],
      index=2
    )
    if interp == 'Inverse Distance Weighting (IDW)':
      st.write('A power value of 2 and 5 closest points are used for IDW.')
    if interp == 'Triangulated Irregular Network (TIN)':
      st.write('Linear interpolation based on Delaunay triangulation.')
      # n_sample = st.number_input(
      #   label='Number of closest points',
      #   min_value=3,
      #   max_value=10,
      #   step=1,
      #   value=5
      # )
      # p = st.number_input(
      #   label='Power value',
      #   min_value=2,
      #   max_value=5,
      #   step=1,
      #   value=2
      # )    

if data_option == 'Raw XYZ data file':
  x, y, z = readXYZFile(input_file, skip_rows, delimiter, decimal)
elif data_option == 'Latitude and longitude bounds' and lat_long_button == True:
  x, y, z = requestDataFromOpenElevation(lat1, long1, lat2, long2)
  #lat_long_button = False
else:
  xy_lat_long = False
  x, y, z = readSampleData('survey_data.csv')

x = [float(i) for i in x]
y = [float(i) for i in y]
#convertXYToDistance(x, y)

#if xy_lat_long:
#  pass
if interp == 'Nearest Neighbor (NN)':
  x_terrain, y_terrain, z_terrain, N = spatialInterpolationNN(x, y, z, grid_size)
elif interp == 'Inverse Distance Weighting (IDW)':
  x_terrain, y_terrain, z_terrain, N = spatialInterpolationIDW(x, y, z, grid_size)
elif interp == 'Triangulated Irregular Network (TIN)':
  x_terrain, y_terrain, z_terrain, N = spatialInterpolationTIN(x, y, z, grid_size)

with st.sidebar:
  # Expander for generating profile
  exp3 = st.beta_expander('Generate Vertical Profile', expanded=False)
  with exp3:
    st.markdown('*Define the coordinates of a cross-section line: (x0, y0) - (x1, y1)*')
    x0 = st.slider(
      label='x0',
      value=max(x),
      min_value=min(x),
      max_value=max(x),
      step=np.ceil((max(x) - min(x)) / 50)
    )
    y0 = st.slider(
      label='y0',
      value=min(y),
      min_value=min(y),
      max_value=max(y),
      step=np.ceil((max(y) - min(y)) / 50)
    )
    x1 = st.slider(
      label='x1',
      value=min(x),
      min_value=min(x),
      max_value=max(x),
      step=np.ceil((max(x) - min(x)) / 50)
    )
    y1 = st.slider(
      label='y1',
      value=max(y),
      min_value=min(y),
      max_value=max(y),
      step=np.ceil((max(y) - min(y)) / 50)
    )    
    # x0, x1 = st.slider(
    #   label='Start and end x-coordinates: (x0, x1)',
    #   value=(min(x), max(x)),
    #   min_value=min(x),
    #   max_value=max(x)
    # )
    # y0, y1 = st.slider(
    #   label='Start and end y-coordinates: (y0, y1)',
    #   value=(min(y), max(y)),
    #   min_value=min(y),
    #   max_value=max(y)
    # )
    show_plane = st.checkbox(
      label='Show/Hide Profile Plane',
      value=False
    )
  
  # Exapnder for 3D terrain visualization option inputs
  exp4 = st.beta_expander('Terrain Visualization Options', expanded=True)
  with exp4:
    color_scale = st.selectbox(
      label='Color scale',
      options=['Viridis', 'Jet', 'Burg', 'Sunset']
    )
    show_scale = st.checkbox(
      label='Show Elevation Scale',
      value=True
    )
    show_contours = st.checkbox(
      label='Show Contours',
      value=False
    )
    project_contours = st.checkbox(
      label='Project Contours on XY Plane',
      value=False
    )        



xp, yp, zp = eqnOfProfilePlane(x0, y0, x1, y1, x, y, z)
profile_x, profile_z = getElevationProfile(x0, y0, x1, y1, x, y, z, N, interp)

fig = go.Figure(
  data=[
    go.Surface(
      z=z_terrain,
      x=x_terrain,
      y=y_terrain,
      name='Terrain',
      colorscale=color_scale,
      showscale=show_scale,
      #connectgaps=True,
      contours_z=dict(
        show=show_contours,
        start=min(z),
        end=max(z),
        size=(max(z) - min(z)) / N,
        usecolormap=True,
        highlightcolor="limegreen",
        project_z=project_contours
      )      
    ),
    go.Surface(
      z=zp,
      x=xp,
      y=yp,
      visible=show_plane,
      name='Plane',
      colorscale='Greys',
      showscale=False,
      opacity=0.3
    )
  ]
)
fig.update_layout(
  autosize=False,  
  width=1400,
  height=800,
  paper_bgcolor='rgb(243,243,243)',
  scene_camera=dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=-0.4),
    eye=dict(x=2, y=2, z=1)
  ),
  scene=dict(
    aspectratio=dict(x=2, y=2, z=0.5),
    xaxis = dict(
      showgrid=False,      
      showticklabels=False,
      showaxeslabels=False,
      showbackground=False
    ), #dict(range=[x_min_grid,x_max_grid]),
    yaxis = dict(
      showgrid=False,
      showticklabels=False,
      showaxeslabels=False,
      showbackground=False
    ), #dict(range=[y_min_grid,y_max_grid])
    zaxis = dict(
      showgrid=False,
      showticklabels=False,
      showaxeslabels=False
      #range=[15, 20]
      #backgroundcolor='lightgrey'
    )
  )  
)
#st.plotly_chart(fig, sharing='streamlit')
st.write(fig)
#fig.write_image("fig1.svg")



fig2 = go.Figure(data=go.Scatter(x=profile_x, y=profile_z, mode='lines'))
fig2.update_layout(
  width=1400,
  height=400
)
st.write(fig2)

# with st.beta_container():
#   col1, col2 = st.beta_columns((1, 2))
#   with col1:    
#     st.write('Profile')  
#   with col2:
#     st.write(fig2)


st.sidebar.markdown('''
  Author: [Yared Bekele (PhD)](https://yaredwb.com) [GitHub](https://github.com/yaredwb) [LinkedIn](https://www.linkedin.com/in/yaredworku/) [Twitter](https://twitter.com/yaredwb)
''')