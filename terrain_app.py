import streamlit as st
import numpy as np
from process_data import *
from spatial_interp import *
from plot import *

st.set_page_config(
  page_title='3D Terrain Generator',
  page_icon='🏔',
  layout='wide',
  initial_sidebar_state='auto'
)

# st.title('3D Terrain Generator')

with st.sidebar:
  st.image('3D-terrain-generator-logo.png', use_column_width=True)
  # Exapnder for generating terrain from data source
  exp1 = st.expander('Elevation Data Source', expanded=False)  
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
  exp2 = st.expander('Spatial Interpolation Settings')
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
  x, y, z = read_xyz_file(input_file, skip_rows, delimiter, decimal)
elif data_option == 'Latitude and longitude bounds' and lat_long_button == True:
  x, y, z = request_data_from_open_elevation(lat1, long1, lat2, long2)
  #lat_long_button = False
else:
  xy_lat_long = False
  x, y, z = read_sample_data('survey_data.csv')

x = [float(i) for i in x]
y = [float(i) for i in y]
#convertXYToDistance(x, y)

#if xy_lat_long:
#  pass
if interp == 'Nearest Neighbor (NN)':
  x_terrain, y_terrain, z_terrain, N = spatial_interp_NN(x, y, z, grid_size)
elif interp == 'Inverse Distance Weighting (IDW)':
  x_terrain, y_terrain, z_terrain, N = spatial_interp_IDW(x, y, z, grid_size)
elif interp == 'Triangulated Irregular Network (TIN)':
  x_terrain, y_terrain, z_terrain, N = spatial_interp_TIN(x, y, z, grid_size)

with st.sidebar:
  # Expander for generating profile
  exp3 = st.expander('Generate Vertical Profile')
  with exp3:
    st.markdown('*Define the coordinates of a cross-section line: (x1, y1) - (x2, y2)*')
    x1 = st.slider(
      label='x1',
      value=0.96*max(x),
      min_value=min(x),
      max_value=max(x),
      step=np.ceil((max(x) - min(x)) / 50)
    )
    y1 = st.slider(
      label='y1',
      value=1.03*min(y),
      min_value=min(y),
      max_value=max(y),
      step=np.ceil((max(y) - min(y)) / 50)
    )
    x2 = st.slider(
      label='x2',
      value=1.03*min(x),
      min_value=min(x),
      max_value=max(x),
      step=np.ceil((max(x) - min(x)) / 50)
    )
    y2 = st.slider(
      label='y2',
      value=0.98*max(y),
      min_value=min(y),
      max_value=max(y),
      step=np.ceil((max(y) - min(y)) / 50)
    )    
    # x1, x2 = st.slider(
    #   label='Start and end x-coordinates: (x1, x2)',
    #   value=(min(x), max(x)),
    #   min_value=min(x),
    #   max_value=max(x)
    # )
    # y1, y2 = st.slider(
    #   label='Start and end y-coordinates: (y1, y2)',
    #   value=(min(y), max(y)),
    #   min_value=min(y),
    #   max_value=max(y)
    # )
    show_plane = st.checkbox(
      label='Show/Hide Profile Plane',
      value=True
    )
  
  # Exapnder for 3D terrain visualization option inputs
  exp4 = st.expander('Terrain Visualization Options')
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



xp, yp, zp = eqn_of_profile_plane(x1, y1, x2, y2, z)
profile_x, profile_z = get_elevation_profile(x1, y1, x2, y2, x, y, z, N, interp)


fig1 = terrain_plot(x_terrain, y_terrain, z_terrain, z, N, xp, yp, zp,
                   color_scale, show_scale, show_contours,
                   project_contours, show_plane)
st.write(fig1)
#fig.write_image("fig1.svg")

fig2 = elevation_profile_plot(profile_x, profile_z)
st.write(fig2)

st.markdown('''
  Author: [Yared Bekele (PhD)](https://yaredwb.com) [GitHub](https://github.com/yaredwb) [LinkedIn](https://www.linkedin.com/in/yaredworku/) [Twitter](https://twitter.com/yaredwb)
''')