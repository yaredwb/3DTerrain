import plotly.graph_objects as go


def terrain_plot(x_terrain, y_terrain, z_terrain, z, N, xp, yp, zp,
                 color_scale='Viridis', show_scale=True,
                 show_contours=False, project_contours=False,
                 show_plane=True):
    """
    Returns a 3D plot of the terrain.
    Parameters:
      x_terrain: the x data of the terrain
      y_terrain: the y data of the terrain
      z_terrain: the z data of the terrain
      z: the original z data
      N: the grid size
      xp: the x data of the profile plane
      yp: the y data of the profile plane
      zp: the z data of the profile plane
      color_scale: the color scale
      show_scale: whether to show the elevation scale
      show_contours: whether to show the contours
      project_contours: whether to project the contours on XY plane
      show_plane: whether to show the profile plane    
    """
    # Create the figure
    fig = go.Figure(
        data=[
            go.Surface(
                z=z_terrain,
                x=x_terrain,
                y=y_terrain,
                name='Terrain',
                colorscale=color_scale,
                showscale=show_scale,
                # connectgaps=True,
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
            eye=dict(x=2, y=1, z=1)
        ),
        scene=dict(
            aspectratio=dict(x=2, y=2, z=0.5),
            xaxis=dict(
                showgrid=False,
                showticklabels=False,
                showaxeslabels=False,
                showbackground=False
            ),  # dict(range=[x_min_grid,x_max_grid]),
            yaxis=dict(
                showgrid=False,
                showticklabels=False,
                showaxeslabels=False,
                showbackground=False
            ),  # dict(range=[y_min_grid,y_max_grid])
            zaxis=dict(
                showgrid=False,
                showticklabels=False,
                showaxeslabels=False
                #range=[15, 20]
                # backgroundcolor='lightgrey'
            )
        )
    )

    return fig


def elevation_profile_plot(profile_x, profile_z):
    """
    Returns a 2D plot of the elevation profile.
    Parameters:
      profile_x: the x data of the elevation profile
      profile_z: the z data of the elevation profile
    """
    # Create the figure
    fig = go.Figure(
        data=[
            go.Scatter(
                x=profile_x,
                y=profile_z,
                name='Elevation Profile',
                mode='lines',
                line=dict(color='black', width=3),
                fill='tonexty',
                fillcolor='rgba(0,0,0,0.2)'
            ),
            go.Scatter(
                x=profile_x,
                y=[0.999*min(profile_z)] * len(profile_x),
                mode=None
            )
        ]
    )
    fig.update_layout(
        autosize=False,
        width=1400,
        height=400,
        paper_bgcolor='rgb(243,243,243)',
        yaxis_range=[0.999*min(profile_z), 1.001*max(profile_z)],
        showlegend=False
    )

    return fig
