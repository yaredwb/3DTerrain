![](3D-terrain-generator-logo.png?raw=true "3D Terrain Generator")

This is a web app to generate a 3D terrain from different data sources. The app is deployed on Heroku and may be accessed here: [3D Terrain Generator](https://generate-3d-terrain.herokuapp.com/).

The types of elevation data sources supported at the moment are:

- Raw XYZ data files
- Elevation data from Open Elevation based on latitude and longitude bounds.

Spatial interpolation methods used include:

- Nearest Neighbor (NN)
- Inverse Distance Weighting (IDW)
- Triangulated Irregular Network (TIN)

Elevation profiles may be generated between any two points in the terrain. The generated elevation profile can be exported to a CSV file and downloaded.

There are some known issues in the app that will hopefully be fixed soon.