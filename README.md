![](3D-terrain-generator-logo.png?raw=true "3D Terrain Generator")

This web app generates interactive 3D terrain visualizations from different data sources. The app is deployed on Streamlit Cloud and may be accessed here: [3D Terrain Generator](https://3d-terrain-generator.streamlit.app/).

The types of elevation data sources supported at the moment are:

- Raw XYZ data files: These should be text files (`.txt`, `.csv`, `.xyz`) where each row contains X, Y, and Z coordinates (in that order), separated by a delimiter (space, comma, semicolon, or tab). The application allows you to specify the delimiter and the decimal separator (dot or comma) during upload.
- Elevation data from Open Elevation based on latitude and longitude bounds.

Spatial interpolation methods used include:

- Nearest Neighbor (NN)
- Inverse Distance Weighting (IDW)
- Triangulated Irregular Network (TIN)

Elevation profiles may be generated between any two points in the terrain. The generated elevation profile can be exported to a CSV file and downloaded.

## Project Structure

- `terrain_app.py`: The main Streamlit application file. Handles UI, user inputs, and orchestrates data processing and plotting.
- `process_data.py`: Contains functions for reading and processing input data from various sources.
- `spatial_interp.py`: Implements spatial interpolation algorithms (NN, IDW, TIN) and elevation profile calculations.
- `plot.py`: Generates 3D terrain plots and 2D elevation profiles using Plotly.
- `helper_functions.py`: Provides utility functions for calculations.
- `requirements.txt`: Lists the Python dependencies for the project.
- `survey_data.csv`: Default sample XYZ data.

## How to Use

1.  **Access the App:** Open the [3D Terrain Generator](https://3d-terrain-generator.streamlit.app/) in your browser.
2.  **Select Data Source (Sidebar):**
    *   **Sample data:** Uses the built-in `survey_data.csv`.
    *   **Raw XYZ data file:** Upload your own file. Specify the number of rows to skip (if any), the data delimiter, and the decimal separator.
    *   **Latitude and longitude bounds:** Enter the coordinates for the desired area to fetch data from Open Elevation API.
3.  **Configure Spatial Interpolation (Sidebar):**
    *   Choose an interpolation method: Nearest Neighbor (NN), Inverse Distance Weighting (IDW), or Triangulated Irregular Network (TIN).
    *   Select the desired grid size for interpolation.
4.  **View Terrain:** The 3D terrain plot will update automatically based on your selections.
5.  **Generate Elevation Profile (Sidebar):**
    *   Adjust the sliders to define the start (x1, y1) and end (x2, y2) points of a cross-section line on the terrain.
    *   The 2D elevation profile plot will be displayed below the 3D terrain.
    *   You can toggle the visibility of the profile plane on the 3D plot.
6.  **Customize Visualization (Sidebar):**
    *   Change the color scale, toggle the elevation scale, and show/hide contours on the 3D plot.
7.  **Download Profile (Sidebar):**
    *   Export the generated elevation profile data as a CSV file.

## Dependencies

The project requires Python and the libraries listed in `requirements.txt`. Key dependencies include:
- Streamlit
- NumPy
- SciPy
- Pandas
- Plotly
