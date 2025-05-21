import json
import numpy as np
import pandas as pd
import urllib.request
from io import StringIO


def read_sample_data(sample_file):
    """
    Reads the sample data from the sample_file and 
    returns the xyz data.  
    """

    df = pd.read_csv('survey_data.csv', delimiter=',', header=None)
    x = [float(i) for i in df[0].values.tolist()]
    y = [float(i) for i in df[1].values.tolist()]
    z = [float(i) for i in df[2].values.tolist()]

    return x, y, z


def read_xyz_file(input_file, nrows_to_skip, delimiter, decimal):
    """
    Reads the input file and returns the xyz data.
    Parameters:
      input_file: the input file name
      nrows_to_skip: the number of rows to skip
      delimiter: the delimiter for the input file
      decimal: the decimal for the input file
    """
    if delimiter == 'Space':
        delim = None
    elif delimiter == 'Comma':
        delim = ','
    elif delimiter == 'Semicolon':
        delim = ';'
    else:
        delim = '\t'

    file_content = str(input_file.getvalue(), 'utf-8')
    data = StringIO(file_content)
    lines = data.readlines()

    x, y, z = [], [], []
    for line in lines:
        data = line.split(delim)
        if decimal == 'Comma':
            data = [x.replace(',', '.') for x in data]
        x.append(float(data[0]))
        y.append(float(data[1]))
        z.append(float(data[2]))

    return x, y, z


def request_data_from_open_elevation(lat1, long1, lat2, long2):
    """
    Requests data from the open elevation API.
    Parameters:
      lat1: the latitude of the first point
      long1: the longitude of the first point
      lat2: the latitude of the second point
      long2: the longitude of the second point
    """
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
            response = urllib.request.Request(
                url, json_locs,
                headers={'Content-Type': 'application/json'}
            )
            response_file = urllib.request.urlopen(response)
        except (OSError, urllib.error.HTTPError):
            count += 1
            continue
        break
    
    if count > number_of_tries:
        return None, None, None

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
