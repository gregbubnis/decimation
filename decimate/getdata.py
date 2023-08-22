#!/usr/bin/env python3

import time
import os
import hashlib
from requests import post
from pandas import json_normalize
import numpy as np
import pandas as pd

def get_elevation_post_opentopodata(data=None):
    """post request to the opentopodata API

    - https://www.opentopodata.org/
    - max 100 locations per POST request
    - max 1 call per second
    - max 1000 calls per day
    """
    locs = '|'.join(['%f,%f' % (x,y) for x,y in data])
    data = dict(locations=locs, interpolation='cubic')
    url = "https://api.opentopodata.org/v1/srtm30m"
    #url = "https://api.opentopodata.org/v1/eudem25m"
    r = post(url, json=data, timeout=300)
    if r.status_code == 200 or r.status_code == 201:
        return json_normalize(r.json(), 'results')
    else:
        print('FAIL: status_code:', r.status_code)
        return None

def get_area(corners=None, stride=50):
    """download an area/grid from the opentopodata api

    POST requests are chunked to 100 points apiece and spaced 1.1s apart and
    can be interrupted/resumed if the partial csv files are not deleted

    Args:
        corners (list): list of two (lat,lon) tuples for rectangle corners
        stride (float): data spacing (in meters)
    Returns:
        data (DataFrame): `x`, `y` columns are lon and lat in meters (elevation
            is also in meters)

    - stride is in meters
    - returns a DataFrame
    """
    os.makedirs('csv_chunks', exist_ok=True)

    corners = np.asarray(corners)
    lon = [np.min(corners[:, 1]), np.max(corners[:, 1])]
    lat = [np.min(corners[:, 0]), np.max(corners[:, 0])]
    lon_scale = np.cos(np.mean(lat)/180*np.pi)
    x_stride = stride/111319./lon_scale
    y_stride = stride/111319.
    lonx = np.arange(*lon, x_stride)
    laty = np.arange(*lat, y_stride)

    #print('lon scale', lon_scale)

    latlon = []
    for iy, y in enumerate(laty):
        for ix, x in enumerate(lonx):
            latlon.append((y, x))

    # chunk the requests and then reassemble 
    chunksize = 100
    tag = hashlib.md5((str([str(lat), str(lon), str(stride)])).encode()).hexdigest()[:8]
    df_todo = pd.DataFrame(data=latlon, columns=['lat', 'lon'])
    df_todo['status'] = 0
    df_todo['chunk'] = np.arange(len(df_todo)) // chunksize
    all_csv = []

    print('REQUESTING %i elevation points' % len(df_todo))
    for i in df_todo['chunk'].unique():
        csv = 'csv_chunks/data_%s_%4.4i.csv' % (tag, i)      
        all_csv.append(csv)
        if os.path.isfile(csv):
            print('  HAVE %s' % csv) 
        else:
            time.sleep(1.1)
            print(' FETCH %s' % csv)
            df_chunk = df_todo[df_todo['chunk']==i]
            req = [tuple(x) for x in df_chunk[['lat', 'lon']].values]
            data = get_elevation_post_opentopodata(data=req)
            data.to_csv(csv)

    # reassemble
    data = pd.concat([pd.read_csv(x, index_col=0) for x in all_csv], ignore_index=True)
    data['y'] = ((data['location.lat']-np.mean(laty))*111319.).astype(int)
    data['x'] = ((data['location.lng']-np.mean(lonx))*111319.*lon_scale).astype(int)
    uniq_x = data['x'].unique()
    uniq_y = data['y'].unique()
    xx = dict(zip(uniq_x, range(len(uniq_x))))
    yy = dict(zip(uniq_y, range(len(uniq_y))))
    data['ix'] = [xx[x] for x in data['x']]
    data['iy'] = [yy[x] for x in data['y']]

    return data



if __name__ == "__main__":

    pass
