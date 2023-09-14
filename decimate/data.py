#!/usr/bin/env python3

import time
import os
import hashlib
import itertools
from requests import post
import xml.etree.ElementTree as ET
from pandas import json_normalize
import numpy as np
import pandas as pd
import shapely.geometry


def parse_kml(f):
    """parse a google earth kml file and return xyz path coordinates as ndarray

    No guarantee this works with arbitrary kml or google earth versions

    Args:
        f (str): kml file
    Returns:
        xyz (ndarray): lat/lon/?? coords of the path in the kml file
    """
    tree = ET.parse(f)
    txt = tree.find('.//{http://www.opengis.net/kml/2.2}coordinates').text
    xyz = np.vstack([[np.asarray([float(k) for k in st.split(',')])] for st in txt.split()])
    xyz[:, 0:2] = xyz[:, [1, 0]]
    return xyz

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


def get_area(perimeter=None, stride=50):
    """download an area/grid from the opentopodata api

    POST requests are chunked to 100 points apiece and spaced 1.1s apart and
    can be interrupted/resumed if the partial csv files are not deleted

    Args:
        perimeter (list): list of two or more (lat, lon) tuples. If two points,
          interpret as rectangle corners, otherwise, if more than two points,
          interpret as a (sorted and closed) polygon perimeter.
        stride (float): grid spacing (in meters)
    Returns:
        data (DataFrame): `x`, `y` columns are lon and lat in meters (elevation
          is also in meters)
    """
    def grid_snap(x0, dx, vals):
        """snap `vals`` to a 1D grid with point `x0`` and spacing `dx``"""
        return x0+((vals-x0+dx*0.49)//dx)*dx

    os.makedirs('csv_chunks', exist_ok=True)
    perimeter = np.asarray(perimeter)

    # convert meter stride to lat/lon stride
    lon_scale = np.abs(np.cos(np.mean(perimeter[:, 0])/180*np.pi))
    lon_stride = stride/111319./lon_scale
    lat_stride = stride/111319.

    # snap perimeter points to grid
    snappy = perimeter*0
    snappy[:, 0] = grid_snap(np.min(perimeter[:, 0]), lat_stride, perimeter[:, 0])
    snappy[:, 1] = grid_snap(np.min(perimeter[:, 1]), lon_stride, perimeter[:, 1])

    # build lat/lon grid
    lon_limits = [np.min(snappy[:, 1]), np.max(snappy[:, 1])]
    lat_limits = [np.min(snappy[:, 0]), np.max(snappy[:, 0])]
    lon_vec = np.arange(*lon_limits, lon_stride)
    lat_vec = np.arange(*lat_limits, lat_stride)
    latlon = list(itertools.product(lat_vec, lon_vec))
    if len(perimeter) > 2:
        snappy_tups = [tuple(x) for x in snappy]
        polygon = shapely.geometry.Polygon(snappy)
        latlon = [x for x in latlon if shapely.geometry.Point(x).within(polygon) and x not in snappy_tups]
        # perimeter points are first and not duplicated
        latlon = snappy_tups + latlon
        is_perimeter = [True if x in snappy_tups else False for x in latlon]

    # download in chunks
    chunksize = 100
    tag = hashlib.md5((str(latlon)).encode()).hexdigest()[:8]
    df_todo = pd.DataFrame(data=latlon, columns=['lat', 'lon'])
    df_todo['chunk'] = np.arange(len(df_todo)) // chunksize
    all_csv = []
    print('REQUESTING %i elevation points (%i chunks)' % (len(df_todo), len(df_todo)//chunksize+1))
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
    data['y'] = ((data['location.lat']-np.mean(lat_vec))*111319.).astype(int)
    data['x'] = ((data['location.lng']-np.mean(lon_vec))*111319.*lon_scale).astype(int)
    data['is_perimeter'] = is_perimeter

    return data

def mesh2stl(x, filename):
    scale = 0.01  #
    from stl import mesh
    f = np.array(x.triangles)
    v = np.array(x.vertices)*scale
    surf = mesh.Mesh(np.zeros(f.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(f):
        for j in range(3):
            surf.vectors[i][j] = v[face[j], :]
    surf.save(file)


if __name__ == "__main__":


    # download halfdome data at 50m resolution
    corners = [(37.7500, -119.5430), (37.7390, -119.5280)]
    data = get_area(perimeter=corners, stride=50)
    #data.to_csv('data_halfdomecrop_srtm30m_0030m.csv', float_format='%8.4f')



    pass
