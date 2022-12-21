import json, pandas as pd, numpy as np
from matplotlib.path import Path as mpl_path

def save_json(obj, path):
    if type(obj) is dict: obj = json.dumps(obj)
    with open(path, 'w') as file:
        file.write(obj); file.close()
    print('Done!')

def points_geojson(df, coords=['EVENTO_LONGITUDE', 'EVENTO_LATITUDE']):
    points_json = {
        "type": "FeatureCollection",
        "features": []
    }
    for idx, row in df.iterrows():
        points_json['features'].append({
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [row[coords[0]], row[coords[1]]],
            },
            'properties': row.drop(coords).to_dict()
        })
    return points_json

def linestring_geojson(df, coords='line', keys=['x', 'y']):
    x, y = keys
    line_json = {
        "type": "FeatureCollection",
        "features": []
    }
    for idx, row in df.iterrows():
        line = row[coords]
        coordinates = ([[point[x], point[y]] for point in line] if keys is not None else line)
        line_json['features'].append({
            'type': 'Feature',
            'geometry': {
                'type': 'LineString',
                'coordinates': coordinates,
            },
            'properties': row.drop(coords).to_dict()
        })
    return line_json


def polygon_geojson(df, coords=['lng_min', 'lng_max', 'lat_min', 'lat_max'], drop=[]):
    polygon_json = {
        "type": "FeatureCollection",
        "features": []
    }
    for idx, row in df.iterrows():
        coordinates = ([[
                    [row[coords[0]], row[coords[2]]],
                    [row[coords[0]], row[coords[3]]],
                    [row[coords[1]], row[coords[3]]],
                    [row[coords[1]], row[coords[2]]],
                    [row[coords[0]], row[coords[2]]],
                ]] if type(coords) is not str else row[coords])
        polygon_json['features'].append({
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': coordinates,
            },
            'properties': row.drop(coords).to_dict()
        })
    return polygon_json

def pointsPolygonIds(df, polygons_geojson, coord_cols, point_key_col, polygon_key_col):
    
    # ---
    # df must be a dataframe of shape (n_points x 3) containig one id column and both latitude and longitude columns

    # Get open events dataframe
    # Get events points array
    points = np.array(list(map(tuple, df[coord_cols].values)))

    # Get polygons dict from geojson
    polygons_dict = {}
    for poly in polygons_geojson['features']:
        if poly['properties'][polygon_key_col] != -1:
            polygons_dict[poly['properties'][polygon_key_col]] = poly['geometry']['coordinates'][0]

    # Get events polygons dict
    events_poly = {}
    for cluster_id, poly in polygons_dict.items():
        mpl_poly =  mpl_path(poly)
        points_msk = mpl_poly.contains_points(points)
        poly_events_df = df[points_msk]
        poly_events_ids = list(poly_events_df[point_key_col])
        for event_id in poly_events_ids:
            events_poly[event_id] = cluster_id

    # Update events dataframe with events polygons ids
    df_ids = pd.Series(- np.ones(len(df)), index=df.index, name=polygon_key_col)
#     df[polygon_key_col] = -1
    for event_id, cluster_id in events_poly.items():
        df_ids[df[point_key_col]==event_id] = cluster_id

    return df_ids