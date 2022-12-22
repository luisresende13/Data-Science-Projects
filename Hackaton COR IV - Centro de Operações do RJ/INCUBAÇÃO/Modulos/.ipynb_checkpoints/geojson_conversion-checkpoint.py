import json, pandas as pd, numpy as np
from matplotlib.path import Path as mpl_path
from geomet import wkt

# ---
# Save and update geojson object

def save_json(obj, path):
    if type(obj) is dict: obj = json.dumps(obj)
    with open(path, 'w') as file:
        file.write(obj); file.close()
    print('Done!')

def drop_props(features, keep_cols):
    for feat in features:
        feat['properties'] = {key: feat['properties'][key] for key in keep_cols}
    return features

def add_props(features, df):
    for feature, (idx, row) in zip(features, df.iterrows()):
        props = row.to_dict()
        if 'properties' not in feature.keys():
            feature['properties'] = {}
        feature['properties'] = {**feature['properties'], **row}
    return features

def update_props_by_id(features, df, id_col, id_col_df=None, drop=[]):
    if id_col_df is None: id_col_df = id_col
    for idx, row in df.iterrows():
        _id, props = row[id_col_df], row.drop(drop + [id_col_df]).to_dict()
        feature = list(filter(lambda feat: feat['properties'][id_col] == _id, features))[0]
        feature['properties'] = {**feature['properties'], **props}
    return features

def update_objs_by_id(objs, df, id_col, id_col_df=None, drop=[]):
    if id_col_df is None: id_col_df = id_col
    for idx, row in df.iterrows():
        _id, props = row[id_col_df], row.drop(drop + [id_col_df]).to_dict()
        obj = list(filter(lambda obj: obj[id_col] == _id, objs))[0]
        obj = {**obj, **props}
    return objs

# ---
# Build geojson object

def geojson_obj(features):
    return {
        "type": "FeatureCollection",
        "features": features
    }

def feature_obj(geometry):
    return {
        'type': 'Feature',
        'geometry': geometry
    }

def add_props(features, df):
    for feat, (idx, row) in zip(features, df.iterrows()):
        feat['properties'] = row.to_dict()
    return features

# ---
# Convert wkt column to geojson

def wkt_geojson(df, wkt_col, props):
    geometries = df[wkt_col].apply(wkt.loads)
    features = geometries.apply(feature_obj)
    features = add_props(features.tolist(), df[props])
    return geojson_obj(features)

# ---
# Convert pandaas dataframe to geojson

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
        line_json['features'].append({
            'type': 'Feature',
            'geometry': {
                'type': 'LineString',
                'coordinates': [[point[x], point[y]] for point in row[coords]],
            },
            'properties': row.drop(coords).to_dict()
        })
    return line_json

def polygon_geojson(df, coords=['lng_min', 'lng_max', 'lat_min', 'lat_max']):
    polygon_json = {
        "type": "FeatureCollection",
        "features": []
    }
    for row in df.iterrows():
        row = row[1]
        polygon_json['features'].append({
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[
                    [row[coords[0]], row[coords[2]]],
                    [row[coords[0]], row[coords[3]]],
                    [row[coords[1]], row[coords[3]]],
                    [row[coords[1]], row[coords[2]]],
                    [row[coords[0]], row[coords[2]]],
                ]],
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
    for event_id, cluster_id in events_poly.items():
        df_ids[df[point_key_col]==event_id] = cluster_id

    return df_ids