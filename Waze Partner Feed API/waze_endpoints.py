import requests, json
import pandas as pd, numpy as np
from matplotlib.path import Path as mpl_path
from datetime import datetime

# Waze partner access token
waze_url = 'https://www.waze.com/partnerhub-api/waze-feed-access-token/c37c11ba-ff9d-4ad5-8ecc-4e4f12e91efb?format=1'

# Load custers polygons geojson data
polygons_geojson_path = '../Hackaton COR IV - Centro de Operações do RJ/ACELERAÇÃO/Dados/Clusters/polygons_micro.geojson'
polygons_geojson = json.loads(open(polygons_geojson_path, 'r').read())

# Get polygons dict from geojson
polygons = {}
for poly in polygons_geojson['features']:
    polygons[poly['properties']['sublabel']] = poly['geometry']['coordinates'][0]

# ---
# Waze partner feed methods

def get_waze_partner_alerts(alert_type='alerts'):
    "alert_type: one of 'alerts', 'irregularities', 'jams'"
    incidents = requests.get(waze_url).json()
    if alert_type not in incidents.keys():
        return None
    df = pd.DataFrame(incidents[alert_type])
    # Data cleaning & preprocessing
    if 'location' in df.columns:
        df[['latitude', 'longitude']] = list(df['location'].map(lambda coords: [coords['x'], coords['y']]))
        df.drop('location', axis=1, inplace=True)
    if 'pubMillis' in df.columns:
        df['pubMillis'] = (df['pubMillis'] / 1000).map(datetime.fromtimestamp)
        df.sort_values('pubMillis', ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df

def get_waze_partner_alerts_extended():
    # Get waze alerts
    alerts = get_waze_partner_alerts(alert_type='alerts')
    # Get events points array
    points = np.array(list(map(tuple, alerts[['latitude', 'longitude']].values)))
    # Get clusters polygons events dict
    events_poly = {}
    for cluster_id, poly in polygons.items():
        mpl_poly =  mpl_path(poly)
        points_msk = mpl_poly.contains_points(points)
        poly_events_df = alerts[points_msk]
        poly_events_ids = list(poly_events_df['uuid'])
        for event_id in poly_events_ids:
            events_poly[event_id] = cluster_id
    
    # Update events dataframe with events polygons ids
    alerts['cluster_id'] = -1
    for event_id, cluster_id in events_poly.items():
        alerts['cluster_id'][alerts['uuid']==event_id] = cluster_id

    # Return extended open events
    return alerts

def get_clusters_waze_partner_alerts():
    # Get waze alerts
    alerts = get_waze_partner_alerts(alert_type='alerts')
    # Get events points array
    points = np.array(list(map(tuple, alerts[['latitude', 'longitude']].values)))
    # Get clusters polygons events dict
    poly_events = []
    for cluster_id, poly in polygons.items():
        mpl_poly =  mpl_path(poly)
        points_msk = mpl_poly.contains_points(points)
        poly_events_df = alerts[points_msk]
        poly_events_ids = list(poly_events_df['uuid'])
        poly_wb_events_ids = list(poly_events_df['uuid'][poly_events_df['subtype']=="HAZARD_WEATHER_FLOOD"])
        poly_events.append({
            'cluster_id': cluster_id,
            'alerts': len(poly_events_ids),
            'alerts_ids': poly_events_ids,
            'alerts_status': int(bool(len(poly_events_ids))),
            'waterbag_alerts': len(poly_wb_events_ids),
            'waterbag_alerts_ids': poly_wb_events_ids,
            'waterbag_alerts_status': int(bool(len(poly_wb_events_ids))),
        })
    return pd.DataFrame(poly_events)