import pandas as pd, numpy as np
from geopy.distance import geodesic

def centroid(lat, lng):
    x_ = sum(lat) / len(lat)
    y_ = sum(lng) / len(lng)
    return x_, y_

def box(lat, lng):
    return lat.min(), lat.max(), lng.min(), lng.max()

def distance(p1, p2):
    return list(map(lambda p1, p2: geodesic(p1, p2).meters, p1.values, p2.values))

def clusters_geometry(lat, lng, labels, include_box=False, include_center_radius=False):
    
    unique_labels = labels.unique()
    centroids, boxes = {}, {}
    
    for label in unique_labels:
        msk = (labels==label).values
        centroids[label] = centroid(lat[msk], lng[msk])
        boxes[label] = box(lat[msk], lng[msk])
    
    centroids = pd.DataFrame(
        list(map(lambda label: centroids[label], labels)),
        columns=['lat_centroid', 'lng_centroid'],
        index=labels.index
    )
    centroids['label_count'] = labels.value_counts().loc[labels].values
    
    boxes = pd.DataFrame(
        list(map(lambda label: boxes[label], labels)),
        columns=['lat_min', 'lat_max', 'lng_min', 'lng_max'],
        index=labels.index
    )

    radius_lat = (boxes['lat_max'] - boxes['lat_min']) / 2
    radius_lng = (boxes['lng_max'] - boxes['lng_min']) / 2
    
    if include_box:
        centroids = pd.concat([centroids, boxes], 1)
        
    if include_center_radius:
        centroids['lat_center'] = boxes['lat_min'] + radius_lat
        centroids['lng_center'] = boxes['lng_min'] + radius_lng
        centroids['horizontal_perimeter'] = distance(centroids[['lat_min', 'lng_center']], centroids[['lat_max', 'lng_center']])
        centroids['vertical_perimeter'] = distance(centroids[['lat_center', 'lng_min']], centroids[['lat_center', 'lng_max']])
        centroids['radius'] = centroids[['horizontal_perimeter', 'vertical_perimeter']].max(1) / 2
        centroids['area_box'] = centroids['horizontal_perimeter'] * centroids['vertical_perimeter']
        centroids['area_circle'] = np.pi * centroids['radius'] ** 2
        centroids['density_box'] = centroids['label_count'] / centroids['area_box']
        centroids['density_circle'] = centroids['label_count'] / centroids['area_circle']
        
            
    return centroids