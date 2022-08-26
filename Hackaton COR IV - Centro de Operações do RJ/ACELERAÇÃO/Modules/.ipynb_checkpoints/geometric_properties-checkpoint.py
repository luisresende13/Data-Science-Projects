import pandas as pd

def centroid(lat, lng):
    x_ = sum(lat) / len(lat)
    y_ = sum(lng) / len(lng)
    return x_, y_

def box(lat, lng):
    return lat.min(), lat.max(), lng.min(), lng.max()
    
def clusters_centroids(lat, lng, labels, include_box=False, include_center_radius=False):
    
    unique_labels = labels.unique()
    centroids, boxes = {}, {}
    
    for label in unique_labels:
        msk = (labels==label).values
        centroids[label] = centroid(lat[msk], lng[msk])
        if include_box: boxes[label] = box(lat[msk], lng[msk])
    
    centroids = pd.DataFrame(
        list(map(lambda label: centroids[label], labels)),
        columns=['lat - centroid', 'lng - centroid'],
        index=labels.index
    )
    
    if include_box:
        boxes = pd.DataFrame(
            list(map(lambda label: boxes[label], labels)),
            columns=['lat_min', 'lat_max', 'lng_min', 'lng_max'],
            index=labels.index
        )
        centroids = pd.concat([centroids, boxes], 1)
        if include_center_radius:
            radius_lat = (boxes['lat_max'] - boxes['lat_min']) / 2
            radius_lng = (boxes['lng_max'] - boxes['lng_min']) / 2
            centroids['radius_lat'] = abs(radius_lat)
            centroids['radius_lng'] = abs(radius_lng)
            centroids['lat - center'] = boxes['lat_min'] + radius_lat
            centroids['lng - center'] = boxes['lng_min'] + radius_lng
            
    return centroids