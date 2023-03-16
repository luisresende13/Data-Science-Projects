import requests, pandas as pd, geopandas as gpd, numpy as np
from urllib.parse import urlencode
from scipy.spatial import distance_matrix
from geopy import distance
from typing import Union
from copy import deepcopy

# baseurl = 'https://octa-api-oayt5ztuxq-ue.a.run.app' # us-east1
# baseurl = 'https://api.octacity.org' # us-central1
# baseurl = 'https://api.octacity.dev' # us-central1

baseurl = 'http://127.0.0.1:5000' # octa.rio api

def df(path, query={}, join_query=False):
    url = f'{baseurl}/{path}?{urlencode(query)}'
    res = requests.get(url)
    if res:
        data = pd.DataFrame(res.json())
        if join_query: return pd.concat([pd.DataFrame(query, data.index), data], axis=1)
        return data
    else:
        print({'type': 'RECORD MANY URL REQUEST FAILED.', 'url': url, 'status_code': res.status_code, 'message': res.reason})
        return None
        
def reindex(df, index):
    df.index = index; return df

def df_query(df, query={}):
    df = deepcopy(df)
    for key, value in query.items():
        if key in df:
            if value is not list: value = [str(value)]
            df = df[df[key].astype(str).isin(value)]
        else: print(f'RECORD MANY ERROR: PROVIDED QUERY KEY NOT FOUND IN URL DATAFRAME COLUMNS. KEY: {key}. COLUMNS: {df.columns}')
    return df

nclosest_fields = ['Codigo', 'distance', 'camera_position', 'prefix', '_id', 'path']

class Cameras:
    
    def get_points(url):
        points = df(url['url'])
        if points is not None:
            if len(points):
                return points.set_index(url['_id'])
        return pd.DataFrame([], pd.Series(dtype=str, name=url['_id']), url['coords'])
                
    def __init__(self, urls:Union[list,None]=None, src:str='static/city/cameras.csv'):
        self.urls = urls
        self._id = 'Codigo'
        self.coords = ['Longitude', 'Latitude']
        self.df = pd.read_csv(src)
        self.lnglat = pd.Series(zip(self.df['Longitude'], self.df['Latitude']))
    
    def geodesic_distances(self, point): 
        return self.lnglat.apply(lambda x: distance.geodesic(point, x).m)

    def geodesic_distance_matrix(self, coords:pd.DataFrame):
        return reindex(coords.apply(self.geodesic_distances, axis=1).T, self.df[self._id])

    def nclosest(self, dists, n, radius, prefix):
        nclosest = []
        nrange = np.array(range(n))
        for col in dists:
            ndists = dists[col].nsmallest(n).reset_index(name='distance')
            ndists['camera_position'] = nrange + 1
            ndists['prefix'] = prefix
            ndists['_id'] = col
            nclosest.append(ndists)
        if not len(nclosest): return pd.DataFrame(columns=nclosest_fields)
        nclosest = pd.concat(nclosest).reset_index(drop=True)
        nclosest['path'] = nclosest['prefix'] + '/' + nclosest['_id'].astype(str)
        return nclosest[nclosest['distance'] <= radius].reset_index(drop=True)
        
    def triggered(self, url:Union[dict,list,None]=None, n:int=3, radius:float=100.0):
        """
        url: None or object or list of objects with fields 'url', 'query', '_id', 'coords' and 'prefix'. If 'query' is of type list, than its elements should be objects with fields 'query' and 'prefix' (upper 'prefix' field will be ignored if present.).
        """
        if url is None: url = self.urls
        if type(url) is list: return pd.concat([self.triggered(urli, n, radius) for urli in url])
        if 'query' not in url: queries = [{'query': {}, 'prefix': url['prefix']}]
        elif type(url['query']) is dict: queries = [{'query': url['query'], 'prefix': url['prefix']}]
        else: queries = url['query']
        points = Cameras.get_points(url)
        return pd.concat([self.nclosest(
            self.geodesic_distance_matrix(df_query(points, query['query'])[url['coords']]),
            n, radius, query['prefix']
        ) for query in queries])