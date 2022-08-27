import googlemaps, pandas as pd, numpy as np
from IPython.display import clear_output as co

def googleReverseGeocode(
    coordinates, coord_ids=None,
    result_type=None, location_type=None,
    language='pt-BR',
    googleAPIKey=None,
    keep_cols =  ['place_id', 'types', 'formatted_address'],
    drop_cols = ['address_components', 'geometry', 'plus_code'], # if included, 'keep_cols' argument is ignored
    keep_geometry_cols = ['location', 'location_type'],
):
    gmaps = googlemaps.Client(key=googleAPIKey) # load google api key
    if coord_ids is None: coord_ids = np.arange(len(coordinates))
    result = []; n_coords = len(coordinates)
    for i, (coords, coord_id) in enumerate(zip(coordinates, coord_ids)):
        res = gmaps.reverse_geocode(
            coords, language='pt-BR',
            result_type='|'.join(result_type),
            location_type='|'.join(location_type)
        )
        df = pd.DataFrame(res); coords_df = []
        if drop_cols is not None:
            found_cols = set(drop_cols).intersection(df.columns)
            keep_cols = df.drop(found_cols, 1).columns
        for j, row in df.iterrows():
            keep_info = row[keep_cols]
            keep_info['search_id'] = coord_id
            location = pd.Series(row['geometry']['location'])
            location['location_type'] = row['geometry']['location_type']
            address = pd.DataFrame(row['address_components'])
            address = pd.Series(
                address['long_name'].values,
                index=address['types'].map(lambda types: ', '.join(types))
            )
            coords_df.append(pd.concat([keep_info, location, address], 0))
        result.append(pd.DataFrame(coords_df))
        print(f'{i+1}/{n_coords} coordinates reversed geocoded.'); co(wait=True)

    print(f'Done! Total of {n_coords} requests.')
    return pd.concat(result, 0)