import pandas as pd

class waterbag_project:
        
    def __init__(
        self, time_serie='clusters', freq='upsample',
        load_waterbags=True, load_stations=False,
        time_features=False, inmet_features=False,
        alerta_features=False,
    ):
        self.load_waterbags = load_waterbags
        self.time_serie = time_serie
        self.freq = freq
        self.load_stations = load_stations
        self.time_features = time_features
        self.inmet_features = inmet_features
        self.alerta_features = alerta_features
        self.path = {
            'waterbags': 'Dados/Catalog/water_bag_catalog_google.csv',
            'clusters': 'Dados/Clusters/clusters_bolsões_micro.csv',
            'inmet': 'Dados/Clean/INMET.csv',
            'alerta_rio': '../../../Dados/Desafio COR-Rio IV/Clean/ALERTA-RIO.csv',
            'waterbags_timeserie': 'Dados/Transform/série_bolsões.csv',
            'waterbags_timeserie_clusters': 'Dados/Transform/série_bolsões_clusters.csv',
            'feat_eng': '../../../Dados/Desafio COR-Rio IV/Engenharia de Features/',
            'time_features': 'time_features.csv',
            'inmet_eng': 'inmet_2-84h.csv',
            'alerta_eng': ('alertario_15-45min.csv', 'alertario_1-4h.csv', 'alertario_5-84h.csv'),
        }
        self.load_data(self.path)
                
    def load_data(self, path):

        # Event labeled time serie
        if self.time_serie == 'waterbags':
            ts = pd.read_csv(path['waterbags_timeserie'], index_col=0)
            ts['event groups'] = ts['event groups'].map(json.loads)
            ts['event ids'] = ts['event ids'].map(json.loads)
            ts.set_index(pd.to_datetime(ts.index), inplace=True)

        # Event labeled time serie per cluster group
        elif self.time_serie == 'clusters':
            ts = pd.read_csv(path['waterbags_timeserie_clusters'], index_col=0)
            ts.set_index(pd.to_datetime(ts.index), inplace=True)

        # INMET metheorological stations' records
        inmet = pd.read_csv(path['inmet'], index_col=0)
        inmet.set_index(pd.to_datetime(inmet.index), inplace=True)

        # Alerta-Rio metheorlogical stations' records
        alerta_rio = pd.read_csv(path['alerta_rio'], index_col=0)
        alerta_rio.set_index(pd.to_datetime(alerta_rio.index), inplace=True)

        # Combine Inmet and Alerta-Rio Stations Time Series by down or up Sampling

        if self.freq is not None:
            # Downsample high frequency time serie
            if self.freq == 'downsample':
                downsample = alerta_rio.resample('H').first()
                self.data = inmet.join(downsample, how='outer') # downsample = None
            # Upsample less frequent time serie
            elif self.freq == 'upsample':
                upsample = inmet.resample('15Min').pad()
                self.data = upsample.join(alerta_rio, how='outer') # upsample = None
#         self.data.index = pd.DatetimeIndex(self.data.index)

                
        # Join engineered features 
        if self.data is not None:
            if self.time_features:
                time_feats = pd.read_csv(self.path['feat_eng'] + path['time_features'], index_col=0)
                time_feats.index = pd.DatetimeIndex(time_feats.index)
                self.data = time_feats.join(self.data, how='right')
            if self.inmet_features:
                inmet_eng = pd.read_csv(self.path['feat_eng'] + self.path['inmet_eng'], index_col=0)
                inmet_eng.index = pd.DatetimeIndex(inmet_eng.index)
                self.data = self.data.join(inmet_eng, how='left')
            if self.alerta_features:
                alerta_eng = pd.concat([
                    pd.read_csv(self.path['feat_eng'] + file, index_col=0) for file in self.path['alerta_eng']
                ], 1)
                alerta_eng.index = pd.DatetimeIndex(alerta_eng.index)
                self.data = self.data.join(alerta_eng, how='left')

            # Reindex target labels to new datatime index
        if self.time_serie is not None and self.freq is not None:
            self.time_serie = ts.reindex(self.data.index).fillna(0) # Not working for downsampling ***            
        
        # Store stations data
        if self.load_stations:
            self.inmet = inmet; self.alerta_rio = alerta_rio
        inmet, alerta_rio = None, None

        # Water bag groupped events collection
        if self.load_waterbags:
            waterbags = pd.read_csv(path['waterbags'], index_col=0)
            clusters = pd.read_csv(path['clusters'], index_col=0)
            waterbags[['EVENTO_INICIO', 'EVENTO_FIM']] = waterbags[['EVENTO_INICIO', 'EVENTO_FIM']].apply(pd.to_datetime)
            self.waterbags = waterbags.join(clusters, how='outer')


from sklearn.preprocessing import LabelEncoder as le

def custom_preprocessing(X, drop_empty_cols=False, label_encode=None, interpolate='linear', fillna='mean'):

    print('Initial shape:', X.shape)

    if drop_empty_cols: # Drop X empty columns and rows
        X.dropna(axis=1, how='all', inplace=True)
        print('Empty columns removed: ', X.shape)

    if label_encode is not None:
        print('Label columns encoded:', list(label_encode))
        LEi = {}
        for col in label_encode:
            LEi[col] = le().fit(X[col])
            X[col] = LEi[col].transform(X[col])

    if interpolate is not None: # Interpolate X missing values
        print('Interpolation:', interpolate)
        X = X.interpolate(interpolate)

    if fillna is not None: # Fill missing values with the minimum column value
        print('Fill missing values:', fillna)
        for col in X:
            if fillna=='min':
                fill_value = X[col].min()
            elif fillna=='mean':
                fill_value = X[col].mean()
            else:
                fill_value = 0
            X[col].fillna(fill_value, inplace=True)

    if label_encode is not None:
        return X, LEi
    return X

from Modulos.imbalanced_selection import groupConsecutiveFlags

# Target selection and train/test split
def select_target(Yi, X, target_id, periods_ahead=None, shift=0, names=None, fill_value=0.0):
    if names is not None: print(f'Selected Target: {names[int(target_id)]} - id: {target_id}', '\n')

    # Select target
    Y = Yi[str(target_id)].loc[X.index].copy()
    y_cnt = Y.value_counts().to_frame('Target')

    ### Target transformation
    if periods_ahead is not None:
        Y = (Y.rolling(periods_ahead, closed='left', min_periods=1).sum().shift(1 - periods_ahead) > 0).astype('float')
        print('Target range:', periods_ahead)
    if shift is not None:
        Y = Y.shift(shift, fill_value=0.0)
        print('Target shift:', shift)

    y_cnt = pd.concat([y_cnt, Y.value_counts().to_frame('Transformed Target')], axis=1)

    # Group target positive class labels by being consecutive in time (group evaluation strategy)
    groups = groupConsecutiveFlags(ts=Y)
    
    return Y, groups, y_cnt

