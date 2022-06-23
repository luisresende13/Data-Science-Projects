import pandas as pd
import os
import sklearn.preprocessing
from IPython.display import clear_output as co 

le = sklearn.preprocessing.LabelEncoder
target = 'area'

class preprocess:
    
    def __init__(self):
        self.name = 'preprocessing functions class'

    def extract_series(data, key0='city_code', key1='product', split_category='Others',
                       split_key='product_type', split_values=['temporary', 'permanent'],
                       sort_by='year', save=False, path='data/series/'):

        cities = data[key0].unique()
        products = data[key1].unique()
        ts_i = {}
        for city_code in cities:
            city_ts = data[data[key0]==city_code]
            for product in products:
                prod_ts = city_ts[city_ts[key1]==product].sort_values(sort_by).copy()
                if len(prod_ts):
                    if product==split_category:
                        for prod_type in split_values:
                            prod_type_ts = prod_ts[prod_ts[split_key]==prod_type].copy()
                            if len(prod_type_ts):
                                key = product+'-'+prod_type
                                ts_i[city_code+'-'+key] = prod_type_ts
                    else:
                        key = product
                        ts_i[city_code+'-'+key] = prod_ts

        #### Saving isolated time series
        if save:
            try:
                os.mkdir(path)
            except:
                for key in ts_i.keys():
                    ts_i[key].to_csv(os.path.join(path, key+'.csv'), index=True)
        return ts_i

    def load_series(path='series/', index_col=0):
        filenames = os.listdir(path)
        series, cnt, n_files = {}, 0, len(filenames)
        freq = range(1, n_files, 10)
        for filename in filenames:
            cnt+=1
            if cnt in freq: print(f'Files loaded: {cnt}/{n_files}'); co(wait=True)
            series[filename[:-4]] = pd.read_csv(path+filename, index_col=index_col)
        print(f'Done! Loaded {n_files} files.')
        return series

    def concat_series(series, le_index=True, save=False, path='data/series.csv'):
        keys= series.keys()
        X = pd.concat([series[key].set_index('year')[target] for key in keys], 1).sort_index()
        X.columns = keys;
        if le_index:
            X.index = le().fit(X.index).transform(X.index)
        if save:
            X.to_csv(path, index=True)
        return X

    def clean_series(series, train_min=1, drop_start_zeros=True, keep_all_zero=True, save=False, path='data/series clean/{}.csv'):
        clean_series = {}
        test_empty = []
        excluded_test_index = []
        n_total = len(series)
        n_values_total = 0
        n_values_test = 0
        n_values_train = 0
        n_series = 0
        n_series_empty = 0
        n_series_test_empty = 0
        n_series_train_min_empty = 0
        n_values = 0
        n_test_empty_total = 0
        n_train_empty_total = 0
        n_non_empty_values = 0
        n_train_zeros = 0

        cnt = 0
        keys = list(series.keys())
        for key in keys:
            cnt+=1; print(f'Cleaned series: {cnt}/{n_total}'); co(wait=True)

            msk = series[key]['year'].isin(['01/01/2016', '01/01/2017'])
            test = series[key][msk]
            train = series[key][msk==False]

            n = len(series[key])
            n_empty = series[key][target].isna().sum()
            n_test_empty = test[target].isna().sum()
            n_train_empty = train[target].isna().sum()

            cond1 = n_test_empty == len(test)
            cond2 = n_train_empty > (len(train) - train_min) # if number of training samples is greater than specified minimum.

            n_values_total += n
            n_values_test += len(test)
            n_values_train += len(train)

            n_series_empty += n==n_empty
            n_test_empty_total+=n_test_empty
            n_train_empty_total+=n_train_empty
            n_series_test_empty += cond1
            n_series_train_min_empty += cond2

            if cond1 or cond2:
                n_series+=1
                n_values+=len(series[key])
                n_non_empty_values+=(len(series[key]) - n_empty)
                excluded_test_index.extend(test[test[target].isna()==False].index.tolist())

            else:
                n_values+=n_empty
                clean_series[key] = series[key].dropna(subset=['area'])
                if drop_start_zeros:
                    if keep_all_zero:
                        test_msk = clean_series[key]['year'].isin(['01/01/2016', '01/01/2017'])
                        train_msk = test_msk==False
                        isNotZero = (clean_series[key][train_msk]['area']!=0).tolist()
                        if sum(isNotZero)!=0:
                            testset = clean_series[key][test_msk]
                            trainset_cut = clean_series[key][train_msk].iloc[isNotZero.index(True):]
                            clean_series[key] = pd.concat([trainset_cut, testset])
                            n_train_zeros+=(sum(train_msk)-len(trainset_cut))

        if save:
            cnt=0
            for key in clean_series.keys():
                cnt+=1; print(f'Saved clean series: {cnt}/{len(clean_series)}'); co(wait=True)
                clean_series[key].to_csv(path.format(key), index=True)
                
        print('Values Count:')
        print( pd.Series(
            [n_total, n_values_total, n_values_test, n_values_train],
            index=['n_total', 'n_values_total', 'n_values_test', 'n_values_train'])
        )
        print(); print('Excluded count:')
        print(
            pd.Series([
                n_series, n_series_empty, n_series_test_empty, n_series_train_min_empty,
                n_values, n_values-n_non_empty_values, n_test_empty_total, n_train_empty_total,
                n_non_empty_values, len(excluded_test_index), n_train_zeros
            ],
            index=[
                'series', 'series_empty', 'series_test_empty', f'series_train_min_<_{train_min}',
                'values', 'values_empty', 'test_empty', 'train_empty',
                'non_empty_values', 'non_empty_test_values', 'n_train_zeros'
            ])
        )
        return clean_series, excluded_test_index

    #### Extract predictive and target variables
    def get_xy(self, data, target='area', fill_na=0):
        ## 1. Extraindo variáveis preditivas
        X = data.drop(target, 1)
        ## 2. Extraindo Variável Álvo
        Y = data[target].copy()
        #### Substituição de Valores Vazios para Variável Alvo
    #     Y[Y.isna()] = fill_na
        return X,Y

    #### Label Encode pandas dataframe
    def label_encode(self, X, base=None):
        if type(base)==type(None): base = X
        X_lab = X.copy()
        for column in X:
            X_lab[column] = le().fit(base[column]).transform(X[column])
        return X_lab

    ## Separação das amostras para treinamento e teste
    def split_ts(self, x, y, col='year', test_values=[42,43]):
        msk = x[col].isin(test_values)
        return [x[msk==False], x[msk], y[msk==False], y[msk]]

    
    ### Train test split of time series by key
    def custom_tts(self, df, le_base, target='area', x_cols=['year'], index_col='year'):
        x, y = self.get_xy(df, target, fill_na=0)
        x_lab = self.label_encode(x[x_cols], base=le_base)
        return self.split_ts(x_lab, y, col=index_col)
    
    #### Extracting test data from series (2016 and 2017)
    def test_data(self, series, data, keys=None):
        if keys is None: keys = list(series.keys())
        Y_e = []
        for key in keys:
            x_t, x_e, y_t, y_e = self.custom_tts(series[key], data)
            Y_e.append(y_e)
        return pd.concat(Y_e)

class category_index:
    
    def __init__(self):
        self.name='extract dataframe indexes by category'
        
    def get_ctgr_combs_indexes(data, prods, col1='product', col2='product_type', sep='-', exclude=['Sorghum', 'Açaí']):
        prod_indexes = {}
        for prod in prods:
            if prod not in exclude:
                if sep in prod: 
                    product, prod_type = prod.split(sep)
                else:
                    product = prod
                df = data[data[col1]==product].copy()        
                if '-' in prod: 
                    df = df[df[col2]==prod_type]
                prod_indexes[prod] = df.index.copy()
        return prod_indexes

    def get_ctgrs_indexes(data, col='product_type', cats=['temporary', 'permanent', 'pasture']):
        cats_indexes = {}
        for cat in cats:
            df = data[data[col]==cat]      
            cats_indexes[cat] = df.index.copy()
        return cats_indexes
    
    #### Mapping between the series test indexes and the series key
    def get_key_index_test_map(series, index):    
        key_test_index_map = {}
        for key in series.keys():    
            key_index = series[key].index
            for entry_index in key_index:
                if entry_index in index:
                    key_test_index_map[entry_index] = key
        return key_test_index_map
    
    #### Mapping between the series keys and associated product_types
    def get_key_prodtype_map(series):
        key_prodtype_map = {}
        for key in series.keys():
            key_prodtype_map[key] = series[key].iloc[0]['product_type']
        return key_prodtype_map
