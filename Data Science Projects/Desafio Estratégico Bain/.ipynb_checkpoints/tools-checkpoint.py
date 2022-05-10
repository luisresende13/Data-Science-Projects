from IPython.display import clear_output as co 
import pandas as pd
import sklearn.preprocessing
le = sklearn.preprocessing.LabelEncoder

target = 'area'
class preprocess:
    
    def __init__(self):
        self.name = 'preprocessing functions class'
    
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
    def split_ts(self, x, y, col='year', horizon=2):
    #     periods = x[col].sort_values().iloc[:len(x)-horizon]
        msk = x[col].isin([42, 43])
        return [x[msk==False], x[msk], y[msk==False], y[msk]]

    
    ### Train test split of time series by key
    def custom_tts(self, key):
        x, y = self.get_xy(series[key], target='area', fill_na=0)
        x_lab = self.label_encode(x[['year']], base=data)
        return self.split_ts(x_lab, y, col='year', horizon=2)
    
    def clean_series(self, series, train_min=1, drop_start_zeros=True, keep_all_zero=True):
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

        cnt, freq = 0, range(1, n_total, 10)
        for key in list(series.keys()):
            cnt+=1
            if cnt in freq: print(f'Cleaned series: {cnt}/{n_total}'); co(wait=True)

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