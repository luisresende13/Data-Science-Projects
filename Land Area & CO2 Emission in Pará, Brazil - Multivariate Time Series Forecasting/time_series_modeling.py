import os
import pandas as pd 
import numpy as np; np.random.seed(25486)
import matplotlib.pyplot as plt
from IPython.display import clear_output as co
import json
import time
import sklearn.utils

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
from pmdarima.arima import auto_arima
from tbats import TBATS

le = sklearn.preprocessing.LabelEncoder
mtrcs = sklearn.metrics._regression
mae = mtrcs.mean_absolute_error
mse = mtrcs.mean_squared_error
mape = mtrcs.mean_absolute_percentage_error
r2 = mtrcs.r2_score
me = mtrcs.max_error
medae = mtrcs.median_absolute_error
evs = mtrcs.explained_variance_score
mpd = mtrcs.mean_poisson_deviance
mgd = mtrcs.mean_gamma_deviance
def wape(ye, yhat):
    return np.abs(ye-yhat).sum()/ye.sum()
def e(ye, yhat):
    return np.abs(yhat-ye).sum()
def estd(ye, yhat):
    return np.abs(yhat-ye).std()
def mpe(ye, yhat):
    return (np.abs(ye-yhat)/ye).mean()

def save_json_file(file, path, filename):
    full_path = './'
    for folder in path.split('/')[:-1]:
        if folder not in os.listdir(full_path):
            os.mkdir(full_path+folder+'/')
        full_path += folder+'/'
    json.dump(file, open(os.path.join(path, filename), 'w'))
    print(f'Json file {filename} saved successfully!')

def save_df(df, path=None, filename=None, index=True):
    full_path = './'
    for folder in path.split('/')[:-1]:
        if folder not in os.listdir(full_path):
            os.mkdir(full_path+folder+'/')
        full_path += folder+'/'
    df.to_csv(os.path.join(path, filename), index=index)

class preprocess:    
    def split_serie(serie, train_size, test_size):
        train = serie.iloc[:train_size]
        test = serie.iloc[train_size:train_size+test_size]
        xt, yt = train.index.values.reshape(-1, 1), train.values
        xe, ye = test.index.values.reshape(-1, 1), test.values
        return xt, yt, xe, ye

class SpecializedModels:
    names = ['AutoReg', 'ARIMA', 'SARIMAX', 'AutoArima', 'TBATS']
    spec_min_train_size = {'AutoReg': 3 , 'ARIMA': 2, 'SARIMAX': 2, 'AutoArima': 3,  'TBATS': 0}     

    def __init__(self):
        self.name='class to perform time series forecasting using specialized models'
    # functions to fit and predict using specialized models    
    def AutoReg_predict(train, x_min=42, x_max=43, lags=1):
        if len(train) <= 3: return np.array([np.nan for i in range(x_max-x_min+1)])
        model = AutoReg(train, lags=lags)
        # fit model
        model_fit = model.fit()
        # make prediction
        return model_fit.predict(x_min, x_max)
    def predict_arima(train, x_min=42, x_max=43, order=(1,1,1)):
        if len(train) <= 2: return np.array([np.nan for i in range(x_max-x_min+1)])
        # fit model
        model = ARIMA(train, order=order)
        try:
            model_fit = model.fit()
            # make prediction
            return model_fit.predict(x_min, x_max)
        except:
            return None
    def predict_SARIMAX(train, x_min, x_max, order=(1,1,1), seasonal_order=(0,0,0,0), exog_train=None, exog_test=None):
        if len(train) <= 2: return np.array([np.nan for i in range(x_max-x_min+1)])
        # fit model
        model = SARIMAX(train, exog=exog_train, order=order, seasonal_order=seasonal_order)
        try:
            model_fit = model.fit(disp=False)
        # make prediction
            return model_fit.predict(x_min, x_max, exog=exog_test)
        except:
            return None
    def predict_auto_arima(train, x_min, x_max):
        n_periods = x_max - x_min + 1
        if len(train) <= 3: return np.array([np.nan for i in range(x_max-x_min+1)])
        model = auto_arima(train, seasonal=False)
        try:
            return model.predict(n_periods=n_periods)
        except:
            return None
    def predict_tbats(train, x_min, x_max):
        steps = x_max - x_min + 1
        model_tbats = TBATS().fit(train)
        return model_tbats.forecast(steps=steps)

def fit_predict(model, xt, yt, xe):
    model_fit = model().fit(xt, yt)
    return model_fit.predict(xe)

class predict:
    
    def predict_by_model_name(model, model_name, xt, yt, xe):
        x_min = len(yt)
        x_max = x_min+len(xe)-1
        if model_name not in SpecializedModels.names:
            return fit_predict(model, xt, yt, xe)
        elif model_name=='AutoReg':
            return SpecializedModels.AutoReg_predict(yt, x_min, x_max)
        elif model_name=='ARIMA':
            return SpecializedModels.predict_arima(yt, x_min, x_max)
        elif model_name=='SARIMAX':
            return SpecializedModels.predict_SARIMAX(yt, x_min, x_max)
        elif model_name=='AutoArima':
            return SpecializedModels.predict_auto_arima(yt, x_min, x_max)
        elif model_name=='TBATS':
            return SpecializedModels.predict_tbats(yt, x_min, x_max)
    
class scoring:
    def __init__(self, scorers, criteria_map):
        self.scorers = scorers
        self.criteria_map = criteria_map
    def score(self, ye, yhat, metrics=['mae', 'mse', 'mape', 'wape', 'r2'], model_name='Predictive Model'):
        return pd.Series([self.scorers[metric](ye, yhat) for metric in metrics], index=metrics, name=model_name)
    def score_by_product(
        self, ye, yhat, indexes, metrics=['mae', 'mse', 'mape', 'wape', 'r2'], model_name='Predictive Model'
    ):
        prod_scrs = []
        for product in indexes.keys():
            prod_index = indexes[product]
            test_index = set(prod_index).intersection(yhat.index)
            if len(test_index):
                Ye, Yhat = ye.loc[test_index], yhat.loc[test_index]
                scrs = self.score(Ye, Yhat, metrics, model_name=product)
            else:
                scrs = pd.Series([np.nan]*len(metrics), index=metrics, name=product)
            prod_scrs.append(scrs)
        prod_scrs_df = pd.concat(prod_scrs, 1)
        prod_scrs_df.index.name = model_name
        return prod_scrs_df
# A mapping between each metric and a value representing wether the metric is optimized for lower or higher values (0 or -1).
criteria_map = {'e': 0, 'estd': 0, 'max_error': 0, 'mae': 0, 'mse': 0, 'medae': 0, 'mape': 0, 'wape': 0, 'r2': -1, 'evs': -1, 'mpe': 0}
# An object ssociating each metric with its name
scorers = {
    'e': e, 'estd': estd, 'max_error': me, 'mae': mae, 'mse': mse,
    'medae': medae, 'mape': mape, 'wape': wape, 'r2': r2, 'evs': evs, 'mpe': mpe
}
# An instance object of the scoring class containing scoring methods to be used be all scoring functions
Scoring = scoring(scorers, criteria_map)
    
class model:

    def __init__(
        self, models_names=None, series=None, keys=None,
        test_size=2, min_train_size=3,
        metrics=['mae', 'mse', 'mape', 'wape', 'r2', 'mpe'],
        regressors=dict(sklearn.utils.all_estimators('regressor')),
    ):
        self.regressors=regressors
        self.models_names=models_names
        self.series=series
        self.keys=keys
        self.test_size=test_size
        self.min_train_size=min_train_size
        self.actual_min_train_size=min_train_size
        self.metrics=metrics

    def score_model_for_serie(self, model, serie, train_size, model_name):
        xt, yt, xe, ye = preprocess.split_serie(serie, train_size, self.test_size)
        yhat = predict.predict_by_model_name(model, model_name, xt, yt, xe)
        if yhat is None: return None
        else: return Scoring.score(ye, yhat, self.metrics, model_name)

    def learning_curve(
        self, model, serie,
        model_name='Predictive Model',
    ):
        scores = []
        n = len(serie)
        train_size_i = range(self.min_train_size, n-self.test_size+1)
        for train_size in train_size_i:
            score = self.score_model_for_serie(model, serie, train_size, model_name)
            if score is None:
                score = pd.Series([np.nan]*len(self.metrics), index=self.metrics, name=model_name)
            scores.append(score)
        scores_df = pd.concat(scores, 1).T
        scores_df.index = train_size_i
        scores_df.columns.name = model_name
        scores_df.index.name = 'train size'
        return scores_df

    def plot_learning_curves(lc):
        ax = plt.figure(figsize=(5, 3*lc.shape[1]), tight_layout=True).add_subplot()
        lc.plot(subplots=True, ax=ax)
        plt.show()


    def models_lc(self, serie, verbose=1):
        self.min_train_size = self.actual_min_train_size + 0
        lc_i = []
        for i, model_name in enumerate(self.models_names):
            if model_name in SpecializedModels.names:
                model = None
                model_min_train = SpecializedModels.spec_min_train_size[model_name]
                if (len(serie) - self.test_size) <= model_min_train: continue
                else: self.min_train_size = model_min_train + 1
            else: model = self.regressors[model_name]
            lc_df = self.learning_curve(model, serie, model_name)
            lc_df['model'] = model_name
            lc_i.append(lc_df)
            if verbose: co(wait=True); print(f'Models scored: {i+1}/{len(self.models_names)}')
        return pd.concat(lc_i)

    def score_keys_models(self, path='scores/', filename='scores.csv', path_partial=None, verbose=0):
        if path is not None and path_partial is not None:
            try: os.mkdir(path+path_partial)
            except: None
        keys_models_scrs = []
        for i, key in enumerate(self.keys):
            serie = self.series[key].copy()
            if len(serie) < (self.min_train_size + self.test_size): continue
            models_scrs = self.models_lc(serie, verbose=0)
            models_scrs['key'] = key
            keys_models_scrs.append(models_scrs)
            if path is not None and path_partial is not None:
                try: os.mkdir(path+path_partial)
                except:
                    try: os.mkdir(path); os.mkdir(path+path_partial)
                    except: None
                models_scrs.to_csv(os.path.join(path+path_partial, key+'-'+filename), index=True)
            if verbose: co(wait=True); print(f"Keys scored: {i+1}/{len(self.keys)} - {filename}")
        
        keys_models_scrs = pd.concat(keys_models_scrs)
        if path is not None and filename is not None:
            try: os.mkdir(path)
            except: None
            keys_models_scrs.to_csv(os.path.join(path, filename), index=True)
        self.keys_models_scrs = keys_models_scrs
        return keys_models_scrs
    
def weight_average(df, order):
    matrix = df#.dropna()
    n_samples = len(matrix)
    weights = np.linspace(1/n_samples, 1, n_samples)**order
    return matrix.T.dot(weights)/sum(weights)

class model_selection:
    
    def __init__(self, criteria=None, min_train_size=None, weight_order=None, n_last=None):
        self.criteria = criteria
        self.min_train_size = min_train_size
        self.weight_order = weight_order
        self.n_last = n_last

    def set_params(self, criteria, min_train_size, weight_order, n_last):
        self.criteria = criteria
        self.min_train_size = min_train_size
        self.weight_order = weight_order
        self.n_last = n_last

    # Calculates learning curve scores averages by provided method
    def average_lc(self, lc_df):
        lc_df = lc_df.loc[self.min_train_size:].iloc[-self.n_last:].copy()
        if len(lc_df)==0:
            return pd.Series([np.nan]*lc_df.shape[1], index=lc_df.columns, name=lc_df.columns.name)
        avg_lc = weight_average(lc_df, self.weight_order)
        avg_lc.name = lc_df.columns.name
        return avg_lc

    def select_model(self, key_scrs):
        avg_lc_i = []
        models_names = key_scrs['model'].unique()
        for j, model_name in enumerate(models_names):
            model_scrs = key_scrs[key_scrs['model']==model_name].drop('model', 1).sort_index()
            avg_scrs = self.average_lc(model_scrs)
            avg_scrs.name = model_name
            avg_lc_i.append(avg_scrs)
        avg_lc_i = pd.concat(avg_lc_i, 1)
        top_model_name = avg_lc_i.loc[self.criteria].sort_values().index[Scoring.criteria_map[self.criteria]]
        return top_model_name
        
    def select_keys_models(
        self, scrs_path='scores/scores.csv', scrs=None,
        save_path='models/', filename=None
    ):
        if scrs is None: scrs = pd.read_csv(scrs_path, index_col=0)
        avg_scrs_i = []
        keys = scrs['key'].unique()
        keys_models = {}
        for i, key in enumerate(keys):            
            key_scrs = scrs[scrs['key']==key].drop('key', 1)
            keys_models[key] = self.select_model(key_scrs)
        
        if save_path is not None and filename is not None:
            save_json_file(keys_models, save_path, filename)
        return keys_models

    def concat_partial_scrs(path):
        scrs, files = [], os.listdir(path)
        for i, file in enumerate(files):
            co(wait=True); print(f'Files loaded: {i}/{len(files)}')
            scrs.append(pd.read_csv(os.path.join(path, file)))
        print(f'Done! {len(files)} score dataframes concatenated.')
        return pd.concat(scrs)

    def model_selection_optimization(
        self, params_dict,
        scrs_path='scores/scores.csv', scrs=None,
        save_path='models/', verbose=0,
    ):
        if scrs is None: scrs = pd.read_csv(scrs_path, index_col=0)
        params_keys_models = {}
        for i, params_key in enumerate(params_dict.keys()):
            if verbose==0: print(f'Evaluating parameters: {params_key} {i+1}/{len(params_dict)}'); co(wait=True)
            self.set_params(**params_dict[params_key])
            params_keys_models[params_key] = self.select_keys_models(
                scrs_path, scrs, save_path, filename=params_key+'.json'
            )
        if verbose==0: co(wait=True); print(f'Done! Saved {len(params_dict)} key-model maps.')
        return params_keys_models

class series_model:
    
    def __init__(
        self, series=None, keys=None, target=None,
        keys_models=None, model_path=None,
        regressors=dict(sklearn.utils.all_estimators('regressor'))
    ):
        if keys_models is None and model_path is not None:
            keys_models = json.load(open(model_path, 'r'))
            if keys is None:
                keys = list(keys_models.keys())      
        self.series=series
        self.keys_models=keys_models
        self.keys=keys
        self.target=target
        self.regressors=regressors

    def set_model(self, keys_models, keys):
        self.keys_models = keys_models
        self.keys = keys
        
    def predict_sequences(
        self, x_min=2, x_max=44, test_size=2,
        min_train_size=1, max_train_size=50,
        min_test_size=1, dropna=True, verbose=1,
    ):
        train_last_i = range(x_min-2, x_max-test_size)
        yhat_i = {train_last+2: [] for train_last in train_last_i}
        for i, key in enumerate(sorted(self.keys)):
            if verbose: co(wait=True); print(f'Keys predicted: {i}/{len(self.keys)}')
            serie = self.series[key].copy(); model_name = self.keys_models[key]
            if model_name in SpecializedModels.names: model = None
            else: model = self.regressors[model_name]
            for train_last in train_last_i:
                train = serie.loc[:train_last].iloc[-max_train_size:]
                test_min = train_last+1; test_max = train_last+test_size                
                test = serie.loc[test_min:test_max]
                xe = np.array(range(test_min, test_max+1)).reshape(-1, 1)
                true_index = [(test.loc[index]['index'] if index in test.index else np.nan) for index in xe.reshape(-1)]
                empty_scrs = pd.Series([np.nan]*test_size, index=true_index)
                if len(train) >= min_train_size and len(test) >= min_test_size:
                    xt, yt = train.index.values.reshape(-1, 1), train[self.target].values
                    yhat = predict.predict_by_model_name(model, model_name, xt, yt, xe)
                    if yhat is not None:
                        yhat_i[train_last+2].append(pd.Series(yhat, index=true_index))
                    else:
                        yhat_i[train_last+2].append(empty_scrs)
                else:
                    yhat_i[train_last+2].append(empty_scrs)
        for key in yhat_i.keys():
            yhat_i[key] = pd.concat(yhat_i[key])
            if dropna: yhat_i[key].dropna(inplace=True)
        return yhat_i
        
    #### Loading key-model maps
    def load_series_models(self, path='models/', filter_by=''):
        maps_filenames = filter(lambda filename: filter_by in filename, os.listdir(path))
        maps = {}
        for filename in maps_filenames:
            if '.json' in filename:
                maps[filename.split('.')[0]] = json.load(open(path+filename, 'r'))
        maps_keys = list(maps.keys())
        return maps, maps_keys

    #### Predicting with key-model map custom models
    def maps_predictions(
        self, x_min, x_max, test_size,
        min_train_size, max_train_size,
        min_test_size, dropna,
        path='models/', filter_by='',
        save_path='predictions/'  # change 'min_test_size' to zero and 'dropna' to False to predict future non-included test_samples (when n > n_max)
    ):
        maps, maps_keys = self.load_series_models(path, filter_by)
        map_models_predictions = {}
        for i, map_key in enumerate(maps_keys):
            co(wait=True); print(f'Predicting with model: {i+1}/{len(maps_keys)} - {map_key}')
            map_model = maps[map_key]
            model_keys = list(map_model.keys())
            self.set_model(map_model, model_keys)
            y_hat_i = self.predict_sequences(x_min, x_max, test_size, min_train_size, max_train_size, min_test_size, dropna, verbose=0)
            map_models_predictions[map_key] = y_hat_i
            if save_path is not None:
                folder_path = f'{save_path}{map_key}/'
                for key in y_hat_i.keys():
                    save_df(y_hat_i[key], folder_path, f'{key}.csv')
        return map_models_predictions

class sequence_scorer:
    def __init__(
        self, Ytrue=None, target=None, criteria='wape', avg=False,
        metrics=['mae', 'estd', 'max_error', 'mse', 'wape', 'r2'],
        indexes=None,
#         yhat_i=None, n=None, test_size=None,
    ):
        self.Ytrue=Ytrue
        self.target=target
        self.criteria=criteria
        self.avg=avg
        self.metrics=metrics
        self.indexes=indexes
    
    def load_csv_folder(self, path):
        dfs = {}
        files = os.listdir(path)
        for file in files:
            if '.csv' in file:
                dfs[int(file.split('.')[0])] = pd.read_csv(os.path.join(path, file), index_col=0)['0']
        return dfs

    def score_indexed_prediction_by_product(self, yhat, name='Custom predictive model'):
        test_index = set(yhat.dropna().index).intersection(self.Ytrue.index)
        Yhat = yhat.loc[test_index].copy()
        Ye = self.Ytrue.loc[test_index].copy()
        prod_scrs = Scoring.score_by_product(Yhat, Ye, self.indexes, self.metrics, name).loc[self.criteria]
        prod_scrs.name = name
        if self.avg: prod_scrs = prod_scrs.mean()
        return prod_scrs

    def score_prediction_sequences(self, yhat_i):
        train_last_scrs = []
        train_last_i = sorted(yhat_i.keys())
        for train_last in train_last_i:
            yhat = yhat_i[train_last].copy()
            train_last_scrs.append(self.score_indexed_prediction_by_product(yhat, name=train_last))
        yhat_i_scrs = pd.DataFrame(train_last_scrs)
        yhat_i_scrs.index.name = 'train last'
        yhat_i_scrs.columns.name = self.criteria
        return yhat_i_scrs

    def score_models_prediction_sequences(self, preds_path='predictions/', filter_by='.'):
        preds_files = [file for file in os.listdir(preds_path) if filter_by not in file]
        preds_scrs = {}
        for i, file in enumerate(preds_files): 
            co(wait=True); print(f'{i+1}/{len(preds_files)} - Loading file: {file}')
            yhat_i = self.load_csv_folder(preds_path+file+'/')
            preds_scrs[file] = self.score_prediction_sequences(yhat_i)
        return preds_scrs
    
    def n_last_stats(self, avg_scrs):
        avgs, es, stds = [{col: [] for col in avg_scrs} for i in [0,1,2]]
        for col in avg_scrs.columns:
            scrs = avg_scrs[col]
            cnt=0
            for i in reversed(range(1, len(scrs)+1)):
                scrs_i = scrs.iloc[-i:]
                mean_i = scrs_i.mean()
                e_i = np.abs(scrs_i - mean_i).sum()/len(scrs_i.dropna())
                std_i = scrs_i.std()
                cnt+=1
                avgs[col].append(mean_i)
                es[col].append(e_i)
                stds[col].append(std_i)
        return ( pd.DataFrame(dic, index=avg_scrs.index) for dic in [avgs, es, stds] )

    def plot_models_sequence_scrs(self, final_scrs, category=None, figsize=(10, 7), ylim=[(None, None) for i in range(4)], legend=False, leg_loc=(1, 0), leg_i=2):
        models_keys = list(final_scrs.keys())
        if category is None:
            scrs = pd.DataFrame({key: final_scrs[key].mean(1) for key in models_keys})
        else:
            scrs = pd.DataFrame({key: final_scrs[key][category] for key in models_keys})
        avg_scrs, e_scrs, std_scrs = self.n_last_stats(scrs)

        fig = plt.figure(figsize=figsize, tight_layout=True)
        ax = [fig.add_subplot(2,2,i) for i in [1,2,3,4]]
        scrs.plot(ax=ax[0])
        avg_scrs.plot(ax=ax[1])
        e_scrs.plot(ax=ax[2])
        std_scrs.plot(ax=ax[3])
        for method, i in zip(['', 'cum. avg.', 'cum. std.', 'cum. e.'], [0,1,2,3]):
            ax[i].set(
                title=f"Custom models' {method} performance (WAPE)\n at predicting 'n' last pair of years" + (f' - category: {category}' if category is not None else ' - categories average'),
                ylabel=f'{method.capitalize()} WAPE error',
                xlabel='"n" last pair of years',
                ylim=ylim[i],
            )
            if not legend: ax[i].legend([])
            if leg_i is not None and i==leg_i: ax[i].legend(loc=leg_loc)
        plt.show()

    def rebuild_prediction_series(self, yhat_i, model, base_year=1974):
        pred_series_first, pred_series_second = [], []
        first_test_i = np.array(sorted(yhat_i.keys()))
        n_keys = len(model)
        pair_index = list(range(0, int(n_keys*2), 2))
        even_index = list(range(1, int(n_keys*2), 2))
        for first_test in first_test_i:
            yhat = yhat_i[first_test].copy()
            year_1 = yhat.iloc[pair_index].copy()
            year_2 = yhat.iloc[even_index].copy()
            pred_series_first.append(year_1.values)
            pred_series_second.append(year_2.values)
        return (
            pd.DataFrame(values, columns=sorted(model.keys()), index=first_test_i+base_year+i-1) for i, values in enumerate([pred_series_first, pred_series_second])
        )

    def plot_random_pred( self,
        X, model,
        yhat_i, base_year=1974,
        exclude=[], n_series=12, n_cols=3, figsize=[5, 3],
        X_params={'marker': 'o', 'ms': 3, 'lw': 5},
        X0_params={'marker': 'x', 'ms': 5},
        X1_params={'marker': 'x', 'ms': 5},
        save_path='plots/',
        filename='predictions.jpg'
    ):
        keys = list(model.keys())
        X0, X1 = self.rebuild_prediction_series(yhat_i, model, base_year)
        X = X.copy(); X.index = X.index.values + base_year
        sample_keys = np.random.choice(list(filter(lambda key: model[key] not in exclude, keys)), n_series, replace=False)
        n_rows = n_series//n_cols if n_series%n_cols==0 else n_series//n_cols+1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0]*n_cols, figsize[1]*n_rows), tight_layout=True)
        row, col = 0, 0
        for key in sample_keys:
            X[key].plot(ax=axes[row][col], **X_params, label='real')
            if X0 is not None: X0[key].plot(ax=axes[row][col], label='one step pred', **X0_params)
            if X1 is not None: X1[key].plot(ax=axes[row][col], label='two step pred', **X1_params)
            axes[row][col].set(
                title=f'Real vs predicted land areas in Pará cities\n{key}\n{model[key]}',
                ylabel='land area (hec)',
                xlabel='years'
            ); axes[row][col].legend()
            col+=1
            if col==n_cols:
                col = 0; row+=1
        if save_path is not None and filename is not None: 
            try: os.mkdir(save_path)
            except: None
            plt.savefig(os.path.join(save_path, filename))
        plt.show()
