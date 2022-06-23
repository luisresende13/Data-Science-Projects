import os
import pandas as pd 
import numpy as np; np.random.seed(25486)
import matplotlib.pyplot as plt
from IPython.display import clear_output as co
import json
import time

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
from pmdarima.arima import auto_arima
from tbats import TBATS

class scoring:
    
    def __init__(self, scorers, criteria_map):
        self.scorers = scorers
        self.criteria_map = criteria_map

    def score(self, ye, yhat, metrics=['mae', 'mse', 'mape', 'wape', 'r2'], model_name='Predictive Model'):
        return pd.Series([self.scorers[metric](ye, yhat) for metric in metrics], index=metrics, name=model_name)    
    
    def score_by_product(self, ye, yhat, indexes, metrics=['mae', 'mse', 'mape', 'wape', 'r2'], model_name='Predictive Model'):
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

class TSmodeling:

    def __init__(self, Scoring):
        self.Scoring = Scoring

    def split_serie(self, serie, train_size, test_size):
        train = serie.iloc[:train_size]
        test = serie.iloc[train_size : train_size+test_size]
        xt, yt = train.index.values.reshape(-1, 1), train.values
        xe, ye = test.index.values.reshape(-1, 1), test.values
        return xt, yt, xe, ye

    def fit_predict(self, model, xt, yt, xe):
        model_fit = model().fit(xt, yt)
        return model_fit.predict(xe)

    def predict_by_model_name(self, model, model_name, xt, yt, xe, specialized_models):
        x_min = len(yt)
        x_max = x_min+len(xe)-1
        if model_name not in specialized_models:
            return self.fit_predict(model, xt, yt, xe)
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

    def score_model_for_serie(self, model, serie, train_size, test_size, metrics, model_name, specialized_models):
        xt, yt, xe, ye = self.split_serie(serie, train_size, test_size)
    #     yhat = fit_predict(model, xt, yt, xe)
        yhat = self.predict_by_model_name(model, model_name, xt, yt, xe, specialized_models)
        if yhat is None:
            return None
        else:
            return self.Scoring.score(ye, yhat, metrics, model_name)

    def learning_curve(
        self, model, serie, test_size=2, min_train_size=1,
        metrics=['mae', 'mse', 'mape', 'wape', 'r2'],
        model_name='Predictive Model',
        specialized_models=['AutoReg', 'ARIMA', 'SARIMAX', 'AutoArima']
    ):
        scores = []
        n = len(serie)
        train_size_i = range(min_train_size, n-test_size+1)
        for train_size in train_size_i:
            score = self.score_model_for_serie(model, serie, train_size, test_size, metrics, model_name, specialized_models)
            if score is None:
                score = pd.Series([np.nan]*len(metrics), index=metrics, name=model_name)
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

    # Calculates learning curve scores averages by provided method
    def average_lc(self, lc_df,  avg_method='simple', min_train_size=None, weight_order=None, n_last=None):
        if avg_method=='simple':
            avg_lc = lc_df.mean()
        elif avg_method=='min_train_size':
            avg_lc = lc_df.loc[min_train_size:].mean()
        elif avg_method=='n_last':
            avg_lc = lc_df.iloc[-n_last:].mean()
        elif avg_method=='weighted':
            n = len(lc_df)
            weights = np.linspace(1/n, 1, n)**weight_order
            avg_lc = lc_df.T.dot(weights)/sum(weights)
        elif avg_method=='train_size_weighted':
            weights = lc_df.index.values**weight_order
            avg_lc = lc_df.T.dot(weights)/sum(weights)
        avg_lc.name = lc_df.columns.name
        return avg_lc

    def models_lc(
        self, models_names, regressors, serie, test_size=2, min_train_size=1,
        metrics=['mae', 'mse', 'mape', 'wape', 'r2'],
        avg_method='simple', avg_min_train_size=None, weight_order=None, avg_n_last=None, verbose=1,
        specialized_models=['AutoReg', 'ARIMA', 'SARIMAX', 'AutoArima'],
        spec_min_train_size = {'AutoReg': 3 , 'ARIMA': 2, 'SARIMAX': 2, 'AutoArima': 3,  'TBATS': 0}
    ):
        default_min_train_size = min_train_size + 0
        lc_i, avg_lc_i = [], []
        n_models = len(models_names)
        times = {}
        for i, model_name in enumerate(models_names):
            if model_name in specialized_models:
                if (len(serie) - test_size) <= spec_min_train_size[model_name]:
                    continue
                else:
                    min_train_size = spec_min_train_size[model_name] + 1
                model = None
            else:
                min_train_size = default_min_train_size
                model = regressors[model_name]
            start = time.time()
            lc_df = self.learning_curve(model, serie, test_size, min_train_size, metrics, model_name, specialized_models)
            avg_lc_i.append(self.average_lc(lc_df, avg_method, avg_min_train_size, weight_order, avg_n_last))
            end = time.time()

            times[model_name] = (end - start)

            lc_df['model'] = model_name
            lc_i.append(lc_df)
            
            if verbose: co(wait=True); print(f'Models scored: {i+1}/{n_models}')
        return pd.concat(avg_lc_i, 1), pd.concat(lc_i), times

    def select_keys_models(
        self, models_names, regressors, series, keys,
        criteria, test_size=2, min_train_size=1,
        metrics=['mae', 'mse', 'mape', 'wape', 'r2'],
        avg_method='simple', avg_min_train_size=None,
        weight_order=None, avg_n_last=None, verbose=0,
        path='models/', scrs_path='scores/', model_name='key_model_map_unnamed',
        specialized_models=['AutoReg', 'ARIMA', 'SARIMAX', 'AutoArima'],
        spec_min_train_size = {'AutoReg': 3, 'ARIMA': 2, 'SARIMAX': 2, 'AutoArima': 3,  'TBATS': 0}
    ):
        keys_models = {}
        keys_models_scrs = {}
        keys_models_scrs_full = []
        for i, key in enumerate(keys):
            serie = series[key].copy()
            if len(serie) < (min_train_size + test_size): continue
            key_models_lc, models_scrs, models_times = self.models_lc(
                models_names, regressors, serie, test_size, min_train_size,
                metrics, avg_method, avg_min_train_size, weight_order, avg_n_last,
                0, specialized_models, spec_min_train_size
            )
            top_model_name = key_models_lc.loc[[criteria]].T.sort_values(criteria).iloc[self.Scoring.criteria_map[criteria]].name
            keys_models[key] = top_model_name
            keys_models_scrs[key] = key_models_lc
            models_scrs['key'] = key
            keys_models_scrs_full.append(models_scrs)
            if verbose: co(wait=True); print(f"Keys scored: {i+1}/{len(keys)} - {model_name}")
        
        keys_models_scrs_full = pd.concat(keys_models_scrs_full)
        #### Saving key-model map
        if path is not None and model_name is not None:
            try: os.mkdir(path)
            except: None
            json.dump(keys_models, open(os.path.join(path, model_name+'.json'), 'w'))
        if scrs_path is not None and model_name is not None:
            try: os.mkdir(scrs_path)
            except: None
            keys_models_scrs_full.to_csv(os.path.join(scrs_path, model_name+'.csv'), index=True)
        return keys_models, keys_models_scrs, keys_models_scrs_full

    def model_selection_params_optimization(
        self, models_names, regressors, params_dict, 
        lab_series, keys, verbose=0,
        path='models/', scrs_path='scores/'
    ):
        if verbose==0: print('MODEL SELECTION PARAMS OPTIMIZATION'); print()
        key_model_maps_dict, key_model_maps_scrs, key_model_maps_scrs_full = {}, {}, {}
        for i, params_key in enumerate(params_dict.keys()):
            if verbose==0: print(f'Evaluating parameters: {params_key} {i+1}/{len(params_dict)}')
            key_model_maps_dict[params_key], key_model_maps_scrs[params_key], key_model_maps_scrs_full[params_key] = self.select_keys_models(
                models_names, regressors, lab_series, keys,
                **params_dict[params_key], verbose=verbose,
                path=path, scrs_path=scrs_path, model_name=params_key
            )
        if verbose==0: co(wait=True); print(f'Done! Saved {len(params_dict)} key-model maps.')
        return key_model_maps_dict, key_model_maps_scrs, key_model_maps_scrs_full

    def predict_custom_model_test_sequences(
        self, series, keys_models, keys, regressors, target,
        n_min=0, n=44, test_size=2,
        min_train_size=1, min_test_size=1, dropna=True,
        specialized_models=['AutoReg', 'ARIMA', 'SARIMAX', 'AutoArima', 'VAR', 'TBATS'],
        out_names = ['VAR', 'TBATS'],
        old_names_dict = {'Yhat_AutoReg': 'AutoReg', 'Yhat_arima': 'ARIMA', 'Yhat_sarimax': 'SARIMAX', 'Yhat_var': 'VAR', 'Yhat_autoarima': 'AutoArima', 'Yhat_tbats': 'TBATS'}
    ):
        train_last_i = range(n_min, n-test_size)
        yhat_i = {train_last+1: [] for train_last in train_last_i}
        for i, key in enumerate(keys):
            co(wait=True); print(f'Keys predicted: {i}/{len(keys)}')
            serie = series[key].copy()
            model_name = keys_models[key]
            if model_name in list(old_names_dict.keys()): model_name = old_names_dict[model_name]
            if model_name in specialized_models:
                model = None
                if model_name in out_names: model_name = 'ARIMA'
            else:
                model = regressors[model_name]
            for train_last in train_last_i:
                test_min = train_last+1
                test_max = train_last+test_size                
                train = serie.loc[:train_last]
                test = serie.loc[test_min:test_max]
                xe = np.array(range(test_min, test_max+1)).reshape(-1, 1)
                true_index = [(test.loc[index]['index'] if index in test.index else np.nan) for index in xe.reshape(-1)]
                empty_scrs = pd.Series([np.nan]*test_size, index=true_index)
                if len(train) >= min_train_size and len(test) >= min_test_size:
                    xt, yt = train.index.values.reshape(-1, 1), train[target].values
                    yhat = self.predict_by_model_name(model, model_name, xt, yt, xe, specialized_models)
                    if yhat is not None:
                        yhat_i[train_last+1].append(pd.Series(yhat, index=true_index))
                    else:
                        yhat_i[train_last+1].append(empty_scrs)
                else:
                    yhat_i[train_last+1].append(empty_scrs)
        for key in yhat_i.keys():
            yhat_i[key] = pd.concat(yhat_i[key])
            if dropna: yhat_i[key].dropna(inplace=True)
        return yhat_i

    def score_indexed_prediction_by_product(
        self, yhat, data, target, indexes, criteria='wape', avg=True,
        metrics=['mae', 'estd', 'max_error', 'mse', 'wape', 'r2'],
        model_name='Custom predictive model'
    ):
        test_index = set(yhat.dropna().index).intersection(data.dropna(subset=[target]).index)
        Yhat = yhat.loc[test_index].copy()
        Ye = data[target].loc[test_index].copy()
        prod_scrs = self.Scoring.score_by_product(Yhat, Ye, indexes, metrics, model_name).loc[criteria]
        prod_scrs.name = model_name
        if avg: prod_scrs = prod_scrs.mean()
        return prod_scrs

    def weight_average(self, df, order):
        matrix = df.dropna()
        n_samples = len(matrix)
        weights = np.linspace(1/n_samples, 1, n_samples)**order
        return matrix.T.dot(weights)/sum(weights)

    #### Loading key-model maps
    def load_key_model_maps(self, path='models/', filter_by=''):
        maps_filenames = filter(lambda filename: filter_by in filename, os.listdir(path))
        maps = {}
        for filename in maps_filenames:
            if '.json' in filename:
                maps[filename.split('.')[0]] = json.load(open(path+filename, 'r'))
        maps_keys = list(maps.keys())
        return maps, maps_keys

    #### Predicting with key-model map custom models
    def maps_predictions(
        self, lab_ind_series, regressors, target, n_min, n, test_size,
        min_train_size, min_test_size, dropna,
        path='models/', filter_by='', save_path='predictions/'  # change 'min_test_size' to zero and 'dropna' to False to predict non-included test_samples (when n > n_max)
    ):
        maps, maps_keys = self.load_key_model_maps(path, filter_by)
        map_models_predictions = {}
        for map_key in maps_keys:
            map_model = maps[map_key]
            sel_keys = list(map_model.keys())
            y_hat_i = self.predict_custom_model_test_sequences(
                lab_ind_series, map_model, sel_keys, regressors, target,
                n_min, n, test_size, min_train_size,
                min_test_size, dropna # change 'min_test_size' to zero and 'dropna' to False to predict non-included test_samples (when n > n_max)
            )
            if save_path is not None:
                folder_path = f'{save_path}{map_key}/'
                try: os.mkdir(save_path)
                except: None
                if map_key not in os.listdir(save_path): os.mkdir(folder_path)
                for key in y_hat_i.keys():
                    y_hat_i[key].to_csv(os.path.join(folder_path, f'{key}.csv'), index=True)
            map_models_predictions[map_key] = y_hat_i
        return map_models_predictions

    def load_csv_folder(self, path):
        dfs = {}
        files = os.listdir(path)
        for file in files:
            if '.csv' in file:
                dfs[int(file.split('.')[0])] = pd.read_csv(os.path.join(path, file), index_col=0)['0']
        return dfs

    def score_prediction_sequences(
        self, yhat_i, data, indexes, target='area',
        n=44, test_size=2, eval_criteria='wape', avg=False,
        metrics=['mae', 'estd', 'max_error', 'mse', 'wape', 'r2']
    ):
        train_last_scrs = []
        train_last_i = range(1, n-test_size+1)
        for train_last in train_last_i:
            yhat = yhat_i[train_last].copy()
            train_last_scrs.append(self.score_indexed_prediction_by_product(
                yhat, data, target, indexes, eval_criteria, avg,
                metrics, train_last
            ))
        yhat_i_scrs = pd.DataFrame(train_last_scrs)
        yhat_i_scrs.index.name = 'train last'
        yhat_i_scrs.columns.name = eval_criteria
        return yhat_i_scrs

    def score_final_models(
        self, data, prodtype_indexes, target, n,
        test_size, eval_criteria, avg, metrics,
        preds_path='predictions/', filter_by='.'
    ):
        preds_files = [file for file in os.listdir(preds_path) if filter_by not in file]
        preds_scrs = {}
        for i, file in enumerate(preds_files): 
            print(f'{i+1}/{len(preds_files)} - Loading file: {file}'); co(wait=True)
            yhat_i = self.load_csv_folder(preds_path+file+'/')
            preds_scrs[file] = self.score_prediction_sequences(
                yhat_i, data, prodtype_indexes, target,
                n, test_size, eval_criteria, avg,
                metrics
            )
        return preds_scrs

    def plot_scrs_stats(
        models_scrs, models_keys,
        params_0 = [0, 1, 3],
        params_1 = [2],
        method='std', #'mean'
        title="Custom models' average performance (WAPE)\n at predicting 'n' last pair of years",
        ylabel='Average WAPE',
        xlabel='"n" last pair of years',
        save_path=None
    ):
        fig = plt.figure(figsize=(10, 4), tight_layout=True)
        axes = fig.subplots(1, 2)
        n_max = len(models_scrs[models_keys[0]])
        avg_scrs_top = []
        avg_scrs_bottom = []
        for n_last_pred in list(range(n_max)):
            if method=='value':
                avg_scrs_top.append(
                    {key: models_scrs[key].mean(1).iloc[n_last_pred] for key in models_keys[params_0]}
                )
                avg_scrs_bottom.append(
                    {key: models_scrs[key].mean(1).iloc[n_last_pred] for key in models_keys[params_1]}
                )

            if method=='mean':
                avg_scrs_top.append(
                    {key: models_scrs[key].mean(1).iloc[n_last_pred:].mean() for key in models_keys[params_0]}
                )
                avg_scrs_bottom.append(
                    {key: models_scrs[key].mean(1).iloc[n_last_pred:].mean() for key in models_keys[params_1]}
                )
            if method=='std':
                avg_scrs_top.append(
                    {key: models_scrs[key].mean(1).iloc[n_last_pred:].std() for key in models_keys[params_0]}
                )
                avg_scrs_bottom.append(
                    {key: models_scrs[key].mean(1).iloc[n_last_pred:].std() for key in models_keys[params_1]}
                )
        pd.DataFrame(avg_scrs_top).plot(ax=axes[0])
        pd.DataFrame(avg_scrs_bottom).plot(ax=axes[1])
        axes[0].set(
            title=title,
            ylabel=ylabel,
            xlabel=xlabel
        )
        axes[1].set(
            title=title,
            ylabel=ylabel,
            xlabel=xlabel
        )
        if save_path is not None:
            try: os.mkdir(''.join(save_path.split('/')[:-1]))
            except: None
            plt.savefig(save_path)
        plt.show()

    def rebuild_prediction_series(yhat_i, models_keys, base_year=1974):

        pred_series_first, pred_series_second = [], []
        n_keys = len(models_keys)
        pair_index = list(range(0, n_keys*2, 2))
        even_index = list(range(1, n_keys*2, 2))
        last_train_max = max(yhat_i.keys())
        last_train_i = np.array(range(1, last_train_max+1))
        for last_train in last_train_i:
            yhat = yhat_i[last_train].copy()
            year_1 = yhat.iloc[pair_index].copy()
            year_2 = yhat.iloc[even_index].copy()
            pred_series_first.append(year_1.values)
            pred_series_second.append(year_2.values)

        X_pred_0 = pd.DataFrame(pred_series_first, columns=models_keys, index=last_train_i+base_year)
        X_pred_1 = pd.DataFrame(pred_series_second, columns=models_keys, index=last_train_i+base_year+1)
        return X_pred_0, X_pred_1

    def plot_random_pred(
        X, keys, models, X_pred_0=None, X_pred_1=None,
        exclude=[], n_series=12, n_cols=3, figsize=[5, 3],
        x_params={'marker': 'o', 'ms': 3, 'lw': 5},
        pred_0_params={'marker': 'x', 'ms': 5},
        pred_1_params={'marker': 'x', 'ms': 5},
    ):

        sample_keys = np.random.choice(list(filter(lambda key: models[key] not in exclude, keys)), n_series)
        n_rows = n_series//n_cols if n_series%n_cols==0 else n_series//n_cols+1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0]*n_cols, figsize[1]*n_rows), tight_layout=True)
        row, col = 0, 0
        for key in sample_keys:
            X[key].plot(ax=axes[row][col], **x_params)
            if X_pred_0 is not None: X_pred_0[key].plot(ax=axes[row][col], **pred_0_params)
            if X_pred_1 is not None: X_pred_1[key].plot(ax=axes[row][col], **pred_1_params)
            axes[row][col].set(
                title=f'Real vs predicted land areas in Para cities\n{models[key]}\n{key}',
                ylabel='land area (hec)',
                xlabel='years'
            )
            col+=1
            if col==n_cols:
                col = 0; row+=1
        plt.show()

class SpecializedModels:
    
    def __init__(self_):
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