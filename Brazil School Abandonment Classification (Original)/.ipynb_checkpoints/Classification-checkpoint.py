import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from IPython.display import clear_output as co
from sklearn.utils import all_estimators
from sklearn.model_selection import cross_val_score as cvs, cross_validate as CV
from sklearn.metrics import accuracy_score as acc, precision_score as ps, r2_score as r2, recall_score as rs
classifiers = dict(all_estimators('classifier'))
metrics = {'accuracy': acc, 'precision': ps, 'recall': rs}

class Classifier:
    
    def __init__(self):
        self.models = classifiers
        self.names = list(classifiers.keys())
        self.metrics = metrics
        
    def binary_target(series, class_0, name, invert=False):
        Y = series.apply(lambda clss: int(clss not in class_0)) # 0 for classes in 'class_0' parameter
        if invert: Y = Y.apply(lambda x: int(not x))
        Y.name = name
        return Y

    def scr(self, model, x, y, xe, ye, scoring=['accuracy']):
        if type(model)==str:
            co(wait=True); print(model)
            yp = self.models[model]().fit(x.values, y).predict(xe)
            return pd.Series([metrics[name](yp, ye) for name in scoring], index=scoring, name=model)
        elif type(model)==list:
            return pd.concat([self.scr(mdl, x, y, xe, ye, scoring) for mdl in model], 1).T
        
    def train_size_scr(self, model, x, y, xe, ye, batch_size=None, n=10, order=1, scoring=['accuracy']):
        batch_scrs = []
        
        if batch_size is not None:
            size_i = np.array(range(batch_size, len(y)+batch_size, batch_size))
        else:
            size_i = len(y) * np.linspace(1/n, 1, n)**order
        for size in size_i:
            xi, yi = x.iloc[:int(size)].copy(), y.iloc[:int(size)].copy()
            scr = self.scr(model, xi, yi, xe, ye, scoring);
            scr['train size'] = len(yi)
            batch_scrs.append(scr)
        return pd.concat(batch_scrs, axis=(0 if type(model)==list else 1))    

    def cv_avg(self, model, _x, _y, cv, scoring):
        if type(model)==list:
            return pd.concat([self.cv_avg(model, _x, _y, cv, scoring) for model in model])
        else:
            mdl = self.models[model]
            cv_scrs = CV(mdl(), _x, _y, cv=cv, scoring=scoring)
            return pd.DataFrame(
                [[cv_scrs[f'test_{metric}'].mean(), cv_scrs[f'test_{metric}'].var(), model] for metric in scoring],
                index=scoring,
                columns=['mean', 'var', 'model']
            )

    def cv_scr(self, models, x, y, cv, describe=True):
        if type(models)==list: return pd.concat([self.cv_scr(model, x, y, cv, describe) for model in models], 1)
        else:
            model = self.models[models]()
            cv_scrs = cvs(model, x, y, cv=cv)
            scr_stts = pd.Series(cv_scrs)
            if describe: scr_stts = scr_stts.describe()
            scr_stts.name = models
            return scr_stts

    def learning_curve(self, model, x, y, batch_size=25, cv=5, xe=None, ye=None, metric='mean'):
        batch_scrs = []
        if type(model)==str:
            batch_scrs = []
            for size in range(batch_size, len(y)+batch_size+1, batch_size):
                xi, yi = x.iloc[:size].copy(), y.iloc[:size].copy()
                scr = self.cv_scr(model, xi, yi, cv); scr.name = size
                batch_scrs.append(scr)
            scrs_df = pd.concat(batch_scrs, 1)
            scrs_df.columns.name = model; scrs_df.index.name='train size'
            return scrs_df
        else:
            for modelname in model:
                batch_scrs.append(self.learning_curve(modelname, x, y, batch_size, cv ,xe, ye).loc[metric].rename(modelname))
            return pd.concat(batch_scrs, 1)

def plot_lc_model_comparison(
    train_size_scrs, metric='accuracy',
    avg_window=7, min_periods=1, center=False,
    figsize=(9, 4.5), legend=True, leg_loc=(1.05, 0),
    title='Model Learning Curve', tight_layout=True, path=None
):
    fig, ax = plt.subplots(figsize=figsize, tight_layout=tight_layout)
    scrs = train_size_scrs.reset_index().set_index('train size')
    metric_scrs = pd.concat([scrs[scrs['index']==model][metric].rename(model) for model in scrs['index'].unique()], 1)
    if avg_window > 1: metric_scrs.rolling(avg_window, min_periods=min_periods, center=center).mean().plot(ax=ax)
    else: metric_scrs.plot(ax=ax)
    ax.set(title=title, ylabel=metric)
    ax.legend(loc=leg_loc)
    if not legend: plt.legend([])
    plt.show()
    if path is not None: plt.savefig(path)
