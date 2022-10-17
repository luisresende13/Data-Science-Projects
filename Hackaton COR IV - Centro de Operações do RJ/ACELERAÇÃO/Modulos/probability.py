import pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler as mms
from sklearn.metrics import classification_report as cr, precision_recall_curve
from IPython.display import clear_output as co

def predict_proba(model, xe):
    try: yprob = model.predict_proba(xe)[:, 1]; print('predict_proba method used.')
    except:
        try: yprob = model.decision_function(xe); print('decision_function method used.')
        except: yprob = model.predict(xe); print('predict method used.')
    return pd.Series(yprob, index=xe.index)

def scale_proba(yprob, threshold=0.0, limit=None):
    yprob = yprob.copy()
    if limit is not None:
        yprob[yprob < limit[0]] = limit[0]
        yprob[yprob > limit[1]] = limit[1]
    msk = yprob >= threshold
    if not msk.all():
        yprob[~msk] = mms((0, 0.49999999)).fit_transform(yprob[~msk].to_frame()).reshape(-1)
    if not msk.any():
        yprob[msk] = mms((0.5, 1)).fit_transform(yprob[msk].to_frame()).reshape(-1)
    return yprob

# Classification report for test probabilities for given threshold
def clf_score(ye, yprob, threshold=0.5):
    yhat = (yprob > threshold).astype('int')
    scr = pd.DataFrame(cr(ye, yhat, digits=4, output_dict=True)).T
    return scr

# Precision-recall curve plot for test probabilities for given threshold
def precision_recall_plot(ye, yprob, thresh_lim=None, recall_lim=None):
    curve = pd.DataFrame(
        precision_recall_curve(ye, yprob, pos_label=1),
        index=['precision', 'recall', 'threshold']
    ).T.set_index('threshold').add_suffix(f' - 1')
    curve['f1 - 1'] = curve.mean(1)
    prec, rec = curve['precision - 1'], curve['recall - 1']
    curve['harmonic mean - 1'] = 2 * prec * rec / (prec + rec)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
    curve.plot(ax=ax[0]); curve.reset_index().plot('recall - 1', ['precision - 1', 'f1 - 1', 'harmonic mean - 1'], ax=ax[1])
    ax[0].set(title='Precision-Recall by Threshold', xlim=thresh_lim); ax[1].set(title='Precision-Recall Curve', xlim=recall_lim)
    return ax

def groups_windows(groups, spread=6, freq=pd.Timedelta(1, 'h')):
    windows = []; wide = spread * freq
    for group in groups.unique():
        group_index = groups.index[groups==group]
        grp_min, grp_max = group_index.min(), group_index.max()
        windows.append((grp_min - wide, grp_max + wide))
    return windows

def window_prob(ye, yprob, time_lim, ax=None):
    yprob = pd.Series(mms().fit_transform(yprob.to_frame()).reshape(-1), index=yprob.index) # scale probability to 0-1 range
    msk = ye.index.to_series().between(*time_lim) # time window limits
    if ax is None: ax = plt.axes()
    yprob[msk].plot(ax=ax)
    ax = ye[msk].plot(ax=ax)
    return ax

def multi_window_prob(ye, yprob, windows, n_cols, title='Probability {} - {}', path=None):
    n_plots = len(windows)
    n_rows = int(n_plots / n_cols if n_plots % n_cols == 0 else n_plots // n_cols + 1)
    figsize = (6 * n_cols, 4 * n_rows)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, tight_layout=True, sharey=True)
    axs = list(axs.reshape(-1))
    i=0
    for ax, time_lim in zip(axs, windows):
        i+=1; co(wait=True); print(f'{i}/{len(windows)}')
        msk = ye.index.to_series().between(time_lim[0], time_lim[1]) # time window limits
        yprob[msk].plot(ax=ax)
        ye[msk].plot(ax=ax)
        time_min, time_max = (time_lim[j].strftime('%d-%h-%y %H:%m') for j in (0, 1))
        ax.set(title=title.format(time_min, time_max))
    if path is not None: plt.savefig(path)
    return fig, axs