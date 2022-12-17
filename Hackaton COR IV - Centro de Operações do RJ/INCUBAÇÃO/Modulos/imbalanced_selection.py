import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RepeatedKFold, cross_validate
from itertools import product
from IPython.display import clear_output as co
from Modulos.probability import predict_proba, scale_proba


#### Metrics and scoring functions
from sklearn.metrics import (
    make_scorer, recall_score, precision_score, f1_score
)

recall_0 = make_scorer(recall_score, pos_label=0, zero_division=1)
recall_1 = make_scorer(recall_score, pos_label=1, zero_division=1)
precision_0 = make_scorer(precision_score, pos_label=0, zero_division=1)
precision_1 = make_scorer(precision_score, pos_label=1, zero_division=1)
f1_0 = make_scorer(f1_score, pos_label=0, zero_division=1)
f1_1 = make_scorer(f1_score, pos_label=1, zero_division=1)

scoring = {
    'accuracy': 'accuracy', 'recall': 'recall',
    'precision': 'precision', 'f1': 'f1',
    'recall-0': recall_0, 'recall-1': recall_1,
    'precision-0': precision_0, 'precision-1': precision_1,
    'f1-0': f1_0, 'f1-1': f1_1
}

# Label consecutive flags groups
def groupConsecutiveFlags(ts, outlier_mark=-1):
    groups = []
    cur_grp, prev_label = -1, 0
    for label in ts:
        if label:
            if label != prev_label: cur_grp += 1
            groups.append(cur_grp)
        else:
            groups.append(outlier_mark)
        prev_label = label
    return pd.Series(groups, index=ts.index, name='group')

def cross_val_predict_proba(estimator, X, Y, cv, calibrate=None):
    yprob_cv = []
    for i, (train, test) in enumerate(cv):
        try:
            estimator.fit(X.iloc[train], Y.iloc[train])
            yprob = predict_proba(estimator, X.iloc[test])
            # Optional scale probabilities
            if calibrate is not None:
                yprob = scale_proba(yprob, threshold=calibrate, limit=None)
            yprob_cv.append(yprob.values)
        except Exception as e:
            yprob_cv.append(np.array([np.nan] * len(test)))
            print('cross_val_predict_proba error:', e)
        co(True); print(f'cv: {i+1}/{len(cv)}')
    return yprob_cv

class group_metrics:

    def group_recall(y, yhat, grps, metric='recall-1'):
        grp_df = []
        for group in grps.unique():
            if group != -1:
                msk = grps==group
                grp_acc = (y[msk] == yhat[msk]).mean()
                grp_df.append([group, grp_acc, int(grp_acc > 0.0)])        
        return pd.DataFrame(grp_df, columns=['label', metric, metric + ' group']).set_index('label')


    def group_precision_recall_stats(ye, yhat, groups, groups_hat):

        grp_df = group_metrics.group_recall(ye, yhat, groups)
        pred_df = group_metrics.group_recall(ye, yhat, groups_hat, metric='precision-1')

        print('Target groups:', grp_df.shape[0], '\n')
        display(grp_df.mean())
        print('\nPrediction groups:', pred_df.shape[0], '\n')
        display(pred_df.mean())

        fig, ax = plt.subplots(1, 2, figsize=(11, 3))
        ax[0] = grp_df['recall-1'].plot.hist(ax=ax[0], title='Group Recall')
        ax[1] = pred_df['precision-1'].plot.hist(ax=ax[1], title='Group Precision')
        return ax

    def group_precision_recall(y, yhat, groups_y, groups_hat):

        ### Inside groups
        grp_df = group_metrics.group_recall(y, yhat, groups_y).mean()
        grp_df['support-1'] = len(groups_y.unique()) - 1

        ### Inside positive predictions
        pred_df = group_metrics.group_recall(y, yhat, groups_hat, metric='precision-1').mean()
        pred_df['support-1 group'] = len(groups_hat.unique()) - 1

        return (df.add_suffix(' avg') for df in (grp_df, pred_df))

    def group_precision_recall_curve(y, yprob, groups_y, num=20):

        thresholds = np.linspace(yprob.min(), yprob.max(), num)
        grp_tunning, pred_tunning = [], []
        for thresh in thresholds:

            #### Probability prediction and groups
            yhat = (yprob > thresh).astype('int')
            groups_hat = groupConsecutiveFlags(ts=yhat)

            grp_df, pred_df = group_metrics.group_precision_recall(y, yhat, groups_y, groups_hat)

            grp_tunning.append(pd.concat([grp_df, pred_df], axis=0))

        grp_curve = pd.DataFrame(grp_tunning, index=thresholds)
        grp_curve['f1-1 avg'] = grp_curve[['precision-1 avg', 'recall-1 avg']].mean(1)
        grp_curve['f1-1 group avg'] = grp_curve[['precision-1 group avg', 'recall-1 group avg']].mean(1)

        return grp_curve

    def group_precision_recall_plot(y, yprob, groups_y, num=20):

        avg_cols = ['precision-1 avg', 'recall-1 avg', 'f1-1 avg']
        group_cols = ['precision-1 group avg', 'recall-1 group avg', 'f1-1 group avg']

        grp_curve = group_metrics.group_precision_recall_curve(y, yprob, groups_y, num)

        fig, ax = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
        grp_curve.plot(y=avg_cols, ax=ax[0][0])
        grp_curve.plot(x='recall-1 avg', y=['precision-1 avg', 'f1-1 avg'], ax=ax[0][1])
        grp_curve.plot(y=group_cols, ax=ax[1][0])
        grp_curve.plot(x='recall-1 group avg', y=['precision-1 group avg', 'f1-1 group avg'], ax=ax[1][1])
        ax[0][0].set_title('Average Group Precision-Recall by Threshold')
        ax[0][1].set_title('Average Group Precision-Recall Curve')
        ax[1][0].set_title('Group Precision-Recall by Threshold')
        ax[1][1].set_title('Group Precision-Recall Curve')
        return ax, grp_curve

    def recall_group(y, yhat, metric='recall-1'):
        if metric=='recall-1':
            grps = groupConsecutiveFlags(y)
        elif metric=='precision-1':
            grps = groupConsecutiveFlags(yhat)
        if (grps == -1).all():
            grp_df = [[np.nan, np.nan, np.nan]]
        else:
            grp_df = []
            for group in np.unique(grps):
                if group != -1:
                    msk = grps==group
                    grp_acc = (y[msk] == yhat[msk]).mean()
                    grp_df.append([group, grp_acc, int(grp_acc > 0.0)])        
        return pd.DataFrame(grp_df, columns=['label', metric + ' avg', metric + ' group']).set_index('label')

    def group_scorer(estimator, X, y):
        yhat = pd.Series(estimator.predict(X), index=X.index)
        recall = group_metrics.recall_group(y, yhat, metric='recall-1').mean()
        precision = group_metrics.recall_group(y, yhat, metric='precision-1').mean()
        score = pd.concat([precision, recall])
        score['recall-1'] = recall_1(estimator, X, y)
        score['precision-1'] = precision_1(estimator, X, y)
        score.sort_index(inplace=True)
        score['f1-1'] = score[['recall-1', 'precision-1']].mean()
        score['f1-1 avg'] = score[['recall-1 avg', 'precision-1 avg']].mean()
        score['f1-1 group'] = score[['recall-1 group', 'precision-1 group']].mean()
        return score.to_dict()

# Minority Group Split Undersample method
class MinorityGroupSplitUndersample:
    
    def __init__(self,
        n_splits=5, train_size=.8, test_size=None,
        train_prct=1, test_prct=None, random_state=None,
        n_repeats=None
    ):
        self.n_splits=n_splits; self.test_size=test_size; self.train_size=train_size;
        self.train_prct=train_prct; self.test_prct=test_prct; self.random_state=random_state;
        self.n_repeats=n_repeats;

    def split(self, X, Y, groups, strategy='GroupShuffleSplit'):    
        X, Y = X.reset_index(drop=True), Y.reset_index(drop=True)
        groups = groups.reset_index(drop=True)
        mino = Y == 1; Y_majo = Y[~mino]
        # Group shuffle split minority
        if strategy == 'GroupShuffleSplit':
            splitter = GroupShuffleSplit(
                n_splits=self.n_splits, test_size=self.test_size,
                train_size=self.train_size, random_state=self.random_state
            )
        elif strategy == 'GroupKFold':
            splitter = GroupKFold(n_splits=self.n_splits)
        elif strategy == 'RepeatedKFold':
            splitter = RepeatedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats)
        t_index, e_index = [], []
        for t_ind, e_ind in splitter.split(X[mino], Y[mino], groups[mino]):
            yt_mino, ye_mino = Y[mino].iloc[t_ind], Y[mino].iloc[e_ind]
            # Undersample train and test majority
            natural_proportion = 1 / mino.mean() - 1  # (1 - mino.mean()) / mino.mean() => negative/positive ratio
            if self.train_prct is None:
                train_prct = natural_proportion # Natural proportion is train_prct is None
            else:
                train_prct = self.train_prct
            yt_majo = Y_majo.sample(int(len(yt_mino) * train_prct), random_state=self.random_state)
            ye_majo = Y_majo.drop(yt_majo.index)
            if self.test_prct is not None: # ALl samples left for testing if test_prct is 'None'
                if self.test_prct == 'natural':
                    test_prct = natural_proportion
                else:
                    test_prct = self.test_prct
                ye_majo = ye_majo.sample(int(len(ye_mino) * test_prct), random_state=self.random_state)
            t_index.append(pd.concat([yt_mino, yt_majo]).index.tolist())
            e_index.append(pd.concat([ye_mino, ye_majo]).index.tolist())
        return zip(t_index, e_index)

def make_grid(params):
    keys = list(params.keys())
    param_list = list(product(*[params[key] for key in keys]))
    return list(map(lambda param_tuple: {key: param for key, param in zip(keys, param_tuple)}, param_list))

split_params_default = dict(
    n_splits=5, # default 5 k folds
    train_size=.79, train_prct=1,
    test_size=.2, test_prct=None,
    random_state=0
)

def groupSplitScore(
    model, Xf, Y, groups,
    split_params=split_params_default,
    strategy='GroupKFold',
    scoring='accuracy', verbose=5
):  
    splitter = MinorityGroupSplitUndersample(**split_params)
    cv = list(splitter.split(Xf, Y, groups, strategy))
    return pd.DataFrame(cross_validate(
        model, Xf, Y, cv=cv,
        scoring=scoring,
        verbose=verbose,
        n_jobs=-1
    ))    

def groupSplitGridSearch(
    model, Xf, Y,
    groups, split_grid,
    strategy='GroupShuffleSplit',
    scoring='accuracy',
    verbose=5
):
    scrs = []
    param_grid = make_grid(split_grid)
    for i, split_params in enumerate(param_grid):
        co(wait=True); print(f'Param grid step: {i+1}/{len(param_grid)}')
        scr = groupSplitScore(
            model, Xf, Y, groups, split_params,
            strategy, scoring, verbose
        )
        for key, value in split_params.items(): scr[key] = value
        scrs.append(scr)
    return pd.concat(scrs)


# Example Usage

# from sklearn.datasets import make_blobs

# X, Y = make_blobs(n_samples=[50, 10], centers=None, n_features=2,
#                   random_state=0)
# X, Y = pd.DataFrame(X), pd.Series(Y)
# groups = groupConsecutiveFlags(ts=Y)

# model = gbc()

# split_params = dict(
#     n_splits=5,
#     train_size=.79, test_size=.2,
#     train_prct=1, test_prct=None,
#     random_state=0
# )

# scr = groupSplitScore(
#     model, X, Y,
#     groups, split_params,
#     strategy='GroupKFold',
#     scoring=scoring, verbose=5
# ); display(scr)

# split_grid = dict(
#     n_splits=[5],
#     train_size=[.8], # train_size_i,
#     test_size=[.2],
#     train_prct=[1, 2, 3, 4, 5],
#     test_prct=[None],
#     random_state=[0]
# )

# scr_grid = groupSplitGridSearch(
#     model, Xf, Y,
#     groups, split_grid,
#     strategy='GroupKFold',
#     scoring=scoring,
#     verbose=5
# ); display(scr_grid)