import pandas as pd
from IPython.display import clear_output as co
from Modulos.cv_samplers import GroupUnderSampleSplit, print_cls_cnt
from Modulos.imbalanced_selection import MinorityGroupSplitUndersample
from sklearn.ensemble import GradientBoostingClassifier as gbc
from sklearn.metrics import classification_report

class ClassificationPipeline:
    
    def __init__(self,
        train_size=0.79, train_prct=1,
        test_size=0.2, test_prct=None,
        n_splits=1, shuffle=True,
        random_state=None,
    ):
        self.train_size=train_size; self.test_size=test_size;
        self.train_prct=train_prct; self.test_prct=test_prct;
        self.n_splits=n_splits; self.shuffle=shuffle;
        self.random_state=random_state

    # Binary classification pipeline function
    def binary(
        self, X, Y, model=gbc(),
        groups=None, strategy=None,
        return_cls_cnt=False
    ):
        if strategy is not None:
            splitter = MinorityGroupSplitUndersample(
                n_splits=self.n_splits,
                train_size=self.train_size, train_prct=self.train_prct,
                test_size=self.test_size, test_prct=self.test_prct,
                random_state=self.random_state, 
            )
            cv = list(splitter.split(
                X, Y, groups, strategy=strategy
            )); t_ind, e_ind = cv[0][0], cv[0][1]
            xt, xe, yt, ye = X.iloc[t_ind], X.iloc[e_ind], Y.iloc[t_ind], Y.iloc[e_ind]
        else:
            guss = GroupUnderSampleSplit(  # Splitter Instance
                train_size=self.train_size, train_prct=self.train_prct,
                test_size=self.test_size, test_prct=self.test_prct,
            )
            xt, xe, yt, ye = guss.undersample(  # Train and test x and y dataframes
                X, Y, shuffle=self.shuffle, random_state=self.random_state
            )

        cls_cnt = print_cls_cnt(
            Y, yt.index, ye.index,
            display_cnt=(not return_cls_cnt),
            return_cnt=return_cls_cnt
        )
        model.fit(xt, yt)  # Fit model with train data
        yhat = model.predict(xe)  # Make predictions
        scrs = pd.DataFrame(cr(ye, yhat, output_dict=True)).T  # Evaluate model prediction
        if return_cls_cnt:
            return scrs, cls_cnt
        return scrs

    # Multiple targets binary classification pipeline function - Changing Traget variable
    def binary_multi_target(
        self,
        X, Yi, model=gbc(),
        groups=None, strategy=None,
    ):
        scrs, cls_cnts = {}, {}
        for i, label in enumerate(Yi):
            co(wait=True); print(f'{i+1}/{Yi.shape[1]} models evaluated.')
            scrs[label], cls_cnts[label] = self.binary(
                X, Yi[label], model=model,
                groups=groups[label], strategy=strategy,
                return_cls_cnt=True
            )
        scrs, cls_cnts = pd.concat(scrs), pd.concat(cls_cnts)
        scrs.index.names = ['group', 'metric']
        cls_cnts.index.names = ['Group', 'Class']
        return scrs, cls_cnts

def classesGroupRecall(scrs, cls_cnts, ignore_first=5):

    fig, ax = plt.subplots(1, 3, figsize=(20, 3.5))

    scrs['recall'].loc[:, ['0', '1', 'macro avg']].unstack('metric').sort_values('1').plot(use_index=False, ax=ax[0])
    ax[0].set(
        title='Groups Recall per Class - Minority class sorted',
        xlabel='Group',
        ylabel='Recall'
    )

    sort_scrs = scrs['recall'].loc[:, ['0', '1', 'macro avg']].unstack('metric').sort_values('macro avg')
    sort_scrs.plot(use_index=False, ax=ax[1])
    ax[1].set(
        title='Groups Recall per Class - Macro avg sorted',
        xlabel='Group',
        ylabel='Recall'
    )

    train_size, test_size = (mms().fit_transform(cls_cnts[col].loc[sort_scrs.index[ignore_first:], 1].to_frame(col)).reshape(-1).tolist() for col in ['Train set', 'Test set'])
    train_size = [np.nan]*ignore_first + train_size 
    test_size = [np.nan]*ignore_first + test_size 

    ax[2].plot(train_size)
    ax[2].plot(test_size)

    return ax