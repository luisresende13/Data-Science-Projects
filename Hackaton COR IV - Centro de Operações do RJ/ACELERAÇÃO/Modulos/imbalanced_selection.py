import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

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

# Minority Group Split Undersample method
class MinorityGroupSplitUndersample:
    
    def __init__(self,
        n_splits=5, train_size=.8, test_size=None,
        train_prct=1, test_prct=None, random_state=None,
    ):
        self.n_splits=n_splits; self.test_size=test_size; self.train_size=train_size;
        self.train_prct=train_prct; self.test_prct=test_prct; self.random_state=random_state;        

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