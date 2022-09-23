import pandas as pd, numpy as np, matplotlib.pyplot as plt

def min_max_filter_stats(data, desc, stat='min', acumulate=None, top_down=True):

    filter_stats = []
    for col in data:
        min_value = desc.loc[stat, col]
        if stat=='min':
            above_min = data[col].fillna(min_value) >= min_value
        elif stat=='max':
            above_min = data[col].fillna(min_value) <= min_value
        filter_stats.append([col, min_value, above_min.sum(), above_min.sum()/len(data)*100])
    filter_stats = pd.DataFrame(filter_stats, columns=['column', 'threshold', 'rows left', 'rows left (%)'])
    filter_stats = filter_stats.sort_values('rows left', ascending=top_down).reset_index(drop=True)

    if acumulate is not None:
        cum_stats = []
        for col in filter_stats['column']:
            min_value = desc.loc[stat, col]
            if stat=='min':
                above_min = data[col].fillna(min_value) >= min_value
            elif stat=='max':
                above_min = data[col].fillna(min_value) <= min_value
            if col == filter_stats['column'].iloc[0]:
                above_min_cum = above_min
            else:
                if acumulate=='or':
                    above_min_cum = above_min_cum | above_min
                elif acumulate=='and':
                    above_min_cum = above_min_cum & above_min
            cum_stats.append([col, min_value, above_min_cum.sum(), above_min_cum.sum()/len(data)*100])
        cum_stats = pd.DataFrame(cum_stats, columns=['column', 'threshold', 'rows left cum', 'rows left cum (%)'])
        return filter_stats, cum_stats
    
    return filter_stats

def min_max_filter(X, min_values=None, max_values=None, n_filters=10, margin_min=0.0, margin_max=0.0):
    X_filt = X.copy()
    if min_values is not None:
        index = min_values.index[:n_filters]
    else:
        index = max_values.index[:n_filters]
    for col in index:
        if min_values is not None:
            min_value =  min_values[col] * (1 + margin_min)
            X_filt = X_filt[X_filt[col].fillna(min_value) >= min_value]
        if max_values is not None:
            max_value =  max_values[col] * (1 - margin_max)
            X_filt = X_filt[X_filt[col].fillna(max_value) <= max_value]
    print('\nRecords:', X.shape[0]); print('Records left:', X_filt.shape[0])
    print('Records left (%):', round(X_filt.shape[0] / X.shape[0] * 100, 2), '%\n')
    return X_filt

def filter_by_stats(
    X, Y, groups=None, n_filters=[10, 0],
    choose_from=None, acumulate='and',
    top_down=True, figsize=(14, 3.5)
):

    #### Extract minority and majority classes records
    mino, majo = X[Y==1], X[~(Y==1)]
    mino_desc = mino.describe()

    #### Filter records by minority minimum values
    stats, cum_stats = min_max_filter_stats(X, mino_desc, 'min', acumulate, top_down)
    stats_max, cum_stats_max = min_max_filter_stats(X, mino_desc, 'max', acumulate, top_down)

    min_values = mino_desc.loc['min'].loc[stats['column']]
    max_values = mino_desc.loc['max'].loc[stats_max['column']]

    if choose_from is not None:
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        stats.set_index('column')['rows left (%)'].head(choose_from).plot(marker='o', ms=5, xticks=[], title='Records above target minimum per column (%)', ax=ax[0])
        stats_max.set_index('column')['rows left (%)'].head(choose_from).plot(marker='o', ms=5, xticks=[], title='Records below target maximum per column (%)', ax=ax[1])
        plt.show(); fig, ax = plt.subplots(1, 2, figsize=figsize)
        cum_stats.set_index('column')['rows left cum (%)'].iloc[:choose_from].plot(xticks=[], marker='o', ms=5, title='Records above target minimum per column - acumulated (%)', ax=ax[0])
        cum_stats_max.set_index('column')['rows left cum (%)'].iloc[:choose_from].plot(xticks=[], marker='o', ms=5, title='Records above target minimum per column - acumulated (%)', ax=ax[1])
        plt.show()
        n_filters[0] = int(input('N° of columns to filter by minimum:'))
        n_filters[1] = int(input('N° of columns to filter by maximum:'))

    print('\nPositive minimum filter:')
    # Filter data by positive class minimum values
    X_filt = min_max_filter(
        X, min_values, # max_values,
        n_filters=n_filters[0], margin_min=0.0, margin_max=0.0
    )

    print('Positive maximum filter:')
    # Filter data by positive class maximum values
    X_filt = min_max_filter(
        X_filt, None, max_values,
        n_filters=n_filters[1], margin_min=0.0, margin_max=0.0
    )

    ### Filter target variable
    Y_filt = Y.loc[X_filt.index]

    cnts = [
        Y_filt.value_counts().to_frame('Class Count'),
        100 * (Y_filt.value_counts().to_frame('Percent left (%)') / Y.value_counts().to_frame('Percent left (%)')).round(4)
    ]

    display(pd.concat(cnts, 1))

    if groups is not None:
        groups_filt = groups.loc[X_filt.index] # Does the same as above
        return X_filt, Y_filt, groups_filt

    return X_filt, Y_filt