import pandas as pd

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