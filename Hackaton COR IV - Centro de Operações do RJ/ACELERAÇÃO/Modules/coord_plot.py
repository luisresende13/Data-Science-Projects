import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns; sns.set()
from sklearn.preprocessing import LabelEncoder as le
from IPython.display import clear_output as co

cmaps = [
    'Pastel1', 'Pastel2', 'Paired', 'Accent',
    'Dark2', 'Set1', 'Set2', 'Set3',
    'tab10', 'tab20', 'tab20b', 'tab20c'
]

def filter_group_size(data, group_col, min_members=10):
    label_cnt = data[group_col].value_counts()
    top_labels = label_cnt[label_cnt > min_members].index
    top_msk = data[group_col].isin(top_labels)
    return data[top_msk]

def cluster_algo_comparison(x, y, data, algs, hide_outliers=False, title='{}/{}', figsize=(12, 7.5), n_cols=2): 
    fig = plt.figure(figsize=figsize, tight_layout=True)
    n_rows =  len(algs)//n_cols if len(algs)%n_cols==0 else len(algs)//2 + 1
    axs = (fig.add_subplot(n_rows, n_cols, i) for i in range(1, len(algs)+1))
    for ax, alg in zip(axs, algs):
        labels = np.unique(alg.labels_)
        for label in labels:
            if hide_outliers and label==-1: continue
            ax.scatter(x, y, data=data[alg.labels_==label])
        ax.set(title=title.format(type(alg).__name__, labels.shape[0]))
    return axs

def connect_coordinates_colored(
    lat, lng, groups, numbers=None,
    cmap='tab10', title='', figsize=None,
    connect=True, scatter=True, ms=30
):
    cmap = plt.get_cmap(cmap)
    labels = pd.Series(le().fit_transform(groups), index=groups.index)
    plt.figure(figsize=figsize)
    for label in labels.unique():
        group_msk = labels==label
        if numbers is not None:
            group_index = numbers[group_msk].sort_values().index
        else:
            group_index = labels[group_msk].index
        group_lat, group_lng = lat.loc[group_index], lng.loc[group_index]
        if connect: plt.plot(group_lat, group_lng)
        if scatter: plt.scatter(group_lat, group_lng, s=ms)
    plt.xlabel('Longitude'); plt.ylabel('Latitude');
    plt.title(title.format(len(labels.unique()))); plt.show()

def min_group_size_progression_plot(
    data, coord_cols, group_col, max_incidents=15,
    color='blue', figsize=None, pause=0.3
):
    route_count = data[group_col].value_counts(); recurrence = []
    for min_incidents in range(max_incidents+1):
        rec_routes = route_count[route_count>=min_incidents]
        rec_data = data[data[group_col].isin(rec_routes.index)]
        n_routes = len(rec_routes)
        n_incidents = rec_data.shape[0]
        p_routes = round(n_routes / route_count.shape[0] * 100, 1)
        p_incidents = round(n_incidents / data.shape[0] * 100, 1)
        recurrence.append([min_incidents, n_routes, n_incidents, p_routes, p_incidents])
        fig = plt.figure(figsize=figsize); ax = fig.add_subplot(111)
        rec_data.plot(
            coord_cols[0], coord_cols[1],
            kind='scatter', c=color, ax=ax,
            title=f'''Min Incidents per Route: {min_incidents} / {max_incidents}.
Routes: {n_routes} / {route_count.shape[0]} ({p_routes} %). Incidents: {n_incidents} / {data.shape[0]} ({p_incidents} %)''',
        )
        plt.show(); plt.pause(pause); co(wait=True)
    return pd.DataFrame(recurrence, columns=['min_incidents', 'n_routes', 'n_incidents', 'p_routes (%)', 'p_incidents (%)']); # recurrence_df

def atemporal_evolution_plot(
    data, coord_cols=['lat', 'lng'], time_col='evento_inicio', color='blue',
    group_col='route', min_per_group=15, cmap=None, lut=None,
    pause=0.3, frame_evolution=25, freq='D',
    figsize=(9, 5), title='Progess: {} / {}',
    path=None,

):
    data_left = data.sort_values(time_col).copy()
    color_code=None

    if group_col is not None:
        if min_per_group is not None: # filter data by groups with at least 'min_per_group' members
            group_count = data[group_col].value_counts()
            groups_left = group_count[group_count>=min_per_group]
            data_left = data[data[group_col].isin(groups_left.index)].copy()
        color_code = pd.Series(le().fit_transform(data_left[group_col]), index=data_left.index).map(plt.get_cmap(cmap, lut=lut))

    n_samples = len(data_left)
    
    if freq is None:
        print_frames = data_left.index[range(0, n_samples+1, frame_evolution)]
    else:
        ts_values = pd.to_datetime(data_left[time_col])
        data_left[time_col] = ts_values
        if group_col is not None:
            color_code.index = ts_values
            color_code.sort_index(inplace=True)
        data_left.set_index(time_col, inplace=True)
        data_left.sort_index(inplace=True)
        ts = data_left.index
        print_frames = pd.date_range(ts.min(), ts.max(), freq=freq)

    coord_data = data_left[coord_cols]
    for index in print_frames:
        
        frame_data = coord_data.loc[:index]
            
        if group_col is not None:
            c = color_code.loc[:index]
            clr = None
        else:
            c = color
            clr = color
            
        if freq is not None:
            date, time = str(index).split()
            Title = title.format(date, time)
        else:
            Title = title.format(len(frame_data), n_samples)
            
        fig = plt.figure(figsize=figsize); ax = fig.add_subplot(111)
        frame_data.plot(
            coord_cols[0], coord_cols[1],
            kind='scatter', c=c,
            title=Title, ax=ax,
#             color=clr
        )
        if path is not None: plt.savefig(path.format(date if freq else index))
        plt.show(); plt.pause(pause); co(wait=True)

def min_samples_clusters_animation(
    data, group_col, max_samples_stop=0.5, freq=0.1,
    coord_cols=['EVENTO_LONGITUDE', 'EVENTO_LATITUDE'], order_col='street_number',
    connect=False, scatter=True,
    cmap=None, figsize=(20, 12),
    title='Incident coordinates Connected and Colored Line & Scatter Plot',
):
    
    label_cnt = data[group_col].value_counts()
    max_samples = label_cnt.max()
    min_samples = label_cnt.min()
    size_range = max_samples - min_samples
    
    if type(max_samples_stop) is float:
        min_samples_i = min_samples + (np.arange(0, max_samples_stop, freq) * size_range).round(0)
    elif type(max_samples_stop) is int:
        min_samples_i = np.arange(min_samples, max_samples_stop+1, freq)
        
    for min_samples in min_samples_i:

        top_data = filter_group_size(data, group_col=group_col, min_members=min_samples)

        co(wait=True)
        connect_coordinates_colored(
            top_data[coord_cols[0]], top_data[coord_cols[1]],
            top_data[group_col], top_data[order_col], cmap=cmap,
            connect=connect, scatter=scatter,
            title=title,
            figsize=figsize,
        )