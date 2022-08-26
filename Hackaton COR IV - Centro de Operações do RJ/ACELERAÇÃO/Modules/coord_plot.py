import pandas as pd, matplotlib.pyplot as plt, seaborn as sns; sns.set()
from sklearn.preprocessing import LabelEncoder as le
from IPython.display import clear_output as co

cmaps = [
    'Pastel1', 'Pastel2', 'Paired', 'Accent',
    'Dark2', 'Set1', 'Set2', 'Set3',
    'tab10', 'tab20', 'tab20b', 'tab20c'
]

def connect_coordinates_colored(lat, lng, groups, numbers, cmap='tab10', title='', figsize=None, connect=True, scatter=True):
    cmap = plt.get_cmap(cmap)
    labels = pd.Series(le().fit_transform(groups), index=groups.index)
    plt.figure(figsize=figsize)
    for label in labels.unique():
        group_index = numbers[labels==label].sort_values().index
        group_lat, group_lng = lat.loc[group_index], lng.loc[group_index]
        if connect: plt.plot(group_lat, group_lng)
        if scatter: plt.scatter(group_lat, group_lng, s=30)
    plt.xlabel('Latitude'); plt.ylabel('Longitude'); plt.title(title); plt.show()

def min_group_size_progression_plot(data, coord_cols, group_col, max_incidents=15, color='blue', figsize=None, pause=0.3):
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
