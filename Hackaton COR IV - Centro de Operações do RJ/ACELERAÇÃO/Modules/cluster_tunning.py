import os, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns; sns.set()
from sklearn import metrics
from Modules.coord_plot import filter_group_size
from IPython.display import clear_output as co

def log_range(order_min=-1, order_max=1, min_value=1, max_value=9):
    log_values = pd.concat([pd.Series(np.arange(1, 10, 1) * 10 ** order) for order in range(order_min, order_max+1)], 0).reset_index(drop=True)
    return log_values[(log_values >= (min_value * 10 ** order_min)) & (log_values <= (max_value * 10 ** order_max))]

def cluster_grid_search(data, alg, param_name, params):
    labels = {}
    for i, param in enumerate(params):
        co(wait=True); print(f'{i+1}/{len(params)} Search grid...')
        alg.set_params(**{param_name: param})
        labels[param] = alg.fit(data).labels_
    return labels

def ndim_centroid(data):
    return data.sum() / len(data)

def inertia(data, labels, order=1):
    grp_inertia = pd.Series(dtype='int64')
    for label in labels.unique():
        grp_data = data[labels==label]
        mass_center = ndim_centroid(grp_data)
        grp_inertia.loc[label] = ((grp_data - mass_center) ** 2).sum(1).sum()
    return grp_inertia

def evaluate_labels(data, labels, param_name, min_samples):
    scrs = []
    params = list(labels.keys())
    for i, param in enumerate(params):
        co(wait=True); print(f'{i+1}/{len(params)} Search grid evaluation...')
        data['label'] = labels[param]
        in_data = data[data['label'] != -1]
        top_data = filter_group_size(in_data, group_col='label', min_members=min_samples) # Excluing clusters with less then 'min_members' samples 
        n_labels = len(data['label'].unique()) - 1
        n_inlabels = len(top_data['label'].unique())
        n_outlabels = n_labels - n_inlabels
        n_in = len(top_data)
        n_out = len(data) - n_in
        n_non_outliers = len(in_data)
        n_outliers = len(data) - n_non_outliers
        n_outliers_label = n_out - n_outliers
        inertia_i = inertia(top_data.drop('label', 1), top_data['label']).sum()
        try: silhouette = metrics.silhouette_score(top_data.drop('label', 1), top_data['label'])
        except: silhouette = np.nan
        try: bouldin = metrics.davies_bouldin_score(top_data.drop('label', 1), top_data['label'])
        except: bouldin = np.nan
        try: calinski = metrics.calinski_harabasz_score(top_data.drop('label', 1), top_data['label'])
        except: calinski = np.nan
        scrs.append([param, inertia_i, silhouette, bouldin, calinski, n_labels, n_inlabels, n_outlabels, n_in, n_out, n_non_outliers, n_outliers, n_outliers_label])
    return pd.DataFrame(scrs, columns=[param_name, 'inertia', 'silhouette', 'bouldin', 'calinski', 'n_labels', 'n_inlabels', 'n_outlabels', 'n_in', 'n_out', 'n_non_outliers', 'n_outliers', 'n_outliers_label'])

def min_samples_analysis(data, labels, param_name, min_samples, max_samples):
    scrs = []
    params = list(labels.keys())
    for i, param in enumerate(params):
        co(wait=True); print(f'{i+1}/{len(params)} Search grid evaluation...')
        data['label'] = labels[param]
        in_data = data[data['label'] != -1]
        for n_samples in range(min_samples, max_samples+1):
            top_data = filter_group_size(in_data, group_col='label', min_members=n_samples) # Excluing clusters with less then 'min_members' samples 
            n_inlabels = len(top_data['label'].unique())
            n_outliers = len(data) - len(in_data)
            inertia_i = inertia(top_data.drop('label', 1), top_data['label']).sum()
            try: silhouette = metrics.silhouette_score(top_data.drop('label', 1), top_data['label'])
            except: silhouette = np.nan
            try: bouldin = metrics.davies_bouldin_score(top_data.drop('label', 1), top_data['label'])
            except: bouldin = np.nan
            try: calinski = metrics.calinski_harabasz_score(top_data.drop('label', 1), top_data['label'])
            except: calinski = np.nan
            scrs.append([param, n_samples, n_inlabels, n_outliers, inertia_i, silhouette, bouldin, calinski])
    return pd.DataFrame(scrs, columns=[param_name, 'min_samples', 'n_inlabels', 'n_outliers', 'inertia', 'silhouette', 'bouldin', 'calinski'])

def rotate_3d_plot(
    data, x, y, z,
    wire=True, surface=False, scatter=False,
    xy_start=0, xy_end=20, z_start=0, z_end=360,
    frames=15, zlim=None,
    title='x - {}° / y - {}°',
    cstride=1, rstride=1,
):
    x_shape, y_shape = (data[XX].unique().shape[0] for XX in [x, y])
    shape = (x_shape, y_shape)

    X = data[x].values.reshape(shape)
    Y = data[y].values.reshape(shape)
    Z = data[z].values.reshape(shape)
    
    xy_range = np.linspace(xy_start, xy_end, frames)
    z_range = np.linspace(z_start, z_end, frames)

    for i in range(frames):
        fig = plt.figure(figsize=(10, 6)); ax = fig.add_subplot(projection='3d')
        # if scatter: ax.scatter(data[x], data[y], data[z])
        if wire: ax.plot_wireframe(X, Y, Z, cstride=cstride, rstride=rstride)
        if surface: ax.plot_surface(X, Y, Z, cstride=cstride, rstride=rstride)
        
        ax.set(
            xlabel=x, ylabel=y, zlabel=z,
            zlim=zlim, 
            title=title.format(round(z_range[i]), round(xy_range[i]))
        )
        ax.view_init(xy_range[i], z_range[i])
        plt.show(); plt.pause(.1); co(wait=True)