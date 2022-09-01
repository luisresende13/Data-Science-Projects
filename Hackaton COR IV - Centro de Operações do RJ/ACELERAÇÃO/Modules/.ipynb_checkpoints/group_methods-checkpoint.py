import numpy as np

# Functions to get new columns for majority class and numeric range, per group

def groups_majority_class(data, group_col, class_col):
    route_info = {}
    for label, df in data.groupby(group_col):
        routes_sorted = df[class_col].value_counts().index
        route_info[label] = (routes_sorted[0] if len(routes_sorted) else np.nan)
    return np.array(list(map(lambda label: route_info[label], data[group_col].values)))

def group_majority_class_numeric_range(data, group_col, class_col, num_col):
    route_info = {}
    for label, df in data.groupby(group_col):
        main_route = df[class_col].value_counts().index[0]
        route_numbers = df[num_col][df[class_col]==main_route]
        num_range = ' - '.join([str(int(route_numbers.max())), str(int(route_numbers.min()))])
        route_info[label] = num_range
    return np.array(list(map(lambda label: route_info[label], data[group_col])))