from sklearn.preprocessing import MinMaxScaler as mms, LabelEncoder as le

def TimeseriesTransformPipeline(data, min_time=None, cut=None, drop_empty_cols=False, label_encode=None, scale=False, interpolate='linear', fillna='min', transform_report=False):
    
    X = data.copy()
    print('Initial data:', X.shape)
    
    if min_time is not None: # Extract features dataset - Time serie section after first recorded incident
        X = X[data.index > min_time]
        print('Time extraction:', X.shape)
    
    if cut is not None:
        X = X.iloc[:cut].copy()

    if drop_empty_cols: # Drop X empty columns and rows
        X.dropna(1, how='all', inplace=True)
        print('Drop empty columns: ', X.shape)

    if label_encode is not None:
        for col in label_encode:
            X[col] = le().fit_transform(X[col])
    
    if scale: # Scale X
        X[X.columns] = mms().fit_transform(X)

    if interpolate is not None: # Interpolate X missing values
        X = X.interpolate(interpolate)

    if fillna is not None: # Fill missing values with the minimum column value
        for col in X:
            if fillna=='min':
                fill_value = X[col].min()
            else:
                fill_value = 0
            X[col].fillna(fill_value, inplace=True)
                
    return X