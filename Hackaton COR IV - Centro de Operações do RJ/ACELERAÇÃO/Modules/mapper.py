from sklearn.preprocessing import LabelEncoder as le
import numpy as np, pandas as pd, matplotlib.pyplot as plt, folium

def rgba_cmap(cMap, number):
    c = np.array(cMap(number))
    c[:3] = c[:3] * 255
    return f'rgba{tuple(c)}'

def plot_markers(
    Map, df, radius=3,
    coord_cols=['lat', 'lng'], group_col='POP_TITULO', cmap=None,
    exclude=["Bolsão d'água em via"],
    touch_coord=True, return_encoder=False, 
):
    
    if touch_coord: Map.add_child(folium.LatLngPopup())

    if group_col is not None:
        LE = le().fit(df[group_col])
        groups = pd.Series(LE.transform(df[group_col]), index=df.index)
        cMap = plt.get_cmap(cmap, lut=len(groups.unique()))

    for index in df.index:
        row = df.loc[index]; coord = row.loc[coord_cols]
        if group_col is not None:
            row_group = row[group_col]
            c = rgba_cmap(cMap, groups.loc[index])
        else:
            row_group, c = None, 'blue'
        if row_group not in exclude:
            folium.CircleMarker(
                location=list(coord), radius=radius,
                color=c, opacity=1, fill=True,
                fill_color=c, fill_opacity=1,
            ).add_to(Map)        
    
    if return_encoder:
        return Map, LE
    return Map

def draw_circles(
    Map, data, loc, radius, popup, tooltip,
    cmap=None, lut=1, LE=None,
    stroke=True, weight=3, fill=True,
    fill_color=None, fill_opacity=.3
):
    cMap = plt.get_cmap(cmap, lut=lut)
    if LE is None: LE = le().fit(data.index)
    labels = pd.Series(LE.transform(data.index), index=data.index)

    for index, row in data.iterrows():
        c = rgba_cmap(cMap, labels.loc[index])
        fill_c = (c if fill_color is None else fill_color)
        folium.Circle(
            row[loc], row[radius] * 1100,
            popup=row[popup], tooltip=row[tooltip],
            color=c, stroke=stroke, weight=weight, fill=fill,
            fill_color=fill_c, fill_opacity=fill_opacity
            
        ).add_to(Map)
    return Map

def draw_rectangles(
    Map, data, loc, popup, tooltip,
    cmap=None, lut=1, LE=None,
    stroke=True, weight=3, fill=True,
    fill_color=None, fill_opacity=.3
):
    cMap = plt.get_cmap(cmap, lut=lut)
    if LE is None: LE = le().fit(data.index)
    labels = pd.Series(LE.transform(data.index), index=data.index)

    for index, row in areas.iterrows():
        c = rgba_cmap(cMap, labels.loc[index])
        fill_c = (c if fill_color is None else fill_color)
        folium.Rectangle(
            [list(row[loc[0]]), list(row[loc[1]])], # row[radius] * 1100,
            popup=row[popup], tooltip=row[tooltip],
            color=c, stroke=stroke, weight=weight, fill=fill,
            fill_color=fill_c, fill_opacity=fill_opacity
            
        ).add_to(Map)
    return Map