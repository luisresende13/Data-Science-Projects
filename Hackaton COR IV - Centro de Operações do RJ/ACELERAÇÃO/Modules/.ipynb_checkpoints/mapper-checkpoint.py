from sklearn.preprocessing import LabelEncoder as le
import pandas as pd, matplotlib.pyplot as plt, folium

def plot_markers(
    df, center=[-22.9037, -43.4276], zoom=10, tiles='OpenStreetMap',
    width='80%', height='80%', radius=3,
    coord_cols=['lat', 'lng'], group_col='POP_TITULO', cmap=None,
    exclude=["Bolsão d'água em via"], touch_coord=True
):

    groups = pd.Series(le().fit_transform(df[group_col]), index=df.index)
    cmap = plt.get_cmap(cmap, lut=len(groups.unique()))
    
    m = folium.Map(location=center, zoom_start=zoom, width=width, height=height, tiles=tiles)
    for index in df.index:
        c = cmap(groups.loc[index])
        c = tuple(list(round(i*255, 5) for i in c)[:3] + [c[3]])
        c = f'rgba{c}'
#         print(c)
        row = df.loc[index]; coord = row.loc[coord_cols]
        row_group = row[group_col]
        if row_group not in exclude:
            folium.CircleMarker(
                location=list(coord),
                radius=radius, color=c, opacity=1,
                fill=True, fill_color=c, fill_opacity=1,
            ).add_to(m)

    if touch_coord: m.add_child(folium.LatLngPopup())
    return m