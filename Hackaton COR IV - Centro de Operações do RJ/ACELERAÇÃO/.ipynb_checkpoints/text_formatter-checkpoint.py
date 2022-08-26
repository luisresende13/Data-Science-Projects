def get_not_number(series):
    obj_index = []
    for index, number in zip(series.index, series):
        try:
            a = float(number) + 1
        except:
            obj_index.append(index)
    return series[obj_index]

import string, numpy as np

def drop_letters(text):
    for char in text:
        if char in string.ascii_letters:
            text = text.replace(char, '')
    return text

def drop_space(text):
    return text.replace(' ', '')

def drop_chars(text, chars=['°', 'º']):
    for char in chars:
        text = text.replace(char, '')
    return text

def split_avg(text, seps=['-', '/', ',']):
    if not text:
        return np.nan
    else:
        for sep in seps:
            if sep in text:
                items = [item for item in text.split(sep) if item]
                if len(items)==0:
                    return np.nan
                elif len(items)==1:
                    return items[0]
                else:
                    try:
                        return str(int(np.mean(np.array(items, dtype='int'))))
                    except:
                        return np.nan
        return text

def text_transform_pipeline(series, functions):
    transformed = series.copy()
    for function in functions:
        transformed = transformed.map(function)
    return transformed