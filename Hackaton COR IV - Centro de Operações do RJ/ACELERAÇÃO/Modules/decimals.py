import numpy as np, pandas as pd

#### Compute and update order of magnitude of float value or series.

def orderOfMagnitude(number):
    return np.floor(np.log10(abs(number)))

def correctMagnitude(number, mag=1):
    if type(number) is float:
        magnitude = orderOfMagnitude(number)
        return number / 10 ** ( orderOfMagnitude(number) - mag )
    else:
        return [correctMagnitude(n, mag) for n in number]

# Replace values below provided decimal precision with NAN in series.

def isBelowPrecision(series, precision=1):
    next_precision = abs(series * 10 ** (precision-1))
    next_abs_dif = next_precision - next_precision.round(0)
    return next_abs_dif == 0

# Compute decimal precision

def dropBelowPrecision(df, precision=1, cols=None, subset='all'): # accepts array
    if cols is None: cols = df.columns
    for col in cols:
        below_msk = isBelowPrecision(df[col], precision=precision)
        if subset=='all':
            df.loc[below_msk] = np.nan
        elif subset=='each':
            df[col].loc[below_msk] = np.nan
    return df    