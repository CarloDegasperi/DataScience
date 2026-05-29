import pandas as pd
import numpy as np

def get_AQI_inq(value, intervals):
    if pd.isna(value):
        return np.nan

    for i, (low, high) in enumerate(intervals):
        if low <= value < high:
            return 25 / (high - low) * (value - low) + 20 * i
               
    return np.nan
