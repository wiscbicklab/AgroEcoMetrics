import numpy as np


def match_weather(weather_datetime_col, data_datetime_col):
    vals = np.searchsorted(weather_datetime_col, data_datetime_col, sorter=None)
    match_times = [None] * len(vals)
    match_indices = [None] * len(vals)
    diffs = [None] * len(vals)
 
    for i in range(len(vals)):
        vals[i] = min(vals[i], len(weather_datetime_col)-1)
        if vals[i] > 0 and abs(data_datetime_col[i]-weather_datetime_col[vals[i]]) > abs(data_datetime_col[i]-weather_datetime_col[vals[i]-1]):
            vals[i] = vals[i]-1
        match_times[i] = weather_datetime_col[vals[i]]
        match_indices[i] = vals[i]
        diffs[i] = abs(data_datetime_col[i]-weather_datetime_col[vals[i]])
    return (data_datetime_col, match_times, match_indices, diffs)
 
 
def get_cols_from_weather_for_matched_indices(weather_data, weather_cols, matched_indices):
    cols = [None]*len(weather_cols)
    for col in range(len(weather_cols)):
        cols[col] = [None]*len(matched_indices)
        for i in range(len(matched_indices)):
            cols[col][i] = weather_data[weather_cols[col]][matched_indices[i]]
 
    return cols