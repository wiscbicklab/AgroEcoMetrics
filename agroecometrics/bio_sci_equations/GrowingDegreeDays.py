# Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from agroecometrics import settings

# Define cardinal temperatures for corn
T_base = None
T_opt = None
T_upper = None

labels = None

# TODO move elsewhere then finish


def load_data(file, start_date=None, end_date=None):
    '''
    Loads a data file and returns a filtered DataFrame.

    file: A string containing the path to your data
    start_date: Optional string in 'YYYY-MM-DD' format to filter data from this date onward
    end_date: Optional string in 'YYYY-MM-DD' format to filter data up to this date
    return: A pandas DataFrame
    '''
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.replace("'", "")

    df[labels['date']] = pd.to_datetime(df[labels['date']])

    if start_date:
        df = df[df[labels['date']] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df[labels['date']] <= pd.to_datetime(end_date)]

    return df.reset_index(drop=True)

def interpolate_temp_data(df):
    # Find number of missing values
    df[labels['soil_temp2']].isna().sum()

    # Replace missing values using linear method
    df[labels['soil_temp2']].interpolate(method='linear', inplace=True)

    # Check that missing values were replaced
    df[labels['soil_temp2']].isna().sum()






