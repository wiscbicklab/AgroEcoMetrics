# Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from agroecometrics import settings

# Load settings labels
labels = settings.get_labels()

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

def curve_number(P, CN=75):
    '''
    Curve number method for runoff estimation.
    P is precipitation in millimeters
    CN is the curve number
    '''
    runoff = np.zeros_like(P)
    S02 = 1000 / CN - 10
    S005 = 1.33 * S02**1.15
    Lambda = 0.05
    Ia = S005 * Lambda
    idx = P > Ia
    runoff[idx] = (P[idx] - Ia)**2 / (P[idx] - Ia + S005)
    return runoff

def compute_rainfall(df):
    '''
    Computes cumulative rainfall and runoff.

    df: DataFrame with rainfall data
    return: DataFrame with additional columns for rainfall and runoff
    '''
    df['RAIN_SUM'] = df[labels['rain']].cumsum()
    df['RUNOFF'] = curve_number(df[labels['rain']] / 25.4, CN=80) * 25.4
    df['RUNOFF_SUM'] = df['RUNOFF'].cumsum()
    return df

def plot_rainfall(df, file_name):
    '''
    Creates a plot of rainfall and runoff over time.

    df: DataFrame with cumulative rainfall and runoff
    file_name: PNG file path for the plot
    return: file_name if successful
    '''
    if not isinstance(file_name, str):
        raise TypeError("file_name must be a string.")
    if not file_name.lower().endswith('.png'):
        raise ValueError("The filename must end with '.png'.")
    dir_name = os.path.dirname(file_name)
    if dir_name and not os.path.exists(dir_name):
        raise FileNotFoundError(f"The directory '{dir_name}' does not exist.")

    plt.figure(figsize=(6, 4))
    plt.plot(df[labels['date']], df['RAIN_SUM'], color='navy', label='Rainfall')
    plt.plot(df[labels['date']], df['RUNOFF_SUM'], color='tomato', label='Runoff')
    plt.ylabel('Rainfall or Runoff (mm)')
    plt.legend()
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    return file_name

# TODO: Move to testing files.
if __name__ == '__main__':
    df = load_data("private/Marshfield_All_Data.csv", '2024-01-01', '2025-01-01')
    df = compute_rainfall(df)
    plot_rainfall(df, 'private/runoff_plot.png')
