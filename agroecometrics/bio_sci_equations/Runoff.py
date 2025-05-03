# Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

df = None

def load_data(file, start_date=None, end_date=None):
    '''
    Loads a data file.

    file: A string containing the path to your data
    start_date: Optional string in 'YYYY-MM-DD' format to filter data from this date onward
    end_date: Optional string in 'YYYY-MM-DD' format to filter data up to this date
    '''
    global df
    df = pd.read_csv(file)

    # Ensure 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Apply date filtering if start_date or end_date is provided
    if start_date:
        df = df[df['Date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['Date'] <= pd.to_datetime(end_date)]

    # Reset index
    df = df.reset_index(drop=True)

def curve_number(P, CN=75):
    '''
    Curve number method proposed by the Soil Conservation Service, More Information Here:
    https://www.wikiwand.com/en/Runoff_curve_number

    P is precipitation in milimeters
    CN is the curve number
    '''
    runoff = np.zeros_like(P)
    
    S02 = 1000/CN - 10
    S005 = 1.33 * S02**1.15
    Lambda = 0.05 # Hawkins, 2002.
    Ia = S005 * Lambda # Initial abstraction (Ia). Rainfall before runoff starts to occur.
    idx = P > Ia
    runoff[idx] = (P[idx] - Ia)**2 / (P[idx] - Ia + S005)

    return runoff

def compute_rainfall():
    # Compute cumulative rainfall
    df['RAIN_SUM'] = df['RAIN'].cumsum()

    # Compute cumulative runoff
    df['RUNOFF'] = curve_number(df['RAIN']/25.4,CN=80)*25.4
    df['RUNOFF_SUM'] = df['RUNOFF'].cumsum()

    # Check Dataframe
    df.head()

def plot_rainfall(file_name):
    '''
    Creates a plot of fainfall and runoff over the time-frame of the data loaded

    file_name: Is the name of the png file the plot will be saved to
    return: the fileName if the file is created otherwise return error
    '''
    # Validate file_name type
    if not isinstance(file_name, str):
        raise TypeError("file_name must be a string.")

    # Validate extension
    if not file_name.lower().endswith('.png'):
        raise ValueError("The filename must end with '.png'.")

    # Optional: Ensure directory exists
    dir_name = os.path.dirname(file_name)
    if dir_name and not os.path.exists(dir_name):
        raise FileNotFoundError(f"The directory '{dir_name}' does not exist.")

    plt.figure(figsize=(6,4))
    plt.plot(df['Date'], df['RAIN_SUM'], color='navy', label='Rainfall')
    plt.plot(df['Date'], df['RUNOFF_SUM'], color='tomato', label='Runoff')
    plt.ylabel('Rainfall or Runoff (mm)')
    plt.legend()
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    return file_name

# TODO: Move to testing files.
if __name__ == '__main__':
    load_data("private/Marshfield_02_11_2023-04_28_2025.csv", '2024-01-01', '2025-01-01')
    compute_rainfall()
    plot_rainfall('private/runoff_plot.png')
