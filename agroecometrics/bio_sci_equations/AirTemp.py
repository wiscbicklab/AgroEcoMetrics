# Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import circmean
import os
from agroecometrics import settings

# Create a range of days to be used for modeling
doy = np.arange(1, 366)

# Define the air temperature model function
model = lambda doy, a, b, c: a + b * np.cos(2*np.pi*((doy - c)/365) + np.pi)

# Data labels
labels = None

def load_data(file, start_date=None, end_date=None):
    '''
    Loads a data file to perform calculations on

    file: A string containing the path to your data
    start_date: Optional string in 'YYYY-MM-DD' format to filter data from this date onward
    end_date: Optional string in 'YYYY-MM-DD' format to filter data up to this date
    return: A pandas DataFrame
    '''
    # Get data Labels
    global labels
    labels = settings.get_labels()
    
    # Load unfiltered data from csv file
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.replace("'", "")  # Clean column names

    # Convert the 'Date' column to datetime objects
    df[labels['date']] = pd.to_datetime(df[labels['date']], errors='coerce', format=labels['date_format'])
    df = df.dropna(subset=[labels['date']])  # Drop rows where 'Date' conversion failed

    # Filter data based on user-provided start and end dates
    if start_date:
        df = df[df[labels['date']] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df[labels['date']] <= pd.to_datetime(end_date)]

    # Select relevant columns and add day-of-year and year columns
    df = df[[labels['date'], labels['temp']]]
    df['DOY'] = df[labels['date']].dt.dayofyear
    df['YEAR'] = df[labels['date']].dt.year
    df = df.reset_index(drop=True)

    return df

def model_temp(df):
    '''
    Creates a model of air temperature from the data and creates daily
    temperature predictions

    df: DataFrame with temperature data
    return: A numpy array of predicted daily temperatures
    '''
    global labels

    # Calculate mean temperature and temperature amplitude
    T_avg = df[labels['temp']].mean()
    T_min, T_max = df.groupby(by='DOY')[labels['temp']].mean().quantile([0.05, 0.95])
    A = (T_max - T_min) / 2

    # Estimate the day of year with minimum temperature using circular mean
    idx_min = df.groupby(by='YEAR')[labels['temp']].idxmin()
    doy_T_min = np.round(df.loc[idx_min, 'DOY'].apply(circmean).mean())

    # Generate daily temperature predictions using the model
    T_pred = model(df['DOY'], T_avg, A, doy_T_min)

    return T_pred

def plot_temp(df, file_name, T_pred):
    '''
    Creates a plot of air temperature over the time-frame of the loaded data

    df: DataFrame with temperature data
    file_name: Filename for saved plot (PNG)
    T_pred: Predicted temperature array
    return: file_name if successful
    '''
    global labels
    
    # Validate input parameters
    if not isinstance(file_name, str):
        raise TypeError("file_name must be a string.")
    if not file_name.lower().endswith('.png'):
        raise ValueError("The filename must end with '.png'.")

    dir_name = os.path.dirname(file_name)
    if dir_name and not os.path.exists(dir_name):
        raise FileNotFoundError(f"The directory '{dir_name}' does not exist.")

    # Plot actual vs predicted temperature
    plt.figure(figsize=(8, 4))
    plt.scatter(df[labels['date']], df[labels['temp']], s=5, color='gray', label="Observed")
    plt.plot(df[labels['date']], T_pred, label="Predicted", color='tomato', linewidth=1)
    plt.ylabel("Air temperature (Celsius)")
    plt.legend()
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    return file_name

# TODO: Move to testing files.
if __name__ == '__main__':
    df = load_data("private/Verona_Temp_Rain.csv", end_date='2025-01-01')
    T_pred = model_temp(df)
    plot_temp(df, 'private/air_temp_plot.png', T_pred)
