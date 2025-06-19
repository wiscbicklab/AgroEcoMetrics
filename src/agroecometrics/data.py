from typing import Optional, Tuple
from scipy.stats import circmean


import numpy as np
import pandas as pd

from agroecometrics import settings


# Gets the acutal labels of columns based on the user settings
labels = settings.get_labels()



# Data File Functions
def load_data(
        file_path: str, 
        date_format: str = '%m/%d/%Y %I:%M %p',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
    '''
    Loads data into a DataFrame from a given CSV file filtered by date

    Args:
        file_path: The path to your csv data file
        start_date: Optional. The date to start filtering from (Inclusive)
        end_date: Optional. The date to stop filtering on (Inclusive)
        date_format: The date_format to use on the file
    
    Returns:
        A DataFrame with the information from the csv file filtered by the specified dates
    '''
    global labels
    # Check that the start date is not after the end date
    if start_date and end_date and pd.to_datetime(start_date) > pd.to_datetime(end_date):
        print("The start date must be on or before the end date")
        return
    
    # Read the data from the given csv file and make the formatting consistent
    df = None
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip().str.replace("'", "")
        df[labels['date']] = pd.to_datetime(df[labels['date']], format=date_format)
    except Exception as e:
        print("Exception occured trying to read file:\t" + file_path)
        print(e.with_traceback)
        return

    # Filter data using the start and end dates
    if start_date:
        df = df[df[labels['date']] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df[labels['date']] <= pd.to_datetime(end_date)]

    # Adds a Year and DOY column to the df
    df['DOY'] = df[labels['date']].dt.dayofyear
    df['YEAR'] = df[labels['date']].dt.year

    return df.reset_index(drop=True)

def interpolate_missing_data(
        df: pd.DataFrame, 
        label_keys: Optional[list[str]] = None,
        method: str = "linear"
        ):
    """
    Interpolates missing data within a DataFrame. Interpolates labels based on
        list of keys in label_keys or all data is label_keys is None.

    Args:
        df:     The DataFrame to interpolate data on
        labels: A list of the label keys to interpolate data on.
        method: The pandas interpolation type to use on the data
    """
    global labels
    if label_keys:
        if len(label_keys) == 0:
            raise ValueError("label_keys must containa list of keys or be none")
        for key in label_keys:
            if key not in labels:
                raise KeyError(key + " was not found in the available labels")
            df[labels[key]].interpolate(method=method, inplace=True)
    else:
        for key in labels.keys:
            df[labels[key]].interpolate(method=method, inplace=True)

# Get parameter predictions from data
def get_yearly_air_temp_params(
        df: pd.DataFrame,
        cutoff: float=0.05
    ) -> Tuple[float, float, float, float, int]:
    """
    Estimates air temperature parameters for yearly fluctuation given air temperature data
    
    Creates parameter estimates for the following equation:
        T(0, t) = T_ave + A(0) * sin[OMEGA * (t - t0)]
    Average temperature is predicted using the mean of the average temperature of all data
    Minimum temperature is predicted by averaging the temperature on each DOY and getting
        the lower percentile defined by the cutoff
    Maximum temperature is predicted by averaging the temperature on each DOY and getting
        the upper percentile defined by the 1 - cutoff
    Thermal Amplitude is calculated as half the difference between the calulated
        minimum and maximum temperature

    Args:
        df: DataFrame containing temperature data, must contain temp_avg as defined in settings.
        cutoff: A float ranging from 0 to 0.5 which determines what percentiles to use
            for minimum and maximum temperature calculations. The default cutoff=0.05 and would
            calculate 5th and 95th percentiles

    Returns:
        T_avg: The average temperature over all of the data
        T_min: The minimum temperature calculated as detailed above
        T_max: The maximum temperature calculated as detailed above
        thermal_amp: The thermal amplitude calculated as detailed above
        min_temp_doy: The approximated DOY that the minimum yearly temperature occurs
    """
    global labels

    # Calculate mean temperature and temperature amplitude
    avg_temp = df[labels["temp_avg"]].mean()
    main_temp, max_temp = df.groupby(by='DOY')[labels["temp_avg"]].mean().quantile([cutoff, 1.0-cutoff])
    thermal_amp = (max_temp - main_temp) / 2

    # Get the index DOY index
    doy_mins = df.groupby(by='YEAR')[labels["temp_avg"]].idxmin()
    min_temp_doy = np.round(df.loc[doy_mins, 'DOY'].apply(circmean).mean())


    return avg_temp, main_temp, max_temp, thermal_amp, min_temp_doy

def get_daily_air_temp_params(
        df: pd.DataFrame,
        date: str,
        date_format: str = '%m/%d/%Y %I:%M %p',
    ) -> Tuple[float, float, float, float, int]:
    """
    Estimates air temperature parameters for daily fluctuation on a given date
    
    Creates parameter estimates for the following equation:
        T(0, t) = T_ave + A(0) * sin[OMEGA * (t - t0)]
    Average temperature is predicted using the mean of all the temperature averages for the given date
    Minimum temperature is predicted by getting the minimum temperature for the given date
    Maximum temperature is predicted by getting the maximum temperature for the given date
    Thermal Amplitude is calculated as half the difference between the calulated
        minimum and maximum temperature

    Args:
        df: DataFrame containing temperature data, must contain 5_minute_temp as defined in settings.
        date: Is the date which parameters will be estimated for.
        date_format: Is the format that the date variable and data use for the date.

    Returns:
        T_avg: The average temperature for the given day
        T_min: The minimum temperature for the given day
        T_max: The maximum temperature for the given day
        thermal_amp: The thermal amplitude calculated as detailed above
        min_temp_time: The approximated time from midnight that the 
            minimum daily temperature occurs
    """
    global labels

    # Gets the data on the target date
    target_date = pd.to_datetime(date, format=date_format)
    daily_df = df[df[labels['date']].dt.date == target_date.date()]

    # Calculates daily average, minimum, and maximum temperature
    avg_temp = daily_df[labels['5_minute_temp']].mean()
    min_temp = min(daily_df[labels['5_minute_temp']])
    max_temp = max(daily_df[labels['5_minute_temp']])
    # Calculates thermal amplitude
    thermal_amp = (max_temp - min_temp) / 2.0

    # Find index of minimum temperature
    idx_min = daily_df[labels['5_minute_temp']].idxmin()
    min_time = daily_df.loc[idx_min, labels['date']].time()
    min_temp_time = min_time.hour * 60 + min_time.minute

    return avg_temp, min_temp, max_temp, thermal_amp, min_temp_time
