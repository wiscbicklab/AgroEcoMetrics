from pathlib import Path
from typing import Optional, Tuple
from scipy.stats import circmean

import numpy as np
import pandas as pd


####    Private UTIL FUNCTIONS    ####
def __csv_file_exists(file_path: Path) -> bool:
    """
    Checks if a Path is a valid path to an existing .CSV file.
    
    Args:
        file_path: The Path to the file being checked.
    
    Returns:
        True if the csv file at the given Path already exists and False otherwise.

    Raises:
        TypeError: If file_path is not a Path object.
        ValueError: If the file is not a csv file.
        FileNotFoundError: If the parent directory of the file does not exist.

    """
    # Check Parameters for validity
    if not isinstance(file_path, Path):
        raise TypeError("file_path must be a pathlib.Path object.")
    if file_path.suffix.lower() != ".csv":
        raise ValueError("The filename must end with '.csv'.")
    if not file_path.parent.exists():
        raise FileNotFoundError(f"The directory '{file_path.parent}' does not exist.")
    
    # Return if the file path exists
    return file_path.exists()

# Data File Functions
def load_data_csv(
        file_path: Path, 
        date_time_column: str,
        date_time_format: str = "%m/%d/%Y %I:%M %p",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
    '''
    Loads data from a csv file into a DataFrame while filtering and cleaning the data.

    Loads data from a csv file into a DataFrame. Strips the column names of whitespace on either end and removes apostrophes.
    Filters data by the given dates. If no start or end date is specified the oldest and newest dates in the data are used respectively.
    Adds columns for the date_norm, DOY, and Year.
        Column names can be found by running AEM.settings.calc_calculation_labels().
        date_norm stores the date normalized to contain the same time, midnight.
        doy stores the number of days since January 1st, where January 1st = 0 and December 31st = 364 or 365 during a leap year.
        year stores the current year.

    Args:
        file_path: The path to your csv data file.
        start_date: Optional. The date to start filtering from (Inclusive).
        end_date: Optional. The date to stop filtering on (Inclusive).
        date_format: The date_format to use on the file.
    
    Returns:
        A DataFrame with the information from the csv file filtered by the specified dates.

    Raises:
        TypeError: If file_path is not a Path object.
        ValueError: If the file extension is not '.csv'.
        FileNotFoundError: If the file could not be found.
    '''

    # Check Parameters
    if not __csv_file_exists(file_path):
        raise FileNotFoundError(f"The file, {file_path},  that you are trying to load does not exist.")
    if start_date and end_date and pd.to_datetime(start_date) > pd.to_datetime(end_date):
        raise ValueError("The end date must be after the start date.\nStart Date:\t{start_date}\nEnd Date:\t{end_date}")
    
    # Read the data from the given csv file and make the formatting consistent
    df = None
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.replace("'", "")

    if date_time_column not in df.columns:
        raise ValueError(f"The given date column name, {date_time_column}, was not found in the csv data!")
    

    df[date_time_column] = pd.to_datetime(df[date_time_column], format=date_time_format)

    # Filter data using the start and end dates
    if start_date:
        df = df[df[date_time_column] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df[date_time_column] <= pd.to_datetime(end_date)]

    # Adds a Year, DOY, and Normalized Date column to the df
    df['DOY'] = df[date_time_column].dt.dayofyear-1
    df['YEAR'] = df[date_time_column].dt.year
    df['NORMALIZED_DATE'] = pd.to_datetime(df[date_time_column], format=date_time_format).dt.normalize()


    return df.reset_index(drop=True)

def interpolate_df(
        df: pd.DataFrame, 
        col_names: Optional[list[str]] = None,
        method: str = "linear"
        ) -> pd.DataFrame:
    """
    Interpolates missing data within a DataFrame.

    Interpolates data using dataframes interpolate function. 
    If a list of column names are provided only those columns are interpolated.

    Args:
        df: The DataFrame to interpolate
        col_names: An optional list of columns names to interpolate the data for
        method: The dataframe interpolation method to use

    Returns:
        The DataFrame.

    Raises:
        KeyError: If one of the col_names was not found in the dataframe
    """
    if col_names:
        if len(col_names) == 0:
            raise ValueError("label_keys must containa list of keys or be none")
        for col in col_names:
            if col not in df.columns:
                raise KeyError(col + " was not found in the df")
            df[col] = df[col].interpolate(method=method)
    else:
        for col in df.columns.values:
            df[col] = df[col].interpolate(method=method)
    
    return df

def save_data_csv(
        df: pd.DataFrame,
        file_path: Path
    ) -> Path:
    """
    Saves your DataFrame to the given csv file.

    Args:
        df: The dataframe to be saved.
        file_path: A Path to where you would like to save the data.
    
    Returns:
        The Path to the newly saved file.

    Raises:
        TypeError: If file_path is not a Path object.
        ValueError: If the file extension is not '.csv'.
        FileNotFoundError: If the parent directory does not exist.
        FileExistsError: If the file already exists.
    """

    # Check Parameters
    if __csv_file_exists(file_path):
        print(f"The file {file_path} already exists.")
        usr_input = input("Are you sure you want OVERWRITE the file(y/n):  ")
        if not usr_input or usr_input != "y":
            raise FileExistsError(f"The file, {file_path}, already exists.")
    
    # Save DataFrame
    df.to_csv(file_path, index=False)

    return file_path


# Fletcher's Functions
def match_weather_datetime(
        weather_dt: np.ndarray,
        data_dt: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Matches each datetime in `data_datetime_col` to the closest datetime 
        in `weather_datetime_col`.

    Args:
        weather_dt: The np array of actual weather date_times
        data_dt:    The np array of date_times to find the closest match for
    
    Returns:
        A tuple of (original times, matched weather times, matched indices, time differences)
    """
    indices = np.searchsorted(weather_dt, data_dt, side="left")
    indices = np.minimum(indices, len(weather_dt) - 1)

    for i in range(len(indices)):
        if indices[i] > 0:
            before = weather_dt[indices[i] - 1]
            after = weather_dt[indices[i]]
            if abs(data_dt[i] - before) < abs(data_dt[i] - after):
                indices[i] -= 1

    matched_times = weather_dt[indices]
    diffs = np.abs(data_dt - matched_times)

    return data_dt, matched_times, indices, diffs
 
def get_weather_data_from_cols(
        weather_df: pd.DataFrame,
        weather_cols: list[str],
        indices: np.ndarray
    ) -> dict[str, list[float]]:
    """
    Extracts values from weather columns at given matched indices.

    Args:
        weather_df:   The DataFrame containing weather data
        weather_cols: The key names to be used in the dictionary
        indices:      The indices in the DataFrame to be used in the dictionary

    Returns:
        A dictionary of {column_name: values}.
    """
    return {
        col: [weather_df[col].iloc[i] for i in indices]
        for col in weather_cols
    }



