from pathlib import Path
from typing import Optional, Tuple
from scipy.stats import circmean


import numpy as np
import pandas as pd

from agroecometrics import settings


# Gets the acutal LABELS of columns based on the user settings
LABELS = settings.get_labels()

####    UTIL FUNCTIONS    ####
def csv_file_exists(file_path: Path):
    """
    Checks is the a Path to a csv file is valid
    
    Args:
        file_path: The Path to the csv file
    
    Returns:
        True if the csv file at the given Path already exists and False otherwise

    Raises:
        TypeError: If file_path is not a Path object.
        ValueError: If the file extension is not '.csv'.
        FileNotFoundError: If the parent directory does not exist.

    """
    if not isinstance(file_path, Path):
        raise TypeError("file_path must be a pathlib.Path object.")
    if file_path.suffix.lower() != ".csv":
        raise ValueError("The filename must end with '.csv'.")
    if file_path.parent.exists():
        return file_path.exists()
    else:
        raise FileNotFoundError(f"The directory '{file_path.parent}' does not exist.")

# Data File Functions
def load_data(
        file_path: Path, 
        date_format: str = LABELS['date_format'],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
    '''
    Loads data into a DataFrame from a given CSV file filtered by date and cleaned

    Loads the data from between the two specificed dates inclusive. If no start or end
        date is specified the oldest and newest dates in the data are used respectively.
        Adss a column with the Date normalized
        Adds a column with the DOY from January 1st. Ie, January 1st = 0 and December 31st = 364
        Adds a column with the Year in an integer representation

    Args:
        file_path: The path to your csv data file
        start_date: Optional. The date to start filtering from (Inclusive)
        end_date: Optional. The date to stop filtering on (Inclusive)
        date_format: The date_format to use on the file
    
    Returns:
        A DataFrame with the information from the csv file filtered by the specified dates

    Raises:
        TypeError: If file_path is not a Path object.
        ValueError: If the file extension is not '.csv'.
        FileNotFoundError: If the file could not be found.
    '''
    global LABELS

    # Check Parameters
    if not csv_file_exists(file_path):
        raise FileNotFoundError(f"The file, {file_path},  that you are trying to load does not exist.")
    if start_date and end_date and pd.to_datetime(start_date) > pd.to_datetime(end_date):
        raise ValueError("The end date must be after the start date.\nStart Date:\t{start_date}\nEnd Date:\t{end_date}")
    
    # Read the data from the given csv file and make the formatting consistent
    df = None
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.replace("'", "")
    df[LABELS['date_time']] = pd.to_datetime(df[LABELS['date_time']], format=date_format)

    # Filter data using the start and end dates
    if start_date:
        df = df[df[LABELS['date_time']] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df[LABELS['date_time']] <= pd.to_datetime(end_date)]

    # Adds a Year, DOY, and Normalized Date column to the df
    df[LABELS['doy']] = df[LABELS['date_time']].dt.dayofyear-1
    df[LABELS['year']] = df[LABELS['date_time']].dt.year
    df[LABELS['date_norm']] = pd.to_datetime(df[LABELS['date_time']], format=date_format).dt.normalize()


    return df.reset_index(drop=True)

def interpolate_missing_data(
        df: pd.DataFrame, 
        label_keys: Optional[list[str]] = None,
        method: str = "linear"
        ):
    """
    Interpolates missing data within a DataFrame. Interpolates LABELS based on
        list of keys in label_keys or all data is label_keys is None.

    Args:
        df:     The DataFrame to interpolate data on
        LABELS: A list of the label keys to interpolate data on.
        method: The pandas interpolation type to use on the data
    """
    global LABELS
    if label_keys:
        if len(label_keys) == 0:
            raise ValueError("label_keys must containa list of keys or be none")
        for key in label_keys:
            if key not in LABELS:
                raise KeyError(key + " was not found in the available LABELS")
            df[LABELS[key]].interpolate(method=method, inplace=True)
    else:
        for key in df.columns.values:
            df[LABELS[key]].interpolate(method=method, inplace=True)

def save_data(
        df: pd.DataFrame,
        file_path: Path
    ) -> Path:
    """
    Saves your DataFrame to the given csv file

    Args:
        df: The dataframe to be saved
        file_path: A Path to where you would like to save the data

    Raises:
        TypeError: If file_path is not a Path object.
        ValueError: If the file extension is not '.csv'.
        FileNotFoundError: If the parent directory does not exist.
        FileExistsError: If the file already exists and the user selects not to override it
    """
    global LABELS

    # Check Parameters
    if csv_file_exists(file_path):
        print("The file {file_path} already exists.")
        usr_input = input("Are you sure you want OVERWRITE the file(y/n):  ")
        if not usr_input or usr_input != "y":
            raise FileExistsError(f"The file, {file_path}, already exists.")
    
    # Save DataFrame
    df.to_csv(file_path, index=False)

    return file_path





