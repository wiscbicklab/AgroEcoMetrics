import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from agroecometrics import settings

# Gets the acutal labels of columns based on the user settings
labels = settings.get_labels()

####    UTIL FUNCTIONS    ####

def check_png_filename(file_name: str):
    if not isinstance(file_name, str):
        raise TypeError("file_name must be a string.")
    if not file_name.lower().endswith('.png'):
        raise ValueError("The filename must end with '.png'.")
    dir_name = os.path.dirname(file_name)
    if dir_name and not os.path.exists(dir_name):
        raise FileNotFoundError(f"The directory '{dir_name}' does not exist.")


####    AIR TEMPERATURE PLOTS    ####

def plot_air_temp(df: pd.DataFrame, file_name: str, T_pred: np.ndarray):
    """
    Creates a plot of air temperature over the time-frame of the loaded data

    Args:
        df: DataFrame with temperature data
        file_name: Filename for saved plot (PNG)
        T_pred: Predicted temperature array
    Return: 
        The file_name that the plot was saved at
    """
    global labels
    
    # Validate input parameters
    check_png_filename(file_name)

    # Plot actual vs predicted temperature
    plt.figure(figsize=(8, 4))
    plt.scatter(df[labels['date']], df[labels['temp']], s=5, color='gray', label="Observed")
    plt.plot(df[labels['date']], T_pred, label="Predicted", color='tomato', linewidth=1)
    plt.ylabel("Air temperature (Celsius)")
    plt.legend()
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()
    return file_name


####    SOIL TEMPERATURE PLOTS    ####

def plot_yearly_soil_temp(T_soil: np.ndarray, file_name: str):
    """
    Creates a plot of modeled soil temperature over a year's time

    Args:
        T_soil: Is the predicted Temperatures for the years
        file_name: Filename for saved plot (PNG)
    Return: 
        The file_name that the plot was saved at
    """
    global labels

    check_png_filename(file_name)

    doy = np.arange(1,366)

    # Create the plot
    plt.figure()
    plt.plot(doy,T_soil)
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    return file_name

def plot_day_temp(T_soil: np.ndarray, depths: np.ndarray, file_name: str):
    """
    Creates a plot of modeled soil temperature at different depths 

    Args:
        T_soil: Is the predicted Temperatures at the given depths
        depths: Are the depths in meters that the temperature predict
        file_name: Filename for saved plot (PNG)
    Return: 
        The file_name that the plot was saved at
    """
    # Create Plot
    plt.figure()
    plt.plot(T_soil, -depths)
    plt.ylabel("Air temperature (Celsius)")
    plt.savefig(file_name, dpi=300, bbox_inches='tight')


####    RAILFALL PLOTS    ####

def plot_rainfall(df: pd.DataFrame, file_name: str):
    """
    Creates a plot of rainfall and runoff over time.

    df: DataFrame with cumulative rainfall and runoff
    file_name: PNG file path for the plot
    return: file_name if successful
    """
    global labels

    check_png_filename(file_name)

    plt.figure(figsize=(6, 4))
    plt.plot(df[labels['date']], df['RAIN_SUM'], color='navy', label='Rainfall')
    plt.plot(df[labels['date']], df['RUNOFF_SUM'], color='tomato', label='Runoff')
    plt.ylabel('Rainfall or Runoff (mm)')
    plt.legend()
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()
    return file_name


####    EVAPOTRANSPIRATION PLOTS    ####

def plot_evapo_data(df: pd.DataFrame, file_name: str, model_data: np.ndarray,
                     model_labels: list[str]):
    global labels

    # Generates a new Plot
    plt.figure(figsize=(10,4))

    # Check Argument Correctness
    check_png_filename(file_name)
    if len(model_data) != len(model_labels):
        raise ValueError("You must provide the same number of model labels and model data")


    # Loop through and plot data from different models
    for i in range(len(model_data)):
        data = model_data[i]
        data_label = model_labels[i]
        plt.plot(df[labels['date']], data, label=data_label)
    
    # Adds plot label
    plt.ylabel('Evapotranspiration (mm/day)')
    plt.legend()
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()
    
    return file_name






