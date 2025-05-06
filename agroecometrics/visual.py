import os

import matplotlib.pyplot as plt

from agroecometrics import settings


# Gets the acutal labels of columns based on the user settings
labels = settings.get_labels()



def check_png_filename(file_name):
    if not isinstance(file_name, str):
        raise TypeError("file_name must be a string.")
    if not file_name.lower().endswith('.png'):
        raise ValueError("The filename must end with '.png'.")
    dir_name = os.path.dirname(file_name)
    if dir_name and not os.path.exists(dir_name):
        raise FileNotFoundError(f"The directory '{dir_name}' does not exist.")


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
    check_png_filename(file_name)

    # Plot actual vs predicted temperature
    plt.figure(figsize=(8, 4))
    plt.scatter(df[labels['date']], df[labels['temp']], s=5, color='gray', label="Observed")
    plt.plot(df[labels['date']], T_pred, label="Predicted", color='tomato', linewidth=1)
    plt.ylabel("Air temperature (Celsius)")
    plt.legend()
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    return file_name

def plot_rain(df, file_name):
    '''
    Creates a plot of rainfall and runoff over time.

    df: DataFrame with cumulative rainfall and runoff
    file_name: PNG file path for the plot
    return: file_name if successful
    '''
    check_png_filename(file_name)

    plt.figure(figsize=(6, 4))
    plt.plot(df[labels['date']], df['RAIN_SUM'], color='navy', label='Rainfall')
    plt.plot(df[labels['date']], df['RUNOFF_SUM'], color='tomato', label='Runoff')
    plt.ylabel('Rainfall or Runoff (mm)')
    plt.legend()
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    return file_name








