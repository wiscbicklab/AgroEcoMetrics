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
    if not file_name.lower().endswith(".png"):
        raise ValueError("The filename must end with '.png'")
    dir_name = os.path.dirname(file_name)
    if dir_name and not os.path.exists(dir_name):
        raise FileNotFoundError(f"The directory '{dir_name}' does not exist.")

def save_plot(file_name: str):
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:  # Only add legend if there are labeled elements
        plt.legend()
    plt.savefig(file_name, dpi=300, bbox_inches="tight", pad_inches=0.4)
    plt.close()


####    AIR TEMPERATURE PLOTS    ####

def plot_air_temp(df: pd.DataFrame, T_pred: np.ndarray, file_name: str):
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
    plt.scatter(df[labels["date"]], df[labels["temp"]], s=5, color="gray", label="Observed")
    plt.plot(df[labels["date"]], T_pred, label="Predicted", color="tomato", linewidth=1)
    plt.ylabel("Air temperature (Celsius)")
    plt.xlabel("Date")
    
    save_plot(file_name)
    return file_name


####    SOIL TEMPERATURE PLOTS    ####

def plot_yearly_soil_temp(T_soil: np.ndarray, file_name: str):
    """
    Creates a plot of modeled soil temperature over a year"s time

    Args:
        T_soil: Is the predicted Temperatures for the years
        file_name: Filename for saved plot (PNG)
    Return: 
        The file_name that the plot was saved at
    """
    check_png_filename(file_name)

    doy = np.arange(1,366)

    # Create the plot
    plt.figure()
    plt.plot(doy,T_soil)
    plt.ylabel("Surface Soil Temperature (Celsius)")
    plt.xlabel("Day of Year")
   
    save_plot(file_name)
    return file_name

def plot_day_soil_temp(T_soil: np.ndarray, depths: np.ndarray, file_name: str):
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
    plt.ylabel("Soil temperature (Celsius)")
    plt.xlabel("Soil Depth (Centimeters)")
    
    save_plot(file_name)
    return file_name

def plot_3d_soil_temp(doy_grid: np.ndarray, z_grid: np.ndarray, 
                      t_grid: np.ndarray, file_name: str):
    """
    
    """
    check_png_filename(file_name)

    # Create figure
    fig = plt.figure(figsize=(10, 6), dpi=80, constrained_layout=True) # 10 inch by 6 inch dpi = dots per inch

    # Get figure axes and convert it to a 3D projection
    ax = fig.add_subplot(111, projection="3d")


    # Add surface plot to axes. Save this surface plot in a variable
    surf = ax.plot_surface(doy_grid, z_grid, t_grid, cmap="viridis", antialiased=False)

    # Add colorbar to figure based on ranges in the surf map.
    fig.colorbar(surf, shrink=0.5, aspect=20)

    # Wire mesh
    frame = surf = ax.plot_wireframe(doy_grid, z_grid, t_grid, linewidth=0.5, color="k", alpha=0.5)

    # Label x,y, and z axis
    ax.set_xlabel("Day of the year")
    ax.set_ylabel("Soil depth [cm]")
    ax.set_zlabel("Soil temperature (Celsius)")

    # Set position of the 3D plot
    ax.view_init(elev=30, azim=35) # elevation and azimuth. Change their value to see what happens.

    save_plot(file_name)
    return file_name


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
    plt.plot(df[labels["date"]], df["RAIN_SUM"], color="navy", label="Rainfall")
    plt.plot(df[labels["date"]], df["RUNOFF_SUM"], color="tomato", label="Runoff")
    plt.ylabel("Rainfall or Runoff (mm)")

    save_plot(file_name)
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
        plt.plot(df[labels["date"]], data, label=data_label)
    
    # Adds plot label
    plt.ylabel("Evapotranspiration (mm/day)")
    plt.xlabel("Date")
    
    save_plot(file_name)
    return file_name






