from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import datetime

import matplotlib.pyplot as plt


from agroecometrics import equations, settings

# Gets the acutal labels of columns based on the user settings
labels = settings.get_labels()


####    UTIL FUNCTIONS    ####

def check_png_filename(file_path: Path):
    """
    Validates that the provided file path ends with '.png' and that the directory exists.

    Args:
        file_path: A Path object representing the output file path.
    
    Raises:
        TypeError: If file_path is not a Path object.
        ValueError: If the file extension is not '.png'.
        FileNotFoundError: If the parent directory does not exist.
    """
    if not isinstance(file_path, Path):
        raise TypeError("file_path must be a pathlib.Path object.")
    if file_path.suffix.lower() != ".png":
        raise ValueError("The filename must end with '.png'.")
    if file_path.parent and not file_path.parent.exists():
        raise FileNotFoundError(f"The directory '{file_path.parent}' does not exist.")

def save_plot(file_path: Path):
    """
    Adds labels to plt figure if they exist, saves the plot to the given file_path, and
        closes the plt plot after saving

    Args:
        file_path: A Path object representing the output file path.

    Raises:
        TypeError: If file_path is not a Path object.
        ValueError: If the file extension is not '.png'.
        FileNotFoundError: If the parent directory does not exist.
    """
    # Validate input parameters
    check_png_filename(file_path)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:  # Only add legend if there are labeled elements
        plt.legend()
    plt.savefig(file_path, dpi=300, bbox_inches="tight", pad_inches=0.4)
    plt.close()


####    AIR TEMPERATURE PLOTS    ####

def plot_air_temp(df: pd.DataFrame, T_pred: np.ndarray, file_path: Path):
    """
    Creates a plot of air temperature over the time-frame of the loaded data

    Args:
        df: DataFrame with temperature data
        T_pred: Predicted temperature array
        file_path: A Path object representing the output file path.

    Returns: 
        The resolved file path where the plot was saved

    Raises:
        TypeError: If file_path is not a Path object.
        ValueError: If the file extension is not '.png'.
        FileNotFoundError: If the parent directory does not exist.
    """
    global labels
    
    # Validate input parameters
    check_png_filename(file_path)

    # Plot actual vs predicted temperature
    plt.figure(figsize=(8, 4))
    plt.scatter(df[labels["date"]], df[labels["temp_avg"]], s=5, color="gray", label="Observed")
    plt.plot(df[labels["date"]], T_pred, label="Predicted", color="tomato", linewidth=1)
    plt.ylabel("Air temperature (Celsius)")
    plt.xlabel("Date")
    
    save_plot(file_path)
    return file_path.resolve


####    SOIL TEMPERATURE PLOTS    ####

def plot_yearly_soil_temp(T_soil: np.ndarray, file_path: Path):
    """
    Creates a plot of modeled soil temperature over a year's time

    Args:
        T_soil:    Is the surface temperatures of the soil for each day of the year
        file_path: A Path object representing the output file path.

    Returns: 
        The filename where the plot was saved

    Raises:
        TypeError: If file_path is not a Path object.
        ValueError: If the file extension is not '.png'.
        FileNotFoundError: If the parent directory does not exist.
    """
    check_png_filename(file_path)

    doy = np.arange(1,366)

    # Create the plot
    plt.figure()
    plt.plot(doy,T_soil)
    plt.ylabel("Surface Soil Temperature (Celsius)")
    plt.xlabel("Day of Year")
   
    save_plot(file_path)
    return file_path

def plot_day_soil_temp(
        T_soil: np.ndarray,
        depths: np.ndarray,
        file_path: Path,
        t_pred: Optional[np.ndarray]=None):
    """
    Creates a plot of modeled soil temperature at different depths 

    Args:
        T_soil: Is the predicted Temperatures at the given depths
        depths: Are the depths in meters that the temperature predict
        file_path: A Path object representing the output file path.

    Returns: 
        The filename where the plot was saved

    Raises:
        TypeError: If file_path is not a Path object.
        ValueError: If the file extension is not '.png'.
        FileNotFoundError: If the parent directory does not exist.
    """
    # Create Plot
    plt.figure()
    plt.plot(T_soil, -depths)
    plt.ylabel("Soil temperature (Celsius)")
    plt.xlabel("Soil Depth (Centimeters)")
    
    
    save_plot(file_path)
    return file_path

def plot_3d_soil_temp(doy_grid: np.ndarray, z_grid: np.ndarray, 
                      t_grid: np.ndarray, file_path: Path):
    """
    Creates a 3d plot of soil temperature at different depths over the course of a year

    Args:
        doy_gird: A 2d np.ndarray with shape (Nz, 365) containing the day of year
                for each plot point.
        z_grid:   A 2d np.ndarray with shape (Nz, 365) containing the soil depth
                for each plot point.
        t_grid:   A 2d np.ndarray with shape (Nz, 365) containing the soil temperature
                for each plot point.
        file_path: A Path object representing the output file path.
    
    Returns: 
        The filename where the plot was saved

    Raises:
        TypeError: If file_path is not a Path object.
        ValueError: If the file extension is not '.png'.
        FileNotFoundError: If the parent directory does not exist.
    """
    check_png_filename(file_path)

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

    save_plot(file_path)
    return file_path

def plot_day_soil_temp_pred(
        air_temp: np.ndarray,
        pred_temp: np.ndarray,
        depth: float,
        file_path: Path,
    ):
    """
    Creates a plot of air temperature and the predicted soil temperature on a particular date

    Args:
        air_temp: Is the air temperature collected every 5 minutes
        pred_temp: Is the soil temperature prediction given in 5 minute intervals
        depth: Is the depth that the soil temperature was predicted at
        file_path: A Path object representing the output file path.

    Returns: 
        The filename where the plot was saved

    Raises:
        TypeError: If file_path is not a Path object.
        ValueError: If the file extension is not '.png'.
        FileNotFoundError: If the parent directory does not exist.
    """
    global labels
    
    # Validate input parameters
    check_png_filename(file_path)
    time_passed = np.arange(0, 1440, 5)
    time = time_passed / 60  # Converts to float hours: [0.0, 0.083, 0.167, ..., 23.917]

    # Plot actual vs predicted temperature
    plt.figure(figsize=(8, 4))
    plt.scatter(time, air_temp, s=5, color="gray", label="Observed")
    plt.plot(time, pred_temp, label="Predicted", color="tomato", linewidth=1)
    plt.ylabel(f"Soil temperature at {depth}cm  (Celsius) and Air temperature  (Celsius)")
    plt.xlabel("Time (Hours)")
    plt.xticks(ticks=[0, 6, 12, 18, 24], labels=["12 AM", "6 AM", "12 PM", "6 PM", "12 AM"])

    
    save_plot(file_path)
    return file_path.resolve

def plot_3d_day_soil_temp_pred(
        time_grid: np.ndarray,
        depth_grid: np.ndarray,
        temp_grid: np.ndarray,
        file_path: Path,
    ):
    """
    Creates a 3D plot of predicted soil temperature over the course of a single day at different depths.

    Args:
        time_grid: A 2D numpy array of shape (n_depths, n_times), where each value is time in minutes.
        depth_grid: A 2D numpy array matching time_grid, where each value is the depth in cm.
        temp_grid: A 2D numpy array of predicted soil temperatures (same shape as time_grid).
        file_path: A Path object representing the output file path.

    Returns:
        The filename where the plot was saved

    Raises:
        TypeError: If file_path is not a Path object.
        ValueError: If the file extension is not '.png'.
        FileNotFoundError: If the parent directory does not exist.
    """
    check_png_filename(file_path)

    # Convert minutes to hours for plotting
    hours_grid = time_grid / 60
    depth_grid = -depth_grid  # Flip so depth increases downward

    # Create figure and axis
    fig = plt.figure(figsize=(10, 6), dpi=100, constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')

    # Surface plot
    surf = ax.plot_surface(hours_grid, depth_grid, temp_grid, cmap="viridis", antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=20)

    # Optional: wireframe overlay
    ax.plot_wireframe(hours_grid, depth_grid, temp_grid, linewidth=0.5, color='k', alpha=0.3)

    # Axis labels
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Soil Depth [cm]")
    ax.set_zlabel("Soil Temperature (°C)")

    # Set viewing angle
    ax.view_init(elev=30, azim=35)

    # Save and return path
    save_plot(file_path)



####    RAILFALL PLOTS    ####

def plot_rainfall(df: pd.DataFrame, file_path: Path):
    """
    Creates a plot of rainfall and runoff over time.

    Args:
        df: DataFrame with cumulative rainfall and runoff
        file_path: A Path object representing the output file path.

    Returns: 
        The filename where the plot was saved

    Raises:
        TypeError: If file_path is not a Path object.
        ValueError: If the file extension is not '.png'.
        FileNotFoundError: If the parent directory does not exist.
    """
    global labels

    check_png_filename(file_path)

    plt.figure(figsize=(6, 4))
    plt.plot(df[labels["date"]], df["RAIN_SUM"], color="navy", label="Rainfall")
    plt.plot(df[labels["date"]], df["RUNOFF_SUM"], color="tomato", label="Runoff")
    plt.ylabel("Rainfall or Runoff (mm)")

    save_plot(file_path)
    return file_path


####    EVAPOTRANSPIRATION PLOTS    ####

def plot_evapo_data(df: pd.DataFrame, file_path: Path, model_data: np.ndarray,
                     model_labels: list[str]):
    """
    Creates a plot of Evapotranspiration model data predictions

    Args:
        df: The DataFrame that the evapotranspiration models were run on.
        file_path: A Path object representing the output file path.
    
    Returns: 
        The filename where the plot was saved

    Raises:
        TypeError: If file_path is not a Path object.
        ValueError: If the file extension is not '.png'.
        FileNotFoundError: If the parent directory does not exist.
    """
    global labels

    # Generates a new Plot
    plt.figure(figsize=(10,4))

    # Check Argument Correctness
    check_png_filename(file_path)
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
    
    save_plot(file_path)
    return file_path


####    GROWING DEGREE DAYS PLOTS    ####

def plot_gdd(df: pd.DataFrame, file_path: Path):
    """
    Creates a plot of the Growing Degree Days that occured over each time segment in the data

    Args:
        df: The DataFrame that Growing Degree Days was calculated on.
                Must also contain "GROWING_DEGREE_DAYS" column generated by model
        file_path: A Path object representing the output file path.

    Returns:
        The filename where the plot was saved

    Raises:
        TypeError: If file_path is not a Path object.
        ValueError: If the file extension is not '.png'.
        FileNotFoundError: If the parent directory does not exist.
    """
    global labels

    check_png_filename(file_path)

    # Extract GDD data
    gdd_data = df[labels['gdd']]

    plt.figure(figsize=(6, 4))
    plt.plot(df[labels["date"]], gdd_data)
    plt.xlabel("Date")
    plt.ylabel(f'Growing degree days {chr(176)}C-d)')

    save_plot(file_path)
    
def plot_gdd_sum(df: pd.DataFrame, file_path: Path):
    """
    Creates a plot of the Cumulative Growing Degree Days that occured in the data


    Args:
        df: The DataFrame that Growing Degree Days was calculated on.
                Must also contain "GROWING_DEGREE_DAYS_SUM" column generated by model
        file_path: A Path object representing the output file path.

    Returns:
        The filename where the plot was saved

    Raises:
        TypeError: If file_path is not a Path object.
        ValueError: If the file extension is not '.png'.
        FileNotFoundError: If the parent directory does not exist.
    """
    global labels

    check_png_filename(file_path)

    # Extract GDD data
    gdd_sum_data = df[labels['gdd_sum']]

    plt.figure(figsize=(6,2))
    plt.plot(df[labels["date"]], gdd_sum_data)
    plt.xlabel("Date")
    plt.ylabel(f'Growing degree days sum {chr(176)}C-d)')

    save_plot(file_path)
    
####    PHOTOPERIOD PLOTS    ####

def plot_yearly_photoperiod(latitude: float, file_path: Path):
    """
    Creates a plot of the photoperiod at a specified latitude over a year's time

    Args:
        latitude: Latitude in decimal degress. Where the northern Hemisphere is positive
        file_path: A Path object representing the output file path.

    Returns:
        The filename where the plot was saved

    Raises:
        TypeError: If file_path is not a Path object.
        ValueError: If the file extension is not '.png'.
        FileNotFoundError: If the parent directory does not exist.
    """
    global labels

    check_png_filename(file_path)

    doy = np.arange(1,366)

    # Set up plot with Title and axes
    plt.figure(figsize=(6,4))
    plt.title('Latitude:' + str(latitude))
    plt.xlabel('Day of the year', size=14)
    plt.ylabel('Photoperiod (hours per day)', size=14)

    # Calulate photoperiods and adds them to the plot
    photoperiods, __, __, __, __, __ = equations.photoperiod_at_latitude(latitude, doy)

    plt.plot(doy, photoperiods, color='k')

    save_plot(file_path)

def plot_daily_photoperiod(doys: np.ndarray, file_path: Path):
    """
    Creates a plot of the photoperiod from -45 to 45 degree latitude on the given day of year

    Args:
        doys: A np.ndarray of the days of year (0-365) where January 1st is 0 and 365 to perform the calculation
        file_path: A Path object representing the output file path.

    Returns:
        The filename where the plot was saved

    Raises:
        TypeError: If file_path is not a Path object.
        ValueError: If the file extension is not '.png'.
        FileNotFoundError: If the parent directory does not exist.
    """
    global labels

    check_png_filename(file_path)

    # Set up plot with Title and axes
    plt.figure(figsize=(6,4))
    plt.xlabel('Latitude (decimal degrees)', size=14)
    plt.ylabel('Photoperiod (hours per day)', size=14)

    # Calculated photoperiods and adds them to the plot
    latitudes = np.linspace(-45, 45, num=180)

    # Loop over each day and plot
    for doy in doys:
        photoperiod, *_ = equations.photoperiod_on_day(latitudes, doy)
        plt.plot(latitudes, photoperiod, label=f'DOY {doy}')
    
    save_plot(file_path)







