from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


from agroecometrics import equations

# Gets the acutal labels of columns based on the user settings


####    UTIL FUNCTIONS    ####

def __check_png_filename(file_path: Path):
    """
    Validates that the provided file path ends with '.png' and that the directory exists.

    Args:
        file_path: A Path object representing the output file path.

    Returns:
        True if the file_path is a png file whose parent folder exists
    
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
    return True

def __save_plot(file_path: Path) -> Path:
    """
    Prepares plot to be saved and saves it.

    Adds labels to plt figure if they exist, saves the plot to the given file_path, and
        closes the plt plot after saving

    Args:
        file_path: A Path object representing the output file path.

    Returns:
        The resolved file path where the plot was saved

    Raises:
        TypeError: If file_path is not a Path object.
        ValueError: If the file extension is not '.png'.
        FileNotFoundError: If the parent directory does not exist.
    """
    # Validate input parameters
    __check_png_filename(file_path)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:  # Only add legend if there are labeled elements
        plt.legend()
    plt.savefig(file_path, dpi=300, bbox_inches="tight", pad_inches=0.4)
    plt.close()

    return file_path.resolve()


####    AIR TEMPERATURE PLOTS    ####

def plot_air_temp(
        air_temps: np.ndarray,
        pred_temps: np.ndarray,
        date_times: np.ndarray,
        file_path: Path
    ) -> Path:
    """
    Creates a plot of air temperature and predicted air temperatures over the time.

    The actual air temperature are plotted as a scatter plot and the predicted air temperatures are ploted as a graph.

    Args:
        air_temps: A numpy array of actual air temperatures. (°C)
        pred_temps: A numpy array of the predicted air temperatures. (°C)
        date_times: A numpy array of date times corrosponding to the air temperatures.
        file_path: A Path object representing the output file path.

    Returns: 
        The resolved file path where the plot was saved.

    Raises:
        TypeError: If file_path is not a Path object.
        ValueError: If the file extension is not '.png'.
        FileNotFoundError: If the parent directory does not exist.
    """
    __check_png_filename(file_path)

    # Plot actual vs predicted temperature
    plt.figure(figsize=(8, 4))
    plt.scatter(date_times, air_temps, s=5, color="red", label="Observed")
    plt.plot(date_times, pred_temps, label="Predicted", color="blue", linewidth=1)
    plt.ylabel("Air temperature (Celsius)")
    plt.xlabel("Date")
    
    return __save_plot(file_path)


####    SOIL TEMPERATURE PLOTS    ####

def plot_yearly_soil_temp(
        soil_temps: np.ndarray,
        file_path: Path
    ) -> Path:
    """
    Creates a plot of modeled soil temperature over a year's time

    Args:
        soil_temps: A numpy array of soil temperatures for each day of the year. (°C)
        file_path: A Path object representing the output file path.

    Returns: 
        The filename where the plot was saved

    Raises:
        TypeError: If file_path is not a Path object.
        ValueError: If the file extension is not '.png'.
        FileNotFoundError: If the parent directory does not exist.
    """
    # Check Parmeters
    __check_png_filename(file_path)

    # Create the soil temperature plot
    doy = np.arange(0,365)
    plt.figure()
    plt.plot(doy,soil_temps)
    plt.ylabel("Surface Soil Temperature (Celsius)")
    plt.xlabel("Day of Year")
   
    return __save_plot(file_path)

def plot_daily_soil_temp(
        soil_temps: np.ndarray,
        depth: int,
        file_path: Path,
    ) -> Path:
    """
    Creates a plot of modeled soil temperature at a given depth.

    Args:
        soil_temp: A numpy array of the predicted soil temperatures. (°C)
        depth: The depth that the soil temperature predictions are made at. (cm)
        file_path: A Path object representing the output file path.

    Returns: 
        The filename where the plot was saved

    Raises:
        TypeError: If file_path is not a Path object.
        ValueError: If the file extension is not '.png'.
        FileNotFoundError: If the parent directory does not exist.
    """
    # Check Parameters
    __check_png_filename(file_path)

    # Create Plot
    plt.figure()
    plt.plot(depth, soil_temps)
    plt.ylabel("Soil temperature (Celsius)")
    plt.xlabel("Soil Depth (Centimeters)")
    
    return __save_plot(file_path)

def plot_yearly_3d_soil_temp(
        doy_grid: np.ndarray,
        depth_grid: np.ndarray, 
        temp_grid: np.ndarray,
        file_path: Path
    ) -> Path:
    """
    Creates a 3d plot of soil temperature at different depths over the course of a year.

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
    # Check Parameters
    __check_png_filename(file_path)

    # Create Plot
    fig = plt.figure(figsize=(10, 6), dpi=80, constrained_layout=True) # 10 inch by 6 inch dpi = dots per inch
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(doy_grid, depth_grid, temp_grid, cmap="viridis", antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=20)
    frame = surf = ax.plot_wireframe(doy_grid, depth_grid, temp_grid, linewidth=0.5, color="k", alpha=0.5)

    # Label x,y, and z axis
    ax.set_xlabel("Day of the year")
    ax.set_ylabel("Soil depth [cm]")
    ax.set_zlabel("Soil temperature (Celsius)")

    # Set position of the 3D plot
    ax.view_init(elev=30, azim=35)

    return __save_plot(file_path)

def plot_modeled_soil_temp(
        air_temp: np.ndarray,
        pred_temps: np.ndarray,
        depths: np.ndarray,
        file_path: Path,
        soil_temp: Optional[np.ndarray] = None,
        colors: list[str] = ["#0072b2", "#009e73", "#cc79a7", "#d55e00", "#F0E442"],
    ) -> Path:
    """
    Creates a plot of air temperature and the predicted soil temperature on a particular date

    Creates a plot showing the given air temperature as a scatter plot, the actual soil temperature
    as a scatter plot if provided, and a graphed approximation of the predicted soil temperature at
    each of the provided depths. The first provided color is used for the air temperature, the next 
    is used for actual soil temperatures if provided, and the next available is used for the predicted
    soil temperatures.

    Args:
        air_temp: A numpy array of the air temperature collected over a single day.
        pred_temps: A 2d numpy array where each row is a set of predicted soil temperatures corrosponding
            to the provided air temperatures. Each row corrospondes to the depth in depths.
        depths: The depth that the soil temperature was predicted at
        file_path: A Path object representing the output file path.
        soil_temp: A numpy array of the collected soil temperature at the same interval as air temperature.
        colors: A list of strings providing the color codes to use for the plot. 5 color blind friendly colors are provided.

    Returns: 
        The filename where the plot was saved

    Raises:
        TypeError: If file_path is not a Path object.
        ValueError: If the file extension is not '.png'.
        FileNotFoundError: If the parent directory does not exist.
        ValueError: If the number of predicted temperatures per set is not equal to the number of provided air temperatures.
        ValueError: If the number of predicted temperature sets is not equal to the number of provided depths.
        ValueError: If the number of required colors is greater than the number of available colors.
    """    
    data_len = len(air_temp)
    # Validate input parameters
    __check_png_filename(file_path)
    if(pred_temps.shape[1] != len(air_temp)):
        raise ValueError(f"{len(air_temp)} air temperatures provided, but {pred_temps.shape[1]} predicted temps provided per depth")
    if(pred_temps.shape[0] != len(depths)):
        raise ValueError(f"{len(depths)} depths provided, but {pred_temps.shape[0]} predicted temps provided")
    if(soil_temp):
        if(pred_temps.shape[0] + 2 > len(colors)):
            raise ValueError("Not enought colors provided")
        elif(pred_temps.shape[0] + 1 > len(colors)):
            raise ValueError("Not enought colors provided")

    
    # Create list of measurement times
    times = np.arange(0, 1440, 1440/len(air_temp))
    
    color_num = 0

    # Add air temperature scatter plot
    plt.scatter(times, air_temp, s=5, color=colors[color_num], label="Air Temperature")
    color_num += 1

    # Add actual soil temperature scatter plot
    if soil_temp is not None:
        plt.scatter(times, soil_temp, s=5, color=colors[color_num], label="Soil Temp")
        color_num += 1

    # Create Soil Temperature Graph
    plt.figure(figsize=(8, 4))
    for i in range(len(depths)):
        depth = depths[i]
        pred_temp = pred_temps[i]
        plt.plot(times, pred_temp, label=f"Predicted: {depth}m", color=colors[color_num], linewidth=1)
        color_num += 1

    

    # Set labels
    plt.ylabel("Temperature  (Celsius)")
    plt.xlabel("Time (Hours)")
    plt.xticks(ticks=[0, 360, 720, 1080, 1440], labels=["12 AM", "6 AM", "12 PM", "6 PM", "12 AM"])

    return __save_plot(file_path)

def plot_3d_modeled_soil_temp(
        time_grid: np.ndarray,
        depth_grid: np.ndarray,
        temp_grid: np.ndarray,
        file_path: Path,
    ) -> Path:
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
        ValueError: If the file extension is not '.png' or if the needed label(s) isn't found in the df
        FileNotFoundError: If the parent directory does not exist.
    """
    __check_png_filename(file_path)

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
    return __save_plot(file_path)


####    EVAPOTRANSPIRATION PLOTS    ####

def plot_evapo_data(
        pred_evapos: np.ndarray,
        date_times: np.ndarray,
        file_path: Path,
        model_labels: list[str],
        colors: list[str] = ["#0072b2", "#009e73", "#cc79a7", "#d55e00", "#F0E442"],
    ) -> Path:
    """
    Creates a plot of different evapotraspiration data over time..

    Args:
        pred_evapos: A 2d Numpy array of evapotranspirations, each row represents a different model. (mm/day)
        date_times: A numpy array of date times corrosponding to the evapotranspiration data of each model.
        file_path: A Path object representing the output file path.
        model_labels: A list of labels corrosponding to each row of the provided evapotranspiration data.
        colors: A list of strings providing the color codes to use for the plot. 5 color blind friendly colors are provided.
    
    Returns: 
        The filename where the plot was saved

    Raises:
        TypeError: If file_path is not a Path object.
        ValueError: If the file extension is not '.png'.
        FileNotFoundError: If the parent directory does not exist.
        ValueError: If the length of evapotranspiration data per depth is not equal to the number of date times provided.
        ValueError: IF the number of model labels is not equal to the number of evapotranspiration models provided.
        ValueError: If the number of of depths provided is greater than the number of colors provided.
    """
    # Check params
    __check_png_filename(file_path)
    if(pred_evapos.shape[1] != len(date_times)):
        raise ValueError(f"{len(date_times)} Date times were provided, but {pred_evapos.shape[1]} data points for each evapotranspiration models provided.")
    if(pred_evapos.shape[0] != len(model_labels)):
        raise ValueError(f"Data for {pred_evapos.shape[0]} evapotransipation models provided, but {len(model_labels)} model labels provided.")
    if(pred_evapos.shape[0] > len(colors)):
        raise ValueError("Not enough colors provided")

    # Generates a new figure
    plt.figure(figsize=(10,4))

    # Add Evapotranspiration models to the figure
    for i in range(len(model_labels)):
        pred_evapo = pred_evapos[i]
        label = model_labels[i]
        plt.plot(date_times, pred_evapo, label=label)
    
    # Adds plot label
    plt.ylabel("Evapotranspiration (mm/day)")
    plt.xlabel("Date")
    
    return __save_plot(file_path)


####    RAILFALL PLOTS    ####

def plot_rainfall(
        rainfall: np.ndarray,
        runoff: np.ndarray,
        date_times: np.ndarray,
        file_path: Path
    ) -> Path:
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
    __check_png_filename(file_path)

    # Create plot with rain and runoff data
    plt.figure(figsize=(6, 4))
    plt.plot(date_times, rainfall, color="blue", label="Rainfall")
    plt.plot(date_times, runoff, color="red", label="Runoff")
    plt.ylabel("Rainfall or Runoff (mm)")

    return __save_plot(file_path)


####    GROWING DEGREE DAYS PLOTS    ####

def plot_gdd(
        gdd: np.ndarray,
        date_times: np.ndarray,
        file_path: Path
    ) -> Path:
    """
    Creates a plot of the Growing Degree Days that occured over each time segment in the data

    Args:
        gdd: The Growing Degree Days that occured 
        date_times: The datetime objects corresponding to the each actual and predicted temp.
        file_path: A Path object representing the output file path.

    Returns:
        The filename where the plot was saved

    Raises:
        TypeError: If file_path is not a Path object.
        ValueError: If the file extension is not '.png' or if the needed label(s) isn't found in the df
        FileNotFoundError: If the parent directory does not exist.
    """
    __check_png_filename(file_path)

    # Create plot
    plt.figure(figsize=(6, 4))
    plt.plot(date_times, gdd)
    plt.xlabel("Date")
    plt.ylabel(f'Growing degree days {chr(176)}C-d)')

    return __save_plot(file_path)
    
def plot_gdd_sum(
        gdd_sum: np.ndarray,
        date_times: np.ndarray,
        file_path: Path
    ) -> Path:
    """
    Creates a plot of the Cumulative Growing Degree Days that occured over the data

    Args:
        gdd_sum: The cummulative Growing Degree Days that have occured since the start of the data.
        date_times: The datetime objects corresponding to the each actual and predicted temp.
        file_path: A Path object representing the output file path.

    Returns:
        The filename where the plot was saved

    Raises:
        TypeError: If file_path is not a Path object.
        ValueError: If the file extension is not '.png' or if the needed label isn't found in the df
        FileNotFoundError: If the parent directory does not exist.
    """
    __check_png_filename(file_path)

    # Create plot
    plt.figure(figsize=(6,2))
    plt.plot(date_times, gdd_sum)
    plt.xlabel("Date")
    plt.ylabel(f'Growing degree days sum {chr(176)}C-d)')

    return __save_plot(file_path)
    

####    PHOTOPERIOD PLOTS    ####

def plot_yearly_photoperiod(lat: float, file_path: Path):
    """
    Creates a plot of the photoperiod at a specified latitude over a year's time. Not accurate near polar regions.

    Args:
        latitude: Latitude in decimal degress. Where the northern hemisphere is .
            positive and the southern hemisphere is negative.
        file_path: A Path object representing the output file path.

    Returns:
        The filename where the plot was saved

    Raises:
        TypeError: If file_path is not a Path object.
        ValueError: If the file extension is not '.png'.
        FileNotFoundError: If the parent directory does not exist.
    """
    # Check Parameters
    __check_png_filename(file_path)

    # Set up plot with Title and axes
    doy = np.arange(0,366)
    plt.figure(figsize=(6,4))
    plt.title('Latitude:' + str(lat))
    plt.xlabel('Day of the year', size=14)
    plt.ylabel('Photoperiod (hours per day)', size=14)

    # Calulate photoperiods and adds them to the plot
    photoperiods, __, __, __, __, __ = equations.photoperiod_at_lat(lat, doy)

    plt.plot(doy, photoperiods, color='k')

    return __save_plot(file_path)

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
    # Check Parameters
    __check_png_filename(file_path)

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
    
    return __save_plot(file_path)







