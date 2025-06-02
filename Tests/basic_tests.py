from pathlib import Path
import numpy as np
import agroecometrics as AEM
import pandas as pd

labels = AEM.settings.get_labels()

def air_temp_tests(df: pd.DataFrame, file_path: Path):
    """
    Tests Air temperature functions and saves air temperature plots
    
    Args:
        df: Dataframe containing the air temperature data to test models on
        file_path: A path to the location where the plot file will be saved
    
    Returns:
        True if the tests ran sucessfully and False otherwise
    """
    try:
        # Air Temperature models
        air_temp_pred = AEM.bio_sci_equations.model_air_temp(df2024)
        AEM.visualizations.plot_air_temp(df, air_temp_pred, file_path.with_name("air_temp_plot.png"))
        return True
    except Exception as e:
        print(e.with_traceback)
        return False

def evapo_transpiation_tests(df: pd.DataFrame, file_path: Path):
    """
    Tests evapotranspiration models and saves a plot with all the evapotranspiration models
    
    Args:
        df: Dataframe containing the required data to test models on
            Required Data: temp_min, temp_max, hmin, hmax, max_gust, DOY, pmin, and pmax.
        file_path: A path to the location where the plot file will be saved
    
    Returns:
        True if the tests ran sucessfully and False otherwise
    """
    global labels

    # Set Latitude and Altitude for Evapo Models
    latitude = 34.0 # Degrees
    altitude = 350.0 # m
    try:
        # Get data from data file
        tmin = df[labels["temp_min"]]
        tmax = df[labels["temp_max"]]
        hmin = df[labels["hmin"]]
        hmax = df[labels["hmax"]]
        wind_avg = df[labels["max_gust"]]*0.5*.44704
        doy = df["DOY"]
        pmin = df[labels["pmin"]]*100
        pmax = df[labels["pmax"]]*100
    except Exception as e:
        print("Error getting data from file, unable to run evapotranspiration model tests")
        print(e.with_traceback)
        return False

    try:
        # Calculate evapo-transpiration models
        evapo_models = np.array([
        AEM.bio_sci_equations.dalton(tmin, tmax, hmin, hmax, wind_avg),
        AEM.bio_sci_equations.penman(tmin, tmax, hmin, hmax, wind_avg),
        AEM.bio_sci_equations.romanenko(tmin, tmax, hmin, hmax),
        AEM.bio_sci_equations.jensen_haise(tmin, tmax, doy, latitude),
        AEM.bio_sci_equations.hargreaves(tmin, tmax, doy, latitude),
        AEM.bio_sci_equations.penman_monteith(tmin, tmax, hmin, hmax, wind_avg, pmin, pmax, doy, latitude, altitude)
        ])
        # Create model Labels for the plot
        evapo_model_labels = ["Dalton", "Penman", "Romanenko", "Jensen-Haise", "Hargreaves", "Penman_Monteith"]
        # Plot Mmodel data
        AEM.visualizations.plot_evapo_data(df, file_path.with_name("Evapo_All.png"), evapo_models, evapo_model_labels)
        return True
    except Exception as e:
        print(e.with_traceback)
        return False
    
def rainfall_runoff_tests(df: pd.DataFrame, file_path: Path):
    """
    Tests rainfall and runoff functions and saves a plot with the rainfall and runoff data
    
    Args:
        df: Dataframe containing the rainfall data for compute the models with
        file_path: A path to the location where the plot file will be saved
    
    Returns:
        True if the tests ran sucessfully and False otherwise
    """
    try:
        file_path = file_path.with_name("runoff_plot.png")
        AEM.bio_sci_equations.rainfall_runoff_to_df(df)
        AEM.visualizations.plot_rainfall(df, file_path)
        return True
    except Exception as e:
        print(e.with_traceback)
        return False

def soil_temp_tests(file_path: Path):
    """
    Test soil temperature functions and saves plots with soil temperature calculations
    
    Args:
        file_path: A path to the location where the plot file will be saved
    
    Returns:
        True if the tests ran sucessfully and False otherwise
    """
    try:
        T_depth = AEM.bio_sci_equations.soil_temp_at_depth(10)
        AEM.visualizations.plot_yearly_soil_temp(T_depth, file_path.with_name("soil_temp.png"))

        T_day, depths = AEM.bio_sci_equations.soil_temp_on_day(10, 500)
        AEM.visualizations.plot_day_soil_temp(T_day, depths, file_path.with_name("day150_temp.png"))

        doy_grid, z_grid, t_grid = AEM.bio_sci_equations.soil_temp_at_depth_on_day(500)
        AEM.visualizations.plot_3d_soil_temp(doy_grid, z_grid, t_grid, file_path.with_name("soil_temp_3d.png"))
        return True
    except Exception as e:
        print(e.with_traceback)
        return False

def gdd_tests(df: pd.DataFrame, file_path: Path):
    """
    Tests growing degree days functions and saves plots for gdd calculations

    Args:
        df: Dataframe containing the rainfall data for compute the models with
        file_path: A path to the location where the plot file will be saved
    
    Returns:
        True if the tests ran sucessfully and False otherwise
    """
    global labels
    try:
        temp_avg = df[labels["temp_avg"]]
    except Exception as e:
        print("Error getting data from file, unable to run gdd model tests")
        print(e.with_traceback)
        return False
    
    try:
        AEM.bio_sci_equations.gdd_to_df(df, temp_avg=temp_avg, temp_base=10)
        AEM.visualizations.plot_gdd(file_path.with_name("gdd_basic.png"))
        AEM.visualizations.plot_gdd_sum(file_path.with_name("gdd_basic_sum.png"))

        AEM.bio_sci_equations.gdd_to_df(df, temp_avg=temp_avg, temp_base=10, temp_opt=28, temp_upper=38)
        AEM.visualizations.plot_gdd(file_path.with_name("gdd_advanced.png"))
        AEM.visualizations.plot_gdd_sum(file_path.with_name("gdd_advanced_sum.png"))
        return True
    except Exception as e:
        print(e.with_traceback)
        return False

def photoperiod_test(file_path: Path):
    """
    Tests photoperiod functions and saves plots for photoperiod calculations

    Args:
        file_path: A path to the location where the plot file will be saved
    
    Returns:
        True if the tests ran sucessfully and False otherwise
    """
    global labels
    try:
        AEM.visualizations.plot_yearly_photoperiod(33.4, file_path.with_name("Yearly_photoperiod.png"))
        AEM.visualizations.plot_daily_photoperiod(np.array([1,90,180,270,360]), file_path.with_name("Daily_photoperiod.png"))
        return True
    except Exception as e:
        return False

def autoregression_tests(file_path: Path):
    try:
        AEM.visualizations.plot_autoregression()
        return True
    except Exception as e:
        print(e.with_traceback)
        return False

if __name__ == "__main__":
    # Get Data
    data_file = "Tests/Data/Marshfield_All_Data.csv"
    df = AEM.bio_sci_equations.load_data(data_file)
    df2023 = AEM.bio_sci_equations.load_data(data_file, start_date="2023-01-01", end_date="2023-12-31")
    df2024 = AEM.bio_sci_equations.load_data(data_file, start_date="2024-01-01", end_date="2024-12-31")
    df2025 = AEM.bio_sci_equations.load_data(data_file, start_date="2025-01-01", end_date="2025-12-31")

    # Save Folder
    folder = Path("Tests/Images/")

    air_temp_tests(df2024, folder)
    evapo_transpiation_tests(df2024, folder)
    rainfall_runoff_tests(df2024, folder)
    soil_temp_tests(folder)
    gdd_tests(df2024, folder)
    photoperiod_test(folder)


