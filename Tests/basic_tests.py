from pathlib import Path
import traceback
import numpy as np
import agroecometrics as AEM
import pandas as pd

LABELS = AEM.settings.get_labels()

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
        air_temp_pred = AEM.equations.model_air_temp(df)
        AEM.visualizations.plot_air_temp(df, air_temp_pred, file_path.joinpath("air_temp_plot.png"))
        return True
    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()
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
    global LABELS

    # Set Latitude and Altitude for Evapo Models
    latitude = 34.0 # Degrees
    altitude = 350.0 # m
    try:
        # Get data from data file
        tmin = df[LABELS["temp_min"]]
        tmax = df[LABELS["temp_max"]]
        hmin = df[LABELS["hmin"]]
        hmax = df[LABELS["hmax"]]
        wind_avg = df[LABELS["max_gust"]]*0.5*.44704
        doy = df[LABELS['doy']]
        pmin = df[LABELS["pmin"]]*100
        pmax = df[LABELS["pmax"]]*100
    except Exception as e:
        print("Error getting data from file, unable to run evapotranspiration model tests")
        print("ERROR:", e)
        traceback.print_exc()
        return False

    try:
        # Calculate evapo-transpiration models
        evapo_models = np.array([
        AEM.equations.dalton(tmin, tmax, hmin, hmax, wind_avg),
        AEM.equations.penman(tmin, tmax, hmin, hmax, wind_avg),
        AEM.equations.romanenko(tmin, tmax, hmin, hmax),
        AEM.equations.jensen_haise(tmin, tmax, doy, latitude),
        AEM.equations.hargreaves(tmin, tmax, doy, latitude),
        AEM.equations.penman_monteith(tmin, tmax, hmin, hmax, wind_avg, pmin, pmax, doy, latitude, altitude)
        ])
        # Create model LABELS for the plot
        evapo_model_LABELS = ["Dalton", "Penman", "Romanenko", "Jensen-Haise", "Hargreaves", "Penman_Monteith"]
        # Plot Mmodel data
        AEM.visualizations.plot_evapo_data(df, file_path.joinpath("Evapo_All.png"), evapo_models, evapo_model_LABELS)
        return True
    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()
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
        file_path = file_path.joinpath("runoff_plot.png")
        AEM.equations.rainfall_runoff_to_df(df)
        AEM.visualizations.plot_rainfall(df, file_path)
        return True
    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()
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
        T_depth = AEM.equations.yearly_soil_temp(10)
        AEM.visualizations.plot_yearly_soil_temp(T_depth, file_path.joinpath("soil_temp.png"))

        T_day, depths = AEM.equations.daily_soil_temp(10, 500)
        AEM.visualizations.plot_daily_soil_temp(T_day, depths, file_path.joinpath("day150_temp.png"))

        doy_grid, z_grid, t_grid = AEM.equations.yearly_3d_soil_temp(500)
        AEM.visualizations.plot_yearly_3d_soil_temp(doy_grid, z_grid, t_grid, file_path.joinpath("soil_temp_3d.png"))
        return True
    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()
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
    global LABELS
    try:
        temp_avg = df[LABELS["temp_avg"]]
    except Exception as e:
        print("Error getting data from file, unable to run gdd model tests")
        print("ERROR:", e)
        traceback.print_exc()
        return False
    
    try:
        AEM.equations.gdd_to_df(df, temp_avg, 10.0)
        AEM.visualizations.plot_gdd(df, file_path.joinpath("gdd_basic.png"))
        AEM.visualizations.plot_gdd_sum(df, file_path.joinpath("gdd_basic_sum.png"))

        AEM.equations.gdd_to_df(df, temp_avg, 10.0, temp_opt=22.0, temp_upper=28.0)
        AEM.visualizations.plot_gdd(df, file_path.joinpath("gdd_advanced.png"))
        AEM.visualizations.plot_gdd_sum(df, file_path.joinpath("gdd_advanced_sum.png"))

        return True
    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()
        return False

def photoperiod_test(file_path: Path):
    """
    Tests photoperiod functions and saves plots for photoperiod calculations

    Args:
        file_path: A path to the location where the plot file will be saved
    
    Returns:
        True if the tests ran sucessfully and False otherwise
    """
    global LABELS
    try:
        AEM.visualizations.plot_yearly_photoperiod(33.4, file_path.joinpath("Yearly_photoperiod.png"))
        AEM.visualizations.plot_daily_photoperiod(np.array([1,90,180,270,360]), file_path.joinpath("Daily_photoperiod.png"))
        return True
    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()
        return False

def air_to_soil_temp_test(df: pd.DataFrame, file_path: Path):
    """
    Tests functions for predicting soil temperature from air temperature and saving plots

    Args:
        file_path: A path to the location where the plot file will be saved

    Returns:
        True if the tests ran sucessfully and False otherwise
    """
    global LABELS 
    try:
        for i in range(10, 31):
            temp_predictions, air_temp, soil_temp = AEM.equations.model_soil_temp(df, 10, f"07/{i}/2024 12:00 AM")
            AEM.visualizations.plot_modeled_soil_temp(air_temp, temp_predictions, 10, file_path.joinpath(f"July_Soil_Predictions/July_{i}_soil_temp_predictions.png"), soil_temp)

        time_grid, depth_grid, temp_grid = AEM.equations.model_3d_soil_temp(df, 10, "07/21/2024 12:00 AM")
        AEM.visualizations.plot_3d_modeled_soil_temp(time_grid, depth_grid, temp_grid, file_path.joinpath("3d_soil_temp_predictions.png"))
        return True
    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()
        return False

def hydraulic_conductivity_test():
    """
    Tests functions for predicting soil temperature from air temperature and saving plots

    Args:
        file_path: A path to the location where the plot file will be saved

    Returns:
        True if the tests ran sucessfully and False otherwise
    """
    global LABELS 
    try:
        calc = AEM.equations.hydraulic_conductivity(.1, .1, .1, .1, 2)
        if calc != 0.1:
            return False
        return True
    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()
        return False

def data_tests(df: pd.DataFrame, file_path: Path):
    df_dict = AEM.data.df_as_dict(df)
    return True



if __name__ == "__main__":
    # Get Data
    data_file = Path("Tests/Data/Marshfield_All_Data.csv")
    df = AEM.data.load_data_csv(data_file)
    df2023 = AEM.data.load_data_csv(data_file, start_date="2023-01-01", end_date="2023-12-31")
    df2024 = AEM.data.load_data_csv(data_file, start_date="2024-01-01", end_date="2024-12-31")
    df2025 = AEM.data.load_data_csv(data_file, start_date="2025-01-01", end_date="2025-12-31")
    df_5_min = AEM.data.load_data_csv(Path("Tests/Data/Arlington_Daily_Data_2.csv"))

    # Save Folder
    folder = Path("/home/scarlett/Documents/Entomology/AgroEcoMetrics/Tests/Images/")

    # Air Temperature Tests
    print("Air_Temp_Tests Passed:\t\t\t" + str(air_temp_tests(df2024, folder)))

    # EvapoTranspiration Tests
    print("Evapo_Transpiration_Tests Passed:\t" + str(evapo_transpiation_tests(df2024, folder)))

    # Rainfall & Runoff Tests
    print("Rainfall_&_Runoff_Tests Passed:\t\t" + str(rainfall_runoff_tests(df2024, folder)))

    # Soil Temperature Tests
    print("Soil_Temp_Tests Passed:\t\t\t" + str(soil_temp_tests(folder)))

    # Growing Degree Day Tests
    print("Growing_Degree_Day_Tests Passed:\t" + str(gdd_tests(df2024, folder)))
    
    # Photoperiod Tests
    print("PhotoPeriod_Tests Passed:\t\t" + str(photoperiod_test(folder)))

    # Air to Soil Temperature Tests
    print("Air_to_Soil_Temp_Tests Passed:\t\t" + str(air_to_soil_temp_test(df_5_min, folder)))

    # Hydraulic Conductivity Tests
    print("Hydraulic_Conductivity_Tests Passed:\t" + str(hydraulic_conductivity_test()))

    print("Data Tests Passed:\t\t\t" + str(data_tests(df, folder)))
    # Attempt to save the updated DataFrame
    #AEM.data.save_data(df2024, Path("/home/scarlett/Documents/Entomology/AgroEcoMetrics/Tests/Data/Saved_Test_Data.cSv"))
    

