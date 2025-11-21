from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import agroecometrics as AEM


def load_np_files(location, files):
  return_tuple = tuple()
  for file in files:
    return_tuple += (np.load(location.with_name(file)),)
  return return_tuple


def data_test() :
  try:
    # Attempt to load the dataset
    file = Path("Daily_Data.csv")
    data = AEM.data.load_data_csv(file, "Date & Time Collected")

    # Check the dataset size
    if(data.shape[0] != 30):
      print("Dataset has an unexpected number of rows")
      return False
    if(data.shape[1] != 16):
      print("Dataset has an unexpected number of columns")
      return False
    
    # Check that the extra columns where added
    if("DOY" not in data.columns):
      print("Expected Column, DOY, not found in loaded dataset")
      return False
    if("YEAR" not in data.columns):
      print("Expected Column, YEAR, not found in loaded dataset")
      return False
    if("NORMALIZED_DATE" not in data.columns):
      print("Expected Column, NORMALIZED_DATE, not found in loaded dataset")
      return False
  

  except Exception:
    print("Exception occured while loading data")
    traceback.print_exc()
    return False
  
  return True # Default return if no tests caused failure


def model_air_temp_test():
  try:
    # Load the Dataset and extract relavent columns
    file = Path("5_Min_Data.csv")
    data = AEM.data.load_data_csv(file, "Date & Time Collected")
    temps = data["5 Minute Average Air Temperature (c)"]
    doy = data["DOY"]
    pred_temp = AEM.equations.model_air_temp(temps, doy)

    # Check results
    expected_results = np.load("expected/air_temp/air_temp_model.npy")
    if(pred_temp.shape[0] != len(expected_results)):
      print("Incorrect length for Model Air Temperature results")
      return False
    if(not np.allclose(pred_temp, expected_results)):
      print("Model air temperature results not as expected")
      return False
    
  except Exception:
    print("Exception occured while running air_temp_tests")
    traceback.print_exc()
    return False
  
  return True # Default return if no tests caused failure


def yearly_soil_temp_test():
  try:
    # Run Model editing all of the different settings
    pred_temp_1 = AEM.equations.yearly_soil_temp(1)
    pred_temp_2 = AEM.equations.yearly_soil_temp(5)
    pred_temp_3 = AEM.equations.yearly_soil_temp(1, 10)
    pred_temp_4 = AEM.equations.yearly_soil_temp(1, thermal_amp=15)
    pred_temp_5 = AEM.equations.yearly_soil_temp(1, thermal_dif=0.5)
    pred_temp_6 = AEM.equations.yearly_soil_temp(1, time_lag=195)

    # Load expected results
    location = Path("expected/yearly_soil_temp/.pny")
    files = [
      "yearly_soil_temp_1.npy",
      "yearly_soil_temp_2.npy",
      "yearly_soil_temp_3.npy",
      "yearly_soil_temp_4.npy",
      "yearly_soil_temp_5.npy",
      "yearly_soil_temp_6.npy"
    ]
    exp_temp_1, exp_temp_2, exp_temp_3, exp_temp_4, exp_temp_5, exp_temp_6 = load_np_files(location, files)

    # Check the output size of the models
    if(pred_temp_1.shape[0] != len(exp_temp_1)):
      print("Size mismatch occured running yearly_soil_temp")
      return False
    if(pred_temp_2.shape[0] != len(exp_temp_2)):
      print("Size mismatch occured running yearly_soil_temp")
      return False
    if(pred_temp_3.shape[0] != len(exp_temp_3)):
      print("Size mismatch occured running yearly_soil_temp")
      return False
    if(pred_temp_4.shape[0] != len(exp_temp_4)):
      print("Size mismatch occured running yearly_soil_temp")
      return False
    if(pred_temp_5.shape[0] != len(exp_temp_5)):
      print("Size mismatch occured running yearly_soil_temp")
      return False
    if(pred_temp_6.shape[0] != len(exp_temp_6)):
      print("Size mismatch occured running yearly_soil_temp")
      return False

    # Check the output of the models
    if(not np.allclose(pred_temp_1, exp_temp_1)):
      print("Incorrect results from yearly_soil_temp")
      return False
    if(not np.allclose(pred_temp_2, exp_temp_2)):
      print("Incorrect results from yearly_soil_temp")
      return False
    if(not np.allclose(pred_temp_3, exp_temp_3)):
      print("Incorrect results from yearly_soil_temp")
      return False
    if(not np.allclose(pred_temp_4, exp_temp_4)):
      print("Incorrect results from yearly_soil_temp")
      return False
    if(not np.allclose(pred_temp_5, exp_temp_5)):
      print("Incorrect results from yearly_soil_temp")
      return False
    if(not np.allclose(pred_temp_6, exp_temp_6)):
      print("Incorrect results from yearly_soil_temp")
      return False


  except Exception:
    print("Exception occured while running yearly_soil_temp_tests")
    traceback.print_exc()
    return False

  return True # Default return if no tests caused failure


def daily_soil_temp_test():
  try:
    # Run Model with multiple settings
    pred_temps_1, pred_depths_1 = AEM.equations.daily_soil_temp(0, 5)
    pred_temps_2, pred_depths_2 = AEM.equations.daily_soil_temp(100, 5)
    pred_temps_3, pred_depths_3 = AEM.equations.daily_soil_temp(365, 5)
    pred_temps_4, pred_depths_4 = AEM.equations.daily_soil_temp(50, 100)
    pred_temps_5, pred_depths_5 = AEM.equations.daily_soil_temp(50, 10, num_depths=10)
    pred_temps_6, pred_depths_6 = AEM.equations.daily_soil_temp(50, 10, surface_temp=10)
    pred_temps_7, pred_depths_7 = AEM.equations.daily_soil_temp(50, 10, thermal_amp=15)
    pred_temps_8, pred_depths_8 = AEM.equations.daily_soil_temp(50, 10, thermal_dif=0.5)
    pred_temps_9, pred_depths_9 = AEM.equations.daily_soil_temp(50, 10, timelag=195)

    # Load expected results
    location = Path("expected/daily_soil_temp/.pny")
    files = [
      "daily_soil_temp_1.npy", "daily_soil_depths_1.npy",
      "daily_soil_temp_2.npy", "daily_soil_depths_2.npy",
      "daily_soil_temp_3.npy", "daily_soil_depths_3.npy",
      "daily_soil_temp_4.npy", "daily_soil_depths_4.npy",
      "daily_soil_temp_5.npy", "daily_soil_depths_5.npy",
      "daily_soil_temp_6.npy", "daily_soil_depths_6.npy",
      "daily_soil_temp_7.npy", "daily_soil_depths_7.npy",
      "daily_soil_temp_8.npy", "daily_soil_depths_8.npy",
      "daily_soil_temp_9.npy", "daily_soil_depths_9.npy"
    ]
    (exp_temps_1, exp_depths_1, exp_temps_2, exp_depths_2, exp_temps_3, exp_depths_3,
    exp_temps_4, exp_depths_4, exp_temps_5, exp_depths_5, exp_temps_6, exp_depths_6,
    exp_temps_7, exp_depths_7, exp_temps_8, exp_depths_8, exp_temps_9, exp_depths_9) = load_np_files(location, files)

    # Compare size of actual and expected results
    if(pred_temps_1.shape[0] != len(exp_temps_1)):
      print("Size mismatch occured running daily_soil_temp")
      return False
    if(pred_depths_1.shape[0] != len(exp_depths_1)):
      print("Size mismatch occured running daily_soil_temp")
      return False
    if(pred_temps_2.shape[0] != len(exp_temps_2)):
      print("Size mismatch occured running daily_soil_temp")
      return False
    if(pred_depths_2.shape[0] != len(exp_depths_2)):
      print("Size mismatch occured running daily_soil_temp")
      return False
    if(pred_temps_3.shape[0] != len(exp_temps_3)):
      print("Size mismatch occured running daily_soil_temp")
      return False
    if(pred_depths_3.shape[0] != len(exp_depths_3)):
      print("Size mismatch occured running daily_soil_temp")
      return False
    if(pred_temps_4.shape[0] != len(exp_temps_4)):
      print("Size mismatch occured running daily_soil_temp")
      return False
    if(pred_depths_4.shape[0] != len(exp_depths_4)):
      print("Size mismatch occured running daily_soil_temp")
      return False
    if(pred_temps_5.shape[0] != len(exp_temps_5)):
      print("Size mismatch occured running daily_soil_temp")
      return False
    if(pred_depths_5.shape[0] != len(exp_depths_5)):
      print("Size mismatch occured running daily_soil_temp")
      return False
    if(pred_temps_6.shape[0] != len(exp_temps_6)):
      print("Size mismatch occured running daily_soil_temp")
      return False
    if(pred_depths_6.shape[0] != len(exp_depths_6)):
      print("Size mismatch occured running daily_soil_temp")
      return False
    if(pred_temps_7.shape[0] != len(exp_temps_7)):
      print("Size mismatch occured running daily_soil_temp")
      return False
    if(pred_depths_7.shape[0] != len(exp_depths_7)):
      print("Size mismatch occured running daily_soil_temp")
      return False
    if(pred_temps_8.shape[0] != len(exp_temps_8)):
      print("Size mismatch occured running daily_soil_temp")
      return False
    if(pred_depths_8.shape[0] != len(exp_depths_8)):
      print("Size mismatch occured running daily_soil_temp")
      return False
    if(pred_temps_9.shape[0] != len(exp_temps_9)):
      print("Size mismatch occured running daily_soil_temp")
      return False
    if(pred_depths_9.shape[0] != len(exp_depths_9)):
      print("Size mismatch occured running daily_soil_temp")
      return False
  
    # Compare contents of the actual and expected results
    if(not np.allclose(pred_temps_1, exp_temps_1)):
      print("Incorrect results from daily_soil_temp")
      return False
    if(not np.allclose(pred_depths_1, exp_depths_1)):
      print("Incorrect results from daily_soil_temp")
      return False
    if(not np.allclose(pred_temps_2, exp_temps_2)):
      print("Incorrect results from daily_soil_temp")
      return False
    if(not np.allclose(pred_depths_2, exp_depths_2)):
      print("Incorrect results from daily_soil_temp")
      return False
    if(not np.allclose(pred_temps_3, exp_temps_3)):
      print("Incorrect results from daily_soil_temp")
      return False
    if(not np.allclose(pred_depths_3, exp_depths_3)):
      print("Incorrect results from daily_soil_temp")
      return False
    if(not np.allclose(pred_temps_4, exp_temps_4)):
      print("Incorrect results from daily_soil_temp")
      return False
    if(not np.allclose(pred_depths_4, exp_depths_4)):
      print("Incorrect results from daily_soil_temp")
      return False
    if(not np.allclose(pred_temps_5, exp_temps_5)):
      print("Incorrect results from daily_soil_temp")
      return False
    if(not np.allclose(pred_depths_5, exp_depths_5)):
      print("Incorrect results from daily_soil_temp")
      return False
    if(not np.allclose(pred_temps_6, exp_temps_6)):
      print("Incorrect results from daily_soil_temp")
      return False
    if(not np.allclose(pred_depths_6, exp_depths_6)):
      print("Incorrect results from daily_soil_temp")
      return False
    if(not np.allclose(pred_temps_7, exp_temps_7)):
      print("Incorrect results from daily_soil_temp")
      return False
    if(not np.allclose(pred_depths_7, exp_depths_7)):
      print("Incorrect results from daily_soil_temp")
      return False
    if(not np.allclose(pred_temps_8, exp_temps_8)):
      print("Incorrect results from daily_soil_temp")
      return False
    if(not np.allclose(pred_depths_8, exp_depths_8)):
      print("Incorrect results from daily_soil_temp")
      return False
    if(not np.allclose(pred_temps_9, exp_temps_9)):
      print("Incorrect results from daily_soil_temp")
      return False
    if(not np.allclose(pred_depths_9, exp_depths_9)):
      print("Incorrect results from daily_soil_temp")
      return False


  except Exception:
    print("Exception occured while running daily_soil_temp_tests")
    traceback.print_exc()
    return False
  
  return True # Default return value if none of the tests failed


def  yearly_3d_soil_temp_test():
  try:
    doys_1, depths_1, temps_1 = AEM.equations.yearly_3d_soil_temp(5)
    doys_2, depths_2, temps_2 = AEM.equations.yearly_3d_soil_temp(10)
    doys_3, depths_3, temps_3 = AEM.equations.yearly_3d_soil_temp(5, 1)
    doys_4, depths_4, temps_4 = AEM.equations.yearly_3d_soil_temp(5, 100, 30, 15, .5, 195)

    # Load expected results
    location = Path("expected/yearly_3d_soil_temp/.npy")
    files = [
      "yearly_doys_1.npy", "yearly_depths_1.npy", "yearly_temps_1.npy",
      "yearly_doys_2.npy", "yearly_depths_2.npy", "yearly_temps_2.npy",
      "yearly_doys_3.npy", "yearly_depths_3.npy", "yearly_temps_3.npy",
      "yearly_doys_4.npy", "yearly_depths_4.npy", "yearly_temps_4.npy"
    ]
    (exp_doys_1, exp_depths_1, exp_temps_1, exp_doys_2, exp_depths_2, exp_temps_2,
     exp_doys_3, exp_depths_3, exp_temps_3, exp_doys_4, exp_depths_4, exp_temps_4) = load_np_files(location, files)

    # Compare size of actual and expected results
    if(doys_1.shape[0] != len(exp_doys_1)):
      print("Size mismatch occured running yearly_3d_soil_temp")
      return False
    if(depths_1.shape[0] != len(exp_depths_1)):
      print("Size mismatch occured running yearly_3d_soil_temp")
      return False
    if(temps_1.shape[0] != len(exp_temps_1)):
      print("Size mismatch occured running yearly_3d_soil_temp")
      return False
    if(doys_2.shape[0] != len(exp_doys_2)):
      print("Size mismatch occured running yearly_3d_soil_temp")
      return False
    if(depths_2.shape[0] != len(exp_depths_2)):
      print("Size mismatch occured running yearly_3d_soil_temp")
      return False
    if(temps_2.shape[0] != len(exp_temps_2)):
      print("Size mismatch occured running yearly_3d_soil_temp")
      return False
    if(doys_3.shape[0] != len(exp_doys_3)):
      print("Size mismatch occured running yearly_3d_soil_temp")
      return False
    if(depths_3.shape[0] != len(exp_depths_3)):
      print("Size mismatch occured running yearly_3d_soil_temp")
      return False
    if(temps_3.shape[0] != len(exp_temps_3)):
      print("Size mismatch occured running yearly_3d_soil_temp")
      return False
    if(doys_4.shape[0] != len(exp_doys_4)):
      print("Size mismatch occured running yearly_3d_soil_temp")
      return False
    if(depths_4.shape[0] != len(exp_depths_4)):
      print("Size mismatch occured running yearly_3d_soil_temp")
      return False
    if(temps_4.shape[0] != len(exp_temps_4)):
      print("Size mismatch occured running yearly_3d_soil_temp")
      return False

    # Compare contents of the actual and expected results
    if(not np.allclose(doys_1, exp_doys_1)):
      print("Incorrect results from yearly_3d_soil_temp")
      return False
    if(not np.allclose(depths_1, exp_depths_1)):
      print("Incorrect results from yearly_3d_soil_temp")
      return False
    if(not np.allclose(temps_1, exp_temps_1)):
      print("Incorrect results from yearly_3d_soil_temp")
      return False
    if(not np.allclose(doys_2, exp_doys_2)):
      print("Incorrect results from yearly_3d_soil_temp")
      return False
    if(not np.allclose(depths_2, exp_depths_2)):
      print("Incorrect results from yearly_3d_soil_temp")
      return False
    if(not np.allclose(temps_2, exp_temps_2)):
      print("Incorrect results from yearly_3d_soil_temp")
      return False
    if(not np.allclose(doys_3, exp_doys_3)):
      print("Incorrect results from yearly_3d_soil_temp")
      return False
    if(not np.allclose(depths_3, exp_depths_3)):
      print("Incorrect results from yearly_3d_soil_temp")
      return False
    if(not np.allclose(temps_3, exp_temps_3)):
      print("Incorrect results from yearly_3d_soil_temp")
      return False
    if(not np.allclose(doys_4, exp_doys_4)):
      print("Incorrect results from yearly_3d_soil_temp")
      return False
    if(not np.allclose(depths_4, exp_depths_4)):
      print("Incorrect results from yearly_3d_soil_temp")
      return False
    if(not np.allclose(temps_4, exp_temps_4)):
      print("Incorrect results from yearly_3d_soil_temp")
      return False


  except Exception:
    print("Exception occured while running yearly_3d_soil_temp_tests")
    traceback.print_exc()
    return False
  
  return True # Default return value if none of the tests failed
    

def model_soil_temp_test():
  try:
    # Load the Dataset and extract relavent columns
    file = Path("5_Min_Data.csv")
    data = AEM.data.load_data_csv(file, "Date & Time Collected")
    temps = data["5 Minute Average Air Temperature (c)"]
    pred_temps_1 = AEM.equations.model_soil_temp(temps, 1)
    pred_temps_2 = AEM.equations.model_soil_temp(temps, 2)
    pred_temps_3 = AEM.equations.model_soil_temp(temps, 5)
    pred_temps_4 = AEM.equations.model_soil_temp(temps, 10)

    # Load expected results
    location = Path("expected/model_soil_temp/.npy")
    files = ["temps_1.npy", "temps_2.npy", "temps_3.npy", "temps_4.npy"]
    exp_temps_1, exp_temps_2, exp_temps_3, exp_temps_4 = load_np_files(location, files)

    # Compare size of actual and expected results
    if(pred_temps_1.shape[0] != len(exp_temps_1)):
      print("Size mismatch occured running model_soil_temp")
      return False
    if(pred_temps_2.shape[0] != len(exp_temps_2)):
      print("Size mismatch occured running model_soil_temp")
      return False
    if(pred_temps_3.shape[0] != len(exp_temps_3)):
      print("Size mismatch occured running model_soil_temp")
      return False
    if(pred_temps_4.shape[0] != len(exp_temps_4)):
      print("Size mismatch occured running model_soil_temp")
      return False

    # Compare contents of the actual and expected results
    if(not np.allclose(pred_temps_1, exp_temps_1)):
      print("Incorrect results from model_soil_temp")
      return False
    if(not np.allclose(pred_temps_2, exp_temps_2)):
      print("Incorrect results from model_soil_temp")
      return False
    if(not np.allclose(pred_temps_3, exp_temps_3)):
      print("Incorrect results from model_soil_temp")
      return False
    if(not np.allclose(pred_temps_4, exp_temps_4)):
      print("Incorrect results from model_soil_temp")
      return False


  except Exception:
    print("Exception occured while running model_soil_temp_tests")
    traceback.print_exc()
    return False
  
  return True # Default return value if none of the tests failed


def model_3d_soil_temp_test():
  try:
    # Load the Dataset and extract relavent columns
    file = Path("5_Min_Data.csv")
    data = AEM.data.load_data_csv(file, "Date & Time Collected")
    temps = data["5 Minute Average Air Temperature (c)"]
    pred_time_1, pred_depth_1, pred_temp_1 = AEM.equations.model_3d_soil_temp(temps, 1)
    pred_time_2, pred_depth_2, pred_temp_2 = AEM.equations.model_3d_soil_temp(temps, 5)
    pred_time_3, pred_depth_3, pred_temp_3 = AEM.equations.model_3d_soil_temp(temps, 5, 100)
    pred_time_4, pred_depth_4, pred_temp_4 = AEM.equations.model_3d_soil_temp(temps, 10, 2000)

    location = Path("expected/model_3d_soil_temp/.npy")
    files = [
      "time_1.npy", "depth_1.npy", "temp_1.npy", "time_2.npy", "depth_2.npy", "temp_2.npy",
      "time_3.npy", "depth_3.npy", "temp_3.npy", "time_4.npy", "depth_4.npy", "temp_4.npy"
    ]
    (exp_time_1, exp_depth_1, exp_temp_1, exp_time_2, exp_depth_2, exp_temp_2,
     exp_time_3, exp_depth_3, exp_temp_3, exp_time_4, exp_depth_4, exp_temp_4) = load_np_files(location, files)

    # Compare size of actual and expected results
    if(pred_time_1.shape[0] != len(exp_time_1)):
      print("Size mismatch occured running model_3d_soil_temp")
      return False
    if(pred_depth_1.shape[0] != len(exp_depth_1)):
      print("Size mismatch occured running model_3d_soil_temp")
      return False
    if(pred_temp_1.shape[0] != len(exp_temp_1)):
      print("Size mismatch occured running model_3d_soil_temp")
      return False
    if(pred_time_2.shape[0] != len(exp_time_2)):
      print("Size mismatch occured running model_3d_soil_temp")
      return False
    if(pred_depth_2.shape[0] != len(exp_depth_2)):
      print("Size mismatch occured running model_3d_soil_temp")
      return False
    if(pred_temp_2.shape[0] != len(exp_temp_2)):
      print("Size mismatch occured running model_3d_soil_temp")
      return False
    if(pred_time_3.shape[0] != len(exp_time_3)):
      print("Size mismatch occured running model_3d_soil_temp")
      return False
    if(pred_depth_3.shape[0] != len(exp_depth_3)):
      print("Size mismatch occured running model_3d_soil_temp")
      return False
    if(pred_temp_3.shape[0] != len(exp_temp_3)):
      print("Size mismatch occured running model_3d_soil_temp")
      return False
    if(pred_time_4.shape[0] != len(exp_time_4)):
      print("Size mismatch occured running model_3d_soil_temp")
      return False
    if(pred_depth_4.shape[0] != len(exp_depth_4)):
      print("Size mismatch occured running model_3d_soil_temp")
      return False
    if(pred_temp_4.shape[0] != len(exp_temp_4)):
      print("Size mismatch occured running model_3d_soil_temp")
      return False

    # Compare contents of the actual and expected results
    if(not np.allclose(pred_time_1, exp_time_1)):
      print("Incorrect results from model_3d_soil_temp")
      return False
    if(not np.allclose(pred_depth_1, exp_depth_1)):
      print("Incorrect results from model_3d_soil_temp")
      return False
    if(not np.allclose(pred_temp_1, exp_temp_1)):
      print("Incorrect results from model_3d_soil_temp")
      return False
    if(not np.allclose(pred_time_2, exp_time_2)):
      print("Incorrect results from model_3d_soil_temp")
      return False
    if(not np.allclose(pred_depth_2, exp_depth_2)):
      print("Incorrect results from model_3d_soil_temp")
      return False
    if(not np.allclose(pred_temp_2, exp_temp_2)):
      print("Incorrect results from model_3d_soil_temp")
      return False
    if(not np.allclose(pred_time_3, exp_time_3)):
      print("Incorrect results from model_3d_soil_temp")
      return False
    if(not np.allclose(pred_depth_3, exp_depth_3)):
      print("Incorrect results from model_3d_soil_temp")
      return False
    if(not np.allclose(pred_temp_3, exp_temp_3)):
      print("Incorrect results from model_3d_soil_temp")
      return False
    if(not np.allclose(pred_time_4, exp_time_4)):
      print("Incorrect results from model_3d_soil_temp")
      return False
    if(not np.allclose(pred_depth_4, exp_depth_4)):
      print("Incorrect results from model_3d_soil_temp")
      return False
    if(not np.allclose(pred_temp_4, exp_temp_4)):
      print("Incorrect results from model_3d_soil_temp")
      return False


  except Exception:
    print("Exception occured while running model_3d_soil_temp_tests")
    traceback.print_exc()
    return False
  
  return True # Default return value if none of the tests failed


def evapotranspiration_test():
  try:
    # Load the Dataset and extract relavent columns
    file = Path("Daily_Data.csv")
    data = AEM.data.load_data_csv(file, "Date & Time Collected")
    min_temp = data["Daily Minimum Air Temperature (c)"]
    max_temp = data["Daily Maximum Air Temperature (c)"]
    min_rh = data["Daily Minimum Relative Humidity (pct)"]
    max_rh = data["Daily Maximum Relative Humidity (pct)"]
    min_p = data["Daily Minimum Pressure (mb)"] * 100 # Convert to Pascal
    max_p = data["Daily Maximum Pressure (mb)"] * 100 # Convert to Pascal
    avg_wind = data["Daily Average Wind Speed (m/s)"]
    doys = data["DOY"]
    solar_rad =  data["Daily Total Solar Radiation (mj)"]

    pred_dalton = AEM.equations.dalton(min_temp, max_temp, min_rh, max_rh, avg_wind)
    pred_penman = AEM.equations.penman(min_temp, max_temp, min_rh, max_rh, avg_wind)
    pred_romananko = AEM.equations.romanenko(min_temp, max_temp, min_rh, max_rh)
    pred_jensen_haise = AEM.equations.jensen_haise(min_temp, max_temp, doys, 43.0665139)
    pred_hargreaves = AEM.equations.hargreaves(min_temp, max_temp, doys, 43.0665139)
    pred_penman_monteith = AEM.equations.penman_monteith(min_temp, max_temp, min_rh, max_rh, min_p, max_p, avg_wind, doys, 43.0665139, 262, solar_rad)

    location = Path("expected/evapotranspiration/.npy")
    files = [
      "dalton.npy", "penman.npy", "romananko.npy",
      "jensen_haise.npy", "hargreaves.npy", "penman_monteith.npy"
    ]
    (exp_dalton, exp_penman, exp_romananko, exp_jensen_haise, 
     exp_hargreaves, exp_penman_monteith) = load_np_files(location, files)
    
    # Compare size of actual and expected results
    if(pred_dalton.shape[0] != len(exp_dalton)):
      print("Size mismatch occured running dalton evapotranspiration model")
      return False
    if(pred_penman.shape[0] != len(exp_penman)):
      print("Size mismatch occured running penman evapotranspiration model")
      return False
    if(pred_romananko.shape[0] != len(exp_romananko)):
      print("Size mismatch occured running romananko evapotranspiration model")
      return False
    if(pred_jensen_haise.shape[0] != len(exp_jensen_haise)):
      print("Size mismatch occured running jensen haise evapotranspiration model")
      return False
    if(pred_hargreaves.shape[0] != len(exp_hargreaves)):
      print("Size mismatch occured running hargreaves evapotranspiration model")
      return False
    if(pred_penman_monteith.shape[0] != len(exp_penman_monteith)):
      print("Size mismatch occured running penman monteith evapotranspiration model")
      return False
    
    # Compare contents of the actual and expected results
    if(not np.allclose(pred_dalton, exp_dalton)):
      print("Incorrect results from dalton evapotransipration model")
      return False
    if(not np.allclose(pred_penman, exp_penman)):
      print("Incorrect results from penman evapotransipration model")
      return False
    if(not np.allclose(pred_romananko, exp_romananko)):
      print("Incorrect results from romananko evapotransipration model")
      return False
    if(not np.allclose(pred_jensen_haise, exp_jensen_haise)):
      print("Incorrect results from jensen_haise evapotransipration model")
      return False
    if(not np.allclose(pred_hargreaves, exp_hargreaves)):
      print("Incorrect results from hargreaves evapotransipration model")
      return False
    if(not np.allclose(pred_penman_monteith, exp_penman_monteith)):
      print("Incorrect results from penman monteith evapotransipration model")
      return False
  
  except Exception:
    print("Exception occured while running evapotranspiration models")
    traceback.print_exc()
    return False
  
  return True # Default return value if none of the tests failed


def model_runoff_test():
  try:
    # Load the Dataset and extract relavent columns
    file = Path("Daily_Data.csv")
    data = AEM.data.load_data_csv(file, "Date & Time Collected")
    rainfall = data["Daily Total Rain (mm)"]

    pred_runoff_1 = AEM.equations.model_runoff(rainfall, 1)
    pred_runoff_2 = AEM.equations.model_runoff(rainfall, 25)
    pred_runoff_3 = AEM.equations.model_runoff(rainfall, 50)
    pred_runoff_4 = AEM.equations.model_runoff(rainfall)
    pred_runoff_5 = AEM.equations.model_runoff(rainfall, 99)

    location = Path("expected/model_runoff/.npy")
    files = ["runoff_1.npy", "runoff_2.npy", "runoff_3.npy", "runoff_4.npy", "runoff_5.npy"]
    (exp_runoff_1, exp_runoff_2, exp_runoff_3, exp_runoff_4, exp_runoff_5) = load_np_files(location, files)

    # Compare size of actual and expected results
    if(pred_runoff_1.shape[0] != len(exp_runoff_1)):
      print("Size mismatch occured running runoff model")
      return False
    if(pred_runoff_2.shape[0] != len(exp_runoff_2)):
      print("Size mismatch occured running runoff model")
      return False
    if(pred_runoff_3.shape[0] != len(exp_runoff_3)):
      print("Size mismatch occured running runoff model")
      return False
    if(pred_runoff_4.shape[0] != len(exp_runoff_4)):
      print("Size mismatch occured running runoff model")
      return False
    if(pred_runoff_5.shape[0] != len(exp_runoff_5)):
      print("Size mismatch occured running runoff model")
      return False
    
    # Compare contents of the actual and expected results
    if(not np.allclose(pred_runoff_1, exp_runoff_1)):
      print("Incorrect results from runoff model")
      return False
    if(not np.allclose(pred_runoff_2, exp_runoff_2)):
      print("Incorrect results from runoff model")
      return False
    if(not np.allclose(pred_runoff_3, exp_runoff_3)):
      print("Incorrect results from runoff model")
      return False
    if(not np.allclose(pred_runoff_4, exp_runoff_4)):
      print("Incorrect results from runoff model")
      return False
    if(not np.allclose(pred_runoff_5, exp_runoff_5)):
      print("Incorrect results from runoff model")
      return False
    

  except Exception:
    print("Exception occured while running runnoff tests")
    traceback.print_exc()
    return False
  
  return True # Default return value if none of the tests failed


def model_gdd_test():
  try:
    # Load the Datasets and extract relavent columns
    daily_file = Path("Daily_Data.csv")
    daily_data = AEM.data.load_data_csv(daily_file, "Date & Time Collected")
    daily_avg_temp = daily_data["Daily Average Air Temperature (c)"]
    minute_file = Path("5_Min_Data.csv")
    minute_data = AEM.data.load_data_csv(minute_file, "Date & Time Collected")
    minute_avg_temp = minute_data["5 Minute Average Air Temperature (c)"]


    pred_day_gdd_1, pred_day_gdd_sum_1 = AEM.equations.model_gdd(daily_avg_temp, 18)
    pred_day_gdd_2, pred_day_gdd_sum_2 = AEM.equations.model_gdd(daily_avg_temp, 20)
    pred_day_gdd_3, pred_day_gdd_sum_3 = AEM.equations.model_gdd(daily_avg_temp, 26)  # Shoud be 0
    pred_day_gdd_4, pred_day_gdd_sum_4 = AEM.equations.model_gdd(daily_avg_temp, 18, 22, 25)
    pred_day_gdd_5, pred_day_gdd_sum_5 = AEM.equations.model_gdd(daily_avg_temp, 20, 22, 25)
    pred_day_gdd_6, pred_day_gdd_sum_6 = AEM.equations.model_gdd(daily_avg_temp, 18, 20, 26)
    pred_day_gdd_7, pred_day_gdd_sum_7 = AEM.equations.model_gdd(daily_avg_temp, 10)

    pred_min_gdd_1, pred_min_gdd_sum_1 = AEM.equations.model_gdd(minute_avg_temp, 18, time_duration=1/(12*24))
    pred_min_gdd_2, pred_min_gdd_sum_2 = AEM.equations.model_gdd(minute_avg_temp, 20, time_duration=1/(12*24))
    pred_min_gdd_3, pred_min_gdd_sum_3 = AEM.equations.model_gdd(minute_avg_temp, 29, time_duration=1/(12*24))  # Shoud be 0
    pred_min_gdd_4, pred_min_gdd_sum_4 = AEM.equations.model_gdd(minute_avg_temp, 18, 22, 25, 1/(12*24))
    pred_min_gdd_5, pred_min_gdd_sum_5 = AEM.equations.model_gdd(minute_avg_temp, 20, 22, 25, 1/(12*24))
    pred_min_gdd_6, pred_min_gdd_sum_6 = AEM.equations.model_gdd(minute_avg_temp, 18, 20, 28, 1/(12*24))
    pred_min_gdd_7, pred_min_gdd_sum_7 = AEM.equations.model_gdd(minute_avg_temp, 10, time_duration=1/(12*24))


    # Load expected results
    location = Path("expected/model_gdd/.npy")
    files = [
      "daily_gdd_1.npy", "daily_gdd_2.npy", "daily_gdd_3.npy",
      "daily_gdd_4.npy", "daily_gdd_5.npy", "daily_gdd_6.npy", "daily_gdd_7.npy",
      "daily_gdd_sum_1.npy", "daily_gdd_sum_2.npy", "daily_gdd_sum_3.npy",
      "daily_gdd_sum_4.npy", "daily_gdd_sum_5.npy", "daily_gdd_sum_6.npy", "daily_gdd_sum_7.npy",
      "minute_gdd_1.npy", "minute_gdd_2.npy", "minute_gdd_3.npy",
      "minute_gdd_4.npy", "minute_gdd_5.npy", "minute_gdd_6.npy", "minute_gdd_7.npy",
      "minute_gdd_sum_1.npy", "minute_gdd_sum_2.npy", "minute_gdd_sum_3.npy",
      "minute_gdd_sum_4.npy", "minute_gdd_sum_5.npy", "minute_gdd_sum_6.npy", "minute_gdd_sum_7.npy",
    ]
    (exp_day_gdd_1, exp_day_gdd_2, exp_day_gdd_3, exp_day_gdd_4, exp_day_gdd_5,
     exp_day_gdd_6, exp_day_gdd_7, exp_day_gdd_sum_1, exp_day_gdd_sum_2, exp_day_gdd_sum_3, 
     exp_day_gdd_sum_4, exp_day_gdd_sum_5, exp_day_gdd_sum_6, exp_day_gdd_sum_7, exp_min_gdd_1,
     exp_min_gdd_2, exp_min_gdd_3, exp_min_gdd_4, exp_min_gdd_5, exp_min_gdd_6, exp_min_gdd_7, 
     exp_min_gdd_sum_1, exp_min_gdd_sum_2, exp_min_gdd_sum_3, exp_min_gdd_sum_4,
     exp_min_gdd_sum_5, exp_min_gdd_sum_6, exp_min_gdd_sum_7
    ) = load_np_files(location, files)

    # Compare size of actual and expected results
    if(pred_day_gdd_1.shape[0] != len(exp_day_gdd_1)):
      print("Size mismatch occured running growing degree day model on daily data")
      return False
    if(pred_day_gdd_2.shape[0] != len(exp_day_gdd_2)):
      print("Size mismatch occured running growing degree day model on daily data")
      return False
    if(pred_day_gdd_3.shape[0] != len(exp_day_gdd_3)):
      print("Size mismatch occured running growing degree day model on daily data")
      return False
    if(pred_day_gdd_4.shape[0] != len(exp_day_gdd_4)):
      print("Size mismatch occured running growing degree day model on daily data")
      return False
    if(pred_day_gdd_5.shape[0] != len(exp_day_gdd_5)):
      print("Size mismatch occured running growing degree day model on daily data")
      return False
    if(pred_day_gdd_6.shape[0] != len(exp_day_gdd_6)):
      print("Size mismatch occured running growing degree day model on daily data")
      return False
    if(pred_day_gdd_7.shape[0] != len(exp_day_gdd_7)):
      print("Size mismatch occured running growing degree day model on daily data")
      return False
    if(pred_day_gdd_sum_1.shape[0] != len(exp_day_gdd_sum_1)):
      print("Size mismatch occured running growing degree day model on daily data")
      return False
    if(pred_day_gdd_sum_2.shape[0] != len(exp_day_gdd_sum_2)):
      print("Size mismatch occured running growing degree day model on daily data")
      return False
    if(pred_day_gdd_sum_3.shape[0] != len(exp_day_gdd_sum_3)):
      print("Size mismatch occured running growing degree day model on daily data")
      return False
    if(pred_day_gdd_sum_4.shape[0] != len(exp_day_gdd_sum_4)):
      print("Size mismatch occured running growing degree day model on daily data")
      return False
    if(pred_day_gdd_sum_5.shape[0] != len(exp_day_gdd_sum_5)):
      print("Size mismatch occured running growing degree day model on daily data")
      return False
    if(pred_day_gdd_sum_6.shape[0] != len(exp_day_gdd_sum_6)):
      print("Size mismatch occured running growing degree day model on daily data")
      return False
    if(pred_day_gdd_sum_7.shape[0] != len(exp_day_gdd_sum_7)):
      print("Size mismatch occured running growing degree day model on daily data")
      return False
    if(pred_min_gdd_1.shape[0] != len(exp_min_gdd_1)):
      print("Size mismatch occured running growing degree day model on 5 minute data")
      return False
    if(pred_min_gdd_2.shape[0] != len(exp_min_gdd_2)):
      print("Size mismatch occured running growing degree day model on 5 minute data")
      return False
    if(pred_min_gdd_3.shape[0] != len(exp_min_gdd_3)):
      print("Size mismatch occured running growing degree day model on 5 minute data")
      return False
    if(pred_min_gdd_4.shape[0] != len(exp_min_gdd_4)):
      print("Size mismatch occured running growing degree day model on 5 minute data")
      return False
    if(pred_min_gdd_5.shape[0] != len(exp_min_gdd_5)):
      print("Size mismatch occured running growing degree day model on 5 minute data")
      return False
    if(pred_min_gdd_6.shape[0] != len(exp_min_gdd_6)):
      print("Size mismatch occured running growing degree day model on 5 minute data")
      return False
    if(pred_min_gdd_7.shape[0] != len(exp_min_gdd_7)):
      print("Size mismatch occured running growing degree day model on 5 minute data")
      return False
    if(pred_min_gdd_sum_1.shape[0] != len(exp_min_gdd_sum_1)):
      print("Size mismatch occured running growing degree day model on 5 minute data")
      return False
    if(pred_min_gdd_sum_2.shape[0] != len(exp_min_gdd_sum_2)):
      print("Size mismatch occured running growing degree day model on 5 minute data")
      return False
    if(pred_min_gdd_sum_3.shape[0] != len(exp_min_gdd_sum_3)):
      print("Size mismatch occured running growing degree day model on 5 minute data")
      return False
    if(pred_min_gdd_sum_4.shape[0] != len(exp_min_gdd_sum_4)):
      print("Size mismatch occured running growing degree day model on 5 minute data")
      return False
    if(pred_min_gdd_sum_5.shape[0] != len(exp_min_gdd_sum_5)):
      print("Size mismatch occured running growing degree day model on 5 minute data")
      return False
    if(pred_min_gdd_sum_6.shape[0] != len(exp_min_gdd_sum_6)):
      print("Size mismatch occured running growing degree day model on 5 minute data")
      return False
    if(pred_min_gdd_sum_7.shape[0] != len(exp_min_gdd_sum_7)):
      print("Size mismatch occured running growing degree day model on 5 minute data")
      return False

    # Compare contents of the actual and expected results
    if(not np.allclose(pred_day_gdd_1, exp_day_gdd_1)):
      print("Incorrect results from growing degree day model on daily data")
      return False
    if(not np.allclose(pred_day_gdd_2, exp_day_gdd_2)):
      print("Incorrect results from growing degree day model on daily data")
      return False
    if(not np.allclose(pred_day_gdd_3, exp_day_gdd_3)):
      print("Incorrect results from growing degree day model on daily data")
      return False
    if(not np.allclose(pred_day_gdd_4, exp_day_gdd_4)):
      print("Incorrect results from growing degree day model on daily data")
      return False
    if(not np.allclose(pred_day_gdd_5, exp_day_gdd_5)):
      print("Incorrect results from growing degree day model on daily data")
      return False
    if(not np.allclose(pred_day_gdd_6, exp_day_gdd_6)):
      print("Incorrect results from growing degree day model on daily data")
      return False
    if(not np.allclose(pred_day_gdd_7, exp_day_gdd_7)):
      print("Incorrect results from growing degree day model on daily data")
      return False
    if(not np.allclose(pred_day_gdd_sum_1, exp_day_gdd_sum_1)):
      print("Incorrect results from growing degree day model on daily data")
      return False
    if(not np.allclose(pred_day_gdd_sum_2, exp_day_gdd_sum_2)):
      print("Incorrect results from growing degree day model on daily data")
      return False
    if(not np.allclose(pred_day_gdd_sum_3, exp_day_gdd_sum_3)):
      print("Incorrect results from growing degree day model on daily data")
      return False
    if(not np.allclose(pred_day_gdd_sum_4, exp_day_gdd_sum_4)):
      print("Incorrect results from growing degree day model on daily data")
      return False
    if(not np.allclose(pred_day_gdd_sum_5, exp_day_gdd_sum_5)):
      print("Incorrect results from growing degree day model on daily data")
      return False
    if(not np.allclose(pred_day_gdd_sum_6, exp_day_gdd_sum_6)):
      print("Incorrect results from growing degree day model on daily data")
      return False
    if(not np.allclose(pred_day_gdd_sum_7, exp_day_gdd_sum_7)):
      print("Incorrect results from growing degree day model on daily data")
      return False
    if(not np.allclose(pred_min_gdd_1, exp_min_gdd_1)):
      print("Incorrect results from growing degree day model on 5 minute data")
      return False
    if(not np.allclose(pred_min_gdd_2, exp_min_gdd_2)):
      print("Incorrect results from growing degree day model on 5 minute data")
      return False
    if(not np.allclose(pred_min_gdd_3, exp_min_gdd_3)):
      print("Incorrect results from growing degree day model on 5 minute data")
      return False
    if(not np.allclose(pred_min_gdd_4, exp_min_gdd_4)):
      print("Incorrect results from growing degree day model on 5 minute data")
      return False
    if(not np.allclose(pred_min_gdd_5, exp_min_gdd_5)):
      print("Incorrect results from growing degree day model on 5 minute data")
      return False
    if(not np.allclose(pred_min_gdd_6, exp_min_gdd_6)):
      print("Incorrect results from growing degree day model on 5 minute data")
      return False
    if(not np.allclose(pred_min_gdd_7, exp_min_gdd_7)):
      print("Incorrect results from growing degree day model on 5 minute data")
      return False
    if(not np.allclose(pred_min_gdd_sum_1, exp_min_gdd_sum_1)):
      print("Incorrect results from growing degree day model on 5 minute data")
      return False
    if(not np.allclose(pred_min_gdd_sum_2, exp_min_gdd_sum_2)):
      print("Incorrect results from growing degree day model on 5 minute data")
      return False
    if(not np.allclose(pred_min_gdd_sum_3, exp_min_gdd_sum_3)):
      print("Incorrect results from growing degree day model on 5 minute data")
      return False
    if(not np.allclose(pred_min_gdd_sum_4, exp_min_gdd_sum_4)):
      print("Incorrect results from growing degree day model on 5 minute data")
      return False
    if(not np.allclose(pred_min_gdd_sum_5, exp_min_gdd_sum_5)):
      print("Incorrect results from growing degree day model on 5 minute data")
      return False
    if(not np.allclose(pred_min_gdd_sum_6, exp_min_gdd_sum_6)):
      print("Incorrect results from growing degree day model on 5 minute data")
      return False
    if(not np.allclose(pred_min_gdd_sum_7, exp_min_gdd_sum_7)):
      print("Incorrect results from growing degree day model on 5 minute data")
      return False

  except Exception:
    print("Exception occured while running growing degree day tests")
    traceback.print_exc()
    return False
  
  return True # Default return value if none of the tests failed


def photoperiod_test():
  try:
    doys = np.asarray([0, 45, 90, 135, 180, 225, 270, 315, 360])
    (pred_day_len_1, pred_angle_1, pred_z_dist_1, pred_anomaly_1,
     pred_declination_1, pred_delta_1) = AEM.equations.photoperiod_at_lat(doys, 0)
    (pred_day_len_2, pred_angle_2, pred_z_dist_2, pred_anomaly_2,
     pred_declination_2, pred_delta_2) = AEM.equations.photoperiod_at_lat(doys, 15)
    (pred_day_len_3, pred_angle_3, pred_z_dist_3, pred_anomaly_3,
     pred_declination_3, pred_delta_3) = AEM.equations.photoperiod_at_lat(doys, 33)
    doys = np.asarray([90, 90, 90, 90, 90, 90, 90, 90])
    lats = np.asarray([0, 5, 10, 15, 20, 25, 30, 40])
    (pred_day_len_4, pred_angle_4, pred_z_dist_4, pred_anomaly_4,
     pred_declination_4, pred_delta_4) = AEM.equations.photoperiod_on_day(doys, lats)
    (pred_day_len_5, pred_angle_5, pred_z_dist_5, pred_anomaly_5,
     pred_declination_5, pred_delta_5) = AEM.equations.photoperiod_on_day(doys, lats)
    (pred_day_len_6, pred_angle_6, pred_z_dist_6, pred_anomaly_6,
     pred_declination_6, pred_delta_6) = AEM.equations.photoperiod_on_day(doys, lats)
    pred_float_vals = np.asarray([
      pred_angle_1, pred_z_dist_1, pred_angle_2, pred_z_dist_2, pred_angle_3,
      pred_z_dist_3, pred_angle_4, pred_z_dist_4, pred_angle_5, pred_z_dist_5,
      pred_angle_6, pred_z_dist_6
    ])
    
    location = Path("expected/photoperiod/.npy")
    files = [
      "day_length_1.npy", "sun_mean_anomaly_1.npy", "sun_declination_1.npy", "delta_1.npy",
      "day_length_2.npy", "sun_mean_anomaly_2.npy", "sun_declination_2.npy", "delta_2.npy",
      "day_length_3.npy", "sun_mean_anomaly_3.npy", "sun_declination_3.npy", "delta_3.npy",
      "day_length_4.npy", "sun_mean_anomaly_4.npy", "sun_declination_4.npy", "delta_4.npy",
      "day_length_5.npy", "sun_mean_anomaly_5.npy", "sun_declination_5.npy", "delta_5.npy",
      "day_length_6.npy", "sun_mean_anomaly_6.npy", "sun_declination_6.npy", "delta_6.npy",
      "float_vals.npy" ]
    (exp_day_len_1, exp_anomaly_1, exp_declination_1, exp_delta_1, exp_day_len_2,
     exp_anomaly_2, exp_declination_2, exp_delta_2, exp_day_len_3, exp_anomaly_3,
     exp_declination_3, exp_delta_3, exp_day_len_4, exp_anomaly_4, exp_declination_4,
     exp_delta_4, exp_day_len_5, exp_anomaly_5, exp_declination_5, exp_delta_5,
     exp_day_len_6, exp_anomaly_6, exp_declination_6, exp_delta_6, exp_float_vals) = load_np_files(location, files)
    
    
    # Compare size of actual and expected results
    if(pred_day_len_1.shape[0] != len(exp_day_len_1)):
      print("Size mismatch occured running photoperiod_at_lat")
      return False
    if(pred_anomaly_1.shape[0] != len(exp_anomaly_1)):
      print("Size mismatch occured running photoperiod_at_lat")
      return False
    if(pred_declination_1.shape[0] != len(exp_declination_1)):
      print("Size mismatch occured running photoperiod_at_lat")
      return False
    if(pred_delta_1.shape[0] != len(exp_delta_1)):
      print("Size mismatch occured running photoperiod_at_lat")
      return False
    if(pred_day_len_2.shape[0] != len(exp_day_len_2)):
      print("Size mismatch occured running photoperiod_at_lat")
      return False
    if(pred_anomaly_2.shape[0] != len(exp_anomaly_2)):
      print("Size mismatch occured running photoperiod_at_lat")
      return False
    if(pred_declination_2.shape[0] != len(exp_declination_2)):
      print("Size mismatch occured running photoperiod_at_lat")
      return False
    if(pred_delta_2.shape[0] != len(exp_delta_2)):
      print("Size mismatch occured running photoperiod_at_lat")
      return False
    if(pred_day_len_3.shape[0] != len(exp_day_len_3)):
      print("Size mismatch occured running photoperiod_at_lat")
      return False
    if(pred_anomaly_3.shape[0] != len(exp_anomaly_3)):
      print("Size mismatch occured running photoperiod_at_lat")
      return False
    if(pred_declination_3.shape[0] != len(exp_declination_3)):
      print("Size mismatch occured running photoperiod_at_lat")
      return False
    if(pred_delta_3.shape[0] != len(exp_delta_3)):
      print("Size mismatch occured running photoperiod_on_day")
      return False
    if(pred_day_len_4.shape[0] != len(exp_day_len_4)):
      print("Size mismatch occured running photoperiod_on_day")
      return False
    if(pred_anomaly_4.shape[0] != len(exp_anomaly_4)):
      print("Size mismatch occured running photoperiod_on_day")
      return False
    if(pred_declination_4.shape[0] != len(exp_declination_4)):
      print("Size mismatch occured running photoperiod_on_day")
      return False
    if(pred_delta_4.shape[0] != len(exp_delta_4)):
      print("Size mismatch occured running photoperiod_on_day")
      return False
    if(pred_day_len_5.shape[0] != len(exp_day_len_5)):
      print("Size mismatch occured running photoperiod_on_day")
      return False
    if(pred_anomaly_5.shape[0] != len(exp_anomaly_5)):
      print("Size mismatch occured running photoperiod_on_day")
      return False
    if(pred_declination_5.shape[0] != len(exp_declination_5)):
      print("Size mismatch occured running photoperiod_on_day")
      return False
    if(pred_delta_5.shape[0] != len(exp_delta_5)):
      print("Size mismatch occured running photoperiod_on_day")
      return False
    if(pred_day_len_6.shape[0] != len(exp_day_len_6)):
      print("Size mismatch occured running photoperiod_on_day")
      return False
    if(pred_anomaly_6.shape[0] != len(exp_anomaly_6)):
      print("Size mismatch occured running photoperiod_on_day")
      return False
    if(pred_declination_6.shape[0] != len(exp_declination_6)):
      print("Size mismatch occured running photoperiod_on_day")
      return False
    if(pred_delta_6.shape[0] != len(exp_delta_6)):
      print("Size mismatch occured running photoperiod_on_day")
      return False
    
    # Compare contents of the actual and expected results
    if(not np.allclose(pred_day_len_1, exp_day_len_1)):
      print("Incorrect results from photoperiod_at_lat")
      return False
    if(not np.allclose(pred_anomaly_1, exp_anomaly_1)):
      print("Incorrect results from photoperiod_at_lat")
      return False
    if(not np.allclose(pred_declination_1, exp_declination_1)):
      print("Incorrect results from photoperiod_at_lat")
      return False
    if(not np.allclose(pred_delta_1, exp_delta_1)):
      print("Incorrect results from photoperiod_at_lat")
      return False
    if(not np.allclose(pred_day_len_2, exp_day_len_2)):
      print("Incorrect results from photoperiod_at_lat")
      return False
    if(not np.allclose(pred_anomaly_2, exp_anomaly_2)):
      print("Incorrect results from photoperiod_at_lat")
      return False
    if(not np.allclose(pred_declination_2, exp_declination_2)):
      print("Incorrect results from photoperiod_at_lat")
      return False
    if(not np.allclose(pred_delta_2, exp_delta_2)):
      print("Incorrect results from photoperiod_at_lat")
      return False
    if(not np.allclose(pred_day_len_3, exp_day_len_3)):
      print("Incorrect results from photoperiod_at_lat")
      return False
    if(not np.allclose(pred_anomaly_3, exp_anomaly_3)):
      print("Incorrect results from photoperiod_at_lat")
      return False
    if(not np.allclose(pred_declination_3, exp_declination_3)):
      print("Incorrect results from photoperiod_at_lat")
      return False
    if(not np.allclose(pred_delta_3, exp_delta_3)):
      print("Incorrect results from photoperiod_at_lat")
      return False
    if(not np.allclose(pred_day_len_4, exp_day_len_4)):
      print("Incorrect results from photoperiod_on_day")
      return False
    if(not np.allclose(pred_anomaly_4, exp_anomaly_4)):
      print("Incorrect results from photoperiod_on_day")
      return False
    if(not np.allclose(pred_declination_4, exp_declination_4)):
      print("Incorrect results from photoperiod_on_day")
      return False
    if(not np.allclose(pred_delta_4, exp_delta_4)):
      print("Incorrect results from photoperiod_on_day")
      return False
    if(not np.allclose(pred_day_len_5, exp_day_len_5)):
      print("Incorrect results from photoperiod_on_day")
      return False
    if(not np.allclose(pred_anomaly_5, exp_anomaly_5)):
      print("Incorrect results from photoperiod_on_day")
      return False
    if(not np.allclose(pred_declination_5, exp_declination_5)):
      print("Incorrect results from photoperiod_on_day")
      return False
    if(not np.allclose(pred_delta_5, exp_delta_5)):
      print("Incorrect results from photoperiod_on_day")
      return False
    if(not np.allclose(pred_day_len_6, exp_day_len_6)):
      print("Incorrect results from photoperiod_on_day")
      return False
    if(not np.allclose(pred_anomaly_6, exp_anomaly_6)):
      print("Incorrect results from photoperiod_on_day")
      return False
    if(not np.allclose(pred_declination_6, exp_declination_6)):
      print("Incorrect results from photoperiod_on_day")
      return False
    if(not np.allclose(pred_delta_6, exp_delta_6)):
      print("Incorrect results from photoperiod_on_day")
      return False
    if(not np.allclose(pred_float_vals, exp_float_vals)):
      print("Incorrect results from photoperiod_on_day")
      return False


  except Exception:
    print("Exception occured while running photoperiod tests")
    traceback.print_exc()
    return False
  
  return True # Default return value if none of the tests failed


def water_flow_test():
  try:
    pred_hydro_conduct_1 = AEM.equations.hydraulic_conductivity(0.5, -10, .1, 4.2, 3.2)
    pred_hydro_conduct_2 = AEM.equations.hydraulic_conductivity(0.5, -10, .1, 4.2, 4.8)
    pred_hydro_conduct_3 = AEM.equations.hydraulic_conductivity(0.5, -10, .1, 3.2, 3.2)
    pred_hydro_conduct_4 = AEM.equations.hydraulic_conductivity(0.5, -10, .18, 4.2, 4.8)
    pred_hydro_conduct_5 = AEM.equations.hydraulic_conductivity(0.5, -20, .1, 4.2, 3.2)
    pred_hydro_conduct_6 = AEM.equations.hydraulic_conductivity(0.5, -20, .18, 4.2, 4.8)
    pred_hydro_conduct_7 = AEM.equations.hydraulic_conductivity(0.01, -65, .18, 5.7, 4.4)
    pred_hydro_conduct_8 = AEM.equations.hydraulic_conductivity(0.01, -65, .18, 5.7, 5.6)
    pred_hydro_conduct_9 = AEM.equations.hydraulic_conductivity(0.01, -65, .18, 5.7, 4.4)
    pred_hydro_conduct_10 = AEM.equations.hydraulic_conductivity(0.01, -65, .28, 6.7, 5.6)
    pred_hydro_conduct_11 = AEM.equations.hydraulic_conductivity(0.01, -82, .28, 5.7, 4.4)
    pred_hydro_conduct_12 = AEM.equations.hydraulic_conductivity(0.01, -82, .18, 6.7, 5.6)
    pred_hydro_conduct_13 = AEM.equations.hydraulic_conductivity(0.005, -65, .28, 6.3, 7)
    pred_hydro_conduct_14 = AEM.equations.hydraulic_conductivity(0.005, -65, .37, 6.3, 7.2)
    pred_hydro_conduct_15 = AEM.equations.hydraulic_conductivity(0.005, -65, .28, 6.7, 7)
    pred_hydro_conduct_16 = AEM.equations.hydraulic_conductivity(0.005, -77, .37, 7.1, 7.2)
    pred_hydro_conduct_17 = AEM.equations.hydraulic_conductivity(0.005, -62, .28, 6.3, 7)
    pred_hydro_conduct_18 = AEM.equations.hydraulic_conductivity(0.005, -77, .37, 7.1, 7.2)
    pred_hydro_conduct_19 = AEM.equations.hydraulic_conductivity(0.00025, -148, .42, 9.2, 12.3)
    pred_hydro_conduct_20 = AEM.equations.hydraulic_conductivity(0.00025, -148, .42, 13.4, 13.6)
    pred_hydro_conduct_21 = AEM.equations.hydraulic_conductivity(0.00025, -148, .42, 9.2, 12.3)
    pred_hydro_conduct_22 = AEM.equations.hydraulic_conductivity(0.00025, -261, .57, 14.4, 13.6)
    pred_hydro_conduct_23 = AEM.equations.hydraulic_conductivity(0.00025, -148, .42, 12.2, 12.3)
    pred_hydro_conduct_24 = AEM.equations.hydraulic_conductivity(0.00025, -333, .57, 13.4, 13.6)
    pred_water_infiltration_1 = AEM.equations.cummulative_water_infiltration(0.1, 10**-4, -0.5, -25, 1800)
    pred_water_infiltration_2 = AEM.equations.cummulative_water_infiltration(0.1, 10**-4, -0.5, -25, 3600)
    pred_water_infiltration_3 = AEM.equations.cummulative_water_infiltration(0.1, 10**-4, -0.5, -25, 7200)
    pred_water_infiltration_4 = AEM.equations.cummulative_water_infiltration(0.3, 10**-4, -0.5, -60, 3600)
    pred_water_infiltration_5 = AEM.equations.cummulative_water_infiltration(0.3, 10**-4, -3.2, -60, 3600)
    pred_water_infiltration_6 = AEM.equations.cummulative_water_infiltration(0.3, 10**-4, -3.2, -60, 7200)
    pred_water_infiltration_7 = AEM.equations.cummulative_water_infiltration(0.1, 10**-6, -3.2, -100, 3600)
    pred_water_infiltration_8 = AEM.equations.cummulative_water_infiltration(0.1, 10**-6, -4.9, -100, 3600)
    pred_water_infiltration_9 = AEM.equations.cummulative_water_infiltration(0.2, 10**-6, -4.9, -300, 3600)
    pred_water_infiltration_10 = AEM.equations.cummulative_water_infiltration(0.25, 10**-6, -5.0, -300, 7200)
    pred_water_infiltration_11 = AEM.equations.cummulative_water_infiltration(0.3, 10**-6, -5.0, -350, 1800)
    pred_water_infiltration_12 = AEM.equations.cummulative_water_infiltration(0.3, 10**-6, -5.0, -350, 3600)
    pred_hydro_vals = np.asarray([
      pred_hydro_conduct_1, pred_hydro_conduct_2, pred_hydro_conduct_3, pred_hydro_conduct_4, pred_hydro_conduct_5,
      pred_hydro_conduct_6, pred_hydro_conduct_7, pred_hydro_conduct_8, pred_hydro_conduct_9, pred_hydro_conduct_10,
      pred_hydro_conduct_11, pred_hydro_conduct_12, pred_hydro_conduct_13, pred_hydro_conduct_14, pred_hydro_conduct_15,
      pred_hydro_conduct_16, pred_hydro_conduct_17, pred_hydro_conduct_18, pred_hydro_conduct_19, pred_hydro_conduct_20,
      pred_hydro_conduct_21, pred_hydro_conduct_22, pred_hydro_conduct_23, pred_hydro_conduct_24,
    ])

    location = Path("expected/water_flow/.npy")

    files = [
      "infiltration_1.npy", "infiltration_2.npy", "infiltration_3.npy", "infiltration_4.npy", 
      "infiltration_5.npy", "infiltration_6.npy", "infiltration_7.npy", "infiltration_8.npy", 
      "infiltration_9.npy", "infiltration_10.npy", "infiltration_11.npy", "infiltration_12.npy", 
      "hydraulic_conductivity.npy",
    ]
    (exp_water_infiltration_1, exp_water_infiltration_2, exp_water_infiltration_3, 
     exp_water_infiltration_4, exp_water_infiltration_5, exp_water_infiltration_6, 
     exp_water_infiltration_7, exp_water_infiltration_8, exp_water_infiltration_9, 
     exp_water_infiltration_10, exp_water_infiltration_11, exp_water_infiltration_12,
     exp_hydro_vals) = load_np_files(location, files)
    
     # Compare size of actual and expected results
    if(pred_water_infiltration_1.shape[0] != len(exp_water_infiltration_1)):
      print("Size mismatch occured running cummulative_water_infiltration")
      return False
    if(pred_water_infiltration_2.shape[0] != len(exp_water_infiltration_2)):
      print("Size mismatch occured running cummulative_water_infiltration")
      return False
    if(pred_water_infiltration_3.shape[0] != len(exp_water_infiltration_3)):
      print("Size mismatch occured running cummulative_water_infiltration")
      return False
    if(pred_water_infiltration_4.shape[0] != len(exp_water_infiltration_4)):
      print("Size mismatch occured running cummulative_water_infiltration")
      return False
    if(pred_water_infiltration_5.shape[0] != len(exp_water_infiltration_5)):
      print("Size mismatch occured running cummulative_water_infiltration")
      return False
    if(pred_water_infiltration_6.shape[0] != len(exp_water_infiltration_6)):
      print("Size mismatch occured running cummulative_water_infiltration")
      return False
    if(pred_water_infiltration_7.shape[0] != len(exp_water_infiltration_7)):
      print("Size mismatch occured running cummulative_water_infiltration")
      return False
    if(pred_water_infiltration_8.shape[0] != len(exp_water_infiltration_8)):
      print("Size mismatch occured running cummulative_water_infiltration")
      return False
    if(pred_water_infiltration_9.shape[0] != len(exp_water_infiltration_9)):
      print("Size mismatch occured running cummulative_water_infiltration")
      return False
    if(pred_water_infiltration_10.shape[0] != len(exp_water_infiltration_10)):
      print("Size mismatch occured running cummulative_water_infiltration")
      return False
    if(pred_water_infiltration_11.shape[0] != len(exp_water_infiltration_11)):
      print("Size mismatch occured running cummulative_water_infiltration")
      return False
    if(pred_water_infiltration_12.shape[0] != len(exp_water_infiltration_12)):
      print("Size mismatch occured running cummulative_water_infiltration")
      return False

    # Compare contents of the actual and expected results
    if(not np.allclose(pred_hydro_vals, exp_hydro_vals)):
      print("Incorrect results from hydraulic_conductivity")
      return False
    if(not np.allclose(pred_water_infiltration_1, exp_water_infiltration_1)):
      print("Incorrect results from cummulative_water_infiltration")
      return False
    if(not np.allclose(pred_water_infiltration_2, exp_water_infiltration_2)):
      print("Incorrect results from cummulative_water_infiltration")
      return False
    if(not np.allclose(pred_water_infiltration_3, exp_water_infiltration_3)):
      print("Incorrect results from cummulative_water_infiltration")
      return False
    if(not np.allclose(pred_water_infiltration_4, exp_water_infiltration_4)):
      print("Incorrect results from cummulative_water_infiltration")
      return False
    if(not np.allclose(pred_water_infiltration_5, exp_water_infiltration_5)):
      print("Incorrect results from cummulative_water_infiltration")
      return False
    if(not np.allclose(pred_water_infiltration_6, exp_water_infiltration_6)):
      print("Incorrect results from cummulative_water_infiltration")
      return False
    if(not np.allclose(pred_water_infiltration_7, exp_water_infiltration_7)):
      print("Incorrect results from cummulative_water_infiltration")
      return False
    if(not np.allclose(pred_water_infiltration_8, exp_water_infiltration_8)):
      print("Incorrect results from cummulative_water_infiltration")
      return False
    if(not np.allclose(pred_water_infiltration_9, exp_water_infiltration_9)):
      print("Incorrect results from cummulative_water_infiltration")
      return False
    if(not np.allclose(pred_water_infiltration_10, exp_water_infiltration_10)):
      print("Incorrect results from cummulative_water_infiltration")
      return False
    if(not np.allclose(pred_water_infiltration_11, exp_water_infiltration_11)):
      print("Incorrect results from cummulative_water_infiltration")
      return False
    if(not np.allclose(pred_water_infiltration_12, exp_water_infiltration_12)):
      print("Incorrect results from cummulative_water_infiltration")
      return False


  except Exception:
    print("Exception occured while running water flow tests")
    traceback.print_exc()
    return False
  
  return True # Default return value if none of the tests failed



if __name__ == "__main__":
  if(not data_test()):
     print("Loading Data Test: Failed\n")
  else:
    print("Loading Data Test: Passed\n")

  if(not model_air_temp_test()):
    print("Modeling Air Temperature Test: Failed\n")
  else:
    print("Modeling Air Temperature Test: Passed\n")
  
  if(not yearly_soil_temp_test()):
    print("Yearly Soil Temperature Test: Failed\n")
  else:
    print("Yearly Soil Temperature Test: Passed\n")

  if(not daily_soil_temp_test()):
    print("Daily Soil Temperature Test: Failed\n")
  else:
    print("Daily Soil Temperature Test: Passed\n")

  if(not yearly_3d_soil_temp_test()):
    print("Yearly 3D Soil Temperature Test: Failed\n")
  else:
    print("Yearly 3D Soil Temperature Test: Passed\n")

  if(not model_soil_temp_test()):
    print("Model Soil Temperature Test: Failed\n")
  else:
    print("Model Soil Temperature Test: Passed\n")

  if(not model_3d_soil_temp_test()):
    print("Model 3D Soil Temperature Test: Failed\n")
  else:
    print("Model 3D Soil Temperature Test: Passed\n")

  if(not evapotranspiration_test()):
    print("Evapotranspiration Test: Failed\n")
  else:
    print("Evapotranspiration Test: Passed\n")

  if(not model_runoff_test()):
    print("Model Runoff Test: Failed\n")
  else:
    print("Model Runoff Test: Passed\n")

  if(not model_gdd_test()):
    print("Model Growing Degree Days Test: Failed\n")
  else:
    print("Model Growing Degree Days Test: Passed\n")

  if(not photoperiod_test()):
    print("Photoperiod Test: Failed\n")
  else:
    print("Photoperiod Test: Passed\n")

  if(not water_flow_test()):
    print("Water Flow Test: Failed\n")
  else:
    print("Water Flow Test: Passed\n")
  



