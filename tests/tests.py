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
      "pred_time_1.npy", "pred_depth_1.npy", "pred_temp_1.npy",
      "pred_time_2.npy", "pred_depth_2.npy", "pred_temp_2.npy",
      "pred_time_3.npy", "pred_depth_3.npy", "pred_temp_3.npy",
      "pred_time_4.npy", "pred_depth_4.npy", "pred_temp_4.npy"
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
    min_temp = data["Daily Maximum Air Temperature (c)"]
    max_temp = data["Daily Minimum Air Temperature (c)"]
    min_rh = data["Daily Maximum Relative Humidity (pct)"]
    max_rh = data["Daily Minimum Relative Humidity (pct)"]
    min_p = data["Daily Minimum Pressure (mb)"] * 100 # Convert to Pascal
    max_p = data["Daily Minimum Pressure (mb)"] * 100 # Convert to Pascal
    avg_wind = data["Daily Average Wind Speed (m/s)"]
    doys = data["DOYS"]
    solar_rad =  data["Daily Total Solar Radiation (mj)"]

    pred_dalton = AEM.equations.dalton(min_temp, max_temp, min_rh, max_rh, avg_wind)
    pred_penman = AEM.equations.penman(min_temp, max_temp, min_rh, max_rh, avg_wind)
    pred_romananko = AEM.equations.romanenko(min_temp, max_temp, min_rh, max_rh)
    pred_jensen_haise = AEM.equations.jensen_haise(min_temp, max_temp, doys, 43.0665139)
    pred_hargreaves = AEM.equations.hargreaves()
    pred_penman_monteith =  AEM.equations.penman_monteith()

  
  except Exception:
    print("Exception occured while running model_3d_soil_temp_tests")
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
  



