from typing import Optional, Tuple, Union
from scipy.stats import circmean

from bokeh.plotting import figure, show, output_notebook, ColumnDataSource, save
from bokeh.layouts import row
from bokeh.models import HoverTool
from bokeh.io import export_svgs

import numpy as np
import pandas as pd

from agroecometrics import settings


# Gets the acutal labels of columns based on the user settings
labels = settings.get_labels()

# Defines types for type hints
series_type = Union[float, list[float], np.ndarray]






# Data File Functions
def load_data(
        file_path: str, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None, 
        date_format: str ='%m/%d/%Y %I:%M %p'
    ) -> pd.DataFrame:
    '''
    Loads data into a DataFrame from a given CSV file filtered by date

    Args:
        file_path: The path to your csv data file
        start_date: Optional. The date to start filtering from (Inclusive)
        end_date: Optional. The date to stop filtering on (Inclusive)
        date_format: The date_format to use on the file
    
    Returns:
        A DataFrame with the information from the csv file filtered by the specified dates
    '''
    # Check that the start date is not after the end date
    if start_date and end_date and pd.to_datetime(start_date) > pd.to_datetime(end_date):
        print("The start date must be on or before the end date")
        return
    
    # Read the data from the given csv file and make the formatting consistent
    df = None
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip().str.replace("'", "")
        df[labels['date']] = pd.to_datetime(df[labels['date']], format=date_format)
    except Exception as e:
        print("Exception occured trying to read file:\t" + file_path)
        print(e.with_traceback)
        return

    # Filter data using the start and end dates
    if start_date:
        df = df[df[labels['date']] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df[labels['date']] <= pd.to_datetime(end_date)]

    # Adds a Year and DOY column to the df
    df['DOY'] = df[labels['date']].dt.dayofyear
    df['YEAR'] = df[labels['date']].dt.year

    return df.reset_index(drop=True)


# Helper Functions
def compute_esat(temp: series_type) -> np.ndarray:
    """
    Function that computes saturation vapor pressure based Tetens formula
    
    Args:
        temp: Temperature in Celsius to calculate the saturation vapor pressure on
    
    Returns:
        Computed saturation vapor pressure in kPa
    """
    e_sat = 0.6108 * np.exp(17.27 * temp/(temp+237.3)) 
    return e_sat

def compute_Ra(doy: series_type, latitude: float) -> np.ndarray:
    """
    Function that computes extra-terrestrial solar radiation
        based on the FAO Penman-Monteith method
    
    Args:
        doy: Day of year (0-365) where January 1st is 0 and 365
        latitude: Latitude of the location in degrees

    Returns:
        Extra-terrestrial solar radiation (Ra) in MJ/m²/day
    """
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy/365) # Inverse relative distance Earth-Sun
    phi = np.pi / 180 * latitude # Latitude in radians
    d = 0.409 * np.sin((2 * np.pi * doy/365) - 1.39) # Solar delcination
    omega = np.arccos(-np.tan(phi) * np.tan(d)) # Sunset hour angle
    Gsc = 0.0820 # Solar constant
    Ra = 24 * 60 / np.pi * Gsc * dr
    Ra = Ra * (omega * np.sin(phi) * np.sin(d) + np.cos(phi) * np.cos(d) * np.sin(omega))
    return Ra


# Air Temperature model
def model_air_temp(df: pd.DataFrame) -> np.ndarray:
    '''
    Uses the air temperature data to find parameter estimates for the air temperature model
        and creates the temperature predictions with the model

    Args:
        df: DataFrame containing temperature data
    
    Returns: 
        A numpy array of predicted daily temperatures
    '''
    global labels

    # Calculate mean temperature and temperature amplitude
    T_avg = df[labels['temp']].mean()
    T_min, T_max = df.groupby(by='DOY')[labels['temp']].mean().quantile([0.05, 0.95])
    A = (T_max - T_min) / 2

    # Estimate the day of year with minimum temperature using circular mean
    idx_min = df.groupby(by='YEAR')[labels['temp']].idxmin()
    doy_T_min = np.round(df.loc[idx_min, 'DOY'].apply(circmean).mean())

    # Generate daily temperature predictions using the model
    T_pred = T_avg + A * np.cos(2 * np.pi * ((df['DOY'] - doy_T_min) / 365) + np.pi)

    return np.array(T_pred)


# Soil Temperature models
def model_soil_temp_at_depth(
        depth: series_type,
        avg_temp: int=25,
        thermal_amp: int = 10,
        thermal_dif: float = 0.203,
        time_lag: int = 15
    ) -> np.ndarray:
    """
    Models soil temperature over one year at the given depth

    Args:
        depth: The depth in meters to model the soil temperature
        avg_temp: The annual average surface temperature in Celsius
        thermal_amp: The annual thermal amplitude of the soil surface in Celsius
        thermal_dif: The Thermal diffusivity obtained from KD2 Pro instrument [mm^2/s]
        time_lag: The time lag in days (0-365) where January 1st is 0 and 365

    Return:
        Predicted soil temperature at the given depth for each day of the year
    """
    # Set Constants
    OMEGA = 2*np.pi/365

    thermal_dif = thermal_dif / 100 * 86400 # convert to cm^2/day
    phase_const = np.pi/2 + OMEGA*time_lag # Phase constant
    damp_depth = (2*thermal_dif/OMEGA)**(1/2) # Damping depth 

    doy = np.arange(1,366)
    T_soil = avg_temp + thermal_amp * np.exp(-depth / damp_depth) * \
        np.sin(OMEGA * doy - depth / damp_depth - phase_const)
    
    return  T_soil

def model_day_soil_temp(
        doy: int,
        max_depth: int,
        Nz: int = 100,
        avg_temp: int = 25,
        thermal_amp: int = 10,
        thermal_dif: int = 0.203,
        timelag: int = 15
    ) -> pd.DataFrame:
    """
    Models soil temperature on a particular day of the year
    
    Args:
        doy: Day of year (0-365) where January 1st is 0 and 365
        max_depth: The maximum depth in centimeters to model the soil temperature
        Nz: The number of interpolated depths to caluculate soil temperature at
        avg_temp: The annual average surface temperature in Celcius
        thermal_amp: The annual thermal amplitude of the soil surface in Celsius
        thermal_dif: The Thermal diffusivity obtained from KD2 Pro instrument [mm^2/s]
        time_lag: The time lag in days (0-365) where January 1st is 0 and 365

    Returns:
        A DataFrame with a depth and temp column
    """
    # Set Constants
    OMEGA = 2*np.pi/365

    thermal_dif = thermal_dif / 100 * 86400 # convert to cm^2/day
    phase_const = np.pi/2 + OMEGA*timelag # Phase constant
    damp_depth = (2*thermal_dif/OMEGA)**(1/2) # Damping depth 

    depths = np.linspace(0, max_depth, Nz) # Interpolation depths
    
    T_soil = avg_temp + thermal_amp * np.exp(-depths / damp_depth) * \
        np.sin(OMEGA * doy - depths / damp_depth - phase_const)
    
    df = pd.DataFrame({
        "depth": depths,
        "temp": T_soil
    })

    return df

def model_soil_temp_3d(
        max_depth: int,
        Nz: int = 1000,
        avg_temp: int = 25,
        thermal_amp: int = 10,
        thermal_dif: int = 0.203,
        timelag: int = 15
    ) -> pd.DataFrame:
    """
    Models soil temperature over a full year (0–365) and across depth.
    
    Args:
        max_depth: The maximum depth in centimeters to model the soil temperature
        Nz: The number of interpolated depths to caluculate soil temperature at
        avg_temp: The annual average surface temperature in Celcius
        thermal_amp: The annual thermal amplitude of the soil surface in Celsius
        thermal_dif: The Thermal diffusivity obtained from KD2 Pro instrument [mm^2/s]
        time_lag: The time lag in days (0-365) where January 1st is 0 and 365
    
    Returns:
        A DataFrame with columns: 'day', 'depth', 'temp'.
    """
    # Set Constants
    OMEGA = 2*np.pi/365

    thermal_dif = thermal_dif / 100 * 86400 # convert to cm^2/day
    phase_const = np.pi/2 + OMEGA*timelag # Phase constant
    damp_depth = (2*thermal_dif/OMEGA)**(1/2) # Damping depth 

    doy = np.arange(1,366)
    depths = np.linspace(0, max_depth, Nz) # Interpolation depths
    doy_grid,z_grid = np.meshgrid(doy,depths)
    
    t_grid = avg_temp + thermal_amp * np.exp(-z_grid/damp_depth) * np.sin(OMEGA*doy_grid - z_grid/damp_depth - phase_const)

    # Flatten grids and create DataFrame
    df = pd.DataFrame({
        "day": doy_grid.ravel(),
        "depth": z_grid.ravel(),
        "temp": t_grid.ravel()
    })

    return df


# EvapoTranspiration Models
def dalton(
        temp_min: series_type,
        temp_max: series_type,
        RH_min: series_type,
        RH_max: series_type,
        wind_speed: series_type
    ) -> np.ndarray:
    """
    Potential evaporation model proposed by Dalton in 1802
    
    Args:
        temp_min: The minimum daily temperature in Celsius
        temp_max: The maximum daily temperature in Celsius
        RH_min:   The minimum daily relative humidity (range: 0.0-1.0)
        RH_max:   The maximum daily relative humidity (range: 0.0-1.0)
        wind_speed: The average daily wind speed in meters per second

    Returns:
        The Dalton models predictions for evapotranspiration clipped to a minimum of zero
    """
    # Ensures parameter saftey
    if not isinstance(temp_min, float) and not \
        (len(temp_min) == len(temp_max) == len(RH_min) == len(RH_max) == len(wind_speed)):
        raise ValueError("All inputs must be the same length")
    
    # Model calulations
    e_sat_min = compute_esat(temp_min)
    e_sat_max = compute_esat(temp_max)
    e_sat = (e_sat_min + e_sat_max)/2
    e_atm = (e_sat_min*(RH_max/100) + e_sat_max*(RH_min/100))/ 2
    PET = (3.648 + 0.7223*wind_speed)*(e_sat - e_atm)

    # Ensures non-negative evapotranspiration
    PET = np.clip(PET, 0.0, None)

    return PET

def penman(
        temp_min: series_type,
        temp_max: series_type,
        RH_min: series_type,
        RH_max: series_type,
        wind_speed: series_type
    ) -> np.ndarray:
    """
    Potential evapotranspiration model proposed by Penman in 1948

    Args:
        temp_min: The minimum daily temperature in Celsius
        temp_max: The maximum daily temperature in Celsius
        RH_min:   The minimum daily relative humidity (range: 0.0-1.0)
        RH_max:   The maximum daily relative humidity (range: 0.0-1.0)
        wind_speed: The average daily wind speed in meters per second

    Returns:
        The Penman models predictions for evapotranspiration clipped to a minimum of zero
    """
    # Ensure parameter saftey
    if not isinstance(temp_min, float) and not \
        (len(temp_min) == len(temp_max) == len(RH_min) == len(RH_max) == len(wind_speed)):
        raise ValueError("All inputs must be the same length")
    
    # Model calulations
    e_sat_min = compute_esat(temp_min)
    e_sat_max = compute_esat(temp_max)
    e_sat = (e_sat_min + e_sat_max)/2
    e_atm = (e_sat_min*(RH_max/100) + e_sat_max*(RH_min/100))/ 2
    PET = (2.625 + 0.000479/wind_speed)*(e_sat - e_atm)

    # Ensures non-negative evapotranspiration
    PET = np.clip(PET, 0.0, None)

    return PET

def romanenko(
        temp_min: series_type,
        temp_max: series_type,
        RH_min: series_type,
        RH_max: series_type,
    ) -> np.ndarray:
    """
    Potential evaporation model proposed by Romanenko in 1961
    
    Args:
        temp_min: The minimum daily temperature in Celsius
        temp_max: The maximum daily temperature in Celsius
        RH_min:   The minimum daily relative humidity (range: 0.0-1.0)
        RH_max:   The maximum daily relative humidity (range: 0.0-1.0)

    Returns:
        The Romanenko models predictions for evapotranspiration clipped to a minimum of zero
    """
    # Ensure parameter saftey
    if not isinstance(temp_min, float) and not \
        (len(temp_min) == len(temp_max) == len(RH_min) == len(RH_max)):
        raise ValueError("All inputs must be the same length")

    # Model calulations
    temp_avg = (temp_min + temp_max)/2
    RH_avg = (RH_min + RH_max)/2
    PET = 0.00006*(25 + temp_avg)**2*(100 - RH_avg)

    # Ensures non-negative evapotranspiration
    PET = np.clip(PET, 0.0, None)

    return PET

def jensen_haise(
        temp_min: series_type,
        temp_max: series_type,
        doy: series_type,
        latitude: series_type
    ) -> np.ndarray:
    """
    Potential evapotranspiration model proposed by Jensen in 1963
    
    Args:
        temp_min: The minimum daily temperature in Celsius
        temp_max: The maximum daily temperature in Celsius
        doy: Day of year (0-365) where January 1st is 0 and 365
        latitude: Latitude of the location in degrees

    Returns:
        The Jensen-Haise models predictions for evapotranspiration clipped to a minimum of zero
    """
    # Ensure parameter saftey
    if not isinstance(temp_min, float) and not (len(temp_min) == len(temp_max)):
        raise ValueError("All inputs must be the same length")
    
    # Model Calculations
    Ra = compute_Ra(doy, latitude)
    T_avg = (temp_min + temp_max)/2
    PET = 0.0102 * (T_avg+3) * Ra

    # Ensures non-negative evapotranspiration
    PET = np.clip(PET, 0.0, None)

    return PET

def hargreaves(
        temp_min: series_type,
        temp_max: series_type,
        doy: series_type,
        latitude: series_type
    ) -> np.ndarray:
    """
    Potential evapotranspiration model proposed by Hargreaves in 1982
    
    Args:
        temp_min: The minimum daily temperature in Celsius
        temp_max: The maximum daily temperature in Celsius
        doy: Day of year (0-365) where January 1st is 0 and 365
        latitude: Latitude of the location in degrees
        
    Returns:
        The Hargreaves models predictions for evapotranspiration clipped to a minimum of zero
    """
    # Ensure parameter saftey
    if not isinstance(temp_min, float) and not (len(temp_min) == len(temp_max)):
        raise ValueError("All inputs must be the same length")
    
    # Model Calulations
    Ra = compute_Ra(doy, latitude)
    T_avg = (temp_min + temp_max)/2
    PET = 0.0023 * Ra * (T_avg + 17.8) * (temp_max - temp_min)**0.5

    # Ensures non-negative evapotranspiration
    PET = np.clip(PET, 0.0, None)

    return PET

def penman_monteith(
        temp_min: series_type,
        temp_max: series_type,
        RH_min: series_type,
        RH_max: series_type,
        solar_rad: series_type,
        wind_speed: series_type,
        doy: series_type,
        latitude: float,
        altitude: float,
        wind_height: int = 1.5,
    ) -> np.ndarray:
    """
    Potential evapotranspiration model proposed by Penman in 1948 
        and revised by Monteith in 1965
    
    Args:
        temp_min: The minimum daily temperature in Celsius
        temp_max: The maximum daily temperature in Celsius
        RH_min:   The minimum daily relative humidity (range: 0.0-1.0)
        RH_max:   The maximum daily relative humidity (range: 0.0-1.0)
        solar_rad: The extra-terrestrial solar radiation (Ra) in MJ/m²/day
        wind_speed: The average daily wind speed in meters per second
        doy:      Day of year (0-365) where January 1st is 0 and 365
        latitude: Latitude of the location in degrees
        altitude: Altitude of the location in meters
        wind_height: Height of measurment for wind speed
        
    Returns:
        The Hargreaves models predictions for evapotranspiration clipped to
            a minimum of zero
    """
    # Ensure parameter saftey
    if not isinstance(temp_min, float) and not \
        (len(temp_min) == len(temp_max) == len(RH_min) == len(RH_max) == len(wind_speed) == \
         len(solar_rad) == len(wind_speed) == len(doy) == len(latitude) == len(altitude)):
        raise ValueError("All inputs must be the same length")
    
    temp_avg = (temp_min + temp_max)/2
    atm_pressure = 101.3 * ((293 - 0.0065 * altitude) / 293)**5.26 # Can be also obtained from weather station
    Cp = 0.001013; # Approx. 0.001013 for average atmospheric conditions
    gamma = (Cp * atm_pressure) / (0.622 * 2.45) # Approx. 0.000665

    # Wind speed Adjustment
    wind_speed_2m = wind_speed * (4.87 / np.log((67.8 * wind_height) - 5.42))  # Eq. 47, FAO-56 wind height in [m]

    # Calculates air humidity and vapor pressure
    delta = 4098 * (0.6108 * np.exp(17.27 * temp_avg / (temp_avg  + 237.3)))
    delta = delta / (temp_avg  + 237.3)**2
    e_temp_max = 0.6108 * np.exp(17.27 * temp_max / (temp_max + 237.3)) # Eq. 11, //FAO-56
    e_temp_min = 0.6108 * np.exp(17.27 * temp_min / (temp_min + 237.3))
    e_saturation = (e_temp_max + e_temp_min) / 2
    e_actual = (e_temp_min * (RH_max / 100) + e_temp_max * (RH_min / 100)) / 2

    # Calculates solar radiation
    Ra = compute_Ra(doy, latitude)

    # Clear Sky Radiation: Rso (MJ/m2/day)
    Rso =  (0.75 + (2 * 10**-5) * altitude) * Ra  # Eq. 37, FAO-56

    # Rs/Rso = relative shortwave radiation (limited to <= 1.0)
    alpha = 0.23 # 0.23 for hypothetical grass reference crop
    Rns = (1 - alpha) * solar_rad # Eq. 38, FAO-56
    sigma  = 4.903 * 10**-9
    maxTempK = temp_max + 273.16
    minTempK = temp_min + 273.16
    # Eq. 39, FAO-56
    Rnl =  sigma * (maxTempK**4 + minTempK**4)
    Rnl = Rnl / 2 * (0.34 - 0.14 * np.sqrt(e_actual))
    Rnl = Rnl * (1.35 * (solar_rad / Rso) - 0.35) 
    Rn = Rns - Rnl # Eq. 40, FAO-56

    # Soil heat flux density
    soil_heat_flux = 0 # Eq. 42, FAO-56 G = 0 for daily time steps  [MJ/m2/day]

    # ETo calculation
    PET = 0.408 * delta * (solar_rad - soil_heat_flux) + gamma
    PET = PET * (900 / (temp_avg  + 273))  * wind_speed_2m * (e_saturation - e_actual)
    PET = PET / (delta + gamma * (1 + 0.34 * wind_speed_2m))


    # Ensures non-negative evapotranspiration
    PET = np.clip(PET, 0.0, None)

    return np.round(PET,2)


# Rain/Runoff Models
def model_runoff(precip: series_type, cn: int = 75) -> pd.DataFrame:
    '''
    Uses Curve Number to estimate runoff from rainfall

    Args:
        precip: The daily amount of precipitation in millimeters
        cn: The curve number to use in the calculation

    Returns:
        The estimated runoff
    '''
    # Convert precitation to inches
    precip_inch = precip / 25.4

    # Model Calulations
    runoff = np.zeros_like(precip_inch)
    S02 = 1000 / cn - 10
    S005 = 1.33 * S02**1.15
    Lambda = 0.05
    Ia = S005 * Lambda
    idx = precip_inch > Ia
    runoff[idx] = (precip_inch[idx] - Ia)**2 / (precip_inch[idx] - Ia + S005)

    return runoff * 25.4 # Convert runoff back to millimeters

def rainfall_runoff_to_df(df: pd.DataFrame, cn: int = 75):
    '''
    Computes cumulative rainfall, cummulative runoff, and daily runoff and adds
        the computed data to the given DataFrame

    Args:
        df: The DataFrame containing the rainfall data
        cn: The curve number to be used in the runoff calculation
    '''
    df['RAIN_SUM'] = df[labels['rain']].cumsum()
    df['RUNOFF'] = model_runoff(df[labels['rain']], cn=cn)
    df['RUNOFF_SUM'] = df['RUNOFF'].cumsum()



# Fletcher's Functions
def match_weather_datetime(
        weather_dt: np.ndarray,
        data_dt: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Matches each datetime in `data_datetime_col` to the closest datetime 
        in `weather_datetime_col`.

    Args:
        weather_dt: The np array of actual weather date_times
        data_dt:    The np array of date_times to find the closest match for
    
    Returns:
        A tuple of (original times, matched weather times, matched indices, time differences)
    """
    indices = np.searchsorted(weather_dt, data_dt, side="left")
    indices = np.minimum(indices, len(weather_dt) - 1)

    for i in range(len(indices)):
        if indices[i] > 0:
            before = weather_dt[indices[i] - 1]
            after = weather_dt[indices[i]]
            if abs(data_dt[i] - before) < abs(data_dt[i] - after):
                indices[i] -= 1

    matched_times = weather_dt[indices]
    diffs = np.abs(data_dt - matched_times)

    return data_dt, matched_times, indices, diffs
 
def get_weather_data_from_cols(
        weather_df: pd.DataFrame,
        weather_cols: list[str],
        indices: np.ndarray
    ) -> dict[str, list[float]]:
    """
    Extracts values from weather columns at given matched indices.

    Args:
        weather_df:   The DataFrame containing weather data
        weather_cols: The key names to be used in the dictionary
        indices:      The indices in the DataFrame to be used in the dictionary

    Returns:
        A dictionary of {column_name: values}.
    """
    return {
        col: [weather_df[col].iloc[i] for i in indices]
        for col in weather_cols
    }



