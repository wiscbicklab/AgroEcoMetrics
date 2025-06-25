from typing import Optional, Tuple
from scipy.stats import circmean

import numpy as np
import pandas as pd

from agroecometrics import settings
import agroecometrics as AEM


# Gets the acutal LABELS of columns based on the user settings
LABELS = settings.get_labels()
DAY_SECONDS = 24*60*60
YEAR_SECONDS = DAY_SECONDS*365

# Helper Functions
def compute_esat(temp: np.ndarray) -> np.ndarray:
    """
    Computes saturation vapor pressure based Tetens formula
    
    Args:
        temp: Temperature to calculate the saturation vapor pressure (°C)
    
    Returns:
        Computed saturation vapor pressure (kPa)
    """
    e_sat = 0.6108 * np.exp(17.27 * temp/(temp+237.3)) 
    return e_sat

def compute_Ra(doy: np.ndarray, latitude: float) -> np.ndarray:
    """
    Computes extra-terrestrial solar radiation using the FAO Penman-Monteith method
    
    Args:
        doy: Day of year (0-365) where January 1st is 0 and 365
        latitude: Latitude in decimal degress. Where the northern hemisphere is 
            positive and the southern hemisphere is negative

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
    Creates an Air temperature model and creates air temperature estimates using model

    Args:
        df: DataFrame containing temperature data
    
    Returns: 
        A numpy array of predicted daily temperatures over the period of the dataframe
    '''
    global LABELS

    # Check Parameters
    if LABELS['temp_avg'] not in df.columns:
        raise ValueError(f"{LABELS['temp_avg']} was not found in the df. Please check your temp_avg label.")
    if LABELS['date_time'] not in df.columns:
        raise ValueError(f"{LABELS['date_time']} was not found in the df. Please check your date_time label.")
    
    # Extract air temperatures
    air_temp = df[LABELS['temp_avg']]

    # Estimate sinusoidal parameters
    avg_temp = air_temp.mean()    # Units: °C
    thermal_amp = (max(air_temp) - min(air_temp)) / 2 # Units: °C
    min_date = df.loc[air_temp.idxmin(), LABELS['date_time']] # Datetime object
    min_doy = (min_date - pd.Timestamp(min_date.year, 1, 1)).days # Units: Days

    # Generate daily temperature predictions using the model
    T_pred = avg_temp + thermal_amp * np.cos(2 * np.pi * ((df[LABELS['doy']] - min_doy) / 365) + np.pi)

    return np.asarray(T_pred)


# Soil Temperature models
def soil_temp_at_depth(
        depth: int,
        avg_temp: int=25,
        thermal_amp: int = 10,
        thermal_dif: float = 0.203,
        time_lag: int = 15
    ) -> np.ndarray:
    """
    Models soil temperature over one year at the given depth

    Args:
        depth: The depth to model the soil temperature (m)
        avg_temp: The annual average surface temperature (°C)
        thermal_amp: The annual thermal amplitude of the soil surface (°C)
        thermal_dif: The Thermal diffusivity obtained from KD2 Pro instrument (mm^2 / s)
        time_lag: The time lag between Jaunary 1st and the coldest day of the year (days)

    Return:
        Predicted soil temperature at the given depth for each day of the year (°C)
    """
    # Set Constants
    OMEGA = 2 * np.pi / YEAR_SECONDS # Phase Frequency | Units: 1 / s
    PHASE = np.pi/2 + OMEGA * (time_lag * DAY_SECONDS) # Phase constant | Units: 

    # Calculate Damping Depth
    thermal_dif = thermal_dif / 100    # Unit Conversion: cm^2 / s
    damp_depth = (2*thermal_dif/OMEGA)**(1/2) # Units: cm

    # Estimate the temperatures for each day of the year
    doy = np.arange(0,365)*DAY_SECONDS # Units: s
    soil_temp = avg_temp + \
        thermal_amp * np.exp(-depth / damp_depth) * \
        np.sin(OMEGA * doy - depth / damp_depth - PHASE)
    
    return  np.asarray(soil_temp)

def soil_temp_on_day(
        doy: int,
        max_depth: int,
        Nz: int = 100,
        avg_temp: int = 25,
        thermal_amp: int = 10,
        thermal_dif: int = 0.203,
        timelag: int = 15
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Models soil temperature on a particular day of the year
    
    Args:
        doy: Day of year (0-365) where January 1st is 0 and 365
        max_depth: The maximum depth in centimeters to model the soil temperature
        Nz: The number of interpolated depths to caluculate soil temperature at
        avg_temp: The annual average surface temperature (°C)
        thermal_amp: The annual thermal amplitude of the soil surface (°C)
        thermal_dif: The Thermal diffusivity obtained from KD2 Pro instrument [mm^2/s]
        time_lag: The time lag in days (0-365) where January 1st is 0 and 365

    Returns:
        An np array of the soil temps and an np array of the depths of the soil temps (°C)
    """
    # Set Constants
    OMEGA = 2 * np.pi / (YEAR_SECONDS) # Phase Frequency | Units: 1 / s
    PHASE = np.pi / 2 + OMEGA * (timelag * DAY_SECONDS) # Phase constant

    # Calculate Damping Depth
    thermal_dif = thermal_dif / 100    # Unit Conversion: cm^2 / s
    damp_depth = (2 * thermal_dif / OMEGA) ** 0.5 # Units: cm

    # Estimate the temperatures for each depth
    depths = np.linspace(0, max_depth, Nz) # Interpolation depths | Units: cm
    T_soil = avg_temp + \
        thermal_amp * np.exp(-depths / damp_depth) * \
        np.sin(OMEGA * doy - depths / damp_depth - PHASE)

    return (np.asarray(depths), np.asarray(T_soil))

def soil_temp_at_depth_on_day(
        max_depth: int,
        Nz: int = 1000,
        avg_temp: int = 25,
        thermal_amp: int = 10,
        thermal_dif: int = 0.203,
        timelag: int = 15
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Models soil temperature over a full year (0–365) and across depth.
    
    Args:
        max_depth: The maximum depth in centimeters to model the soil temperature
        Nz: The number of interpolated depths to calculate soil temperature at
        avg_temp: The annual average surface temperature (°C)
        thermal_amp: The annual thermal amplitude of the soil surface (°C)
        thermal_dif: The Thermal diffusivity obtained from KD2 Pro instrument [mm^2/s]
        timelag: The time lag in days (0-365) where January 1st is 0 and 365
    
    Returns:
        A tuple of (doy_grid, z_grid, temp_grid), where each is a 2D NumPy array.
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

    return doy_grid, z_grid, t_grid

def model_soil_temp_from_air_temp(
        df: pd.DataFrame,
        depth: float,
        date: str,
        date_format: str = LABELS['date_format'],
        thermal_dif: int = 0.203,
    ) -> np.ndarray:
    """
    Creates temperature predictions for every 5 minutes for a modeled date

    Estimates the parameters for a sinusodial function modeling the daily surface temperature.
    Uses the parameter estimations to create a model of soil temperatures at different depths

    Args:
        df: DataFrame containing temperature data, must contain 5_minute_temp as defined in settings.
        depth: The depth in centimeters to model the soil temperature
        date: Is the date which parameters will be estimated for.
        date_format: Is the format that the date variable and data use for the date.
        thermal_dif: The Thermal diffusivity obtained from KD2 Pro instrument [mm^2/s]

    Returns:
        A numpy array containing the predicted temperatures (°C) every 5 minutes 
            starting at midnight at the specified depth and a numpy array containing
            the given air temperatures
    """
    # Constants
    global LABELS
    OMEGA = 2 * np.pi / DAY_SECONDS # Phase Frequency | Units: 1 / s

    # Calculate Damping Depth
    thermal_dif = thermal_dif / 100 # Unit Conversion: cm^2 / s
    damp_depth = (2 * thermal_dif / OMEGA) ** 0.5 # Units: cm

    # Find all the entries in the dataframe with the correct date and sort by time
    target_date = pd.to_datetime(date, format=date_format).normalize()
    dates = pd.to_datetime(df[LABELS['date_norm']], format=date_format)
    daily_df = df[dates == target_date].sort_values(LABELS['date_time'])
    # Raise an exception if no data for the specified date could be found
    if len(daily_df) == 0:
        raise ValueError(f"Could not find data in the given dataframe for:\t{date}")

    # Extract the 5-minute air temperatures and ensure all exist
    air_temp = daily_df[LABELS['5_minute_temp']]
    if len(air_temp) != DAY_SECONDS / (5*60):
        raise ValueError(f"Expected 288 5-minute temperature values for {target_date.date()}, got {len(air_temp)}")

    # Estimate sinusoidal parameters
    avg_temp = air_temp.mean()    # Units: °C
    thermal_amp = (max(air_temp) - min(air_temp)) / 2 # Units: °C
    min_time = daily_df.loc[air_temp.idxmin(), LABELS['date_time']]
    timelag = (min_time.hour * 60 + min_time.minute) * 60 # Units: s

    # Generate predictions for every 5 minutes
    time = np.arange(0, DAY_SECONDS, 5 * 60) # Units: s
    temp_predictions = avg_temp + \
        thermal_amp * np.exp(-depth / damp_depth) * \
        np.sin(OMEGA * (time - timelag) - (depth / damp_depth))


    # DEBUGGING
    soil_temp_actual = daily_df[LABELS['soil_temp4']]
    # DEBUGGING


    return temp_predictions, air_temp.to_numpy(), soil_temp_actual

def model_soil_temp_at_depth_from_air_temp(
        df: pd.DataFrame,
        max_depth: float,
        date: str,
        Nz: int = 1000,
        date_format: str = '%m/%d/%Y %I:%M %p',
        thermal_dif: float = 0.203
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Models soil temperature over a full day (in 5-minute intervals) across depth 
    using air temperature data to parameterize a sinusoidal model.

    Args:
        df: DataFrame containing temperature data; must contain '5_minute_temp' and datetime.
        max_depth: Maximum depth in centimeters to model the soil temperature.
        date: The date (string) for which parameters will be estimated.
        Nz: Number of interpolated depth points.
        date_format: The format used for the date string.
        thermal_dif: Thermal diffusivity [mm^2/s] from KD2 Pro instrument.

    Returns:
        A tuple of (time_grid, depth_grid, temp_grid), each a 2D NumPy array.
    """
    # Constants
    global LABELS
    OMEGA = 2 * np.pi / DAY_SECONDS # Phase Frequency | Units: 1 / s

    # Calculate Damping Depth
    thermal_dif = thermal_dif / 100 # Unit Conversion: cm^2 / s
    damp_depth = (2 * thermal_dif / OMEGA) ** 0.5 # Units: cm

    # Find all the entries in the dataframe with the correct date and sort by time
    target_date = pd.to_datetime(date, format=date_format).normalize()
    dates = pd.to_datetime(df[LABELS['date_norm']], format=date_format)
    daily_df = df[dates == target_date].sort_values(LABELS['date_time'])
    # Raise an exception if no data for the specified date could be found
    if len(daily_df) == 0:
        raise ValueError(f"Could not find data in the given dataframe for:\t{date}")

    # Extract the 5-minute air temperatures and ensure all exist
    air_temp = daily_df[LABELS['5_minute_temp']]
    if len(air_temp) != DAY_SECONDS / (5*60):
        raise ValueError(f"Expected 288 5-minute temperature values for {target_date.date()}, got {len(air_temp)}")

    # Estimate sinusoidal parameters
    avg_temp = air_temp.mean()    # Units: °C
    thermal_amp = (max(air_temp) - min(air_temp)) / 2 # Units: °C
    min_time = daily_df.loc[air_temp.idxmin(), LABELS['date_time']]
    timelag = (min_time.hour * 60 + min_time.minute) * 60 # Units: s

    # Create time and depth grids
    time = np.arange(0, DAY_SECONDS, 5)
    depths = np.linspace(max_depth/Nz, max_depth, Nz)
    time_grid, depth_grid = np.meshgrid(time, depths)

    # Generate predictions for the given times and depths
    temp_grid = avg_temp + \
        thermal_amp * np.exp(-depth_grid / damp_depth) * \
        np.sin(OMEGA * (time_grid - timelag) - depth_grid / damp_depth)

    return time_grid, depth_grid, temp_grid


# EvapoTranspiration Models
def dalton(
        temp_min: np.ndarray,
        temp_max: np.ndarray,
        RH_min: np.ndarray,
        RH_max: np.ndarray,
        wind_speed: np.ndarray
    ) -> np.ndarray:
    """
    Computes Evapotranspiration using the Dalton model 
    
    Args:
        temp_min: The minimum daily temperature (°C)
        temp_max: The maximum daily temperature (°C)
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
        temp_min: np.ndarray,
        temp_max: np.ndarray,
        RH_min: np.ndarray,
        RH_max: np.ndarray,
        wind_speed: np.ndarray
    ) -> np.ndarray:
    """
    Computes evapotranspiration using the Penman model

    Args:
        temp_min: The minimum daily temperature (°C)
        temp_max: The maximum daily temperature (°C)
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
        temp_min: np.ndarray,
        temp_max: np.ndarray,
        RH_min: np.ndarray,
        RH_max: np.ndarray,
    ) -> np.ndarray:
    """
    Potential evaporation model proposed by Romanenko in 1961
    
    Args:
        temp_min: The minimum daily temperature (°C)
        temp_max: The maximum daily temperature (°C)
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
        temp_min: np.ndarray,
        temp_max: np.ndarray,
        doy: np.ndarray,
        latitude: np.ndarray
    ) -> np.ndarray:
    """
    Computes evapotranspiration using the Jensen model
    
    Args:
        temp_min: The minimum daily temperature (°C)
        temp_max: The maximum daily temperature (°C)
        doy: Day of year (0-365) where January 1st is 0 and 365
        latitude: Latitude in decimal degress. Where the northern hemisphere is 
            positive and the southern hemisphere is negative

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
        temp_min: np.ndarray,
        temp_max: np.ndarray,
        doy: np.ndarray,
        latitude: float
    ) -> np.ndarray:
    """
    Computes evapotranspiration using the Hargreaves model
    
    Args:
        temp_min: The minimum daily temperature (°C)
        temp_max: The maximum daily temperature (°C)
        doy: Day of year (0-365) where January 1st is 0 and 365
        latitude: Latitude in decimal degress. Where the northern hemisphere is 
            positive and the southern hemisphere is negative
        
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
        temp_min: np.ndarray,
        temp_max: np.ndarray,
        RH_min: np.ndarray,
        RH_max: np.ndarray,
        p_min: np.ndarray,
        p_max: np.ndarray,
        wind_speed: np.ndarray,
        doy: np.ndarray,
        latitude: float,
        altitude: float,
        wind_height: int = 1.5,
    ) -> np.ndarray:
    """
    PComputer evapotranspiration using the penman-monteith model
    
    Args:
        temp_min: The minimum daily temperature (°C)
        temp_max: The maximum daily temperature (°C)
        RH_min:   The minimum daily relative humidity (range: 0.0-1.0)
        RH_max:   The maximum daily relative humidity (range: 0.0-1.0)
        p_min:    The minimum daily atmospheric pressure in Pa
        p_max:    The maximum daily atmospheric pressure in Pa
        wind_speed: The average daily wind speed in meters per second
        doy:      Day of year (0-365) where January 1st is 0 and 365
        latitude: Latitude in decimal degress. Where the northern hemisphere is 
            positive and the southern hemisphere is negative
        altitude: Altitude of the location in meters
        wind_height: Height of measurment for wind speed
        
    Returns:
        The Hargreaves models predictions for evapotranspiration clipped to
            a minimum of zero
    """
    # Ensure parameter saftey
    if not isinstance(temp_min, float) and not (len(temp_min) == len(temp_max) == len(RH_min)\
                                     == len(RH_max) ==  len(wind_speed) == len(wind_speed)):
        raise ValueError("All inputs must be the same length")
    
    temp_avg = (temp_min + temp_max)/2
    atm_pressure = (p_min+p_max)/2 # Can be also obtained from weather station
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
    solar_rad = compute_Ra(doy, latitude)

    # Clear Sky Radiation: Rso (MJ/m2/day)
    Rso =  (0.75 + (2 * 10**-5) * altitude) * solar_rad  # Eq. 37, FAO-56

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

def EvapoTranspiration_to_df(
        df: pd.DataFrame,
        model_name: str,
        temp_min: Optional[np.ndarray] = None,
        temp_max: Optional[np.ndarray] = None,
        RH_min: Optional[np.ndarray] = None,
        RH_max: Optional[np.ndarray] = None,
        wind_speed: Optional[np.ndarray] = None,
        p_min: Optional[np.ndarray] = None,
        p_max: Optional[np.ndarray] = None,
        doy: Optional[np.ndarray] = None,
        latitude: Optional[float] = None,
        altitude: Optional[float] = None,
        wind_height: int = 1.5
    ) -> pd.DataFrame:
    """
    Adds daily evapotranspiration and its cumulative sum to the given DataFrame 
    using a specified model and its required parameters.

    Args:
        df: The DataFrame to add the evapotranspiration data to
        model_name: Name of the model to use. Options:
                ["dalton", "penman", "romanenko", "jensen_haise", "hargreaves", "penman_monteith"]
        temp_min:   The minimum daily temperature (°C)
        temp_max:   The maximum daily temperature (°C)
        RH_min:     The minimum daily relative humidity (range: 0.0-1.0)
        RH_max:     The maximum daily relative humidity (range: 0.0-1.0)
        wind_speed: The average daily wind speed in meters per second
        p_min:      The minimum daily atmospheric pressure in Pa
        p_max:      The maximum daily atmospheric pressure in Pa
        doy:        Day of year (0-365) where January 1st is 0 and 365
        latitude:   Latitude in decimal degress. Where the northern hemisphere is 
            positive and the southern hemisphere is negative
        altitude:   Altitude of the location in meters
        wind_height: Height of measurement for wind speed in meters
    """
    model_name = model_name.lower()
    if model_name == "dalton":
        pet = dalton(temp_min, temp_max, RH_min, RH_max, wind_speed)
    elif model_name == "penman":
        pet = penman(temp_min, temp_max, RH_min, RH_max, wind_speed)
    elif model_name == "romanenko":
        pet = romanenko(temp_min, temp_max, RH_min, RH_max)
    elif model_name == "jensen_haise":
        pet = jensen_haise(temp_min, temp_max, doy, latitude)
    elif model_name == "hargreaves":
        pet = hargreaves(temp_min, temp_max, doy, latitude)
    elif model_name == "penman_monteith":
        pet = penman_monteith(temp_min, temp_max, RH_min, RH_max, p_min, p_max,
                              wind_speed, doy, latitude, altitude, wind_height)
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Must be one of: "
                         "['dalton', 'penman', 'romanenko', 'jensen_haise', 'hargreaves', 'penman_monteith'].")

    df[LABELS['evapotranspiration'] + "-" + model_name] = pet


# Rain/Runoff Models
def model_runoff(precip: np.ndarray, cn: int = 75) -> pd.DataFrame:
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
    """
    Adds rainfall sum, runoff sum, and runoff to the DataFrame

    Args:
        df: The DataFrame containing the rainfall data
        cn: The curve number to be used in the runoff calculation
    """
    global LABELS

    # Check Parameters
    if LABELS['rain'] not in df.columns:
        raise ValueError(f"{LABELS['rain']} was not found in the df. Please check your rain label.")
    
    # Add data to the df
    df[LABELS["rain_sum"]] = df[LABELS["rain"]].cumsum()
    df[LABELS["runoff"]] = model_runoff(df[LABELS["rain"]], cn=cn)
    df[LABELS["runoff_sum"]] = df[LABELS["runoff"]].cumsum()


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


# Growing Degree Days
def model_gdd(
        temp_avg: np.ndarray,
        temp_base: float,
        temp_opt: Optional[float] = None,
        temp_upper: Optional[float] = None,
        duration_time: float = 1.0
    ) -> np.ndarray:
    """
    Model how many growing degree days have passed

    Models Growing Degree days using a minimum base temperature, an optional 
        optimal temperature, an optional maximum growing temperature, and 
        the average temperature over the recorded durations
        
    Args:
        temp_avg:  The average temperature over the duration_time (°C)
        temp_base: The minimum temperature a given crop will grow at (°C)
        temp_opt: The optimal temperature a given crop will grow (°C), 
                            above this temperature growing will slow linearly
        temp_upper: The maximum temperature a given crop will grow (°C)
        duration_time: The number of days that each temp_avg represents
    
    Returns:
        An np.ndarray with the calculated Growing Degree Days for each daily temperature
    """
    # Validate parameter input
    if(duration_time <= 0):
        raise ValueError("The time duration must be positive")
    if temp_avg is None or (not np.isscalar(temp_avg) and len(temp_avg) == 0):
        raise ValueError("You must provide temperature averages, None provided")
    if temp_opt is None and temp_upper is not None or\
        temp_upper is None and temp_opt is not None:
        raise ValueError("You must provide both upper and optimal temperatures or neither")
    
    # Ensure the temp_avg is a numpy array
    temp_avg = np.asarray(temp_avg)
    
    # Simply GDD calculation
    if temp_opt is None and temp_upper is None:
        gdd_days = (temp_avg - temp_base)*duration_time
        return np.maximum(0, gdd_days)

    # Initialize GDD array
    gdd = np.zeros_like(temp_avg)

    # Vectorized masks
    below_opt = temp_avg <= temp_opt
    above_opt = temp_avg > temp_opt 
    
    # Compute GDD where temp < temp_opt
    gdd[below_opt] = np.maximum( 0, (temp_avg[below_opt] - temp_base) * duration_time)
    
    # Compute GDD where temp >= temp_opt
    gdd_max = (temp_opt - temp_base) * duration_time
    gdd[above_opt] = np.maximum(0, 
        gdd_max * (temp_upper - temp_avg[above_opt]) / (temp_upper - temp_opt),)
    
    return gdd

def gdd_to_df(
        df: pd.DataFrame,
        temp_avg: np.ndarray,
        temp_base: float,
        temp_opt: Optional[float] = None,
        temp_upper: Optional[float] = None,
        duration_time: float = 1
    ):
    """
    Models Growing Degree days using a minimum base temperature, an optional 
        optimal temperature, an optional maximum growing temperature, and 
        the average temperature over the recorded durations.
        Adds the Growing Degree days and cummulative GDD to the given DataFrame

    Args:
        df: The DataFrame to add the GDD data to
        temp_avg:  The average temperature over the duration_time (°C)
        temp_base: The minimum temperature a given crop will grow at (°C)
        temp_optimal: The optimal temperature a given crop will grow (°C), 
                            above this temperature growing will slow linearly
        temp_upper: The maximum temperature a given crop will grow (°C)
        duration_time: The number of days that each temp_avg represents
    """
    global LABELS
    
    # Calculate gdd
    gdd = model_gdd(temp_avg, temp_base, temp_opt, temp_upper, duration_time)

    df[LABELS['gdd']] = gdd
    df[LABELS['gdd_sum']] = gdd.cumsum()


# Photo Period Tools
def photoperiod_at_latitude(latitude: float, doy: np.ndarray):
    """
    Computes photoperiod for a given latitude and day of year. Not accurate near polar regions.

    Args:
        latitude: Latitude in decimal degress. Where the northern hemisphere is 
            positive and the southern hemisphere is negative
        doy: np.ndarray of the days of year (0-365) where January 1st is 0 and 365

    Returns:
        Photoperiod, daylight hours, at the given latitude for the given days.
        The angle of the sum below the horizon.
        The zenithal distance of the sun in degrees
        The mean anomaly of the sun
        The declination of the sun in degrees
        Lambda
        Delta
    """
    # Convert latitude to radians
    latitude = np.radians(latitude)
    
    # Angle of the sun below the horizon. Civil twilight is -4.76 degrees.
    light_intensity = 2.206 * 10**-3
    B = -4.76 - 1.03 * np.log(light_intensity) # Eq. [5].

    # Zenithal distance of the sun in degrees
    alpha = np.radians(90 + B) # Eq. [6]. Value at sunrise and sunset.
    
    # Mean anomaly of the sun. It is a convenient uniform measure of 
    # how far around its orbit a body has progressed since pericenter.
    M = 0.9856*doy - 3.251 # Eq. [4].
    
    # Declination of sun in degrees
    Lambda = M + 1.916*np.sin(np.radians(M)) + 0.020*np.sin(np.radians(2*M)) + 282.565 # Eq. [3]. Lambda
    C = np.sin(np.radians(23.44)) # 23.44 degrees is the orbital plane of Earth around the Sun
    delta = np.arcsin(C*np.sin(np.radians(Lambda))) # Eq. [2].

    # Calculate daylength in hours, defining sec(x) = 1/cos(x)
    P = 2/15 * np.degrees( np.arccos( np.cos(alpha) * (1/np.cos(latitude)) * (1/np.cos(delta)) - np.tan(latitude) * np.tan(delta) ) ) # Eq. [1].

    return P, B, alpha, M, Lambda, np.degrees(delta)
    
def photoperiod_on_day(latitude: np.ndarray, doys: np.ndarray):
    """
    Computes photoperiod for a given near polar regions.

    Args:
        latitude: Latitude in decimal degress. Where the northern hemisphere is 
            positive and the southern hemisphere is negative
        doys: The day of year (0-365) where January 1st is 0 and 365 to perform the calculation

    Returns:
        Photoperiod, daylight hours, for the given latitudes on the given day.
        The angle of the sum below the horizon.
        The zenithal distance of the sun in degrees
        The mean anomaly of the sun
        The declination of the sun in degrees
        Lambda from the equation
        Delta from the equation
    
    References:
        Keisling, T.C., 1982. Calculation of the Length of Day 1. Agronomy Journal, 74(4), pp.758-759.
    """
    # Convert latitude to radians and convert shapes
    latitude = np.radians(np.asarray(latitude)).reshape(-1, 1)  # shape (N, 1)
    doys = np.asarray(doys).reshape(1, -1) 
    
    # Angle of the sun below the horizon. Civil twilight is -4.76 degrees.
    light_intensity = 2.206 * 10**-3
    B = -4.76 - 1.03 * np.log(light_intensity) # Eq. [5].

    # Zenithal distance of the sun in degrees
    alpha = np.radians(90 + B) # Eq. [6]. Value at sunrise and sunset.
    
    # Mean anomaly of the sun. It is a convenient uniform measure of 
    # how far around its orbit a body has progressed since pericenter.
    M = 0.9856*doys - 3.251 # Eq. [4].
    
    # Declination of sun in degrees
    Lambda = M + 1.916*np.sin(np.radians(M)) + 0.020*np.sin(np.radians(2*M)) + 282.565 # Eq. [3]. Lambda
    C = np.sin(np.radians(23.44)) # 23.44 degrees is the orbital plane of Earth around the Sun
    delta = np.arcsin(C*np.sin(np.radians(Lambda))) # Eq. [2].

    # Calculate daylength in hours, defining sec(x) = 1/cos(x)
    P = 2/15 * np.degrees( np.arccos( np.cos(alpha) * (1/np.cos(latitude)) * (1/np.cos(delta)) - np.tan(latitude) * np.tan(delta) ) ) # Eq. [1].

    return P, B, alpha, M, Lambda, np.degrees(delta)


# Soil Water Flow Functions
def hydraulic_conductivity(
        k_s: float,
        psi_e: float,
        theta: float,
        theta_s: float,
        b: float,
    ):
    """
    Estimates the hydraulic conductivity of soil
    
    Args:
        k_s: The saturated conductivity of the soil  (kg * s / m^3)
        psi_e: The air entry water potential of the soil  (J / kg)
        theta: Is the volumetric water content of the soil (m^3 / m^3)
        theta_s: Is the saturation water content of the soil (m^3 / m^3)
        b: Is the exponent of moisture release parameter

    Returns:
        The calculated hydraulic conductivity of the soil given the parameters ()
    """
    psi_m = psi_e* (theta / theta_s) ** (-b)

    k_psi_m = k_s * (psi_e / psi_m) ** (2 + 3/b)

    return k_psi_m

def cummulative_water_infultration(
        delta_theta: float,
        K_avg: float,
        psi_mi: float,
        psi_mf: float,
        time: int,
        Nz: int=100
    ):
    """
    Calculates the cummulative vertical water infultration.
    
    Calculates the cummulative vertical water infultration for the given soil parameters.
        Calculations are made for equally spaced times from [time/Nz to time]
    Args:
        delta_theta: The Volume Fraction of water
        K_avg: The average hydraulic conductivity of the wet soil (kg * s /m^3)
        psi_mi: The infultration boudary's water potential (J / kg)
        psi_mf: The wetting front's water potentail (J / kg)
        time: The length of time to calcute the cummulative water infultration (s)
        Nz: The number of interpolated times to calculate

    
    Returns:
        A numpy array of length Nz representing the cummulative vertical water
            infultration over time. Where the first value represents the 
            infultraiton after zero time, the second value represents the
            infultration after time/Nz time, and the last vaule represents the
            infultration after time.
    """
    # Define Constants
    WATER_DENSITY = 1000 # Units: Kg / m^3
    LITTLE_G = 9.80665 # Units: m / s^2

    if delta_theta < 0 or delta_theta > 1.0:
        raise ValueError(f"The volume Fraction of water must be between [0,1], but {delta_theta} was provided")

    # Calculate water infultration at each of the time
    times = np.linspace(0, time, Nz)
    water_inful = 2 * WATER_DENSITY * delta_theta * K_avg * (psi_mi - psi_mf) * times
    water_inful =  water_inful ** 0.5 + LITTLE_G * K_avg * times

    return water_inful



