from typing import Optional, Tuple
from scipy.stats import circmean

import numpy as np
import pandas as pd

import agroecometrics as AEM


# Constants Representing the number of seconds in a Day and Year
DAY_SECONDS = 24*60*60
YEAR_SECONDS = DAY_SECONDS*365

# Helper Functions
def __compute_vapor_sat_pressure(temp: np.ndarray) -> np.ndarray:
    """
    Computes saturation vapor pressure based Tetens formula.
    
    Args:
        temp: Temperature used to calculate the saturation vapor pressure (°C).
    
    Returns:
        Computed saturation vapor pressure (kPa).
    """
    e_sat = 0.6108 * np.exp(17.27 * temp/(temp+237.3)) 
    return e_sat

def __compute_solar_radiation(doy: np.ndarray, lat: float) -> np.ndarray:
    """
    Computes extra-terrestrial solar radiation using the FAO Penman-Monteith method.
    
    Args:
        doy: Day of year (0-365) where January 1st is 0 and 365.
        lat: Latitude in decimal degress. Where the northern hemisphere is positive and the southern hemisphere is negative.

    Returns:
        Extra-terrestrial solar radiation (Ra) in MJ/m²/day.
    """
    lat_radians = np.pi / 180 * lat 
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy/365) # Inverse relative distance Earth-Sun
    solar_declination = 0.409 * np.sin((2 * np.pi * doy/365) - 1.39) # Solar delcination
    Gsc = 0.0820 # Solar constant

    sunset_hour_angle = np.arccos(-np.tan(lat_radians) * np.tan(solar_declination)) # Sunset hour angle

    Ra = 24 * 60 / np.pi * Gsc * dr
    Ra = Ra * (sunset_hour_angle * np.sin(lat_radians) * np.sin(solar_declination) + np.cos(lat_radians) * np.cos(solar_declination) * np.sin(sunset_hour_angle))

    return Ra

# Air Temperature model
def model_air_temp(air_temps: np.ndarray, date_times: np.ndarray) -> np.ndarray:
    '''
    Creates an air temperature model and creates air temperature estimates using model.

    Creates an air temperature model by finding the best fit for a sinusoidal function.
    Estimates Parameters using the given data.
    Creates a temperature estimate for each day in the df using the model

    Args:
        air_temps: The daily average air temperature (°C)
        date_times: The datetime corresponding to each air temperature measurement
    
    Returns: 
        A numpy array of predicted daily temperatures over the period of the dataframe
    '''
    # Estimate sinusoidal parameters
    avg_temp = np.mean(air_temps)    # Mean Temperature | Units: °C
    thermal_amp = (np.max(air_temps) - np.min(air_temps)) / 2  # | Units: °C

    # Use the day with the minimum and maximum temperatures to calulate time_laf
    min_idx = np.argmin(air_temps[:int(len(air_temps) / 2)])
    min_date = pd.to_datetime(date_times[min_idx])
    max_idx = np.argmax(air_temps)
    max_date = pd.to_datetime(date_times[max_idx])
    time_lag_date = min_date + (max_date - pd.Timedelta(days=183) - min_date) / 2
    time_lag = (time_lag_date - pd.Timestamp(time_lag_date.year, 1, 1)).days  # days since Jan 1

    # Convert all dates to day of year
    date_times = pd.to_datetime(date_times)
    doys = date_times.dt.dayofyear

    # Generate sinusoidal temperature predictions
    pred_temp = avg_temp + thermal_amp * np.cos(
        2 * np.pi * ((doys - time_lag) / 365.0) + np.pi
    )

    return np.asarray(pred_temp)


# Soil Temperature models
def yearly_soil_temp(
        depth: int,
        avg_temp: int=25,
        thermal_amp: int = 10,
        thermal_diffusivity: float = 0.203,
        time_lag: int = 15
    ) -> np.ndarray:
    """
    Models soil temperature over one year at the given depth.

    Creates a model to estimate soil temperature given a day of year and depth.
    Predicts the soil temperature everyday at the specified depth.

    Args:
        depth: The depth to model the soil temperature (m).
        avg_temp: The annual average surface temperature (°C).
        thermal_amp: The annual thermal amplitude of the soil surface (°C).
        thermal_dif: The Thermal diffusivity obtained from KD2 Pro instrument (mm^2 / s).
        time_lag: The time lag between Jaunary 1st and the coldest day of the year (days).

    Return:
        Predicted soil temperature at the given depth for each day of the year (°C).
    """
    # Set Constants
    phase_frequency = 2 * np.pi / YEAR_SECONDS # Phase Frequency | Units: 1 / s
    PHASE = np.pi/2 + phase_frequency * (time_lag * DAY_SECONDS) # Phase constant | Units: None

    # Calculate Damping Depth
    thermal_diffusivity = thermal_diffusivity / 100    # Unit Conversion | Units: cm^2 / s
    damp_depth = (2*thermal_diffusivity/phase_frequency)**(1/2) # Units: cm

    # Estimate the temperatures for each day of the year
    doy = np.arange(0,365)*DAY_SECONDS # Units: s
    soil_temp = avg_temp + \
        thermal_amp * np.exp(-depth / damp_depth) * \
        np.sin(phase_frequency * doy - depth / damp_depth - PHASE)
    
    return  np.asarray(soil_temp)

def daily_soil_temp(
        doy: int,
        max_depth: int,
        interpolations: int = 100,
        avg_temp: int = 25,
        thermal_amp: int = 10,
        thermal_diffusivity: int = 0.203,
        timelag: int = 15
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Models soil temperature on a particular day of the year

    Creates a model to estimate soil temperature given a day of year and depth.
    Calculates depths to use by linearly interpolating the max_depth by the number of interpolations.
    Uses the model to estimate the soil temperature at a range of depths on the specified day of year.
    
    Args:
        doy: Day of year (0-365) where January 1st is 0 and 365.
        max_depth: The maximum depth in centimeters to model the soil temperature.
        interpolations: The number of interpolated depths to caluculate soil temperature at.
        avg_temp: The annual average surface temperature (°C).
        thermal_amp: The annual thermal amplitude of the soil surface (°C).
        thermal_diffusivity: The Thermal diffusivity obtained from KD2 Pro instrument [mm^2/s].
        time_lag: The time lag in days (0-365) where January 1st is 0 and 365.

    Returns:
        A tuple containing two numpy arrays.
            The first array contains the soil temperature predictions (°C)
            The second array contians the depth of each prediction (cm)
    """
    # Set Constants
    phase_frequency = 2 * np.pi / (YEAR_SECONDS) # Phase Frequency | Units: 1 / s
    PHASE = np.pi / 2 + phase_frequency * (timelag * DAY_SECONDS) # Phase constant | Units: None

    # Calculate Damping Depth
    thermal_diffusivity = thermal_diffusivity / 100    # Unit Conversion | Units: cm^2 / s
    damp_depth = (2 * thermal_diffusivity / phase_frequency) ** 0.5 # Units: cm

    # Estimate the temperatures for each depth
    soil_depths = np.linspace(max_depth/interpolations, max_depth, interpolations) # Interpolation depths | Units: cm
    soil_temp = avg_temp + \
        thermal_amp * np.exp(-soil_depths / damp_depth) * \
        np.sin(phase_frequency * doy - soil_depths / damp_depth - PHASE)

    return (np.asarray(soil_temp), np.asarray(soil_depths))

def yearly_3d_soil_temp(
        max_depth: int,
        interpolations: int = 1000,
        avg_temp: int = 25,
        thermal_amp: int = 10,
        thermal_diffusivity: int = 0.203,
        timelag: int = 15
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Models soil temperature over a full year (0–365) and across depth.

    Creates a model to estimate soil temperature given a day of year and depth.
    Calculates depths to use by linearly interpolating the max_depth by the number of interpolations
    Uses the model to create a matrix of estimations for each day of the year at each depth.
        
    Args:
        max_depth: The maximum depth in centimeters to model the soil temperature
        interpolations: The number of interpolated depths to calculate soil temperature at
        avg_temp: The annual average surface temperature (°C)
        thermal_amp: The annual thermal amplitude of the soil surface (°C)
        thermal_diffusivity: The Thermal diffusivity obtained from KD2 Pro instrument [mm^2/s]
        timelag: The time lag in days (0-365) where January 1st is 0 and 365
    
    Returns:
        A tuple of (doy_grid, z_grid, temp_grid), where each is a 2D NumPy array. 
        Each grid represents a Matrix of depths and doys.
        doy_grid varies the day of year between columns. Each columns represents the same doy.
        depth_grid varies the depth between rows. Each row represents the same depth
        temp_grid varies between rows and columns. 
        Each point represents the predicted temperature for the doy and depth indicated by the same point in the doy_grid and depth_grid respectively
    """
    # Set Constants
    phase_frequency = 2 * np.pi / 365 # Phase Frequency | Units: 1 / s
    PHASE = np.pi/2 + phase_frequency*timelag # Phase constant | Units: None

    thermal_diffusivity = thermal_diffusivity * 0.01 * 86400 # Unit conversion | Units: cm^2/day
    damp_depth = (2*thermal_diffusivity/phase_frequency)**(1/2) # Damping depth 

    doys = np.arange(1,366)
    depths = np.linspace(0, max_depth, interpolations) # Interpolation depths
    doy_grid, depth_grid = np.meshgrid(doys,depths)
    
    temp_grid = avg_temp + thermal_amp * np.exp(-depth_grid/damp_depth) * np.sin(phase_frequency*doy_grid - depth_grid/damp_depth - PHASE)

    return doy_grid, depth_grid, temp_grid

def model_soil_temp(
        air_temps: np.ndarray,
        depth: float,
        thermal_diffusivity: int = 0.203,
    ) -> np.ndarray:
    """
    Creates soil temperature predictions for each air temperature provided over the course of a day

    Estimates the parameters for a sinusodial function modeling the daily surface temperature.
    Uses the parameter estimations to create a model of soil temperatures at different depths.
    Predicts the soil temperatue at the given depth for each air temperature provided

    Args:
        air_temps: The soil surface air temperatures for a given day (°C).
        depth: The depth to model the soil temperature (cm).
        thermal_diffusivity: The Thermal diffusivity obtained from KD2 Pro instrument [mm^2/s].

    Returns:
        A numpy array containing the predicted temperatures (°C) every 5 minutes 
            starting at midnight at the specified depth and a numpy array containing
            the given air temperatures
    """
    # Constants
    phase_frequency = 2 * np.pi / DAY_SECONDS # Phase Frequency | Units: 1 / s

    # Calculate Damping Depth
    thermal_diffusivity = thermal_diffusivity / 100 # Unit Conversion: cm^2 / s
    damp_depth = (2 * thermal_diffusivity / phase_frequency) ** 0.5 # Units: cm

    measurement_offset = DAY_SECONDS/len(air_temps)

    # Estimate sinusoidal parameters
    avg_temp = np.mean(air_temps)    # Units: °C
    thermal_amp = (max(air_temps) - min(air_temps)) / 2 # Units: °C

    # Find TimeLag
    min_idx = np.argmin(air_temps)
    timelag = min_idx*measurement_offset

    # Generate predictions for every air temperature provided
    times = [i*measurement_offset for i in len(air_temps)] # Units: s
    temp_predictions = avg_temp + \
        thermal_amp * np.exp(-depth / damp_depth) * \
        np.sin(phase_frequency * (times - timelag) - (depth / damp_depth))

    return temp_predictions

def model_3d_soil_temp(
        air_temps: np.ndarray,
        max_depth: float,
        interpolations: int = 1000,
        thermal_diffusivity: float = 0.203
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Models soil temperature over a full day (in 5-minute intervals) across depth 
    using air temperature data to parameterize a sinusoidal model.

    Args:
        air_temps: The soil surface air temperatures for a given day (°C).
        max_depth: Maximum depth to model the soil temperature (cm).
        interpolations: Number of interpolated depth points.
        thermal_diffusivity: Thermal diffusivity [mm^2/s] from KD2 Pro instrument.

    Returns:
        A tuple of (time_grid, depth_grid, temp_grid), each a 2D NumPy array.
    """
    # Constants
    phase_frequency = 2 * np.pi / DAY_SECONDS # Phase Frequency | Units: 1 / s

    # Calculate Damping Depth
    thermal_diffusivity = thermal_diffusivity / 100 # Unit Conversion: cm^2 / s
    damp_depth = (2 * thermal_diffusivity / phase_frequency) ** 0.5 # Units: cm


    measurement_offset = DAY_SECONDS/len(air_temps)

    # Estimate sinusoidal parameters
    avg_temp = np.mean(air_temps)    # Units: °C
    thermal_amp = (max(air_temps) - min(air_temps)) / 2 # Units: °C

    # Find TimeLag
    min_idx = np.argmin(air_temps)
    timelag = min_idx*measurement_offset

    # Create time and depth grids
    times = [i*measurement_offset for i in len(air_temps)] # Units: s
    depths = np.linspace(max_depth/interpolations, max_depth, interpolations)
    time_grid, depth_grid = np.meshgrid(times, depths)

    # Generate predictions for the given times and depths
    temp_grid = avg_temp + \
        thermal_amp * np.exp(-depth_grid / damp_depth) * \
        np.sin(phase_frequency * (time_grid - timelag) - depth_grid / damp_depth)

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
        The Dalton models predictions for evapotranspiration clipped to a minimum of zero (mm/day)
    """
    # Ensures parameter saftey
    if not isinstance(temp_min, float) and not \
        (len(temp_min) == len(temp_max) == len(RH_min) == len(RH_max) == len(wind_speed)):
        raise ValueError("All inputs must be the same length")
    
    # Model calulations
    e_sat_min = __compute_vapor_sat_pressure(temp_min)
    e_sat_max = __compute_vapor_sat_pressure(temp_max)
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
    e_sat_min = __compute_vapor_sat_pressure(temp_min)
    e_sat_max = __compute_vapor_sat_pressure(temp_max)
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
    Ra = __compute_solar_radiation(doy, latitude)
    T_avg = (temp_min + temp_max)/2
    PET = 0.0102 * (T_avg+3) * Ra

    # Ensures non-negative evapotranspiration
    PET = np.clip(PET, 0.0, None)

    return PET

def hargreaves(
        temp_min: np.ndarray,
        temp_max: np.ndarray,
        doys: np.ndarray,
        latitude: float
    ) -> np.ndarray:
    """
    Computes evapotranspiration using the Hargreaves model
    
    Args:
        temp_min: The minimum daily temperature (°C)
        temp_max: The maximum daily temperature (°C)
        doys: Array of Day of year's (0-365) where January 1st is 0 and 365
        latitude: Latitude in decimal degress. Where the northern hemisphere is 
            positive and the southern hemisphere is negative
        
    Returns:
        The Hargreaves models predictions for evapotranspiration clipped to a minimum of zero
    """
    # Ensure parameter saftey
    if not isinstance(temp_min, float) and not (len(temp_min) == len(temp_max)):
        raise ValueError("All inputs must be the same length")
    
    # Model Calulations
    Ra = __compute_solar_radiation(doys, latitude)
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
        solar_rad: Optional[np.ndarray]=None,
        wind_height: int = 1.5,
    ) -> np.ndarray:
    """
    Computed evapotranspiration using the penman-monteith model
    
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
    
    if solar_rad is not None:
        if len(solar_rad) != len(temp_min):
               raise ValueError("All inputs must be the same length")
    else:
        # Calculates solar radiation
        solar_rad = __compute_solar_radiation(doy, latitude)
    
    temp_avg = (temp_min + temp_max)/2
    atm_pressure = (p_min+p_max)/2 # Can be also obtained from weather station
    Cp = 0.001013; # Approx. 0.001013 for average atmospheric conditions
    gamma = 0.000665* (Cp * atm_pressure)

    # Wind speed Adjustment
    wind_speed_at_height = wind_speed * (4.87 / np.log((67.8 * wind_height) - 5.42))  # Eq. 47, FAO-56 wind height in [m]

    # Calculates air humidity and vapor pressure
    delta = 4098 * (0.6108 * np.exp(17.27 * temp_avg / (temp_avg  + 237.3)))
    delta = delta / (temp_avg  + 237.3)**2
    e_temp_max = 0.6108 * np.exp(17.27 * temp_max / (temp_max + 237.3)) # Eq. 11, //FAO-56
    e_temp_min = 0.6108 * np.exp(17.27 * temp_min / (temp_min + 237.3))
    e_saturation = (e_temp_max + e_temp_min) / 2
    e_actual = (e_temp_min * (RH_max / 100) + e_temp_max * (RH_min / 100)) / 2

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
    PET = 0.408 * delta * (Rn - soil_heat_flux)
    PET = PET + gamma * (900 / (temp_avg  + 273))  * wind_speed_at_height * (e_saturation - e_actual)
    PET = PET / (delta + gamma * (1 + 0.34 * wind_speed_at_height))


    # Ensures non-negative evapotranspiration
    PET = np.clip(PET, 0.0, None)

    return np.round(PET,2)

# Rain/Runoff Models
def model_runoff(rainfall: np.ndarray, curve_number: int = 75) -> pd.DataFrame:
    '''
    Uses Curve Number to estimate runoff from rainfall

    Args:
        precipitation: The daily amount of precipitation in millimeters
        curve_number: The curve number to use in the calculation

    Returns:
        The estimated runoff
    '''
    rainfall = rainfall / 25.4 # Unit Conversion | Units: In

    # Model Calulations
    runoff = np.zeros_like(rainfall)
    S02 = 1000 / curve_number - 10
    S005 = 1.33 * S02**1.15
    Lambda = 0.05
    Ia = S005 * Lambda
    idx = rainfall > Ia
    runoff[idx] = (rainfall[idx] - Ia)**2 / (rainfall[idx] - Ia + S005)

    return runoff * 25.4 # Unit Conversion | Units: mm

# Growing Degree Days
def model_gdd(
        temp_avg: np.ndarray,
        temp_base: float,
        temp_opt: Optional[float] = None,
        temp_upper: Optional[float] = None,
        time_duration: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        time_duration: The number of days that each temp_avg represents
    
    Returns:
        A tuple containing two np.ndarray's
        The first contains Growing Degree Days for each daily temperature
        The second contains the cummulative sum of Growing Degree Days for each daily temperature
    """
    # Validate parameter input
    if(time_duration <= 0):
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
        gdd_days = (temp_avg - temp_base)*time_duration
        gdd_days = np.maximum(0, gdd_days)
        return gdd_days, gdd_days.cumsum()

    # Initialize GDD array
    gdd = np.zeros_like(temp_avg)

    # Vectorized masks
    below_opt = temp_avg <= temp_opt
    above_opt = temp_avg > temp_opt 
    
    # Compute GDD where temp < temp_opt
    gdd[below_opt] = np.maximum( 0, (temp_avg[below_opt] - temp_base) * time_duration)
    
    # Compute GDD where temp >= temp_opt
    gdd_max = (temp_opt - temp_base) * time_duration
    gdd[above_opt] = np.maximum(0, 
        gdd_max * (temp_upper - temp_avg[above_opt]) / (temp_upper - temp_opt),)
    
    return gdd, gdd.cumsum()


# Photo Period Tools
def photoperiod_at_lat(lat: float, doy: np.ndarray) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
    """
    Computes photoperiod for a given latitude and day of year. Not accurate near polar regions.

    Args:
        latitude: Latitude in decimal degress. Where the northern hemisphere is 
            positive and the southern hemisphere is negative
        doy: np.ndarray of the days of year (0-365) where January 1st is 0 and 365

    Returns:
        Photoperiod, daylight hours, at the given latitude for the given days.
        The angle of the sum below the horizon.
        The zenithal distance of the sun in degrees.
        The mean anomaly of the sun.
        Lambda.
        Delta.
    """
    # Convert latitude to radians
    lat_radians = np.radians(lat)
    
    # Angle of the sun below the horizon. Civil twilight is -4.76 degrees.
    light_intensity = 2.206 * 10**-3
    sun_angle = -4.76 - 1.03 * np.log(light_intensity) # Eq. [5].

    # Zenithal distance of the sun in degrees
    sun_zenithal_dist = np.radians(90 + sun_angle) # Eq. [6]. Value at sunrise and sunset.
    
    # Mean anomaly of the sun. It is a convenient uniform measure of 
    # how far around its orbit a body has progressed since pericenter.
    sun_mean_anomaly = 0.9856*doy - 3.251 # Eq. [4].
    
    # Declination of sun in degrees
    sun_declenation = sun_mean_anomaly + 1.916*np.sin(np.radians(sun_mean_anomaly)) + 0.020*np.sin(np.radians(2*sun_mean_anomaly)) + 282.565 # Eq. [3]. Lambda
    C = np.sin(np.radians(23.44)) # 23.44 degrees is the orbital plane of Earth around the Sun
    delta = np.arcsin(C*np.sin(np.radians(sun_declenation))) # Eq. [2].

    # Calculate daylength in hours, defining sec(x) = 1/cos(x)
    day_length = 2/15 * np.degrees( np.arccos( np.cos(sun_zenithal_dist) * (1/np.cos(lat_radians)) * (1/np.cos(delta)) - np.tan(lat_radians) * np.tan(delta) ) ) # Eq. [1].

    return day_length, sun_angle, sun_zenithal_dist, sun_mean_anomaly, sun_declenation, np.degrees(delta)
    
def photoperiod_on_day(lat: np.ndarray, doys: np.ndarray) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
    """
    Computes photoperiod for a given near polar regions.

    Args:
        lat: Latitude in decimal degress. Where the northern hemisphere is positive and the southern hemisphere is negative.
        doys: The day of year (0-365) where January 1st is 0 and 365 to perform the calculation.

    Returns:
        Photoperiod, daylight hours, for the given latitudes on the given day.
        The angle of the sum below the horizon.
        The zenithal distance of the sun in degrees.
        The mean anomaly of the sun.
        The declination of the sun in degrees.
        Lambda from the equation.
        Delta from the equation.
    
    References:
        Keisling, T.C., 1982. Calculation of the Length of Day 1. Agronomy Journal, 74(4), pp.758-759.
    """
    # Convert latitude to radians and convert shapes
    lat = np.radians(np.asarray(lat)).reshape(-1, 1)  # shape (N, 1)
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
    P = 2/15 * np.degrees( np.arccos( np.cos(alpha) * (1/np.cos(lat)) * (1/np.cos(delta)) - np.tan(lat) * np.tan(delta) ) ) # Eq. [1].

    return P, B, alpha, M, Lambda, np.degrees(delta)


# Soil Water Flow Functions
def hydraulic_conductivity(
        sat_hydraulic_conductivity: float,
        air_entry_water_potential: float,
        volumetric_water_content: float,
        sat_water_content: float,
        b: float,
    ) -> float:
    """
    Estimates the hydraulic conductivity of soil.
    
    Args:
        sat_hydraulic_conductivity: The saturated conductivity of the soil  (kg * s / m^3).
        air_entry_water_potential: The air entry water potential of the soil  (J / kg).
        volumetric_water_content: Is the volumetric water content of the soil (m^3 / m^3).
        sat_water_content: Is the saturation water content of the soil (m^3 / m^3).
        b: Is the exponent of moisture release.

    Returns:
        The calculated hydraulic conductivity of the soil given the parameters ().
    """
    wetting_front_water_potential = air_entry_water_potential* (volumetric_water_content / sat_water_content) ** (-b)

    hydraulic_conductivity = sat_hydraulic_conductivity * (air_entry_water_potential / wetting_front_water_potential) ** (2 + 3/b)

    return hydraulic_conductivity

def cummulative_water_infiltration(
        water_vol_fraction: float,
        avg_sat_hydraulic_conductivity: float,
        infultration_water_potential: float,
        wetting_front_water_potential: float,
        max_time: int,
        interpolations: int=100
    ) -> np.array:
    """
    Calculates the cummulative vertical water infultration.
    
    Calculates the cummulative vertical water infultration for the given soil parameters.
    Calculations are made for equally spaced times from [0 to max_time].
    
    Args:
        water_vol_fraction: The Volume Fraction of water.
        avg_sat_hydraulic_conductivity: The average hydraulic conductivity of the wet soil (kg * s /m^3).
        infultration_water_potential: The infultration boudary's water potential (J / kg).
        wetting_front_water_potential: The wetting front's water potential (J / kg).
        max_time: The length of time to calcute the cummulative water infultration (s).
        interpolations: The number of interpolated times to calculate.
  
    Returns:
        A numpy array of length 'interpolations' representing the cummulative vertical water infultration over equal time segments.
        Where the first value represents the infultration after max_time/interpolations time, and the last vaule represents the infultration after max_time.
    """
    # Define Constants
    WATER_DENSITY = 1000 # Units: Kg / m^3
    LITTLE_G = 9.80665 # Units: m / s^2

    if water_vol_fraction < 0 or water_vol_fraction > 1.0:
        raise ValueError(f"The volume Fraction of water must be between [0,1], but {water_vol_fraction} was provided")

    # Calculate water infultration at each of the time
    times = np.linspace(max_time/interpolations, max_time, interpolations)
    water_inful = 2 * WATER_DENSITY * water_vol_fraction * avg_sat_hydraulic_conductivity * (infultration_water_potential - wetting_front_water_potential) * times
    water_inful =  water_inful ** 0.5 + LITTLE_G * avg_sat_hydraulic_conductivity * times

    return water_inful



