from typing import Optional, Tuple
from scipy.optimize import curve_fit

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
        temp: Numpy array of temperatures. (°C)
    
    Returns:
        Numpy array of computed saturation vapor pressure. (kPa)
    """
    return  0.6108 * np.exp(17.27 * temp/(temp+237.3)) 

def __compute_solar_radiation(doys: np.ndarray, lat: float) -> np.ndarray:
    """
    Computes extra-terrestrial solar radiation using the FAO Penman-Monteith method.
    
    Args:
        doys: Numpy array containing the day of year, days since January 1st. (January 1st = 0 and December 31st = 364)
        lat: The latitude to compute solar radiations at. (Degrees, North is positive)
    Returns:
        A Numpy array of extra-terrestrial solar radiation on the given days. (MJ/(m² * day)).
    """
    lat = np.pi / 180 * lat     # Convert latitude to radians

    dr = 1 + 0.033 * np.cos(2 * np.pi * doys/365) # Inverse relative distance Earth-Sun
    sol_declination = 0.409 * np.sin((2 * np.pi * doys/365) - 1.39)

    sunset_hour_angle = np.arccos(-np.tan(lat) * np.tan(sol_declination))

    sol_rad = 24 * 60 / np.pi * 0.0820 * dr
    sol_rad = sol_rad * (sunset_hour_angle * np.sin(lat) * np.sin(sol_declination) + np.cos(lat) * np.cos(sol_declination) * np.sin(sunset_hour_angle))
    return sol_rad


# Air Temperature model
def model_air_temp(temps: np.ndarray, date_times: np.ndarray) -> np.ndarray:
    """
    Generates a daily air temperature estimate by creating a model from actual air temperatures.

    Creates a sinusoidal model of air temperature using best fit on the provided data.
    Then creates daily air temperature predictions over the entire range of dates provided.

    Args:
        temps: A numpy array of average daily air temperatures. (°C)
        date_times: A numpy array of date time objects correspoinding to the air temperatures.
    
    Returns: 
        A numpy array of predicted daily temperatures from the first to the last day provided, inclusive.
    """
    # Estimate sinusoidal parameters
    thermal_amp = (np.max(temps) - np.min(temps)) / 2
    avg_temp = np.mean(temps)
    phase_shift = 15
    param_estimate = [thermal_amp, avg_temp, phase_shift]

    # Convert all dates to day of year
    date_times = pd.to_datetime(date_times)
    doys = date_times.dt.dayofyear-1

    def model(doy, amplitude, phase_shift, offset):
        return amplitude * np.sin((2 * np.pi *doy / 365.25) + phase_shift) + offset
    
    params, __ = curve_fit(model, doys, temps, p0=param_estimate)

    # Generate sinusoidal temperature predictions
    pred_temp = model(doys, params[0], params[1], params[2])

    return np.asarray(pred_temp)


# Soil Temperature models
def yearly_soil_temp(
        depth: int,
        surface_temp: int=25,
        thermal_amp: int = 10,
        thermal_dif: float = 0.203,
        time_lag: int = 15
    ) -> np.ndarray:
    """
    Models average soil temperature at a given depth for each day of a year

    Args:
        depth: The depth to model the soil temperature. (m)
        surface_temp: The annual average surface temperature. (°C)
        thermal_amp: The annual thermal amplitude of the soil surface. (°C)
        thermal_dif: The thermal diffusivity of the soil to estimate. (mm^2 / s)
        time_lag: The difference between January 1st and the coldest day of the year. (Days)

    Return:
        A numpy array of the daily soil temperature predictions. (°C).
    """
    # Set Constants
    PHASE_FREQ = 2 * np.pi / 365
    PHASE_SHIFT = time_lag * PHASE_FREQ - np.pi/2

    thermal_dif = thermal_dif / 100    # Unit Conversion

    # Estimate the temperatures for each day of the year
    doy = np.arange(0,365)
    damp_depth = (2*thermal_dif/PHASE_FREQ)**(1/2)
    soil_temp = np.sin(PHASE_FREQ * doy - depth / damp_depth - PHASE_SHIFT)
    soil_temp *= thermal_amp * np.exp(-depth / damp_depth)
    soil_temp += surface_temp
    
    return  np.asarray(soil_temp)

def daily_soil_temp(
        doy: int,
        max_depth: int,
        num_depths: int = 100,
        surface_temp: int = 25,
        thermal_amp: int = 10,
        thermal_dif: int = 0.203,
        timelag: int = 15
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Models soil temperature at a set of depths on a particular day of the year

    Estimates the soil temperature at the number of depths specified. 
    The first depth is max_depth/num_depths and the last depth is max_depth.
    
    Args:
        doy: The day of year, days since January 1st. (January 1st = 0 and December 31st = 364)
        max_depth: The maximum depth to model the soil temperature. (m)
        num_depths: The number of depths to caluculate soil temperature at. (None)
        surface_temp: The annual average surface temperature. (°C)
        thermal_amp: The annual thermal amplitude of the soil surface. (°C)
        thermal_dif: The thermal diffusivity of the soil to estimate. (mm^2 / s)
        time_lag: The time lag in days (0-365) where January 1st is 0 and 365.

    Returns:
        A tuple containing two numpy arrays.
            - The first contains the soil temperature predictions. (°C)
            - The second contains the depth of each prediction. (cm)
    """
    # Set Constants
    PHASE_FREQ = 2 * np.pi / 365
    PHASE_SHIFT = timelag * PHASE_FREQ - np.pi/2

    thermal_dif = thermal_dif / 100 # Unit Conversion

    soil_depths = np.linspace(max_depth/num_depths, max_depth, num_depths) # Interpolate depths

    # Estimate the temperatures for each depth
    damp_depth = (2 * thermal_dif / PHASE_FREQ) ** 0.5
    soil_temps = np.sin(PHASE_FREQ * doy - soil_depths / damp_depth - PHASE_SHIFT)
    soil_temps *= thermal_amp * np.exp(-soil_depths / damp_depth) 
    soil_temps += surface_temp

    return (np.asarray(soil_temps), np.asarray(soil_depths))

def yearly_3d_soil_temp(
        max_depth: int,
        num_depths: int = 1000,
        surface_temp: int = 25,
        thermal_amp: int = 10,
        thermal_dif: float = 0.203,
        timelag: int = 15
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Models soil temperature over a full year (0-365) and across depth.

    Estimates the soil temperature at the number of depths specified for each day of the year. 
    The first depth is max_depth/num_depths and the last depth is max_depth.
    Creates a matrix of estimations for each day of the year at each depth.
        
    Args:
        max_depth: The maximum depth in centimeters to model the soil temperature. (m)
        num_depths: The number of interpolated depths to calculate soil temperature at.
        surface_temp: The annual average surface temperature. (°C)
        thermal_amp: The annual thermal amplitude of the soil surface. (°C)
        thermal_dif: The thermal diffusivity of the soil to model. (mm^2 / s)
        timelag: The time lag in days (0-365) where January 1st is 0 and 365
    
    Returns:
        A tuple of (doy_grid, z_grid, temp_grid), where each is a 2D NumPy array. 
            - doy_grid: varies across columns. (same DOY per column)
            - z_grid: varies across rows. (same depth per row) (m)
            - temp_grid: soil temperature at each depth and DOY. (°C)
    """
    doys = np.arange(0, 365)  # Days of year
    depths = np.linspace(max_depth / num_depths, max_depth, num_depths)  # Depths

    # Initialize temperature matrix [depth x day]
    temp_grid = np.zeros((num_depths, len(doys)))

    for j, doy in enumerate(doys):
        soil_temps, _ = daily_soil_temp(
            doy, max_depth, num_depths, surface_temp, thermal_amp, thermal_dif, timelag
        )
        temp_grid[:, j] = soil_temps

    # Create matching meshgrids
    doy_grid, z_grid = np.meshgrid(doys, depths)

    return doy_grid, z_grid, temp_grid

def model_soil_temp(
        air_temps: np.ndarray,
        depth: float,
        thermal_dif: int = 0.000001,
    ) -> np.ndarray:
    """
    Creates soil temperature predictions for each air temperature provided over the course of a day

    Creates a sinusodial model of air temperature using best fir on the provided data
    Then uses that model to estimate soil temperature at the given depth for each temperature given.
    Air temperature are assumed to represent a single day and be equally spaced out through the entire day.

    Args:
        air_temps: A numpy array of air temperatures at the soil surface. (°C)
        depth: The depth to model the soil temperature. (m)
        thermal_dif: The thermal diffusivity of the soil to model. (m^2 / s)

    Returns:
        A numpy array containing the predicted temperatures. (°C)
    """
    # Estimate sinusoidal parameters
    thermal_amp = (np.max(air_temps) - np.min(air_temps))
    avg_temp = np.mean(air_temps)
    phase_shift = 3600
    param_estimate = [thermal_amp, avg_temp, phase_shift]

    # Calcuate Times in seconds for each measurement
    times = np.linspace(0, DAY_SECONDS, len(air_temps))

    # Determine constants
    PHASE_FREQ = 2 * np.pi / DAY_SECONDS 

    def model(time, amplitude, phase_shift, avg_temp):
        return amplitude * np.sin((time * PHASE_FREQ) + phase_shift) + avg_temp
    
    # Estimate function parameters
    params, __ = curve_fit(model, times, air_temps, p0=param_estimate)
    damp_depth = np.sqrt(2 * thermal_dif / PHASE_FREQ)

    pred_air_temps = params[0] * np.sin((times * PHASE_FREQ) + params[1]) + params[2]
    
    # Generate soil temperature preditions    
    soil_temps = params[0] * (np.e**(-depth/damp_depth))
    soil_temps *= np.sin((times * PHASE_FREQ) - (depth / damp_depth) + params[1])
    soil_temps += params[2]

    return soil_temps

def model_3d_soil_temp(
        air_temps: np.ndarray,
        max_depth: float,
        num_depths: int = 1000,
        thermal_dif: float = 0.203
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Models soil temperature over a full day (in 5-minute intervals) across depth 
    using air temperature data to parameterize a sinusoidal model.

    Args:
        air_temps: Soil surface air temperatures for a given day. (°C)
        max_depth: Maximum depth to model the soil temperature. (m)
        num_depths: The number of interpolated depths to calculate soil temperature at.
        thermal_dif: The thermal diffusivity of the soil to model. (mm^2 / s)

    Returns:
        (time_grid, depth_grid, temp_grid), each a 2D NumPy array:
            - time_grid: varies across columns. (seconds in day)
            - depth_grid: varies across rows. (m)
            - temp_grid: soil temperature (°C) at each depth and time.
    """
    # Calculate depths to evaluate
    depths = np.linspace(max_depth / num_depths, max_depth, num_depths)

    # Time steps: 5-minute intervals over the full day
    times = np.linspace(0, DAY_SECONDS, len(air_temps))

    # Preallocate matrix: [depth x time]
    temp_grid = np.zeros((num_depths, len(air_temps)))

    # Generate predictions at each depth
    for i, depth in enumerate(depths):
        soil_temps = model_soil_temp(air_temps, depth, thermal_dif)
        temp_grid[i, :] = soil_temps

    # Build matching meshgrids
    time_grid, depth_grid = np.meshgrid(times, depths)

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
        temp_min: A numpy array of minimum daily temperatures. (°C)
        temp_max: A numpy array of maximum daily temperatures. (°C)
        RH_min:   A numpy array of minimum daily relative humidities. (range: 0.0-1.0)
        RH_max:   A numpy array of maximum daily relative humidities. (range: 0.0-1.0)
        wind_speed: A numpy array of average daily wind speeds. (m/s)

    Returns:
        Daily evapotranspiration clipped to a minimum of zero. (mm/day)

    Raises:
        ValueError: If all of the numpy arrays are not the same size
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
        temp_min: A numpy array of minimum daily temperatures. (°C)
        temp_max: A numpy array of maximum daily temperatures. (°C)
        RH_min:   A numpy array of minimum daily relative humidities. (range: 0.0-1.0)
        RH_max:   A numpy array of maximum daily relative humidities. (range: 0.0-1.0)
        wind_speed: A numpy array of average daily wind speeds. (m/s)

    Returns:
        Daily evapotranspiration clipped to a minimum of zero. (mm/day)

    Raises:
        ValueError: If all of the numpy arrays are not the same size
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
        temp_min: A numpy array of minimum daily temperatures. (°C)
        temp_max: A numpy array of maximum daily temperatures. (°C)
        RH_min:   A numpy array of minimum daily relative humidities. (range: 0.0-1.0)
        RH_max:   A numpy array of maximum daily relative humidities. (range: 0.0-1.0)

    Returns:
        Daily evapotranspiration clipped to a minimum of zero. (mm/day)

    Raises:
        ValueError: If all of the numpy arrays are not the same size
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
        doys: np.ndarray,
        lats: np.ndarray
    ) -> np.ndarray:
    """
    Computes evapotranspiration using the Jensen model
    
    Args:
        temp_min: A numpy array of minimum daily temperatures. (°C)
        temp_max: A numpy array of maximum daily temperatures. (°C)
        doys:     A numpy array containing the day of year, days since January 1st. (January 1st = 0 and December 31st = 364)
        lats:     A numpy array of latitudes. (Degrees, North is positive)

    Returns:
        Daily evapotranspiration clipped to a minimum of zero. (mm/day)

    Raises:
        ValueError: If all of the numpy arrays are not the same size
    """
    # Ensure parameter saftey
    if not isinstance(temp_min, float) and not (len(temp_min) == len(temp_max)):
        raise ValueError("All inputs must be the same length")
    
    # Model Calculations
    Ra = __compute_solar_radiation(doys, lats)
    T_avg = (temp_min + temp_max)/2
    PET = 0.0102 * (T_avg+3) * Ra

    # Ensures non-negative evapotranspiration
    PET = np.clip(PET, 0.0, None)

    return PET

def hargreaves(
        temp_min: np.ndarray,
        temp_max: np.ndarray,
        doys: np.ndarray,
        lats: float
    ) -> np.ndarray:
    """
    Computes evapotranspiration using the Hargreaves model
    
    Args:
        temp_min: A numpy array of minimum daily temperatures. (°C)
        temp_max: A numpy array of maximum daily temperatures. (°C)
        doys:     A numpy array containing the day of year, days since January 1st. (January 1st = 0 and December 31st = 364)
        lats:     A numpy array of latitudes. (Degrees, North is positive)
        
    Returns:
        Daily evapotranspiration clipped to a minimum of zero. (mm/day)

    Raises:
        ValueError: If all of the numpy arrays are not the same size
    """
    # Ensure parameter saftey
    if not isinstance(temp_min, float) and not (len(temp_min) == len(temp_max)):
        raise ValueError("All inputs must be the same length")
    
    # Model Calulations
    Ra = __compute_solar_radiation(doys, lats)
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
        doys: np.ndarray,
        lat: float,
        alt: float,
        solar_rad: Optional[np.ndarray]=None,
        wind_height: float = 1.5,
    ) -> np.ndarray:
    """
    Computed evapotranspiration using the penman-monteith model
    
    Args:
        temp_min:   A numpy array of minimum daily temperatures. (°C)
        temp_max:   A numpy array of maximum daily temperatures. (°C)
        RH_min:     A numpy array of minimum daily relative humidities. (range: 0.0-1.0)
        RH_max:     A numpy array of maximum daily relative humidities. (range: 0.0-1.0)
        p_min:      A numpy array of minimum daily atmospheric pressure. (Pa)
        p_max:      A numpy array of maximum daily atmospheric pressure. (Pa)
        wind_speed: A numpy array of average daily wind speeds. (m/s)
        doys:       A numpy array containing the day of year, days since January 1st. (January 1st = 0 and December 31st = 364)
        lat:        The latitude of the location. (Degrees, North is positive)
        alts:       The altitude of the location. (m)
        solar_rad   The soloar radiation occuring over the whole day. ((MJ/(m² * day))
        wind_height: Height of for wind speed measurments. (m)
        
    Returns:
        Daily evapotranspiration clipped to a minimum of zero. (mm/day)

    Raises:
        ValueError: If all of the numpy arrays are not the same size
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
        solar_rad = __compute_solar_radiation(doys, lat)
    
    temp_avg = (temp_min + temp_max)/2
    atm_pressure = (p_min+p_max)/2 # Can be also obtained from weather station
    Cp = 0.001013; # Approx. 0.001013 for average atmospheric conditions
    gamma = 0.000665* (Cp * atm_pressure)

    # Wind speed Adjustment
    wind_speed_at_height = 6.56 * wind_speed / (np.log(67.8 * wind_height) / np.log(np.e)) # Eq. 47, FAO-56 wind height in [m]

    # Calculates air humidity and vapor pressure
    delta = 4098 * (0.6108 * np.exp(17.27 * temp_avg / (temp_avg  + 237.3)))
    delta = delta / (temp_avg  + 237.3)**2
    e_temp_max = 0.6108 * np.exp(17.27 * temp_max / (temp_max + 237.3)) # Eq. 11, //FAO-56
    e_temp_min = 0.6108 * np.exp(17.27 * temp_min / (temp_min + 237.3))
    e_saturation = (e_temp_max + e_temp_min) / 2
    e_actual = (e_temp_min * (RH_max / 100) + e_temp_max * (RH_min / 100)) / 2

    # Clear Sky Radiation: Rso (MJ/m2/day)
    Rso =  (0.75 + (1 / 50000) * alt) * solar_rad  # Eq. 37, FAO-56

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
    """
    Uses Curve Number to estimate runoff from rainfall

    Args:
        precipitation: A numpy array of daily precipitation. (mm)
        curve_number: The curve number.

    Returns:
        The runoff estimation. (mm)
    """
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
    Model how many growing degree days have passed.

    Models Growing Degree days using a minimum base temperature, an optional optimal temperature,
    an optional maximum growing temperature, and the average temperature over the recorded durations.
    If only a base temperature is provided GDD start accumulating above the base temperature, and 
    accumulate faster the higher the temperature gets. If optimal and upper temperature are provided
    the accumulation starts above the base temperature, is greatest at the optimal temperature, and 
    then decreases linearly down to zero at the upper temperature.
        
    Args:
        temp_avg:  A numpy array of average temperatures. (°C)
        temp_base: The minimum temperature to accumulate GDD. (°C)
        temp_opt: The optimal temperature to accumulate GDD. (°C)
        temp_upper: The maximum temperature to accumulate GDD. (°C)
        time_duration: The length of time each temp_avg represents. (Days)
    
    Returns:
        A tuple containing two numpy arrays.
            - The first contains Growing Degree Days for each day.
            - The second contains the cummulative sum of Growing Degree Days for each day.
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


# Photoperiod Tools
def photoperiod_at_lat(doys: np.ndarray, lat: float) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
    """
    Computes photoperiod for a given latitude and day of year. Not accurate near polar regions.

    Args:
        doys: A numpy array containing the day of year, days since January 1st. (January 1st = 0 and December 31st = 364)
        lat:  The latitude to compute photoperiod at. (Degrees, North is positive)

    Returns:
        A tuple containing
            - Photoperiod, daylight hours, at the given latitude for the given days.
            - The angle of the sum below the horizon.
            - The zenithal distance of the sun in degrees.
            - The mean anomaly of the sun.
            - Lambda.
            - Delta.
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
    sun_mean_anomaly = 0.9856*doys - 3.251 # Eq. [4].
    
    # Declination of sun in degrees
    sun_declenation = sun_mean_anomaly + 1.916*np.sin(np.radians(sun_mean_anomaly)) + 0.020*np.sin(np.radians(2*sun_mean_anomaly)) + 282.565 # Eq. [3]. Lambda
    C = np.sin(np.radians(23.44)) # 23.44 degrees is the orbital plane of Earth around the Sun
    delta = np.arcsin(C*np.sin(np.radians(sun_declenation))) # Eq. [2].

    # Calculate daylength in hours, defining sec(x) = 1/cos(x)
    day_length = 2/15 * np.degrees( np.arccos( np.cos(sun_zenithal_dist) * (1/np.cos(lat_radians)) * (1/np.cos(delta)) - np.tan(lat_radians) * np.tan(delta) ) ) # Eq. [1].

    return day_length, sun_angle, sun_zenithal_dist, sun_mean_anomaly, sun_declenation, np.degrees(delta)
    
def photoperiod_on_day(doys: np.ndarray, lats: np.ndarray) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
    """
    Computes photoperiod for a given near polar regions.

    Args:
        doys: A numpy array containing the day of year, days since January 1st. (January 1st = 0 and December 31st = 364)
        lats: A numpy array of latitudes to compute photoperiod at. (Degrees, North is positive)

    Returns:
        A tuple containing
            Photoperiod, daylight hours, for the given latitudes on the given day.
            - The angle of the sum below the horizon.
            - The zenithal distance of the sun in degrees.
            - The mean anomaly of the sun.
            - The declination of the sun in degrees.
            - Lambda from the equation.
            - Delta from the equation.
    
    References:
        Keisling, T.C., 1982. Calculation of the Length of Day 1. Agronomy Journal, 74(4), pp.758-759.
    """
    # Convert latitude to radians and convert shapes
    lats = np.radians(np.asarray(lats)).reshape(-1, 1)  # shape (N, 1)
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
    P = 2/15 * np.degrees( np.arccos( np.cos(alpha) * (1/np.cos(lats)) * (1/np.cos(delta)) - np.tan(lats) * np.tan(delta) ) ) # Eq. [1].

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
        A numpy array of length 'interpolations' representing the cummulative vertical water infultration.
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


