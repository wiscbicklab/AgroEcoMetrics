from scipy.stats import circmean

import numpy as np
import pandas as pd

from agroecometrics import settings


# Gets the acutal labels of columns based on the user settings
labels = settings.get_labels()


def load_data(file, start_date=None, end_date=None):
    '''
    Loads a data file and returns a filtered DataFrame.

    file: A string containing the path to your data
    start_date: Optional string in 'YYYY-MM-DD' format to filter data from this date onward
    end_date: Optional string in 'YYYY-MM-DD' format to filter data up to this date
    return: A pandas DataFrame
    '''
    # Read the data from the given csv file
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.replace("'", "")

    # Ensures that the data colomn is datetime format
    df[labels['date']] = pd.to_datetime(df[labels['date']], format='%m/%d/%Y %I:%M %p')

    # Uses start and end date to filter the data
    if start_date:
        df = df[df[labels['date']] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df[labels['date']] <= pd.to_datetime(end_date)]

    # Adds a Year and DOY column to the df based on the date column
    df['DOY'] = df[labels['date']].dt.dayofyear
    df['YEAR'] = df[labels['date']].dt.year

    return df.reset_index(drop=True)


# Helper Functions
def compute_esat(T):
    """Function that computes saturation vapor pressure based Tetens formula"""
    e_sat = 0.6108 * np.exp(17.27 * T/(T+237.3)) 
    return e_sat

def compute_Ra(doy, latitude):
    """Function that computes extra-terrestrial solar radiation"""
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy/365) # Inverse relative distance Earth-Sun
    phi = np.pi / 180 * latitude # Latitude in radians
    d = 0.409 * np.sin((2 * np.pi * doy/365) - 1.39) # Solar delcination
    omega = np.arccos(-np.tan(phi) * np.tan(d)) # Sunset hour angle
    Gsc = 0.0820 # Solar constant
    Ra = 24 * 60 / np.pi * Gsc * dr * (omega * np.sin(phi) * np.sin(d) + np.cos(phi) * np.cos(d) * np.sin(omega))
    return Ra


# Air Temperature model
def model_air_temp(df):
    '''
    Uses the air temperature data to find parameter estimates for the air temperature model
        and creates the temperature predictions given the model

    df: DataFrame with temperature data
    return: A numpy array of predicted daily temperatures
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

    return T_pred

# Soil Temperature models
def model_soil_temp_at_depth(depth, avg_temp=25, thermal_amp=10, thermal_dif=0.203, time_lag=15):
    """
    Models soil temperature over a year at the given depth

    Args:
        depth: The depth to model the soil temperature at, in meters
        avg_temp: The annual average surface temperature, in Celsius
        thermal_amp: The annual thermal amplitude of the soil surface, in Celsius
        thermal_dif: The Thermal diffusivity obtained from KD2 Pro instrument [mm^2/s]
        time_lag: The time lag in days from January 1st
    Return:
        Predicted soil temperature at the given depth for each day of the year
    """
    # Set Constants
    OMEGA = 2*np.pi/365

    thermal_dif = thermal_dif / 100 * 86400 # convert to cm^2/day
    phase_const = np.pi/2 + OMEGA*time_lag # Phase constant
    damp_depth = (2*thermal_dif/OMEGA)**(1/2) # Damping depth 

    doy = np.arange(1,366)
    T_soil = np.sin(OMEGA*doy - depth/damp_depth - phase_const)
    T_soil = avg_temp + thermal_amp * np.exp(-depth/damp_depth) * T_soil
    return  T_soil

def model_day_soil_temp(doy, max_depth, Nz=100, avg_temp=25, thermal_amp=10, thermal_dif=0.203, timelag=15):
    """
    Models soil temperature on a particular day of the year
    
    Args:
        doy: Is the day to model soil temperature at, given as days since January first
        max_depth: The maximum depth to model the soil temperature at, in centimeters
        Nz: The number of interpolated depths to caluculate soil temperature at
        avg_temp: The annual average surface temperature in, Celsius
        thermal_amp: The annual thermal amplitude of the soil surface, in Celsius
        thermal_dif: The Thermal diffusivity obtained from KD2 Pro instrument [mm^2/s]
        time_lag: The time lag in days since January 1st
    Return:
        Predicted soil temperature at Nz number of depths from max_depth/Nz to max_depth
        Depths that the soil temperature was modeled at 
    """
    # Set Constants
    OMEGA = 2*np.pi/365

    thermal_dif = thermal_dif / 100 * 86400 # convert to cm^2/day
    phase_const = np.pi/2 + OMEGA*timelag # Phase constant
    damp_depth = (2*thermal_dif/OMEGA)**(1/2) # Damping depth 
    depths = np.linspace(0, max_depth, Nz) # Interpolation depths

    T_soil = avg_temp + thermal_amp * np.exp(-depths/damp_depth) * np.sin(OMEGA*doy - depths/damp_depth - phase_const)
    return T_soil, depths


# EvapoTranspiration Models
def dalton(T_min,T_max,RH_min,RH_max,wind_speed):
    """Potential evaporation model proposed by Dalton in 1802"""
    e_sat_min = compute_esat(T_min)
    e_sat_max = compute_esat(T_max)
    e_sat = (e_sat_min + e_sat_max)/2
    e_atm = (e_sat_min*(RH_max/100) + e_sat_max*(RH_min/100))/ 2
    PE = (3.648 + 0.7223*wind_speed)*(e_sat - e_atm)
    return PE

def penman(T_min,T_max,RH_min,RH_max,wind_speed):
    """Potential evapotranspiration model proposed by Penman in 1948"""
    e_sat_min = compute_esat(T_min)
    e_sat_max = compute_esat(T_max)
    e_sat = (e_sat_min + e_sat_max)/2
    e_atm = (e_sat_min*(RH_max/100) + e_sat_max*(RH_min/100))/ 2
    PET = (2.625 + 0.000479/wind_speed)*(e_sat - e_atm)
    return PET

def romanenko(T_min,T_max,RH_min,RH_max):
    """Potential evaporation model proposed by Romanenko in 1961"""
    T_avg = (T_min + T_max)/2
    RH_avg = (RH_min + RH_max)/2
    PET = 0.00006*(25 + T_avg)**2*(100 - RH_avg)
    return PET

def jensen_haise(T_min,T_max,doy,latitude):
    """Potential evapotranspiration model proposed by Jensen in 1963"""
    Ra = compute_Ra(doy, latitude)
    T_avg = (T_min + T_max)/2
    PET = 0.0102 * (T_avg+3) * Ra
    return PET

def hargreaves(T_min,T_max,doy,latitude):
    """Potential evapotranspiration model proposed by Hargreaves in 1982"""
    Ra = compute_Ra(doy, latitude)
    T_avg = (T_min + T_max)/2
    PET = 0.0023 * Ra * (T_avg + 17.8) * (T_max - T_min)**0.5
    return PET

def penman_monteith(T_min,T_max,RH_min,RH_max,solar_rad,wind_speed,doy,latitude,altitude):
    T_avg = (T_min + T_max)/2
    atm_pressure = 101.3 * ((293 - 0.0065 * altitude) / 293)**5.26 # Can be also obtained from weather station
    Cp = 0.001013; # Approx. 0.001013 for average atmospheric conditions
    epsilon =  0.622
    Lambda = 2.45
    gamma = (Cp * atm_pressure) / (epsilon * Lambda) # Approx. 0.000665

    ##### Wind speed
    wind_height = 1.5 # Most common height in meters
    wind_speed_2m = wind_speed * (4.87 / np.log((67.8 * wind_height) - 5.42))  # Eq. 47, FAO-56 wind height in [m]

    ##### Air humidity and vapor pressure
    delta = 4098 * (0.6108 * np.exp(17.27 * T_avg / (T_avg  + 237.3))) / (T_avg  + 237.3)**2
    e_temp_max = 0.6108 * np.exp(17.27 * T_max / (T_max + 237.3)) # Eq. 11, //FAO-56
    e_temp_min = 0.6108 * np.exp(17.27 * T_min / (T_min + 237.3))
    e_saturation = (e_temp_max + e_temp_min) / 2
    e_actual = (e_temp_min * (RH_max / 100) + e_temp_max * (RH_min / 100)) / 2

    ##### Solar radiation
    
    # Extra-terrestrial solar radiation
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy/365)  # Eq. 23, FAO-56
    phi = np.pi / 180 * latitude # Eq. 22, FAO-56
    d = 0.409 * np.sin((2 * np.pi * doy/365) - 1.39)
    omega = np.arccos(-np.tan(phi) * np.tan(d))
    Gsc = 0.0820 # Approx. 0.0820
    Ra = 24 * 60 / np.pi * Gsc * dr * (omega * np.sin(phi) * np.sin(d) + np.cos(phi) * np.cos(d) * np.sin(omega))

    # Clear Sky Radiation: Rso (MJ/m2/day)
    Rso =  (0.75 + (2 * 10**-5) * altitude) * Ra  # Eq. 37, FAO-56

    # Rs/Rso = relative shortwave radiation (limited to <= 1.0)
    alpha = 0.23 # 0.23 for hypothetical grass reference crop
    Rns = (1 - alpha) * solar_rad # Eq. 38, FAO-56
    sigma  = 4.903 * 10**-9
    maxTempK = T_max + 273.16
    minTempK = T_min + 273.16
    Rnl =  sigma * (maxTempK**4 + minTempK**4) / 2 * (0.34 - 0.14 * np.sqrt(e_actual)) * (1.35 * (solar_rad / Rso) - 0.35) # Eq. 39, FAO-56
    Rn = Rns - Rnl # Eq. 40, FAO-56

    # Soil heat flux density
    soil_heat_flux = 0 # Eq. 42, FAO-56 G = 0 for daily time steps  [MJ/m2/day]

    # ETo calculation
    PET = (0.408 * delta * (solar_rad - soil_heat_flux) + gamma * (900 / (T_avg  + 273)) * wind_speed_2m * (e_saturation - e_actual)) / (delta + gamma * (1 + 0.34 * wind_speed_2m))
    return np.round(PET,2)


# Rain/Runoff Models
def curve_number(P, CN=75):
    '''
    Curve number method for runoff estimation.
    P is precipitation in millimeters
    CN is the curve number
    '''
    runoff = np.zeros_like(P)
    S02 = 1000 / CN - 10
    S005 = 1.33 * S02**1.15
    Lambda = 0.05
    Ia = S005 * Lambda
    idx = P > Ia
    runoff[idx] = (P[idx] - Ia)**2 / (P[idx] - Ia + S005)
    return runoff

def model_rainfall(df, cn=80):
    '''
    Computes cumulative rainfall and runoff, and adds the data to the DataFrame
        with the labels RAIN_SUM, RUNOFF, and RUNOFF_SUM

    df: DataFrame with rainfall data
    cn: curve number for runoff model
    '''
    df['RAIN_SUM'] = df[labels['rain']].cumsum()
    df['RUNOFF'] = curve_number(df[labels['rain']] / 25.4, CN=cn) * 25.4
    df['RUNOFF_SUM'] = df['RUNOFF'].cumsum()
    return df




# Fletcher's Functions
def match_weather(weather_datetime_col, data_datetime_col):
    vals = np.searchsorted(weather_datetime_col, data_datetime_col, sorter=None)
    match_times = [None] * len(vals)
    match_indices = [None] * len(vals)
    diffs = [None] * len(vals)
 
    for i in range(len(vals)):
        vals[i] = min(vals[i], len(weather_datetime_col)-1)
        if vals[i] > 0 and abs(data_datetime_col[i]-weather_datetime_col[vals[i]]) > abs(data_datetime_col[i]-weather_datetime_col[vals[i]-1]):
            vals[i] = vals[i]-1
        match_times[i] = weather_datetime_col[vals[i]]
        match_indices[i] = vals[i]
        diffs[i] = abs(data_datetime_col[i]-weather_datetime_col[vals[i]])
    return (data_datetime_col, match_times, match_indices, diffs)
 
def get_cols_from_weather_for_matched_indices(weather_data, weather_cols, matched_indices):
    cols = [None]*len(weather_cols)
    for col in range(len(weather_cols)):
        cols[col] = [None]*len(matched_indices)
        for i in range(len(matched_indices)):
            cols[col][i] = weather_data[weather_cols[col]][matched_indices[i]]
 
    return cols



