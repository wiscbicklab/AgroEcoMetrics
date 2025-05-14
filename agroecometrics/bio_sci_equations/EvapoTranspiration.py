# Import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from agroecometrics import settings
import _util 

labels = None

def load_data(file, start_date=None, end_date=None):
    '''
    Loads a data file and returns a filtered DataFrame.

    file: A string containing the path to your data
    start_date: Optional string in 'YYYY-MM-DD' format to filter data from this date onward
    end_date: Optional string in 'YYYY-MM-DD' format to filter data up to this date
    return: A pandas DataFrame
    '''
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.replace("'", "")
    
    global labels
    labels = settings.get_labels()

    df[labels['date']] = pd.to_datetime(df[labels['date']])

    if start_date:
        df = df[df[labels['date']] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df[labels['date']] <= pd.to_datetime(end_date)]

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


# Models
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

def plot_evapo_data(df, file_name, model_data, model_labels):

    # Check Argument Correctness
    _util.check_filename(file_name)
    if len(model_data) != len(model_labels):
        raise ValueError("You must provide the same number of model labels and model data")


    # Loop through and plot data from different models
    for i in range(model_data):
        data = model_data[i]
        data_label = model_labels[i]
        plt.plot(df[settings['date']], data, label=data_label)
    
    # Adds plot label
    plt.ylabel('Evapotranspiration (mm/day)')
    plt.legend()
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    
    return file_name

 
 #TODO: Move to tests file
if __name__ == '__main__':
    # Load Data
    df = load_data("private/Marshfield_All_Data.csv", '2024-01-01', '2025-01-01')
    # Set Latitude and Altitude for models which require it
    latitude = 34
    altitude = 350 # m

    # Calculate Model Data
    models = [
    dalton(df[labels['tmin']], df[labels['tmax']], df[labels['hmin']], df[labels['hmax']], df[labels['w2avg']]),
    penman(df[labels['tmin']], df[labels['tmax']], df[labels['hmin']], df[labels['hmax']], df[labels['w2avg']]),
    romanenko(df[labels['tmin']], df[labels['tmax']], df[labels['hmin']], df[labels['hmax']]),
    jensen_haise(df[labels['tmin']], df[labels['tmax']], df['DOY'], latitude),
    hargreaves(df[labels['tmin']], df[labels['tmin']], df[labels['tmin']], latitude),
    penman_monteith(df[labels['tmin']], df[labels['tmax']], df[labels['hmin']], df[labels['hmax']], df['ATOT'], df[labels['w2avg']], df['DOY'], latitude, altitude)
    ]

    # Labels for the models
    model_labels = ['Dalton', 'Penman', 'Romanenko', 'Jensen-Haise', 'Hargreaves', 'Penman-Monteith']

    # Plot Model Data
    plot_evapo_data(df, 'Evapo_All.png', models, model_labels)

