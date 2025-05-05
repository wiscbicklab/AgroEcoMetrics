# Import modules
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from agroecometrics import settings
import os
import _util

# Constants
avg_temp = 25 # Annual average temperature at the soil surface
thermal_amp = 10    # Annual thermal amplitude at the soil surface
thermal_dif = 0.203    # Thermal diffusivity obtained from KD2 Pro instrument [mm^2/s]
thermal_dif = thermal_dif / 100 * 86400 # convert to cm^2/day
OMEGA = 2*np.pi/365
timelag = 15   # Time lag in days from January 1
phase_const = np.pi/2 + OMEGA*timelag # Phase constant
damp_depth = (2*thermal_dif/OMEGA)**(1/2) # Damping depth 

# Define model as lambda function
model = lambda doy,z: avg_temp + thermal_amp * np.exp(-z/damp_depth) * np.sin(OMEGA*doy - z/damp_depth - phase_const)

# Data labels
labels = settings.get_labels()

def get_model_const():
    return avg_temp, thermal_amp, thermal_dif, timelag, phase_const, damp_depth

def set_model_const(T_avg=None, A0=None, D=None, t_0=None):
    """
    Update global thermal model constants.
    
    Parameters:
    - T_avg: new average surface temperature (°C)
    - A0: new thermal amplitude (°C)
    - D: new thermal diffusivity (mm²/s)
    - t_0: new time lag (days from Jan 1)
    """
    global avg_temp, thermal_amp, thermal_dif, timelag, phase_const, damp_depth

    if T_avg is not None:
        avg_temp = T_avg
    if A0 is not None:
        thermal_amp = A0
    if D is not None:
        thermal_dif = D / 100 * 86400  # convert to cm²/day
    if t_0 is not None:
        timelag = t_0

    # Always update derived constants
    phase_const = np.pi / 2 + OMEGA * timelag
    damp_depth = (2 * thermal_dif / OMEGA) ** 0.5

def plot_soil_temp(depth, file_name):
    '''
    Creates a plot of the soil temperature at the specified depth over the data loaded

    depth: the depth to model the soil temperature at
    '''
    # Validate input parameters
    _util.check_png_filename(file_name)
    
    # Use the model do calulate temperature data
    doy = np.arange(1,366)
    T_soil = model(doy,depth)

    # Create the plot
    plt.figure()
    plt.plot(doy,T_soil)
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    return file_name

def plot_day_temp(doy, max_depth, file_name):
    Nz = 100  # Number of interpolations

    z = np.linspace(0, max_depth, Nz)
    T = model(doy, z)

    # Create Plot
    plt.figure()
    plt.plot(T, -z)
    plt.ylabel("Air temperature (Celsius)")
    plt.savefig(file_name, dpi=300, bbox_inches='tight')

#TODO Move to tests section
if __name__ == '__main__':
    print(get_model_const)
    plot_soil_temp(10, "private/soil_temp.png")
    plot_day_temp(150, 500, "private/day150_temp.png")