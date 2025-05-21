import agroecometrics as AEM
import pandas as pd

labels = AEM.settings.get_labels()

def air_temp_tests(df: pd.DataFrame, file_name: str):
    # Air Temperature models
    air_temp_pred = AEM.bio_sci_equations.model_air_temp(df2024)
    AEM.visualizations.plot_air_temp(df, air_temp_pred, file_name)

def evapo_transpiation_tests(df: pd.DataFrame, file_name: str):
    global labels

    # Set Latitude and Altitude for Evapo Models
    latitude = 34.0 # Degrees
    altitude = 350.0 # m

    # Get data from data file
    tmin = df[labels['tmin']]
    tmax = df[labels['tmax']]
    hmin = df[labels['hmin']]
    hmax = df[labels['hmax']]
    wind_avg = df[labels['max_gust']]*0.5*.44704
    doy = df['DOY']
    pmin = df[labels['pmin']]*100
    pmax = df[labels['pmax']]*100

    # Calculate EVapo model Data
    evapo_models = [
    AEM.bio_sci_equations.dalton(tmin, tmax, hmin, hmax, wind_avg),
    AEM.bio_sci_equations.penman(tmin, tmax, hmin, hmax, wind_avg),
    AEM.bio_sci_equations.romanenko(tmin, tmax, hmin, hmax),
    AEM.bio_sci_equations.jensen_haise(tmin, tmax, doy, latitude),
    AEM.bio_sci_equations.hargreaves(tmin, tmax, doy, latitude),
    AEM.bio_sci_equations.penman_monteith(tmin, tmax, hmin, hmax, wind_avg, pmin, pmax, doy, latitude, altitude)
    ]

    # Labels for the models
    evapo_model_labels = ['Dalton', 'Penman', 'Romanenko', 'Jensen-Haise', 'Hargreaves', 'Penman_Monteith']

    # Plot Model Data
    AEM.visualizations.plot_evapo_data(df, file_name, evapo_models, evapo_model_labels)

def runoff_tests(df: pd.DataFrame, file_name: str):
    AEM.bio_sci_equations.rainfall_runoff_to_df(df)
    AEM.visualizations.plot_rainfall(df, file_name)

def soil_temp_tests(file_names: list[str]):
    T_depth = AEM.bio_sci_equations.model_soil_temp_at_depth(10)
    AEM.visualizations.plot_yearly_soil_temp(T_depth, file_names[0])

    T_day, depths = AEM.bio_sci_equations.model_day_soil_temp(10, 500)
    AEM.visualizations.plot_day_soil_temp(T_day, depths, file_names[1])

    doy_grid, z_grid, t_grid = AEM.bio_sci_equations.model_soil_temp_3d(500)
    AEM.visualizations.plot_3d_soil_temp(doy_grid, z_grid, t_grid, file_names[2])


if __name__ == '__main__':
    # Get Data
    data_file = "Tests/Data/Marshfield_All_Data.csv"
    df = AEM.bio_sci_equations.load_data(data_file)
    df2023 = AEM.bio_sci_equations.load_data(data_file, start_date="2023-01-01", end_date="2023-12-31")
    df2024 = AEM.bio_sci_equations.load_data(data_file, start_date="2024-01-01", end_date="2024-12-31")
    df2025 = AEM.bio_sci_equations.load_data(data_file, start_date="2025-01-01", end_date="2025-12-31")

    # Save Folder
    folder = "Tests/Images/"

    air_temp_tests(df2024, folder+"air_temp_plot.png")
    evapo_transpiation_tests(df2024, folder+"Evapo_All.png")
    runoff_tests(df2024, folder+"runoff_plot.png")
    soil_temp_tests([folder+"soil_temp.png", folder+"day150_temp.png", folder+"soil_temp_3d.png"])


