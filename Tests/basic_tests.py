import agroecometrics as AEM
import pandas as pd

def air_temp_tests(df: pd.DataFrame, file_name: str):
    # Air Temperature models
    air_temp_pred = AEM.bio_sci_equations.model_air_temp(df2024)
    AEM.visualizations.plot_air_temp(df, air_temp_pred, file_name)

def evapo_transpiation_tests(df: pd.DataFrame, file_name: str):
    # Set Latitude and Altitude for Evapo Models
    latitude = 34
    altitude = 350 # m

    # Calculate EVapo model Data
    evapo_models = [
    AEM.bio_sci_equations.dalton(df[labels['tmin']], df[labels['tmax']], df[labels['hmin']], df[labels['hmax']], df[labels['w2avg']]),
    AEM.bio_sci_equations.penman(df[labels['tmin']], df[labels['tmax']], df[labels['hmin']], df[labels['hmax']], df[labels['w2avg']]),
    AEM.bio_sci_equations.romanenko(df[labels['tmin']], df[labels['tmax']], df[labels['hmin']], df[labels['hmax']]),
    AEM.bio_sci_equations.jensen_haise(df[labels['tmin']], df[labels['tmax']], df['DOY'], latitude),
    AEM.bio_sci_equations.hargreaves(df[labels['tmin']], df[labels['tmin']], df[labels['tmin']], latitude),
    #TODO: Need to add solar radiation calculations to get this model to work
        # Add this model label back to the list of labels
    #AEM.bio_sci_equat.penman_monteith(df[labels['tmin']], df[labels['tmax']], df[labels['hmin']], df[labels['hmax']], df['solar_rad'], df[labels['w2avg']], df['DOY'], latitude, altitude)
    ]

    # Labels for the models
    evapo_model_labels = ['Dalton', 'Penman', 'Romanenko', 'Jensen-Haise', 'Hargreaves']

    # Plot Model Data
    AEM.visualizations.plot_evapo_data(df, file_name, evapo_models, evapo_model_labels)

def runoff_tests(df: pd.DataFrame, file_name: str):
    runoff = AEM.bio_sci_equations.model_rainfall(df)
    AEM.visualizations.plot_rainfall(runoff, file_name)

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

    labels = AEM.settings.get_labels()

    air_temp_tests(df2024, folder+"air_temp_plot.png")
    evapo_transpiation_tests(df2024, folder+"Evapo_All.png")
    runoff_tests(df2024, folder+"runoff_plot.png")
    soil_temp_tests([folder+"soil_temp.png", folder+"day150_temp.png", folder+"soil_temp_3d.png"])


