import agroecometrics as AEM
import pandas as pd

def air_temp_tests(df: pd.DataFrame):
    # Air Temperature models
    air_temp_pred = AEM.bio_sci_equations.model_air_temp(df2024)
    AEM.visualizations.plot_air_temp(df, 'Tests/Images/air_temp_plot.png', air_temp_pred)

def evapo_transpiation_tests(df: pd.DataFrame):
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
    AEM.visualizations.plot_evapo_data(df, 'Tests/Images/Evapo_All.png', evapo_models, evapo_model_labels)

def runoff_tests(df: pd.DataFrame):
    runoff = AEM.bio_sci_equations.model_rainfall(df)
    AEM.visualizations.plot_rainfall(runoff, 'Tests/Images/runoff_plot.png')

def soil_temp_tests():
    T_depth = AEM.bio_sci_equations.model_soil_temp_at_depth(10)
    AEM.visualizations.plot_yearly_soil_temp(T_depth, "Tests/Images/soil_temp.png")

    T_day, depths = AEM.bio_sci_equations.model_day_soil_temp(10, 500, Nz=1000)
    AEM.visualizations.plot_day_temp(T_day, depths, "Tests/Images/day150_temp.png")

if __name__ == '__main__':
    # Get Data
    data_file = "Tests/Data/Marshfield_All_Data.csv"
    df = AEM.bio_sci_equations.load_data(data_file)
    df2023 = AEM.bio_sci_equations.load_data(data_file, start_date="2023-01-01", end_date="2023-12-31")
    df2024 = AEM.bio_sci_equations.load_data(data_file, start_date="2024-01-01", end_date="2024-12-31")
    df2025 = AEM.bio_sci_equations.load_data(data_file, start_date="2025-01-01", end_date="2025-12-31")

    labels = AEM.settings.get_labels()

    air_temp_tests(df2024)
    evapo_transpiation_tests(df2024)
    runoff_tests(df2024)
    soil_temp_tests()


