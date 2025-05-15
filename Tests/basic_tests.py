import agroecometrics as AEM
import pandas as pd

def air_temp_tests(df: pd.DataFrame):
    # Air Temperature models
    air_temp_pred = AEM.bio_sci_equat.model_air_temp(df2024)
    AEM.visual.plot_air_temp(df, 'Tests/Images/air_temp_plot.png', air_temp_pred)

def evapo_transpiation_tests(df: pd.DataFrame):
    # Set Latitude and Altitude for Evapo Models
    latitude = 34
    altitude = 350 # m

    # Calculate EVapo model Data
    evapo_models = [
    AEM.bio_sci_equat.dalton(df[labels['tmin']], df[labels['tmax']], df[labels['hmin']], df[labels['hmax']], df[labels['w2avg']]),
    AEM.bio_sci_equat.penman(df[labels['tmin']], df[labels['tmax']], df[labels['hmin']], df[labels['hmax']], df[labels['w2avg']]),
    AEM.bio_sci_equat.romanenko(df[labels['tmin']], df[labels['tmax']], df[labels['hmin']], df[labels['hmax']]),
    AEM.bio_sci_equat.jensen_haise(df[labels['tmin']], df[labels['tmax']], df['DOY'], latitude),
    AEM.bio_sci_equat.hargreaves(df[labels['tmin']], df[labels['tmin']], df[labels['tmin']], latitude),
    #TODO: Need to add solar radiation calculations to get this model to work
        # Add this model label back to the list of labels
    #AEM.bio_sci_equat.penman_monteith(df[labels['tmin']], df[labels['tmax']], df[labels['hmin']], df[labels['hmax']], df['solar_rad'], df[labels['w2avg']], df['DOY'], latitude, altitude)
    ]

    # Labels for the models
    evapo_model_labels = ['Dalton', 'Penman', 'Romanenko', 'Jensen-Haise', 'Hargreaves']

    # Plot Model Data
    AEM.visual.plot_evapo_data(df, 'Tests/Images/Evapo_All.png', evapo_models, evapo_model_labels)

def runoff_tests(df: pd.DataFrame):
    runoff = AEM.bio_sci_equat.model_rainfall(df)
    AEM.visual.plot_rainfall(runoff, 'Tests/Images/runoff_plot.png')


if __name__ == '__main__':
    # Get Data
    data_file = "Tests/Data/Marshfield_All_Data.csv"
    df = AEM.bio_sci_equat.load_data(data_file)
    df2023 = AEM.bio_sci_equat.load_data(data_file, start_date="2023-01-01", end_date="2023-12-31")
    df2024 = AEM.bio_sci_equat.load_data(data_file, start_date="2024-01-01", end_date="2024-12-31")
    df2025 = AEM.bio_sci_equat.load_data(data_file, start_date="2025-01-01", end_date="2025-12-31")

    labels = AEM.settings.get_labels()

    air_temp_tests(df2024)
    evapo_transpiation_tests(df2024)
    runoff_tests(df2024)


