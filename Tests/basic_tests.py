import agroecometrics as AEM


def temp_tests(df):
    # Air Temperature models
    air_temp_pred = AEM.bio_sci_equat.model_air_temp(df2024)

def evapo_transpiation_tests(df):
    def temp_tests(df):
    # Air Temperature models
    air_temp_pred = AEM.bio_sci_equat.model_air_temp(df2024)


if __name__ == '__main__':
    # Get Data
    df = AEM.bio_sci_equat.load_data("Data/Marshfield_All_Data.csv")
    df2023 = AEM.bio_sci_equat.load_data("Data/Marshfield_All_Data.csv", start_date="2023-01-01", end_date="2023-12-31")
    df2024 = AEM.bio_sci_equat.load_data("Data/Marshfield_All_Data.csv", start_date="2024-01-01", end_date="2024-12-31")
    df2025 = AEM.bio_sci_equat.load_data("Data/Marshfield_All_Data.csv", start_date="2025-01-01", end_date="2025-12-31")

    labels = AEM.settings.get_labels()

    temp_tests(df2024)


    # Evapo Transpiration  models

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
    AEM.bio_sci_equat.penman_monteith(df[labels['tmin']], df[labels['tmax']], df[labels['hmin']], df[labels['hmax']], df['ATOT'], df[labels['w2avg']], df['DOY'], latitude, altitude)
    ]

    # Labels for the models
    evapo_model_labels = ['Dalton', 'Penman', 'Romanenko', 'Jensen-Haise', 'Hargreaves', 'Penman-Monteith']






