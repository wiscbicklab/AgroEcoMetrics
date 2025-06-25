# Labels are used to convert data references to the actual names of the data used in the csv files
_data_labels = {
    # Date Labels
    "date_time": "Date & Time Collected",
    "date_format": "%m/%d/%Y %I:%M %p",

    # Air Temperature Labels
    "temp_avg": "T_DAILY_AVG",
    "temp_min": "Daily Minimum Air Temperature (c)",
    "temp_max": "Daily Maximum Air Temperature (c)",
    "5_minute_temp": "5 Minute Average Air Temperature (c)",

    # Soil Temperature Labels
    "soil_temp2": "5 Minute Average Soil Temperature at 2 Inches (c)",
    "soil_temp4": "5 Minute Average Soil Temperature at 4 Inches (c)",
    "soil_temp8": "5 Minute Average Soil Temperature at 8 Inches (c)",
    "soil_temp20": "5 Minute Average Soil Temperature at 20 Inches (c)",
    "soil_temp40": "5 Minute Average Soil Temperature at 40 Inches (c)",

    # Humidity Labels
    "hmin": "Daily Minimum Relative Humidity (pct)",
    "hmax": "Daily Maximum Relative Humidity (pct)",

    # Air Pressure and Wind labels
    "pmin": "Daily Minimum Pressure (mb)",
    "pmax": "Daily Maximum Pressure (mb)",
    "max_gust": "Daily Maximum Wind Gust (m/h)",

    # Misellaneous Labels
    "solar_rad":  "Daily solar radiation",
    "rain": "Daily Total Rain (mm)",
}

# Calulation Labels allow shorttened references to the names that are used
#   to store calculation data in the csv
_calc_labels = {
    # Growing Degree Days (GDD) Labels
    "gdd": "GROWING_DEGREE_DAYS",
    "gdd_sum": "GROWING_DEGREE_DAYS_SUM",

    # Rainfall Lables
    "runoff": "RUNOFF",
    "rain_sum": "RAIN_SUM",
    "runoff_sum": "RUNOFF_SUM",

    # Other Labels
    "evapo": "EVAPOTRANSPIRATION",

    # Date and Time Labels
    "date_norm": "NORMALIZED_DATE",
    "doy": "DAYS_SINCE_JANUARY_1ST",
    "year": "YEAR",

}

def get_labels():
    """Return: The current label dictionary."""
    return {**_data_labels, **_calc_labels}

def get_calculation_lables():
    """Return: The current labels used to store calculated data"""
    return _calc_labels.copy()

def set_labels(**kwargs):
    """Override default labels with user-defined ones."""
    for key, value in kwargs.items():
        if key in _data_labels:
            _data_labels[key] = value
        else:
            raise KeyError(f"Invalid label key: '{key}'. Allowed keys are: {list(_data_labels.keys())}")
