# Labels are used to convert a generic label for a type of data to the 
#       actual label used with in the csv file.
_labels = {
    "date": "Date & Time Collected",
    "date_format": "%m/%d/%Y %I:%M %p",
    "temp_avg": "T_DAILY_AVG",
    "temp_min": "Daily Minimum Air Temperature (c)",
    "temp_max": "Daily Maximum Air Temperature (c)",
    "rain": "Daily Total Rain (mm)",
    "soiltemp2": "5 Minute Average Soil Temperature at 2 Inches (c)",
    "soil_temp4": "5 Minute Average Soil Temperature at 4 Inches (c)",
    "soil_temp8": "5 Minute Average Soil Temperature at 8 Inches (c)",
    "soil_temp20": "5 Minute Average Soil Temperature at 20 Inches (c)",
    "soil_temp40": "5 Minute Average Soil Temperature at 40 Inches (c)",
    "hmin": "Daily Minimum Relative Humidity (pct)",
    "hmax": "Daily Maximum Relative Humidity (pct)",
    "pmin": "Daily Minimum Pressure (mb)",
    "pmax": "Daily Maximum Pressure (mb)",
    "max_gust": "Daily Maximum Wind Gust (m/h)",
    "solar_rad":  "Daily solar radiation",
    "5_minute_temp": "5 Minute Average Air Temperature (f)",
}

def get_labels():
    """Return: The current label dictionary."""
    return _labels.copy()

def set_labels(**kwargs):
    """Override default labels with user-defined ones."""
    for key, value in kwargs.items():
        if key in _labels:
            _labels[key] = value
        else:
            raise KeyError(f"Invalid label key: '{key}'. Allowed keys are: {list(_labels.keys())}")
