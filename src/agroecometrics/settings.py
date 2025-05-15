# Default labels
_labels = {
    "date": "Date",
    "date_format": "%m/%d/%Y %I:%M %p",
    "temp": "T_DAILY_AVG",
    "rain": "Daily Total Rain (mm)",
    "soiltemp2": "5 Minute Average Soil Temperature at 2 Inches (c)",
    "soil_temp4": "5 Minute Average Soil Temperature at 4 Inches (c)",
    "soil_temp8": "5 Minute Average Soil Temperature at 8 Inches (c)",
    "soil_temp20": "5 Minute Average Soil Temperature at 20 Inches (c)",
    "soil_temp40": "5 Minute Average Soil Temperature at 40 Inches (c)",
    "tmin": "Daily Minimum Air Temperature (c)",
    "tmax": "Daily Maximum Air Temperature (c)",
    "hmin": "Daily Minimum Relative Humidity (pct)",
    "hmax": "Daily Maximum Relative Humidity (pct)",
    "w2avg": "Daily Maximum Wind Gust (m/h)",
    "solar_rad":  "Daily solar radiation"


}

def get_labels():
    """Return current labels."""
    return _labels.copy()

def set_labels(**kwargs):
    """Override default labels with user-defined ones."""
    for key, value in kwargs.items():
        if key in _labels:
            _labels[key] = value
        else:
            raise KeyError(f"Invalid label key: '{key}'. Allowed keys are: {list(_labels.keys())}")
