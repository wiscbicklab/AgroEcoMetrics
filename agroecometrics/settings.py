# Default labels
_labels = {
    "date": "Date",
    "temp": "T_DAILY_AVG",
    "rain": "Daily Total Rain (mm)",  # optional future use
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
