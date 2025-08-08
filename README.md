# AgroEcoMetrics

AgroEcoMetrics is a useful tool for manipulating agricultural and Ecological data

Source code:    ([https://github.com/wiscbicklab/AgroEcoMetrics](https://github.com/wiscbicklab/AgroEcoMetrics))

Bug reports:    ([https://github.com/wiscbicklab/AgroEcoMetrics/issues](https://github.com/wiscbicklab/AgroEcoMetrics/issues))

Documentation:  ([https://wiscbicklab.github.io/AgroEcoMetrics/](https://wiscbicklab.github.io/AgroEcoMetrics/))

It provides:

- Methods for calulating agricultural and ecological data
- Methods for manipulating and cleaning agriculture and ecological data
- Methods for visualizing agricultural and ecological data

## Submodules Overview

### `agroecometrics.data`

Provides utilities for loading, cleaning, interpolating, manipulating, and saving agricultural datasets. Includes functions to:

- Check CSV file validity
- Load and filter data by date range
- Interpolate missing data
- Save processed DataFrames
- Match date times between numpy arrays
- Get a pandas DataFrame as a dictionary

### `agroecometrics.equations`

Contains models and equations for ecological and agricultural analysis, including:

- **TEMPERATURE MODELS**: Soil and Air Temperature Prediction Models.
- **Evapotranspiration Models**: Dalton, Penman, Hargreaves, etc.
- **Crop Models**: Calculate Growing Degree Days (GDD)
- **Photoperiod Models**: Photoperiod Predictions.
- **Water Movement Models**: Infiltration and hydraulic conductivity

### `agroecometrics.visualizations`

Provides methods for creating plots from the calulations made in equations.

- Air Temperature plots
- Soil temperature plots and 3d mesh graphs
- Rainfall and runoff plot
- Growing degree day plots
- Photoperiod prediction plots

## Getting Started

### Installation

Install via pip:

```bash
pip install AgroEcoMetrics
```

### Quick Example

```python
from pathlib import Path
from agroecometrics as AEM
import pandas as pd

# Load your data
data_path = Path("**your_weather_data.csv**")
image_path = Path("**your_saved_plot.png**")
df = AEM.data.load_data_csv(data_path, "**date_time_col_name**", start_date='2024-01-01', end_date='2024-12-31')

# Create a Graph of air temperature on a particular day
date_times = df["**date_time_col_name**"]
avg_air_temp = df['**avg_air_temp_col_name**']
air_temp_pred = AEM.equations.model_air_temp(avg_air_temp)
AEM.visualizations.plot_air_temp(avg_air_temp, air_temp_pred, date_times, image_path)
```

This script loads weather data filtered to only 2024, creates an air temperature models from the data, and saves a plot of the predicted temperatures from the model against the actual temperatures.

