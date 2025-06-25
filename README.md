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

Provides utilities for loading, cleaning, interpolating, and saving agricultural datasets. Includes functions to:

- Check CSV file validity
- Load and filter data by date range
- Interpolate missing data
- Save processed DataFrames

### `agroecometrics.equations`

Contains models and equations for ecological and agricultural analysis, including:

- **Evapotranspiration Models**: Dalton, Penman, Hargreaves, etc.
- **Solar Radiation Calculations**: Compute extraterrestrial radiation (`Ra`)
- **Vapor Pressure Calculations**: Estimate saturation vapor pressure (`esat`)
- **Water Movement Models**: Infiltration and hydraulic conductivity
- **Crop Models**: Calculate Growing Degree Days (GDD)

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
file_path = Path("your_weather_data.csv")
df = AEM.data.load_data(file_path)

# Create a Graph of air temperature on a particular day
air_temp_pred = AEM.equations.model_air_temp(df)
AEM.visualizations.plot_air_temp(df, air_temp_pred, Path("your_saved_plot.png"))

# Add Growing Degree Days (GDD) to your dataset
gdd_to_df(df, temp_avg=df["TempAvg"].values, temp_base=10)

# Save your CSV file with new data
AEM.data.save_data(df, Path('your_updated_data.csv'))
```

This script loads weather data, computes growing degree days, and adds the results to your DataFrame.

