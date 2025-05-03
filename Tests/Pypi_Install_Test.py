import agroecometrics as AEM
import os


if __name__ == '__main__':
    AEM.bio_sci_equations.Runoff.load_data("~/Documents/Entomology_Lab/agro-eco-metrics/private/Marshfield_02_11_2023-04_28_2025.csv", [2024])
    AEM.bio_sci_equations.Runoff.compute_rainfall()
    file_path = os.path.expanduser('~/Documents/Entomology_Lab/agro-eco-metrics/private/runoff_plot.png')
    AEM.bio_sci_equations.Runoff.plot_rainfall(file_path)