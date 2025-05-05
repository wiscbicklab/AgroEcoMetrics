import os

def check_png_filename(file_name):
    if not isinstance(file_name, str):
        raise TypeError("file_name must be a string.")
    if not file_name.lower().endswith('.png'):
        raise ValueError("The filename must end with '.png'.")
    dir_name = os.path.dirname(file_name)
    if dir_name and not os.path.exists(dir_name):
        raise FileNotFoundError(f"The directory '{dir_name}' does not exist.")