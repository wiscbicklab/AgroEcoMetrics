# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))



project = 'agro-eco-metrics'
copyright = "2025, 'Scarlett Olson'"
author = "'Scarlett Olson'"
release = '0.0.3b1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",      # Auto-generates documentation from docstrings
    "sphinx.ext.napoleon",     # Supports NumPy/Google-style docstrings
    "sphinx.ext.viewcode",     # Adds links to highlighted source code
    "sphinx.ext.autosummary",  # Summarizes module/class/function listings
    "sphinx_autodoc_typehints",  # Optional, needs pip install
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
