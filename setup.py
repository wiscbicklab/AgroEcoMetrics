from setuptools import setup, find_packages

setup(
    name='agroecometrics',
    version='0.0.2b2',
    author='Scarlett Olson',
    author_email='Scarlett.Olson@wisc.edu',
    description='This package contains useful tools for Aggriculture and Ecological researchers to clean,' \
    'manipulate, display, and convert data.',
    packages=find_packages(include=["agroecometrics", "agroecometrics.*"]),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib'
    ],
    classifiers=[
    'Programming Language :: Python :: 3',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    Homepage = "https://github.com/wiscbicklab/AgroEcoMetrics",
    Issues = "https://github.com/wiscbicklab/AgroEcoMetrics/issues"
)