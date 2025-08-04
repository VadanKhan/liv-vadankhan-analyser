import sys
import time
import os
import traceback
import warnings
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from typing import Union, List
from collections import defaultdict
from tqdm import tqdm
import configparser
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import plotly.io as pio

pio.renderers.default = "notebook"
# import plotly.graph_objs as go
# from plotly.subplots import make_subplots


# import statsmodels.api as sm
import scipy.stats as st

# from scipy.stats import linregress
from scipy.signal import cheby1, filtfilt, savgol_filter
from scipy.optimize import curve_fit, minimize, least_squares
from scipy.signal import find_peaks
from sklearn.linear_model import RANSACRegressor, LinearRegression

CURRENT_DIR = Path(os.getcwd())
# Move to the root directory
ROOT_DIR = CURRENT_DIR.parents[0]  # Adjust the number based on your folder structure
# Add the root directory to the system path
sys.path.append(str(ROOT_DIR))
# Import the importlib module
import importlib

# import function implementations
import stst_urls

# Reload the modules
importlib.reload(stst_urls)
# Re-import the functions


from stst_urls import GTX_URL

RESULTS_FILE_PATH = ROOT_DIR / "results"
EXPORTS_FILEPATH = ROOT_DIR / "exports"


# Create the exports folder if it doesn't exist
if not os.path.exists(EXPORTS_FILEPATH):
    os.makedirs(EXPORTS_FILEPATH)
# print(EXPORTS_FILEPATH)
