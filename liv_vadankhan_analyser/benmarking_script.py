import sys
import time
import os
import traceback

# import warnings
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from typing import Union, List
from collections import defaultdict

# from tqdm import tqdm

import pandas as pd
import numpy as np

# import matplotlib.pyplot as plt

# import plotly.io as pio

# pio.renderers.default = "notebook"
# import plotly.graph_objs as go
# from plotly.subplots import make_subplots


import statsmodels.api as sm
import scipy.stats as st
from scipy.stats import linregress
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

# ------------------------- FILELINK FINDER ------------------------ #


def liv_raw_filelink_finder(
    wafer_codes, fileserver_link: str, product_code: Union[str, List[str]] = "QC", select_earliest: bool = False
):
    if isinstance(product_code, str):
        product_codes = [product_code]
    else:
        product_codes = product_code

    subdirectory_map = {wafer_code: None for wafer_code in wafer_codes}

    for code in product_codes:
        base_url = f"{fileserver_link}{code}/"
        # print(f"Checking fileserver link: {base_url}")

        try:
            response = requests.get(base_url, verify=False)
            soup = BeautifulSoup(response.content, "html.parser")
            links = soup.find_all("a")

            for wafer_code in wafer_codes:
                if subdirectory_map[wafer_code] is not None:
                    continue
                for link in links:
                    href = link.get("href")
                    if href and wafer_code in href:
                        subdirectory_map[wafer_code] = base_url + href
                        break
        except Exception as e:
            print(f"Failed to fetch from {base_url}: {e}")

    wafer_records = []
    missing = []

    for wafer_code in wafer_codes:
        subdirectory_url = subdirectory_map.get(wafer_code)

        if not subdirectory_url:
            missing.append(wafer_code)
            continue

        try:
            response = requests.get(subdirectory_url, verify=False)
            soup = BeautifulSoup(response.content, "html.parser")
            links = soup.find_all("a")
        except Exception as e:
            print(f"Failed to access subdir for {wafer_code}: {e}")
            missing.append(wafer_code)
            continue

        cmp = (lambda a, b: a < b) if select_earliest else (lambda a, b: a > b)
        init_time = "99999999999999" if select_earliest else ""

        latest_file = None
        latest_cod_file = None
        latest_degradation_file = None
        latest_time = init_time
        latest_cod_time = init_time
        latest_degradation_time = init_time
        machine_name = None
        proc_cod70 = None
        proc_cod250 = None
        proc_cod_base = None

        for link in links:
            href = link.get("href")
            if not href:
                continue

            if "RAW" in href:
                time_str = href[-18:-4]
                if not machine_name:
                    machine_name = href[:6]

                if "COD250" in href:
                    if cmp(time_str, latest_cod_time):
                        latest_cod_time = time_str
                        latest_cod_file = subdirectory_url + href

                elif "COD70" in href:
                    if cmp(time_str, latest_degradation_time):
                        latest_degradation_time = time_str
                        latest_degradation_file = subdirectory_url + href

                else:
                    if cmp(time_str, latest_time):
                        latest_time = time_str
                        latest_file = subdirectory_url + href

            elif "processed" in href and "COD" in href:
                full_url = subdirectory_url + href
                if "COD250" in href and proc_cod250 is None:
                    proc_cod250 = full_url
                elif "COD70" in href and proc_cod70 is None:
                    proc_cod70 = full_url
                elif "COD" in href and "COD250" not in href and "COD70" not in href and proc_cod_base is None:
                    proc_cod_base = full_url

        if latest_file is None:
            missing.append(wafer_code)
            continue  # Skip wafers with no valid RAW file

        wafer_records.append(
            {
                "wafer_code": wafer_code,
                "file_url": latest_file,
                "file_cod_url": latest_cod_file,
                "file_degradation_url": latest_degradation_file,
                "file_time": latest_time if latest_file else None,
                "file_cod_time": latest_cod_time if latest_cod_file else None,
                "file_degradation_time": latest_degradation_time if latest_degradation_file else None,
                "machine": machine_name,
                "proc_cod70_url": proc_cod70,
                "proc_cod250_url": proc_cod250,
                "proc_cod_base_url": proc_cod_base,
            }
        )

        # Optional: If you want to keep all entries even with missing data, comment out the `continue` above
        # and always append to wafer_records, just using None values

    return wafer_records, missing


# ----------------------------- INPUTS ----------------------------- #

wafer_codes = [
    "QCI3N",
    # "QCI3S",
    # "QCI85",
]

ANALYSIS_RUN_NAME = "optimising_ith"
SUBARU_DECODER = "QC WAFER_LAYOUT 24Dec.csv"
HALO_DECODER = "HALO_DECODER_NE-rev1_1 logic_coords_annotated.csv"


print(f"Number of Wafers: {len(wafer_codes)}")


wafer_records, missing_wafers = liv_raw_filelink_finder(wafer_codes, GTX_URL, ["NV", "QD", "QC"], select_earliest=False)

for record in wafer_records:
    print(record["wafer_code"], record["file_url"])


file_urls = [w["file_url"] for w in wafer_records]
wafer_codes = [w["wafer_code"] for w in wafer_records]

# ------------------ GENERAL PROCESSING FUNCTIONS ------------------ #


def basic_sweep_analysis(df):
    """
    Compute first and second order differentials for voltage (Vf) and photodiode signal (PD)
    while ensuring calculations remain per device.
    Additionally, compute min and max PD per touchdown and clone max PD across the sweep.
    """
    df["dV/dI"] = df.groupby("TOUCHDOWN")["Vf"].diff()
    df["dP/dI"] = df.groupby("TOUCHDOWN")["PD"].diff()
    df["d2V/dI2"] = df.groupby("TOUCHDOWN")["dV/dI"].diff()
    df["d2P/dI2"] = df.groupby("TOUCHDOWN")["dP/dI"].diff()

    df["MAX_PD"] = df.groupby("TOUCHDOWN")["PD"].transform("max")
    df["MIN_PD"] = df.groupby("TOUCHDOWN")["PD"].transform("min")
    return df


def flag_no_laser(intensity, max_pd=1):
    """
    Returns True if the max intensity is below the threshold (i.e., no laser detected),
    otherwise returns False.

    Parameters:
        intensity (array-like): 1D array of PD values.
        max_pd (float): Threshold below which it is flagged as "NO LASER".

    Returns:
        bool: True if "NO LASER", False otherwise.
    """
    if np.max(intensity) < max_pd:
        return True  # "NO LASER"
    else:
        return False


# Linear model for line fitting
def linear_model(x, slope, intercept):
    return slope * x + intercept


# Least Absolute Residuals fitting function using L1 norm
def least_absolute_residuals_fit(x, y, model, initial_guess, bounds):
    def objective(params):
        return np.sum(np.abs(model(x, *params) - y))

    result = minimize(objective, initial_guess, bounds=bounds, method="L-BFGS-B", options={"maxiter": 1000})

    residuals = np.abs(model(x, *result.x) - y)
    mean_abs_error = np.mean(residuals)

    return result.x, mean_abs_error


# ------------------------------- ITH ------------------------------ #


# Gaussian model for fitting
def gaussian(x, a, x0, sigma):
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def gaussian_residuals(p, x, y):
    return gaussian(x, *p) - y


def find_ith_value_labview(intensity, current, use_ransac_right=True, ransac_threshold=0.1):
    # try:
    # 1) Trim data to only include values >2 and <=35 mA
    mask_trimmed = (current > 1) & (current <= 34)
    if not np.any(mask_trimmed):
        print("Warning: No data points between 2 and 35 mA.")
        return 0
    current = current[mask_trimmed]
    intensity = intensity[mask_trimmed]

    # 2) Interpolate to double resolution (spacing of 0.5 mA)
    current_interp = np.arange(np.min(current), np.max(current) + 0.1, 0.5)
    intensity_interp = np.interp(current_interp, current, intensity)

    current = current_interp
    intensity = intensity_interp

    # 3) Normalize intensity using min-max scaling (first operation)
    min_intensity = np.min(intensity)
    max_intensity = np.max(intensity)
    intensity_norm = (intensity - min_intensity) / (max_intensity - min_intensity)

    # 4) Sort data by current to ensure proper processing
    sorted_indices = np.argsort(current)
    current = current[sorted_indices]
    intensity_norm = intensity_norm[sorted_indices]

    # 5) Apply Chebyshev high-pass filter (order 2, ripple 0.1 dB, bandpass 0.15–0.45)
    b, a = cheby1(N=2, rp=0.1, Wn=0.45, btype="lowpass", fs=1)
    filtered_intensity = filtfilt(b, a, intensity_norm)

    # 5a) Initial linear fit via least absolute residuals on filtered data using QuantReg
    X = sm.add_constant(current)  # Adds intercept term
    model = sm.QuantReg(filtered_intensity, X)
    res = model.fit(q=0.5)
    slope_left = res.params[1]
    intercept_left = res.params[0]
    initial_abs_residual_total = np.mean(np.abs(filtered_intensity - res.predict(X)))

    # print(initial_abs_residual_total)
    if initial_abs_residual_total > 1:
        print("Warning: Initial L1 fit residual too high.")
        return 0  # DEBUG: STILL TRY

    # 6) First Savitzky-Golay smoothing (5,1)
    smoothed_intensity = savgol_filter(filtered_intensity, window_length=5, polyorder=1)

    # 7) Second Savitzky-Golay smoothing before differential (3,2)
    smoothed_intensity = savgol_filter(smoothed_intensity, window_length=3, polyorder=2)

    # 8) Compute first derivative (renamed to dL_dI)
    dL_dI = np.gradient(smoothed_intensity, current)

    # 9) Smooth first derivative (6,2)
    smoothed_dL_dI = savgol_filter(dL_dI, window_length=6, polyorder=2)

    # 10) Compute second derivative (renamed to d2L_dI2)
    d2L_dI2 = np.gradient(smoothed_dL_dI, current)

    # 11) Smooth second derivative (6,2)
    smoothed_d2L_dI2 = savgol_filter(d2L_dI2, window_length=6, polyorder=2)

    # 11a) Set negative second derivative values to zero (LabVIEW-like behavior)
    smoothed_d2L_dI2[smoothed_d2L_dI2 < 0] = 0

    # 12) Normalize second derivative and add 0.01
    max_d2L_dI2 = np.max(smoothed_d2L_dI2)
    if max_d2L_dI2 == 0:
        print("Warning: Second derivative all zero after zeroing negatives.")
        return 0
    d2L_dI2_ready = (smoothed_d2L_dI2 / max_d2L_dI2) + 0.01

    # 13) Least Absolute Residuals fitting with initial conditions and bounds
    # --- New logic: Find all peaks above 0.95 ---
    peaks, _ = find_peaks(d2L_dI2_ready, height=0.95)
    if len(peaks) == 0:
        print("Warning: No significant peaks in second derivative.")
        return 0

    # Select the leftmost (lowest current) among the high peaks
    selected_peak_idx = peaks[np.argmin(current[peaks])]
    x0_guess = current[selected_peak_idx]
    a_guess = d2L_dI2_ready[selected_peak_idx]
    sigma_guess = (max(current) - min(current)) / 5
    initial_guess = [a_guess, x0_guess, sigma_guess]

    a_min = 0.1 * np.max(d2L_dI2_ready)
    a_max = 2.0 * np.max(d2L_dI2_ready)
    x0_min = max(10, min(current))
    x0_max = min(30, max(current))
    sigma_min = 0.2
    sigma_max = (x0_max - x0_min) / 2
    bounds = ((a_min, a_max), (x0_min, x0_max), (sigma_min, sigma_max))

    popt, _ = least_absolute_residuals_fit(current, d2L_dI2_ready, gaussian, initial_guess, bounds)
    median_x = popt[1]

    # 14) Validate split point
    if not (2 <= median_x <= 35):
        print("Warning: Gaussian fit split point out of usual bounds.")
        return 0

    # 15) Linear fit on left segment
    left_mask = current <= median_x
    if not np.any(left_mask):
        print("Warning: No data points on the left segment for fitting.")
        return 0
    current_left = current[left_mask]
    intensity_left = intensity_norm[left_mask]
    A_left = np.vstack([current_left, np.ones_like(current_left)]).T
    solution_left, _, _, _ = np.linalg.lstsq(A_left, intensity_left, rcond=None)
    slope_left, intercept_left = solution_left

    # 16) Linear fit on right segment
    right_mask = current > median_x
    if not np.any(right_mask):
        print("Warning: No data points on the right segment for fitting.")
        return 0
    current_right = current[right_mask]
    intensity_right = intensity_norm[right_mask]

    # Optionally remove last N points to avoid noisy tail
    N = 15
    current_right = current_right[:-N]
    intensity_right = intensity_right[:-N]
    if len(current_right) < 10:
        print("Warning: Fewer than 10 data points for stimulated emission fit.")
        return 0

    if use_ransac_right:
        # Fit with RANSAC
        X = current_right.reshape(-1, 1)
        y = intensity_right
        ransac = RANSACRegressor(LinearRegression(), residual_threshold=ransac_threshold, max_trials=100)
        ransac.fit(X, y)

        slope_right = ransac.estimator_.coef_[0]
        intercept_right = ransac.estimator_.intercept_
        fitted_right = ransac.predict(X)

        inlier_mask = ransac.inlier_mask_
        mse_right = np.mean((y[inlier_mask] - fitted_right[inlier_mask]) ** 2)
    else:
        # Use ordinary least squares
        A_right = np.vstack([current_right, np.ones_like(current_right)]).T
        solution_right, _, _, _ = np.linalg.lstsq(A_right, intensity_right, rcond=None)
        slope_right, intercept_right = solution_right
        fitted_right = A_right @ solution_right

        mse_right = np.mean((intensity_right - fitted_right) ** 2)

    if mse_right > 100:
        print("Warning: High MSE in stimulated emission fit.")
        return 0

    # 17) Compute intersection (I_th)
    ith_value = (intercept_right - intercept_left) / (slope_left - slope_right)
    if not (2 <= ith_value <= 35):
        print("Warning: Computed I_th outside bounds normal bounds.")
        return 0  # DEBUG: STILL TRY

    return ith_value


def find_ith_value_labview_timed(intensity, current, use_ransac_right=True, ransac_threshold=0.1, mu_bound_window=5):
    timings = {}
    t0 = time.time()

    # Step 1: Trim data
    mask_trimmed = (current > 1) & (current <= 34)
    if not np.any(mask_trimmed):
        return 0, {"error": "no_trimmed_data"}
    current = current[mask_trimmed]
    intensity = intensity[mask_trimmed]
    timings["trim"] = time.time() - t0
    t0 = time.time()

    # Step 2: Interpolation
    current_interp = np.arange(np.min(current), np.max(current) + 0.1, 0.5)
    intensity_interp = np.interp(current_interp, current, intensity)
    current = current_interp
    intensity = intensity_interp
    timings["interpolation"] = time.time() - t0
    t0 = time.time()

    # Step 3: Normalization
    min_intensity = np.min(intensity)
    max_intensity = np.max(intensity)
    intensity_norm = (intensity - min_intensity) / (max_intensity - min_intensity)
    timings["normalize"] = time.time() - t0
    t0 = time.time()

    # Step 4: Sorting
    sorted_indices = np.argsort(current)
    current = current[sorted_indices]
    intensity_norm = intensity_norm[sorted_indices]
    timings["sort"] = time.time() - t0
    t0 = time.time()

    # # Step 5: Chebyshev filter
    # b, a = cheby1(N=2, rp=0.1, Wn=0.45, btype="lowpass", fs=1)
    # filtered_intensity = filtfilt(b, a, intensity_norm)
    # timings["cheby_filter"] = time.time() - t0
    # t0 = time.time()

    # # Step 5a: Initial L1 fit
    # X = sm.add_constant(current)
    # model = sm.QuantReg(filtered_intensity, X)
    # res = model.fit(q=0.5)
    # slope_left = res.params[1]
    # intercept_left = res.params[0]
    # initial_abs_residual_total = np.mean(np.abs(filtered_intensity - res.predict(X)))
    # if initial_abs_residual_total > 1:
    #     return 0, {"error": "high_l1_residual"}
    # timings["l1_fit"] = time.time() - t0
    # t0 = time.time()

    # Step 6–11: Smoothing + Derivatives (granular timings)
    t0 = time.time()

    # t1 = time.time()
    # smoothed_intensity = savgol_filter(intensity_norm, 5, 1)
    # timings["smooth_1"] = time.time() - t1

    # t2 = time.time()
    # smoothed_intensity = savgol_filter(smoothed_intensity, 3, 2)
    # timings["smooth_2"] = time.time() - t2

    t3 = time.time()
    dL_dI = np.gradient(intensity_norm, current)
    timings["gradient_1"] = time.time() - t3

    # t4 = time.time()
    # smoothed_dL_dI = savgol_filter(dL_dI, 6, 2)
    # timings["smooth_3"] = time.time() - t4

    t5 = time.time()
    d2L_dI2 = np.gradient(dL_dI, current)
    timings["gradient_2"] = time.time() - t5

    t6 = time.time()
    smoothed_d2L_dI2 = savgol_filter(d2L_dI2, 6, 2)
    timings["d2L/dI2_smooth"] = time.time() - t6

    t7 = time.time()
    smoothed_d2L_dI2[smoothed_d2L_dI2 < 0] = 0
    timings["clip_neg"] = time.time() - t7

    # Total smoothing time (optional, includes all above)
    timings["total_smoothing"] = time.time() - t0
    t0 = time.time()  # reset timer for next block

    # Step 12: Normalize 2nd derivative
    max_d2L_dI2 = np.max(smoothed_d2L_dI2)
    if max_d2L_dI2 == 0:
        return 0, {"error": "d2_zero"}
    d2L_dI2_ready = (smoothed_d2L_dI2 / max_d2L_dI2) + 0.01
    timings["d2_norm"] = time.time() - t0
    t0 = time.time()

    # Step 13a: Peak detection
    peaks, _ = find_peaks(d2L_dI2_ready, height=0.95)
    if len(peaks) == 0:
        return 0, {"error": "no_peak"}
    selected_peak_idx = peaks[np.argmin(current[peaks])]
    timings["peak_detection"] = time.time() - t0
    t0 = time.time()

    if np.max(d2L_dI2_ready) < 0.05:
        return 0, {"error": "fit_peak_too_small"}
    if np.std(d2L_dI2_ready) < 1e-3:
        return 0, {"error": "fit_flat"}

    # Step 13b: Gaussian fit
    a_guess = d2L_dI2_ready[selected_peak_idx]
    x0_guess = current[selected_peak_idx]
    sigma_guess = 1
    initial_guess = [a_guess, x0_guess, sigma_guess]
    bounds = (
        (0.9, 1.011),
        ((x0_guess - mu_bound_window, x0_guess + mu_bound_window)),
        (0.5, 3),
    )

    # fit_mask = (current >= x0_guess - fit_window) & (current <= x0_guess + fit_window)
    # if not np.any(fit_mask):
    #     return 0, {"error": "empty_fit_window"}
    # current_fit = current[fit_mask]
    # d2L_dI2_fit = d2L_dI2_ready[fit_mask]

    # # Least Squares Residual Fitting
    # try:
    #     popt, _ = curve_fit(
    #         gaussian,
    #         current,
    #         d2L_dI2_ready,
    #         p0=initial_guess,
    #         bounds=([b[0] for b in bounds], [b[1] for b in bounds]),
    #         maxfev=100,  # cap iterations
    #     )
    # except RuntimeError:
    #     return 0, {"error": "curve_fit_failed"}

    try:
        res = least_squares(
            gaussian_residuals,
            x0=initial_guess,
            bounds=([b[0] for b in bounds], [b[1] for b in bounds]),
            args=(current, d2L_dI2_ready),
            max_nfev=100,
            method="trf",  # default, but can be explicitly specified
        )
        popt = res.x
    except Exception:
        return 0, {"error": "least_squares_failed"}

    median_x = popt[1]
    if not (2 <= median_x <= 35):
        return 0, {"error": "median_x_out_of_bounds"}
    timings["gaussian_fit"] = time.time() - t0
    t0 = time.time()

    # Step 15–16: Line fits left and right
    left_mask = current <= median_x
    right_mask = current > median_x
    if not np.any(left_mask) or not np.any(right_mask):
        return 0, {"error": "no_left_or_right"}
    current_left = current[left_mask]
    intensity_left = intensity_norm[left_mask]
    A_left = np.vstack([current_left, np.ones_like(current_left)]).T
    solution_left, _, _, _ = np.linalg.lstsq(A_left, intensity_left, rcond=None)
    slope_left, intercept_left = solution_left
    timings["left_fit"] = time.time() - t0
    t0 = time.time()

    current_right = current[right_mask]
    intensity_right = intensity_norm[right_mask]
    current_right = current_right[:-15]
    intensity_right = intensity_right[:-15]
    if len(current_right) < 10:
        return 0, {"error": "too_few_right_pts"}

    # Always do simple least squares fit first
    A_right = np.vstack([current_right, np.ones_like(current_right)]).T
    solution_right, _, _, _ = np.linalg.lstsq(A_right, intensity_right, rcond=None)
    fitted_right = A_right @ solution_right
    residuals = np.abs(intensity_right - fitted_right)

    # Decide whether to use RANSAC
    if use_ransac_right and np.any(residuals > ransac_threshold):
        # RANSAC fallback
        X = current_right.reshape(-1, 1)
        y = intensity_right
        ransac = RANSACRegressor(LinearRegression(), residual_threshold=ransac_threshold, max_trials=100)
        ransac.fit(X, y)
        slope_right = ransac.estimator_.coef_[0]
        intercept_right = ransac.estimator_.intercept_
        fitted_right = ransac.predict(X)
        inlier_mask = ransac.inlier_mask_
        mse_right = np.mean((y[inlier_mask] - fitted_right[inlier_mask]) ** 2)
    else:
        # Use OLS result
        slope_right, intercept_right = solution_right
        mse_right = np.mean((intensity_right - fitted_right) ** 2)

    timings["right_fit"] = time.time() - t0
    t0 = time.time()

    if mse_right > 100:
        return 0, {"error": "high_mse"}

    # Step 17: Compute Ith
    ith_value = (intercept_right - intercept_left) / (slope_left - slope_right)
    if not (2 <= ith_value <= 35):
        return 0, {"error": "ith_out_of_bounds"}
    timings["ith_calc"] = time.time() - t0

    return ith_value, timings


# ------------------------------- SE ------------------------------- #
def find_slope_efficiency(intensity, current, ith, mse_threshold=1, fitting_range=15):
    intensity = np.asarray(intensity)
    current = np.asarray(current)

    if len(intensity) != len(current):
        print("Warning: Input arrays have different lengths.")
        return 0

    # Index where current just exceeds I_th
    idx_start = np.argmax(current >= ith)
    idx_end = idx_start + fitting_range

    if idx_end > len(current):
        print("Warning: Not enough points after I_th for slope efficiency fit.")
        return 0

    # Mask the relevant region
    x = current[idx_start:idx_end]
    y = intensity[idx_start:idx_end]

    # Least squares fit: Ax = y
    A = np.vstack([x, np.ones_like(x)]).T
    try:
        solution, residuals, _, _ = np.linalg.lstsq(A, y, rcond=None)
        slope, intercept = solution

        if residuals.size > 0:
            mse = residuals[0] / len(x)
            if mse > mse_threshold:
                print(f"Warning: High MSE in slope efficiency fit ({mse:.4f}).")
                return 0

        return slope

    except Exception as e:
        print(f"Error during slope efficiency calculation: {e}")
        return 0


# ------------------------------- RS ------------------------------- #
def find_series_resistance(voltage, current, ith, mse_threshold=0.5, fitting_range=15):

    voltage = np.asarray(voltage)

    current = np.asarray(current)

    if len(voltage) != len(current):

        print("Warning: Input arrays have different lengths.")

        return 0

    # Index where current just exceeds I_th

    idx_start = np.argmax(current >= ith)

    idx_end = idx_start + fitting_range

    if idx_end > len(current):

        print("Warning: Not enough points after I_th for series resistance fit.")

        return 0

    # Mask the relevant region

    x = current[idx_start:idx_end]

    y = voltage[idx_start:idx_end]

    # Least squares fit: Ax = y

    A = np.vstack([x, np.ones_like(x)]).T

    try:

        solution, residuals, _, _ = np.linalg.lstsq(A, y, rcond=None)

        slope, intercept = solution

        if residuals.size > 0:

            mse = residuals[0] / len(x)

            if mse > mse_threshold:

                print(f"Warning: High MSE in series resistance fit ({mse:.4f}).")

                return 0

        rs_ohms = slope * 1000  # converting to ohms

        return rs_ohms  # Rs = dV/dI

    except Exception as e:

        print(f"Error during series resistance calculation: {e}")

        return 0


def find_series_resistance_with_vnoise(voltage, current, ith, mse_threshold=0.05, fitting_range=15):

    voltage = np.asarray(voltage)

    current = np.asarray(current)

    if len(voltage) != len(current):

        print("Warning: Input arrays have different lengths.")

        return 0, 0

    # Index where current just exceeds I_th

    idx_start = np.argmax(current >= ith)

    idx_end = idx_start + fitting_range

    if idx_end > len(current):

        print("Warning: Not enough points after I_th for series resistance fit.")

        return 0, 0

    # Mask the relevant region

    x = current[idx_start:idx_end]

    y = voltage[idx_start:idx_end]

    # Least squares fit: Ax = y

    A = np.vstack([x, np.ones_like(x)]).T

    try:

        solution, residuals, _, _ = np.linalg.lstsq(A, y, rcond=None)

        slope, intercept = solution

        if residuals.size > 0:

            mse = residuals[0] / len(x)

            if mse > mse_threshold:

                print(f"Warning: High MSE in series resistance fit ({mse:.4f}).")

                return 0, 0

        rs_ohms = slope * 1000  # converting to ohms

        # Calculate voltage noise above I_th (over fitting region)

        dv = np.diff(y)
        di = np.diff(x)

        dv_di = dv / di

        mean_dv_di = np.mean(dv_di)

        voltage_noise = np.mean(np.abs(dv_di - mean_dv_di))

        return rs_ohms, voltage_noise

    except Exception as e:

        print(f"Error during series resistance calculation: {e}")

        return 0, 0


# ------------------------------- SPD ------------------------------ #
def def_find_spd(
    current, intensity, voltage, no_laser_flag, spd_delta_threshold=1, rollover_threshold=200, noise_threshold=25
):
    """
    Process a single device sweep and return SPD evaluation and related parameters.

    Parameters:
    - current: np.ndarray, shape (n_points,)
    - intensity: np.ndarray, shape (n_points,)
    - voltage: np.ndarray, shape (n_points,)
    - spd_delta_threshold: float, threshold for SPD detection

    Returns:
    - spd_eval: str, one of ["ROLLOVER", "NO LASER", "COD", "COD-PR", "COMD", "COBD"]
    - min_delta: float
    - current_at_spd: float
    - peak_power: float
    - current_at_peak_power: float
    - mean_power: float
    - pot_failmode: str, one of ["COD", "COD-PR", "COMD", "COBD"]
    - voltage_noise: float
    """
    assert len(current) == len(intensity) == len(voltage), "Input arrays must be same length."

    # Use diff-style gradient calculation
    delta_I = np.diff(current)
    dP_dI = np.diff(intensity) / delta_I
    dV_dI = np.diff(voltage) / delta_I
    current_mid = current[1:]  # Midpoints for diff-based gradients

    mean_power = np.mean(intensity)
    peak_power = np.max(intensity)
    current_at_peak_power = current[np.argmax(intensity)]

    delta_spd = mean_power - np.abs(dP_dI)
    min_delta = np.min(delta_spd)
    idx_min_delta = np.argmin(delta_spd)
    current_at_spd = current_mid[idx_min_delta]

    if min_delta > spd_delta_threshold:
        spd_eval = "ROLLOVER"
    elif peak_power < 0.5:
        spd_eval = "NO LASER"
    else:
        spd_eval = "COD"

    if spd_eval == "COD" and current_at_spd > rollover_threshold:
        spd_eval = "COD-PR"

    dv_di_at_spd = dV_dI[idx_min_delta]

    if spd_eval in ["COD", "COD-PR"]:
        if dv_di_at_spd > 0:
            pot_failmode = "COMD"
        elif dv_di_at_spd < 0:
            pot_failmode = "COBD"
        else:
            pot_failmode = spd_eval
    else:
        pot_failmode = spd_eval

    # Voltage noise for current >= noise_threshold, using midpoint-based current
    mask_high_current = current_mid >= noise_threshold
    if np.any(mask_high_current):
        dv_di_high_current = dV_dI[mask_high_current]
        voltage_noise = np.mean(np.abs(dv_di_high_current - np.mean(dv_di_high_current)))
    else:
        voltage_noise = np.nan

    # Override spd_eval and pot_failmode if no laser detected
    if no_laser_flag:
        spd_eval = "NO LASER"
        pot_failmode = "NO LASER"

    result = pd.Series(
        {
            "spd_eval": spd_eval,
            "min_delta": min_delta,
            "current_at_spd": current_at_spd,
            "peak_power": peak_power,
            "current_at_peak_power": current_at_peak_power,
            "mean_power": mean_power,
            "pot_failmode": pot_failmode,
            "voltage_noise": voltage_noise,
        }
    )

    return result


def calculate_confidence_interval_wilsonscore(cod_count, total_count, confidence_level=0.95):
    """
    Calculate the Wilson score confidence interval for a proportion.
    """
    if total_count == 0:
        return 0, 0

    proportion = cod_count / total_count
    z = st.norm.ppf(1 - (1 - confidence_level) / 2)
    denominator = 1 + z**2 / total_count
    center_adjusted_probability = proportion + z**2 / (2 * total_count)
    adjusted_standard_deviation = np.sqrt((proportion * (1 - proportion) + z**2 / (4 * total_count)) / total_count)
    lower_bound = (center_adjusted_probability - z * adjusted_standard_deviation) / denominator
    upper_bound = (center_adjusted_probability + z * adjusted_standard_deviation) / denominator

    # Ensure the lower bound is not less than zero
    lower_bound = max(0, lower_bound)

    return lower_bound * 100, upper_bound * 100


def calculate_confidence_interval(cod_count, total_count, confidence_level=0.95):
    """
    Calculate the confidence interval for a proportion.
    """
    proportion = cod_count / total_count
    z = 1.96  # Z-score for 95% confidence interval
    margin_of_error = z * np.sqrt((proportion * (1 - proportion)) / total_count)
    lower_bound = max(0, proportion - margin_of_error)
    upper_bound = proportion + margin_of_error
    return lower_bound * 100, upper_bound * 100


# ------------------------------- KNK ------------------------------ #
def exclusion_based_on_second_derivative(
    d2LdI2_trimmed, current_trimmed, window_size=1, second_deriv_deviation_threshold=0.001, default_kink_current=55
):
    """
    Use second derivative to find curvature start, and apply weighted linear fit.
    If curvature is marked from the first point, dynamically adjust lower bound.

    Parameters:
        d2LdI2_trimmed: array-like, second derivative values after trimming
        current_trimmed: array-like, corresponding trimmed current values
        window_size: int, smoothing window for Savitzky-Golay filter
        second_deriv_deviation_threshold: float, deviation threshold
        default_kink_current: float, default kink current (mA)

    Returns:

        curved_start_idx: int, index where curvature starts
        d2LdI2_smoothed: array, smoothed second derivative
    """
    # Smooth the second derivative
    d2LdI2_smoothed = savgol_filter(d2LdI2_trimmed, window_length=2 * window_size + 1, polyorder=1)

    # Standard deviation from zero method
    lower_threshold = 0 - second_deriv_deviation_threshold
    upper_threshold = 0 + second_deriv_deviation_threshold
    curved_mask = (d2LdI2_smoothed < lower_threshold) | (d2LdI2_smoothed > upper_threshold)
    curved_start_idx = np.argmax(curved_mask) if np.any(curved_mask) else None

    # Special case: curvature marked from index 0
    if curved_start_idx == 0:
        # Dynamically adjust lower threshold based on starting value
        start_val = d2LdI2_smoothed[0]

        if start_val < lower_threshold:
            lower_threshold = start_val - second_deriv_deviation_threshold
        elif start_val > upper_threshold:
            upper_threshold = start_val + second_deriv_deviation_threshold

        # New mask: keep only values deviating more from start_val than threshold
        curved_mask_adjusted = (d2LdI2_smoothed < lower_threshold) | (d2LdI2_smoothed > upper_threshold)

        if np.any(curved_mask_adjusted):
            curved_start_idx = np.argmax(curved_mask_adjusted)
        else:
            # Still no good region, fallback
            curved_start_idx = np.argmin(np.abs(current_trimmed - default_kink_current))

    elif curved_start_idx is None:
        # No significant curvature found
        curved_start_idx = np.argmin(np.abs(current_trimmed - default_kink_current))

    return curved_start_idx, d2LdI2_smoothed, lower_threshold, upper_threshold


def apply_weighted_linear_fit(I_fit, dLdI_fit, I_rest, dLdI_rest, I_full, reduced_weight=0.01, extra_trim=False):
    """Apply weighted linear fit with optional extra trimming of I_fit region."""
    if extra_trim:
        I_fit = I_fit[5:]
        dLdI_fit = dLdI_fit[5:]

    # Concatenate fit and rest data
    I_all = np.concatenate([I_fit, I_rest])
    dLdI_all = np.concatenate([dLdI_fit, dLdI_rest])
    weights = np.concatenate([np.ones_like(I_fit), reduced_weight * np.ones_like(I_rest)])
    sqrt_w = np.sqrt(weights)

    A = np.vstack([I_all, np.ones_like(I_all)]).T
    A_w = sqrt_w[:, np.newaxis] * A
    dLdI_w = sqrt_w * dLdI_all

    solution, _, _, _ = np.linalg.lstsq(A_w, dLdI_w, rcond=None)
    L_linear_fit = solution[0] * I_full + solution[1]

    return L_linear_fit


def redo_fit_if_kink_too_early(
    I_non_spd,
    dLdI_non_spd,
    I_trimmed,
    dLdI_trimmed,
    d2LdI2_trimmed,
    second_deriv_deviation_threshold,
    trim_boundary_points,
    trim_end_points,
    n_points,
    deviation_percentage_threshold,
    point_skip=5,
):
    """
    If kink is within first 5 points, redo the fit on a later range and re-evaluate deviations & kink.
    """
    trim_boundary_points = trim_boundary_points + point_skip
    I_trimmed = I_trimmed[point_skip - 1 :]
    dLdI_trimmed = dLdI_trimmed[point_skip - 1 :]
    d2LdI2_trimmed = d2LdI2_trimmed[point_skip - 1 :]

    # Widen fit range by excluding 5 more initial points
    curved_start_idx, d2LdI2_smoothed, lower_threshold, upper_threshold = exclusion_based_on_second_derivative(
        d2LdI2_trimmed, I_trimmed, second_deriv_deviation_threshold=second_deriv_deviation_threshold
    )

    # ------------------ Split data based on the mask ------------------ #
    I_fit = I_trimmed[:curved_start_idx]
    dLdI_fit = dLdI_trimmed[:curved_start_idx]
    I_rest = I_trimmed[curved_start_idx:]
    dLdI_rest = dLdI_trimmed[curved_start_idx:]

    new_dLdI_linear_fit = apply_weighted_linear_fit(I_fit, dLdI_fit, I_rest, dLdI_rest, I_non_spd)

    # ---------- Deviation Calculating and Kink Classification --------- #
    deviations = 100 * np.abs(dLdI_non_spd - new_dLdI_linear_fit) / np.maximum(np.abs(new_dLdI_linear_fit), 5e-2)
    kink_mask = deviations > deviation_percentage_threshold
    new_kink_index = np.argmax(kink_mask) if np.any(kink_mask) else None
    new_kink_current = I_non_spd[kink_mask][0] if np.any(kink_mask) else 63

    return {
        "kink_index": new_kink_index,
        "kink_current": new_kink_current,
        "dLdI_linear_fit": new_dLdI_linear_fit,
        "curved_start_idx": curved_start_idx,
        "d2LdI2_smoothed": d2LdI2_smoothed,
        "trim_boundary_points": trim_boundary_points,
        "I_trimmed": I_trimmed,
        "dLdI_trimmed": dLdI_trimmed,
        "d2LdI2_trimmed": d2LdI2_trimmed,
        "deviations_low_kink": deviations,
        "lower_threshold": lower_threshold,
        "upper_threshold": upper_threshold,
    }


def find_kink(
    intensity,
    current,
    ith,
    disable_smooth=True,
    trim_end_points=2,
    trim_boundary_points=5,
    deviation_percentage_threshold=10,
    threshold_min_delta=1,
    second_deriv_deviation_threshold=0.0005,
    early_kink_threshold=50,
):
    spd_detected = False
    low_kink_flag = False

    intensity = np.asarray(intensity)
    intensity_norm = intensity / np.max(intensity) if np.max(intensity) > 0 else intensity
    current = np.asarray(current)

    if len(intensity) != len(current):
        print("Warning: Input arrays have different lengths.")
        return 0, spd_detected, low_kink_flag

    # ---------------- Remove Points before I_threshold ---------------- #
    mask = current >= ith
    if not np.any(mask):
        print("Warning: No current values above the threshold found.")
        return 0, spd_detected
    idx_start = np.argmax(mask)

    I_raw = current[idx_start:]
    L_raw = intensity[idx_start:]

    if len(L_raw) < trim_boundary_points + 1:
        print("Warning: Not enough points after Ith for analysis.")
        return 0, spd_detected, low_kink_flag

    # ----------------- Compute delta for SPD filtering ---------------- #
    dLdI_temp = np.gradient(L_raw, I_raw)
    delta = np.mean(L_raw) - np.abs(dLdI_temp)
    min_delta = np.min(delta)
    max_pd = np.max(L_raw)

    # ------------- Normalize and Compute smoothed L and derivatives ------------- #
    I = I_raw
    L_raw_norm = L_raw / np.max(L_raw) if np.max(L_raw) > 0 else L_raw

    LV_window_gen_smooth = 1
    L = (
        L_raw_norm
        if disable_smooth
        else savgol_filter(L_raw_norm, window_length=2 * LV_window_gen_smooth + 1, polyorder=1)
    )

    dLdI = np.gradient(L, I)
    d2LdI2 = np.gradient(np.gradient(L, I), I)

    # ----------------------- SPD Filtering ----------------------- #
    if min_delta > threshold_min_delta:
        spd_eval = "ROLLOVER"
    elif max_pd < 0.01:
        spd_eval = "NO LASER"
    else:
        spd_eval = "COD"

    if spd_eval == "COD":
        cutoff_idx = np.where(delta < threshold_min_delta)[0][0]
        I_non_spd = I[:cutoff_idx]
        L_non_spd = L[:cutoff_idx]
        dLdI_non_spd = dLdI[:cutoff_idx]
        d2LdI2_non_spd = d2LdI2[:cutoff_idx]
        spd_detected = True
        print("SPD DETECTED: TRIMMING ARRAYS")
    else:
        I_non_spd = I
        L_non_spd = L
        dLdI_non_spd = dLdI
        d2LdI2_non_spd = d2LdI2

    if len(I_non_spd) < trim_boundary_points + 1:
        print("Warning: Not enough points after SPD filtering.")
        return 0, spd_detected, low_kink_flag

    # -------------- Hardcoded Trimming of Boundary Points -------------- #
    n_points = len(I_non_spd)
    if n_points < trim_boundary_points + trim_end_points + 1:
        print("Warning: Not enough points after trimming for fit.")
        return 0, spd_detected, low_kink_flag

    I_trimmed = I_non_spd[trim_boundary_points : n_points - trim_end_points]
    L_trimmed = L_non_spd[trim_boundary_points : n_points - trim_end_points]
    dLdI_trimmed = dLdI_non_spd[trim_boundary_points : n_points - trim_end_points]
    d2LdI2_trimmed = d2LdI2_non_spd[trim_boundary_points : n_points - trim_end_points]

    try:
        # ---------------- 2nd Derivative Masking Algorithm ---------------- #
        curved_start_idx, d2LdI2_smoothed, lower_threshold, upper_threshold = exclusion_based_on_second_derivative(
            d2LdI2_trimmed, I_trimmed, second_deriv_deviation_threshold=second_deriv_deviation_threshold
        )

        # ------------------ Split data based on the mask ------------------ #
        I_fit = I_trimmed[:curved_start_idx]
        dLdI_fit = dLdI_trimmed[:curved_start_idx]
        I_rest = I_trimmed[curved_start_idx:]
        dLdI_rest = dLdI_trimmed[curved_start_idx:]

        # ----------------- Linear Stimulated Fit Algorithm ---------------- #
        dLdI_linear_fit = apply_weighted_linear_fit(I_fit, dLdI_fit, I_rest, dLdI_rest, I_non_spd)

        # ---------- Deviation Calculating and Kink Classification --------- #
        deviations = 100 * np.abs(dLdI_non_spd - dLdI_linear_fit) / np.maximum(np.abs(dLdI_linear_fit), 5e-2)
        deviations_to_check = deviations[trim_boundary_points : n_points - trim_end_points]
        kink_mask = deviations_to_check > deviation_percentage_threshold
        # print(kink_mask)
        kink_current = I_trimmed[kink_mask][0] if np.any(kink_mask) else 63

        # ----------------------- Handle early kinks ----------------------- #
        kink_index = np.argmax(kink_mask) if np.any(kink_mask) else None
        # print(f"kink_index = {kink_index}")
        early_deviations = deviations[:trim_boundary_points]
        mean_early_deviations = np.mean(early_deviations)
        # print(f"mean_early_deviations = {mean_early_deviations}")

        if kink_index is not None and kink_index <= 5 or mean_early_deviations > early_kink_threshold:
            print("Warning: Low Kink Detected, Using Low Kink Countermeasures")
            low_kink_flag = True
            redo_result = redo_fit_if_kink_too_early(
                I_non_spd=I_non_spd,
                dLdI_non_spd=dLdI_non_spd,
                I_trimmed=I_trimmed,
                dLdI_trimmed=dLdI_trimmed,
                d2LdI2_trimmed=d2LdI2_trimmed,
                second_deriv_deviation_threshold=second_deriv_deviation_threshold,
                trim_boundary_points=trim_boundary_points,
                trim_end_points=trim_end_points,
                n_points=n_points,
                deviation_percentage_threshold=deviation_percentage_threshold,
            )
            dLdI_linear_fit = redo_result["dLdI_linear_fit"]
            kink_index = redo_result["kink_index"]
            kink_current = redo_result["kink_current"]
            curved_start_idx = redo_result["curved_start_idx"]
            d2LdI2_smoothed = redo_result["d2LdI2_smoothed"]
            trim_boundary_points = redo_result["trim_boundary_points"]
            I_trimmed = redo_result["I_trimmed"]
            dLdI_trimmed = redo_result["dLdI_trimmed"]
            d2LdI2_trimmed = redo_result["d2LdI2_trimmed"]
            deviations_to_check = redo_result["deviations_low_kink"]
            lower_threshold = redo_result["lower_threshold"]
            upper_threshold = redo_result["upper_threshold"]

        return kink_current, spd_detected, low_kink_flag

    except Exception as e:
        print(f"Error during kink detection: {e}")
        traceback.print_exc()
        return 0, spd_detected, low_kink_flag, low_kink_flag


# ---------------------------- TRANSFORM --------------------------- #
def transform_raw_liv_file_every_nth_laser_chunked(
    file_url, decoder_file_path, machine_code, wafer_id, sampling_freq=10000, chunksize=10000
):
    print(f"Starting chunked transformation for {wafer_id}...")
    # transform_time_accum = 0
    # read_time_accum = 0

    # Read decoder once
    decoder_df = None
    # decoder_read_time = 0
    if decoder_file_path.exists():
        # start_read_decoder = time.time()
        decoder_df = pd.read_csv(decoder_file_path)
        # decoder_read_time = time.time() - start_read_decoder
        # read_time_accum += decoder_read_time
        # print(f"✓ Decoder file read in {decoder_read_time:.2f} seconds")
        if "YMIN" not in decoder_df.columns or "XMIN" not in decoder_df.columns:
            print("Warning: Decoder file missing YMIN/XMIN columns.")
            raise ValueError("ERROR: Decoder Matching Failed! Perhaps the wrong decoder file was used")
    else:
        print(f"Warning: Decoder file {decoder_file_path} not found.")
        raise ValueError("ERROR: Decoder Matching Failed! Perhaps the wrong decoder name was used")

    # Time the creation of the chunked reader
    # start_read_mainfile = time.time()
    reader = pd.read_csv(file_url, skiprows=19, chunksize=chunksize)
    # mainfile_reader_init_time = time.time() - start_read_mainfile
    # read_time_accum += mainfile_reader_init_time
    # print(f"✓ Main file chunked reader initialized in {mainfile_reader_init_time:.2f} seconds")

    chunk_idx = 0
    while True:
        try:
            # start_chunk_read = time.time()
            chunk = next(reader)
            # chunk_read_time = time.time() - start_chunk_read
            # read_time_accum += chunk_read_time
            # print(f"✓ Chunk {chunk_idx + 1} read in {chunk_read_time:.2f} seconds")

            # start_chunk_transform = time.time()

            chunk = chunk[chunk["TOUCHDOWN"] % sampling_freq == 0]
            if chunk.empty:
                print(f"Chunk {chunk_idx + 1}: No lasers matched the sampling condition.")
                chunk_idx += 1
                continue

            # Extract peak wavelengths
            peak_wavelength_col = "Peak_Wavelength@35mA"
            if peak_wavelength_col in chunk.columns:
                peak_wavelengths = chunk.set_index("TOUCHDOWN")[peak_wavelength_col].to_dict()
            else:
                peak_wavelengths = {}
                print(f"Warning: '{peak_wavelength_col}' column not found in chunk.")

            # Subset Vf/PD columns
            col_names = chunk.columns
            selected_cols = [col for col in col_names if "Vf" in col or "PD" in col]
            df_subset = chunk[selected_cols].copy()
            df_subset.drop(columns=[col for col in df_subset.columns if "Vf@" in col or "PD@" in col], inplace=True)

            # Transpose and split
            df_transposed = df_subset.transpose()
            df_transposed.reset_index(inplace=True)
            df_transposed.columns = ["Label"] + list(range(1, len(df_transposed.columns)))
            df_transposed.loc[-1] = df_transposed.columns
            df_transposed.index += 1
            df_transposed.sort_index(inplace=True)

            df_vf = df_transposed[df_transposed["Label"].str.contains("Vf")].drop(columns=["Label"])
            df_pd = df_transposed[df_transposed["Label"].str.contains("PD")].drop(columns=["Label"])

            n_meas = df_vf.shape[0]
            n_devices = df_vf.shape[1]

            # Concatenate Vf
            df_concat_vf = pd.concat([df_vf[col] for col in df_vf.columns], ignore_index=True).to_frame(name="Vf")
            df_concat_vf["TOUCHDOWN"] = chunk["TOUCHDOWN"].repeat(n_meas).values

            # Concatenate PD
            df_concat_pd = pd.concat([df_pd[col] for col in df_pd.columns], ignore_index=True).to_frame(name="PD")

            # Combine
            df_raw_sweeps = pd.concat([df_concat_vf, df_concat_pd], axis=1)

            # Merge coordinates
            if all(c in chunk.columns for c in ["TOUCHDOWN", "STX_WAFER_X_UM", "STX_WAFER_Y_UM"]):
                df_raw_sweeps = df_raw_sweeps.merge(
                    chunk[["TOUCHDOWN", "STX_WAFER_X_UM", "STX_WAFER_Y_UM"]], on="TOUCHDOWN", how="left"
                )
            else:
                print("Missing coordinate columns in chunk.")

            # Merge decoder info
            if decoder_df is not None:
                df_raw_sweeps = df_raw_sweeps.merge(
                    decoder_df[["YMIN", "XMIN", "TE_LABEL", "TYPE"]],
                    left_on=["STX_WAFER_Y_UM", "STX_WAFER_X_UM"],
                    right_on=["YMIN", "XMIN"],
                    how="left",
                ).drop(columns=["YMIN", "XMIN"], errors="ignore")

            # Final columns
            df_raw_sweeps.rename(columns={"STX_WAFER_X_UM": "X_UM", "STX_WAFER_Y_UM": "Y_UM"}, inplace=True)
            df_raw_sweeps["LDI_mA"] = [i % n_meas + 1 for i in range(len(df_raw_sweeps))]
            df_raw_sweeps.insert(0, "WAFER_ID", wafer_id)
            df_raw_sweeps.insert(0, "MACH", machine_code)

            # chunk_transform_time = time.time() - start_chunk_transform
            # transform_time_accum += chunk_transform_time
            # print(f"✓ Chunk {chunk_idx + 1} transformed in {chunk_transform_time:.2f} seconds")

            yield df_raw_sweeps, n_meas, n_devices, sampling_freq, peak_wavelengths
            chunk_idx += 1

        except StopIteration:
            break

    print(f"\n✓ All chunks transformed for {wafer_id}")
    # print("⏱ Transformation Timing Summary (seconds):")
    # print(f"Total decoder read time:    {decoder_read_time:.2f} seconds")
    # print(f"Main file init read time:   {mainfile_reader_init_time:.2f} seconds")
    # print(f"Total chunk read time:      {read_time_accum - decoder_read_time - mainfile_reader_init_time:.2f} seconds")
    # print(f"Total reading time:         {read_time_accum:.2f} seconds")
    # print(f"Total transformation time:  {transform_time_accum:.2f} seconds\n")


def transform_raw_liv_file_every_nth_laser_chunked_timed(
    file_url, decoder_file_path, machine_code, wafer_id, sampling_freq=10000, chunksize=10000
):
    print(f"Starting chunked transformation for {wafer_id}...")
    transform_time_accum = 0
    read_time_accum = 0

    # Read decoder once
    decoder_df = None
    decoder_read_time = 0
    if decoder_file_path.exists():
        start_read_decoder = time.time()
        decoder_df = pd.read_csv(decoder_file_path)
        decoder_read_time = time.time() - start_read_decoder
        read_time_accum += decoder_read_time
        print(f"✓ Decoder file read in {decoder_read_time:.2f} seconds")
        if "YMIN" not in decoder_df.columns or "XMIN" not in decoder_df.columns:
            print("Warning: Decoder file missing YMIN/XMIN columns.")
            raise ValueError("ERROR: Decoder Matching Failed! Perhaps the wrong decoder file was used")
    else:
        print(f"Warning: Decoder file {decoder_file_path} not found.")
        raise ValueError("ERROR: Decoder Matching Failed! Perhaps the wrong decoder name was used")

    # Time the creation of the chunked reader
    start_read_mainfile = time.time()
    reader = pd.read_csv(file_url, skiprows=19, chunksize=chunksize)
    mainfile_reader_init_time = time.time() - start_read_mainfile
    read_time_accum += mainfile_reader_init_time
    print(f"✓ Main file chunked reader initialized in {mainfile_reader_init_time:.2f} seconds")

    chunk_idx = 0
    while True:
        try:
            start_chunk_read = time.time()
            chunk = next(reader)
            chunk_read_time = time.time() - start_chunk_read
            read_time_accum += chunk_read_time
            # print(f"✓ Chunk {chunk_idx + 1} read in {chunk_read_time:.2f} seconds")

            start_chunk_transform = time.time()

            chunk = chunk[chunk["TOUCHDOWN"] % sampling_freq == 0]
            if chunk.empty:
                print(f"Chunk {chunk_idx + 1}: No lasers matched the sampling condition.")
                chunk_idx += 1
                continue

            # Extract peak wavelengths
            peak_wavelength_col = "Peak_Wavelength@35mA"
            if peak_wavelength_col in chunk.columns:
                peak_wavelengths = chunk.set_index("TOUCHDOWN")[peak_wavelength_col].to_dict()
            else:
                peak_wavelengths = {}
                print(f"Warning: '{peak_wavelength_col}' column not found in chunk.")

            # Subset Vf/PD columns
            col_names = chunk.columns
            selected_cols = [col for col in col_names if "Vf" in col or "PD" in col]
            df_subset = chunk[selected_cols].copy()
            df_subset.drop(columns=[col for col in df_subset.columns if "Vf@" in col or "PD@" in col], inplace=True)

            # Transpose and split
            df_transposed = df_subset.transpose()
            df_transposed.reset_index(inplace=True)
            df_transposed.columns = ["Label"] + list(range(1, len(df_transposed.columns)))
            df_transposed.loc[-1] = df_transposed.columns
            df_transposed.index += 1
            df_transposed.sort_index(inplace=True)

            df_vf = df_transposed[df_transposed["Label"].str.contains("Vf")].drop(columns=["Label"])
            df_pd = df_transposed[df_transposed["Label"].str.contains("PD")].drop(columns=["Label"])

            n_meas = df_vf.shape[0]
            n_devices = df_vf.shape[1]

            # Concatenate Vf
            df_concat_vf = pd.concat([df_vf[col] for col in df_vf.columns], ignore_index=True).to_frame(name="Vf")
            df_concat_vf["TOUCHDOWN"] = chunk["TOUCHDOWN"].repeat(n_meas).values

            # Concatenate PD
            df_concat_pd = pd.concat([df_pd[col] for col in df_pd.columns], ignore_index=True).to_frame(name="PD")

            # Combine
            df_raw_sweeps = pd.concat([df_concat_vf, df_concat_pd], axis=1)

            # Merge coordinates
            if all(c in chunk.columns for c in ["TOUCHDOWN", "STX_WAFER_X_UM", "STX_WAFER_Y_UM"]):
                df_raw_sweeps = df_raw_sweeps.merge(
                    chunk[["TOUCHDOWN", "STX_WAFER_X_UM", "STX_WAFER_Y_UM"]], on="TOUCHDOWN", how="left"
                )
            else:
                print("Missing coordinate columns in chunk.")

            # Merge decoder info
            if decoder_df is not None:
                df_raw_sweeps = df_raw_sweeps.merge(
                    decoder_df[["YMIN", "XMIN", "TE_LABEL", "TYPE"]],
                    left_on=["STX_WAFER_Y_UM", "STX_WAFER_X_UM"],
                    right_on=["YMIN", "XMIN"],
                    how="left",
                ).drop(columns=["YMIN", "XMIN"], errors="ignore")

            # Final columns
            df_raw_sweeps.rename(columns={"STX_WAFER_X_UM": "X_UM", "STX_WAFER_Y_UM": "Y_UM"}, inplace=True)
            df_raw_sweeps["LDI_mA"] = [i % n_meas + 1 for i in range(len(df_raw_sweeps))]
            df_raw_sweeps.insert(0, "WAFER_ID", wafer_id)
            df_raw_sweeps.insert(0, "MACH", machine_code)

            chunk_transform_time = time.time() - start_chunk_transform
            transform_time_accum += chunk_transform_time
            # print(f"✓ Chunk {chunk_idx + 1} transformed in {chunk_transform_time:.2f} seconds")

            yield df_raw_sweeps, n_meas, n_devices, sampling_freq, peak_wavelengths
            chunk_idx += 1

        except StopIteration:
            break

    print(f"\n✓ All chunks transformed for {wafer_id}")
    print("⏱ Transformation Timing Summary (seconds):")
    print(f"Total decoder read time:    {decoder_read_time:.2f} seconds")
    print(f"Main file init read time:   {mainfile_reader_init_time:.2f} seconds")
    print(f"Total chunk read time:      {read_time_accum - decoder_read_time - mainfile_reader_init_time:.2f} seconds")
    print(f"Total reading time:         {read_time_accum:.2f} seconds")
    print(f"Total transformation time:  {transform_time_accum:.2f} seconds\n")


# ----------------------------- PROCESS ---------------------------- #
def stream_process_lasing_parameters(
    filepath,
    summary_output_path,
    wafer_code,
    machine_code,
    decoder_df,
    sampling_freq=400,
    chunksize=10000,
    export_buffer=10000,
):

    accumulator = {}
    summary_records = []

    # Timers
    # t_flag_no_laser = 0
    # t_ith = 0
    # t_slope = 0
    # t_series_r = 0
    # t_kink = 0
    # t_export = 0
    # t_groupby = 0

    for chunk, n_meas, _, _, peak_wavelengths in transform_raw_liv_file_every_nth_laser_chunked(
        filepath, decoder_df, machine_code, wafer_code, sampling_freq, chunksize
    ):
        # start_groupby = time.time()  # ⬅️ START
        grouped = chunk.groupby("TE_LABEL")
        # t_groupby += time.time() - start_groupby  # ⬅️ END

        for te_label, group in grouped:
            if te_label not in accumulator:
                accumulator[te_label] = group.copy()
            else:
                accumulator[te_label] = pd.concat([accumulator[te_label], group], ignore_index=True)

            if len(accumulator[te_label]) >= n_meas:
                full_data = accumulator[te_label]
                touchdown = full_data["TOUCHDOWN"].iloc[0] if "TOUCHDOWN" in full_data else ""

                current = full_data["LDI_mA"].values
                intensity = full_data["PD"].values
                voltage = full_data["Vf"].values

                # start = time.time()
                no_laser_flag = flag_no_laser(intensity)
                # t_flag_no_laser += time.time() - start

                flag = "NO LASER" if no_laser_flag else np.nan

                # start = time.time()
                ith = find_ith_value_labview(intensity, current)
                # t_ith += time.time() - start

                # start = time.time()
                slope = find_slope_efficiency(intensity, current, ith) if ith else 0
                # t_slope += time.time() - start

                # start = time.time()
                series_r = find_series_resistance(voltage, current, ith) if ith else 0
                # t_series_r += time.time() - start

                peak_wl = peak_wavelengths.get(touchdown, np.nan) if peak_wavelengths else np.nan

                # start = time.time()
                kink_current, spd_detected, low_kink_flag = find_kink(intensity, current, ith)
                # t_kink += time.time() - start

                record = {
                    "WAFER_ID": wafer_code,
                    "MACH": full_data["MACH"].iloc[0] if "MACH" in full_data else "",
                    "TE_LABEL": te_label,
                    "TOUCHDOWN": touchdown,
                    "I_THRESHOLD": ith,
                    "SLOPE_EFFICIENCY": slope,
                    "SERIES_RESISTANCE": series_r,
                    "PEAK_WAVELENGTH_35mA": peak_wl,
                    "KINK1": kink_current,
                    "TYPE": full_data["TYPE"].iloc[0],
                    "X_UM": full_data["X_UM"].iloc[0],
                    "Y_UM": full_data["Y_UM"].iloc[0],
                    "FLAG": flag,
                }
                summary_records.append(record)

                del accumulator[te_label]

                if len(summary_records) >= export_buffer:
                    # start = time.time()
                    pd.DataFrame(summary_records).to_csv(
                        Path(summary_output_path), mode="a", header=not summary_output_path.exists(), index=False
                    )
                    # t_export += time.time() - start
                    summary_records.clear()

    print(f"Data Exporting to: {summary_output_path}")
    if summary_records:
        # start = time.time()
        pd.DataFrame(summary_records).to_csv(
            summary_output_path, mode="a", header=not summary_output_path.exists(), index=False
        )
        # t_export += time.time() - start

    print(f"✓ Device summary exported for {wafer_code}")

    # total_proc_time = t_export + t_kink + t_series_r + t_slope + t_ith + t_flag_no_laser + t_groupby

    # # Summary of time spent
    # print("\n⏱ Processing Timing Summary (seconds):")
    # print(f"  groupby:             {t_groupby:.3f}")
    # print(f"  flag_no_laser:       {t_flag_no_laser:.3f}")
    # print(f"  find_ith:            {t_ith:.3f}")
    # print(f"  find_slope_eff:      {t_slope:.3f}")
    # print(f"  find_series_res:     {t_series_r:.3f}")
    # print(f"  find_kink:           {t_kink:.3f}")
    # print(f"  CSV Exporting:       {t_export:.3f}")
    # print(f"  Total Processing:    {total_proc_time:.3f}\n")


def stream_process_lasing_parameters_timed(
    filepath,
    summary_output_path,
    wafer_code,
    machine_code,
    decoder_df,
    sampling_freq=250,
    chunksize=10000,
    export_buffer=10000,
):

    accumulator = {}
    summary_records = []
    t_ith_breakdown = defaultdict(float)

    # Timers
    t_flag_no_laser = 0
    t_ith = 0
    t_slope = 0
    t_series_r = 0
    t_kink = 0
    t_export = 0
    t_groupby = 0

    for chunk, n_meas, _, _, peak_wavelengths in transform_raw_liv_file_every_nth_laser_chunked_timed(
        filepath, decoder_df, machine_code, wafer_code, sampling_freq, chunksize
    ):
        start_groupby = time.time()  # ⬅️ START
        grouped = chunk.groupby("TE_LABEL")
        t_groupby += time.time() - start_groupby  # ⬅️ END

        for te_label, group in grouped:
            if te_label not in accumulator:
                accumulator[te_label] = group.copy()
            else:
                accumulator[te_label] = pd.concat([accumulator[te_label], group], ignore_index=True)

            if len(accumulator[te_label]) >= n_meas:
                full_data = accumulator[te_label]
                touchdown = full_data["TOUCHDOWN"].iloc[0] if "TOUCHDOWN" in full_data else ""

                current = full_data["LDI_mA"].values
                intensity = full_data["PD"].values
                voltage = full_data["Vf"].values

                start = time.time()
                no_laser_flag = flag_no_laser(intensity)
                t_flag_no_laser += time.time() - start

                flag = "NO LASER" if no_laser_flag else np.nan

                start = time.time()
                ith, ith_timings = find_ith_value_labview_timed(intensity, current)
                t_ith += time.time() - start

                # Accumulate internal timings only if ith_timings is valid
                if isinstance(ith_timings, dict) and all(isinstance(v, (int, float)) for v in ith_timings.values()):
                    for key, dt in ith_timings.items():
                        t_ith_breakdown[key] += dt

                start = time.time()
                slope = find_slope_efficiency(intensity, current, ith) if ith else 0
                t_slope += time.time() - start

                start = time.time()
                series_r = find_series_resistance(voltage, current, ith) if ith else 0
                t_series_r += time.time() - start

                peak_wl = peak_wavelengths.get(touchdown, np.nan) if peak_wavelengths else np.nan

                start = time.time()
                kink_current, spd_detected, low_kink_flag = find_kink(intensity, current, ith)
                t_kink += time.time() - start

                record = {
                    "WAFER_ID": wafer_code,
                    "MACH": full_data["MACH"].iloc[0] if "MACH" in full_data else "",
                    "TE_LABEL": te_label,
                    "TOUCHDOWN": touchdown,
                    "I_THRESHOLD": ith,
                    "SLOPE_EFFICIENCY": slope,
                    "SERIES_RESISTANCE": series_r,
                    "PEAK_WAVELENGTH_35mA": peak_wl,
                    "KINK1": kink_current,
                    "TYPE": full_data["TYPE"].iloc[0],
                    "X_UM": full_data["X_UM"].iloc[0],
                    "Y_UM": full_data["Y_UM"].iloc[0],
                    "FLAG": flag,
                }
                summary_records.append(record)

                del accumulator[te_label]

                if len(summary_records) >= export_buffer:
                    start = time.time()
                    pd.DataFrame(summary_records).to_csv(
                        Path(summary_output_path), mode="a", header=not summary_output_path.exists(), index=False
                    )
                    t_export += time.time() - start
                    summary_records.clear()

    print(f"Data Exporting to: {summary_output_path}")
    if summary_records:
        start = time.time()
        pd.DataFrame(summary_records).to_csv(
            summary_output_path, mode="a", header=not summary_output_path.exists(), index=False
        )
        t_export += time.time() - start

    print(f"✓ Device summary exported for {wafer_code}")

    total_proc_time = t_export + t_kink + t_series_r + t_slope + t_ith + t_flag_no_laser + t_groupby

    # Summary of time spent
    print("\n⏱ Processing Timing Summary (seconds):")
    print(f"  groupby:             {t_groupby:.3f}")
    print(f"  flag_no_laser:       {t_flag_no_laser:.3f}")
    print(f"  find_ith:            {t_ith:.3f}")
    print(f"  find_slope_eff:      {t_slope:.3f}")
    print(f"  find_series_res:     {t_series_r:.3f}")
    print(f"  find_kink:           {t_kink:.3f}")
    print(f"  CSV Exporting:       {t_export:.3f}")
    print(f"  Total Processing:    {total_proc_time:.3f}\n")

    print("Ith internal breakdown:")
    for k, v in t_ith_breakdown.items():
        print(f"  {k:16s}: {v:.4f} s")


for record in wafer_records:
    filepath = record["file_url"]
    wafer_code = record["wafer_code"]
    machine_code = record["machine"]
    product_code = wafer_code[:2]
    if product_code in ["QD", "NV", "NK", "NE"]:
        print("Halo Wafer Type Detected")
        decoder_path = ROOT_DIR / "decoders" / HALO_DECODER
    else:
        print("Subaru Wafer Type Detected")
        decoder_path = ROOT_DIR / "decoders" / SUBARU_DECODER

    summary_output_path = EXPORTS_FILEPATH / f"{ANALYSIS_RUN_NAME}_{wafer_code}_device_summary.csv"

    start = time.time()
    stream_process_lasing_parameters_timed(
        filepath,
        summary_output_path,
        wafer_code,
        machine_code,
        decoder_path,
    )
    print(f"Total processing time for {wafer_code}: {time.time() - start:.2f} seconds.\n\n")
