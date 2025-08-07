import sys
import time
import os
import traceback
import io
from pathlib import Path
import configparser
from datetime import datetime

import pandas as pd
import numpy as np

import scipy.stats as st
from scipy.signal import savgol_filter
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from sklearn.linear_model import RANSACRegressor, LinearRegression

from tqdm import tqdm
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading


CURRENT_DIR = Path(os.getcwd())
ROOT_DIR = CURRENT_DIR  # Adjust the number based on your folder structure
sys.path.append(str(ROOT_DIR))  # Add the root directory to the system path

EXPORTS_FILEPATH = ROOT_DIR / "exports"
if not os.path.exists(EXPORTS_FILEPATH):
    os.makedirs(EXPORTS_FILEPATH)

RESULTS_PATH = ROOT_DIR / "results"
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

SUBARU_DECODER = "QC WAFER_LAYOUT 24Dec.csv"
HALO_DECODER = "HALO_DECODER_NE-rev1_1 logic_coords_annotated.csv"
MONITORED_PATH = ROOT_DIR / "monitored_folder"
LOG_PATH = RESULTS_PATH / "detection_log.txt"
INI_NAME = "PROBEINF.ini"
INI_LOCATION = ROOT_DIR / "dummy_location" / INI_NAME
ANALYSIS_RUN_NAME = "python"  # Replace or parameterize as needed
VERSION = 1.0


# ------------------------------------------------------------------ #
#                       Photonic Analysis Code                       #
# ------------------------------------------------------------------ #
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


# ------------------------------- ITH ------------------------------ #
def gaussian(x, a, x0, sigma):
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def gaussian_residuals(p, x, y):
    return gaussian(x, *p) - y


def find_ith_value(
    intensity,
    current,
    use_ransac_right=True,
    ransac_threshold=0.1,
    mu_bound_window=5,
    fit_window=10,
    sigma_guess_default=0.7,
    fit_skip_threshold=1e-3,
    trim_last_points=True,
):

    # 1) Trim data to only include values >2 and <=35 mA
    mask_trimmed = (current > 1) & (current <= 34)
    if not np.any(mask_trimmed):
        # print("Warning: No data points between 2 and 35 mA.")
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

    # # 5) Apply Chebyshev high-pass filter (order 2, ripple 0.1 dB, bandpass 0.15–0.45)
    # b, a = cheby1(N=2, rp=0.1, Wn=0.45, btype="lowpass", fs=1)
    # filtered_intensity = filtfilt(b, a, intensity_norm)

    # # 5a) Initial linear fit via least absolute residuals on filtered data using QuantReg
    # X = sm.add_constant(current)  # Adds intercept term
    # model = sm.QuantReg(filtered_intensity, X)
    # res = model.fit(q=0.5)
    # slope_left = res.params[1]
    # intercept_left = res.params[0]
    # initial_abs_residual_total = np.mean(np.abs(filtered_intensity - res.predict(X)))

    # # print(initial_abs_residual_total)
    # if initial_abs_residual_total > 1:
    #     print("Warning: Initial L1 fit residual too high.")
    #     return 0  # DEBUG: STILL TRY

    # # 6) First Savitzky-Golay smoothing (5,1)
    # smoothed_intensity = savgol_filter(filtered_intensity, window_length=5, polyorder=1)

    # # 7) Second Savitzky-Golay smoothing before differential (3,2)
    # smoothed_intensity = savgol_filter(smoothed_intensity, window_length=3, polyorder=2)

    # 8) Compute first derivative (renamed to dL_dI)
    dL_dI = np.gradient(intensity_norm, current)

    # # 9) Smooth first derivative (6,2)
    # smoothed_dL_dI = savgol_filter(dL_dI, window_length=6, polyorder=2)

    # 10) Compute second derivative (renamed to d2L_dI2)
    d2L_dI2 = np.gradient(dL_dI, current)

    # 11) Smooth second derivative (6,2)
    smoothed_d2L_dI2 = savgol_filter(d2L_dI2, window_length=6, polyorder=2)

    # 11a) Set negative second derivative values to zero (LabVIEW-like behavior)
    smoothed_d2L_dI2[smoothed_d2L_dI2 < 0] = 0

    # 12) Normalize second derivative and add 0.01
    max_d2L_dI2 = np.max(smoothed_d2L_dI2)
    if max_d2L_dI2 == 0:
        # print("Warning: Second derivative all zero after zeroing negatives.")
        return 0
    d2L_dI2_ready = (smoothed_d2L_dI2 / max_d2L_dI2) + 0.01

    # Step 13a: Peak detection
    # --- New logic: Find all peaks above 0.95 ---
    peaks, _ = find_peaks(d2L_dI2_ready, height=0.95)
    if len(peaks) == 0:
        # print("Warning: No significant peaks in second derivative.")
        return 0
    selected_peak_idx = peaks[np.argmin(current[peaks])]

    # Step 13a: Subsampling Algorithm
    a_guess = d2L_dI2_ready[selected_peak_idx]
    sigma_guess = sigma_guess_default

    if selected_peak_idx <= 0 or selected_peak_idx >= len(current) - 1:
        x0_guess = current[selected_peak_idx]  # Edge case: fallback to raw peak index
    else:
        x_peak = current[selected_peak_idx]
        y_peak = d2L_dI2_ready[selected_peak_idx]

        x_left = current[selected_peak_idx - 1]
        x_right = current[selected_peak_idx + 1]
        y_left = d2L_dI2_ready[selected_peak_idx - 1]
        y_right = d2L_dI2_ready[selected_peak_idx + 1]

        delta_left = abs(y_peak - y_left)
        delta_right = abs(y_peak - y_right)

        # Pick side with smaller delta (closer in value to the peak)
        if delta_left < delta_right:
            # Interpolate between left and center
            x_between = (x_left + x_peak) / 2
            x_spacing = abs(x_peak - x_left)
            interp_ratio = delta_left / (delta_left + delta_right + 1e-12)
            x0_guess = x_between + interp_ratio * x_spacing
        elif delta_right < delta_left:
            # Interpolate between right and center
            x_between = (x_right + x_peak) / 2
            x_spacing = abs(x_right - x_peak)
            interp_ratio = delta_right / (delta_left + delta_right + 1e-12)
            x0_guess = x_between - interp_ratio * x_spacing
        else:
            interp_ratio = 0
            x0_guess = x_peak

        # Optional: print diagnostic
        # print(f"[x0 refined guess] left Δ={delta_left:.4f}, right Δ={delta_right:.4f}, interp_ratio={interp_ratio:.4f}")

    if np.max(d2L_dI2_ready) < 0.9:
        return 0

    # Step 13b: Gaussian fit
    initial_guess = [a_guess, x0_guess, sigma_guess]
    bounds = (
        (0.9, 1.011),
        ((x0_guess - mu_bound_window, x0_guess + mu_bound_window)),
        (0.5, 1.5),
    )

    fit_mask = (current >= x0_guess - fit_window) & (current <= x0_guess + fit_window)
    if not np.any(fit_mask):
        return 0
    current_fit = current[fit_mask]
    d2L_dI2_fit = d2L_dI2_ready[fit_mask]

    # Step 13b.1: Try default Gaussian guess and calculate MSE
    default_fit = gaussian(current_fit, *initial_guess)
    default_mse = np.mean((d2L_dI2_fit - default_fit) ** 2)

    # Step 13b.2 If sufficient, use guess. If not use curve fit
    if default_mse < fit_skip_threshold:
        # Skip least_squares – use initial guess directly
        popt = initial_guess
    else:
        # Fit with least_squares
        try:
            res = least_squares(
                gaussian_residuals,
                x0=initial_guess,
                bounds=([b[0] for b in bounds], [b[1] for b in bounds]),
                args=(current_fit, d2L_dI2_fit),
                max_nfev=100,
                method="trf",
            )
            popt = res.x
        except Exception:
            # print("Warning: least squares failed")
            return 0, {"error": "least_squares_failed"}

    median_x = popt[1]

    # 14) Validate split point
    if not (2 <= median_x <= 35):
        # print("Warning: Gaussian fit split point out of usual bounds.")
        return 0

    # 15) Linear fit on left segment
    left_mask = current <= median_x
    if not np.any(left_mask):
        # print("Warning: No data points on the left segment for fitting.")
        return 0
    current_left = current[left_mask]
    intensity_left = intensity_norm[left_mask]
    A_left = np.vstack([current_left, np.ones_like(current_left)]).T
    solution_left, _, _, _ = np.linalg.lstsq(A_left, intensity_left, rcond=None)
    slope_left, intercept_left = solution_left

    # 16) Linear fit on right segment
    right_mask = current > median_x
    if not np.any(right_mask):
        # print("Warning: No data points on the right segment for fitting.")
        return 0
    current_right = current[right_mask]
    intensity_right = intensity_norm[right_mask]

    # Optionally remove last N points to avoid noisy tail
    if trim_last_points:
        N = 15
        current_right = current_right[:-N]
        intensity_right = intensity_right[:-N]
    if len(current_right) < 10:
        # print("Warning: Fewer than 10 data points for stimulated emission fit.")
        return 0

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

    if mse_right > 3:
        # print("Warning: High MSE in stimulated emission fit.")
        return 0

    # 17) Compute intersection (I_th)
    ith_value = (intercept_right - intercept_left) / (slope_left - slope_right)
    if not (2 <= ith_value <= 35):
        # print("Warning: Computed I_th outside bounds normal bounds.")
        return 0  # DEBUG: STILL TRY

    return ith_value


# ------------------------------- SE ------------------------------- #
def find_slope_efficiency(intensity, current, ith, mse_threshold=1, fitting_range=15):
    intensity = np.asarray(intensity)
    current = np.asarray(current)

    if len(intensity) != len(current):
        # print("Warning: Input arrays have different lengths.")
        return 0

    # Index where current just exceeds I_th
    idx_start = np.argmax(current >= ith)
    idx_end = idx_start + fitting_range

    if idx_end > len(current):
        # print("Warning: Not enough points after I_th for slope efficiency fit.")
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
                # print(f"Warning: High MSE in slope efficiency fit ({mse:.4f}).")
                return 0

        return slope

    except Exception as e:
        # print(f"Error during slope efficiency calculation: {e}")
        return 0


# ------------------------------- RS ------------------------------- #
def find_series_resistance(voltage, current, ith, mse_threshold=0.5, fitting_range=15):
    voltage = np.asarray(voltage)
    current = np.asarray(current)
    if len(voltage) != len(current):
        # print("Warning: Input arrays have different lengths.")
        return 0

    # Index where current just exceeds I_th
    idx_start = np.argmax(current >= ith)
    idx_end = idx_start + fitting_range
    if idx_end > len(current):
        # print("Warning: Not enough points after I_th for series resistance fit.")
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
                # print(f"Warning: High MSE in series resistance fit ({mse:.4f}).")
                return 0
        rs_ohms = slope * 1000  # converting to ohms
        return rs_ohms  # Rs = dV/dI
    except Exception as e:
        # print(f"Error during series resistance calculation: {e}")
        return 0


# ------------------------------- SPD ------------------------------ #
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


def find_spd(
    current,
    intensity,
    voltage,
    spd_delta_threshold=1,
    no_laser_flag=False,
    peak_power_threshold=1,
    rollover_threshold=200,
    noise_threshold=25,
):

    assert len(current) == len(intensity) == len(voltage), "Input arrays must be same length."

    # Use diff-style gradient calculation
    delta_I = np.diff(current)
    dP_dI = np.diff(intensity) / delta_I
    dV_dI = np.diff(voltage) / delta_I

    current_mid = current[1:]  # Midpoints for diff-based gradients

    mean_power = np.mean(intensity)
    peak_power = np.max(intensity)
    current_at_peak_power = current[np.argmax(intensity)]
    max_current = np.max(current)

    delta_spd = mean_power - np.abs(dP_dI)
    min_delta = np.min(delta_spd)
    idx_min_delta = np.argmin(delta_spd)
    current_at_spd = current_mid[idx_min_delta]

    # Find the first index where delta_spd < threshold
    first_spd_idx = np.argmax(delta_spd < spd_delta_threshold) if np.any(delta_spd < spd_delta_threshold) else None
    first_spd_current = current_mid[first_spd_idx] if first_spd_idx is not None else max_current

    if min_delta > spd_delta_threshold:
        spd_eval = "ROLLOVER"
    elif peak_power < peak_power_threshold:
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
        current_at_spd = max_current
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
            "first_spd_current": first_spd_current,
            "peak_power": peak_power,
            "current_at_peak_power": current_at_peak_power,
            "mean_power": mean_power,
            "pot_failmode": pot_failmode,
            "voltage_noise": voltage_noise,
        }
    )

    return result


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
    voltage,
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
    first_spd_current = 0
    spd_current = 0

    intensity = np.asarray(intensity)
    current = np.asarray(current)
    max_current = np.max(current)

    if len(intensity) != len(current):
        print("Warning: Input arrays have different lengths.")
        return {
            "kink_current": 0,
            "spd_detected": spd_detected,
            "spd_current": spd_current,
            "first_spd_current": first_spd_current,
            "low_kink_flag": low_kink_flag,
        }

    # ---------------- Remove Points before I_threshold ---------------- #
    mask = current >= ith
    if not np.any(mask):
        print("Warning: No current values above the threshold found.")
        return {
            "kink_current": 0,
            "spd_detected": spd_detected,
            "spd_current": spd_current,
            "first_spd_current": first_spd_current,
            "low_kink_flag": low_kink_flag,
        }
    idx_start = np.argmax(mask)

    I_raw = current[idx_start:]
    L_raw = intensity[idx_start:]
    V = voltage[idx_start:]

    if len(L_raw) < trim_boundary_points + 4:
        print("Warning: Not enough points after Ith for analysis.")
        return {
            "kink_current": 0,
            "spd_detected": spd_detected,
            "spd_current": spd_current,
            "first_spd_current": first_spd_current,
            "low_kink_flag": low_kink_flag,
        }

    # # ----------------- Compute delta for SPD filtering ---------------- #
    # dLdI_temp = np.gradient(L_raw, I_raw)
    # delta = np.mean(L_raw) - np.abs(dLdI_temp)
    # min_delta = np.min(delta)
    # max_pd = np.max(L_raw)

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
    spd_results = find_spd(current, intensity, voltage, spd_delta_threshold=threshold_min_delta)
    spd_eval = spd_results["spd_eval"]
    first_spd_current = spd_results["first_spd_current"]
    spd_current = spd_results["current_at_spd"]
    if spd_eval not in ["COD", "COD-PR"]:
        first_spd_current = max_current
        spd_current = max_current

    if spd_eval in ["COD", "COD-PR"]:
        # Find index in I array where current >= first_spd_current
        cutoff_idx = np.argmax(I >= spd_current)
        I_non_spd = I[:cutoff_idx]
        L_non_spd = L[:cutoff_idx]
        dLdI_non_spd = dLdI[:cutoff_idx]
        d2LdI2_non_spd = d2LdI2[:cutoff_idx]
        spd_detected = True
        # print("SPD DETECTED: TRIMMING ARRAYS")
    else:
        I_non_spd = I
        L_non_spd = L
        dLdI_non_spd = dLdI
        d2LdI2_non_spd = d2LdI2

    if len(I_non_spd) < trim_boundary_points + 1:
        print("Warning: Not enough points after SPD filtering.")
        return {
            "kink_current": 0,
            "spd_detected": spd_detected,
            "spd_current": spd_current,
            "first_spd_current": first_spd_current,
            "low_kink_flag": low_kink_flag,
        }

    # -------------- Hardcoded Trimming of Boundary Points -------------- #
    n_points = len(I_non_spd)
    if n_points < trim_boundary_points + trim_end_points + 3:
        print("Warning: Not enough points after trimming for fit.")
        return {
            "kink_current": 0,
            "spd_detected": spd_detected,
            "spd_current": spd_current,
            "first_spd_current": first_spd_current,
            "low_kink_flag": low_kink_flag,
        }

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

        return {
            "kink_current": kink_current,
            "spd_detected": spd_detected,
            "spd_current": spd_current,
            "first_spd_current": first_spd_current,
            "low_kink_flag": low_kink_flag,
        }

    except Exception as e:
        print(f"Error during kink detection: {e}")
        traceback.print_exc()
        return {
            "kink_current": 0,
            "spd_detected": spd_detected,
            "spd_current": spd_current,
            "first_spd_current": first_spd_current,
            "low_kink_flag": low_kink_flag,
        }


def find_kink_fast_spd(
    intensity,
    current,
    voltage,
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
    spd_current = 63

    intensity = np.asarray(intensity)
    current = np.asarray(current)

    if len(intensity) != len(current):
        # print("Warning: Input arrays have different lengths.")
        return {
            "kink_current": 0,
            "spd_detected": spd_detected,
            "spd_current": spd_current,
            "low_kink_flag": low_kink_flag,
        }

    # ---------------- Remove Points before I_threshold ---------------- #
    mask = current >= ith
    if not np.any(mask):
        # print("Warning: No current values above the threshold found.")
        return {
            "kink_current": 0,
            "spd_detected": spd_detected,
            "spd_current": spd_current,
            "low_kink_flag": low_kink_flag,
        }
    idx_start = np.argmax(mask)

    I_raw = current[idx_start:]
    L_raw = intensity[idx_start:]
    V = voltage[idx_start:]

    if len(L_raw) < trim_boundary_points + 1:
        # print("Warning: Not enough points after Ith for analysis.")
        return {
            "kink_current": 0,
            "spd_detected": spd_detected,
            "spd_current": spd_current,
            "low_kink_flag": low_kink_flag,
        }

    # # ----------------- Compute delta for SPD filtering ---------------- #
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
        spd_current = I[cutoff_idx]  # <-- Set spd_current to first current in cutoff region
        I_non_spd = I[:cutoff_idx]
        L_non_spd = L[:cutoff_idx]
        dLdI_non_spd = dLdI[:cutoff_idx]
        d2LdI2_non_spd = d2LdI2[:cutoff_idx]
        spd_detected = True
        # print("SPD DETECTED: TRIMMING ARRAYS")
    else:
        I_non_spd = I
        L_non_spd = L
        dLdI_non_spd = dLdI
        d2LdI2_non_spd = d2LdI2

    if len(I_non_spd) < trim_boundary_points + 1:
        # print("Warning: Not enough points after SPD filtering.")
        return {
            "kink_current": 0,
            "spd_detected": spd_detected,
            "spd_current": spd_current,
            "low_kink_flag": low_kink_flag,
        }

    # -------------- Hardcoded Trimming of Boundary Points -------------- #
    n_points = len(I_non_spd)
    if n_points < trim_boundary_points + trim_end_points + 4:
        # print("Warning: Not enough points after trimming for fit.")
        return {
            "kink_current": 0,
            "spd_detected": spd_detected,
            "spd_current": spd_current,
            "low_kink_flag": low_kink_flag,
        }

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
        kink_current = I_trimmed[kink_mask][0] if np.any(kink_mask) else 63  # IF NO ERROR: KINK SET TO 63

        # ----------------------- Handle early kinks ----------------------- #
        kink_index = np.argmax(kink_mask) if np.any(kink_mask) else None
        # print(f"kink_index = {kink_index}")
        early_deviations = deviations[:trim_boundary_points]
        mean_early_deviations = np.mean(early_deviations)
        # print(f"mean_early_deviations = {mean_early_deviations}")

        if kink_index is not None and kink_index <= 5 or mean_early_deviations > early_kink_threshold:
            # print("Warning: Low Kink Detected, Using Low Kink Countermeasures")
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

        return {
            "kink_current": kink_current,
            "spd_detected": spd_detected,
            "spd_current": spd_current,
            "low_kink_flag": low_kink_flag,
        }

    except Exception as e:
        # print(f"Error during kink detection: {e}")
        traceback.print_exc()
        return {
            "kink_current": 0,
            "spd_detected": spd_detected,
            "spd_current": spd_current,
            "low_kink_flag": low_kink_flag,
        }


# ------------------------------------------------------------------ #
#                   Laser Processing Pipeline Code                   #
# ------------------------------------------------------------------ #
def transform_raw_liv_file_every_nth_laser_chunked(
    file_url, decoder_file_path, machine_code, wafer_id, sampling_freq=10000, chunksize=10000
):
    # print(f"Starting chunked transformation for {wafer_id}...")
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
            tqdm.write("Warning: Decoder file missing YMIN/XMIN columns.")
            raise ValueError("ERROR: Decoder Matching Failed! Perhaps the wrong decoder file was used")
    else:
        tqdm.write(f"Warning: Decoder file {decoder_file_path} not found.")
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
                tqdm.write(f"Chunk {chunk_idx + 1}: No lasers matched the sampling condition.")
                chunk_idx += 1
                continue

            # Extract peak wavelengths
            peak_wavelength_col = "Peak_Wavelength@35mA"
            if peak_wavelength_col in chunk.columns:
                peak_wavelengths = chunk.set_index("TOUCHDOWN")[peak_wavelength_col].to_dict()
            else:
                peak_wavelengths = {}
                tqdm.write(f"Warning: '{peak_wavelength_col}' column not found in chunk.")

            # Extract Leakage Current
            preferred_columns = ["Leakage_current", "Leakage_current (uA)"]
            leakage_currents = {}

            for col in preferred_columns:
                if col in chunk.columns:
                    leakage_currents = chunk.set_index("TOUCHDOWN")[col].to_dict()
                    break
            else:
                tqdm.write(f"Warning: None of the expected leakage current columns found: {preferred_columns}")

            # Extract Reverse Voltage
            reverse_voltage_col = "Reverse_voltage"
            if reverse_voltage_col in chunk.columns:
                reverse_voltages = chunk.set_index("TOUCHDOWN")[reverse_voltage_col].to_dict()
            else:
                reverse_voltages = {}
                tqdm.write(f"Warning: '{reverse_voltage_col}' column not found in chunk.")

            # Extract Cube Numbers
            cube_num_col = "CUBE"
            if cube_num_col in chunk.columns:
                cube_numbers = chunk.set_index("TOUCHDOWN")[cube_num_col].to_dict()
            else:
                cube_numbers = {}
                tqdm.write(f"Warning: '{cube_num_col}' column not found in chunk.")

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
                tqdm.write("Missing coordinate columns in chunk.")

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

            yield {
                "chunk": df_raw_sweeps,
                "n_meas": n_meas,
                "n_devices": n_devices,
                "sampling_freq": sampling_freq,
                "peak_wavelengths": peak_wavelengths,
                "leakage_currents": leakage_currents,
                "reverse_voltages": reverse_voltages,
                "cube_numbers": cube_numbers,
            }
            chunk_idx += 1

        except StopIteration:
            break

    tqdm.write(f"\n✓ All chunks transformed for {wafer_id}")
    # print("⏱ Transformation Timing Summary (seconds):")
    # print(f"Total decoder read time:    {decoder_read_time:.2f} seconds")
    # print(f"Main file init read time:   {mainfile_reader_init_time:.2f} seconds")
    # print(f"Total chunk read time:      {read_time_accum - decoder_read_time - mainfile_reader_init_time:.2f} seconds")
    # print(f"Total reading time:         {read_time_accum:.2f} seconds")
    # print(f"Total transformation time:  {transform_time_accum:.2f} seconds\n")


def stream_process_lasing_parameters(
    filepath,
    summary_output_path,
    dat_output_path,
    ini_data,
    decoder_path,
    sampling_freq=10000,
    chunksize=10000,
    export_buffer=1000,
):

    accumulator = {}
    summary_records = []
    melted_records = []
    dat_file_exists = dat_output_path.exists()
    total_processed = 0  # 👈 Counter for total exported rows
    total_exported_rows = 0  # 👈 Counter for total exported rows
    pt_counter = 0  # 👈 Counter for melted .dat file rows

    # ⬇️ Unpack ini_data dictionary
    INI_LOT = ini_data["INI_LOT"]
    wafer_code = INI_LOT
    INI_WAFER = ini_data["INI_WAFER"]
    INI_OP = ini_data["INI_OP"]
    INI_STAGE = ini_data["INI_STAGE"]
    INI_STEP = ini_data["INI_STEP"]
    INI_PROD = ini_data["INI_PROD"]
    product_code = INI_PROD
    INI_MACH = ini_data["INI_MACH"]
    machine_code = INI_MACH
    INI_EMPID = ini_data["INI_EMPID"]
    INI_MEAS_TYPE = ini_data["INI_MEAS_TYPE"]
    INI_PDATETIME = ini_data["INI_PDATETIME"]
    INI_XDIVIDING_FACTOR = ini_data["INI_XDIVIDING_FACTOR"]
    INI_YDIVIDING_FACTOR = ini_data["INI_YDIVIDING_FACTOR"]
    FAC = ini_data.get("FAC", "STST")  # default fallback
    CARD = ini_data.get("CARD", 0)
    COND = ini_data.get("COND", "Py_v1.0")
    STRUCT = ini_data.get("STRUCT", "L")
    PRODUCT_TYPE = ini_data["PRODUCT_TYPE"]
    UNIT_DICT = ini_data["UNIT_DICT"]

    columns_local = [
        "WAFER_ID",
        "WAFER_TYPE",
        "MACH_NUM",
        "LIV_ANALYZER_VERSION",
        "TE_LABEL",
        "TOUCHDOWN",
        "ITH_MA",
        "SLOPE_EFF_MW_MA",
        "RS_FIT_OHMS",
        "PEAK_WAVE_AT35_NM",
        "CURRENT_AT_SPD",
        "KINK1_CURRENT_MA",
        "LEAK_CURR_UA",
        "REVERSE_VOLTS_V",
        "CUBE_NUM",
        "TYPE",
        "STX_WAFER_X_UM",
        "STX_WAFER_Y_UM",
        "FLAG_LOW_POUT",
    ]

    columns_levee = [
        "PDATETIME",
        "LOT",
        "WAFER",
        "OP",
        "STAGE",
        "STEP",
        "PROD",
        "MACH",
        "FAC",
        "CARD",
        "EMPID",
        "FLD",
        "MEAS_TYPE",
        "COND",
        "PT",
        "PT_LOC_X",
        "PT_LOC_Y",
        "STRUCT",
        "PARAM",
        "MEAS_PT",
        "UNITS",
    ]

    file_exists = Path(summary_output_path).exists()  # 👈 Avoid repeated checks

    tqdm.write("Processing Initiated:")

    for transformed_data in transform_raw_liv_file_every_nth_laser_chunked(
        filepath, decoder_path, machine_code, wafer_code, sampling_freq, chunksize
    ):
        processed_in_chunk = 0  # count lasers processed in this chunk

        chunk = transformed_data["chunk"]
        n_meas = transformed_data["n_meas"]
        peak_wavelengths = transformed_data["peak_wavelengths"]
        leakage_currents = transformed_data["leakage_currents"]
        reverse_voltages = transformed_data["reverse_voltages"]
        cube_numbers = transformed_data["cube_numbers"]

        # start_groupby = time.time()  # ⬅️ START
        grouped = chunk.groupby("TE_LABEL")
        # t_groupby += time.time() - start_groupby  # ⬅️ END

        for te_label, group in grouped:

            if te_label not in accumulator:
                accumulator[te_label] = [group]
            else:
                accumulator[te_label].append(group)

            if sum(len(g) for g in accumulator[te_label]) >= n_meas:
                full_data = pd.concat(accumulator[te_label], ignore_index=True)
                touchdown = full_data["TOUCHDOWN"].iloc[0] if "TOUCHDOWN" in full_data else ""

                current = full_data["LDI_mA"].values
                intensity = full_data["PD"].values
                voltage = full_data["Vf"].values

                no_laser_flag = flag_no_laser(intensity)

                if no_laser_flag:
                    flag = 1
                    ith = 0
                    slope_eff = 0
                    series_r = 0
                    kink_current = 0
                    peak_wl = 0
                    leakage_current = 0
                    reverse_voltage = 0
                    cube_number = 0
                    spd_current = 0
                    KNKPPD_BL = 0  # Dummy Values set in lieu of generation of these parameters that labview does
                    KNKKMM_BL = 0  # Dummy Values set in lieu of generation of these parameters that labview does
                else:
                    flag = 0
                    ith = find_ith_value(intensity, current)

                    slope_eff = find_slope_efficiency(intensity, current, ith) if ith else 0

                    series_r = find_series_resistance(voltage, current, ith) if ith else 0
                    kink_results = find_kink_fast_spd(intensity, current, voltage, ith)
                    kink_current = kink_results["kink_current"]
                    spd_current = kink_results["spd_current"]

                    peak_wl = peak_wavelengths.get(touchdown, np.nan) if peak_wavelengths else np.nan
                    leakage_current = leakage_currents.get(touchdown, np.nan) if leakage_currents else np.nan
                    reverse_voltage = reverse_voltages.get(touchdown, np.nan) if reverse_voltages else np.nan
                    cube_number = cube_numbers.get(touchdown, np.nan) if cube_numbers else "not_found"

                    KNKPPD_BL = 99999  # Dummy Values set in lieu of generation of these parameters that labview does
                    KNKKMM_BL = 99999  # Dummy Values set in lieu of generation of these parameters that labview does

                summary_records.append(
                    [
                        wafer_code,
                        product_code,
                        machine_code,
                        f"py_v{VERSION}",
                        te_label,
                        touchdown,
                        ith,
                        slope_eff,
                        series_r,
                        peak_wl,
                        spd_current,
                        kink_current,
                        leakage_current,
                        reverse_voltage,
                        cube_number,
                        full_data["TYPE"].iloc[0],
                        full_data["X_UM"].iloc[0],
                        full_data["Y_UM"].iloc[0],
                        flag,
                    ]
                )

                melted_entries = [
                    ("ITH_BL", ith),
                    ("SE_BL", slope_eff),
                    ("RS_BL", series_r),
                    ("PW_BL", peak_wl),
                    ("ISPD_BL", spd_current),
                    ("KNK1_BL", kink_current),
                    ("KNKPPD_BL", KNKPPD_BL),
                    ("KNKMM_BL", KNKKMM_BL),
                    ("ILK_BL", leakage_current),
                    ("RV_BL", reverse_voltage),
                ]

                for param, val in melted_entries:
                    melted_records.append(
                        [
                            INI_PDATETIME,
                            INI_LOT,
                            INI_WAFER,
                            INI_OP,
                            INI_STAGE,
                            INI_STEP,
                            INI_PROD,
                            INI_MACH,
                            FAC,
                            CARD,
                            INI_EMPID,
                            cube_number,
                            INI_MEAS_TYPE,
                            COND,
                            pt_counter,
                            convert_x_position(
                                full_data["X_UM"].iloc[0],
                                dividing_factor=INI_XDIVIDING_FACTOR,
                                product_type=PRODUCT_TYPE,
                            ),
                            convert_y_position(
                                full_data["Y_UM"].iloc[0],
                                dividing_factor=INI_YDIVIDING_FACTOR,
                                product_type=PRODUCT_TYPE,
                            ),
                            STRUCT,
                            param,
                            val,
                            UNIT_DICT.get(param, ""),
                        ]
                    )

                    pt_counter += 1  # 👈 increment once per row

                del accumulator[te_label]

                if len(summary_records) >= export_buffer:
                    df_to_export = pd.DataFrame(summary_records, columns=columns_local)
                    df_to_export.to_csv(Path(summary_output_path), mode="a", header=not file_exists, index=False)
                    file_exists = True
                    total_exported_rows += len(df_to_export)  # 👈 Update counter
                    tqdm.write(f"🔼 Processed {len(df_to_export)} devices. Total exported: {total_exported_rows}")
                    summary_records.clear()
                if len(melted_records) >= export_buffer:
                    df_melted_export = pd.DataFrame(melted_records, columns=columns_levee)
                    df_melted_export.to_csv(
                        dat_output_path, mode="a", header=not dat_file_exists, index=False, sep="\t"
                    )
                    dat_file_exists = True
                    melted_records.clear()

                processed_in_chunk += 1

        total_processed += processed_in_chunk
        tqdm.write(f"🔼 Processed {processed_in_chunk} lasers in this chunk. Total processed: {total_processed}")

    tqdm.write(f"Data Exporting to: {summary_output_path}")
    if summary_records:
        df_to_export = pd.DataFrame(summary_records, columns=columns_local)
        df_to_export.to_csv(summary_output_path, mode="a", header=not file_exists, index=False)
        total_exported_rows += len(df_to_export)
        tqdm.write(f"🔼 Final export of {len(df_to_export)} rows. Total exported: {total_exported_rows}")
    if melted_records:
        df_melted_export = pd.DataFrame(melted_records, columns=columns_levee)
        df_melted_export.to_csv(dat_output_path, mode="a", header=not dat_file_exists, index=False, sep="\t")
        tqdm.write(f"🔼 Final .dat export of {len(df_melted_export)} rows.")

    tqdm.write(f"✓ Device summary exported for {wafer_code}")


def stream_process_lasing_parameters_cod(
    filepath,
    summary_output_path,
    dat_output_path,
    ini_data,
    decoder_path,
    sampling_freq=10000,
    chunksize=10000,
    export_buffer=1000,
):

    accumulator = {}
    summary_records = []
    melted_records = []
    dat_file_exists = dat_output_path.exists()
    total_processed = 0  # 👈 Counter for total exported rows
    total_exported_rows = 0  # 👈 Counter for total exported rows
    pt_counter = 0  # 👈 Counter for melted .dat file rows

    # ⬇️ Unpack ini_data dictionary
    INI_LOT = ini_data["INI_LOT"]
    wafer_code = INI_LOT
    INI_WAFER = ini_data["INI_WAFER"]
    INI_OP = ini_data["INI_OP"]
    INI_STAGE = ini_data["INI_STAGE"]
    INI_STEP = ini_data["INI_STEP"]
    INI_PROD = ini_data["INI_PROD"]
    product_code = INI_PROD
    INI_MACH = ini_data["INI_MACH"]
    machine_code = INI_MACH
    INI_EMPID = ini_data["INI_EMPID"]
    INI_MEAS_TYPE = ini_data["INI_MEAS_TYPE"]
    INI_PDATETIME = ini_data["INI_PDATETIME"]
    INI_XDIVIDING_FACTOR = ini_data["INI_XDIVIDING_FACTOR"]
    INI_YDIVIDING_FACTOR = ini_data["INI_YDIVIDING_FACTOR"]
    FAC = ini_data.get("FAC", "STST")  # default fallback
    CARD = ini_data.get("CARD", 0)
    COND = ini_data.get("COND", "Py_v1.0")
    STRUCT = ini_data.get("STRUCT", "L")
    PRODUCT_TYPE = ini_data["PRODUCT_TYPE"]
    UNIT_DICT = ini_data["UNIT_DICT"]

    columns_local = [
        "WAFER_ID",
        "WAFER_TYPE",
        "MACH_NUM",
        "LIV_ANALYZER_VERSION",
        "TE_LABEL",
        "TOUCHDOWN",
        "ITH_MA",
        "SLOPE_EFF_MW_MA",
        "RS_FIT_OHMS",
        "PEAK_WAVE_AT35_NM",
        "CURRENT_AT_SPD",
        "KINK1_CURRENT_MA",
        "LEAK_CURR_UA",
        "REVERSE_VOLTS_V",
        "CUBE_NUM",
        "TYPE",
        "STX_WAFER_X_UM",
        "STX_WAFER_Y_UM",
        "FLAG_LOW_POUT",
    ]

    columns_levee = [
        "PDATETIME",
        "LOT",
        "WAFER",
        "OP",
        "STAGE",
        "STEP",
        "PROD",
        "MACH",
        "FAC",
        "CARD",
        "EMPID",
        "FLD",
        "MEAS_TYPE",
        "COND",
        "PT",
        "PT_LOC_X",
        "PT_LOC_Y",
        "STRUCT",
        "PARAM",
        "MEAS_PT",
        "UNITS",
    ]

    file_exists = Path(summary_output_path).exists()  # 👈 Avoid repeated checks

    tqdm.write("Processing Initiated:")

    for transformed_data in transform_raw_liv_file_every_nth_laser_chunked(
        filepath, decoder_path, machine_code, wafer_code, sampling_freq, chunksize
    ):
        processed_in_chunk = 0  # count lasers processed in this chunk

        chunk = transformed_data["chunk"]
        n_meas = transformed_data["n_meas"]
        peak_wavelengths = transformed_data["peak_wavelengths"]
        leakage_currents = transformed_data["leakage_currents"]
        reverse_voltages = transformed_data["reverse_voltages"]
        cube_numbers = transformed_data["cube_numbers"]

        # start_groupby = time.time()  # ⬅️ START
        grouped = chunk.groupby("TE_LABEL")
        # t_groupby += time.time() - start_groupby  # ⬅️ END

        for te_label, group in grouped:

            if te_label not in accumulator:
                accumulator[te_label] = [group]
            else:
                accumulator[te_label].append(group)

            if sum(len(g) for g in accumulator[te_label]) >= n_meas:
                full_data = pd.concat(accumulator[te_label], ignore_index=True)
                touchdown = full_data["TOUCHDOWN"].iloc[0] if "TOUCHDOWN" in full_data else ""

                current = full_data["LDI_mA"].values
                intensity = full_data["PD"].values
                voltage = full_data["Vf"].values

                no_laser_flag = flag_no_laser(intensity)

                if no_laser_flag:
                    flag = 1
                    ith = 0
                    slope_eff = 0
                    series_r = 0
                    kink_current = 0
                    peak_wl = 0
                    leakage_current = 0
                    reverse_voltage = 0
                    cube_number = 0
                    spd_current = 0
                    KNKPPD_BL = 0  # Dummy Values set in lieu of generation of these parameters that labview does
                    KNKKMM_BL = 0  # Dummy Values set in lieu of generation of these parameters that labview does
                else:
                    flag = 0
                    ith = find_ith_value(intensity, current)

                    slope_eff = find_slope_efficiency(intensity, current, ith) if ith else 0

                    series_r = find_series_resistance(voltage, current, ith) if ith else 0
                    kink_results = find_kink(intensity, current, voltage, ith)
                    kink_current = kink_results["kink_current"]
                    spd_current = kink_results["spd_current"]

                    peak_wl = peak_wavelengths.get(touchdown, np.nan) if peak_wavelengths else np.nan
                    leakage_current = leakage_currents.get(touchdown, np.nan) if leakage_currents else np.nan
                    reverse_voltage = reverse_voltages.get(touchdown, np.nan) if reverse_voltages else np.nan
                    cube_number = cube_numbers.get(touchdown, np.nan) if cube_numbers else "not_found"

                    KNKPPD_BL = 99999  # Dummy Values set in lieu of generation of these parameters that labview does
                    KNKKMM_BL = 99999  # Dummy Values set in lieu of generation of these parameters that labview does

                summary_records.append(
                    [
                        wafer_code,
                        product_code,
                        machine_code,
                        f"py_v{VERSION}",
                        te_label,
                        touchdown,
                        ith,
                        slope_eff,
                        series_r,
                        peak_wl,
                        spd_current,
                        kink_current,
                        leakage_current,
                        reverse_voltage,
                        cube_number,
                        full_data["TYPE"].iloc[0],
                        full_data["X_UM"].iloc[0],
                        full_data["Y_UM"].iloc[0],
                        flag,
                    ]
                )

                melted_entries = [
                    ("ITH_BL", ith),
                    ("SE_BL", slope_eff),
                    ("RS_BL", series_r),
                    ("PW_BL", peak_wl),
                    ("ISPD_BL", spd_current),
                    ("KNK1_BL", kink_current),
                    ("KNKPPD_BL", KNKPPD_BL),
                    ("KNKMM_BL", KNKKMM_BL),
                    ("ILK_BL", leakage_current),
                    ("RV_BL", reverse_voltage),
                ]

                for param, val in melted_entries:
                    melted_records.append(
                        [
                            INI_PDATETIME,
                            INI_LOT,
                            INI_WAFER,
                            INI_OP,
                            INI_STAGE,
                            INI_STEP,
                            INI_PROD,
                            INI_MACH,
                            FAC,
                            CARD,
                            INI_EMPID,
                            cube_number,
                            INI_MEAS_TYPE,
                            COND,
                            pt_counter,
                            convert_x_position(
                                full_data["X_UM"].iloc[0],
                                dividing_factor=INI_XDIVIDING_FACTOR,
                                product_type=PRODUCT_TYPE,
                            ),
                            convert_y_position(
                                full_data["Y_UM"].iloc[0],
                                dividing_factor=INI_YDIVIDING_FACTOR,
                                product_type=PRODUCT_TYPE,
                            ),
                            STRUCT,
                            param,
                            val,
                            UNIT_DICT.get(param, ""),
                        ]
                    )

                    pt_counter += 1  # 👈 increment once per row

                del accumulator[te_label]

                if len(summary_records) >= export_buffer:
                    df_to_export = pd.DataFrame(summary_records, columns=columns_local)
                    df_to_export.to_csv(Path(summary_output_path), mode="a", header=not file_exists, index=False)
                    file_exists = True
                    total_exported_rows += len(df_to_export)  # 👈 Update counter
                    tqdm.write(f"🔼 Processed {len(df_to_export)} devices. Total exported: {total_exported_rows}")
                    summary_records.clear()
                if len(melted_records) >= export_buffer:
                    df_melted_export = pd.DataFrame(melted_records, columns=columns_levee)
                    df_melted_export.to_csv(
                        dat_output_path, mode="a", header=not dat_file_exists, index=False, sep="\t"
                    )
                    dat_file_exists = True
                    melted_records.clear()

                processed_in_chunk += 1

        total_processed += processed_in_chunk
        tqdm.write(f"🔼 Processed {processed_in_chunk} lasers in this chunk. Total processed: {total_processed}")

    tqdm.write(f"Data Exporting to: {summary_output_path}")
    if summary_records:
        df_to_export = pd.DataFrame(summary_records, columns=columns_local)
        df_to_export.to_csv(summary_output_path, mode="a", header=not file_exists, index=False)
        total_exported_rows += len(df_to_export)
        tqdm.write(f"🔼 Final export of {len(df_to_export)} rows. Total exported: {total_exported_rows}")
    if melted_records:
        df_melted_export = pd.DataFrame(melted_records, columns=columns_levee)
        df_melted_export.to_csv(dat_output_path, mode="a", header=not dat_file_exists, index=False, sep="\t")
        tqdm.write(f"🔼 Final .dat export of {len(df_melted_export)} rows.")

    tqdm.write(f"✓ Device summary exported for {wafer_code}")


def export_wafer_level_cod_summary(
    wafer_code,
    machine_code,
    cod_measured_time,
    device_summary_path,
    wafer_summary_output_folder,
):
    if not device_summary_path.exists():
        print(f"Missing device summary file for {wafer_code}, skipping...")
        return

    df = pd.read_csv(device_summary_path)
    total_count = len(df)

    cod_count = (df["spd_eval"] == "COD").sum()
    cod_percentage = (cod_count / total_count) * 100 if total_count else 0
    lower_bound, upper_bound = calculate_confidence_interval_wilsonscore(cod_count, total_count)

    rollover_count = (df["spd_eval"].isin(["ROLLOVER", "COD-PR"])).sum()
    no_laser_count = (df["spd_eval"] == "NO LASER").sum()

    failmode_counts = df["pot_failmode"].value_counts()
    failmode_counts = failmode_counts.reindex(["COBD", "COMD"], fill_value=0)
    total_fail = failmode_counts.sum()
    cobd_percentage = (failmode_counts["COBD"] / total_fail) * 100 if total_fail else 0
    comd_percentage = (failmode_counts["COMD"] / total_fail) * 100 if total_fail else 0

    summary_dict = {
        "Wafer_Code": [wafer_code, wafer_code, wafer_code],
        "MEAS_TIME": [cod_measured_time, cod_measured_time, cod_measured_time],
        "MACH": [machine_code, "", ""],
        "COD_Percentage": [cod_percentage, lower_bound, upper_bound],
        "ROLLOVER_Count": [rollover_count, "", ""],
        "COD_Count": [cod_count, "", ""],
        "NO_LASER_Count": [no_laser_count, "", ""],
        "COBD_Percentage": [cobd_percentage, "", ""],
        "COMD_Percentage": [comd_percentage, "", ""],
    }

    types = df["TYPE"].unique()
    for t in types:
        type_df = df[df["TYPE"] == t]
        total_type = len(type_df)

        cod_count_type = (type_df["spd_eval"] == "COD").sum()
        cod_percentage_type = (cod_count_type / total_type) * 100 if total_type else 0
        lower_t, upper_t = calculate_confidence_interval_wilsonscore(cod_count_type, total_type)

        rollover_count_type = (type_df["spd_eval"] == "ROLLOVER").sum()
        no_laser_count_type = (type_df["spd_eval"] == "NO LASER").sum()

        failmode_t = type_df["pot_failmode"].value_counts().reindex(["COBD", "COMD"], fill_value=0)
        total_fail_t = failmode_t.sum()
        cobd_percentage_t = (failmode_t["COBD"] / total_fail_t) * 100 if total_fail_t else 0
        comd_percentage_t = (failmode_t["COMD"] / total_fail_t) * 100 if total_fail_t else 0

        summary_dict[f"{t} COD Percentage"] = [cod_percentage_type, lower_t, upper_t]
        summary_dict[f"{t} ROLLOVER Count"] = [rollover_count_type, "", ""]
        summary_dict[f"{t} COD Count"] = [cod_count_type, "", ""]
        summary_dict[f"{t} NO LASER Count"] = [no_laser_count_type, "", ""]
        summary_dict[f"{t} COBD Percentage"] = [cobd_percentage_t, "", ""]
        summary_dict[f"{t} COMD Percentage"] = [comd_percentage_t, "", ""]

    final_summary = pd.DataFrame(summary_dict)

    out_path = Path(wafer_summary_output_folder) / f"{wafer_code}_wafer_cod_summary_{cod_measured_time}_processed.csv"
    final_summary.to_csv(out_path, index=False)
    tqdm.write(f"✓ Wafer-level summary exported to {out_path}\n")


# ------------------------------------------------------------------ #
#                          INI Reading Code                          #
# ------------------------------------------------------------------ #
def convert_x_position(value, dividing_factor, product_type):
    try:
        return value / dividing_factor if dividing_factor else value
    except Exception:
        return np.nan  # or return value if you'd prefer to keep the raw input


def convert_y_position(value, dividing_factor, product_type):
    try:
        return value / dividing_factor if dividing_factor else value
    except Exception:
        return np.nan  # or return value if you'd prefer to keep the raw input


def load_ini_data(ini_path):
    config = configparser.RawConfigParser(strict=False)
    config.read(ini_path)
    mes_dict = dict(config["MES"])
    setup_info_dict = dict(config["LIV_SETUP_INFO"])

    stagestep_str = mes_dict.get("stagestep", "")
    stage_parts = [s.strip() for s in stagestep_str.split(";") if s.strip()]

    dt_obj = datetime.strptime(mes_dict["datetimestamp"], "%Y%m%d%H%M%S")
    pdatetime = dt_obj.strftime("%d%b%Y %H:%M:%S")

    ini_prod = mes_dict["product"]
    product_code_dict = {"HALO": ["QD", "NV", "NK", "NE"]}
    product_type = "HALO" if ini_prod in product_code_dict["HALO"] else "SUBARU"

    return {
        "INI_LOT": mes_dict["lot"],
        "INI_WAFER": mes_dict["wafer1"],
        "INI_OP": mes_dict["op"],
        "INI_STAGE": stage_parts[0] if len(stage_parts) > 0 else "",
        "INI_STEP": stage_parts[1] if len(stage_parts) > 1 else "",
        "INI_PROD": ini_prod,
        "INI_MACH": mes_dict["machine"],
        "INI_EMPID": mes_dict["operator"],
        "INI_MEAS_TYPE": mes_dict["recipe"].strip('"; '),
        "INI_PDATETIME": pdatetime,
        "INI_XDIVIDING_FACTOR": int(setup_info_dict["xdiesize"]),
        "INI_YDIVIDING_FACTOR": int(setup_info_dict["ydiesize"]),
        "INI_FINAL_RECIPE": setup_info_dict["final_recipe"],
        "FAC": "STST",
        "CARD": 0,
        "COND": "Py_v1.0",
        "STRUCT": "L",
        "UNIT_DICT": {
            "ITH_BL": "mA",
            "RS_BL": "Ohms",
            "PW_BL": "nm",
            "RV_BL": "V",
            "ILK_BL": "uA",
            "SE_BL": "mWmA",
            "KNK1_BL": "mA",
            "KNKMM_BL": "mA",
            "ISPD_BL": "mA",
            "KNKPPD_BL": "PCT",
        },
        "PRODUCT_TYPE": product_type,
    }


# ------------------------------------------------------------------ #
#                            Logging Code                            #
# ------------------------------------------------------------------ #
class TeeLogger:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for s in self.streams:
            s.write(message)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


# ------------------------------------------------------------------ #
#                      Watchdog Triggering Code                      #
# ------------------------------------------------------------------ #
def wait_for_file_to_appear_and_be_readable(filepath, max_wait=300, delay=1):
    wait_time = 0
    while not filepath.exists() and wait_time < max_wait:
        print(f"Waiting for {filepath.name} to appear... ({wait_time}s elapsed)")
        time.sleep(delay)
        wait_time += delay

    if not filepath.exists():
        print(f"❌ File {filepath.name} did not appear within {max_wait} seconds.")
        return False

    for attempt in range(max_wait):
        try:
            with open(filepath, "rb"):
                print(f"✅ File {filepath.name} is now ready to read.")
                return True
        except (PermissionError, FileNotFoundError):
            print(f"Waiting for {filepath.name} to be ready... ({attempt}s elapsed)")
            time.sleep(delay)

    print(f"❌ File {filepath.name} was not readable after {max_wait} seconds.")
    return False


def extract_test_type_from_file(file_path):
    file = Path(file_path)

    # Check if file is a CSV, starts with "LIV_", and ends with "_STX" in the stem
    if file.name.startswith("LIV_") and file.suffix == ".csv" and file.stem.upper().endswith("_STX"):
        parts = file.name.split("_")
        if len(parts) >= 3:
            if "COD250" in file.name.upper():
                return "SPD"
            elif "COD70" in file.name.upper():
                return "Rollover"
            else:
                return "Baseline"

    return None


# 🔁 Processing triggering
def initialise_lasing_processing(file_path, ini_dict, test_type, detection_time):
    INI_MACH = ini_dict["INI_MACH"]
    INI_LOT = ini_dict["INI_LOT"]
    wafer_code = INI_LOT  # just for readability
    PRODUCT_TYPE = ini_dict["PRODUCT_TYPE"]

    tqdm.write(f"Fetched Info (from {INI_NAME}): Wafer Code - {INI_LOT}, Machine Code - {INI_MACH}\n")

    if wafer_code:
        if PRODUCT_TYPE == "HALO":
            tqdm.write("Halo Wafer Detected, using Halo Decoder\n")
            decoder_path = ROOT_DIR / "decoders" / HALO_DECODER
        else:
            tqdm.write("Non-Halo Wafer Detected, Using Subaru Decoder\n")
            decoder_path = ROOT_DIR / "decoders" / SUBARU_DECODER

        gtx_output_path = RESULTS_PATH / f"{file_path.stem}_processed.csv"
        levee_output_path = RESULTS_PATH / f"{wafer_code}_{detection_time}.dat"

        # 👉 Choose processing function based on test_type
        if test_type == "SPD":
            tqdm.write("⚙️ SPD test detected, using COD processing function\n")
            stream_process_lasing_parameters_cod(
                file_path,
                summary_output_path=gtx_output_path,
                dat_output_path=levee_output_path,
                ini_data=ini_dict,
                decoder_path=decoder_path,
                sampling_freq=1,
            )
        elif test_type == "Rollover":
            tqdm.write("⚙️ Rollover test detected, using standard processing function\n")
            stream_process_lasing_parameters(
                file_path,
                summary_output_path=gtx_output_path,
                dat_output_path=levee_output_path,
                ini_data=ini_dict,
                decoder_path=decoder_path,
                sampling_freq=1,
            )
        else:
            tqdm.write("⚙️ Baseline test or other, using default processing function\n")
            stream_process_lasing_parameters(
                file_path,
                summary_output_path=gtx_output_path,
                dat_output_path=levee_output_path,
                ini_data=ini_dict,
                decoder_path=decoder_path,
            )

    else:
        message = f"No valid wafer code found in folder: {file_path.parent.name}"
        tqdm.write(message)
        tqdm.write(f"\n\n========== Wafer: UNKNOWN ==========\n")
        tqdm.write(f"❌ {message}\n")


def print_watcher_banner():
    tqdm.write(f"\n\n# --------------------------- LIV Automatic Analyser v{VERSION} -------------------------- #")
    tqdm.write("(Do not close this command window)")
    tqdm.write(f"Watching folder: {MONITORED_PATH}\n")


print_watcher_banner()


# 🔍 Folder monitor logic
class WaferFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith("_STX.csv"):
            return

        def analysis_job():
            file_path = Path(event.src_path)
            detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            detection_time_raw = datetime.now().strftime("%Y%m%d%H%M%S")

            # ---------- Wrap the following code in a buffer that ensures output is appended to logs --------- #
            buffer = io.StringIO()
            tee_out = TeeLogger(sys.__stdout__, buffer)
            tee_err = TeeLogger(sys.__stderr__, buffer)

            sys.stdout = tee_out
            sys.stderr = tee_err

            try:
                tqdm.write(f"[{detection_time}] Detected new file: {file_path.name}\n")

                if not wait_for_file_to_appear_and_be_readable(file_path):
                    message = f"{file_path.name} did not become readable."
                    tqdm.write(message)
                    with open(LOG_PATH, "a", encoding="utf-8") as log_file:
                        log_file.write(f"[{detection_time}] ❌ {message}\n")
                    return

                test_type = extract_test_type_from_file(file_path)
                tqdm.write(f"\nDetected Test Type: {test_type}")

                if test_type not in ("Rollover", "Baseline", "SPD"):
                    message = f"Skipping analysis for test type '{test_type}' in: {file_path.name}"
                    tqdm.write(message)
                    tqdm.write(f"[{detection_time}] ⚠️ {message}\n")
                    print_watcher_banner()
                    return

                ini_dict = load_ini_data(INI_LOCATION)
                initialise_lasing_processing(
                    file_path, ini_dict, test_type, detection_time=detection_time_raw
                )  # unpacks into rest of code

                # ----------------- case statement for last recipe --> shunting + triggering levee ----------------- #
                if ini_dict["INI_MEAS_TYPE"] == ini_dict["INI_FINAL_RECIPE"]:
                    tqdm.write(f"\nFinal Recipe Detected, Engaging Levee Protocol:\n")
                # ------------------------------------ end of last recipe case ----------------------------------- #

                # ✅ Rename file after successful analysis
                timestamp = datetime.now().strftime("RAW%Y%m%d%H%M%S")
                new_name = file_path.stem.replace("STX", timestamp) + file_path.suffix
                new_path = file_path.with_name(new_name)

                try:
                    file_path.rename(new_path)
                    tqdm.write(f"✅ Renamed file to: {new_path.name}")
                    tqdm.write(f"[{detection_time}] ✅ Renamed to: {new_path.name}\n")
                except Exception as e:
                    tqdm.write(f"❌ Failed to rename file: {e}")
                    tqdm.write(f"[{detection_time}] ❌ Rename failed: {e}\n")

                success = True  # Mark success if all above runs without fatal errors

            except Exception as e:
                error_message = str(e)
                tqdm.write(f"❌ Error while processing file: {e}")
                traceback.print_exc(file=buffer)  # Include full traceback in log

            finally:
                # ✅ Write final log
                with open(LOG_PATH, "a", encoding="utf-8") as log_file:
                    lot = ini_dict.get("INI_LOT", "UNKNOWN")
                    log_file.write(f"\n\n========== Wafer: {lot} ==========\n")
                    if success:
                        log_file.write(f"[{detection_time}] ✅ Processed: {file_path.name} (Wafer: {lot})\n")
                    else:
                        log_file.write(f"[{detection_time}] ❌ Failed to process: {file_path.name} (Wafer: {lot})\n")
                        if error_message:
                            log_file.write(f"Reason: {error_message}\n")
                    log_file.write("\n\n---- Processing Output ----\n")
                    log_file.write(buffer.getvalue())
                    log_file.write("---- End of Output ----\n\n")

                # Always restore stdout/stderr
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__
            # ---------------------------------- end of log output wrapping ---------------------------------- #

            print_watcher_banner()

        threading.Thread(target=analysis_job, daemon=True).start()


# 🕵️ Start Watchdog
observer = Observer()
event_handler = WaferFileHandler()
observer.schedule(event_handler, str(MONITORED_PATH), recursive=False)
observer.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
observer.join()
