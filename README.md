# Notebook Usage Instructions for Validation
## Explanation of various sections 
(These sections marked by large text in the notebook):
- `Imports`: Imports the required python packages
- `Input Wafer Codes + Find Raw Filelinks`: 
    - 2 cells where the user can input what wafer codes are analysed (just in a list), and the filenames of any decoder files. The first cells just defines the function that connects to GTX. 
    - "ANALYSIS_RUN_NAME" is added into the names of any exports, so useful to adjust for new runs where you want to preserve old data.
- `(General, ITH, SE...) Functions`: Cells that define the functions that contain the photonics algorithms. If you want to change how something like ITH works, change these functions. 
- `Streamlined Analysis`: 
    - The main calling code for sampled analysis. The first cell calls reads the raw files from `Input Wafer Codes + Find Raw Filelinks` and runs the python analysis and exports CSVs with the CTQs. 
    - Note that you can reduce the sampling rate (defaulted to 10000) at the expense of runtime
    - It exports 1 csv per device. The second cell just merges all the csvs into 1 "batch" csv. 
- `Specific Device Testing`: An alternative calling code that takes an input file with a list of TE_LABELs, and then runs analysis for each of those wafer_code TE_LABEL pairs. 
- `Debugging`: The section of code that can run the code for a specific device. 
    - First cell defines a alternate set of algorithms (eg "find_series_resistance" to "find_series_resistance_debug") that are identical but add some debugging plots for visibility
    - Second cell will run analysis on a set of lasers input, where you input a list of wafers, and corresponding lists of touchdown numbers for each device. 
    - Note that in this second cell, you can change a function name between debugging versions if you want to enable or disable plots.

## Typical Cell Running Ordering to Run Analysis
1) Always run the `Imports` cell first.
2) Import the desired wafer codes you want to analyse, edit the "ANALYSIS_RUN_NAME" for the export names, then run the `Input Wafer Codes + Find Raw Filelinks` cell. If you have a lot of wafers it can take a few minutes. 
3) Run all of the `(General, ITH, SE...) Functions` cells.
4) Then run the `Streamlined Analysis` cells. The first cell will run the analysis can a long time if many wafers are selected. For 220 wafers at 10,000 sampling rate, will be around 1 hour to complete. 


## How the Code and Algorithms Work

This notebook is designed to process raw LIV (Light–Current–Voltage) data from many laser devices in a memory-efficient, streaming fashion. The workflow connects to the file server, locates raw wafer data, decodes device positions, and then applies photonics algorithms to extract critical test quantities (CTQs) such as threshold current (I_th), slope efficiency, series resistance, and kink detection.  

The implementation is modular: each algorithm is encapsulated in a function, and the higher-level "stream processing" functions call these per-device. This makes it easy to swap or debug individual algorithms without rewriting the pipeline.

### Key Algorithms

#### 1. Threshold Current (I_th) Extraction — `find_ith_value`
The **threshold current** is determined by detecting the point at which optical output intensity rises linearly with current. The algorithm:
- Trims the current range to a usable window (e.g. >1 mA and ≤34 mA).
- Optionally applies **RANSAC regression** to robustly fit a linear model while excluding outliers.
- Uses a Gaussian residuals approach to refine the threshold transition point.
- Returns the estimated threshold current (`I_th`) that best explains the "turn-on" behavior of the laser.

This ensures the threshold is not biased by measurement noise or non-linear tails.

#### 2. Slope Efficiency — `find_slope_efficiency`
The **slope efficiency (SE)** is the slope of the L–I curve just above threshold. The algorithm:
- Finds the first index where current exceeds the extracted I_th.
- Selects a fitting window (default ~15 points).
- Fits a straight line to intensity vs. current in that range.
- Returns the slope, representing the conversion efficiency of current into light.

#### 3. Series Resistance — `find_series_resistance`
The **series resistance (R_s)** is derived from the V–I curve above threshold. The method:
- Finds the region just above I_th.
- Performs a linear regression on voltage vs. current in a fitting window.
- The slope of this regression corresponds to the electrical series resistance of the device.

This helps assess device quality and ohmic losses.

#### 4. Kink Detection — `find_kink`, `find_spd`, `exclusion_based_on_second_derivative`, `apply_weighted_linear_fit`
Kinks in the L–I curve indicate changes in optical mode or defects. The code uses a multi-step approach:
- **First derivative (dL/dI)**: Sudden changes in slope are flagged using a **Sudden Power Drop (SPD)** filter.
- **Second derivative filtering**: Excludes spurious detections due to noise by requiring consistent curvature changes.
- **Weighted linear fitting**: Fits pre- and post-kink regions and compares slopes.
- If a kink occurs "too early," a re-fit is attempted to avoid false positives.

The output is the detected kink current and slope change, useful for reliability analysis.

### Data Flow
1. **Raw file loading (streaming)**: Large wafer raw files are read in chunks (`transform_raw_liv_file_every_nth_laser_chunked`) to avoid memory overload.
2. **Per-device analysis**: Each TE_LABEL device is passed through the above algorithms to extract CTQs.
3. **Export**: Results are written per device and then aggregated into batch CSVs.

### Debug Versions
For validation, alternate versions of algorithms exist (e.g. `find_series_resistance_debug`). These add diagnostic plots of fits, residuals, and detected points, helping verify that each algorithm is performing correctly.

---