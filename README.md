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