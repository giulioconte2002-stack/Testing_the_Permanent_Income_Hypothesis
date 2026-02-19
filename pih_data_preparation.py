"""
Permanent Income Hypothesis - Data Preparation and Preliminary Analysis
========================================================================

This script performs all necessary data transformations and preliminary tests
for empirical analysis of the Permanent Income Hypothesis using U.S. aggregate data.

Steps implemented:
1. Load and merge FRED data
2. Convert from monthly to quarterly frequency
3. Deflate nominal series to real terms
4. Log transformation
5. First differencing (growth rates)
6. Construct real interest rate
7. Unit root tests (ADF)
8. Descriptive statistics and correlations
9. Cointegration tests
10. Export final dataset for regression analysis

Author: [Your Name]
Date: 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

# Create output directory if it doesn't exist
OUTPUT_DIR = 'pih_output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}/")
print()

# Manual implementation of ADF test for when statsmodels is not available
def adf_test_manual(y, max_lags=4):
    """
    Simplified ADF test implementation.
    Tests: Δy_t = α + β*t + ρ*y_{t-1} + Σγ_i*Δy_{t-i} + ε_t
    H0: ρ = 0 (unit root)
    """
    from scipy.stats import t as t_dist
    
    n = len(y)
    y = np.array(y)
    
    # Create lagged variables
    y_lag = y[:-1]
    dy = np.diff(y)
    
    # Simple version: no lags of differences
    # Regression: dy_t = α + ρ*y_{t-1} + ε_t
    X = np.column_stack([np.ones(len(y_lag)), y_lag])
    y_reg = dy
    
    # OLS estimation
    from numpy.linalg import inv
    beta = inv(X.T @ X) @ (X.T @ y_reg)
    residuals = y_reg - X @ beta
    sigma2 = np.sum(residuals**2) / (len(y_reg) - 2)
    var_beta = sigma2 * inv(X.T @ X)
    se_rho = np.sqrt(var_beta[1, 1])
    
    # ADF statistic
    adf_stat = beta[1] / se_rho
    
    # Critical values (MacKinnon, approximate for constant + trend)
    critical_values = {'1%': -3.43, '5%': -2.86, '10%': -2.57}
    
    # Approximate p-value (rough)
    if adf_stat < critical_values['1%']:
        p_value = 0.001
    elif adf_stat < critical_values['5%']:
        p_value = 0.03
    elif adf_stat < critical_values['10%']:
        p_value = 0.08
    else:
        p_value = 0.15
    
    return {
        'adf_stat': adf_stat,
        'p_value': p_value,
        'critical_values': critical_values,
        'lags': 0
    }

# Set display options for better output readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)
np.set_printoptions(precision=4, suppress=True)

print("="*80)
print("PERMANENT INCOME HYPOTHESIS - DATA PREPARATION")
print("="*80)
print()

# ==============================================================================
# STEP 1: LOAD DATA
# ==============================================================================
print("STEP 1: Loading FRED data...")

# Load all CSV files
# Update these paths to match where you saved the FRED data files
pcend = pd.read_csv('PCEND.csv')  # Personal Consumption Expenditures: Nondurable Goods
dspi = pd.read_csv('DSPI.csv')    # Disposable Personal Income
pcepi = pd.read_csv('PCEPI.csv')  # PCE Price Index
tb3ms = pd.read_csv('TB3MS.csv')  # 3-Month Treasury Bill Rate
mich = pd.read_csv('MICH.csv')    # Michigan Inflation Expectations

# Convert date columns to datetime
for df in [pcend, dspi, pcepi, tb3ms, mich]:
    df['observation_date'] = pd.to_datetime(df['observation_date'])

print(f"  - PCEND: {len(pcend)} monthly observations")
print(f"  - DSPI: {len(dspi)} monthly observations")
print(f"  - PCEPI: {len(pcepi)} monthly observations")
print(f"  - TB3MS: {len(tb3ms)} monthly observations")
print(f"  - MICH: {len(mich)} monthly observations")
print()

# ==============================================================================
# STEP 2: CONVERT TO QUARTERLY FREQUENCY
# ==============================================================================
print("STEP 2: Converting from monthly to quarterly frequency...")
print("  Method: Taking quarterly averages of monthly data")
print()

def to_quarterly(df, value_col):
    """
    Convert monthly data to quarterly by taking averages.
    
    Parameters:
    -----------
    df : DataFrame with 'observation_date' and value column
    value_col : str, name of the value column
    
    Returns:
    --------
    DataFrame with quarterly data
    """
    df = df.copy()
    df.set_index('observation_date', inplace=True)
    # Resample to quarterly frequency, taking mean of each quarter
    quarterly = df.resample('QE').mean()
    quarterly.reset_index(inplace=True)
    return quarterly

# Convert all series to quarterly
pcend_q = to_quarterly(pcend, 'PCEND')
dspi_q = to_quarterly(dspi, 'DSPI')
pcepi_q = to_quarterly(pcepi, 'PCEPI')
tb3ms_q = to_quarterly(tb3ms, 'TB3MS')
mich_q = to_quarterly(mich, 'MICH')

print(f"  Quarterly observations after conversion: {len(pcend_q)}")
print()

# ==============================================================================
# STEP 3: MERGE ALL SERIES
# ==============================================================================
print("STEP 3: Merging all series into single dataframe...")

# Start with consumption as base
data = pcend_q.copy()
data.columns = ['date', 'C_nominal']

# Merge other series
data = data.merge(dspi_q[['observation_date', 'DSPI']], 
                  left_on='date', right_on='observation_date', how='left')
data.rename(columns={'DSPI': 'Y_nominal'}, inplace=True)
data.drop('observation_date', axis=1, inplace=True)

data = data.merge(pcepi_q[['observation_date', 'PCEPI']], 
                  left_on='date', right_on='observation_date', how='left')
data.rename(columns={'PCEPI': 'price_index'}, inplace=True)
data.drop('observation_date', axis=1, inplace=True)

data = data.merge(tb3ms_q[['observation_date', 'TB3MS']], 
                  left_on='date', right_on='observation_date', how='left')
data.rename(columns={'TB3MS': 'i_nominal'}, inplace=True)
data.drop('observation_date', axis=1, inplace=True)

data = data.merge(mich_q[['observation_date', 'MICH']], 
                  left_on='date', right_on='observation_date', how='left')
data.rename(columns={'MICH': 'pi_expected'}, inplace=True)
data.drop('observation_date', axis=1, inplace=True)

# Restrict to post-1980 period (recommended for structural stability)
data = data[data['date'] >= '1980-01-01'].copy()
data.reset_index(drop=True, inplace=True)

print(f"  Merged dataset: {len(data)} quarterly observations from {data['date'].min()} to {data['date'].max()}")
print()

# ==============================================================================
# STEP 4: DEFLATE TO REAL TERMS
# ==============================================================================
print("STEP 4: Deflating nominal series to real terms...")
print("  Formula: X_real = (X_nominal / PCEPI) * 100")
print()

# Deflate consumption and income
# Multiply by 100 to maintain scale (since PCEPI is an index with base year = 100)
data['C_real'] = (data['C_nominal'] / data['price_index']) * 100
data['Y_real'] = (data['Y_nominal'] / data['price_index']) * 100

print("  Deflation complete. Real series created:")
print(f"    - C_real: Real consumption expenditures")
print(f"    - Y_real: Real disposable income")
print()

# ==============================================================================
# STEP 5: LOG TRANSFORMATION
# ==============================================================================
print("STEP 5: Taking natural logarithms...")
print("  Motivation: Log differences approximate growth rates")
print("  Formula: c_t = ln(C_real_t)")
print()

data['c'] = np.log(data['C_real'])
data['y'] = np.log(data['Y_real'])

print("  Log transformation complete:")
print(f"    - c: Log real consumption")
print(f"    - y: Log real income")
print()

# ==============================================================================
# STEP 6: FIRST DIFFERENCING (GROWTH RATES)
# ==============================================================================
print("STEP 6: Computing first differences (growth rates)...")
print("  Formula: Δc_t = c_t - c_{t-1} ≈ growth rate of consumption")
print()

data['delta_c'] = data['c'].diff()
data['delta_y'] = data['y'].diff()

# Also compute lagged growth rates for regression analysis
data['delta_c_lag1'] = data['delta_c'].shift(1)
data['delta_y_lag1'] = data['delta_y'].shift(1)

print("  First differencing complete:")
print(f"    - delta_c: Change in log consumption (consumption growth)")
print(f"    - delta_y: Change in log income (income growth)")
print(f"    - delta_c_lag1, delta_y_lag1: One-period lags")
print()

# ==============================================================================
# STEP 7: CONSTRUCT REAL INTEREST RATE
# ==============================================================================
print("STEP 7: Constructing real interest rate...")
print("  Formula: r_t = i_t - E_t[π_{t+1}]")
print("  Using Michigan inflation expectations as E_t[π_{t+1}]")
print()

# Real rate = nominal rate - expected inflation
data['r_real'] = data['i_nominal'] - data['pi_expected']

# For periods where Michigan expectations are missing, use backward-looking inflation
# Calculate actual inflation rate
data['pi_actual'] = (data['price_index'] / data['price_index'].shift(1) - 1) * 100

# Fill missing expected inflation with lagged actual inflation (adaptive expectations)
data['pi_expected_filled'] = data['pi_expected'].fillna(data['pi_actual'].shift(1))
data['r_real_filled'] = data['i_nominal'] - data['pi_expected_filled']

print(f"  Real interest rate constructed.")
print(f"  Missing values in expectations: {data['pi_expected'].isna().sum()}")
print(f"  Filled using adaptive expectations where needed")
print()

# ==============================================================================
# STEP 8: CLEAN DATASET - REMOVE ROWS WITH MISSING VALUES
# ==============================================================================
print("STEP 8: Cleaning dataset (removing missing values)...")

# Key variables for analysis
analysis_vars = ['delta_c', 'delta_y', 'delta_c_lag1', 'delta_y_lag1', 'r_real_filled']

# Count missing before
missing_before = data[analysis_vars].isna().sum().sum()

# Drop rows with any missing in key variables
data_clean = data.dropna(subset=analysis_vars).copy()
data_clean.reset_index(drop=True, inplace=True)

missing_after = data_clean[analysis_vars].isna().sum().sum()

print(f"  Missing values before: {missing_before}")
print(f"  Missing values after: {missing_after}")
print(f"  Final sample: {len(data_clean)} observations from {data_clean['date'].min()} to {data_clean['date'].max()}")
print()

# ==============================================================================
# STEP 9: UNIT ROOT TESTS (ADF)
# ==============================================================================
print("="*80)
print("STEP 9: AUGMENTED DICKEY-FULLER TESTS FOR UNIT ROOTS")
print("="*80)
print()
print("Null Hypothesis: Series has a unit root (non-stationary)")
print("Alternative: Series is stationary")
print("Decision rule: Reject H0 if p-value < 0.05")
print()

def adf_test(series, name, max_lags=12):
    """
    Perform ADF test and print results.
    
    The ADF test regression is:
    Δy_t = α + ρy_{t-1} + Σβ_i Δy_{t-i} + ε_t
    
    H0: ρ = 0 (unit root present)
    H1: ρ < 0 (stationary)
    """
    result = adf_test_manual(series.dropna(), max_lags=4)
    
    print(f"{name}:")
    print(f"  ADF Statistic: {result['adf_stat']:.4f}")
    print(f"  p-value (approx): {result['p_value']:.4f}")
    print(f"  Lags used: {result['lags']}")
    print(f"  Critical values:")
    for key, value in result['critical_values'].items():
        print(f"    {key}: {value:.4f}")
    
    if result['p_value'] < 0.05:
        print(f"  ✓ Reject H0: {name} is STATIONARY (p < 0.05)")
    else:
        print(f"  ✗ Fail to reject H0: {name} has UNIT ROOT (p >= 0.05)")
    print()
    
    return result

# Test levels (should have unit root)
print("Testing LEVELS (expect unit root):")
print("-" * 80)
adf_c_level = adf_test(data_clean['c'], 'c (log consumption level)')
adf_y_level = adf_test(data_clean['y'], 'y (log income level)')

# Test first differences (should be stationary)
print("Testing FIRST DIFFERENCES (expect stationarity):")
print("-" * 80)
adf_dc = adf_test(data_clean['delta_c'], 'Δc (consumption growth)')
adf_dy = adf_test(data_clean['delta_y'], 'Δy (income growth)')

# ==============================================================================
# STEP 10: DESCRIPTIVE STATISTICS
# ==============================================================================
print("="*80)
print("STEP 10: DESCRIPTIVE STATISTICS")
print("="*80)
print()

# Summary statistics for key variables
summary_vars = ['delta_c', 'delta_y', 'r_real_filled']
summary_stats = data_clean[summary_vars].describe()

# Add skewness and kurtosis
from scipy import stats
summary_stats.loc['skewness'] = data_clean[summary_vars].apply(lambda x: stats.skew(x, nan_policy='omit'))
summary_stats.loc['kurtosis'] = data_clean[summary_vars].apply(lambda x: stats.kurtosis(x, nan_policy='omit'))

print("Summary Statistics (quarterly growth rates in %, real interest rate in %):")
print(summary_stats.round(4))
print()

# Convert growth rates to annualized percentage for interpretation
print("Annualized interpretation:")
print(f"  Mean consumption growth: {data_clean['delta_c'].mean() * 4 * 100:.2f}% per year")
print(f"  Mean income growth: {data_clean['delta_y'].mean() * 4 * 100:.2f}% per year")
print(f"  Mean real interest rate: {data_clean['r_real_filled'].mean():.2f}% per year")
print()

# ==============================================================================
# STEP 11: CORRELATION ANALYSIS
# ==============================================================================
print("="*80)
print("STEP 11: CORRELATION ANALYSIS")
print("="*80)
print()

# Correlation matrix
corr_vars = ['delta_c', 'delta_y', 'delta_c_lag1', 'delta_y_lag1', 'r_real_filled']
correlation_matrix = data_clean[corr_vars].corr()

print("Correlation Matrix:")
print(correlation_matrix.round(4))
print()

# Key correlations to examine
print("Key correlations for PIH testing:")
print(f"  Corr(Δc_t, Δy_t) = {correlation_matrix.loc['delta_c', 'delta_y']:.4f}")
print(f"    → Contemporary correlation between consumption and income growth")
print(f"  Corr(Δc_t, Δy_{{t-1}}) = {correlation_matrix.loc['delta_c', 'delta_y_lag1']:.4f}")
print(f"    → PIH predicts this should be ≈ 0 (excess sensitivity test)")
print(f"  Corr(Δc_t, Δc_{{t-1}}) = {correlation_matrix.loc['delta_c', 'delta_c_lag1']:.4f}")
print(f"    → Autocorrelation in consumption growth (should be ≈ 0 under PIH)")
print(f"  Corr(Δc_t, r_t) = {correlation_matrix.loc['delta_c', 'r_real_filled']:.4f}")
print(f"    → Intertemporal substitution effect")
print()

# ==============================================================================
# STEP 12: AUTOCORRELATION FUNCTION
# ==============================================================================
print("="*80)
print("STEP 12: AUTOCORRELATION ANALYSIS")
print("="*80)
print()

# Manual autocorrelation calculation
def calculate_acf(series, nlags=8):
    """Calculate autocorrelation function manually"""
    series = np.array(series.dropna())
    mean = np.mean(series)
    c0 = np.sum((series - mean)**2) / len(series)
    
    acf_values = []
    for k in range(1, nlags + 1):
        c_k = np.sum((series[:-k] - mean) * (series[k:] - mean)) / len(series)
        acf_values.append(c_k / c0)
    
    return acf_values

acf_vals = calculate_acf(data_clean['delta_c'], nlags=8)

print("Autocorrelation Analysis for Δc:")
print("  H0: No autocorrelation at each lag")
print("  Rule of thumb: |ACF| > 2/√n suggests significance")
print()

n = len(data_clean['delta_c'].dropna())
threshold = 2 / np.sqrt(n)

print(f"  Sample size: {n}")
print(f"  Significance threshold (±): {threshold:.4f}")
print()
print("  Lag    ACF      Significant?")
print("  " + "-"*30)
for lag, acf_val in enumerate(acf_vals, 1):
    sig = "Yes" if abs(acf_val) > threshold else "No"
    print(f"   {lag:2d}   {acf_val:7.4f}     {sig}")
print()

if any(abs(val) > threshold for val in acf_vals):
    print("  ✗ Evidence of significant autocorrelation (violations of PIH martingale property)")
else:
    print("  ✓ No significant autocorrelation detected (consistent with PIH)")
print()

# ==============================================================================
# STEP 13: COINTEGRATION TEST
# ==============================================================================
print("="*80)
print("STEP 13: ENGLE-GRANGER COINTEGRATION TEST")
print("="*80)
print()
print("Testing whether c and y have a long-run equilibrium relationship")
print("H0: No cointegration (series drift apart)")
print("H1: Cointegration exists (stable long-run relationship)")
print()

# Manual Engle-Granger test
# Step 1: Estimate long-run relationship
c_vals = data_clean['c'].values
y_vals = data_clean['y'].values

# OLS regression: c = α + β*y + u
X_coint = np.column_stack([np.ones(len(y_vals)), y_vals])
from numpy.linalg import inv
beta_coint = inv(X_coint.T @ X_coint) @ (X_coint.T @ c_vals)
residuals_coint = c_vals - X_coint @ beta_coint

# Step 2: Test if residuals are stationary (cointegration)
coint_result = adf_test_manual(residuals_coint)

print(f"Test statistic: {coint_result['adf_stat']:.4f}")
print(f"p-value (approx): {coint_result['p_value']:.4f}")
print(f"Critical values: {coint_result['critical_values']}")
print()

# For Engle-Granger, critical values are more negative than standard ADF
if coint_result['adf_stat'] < -3.5:
    print("  ✓ Reject H0: Evidence of COINTEGRATION")
    print("    → Long-run relationship exists between consumption and income")
    print("    → Consistent with consumption proportional to permanent income")
else:
    print("  ✗ Fail to reject H0: No strong evidence of cointegration")
print()

# Long-run relationship already estimated above
print("Long-run relationship (if cointegrated):")
print(f"  c_t = {beta_coint[0]:.4f} + {beta_coint[1]:.4f} × y_t")
print(f"  → Long-run marginal propensity to consume: {beta_coint[1]:.4f}")
print()

# ==============================================================================
# STEP 14: VISUALIZATION
# ==============================================================================
print("="*80)
print("STEP 14: CREATING VISUALIZATIONS")
print("="*80)
print()

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Permanent Income Hypothesis - Preliminary Data Analysis', fontsize=16, fontweight='bold')

# Plot 1: Levels of consumption and income
ax1 = axes[0, 0]
ax1.plot(data_clean['date'], data_clean['c'], label='Log Real Consumption (c)', linewidth=2)
ax1.plot(data_clean['date'], data_clean['y'], label='Log Real Income (y)', linewidth=2, alpha=0.8)
ax1.set_title('Log Consumption and Income (Levels)', fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Log of Real Values')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Growth rates
ax2 = axes[0, 1]
ax2.plot(data_clean['date'], data_clean['delta_c'] * 100, label='Δc (Consumption Growth)', linewidth=1.5)
ax2.plot(data_clean['date'], data_clean['delta_y'] * 100, label='Δy (Income Growth)', linewidth=1.5, alpha=0.8)
ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
ax2.set_title('Quarterly Growth Rates (%)', fontweight='bold')
ax2.set_xlabel('Date')
ax2.set_ylabel('Quarterly Growth Rate (%)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Scatter plot of consumption vs income growth
ax3 = axes[1, 0]
ax3.scatter(data_clean['delta_y'], data_clean['delta_c'], alpha=0.6, s=30)
# Add regression line
z = np.polyfit(data_clean['delta_y'].dropna(), data_clean['delta_c'].dropna(), 1)
p = np.poly1d(z)
ax3.plot(data_clean['delta_y'], p(data_clean['delta_y']), "r--", linewidth=2, 
         label=f'Fitted: slope={z[0]:.3f}')
ax3.set_title('Consumption Growth vs Income Growth', fontweight='bold')
ax3.set_xlabel('Δy (Income Growth)')
ax3.set_ylabel('Δc (Consumption Growth)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Real interest rate
ax4 = axes[1, 1]
ax4.plot(data_clean['date'], data_clean['r_real_filled'], linewidth=2, color='green')
ax4.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
ax4.set_title('Real Interest Rate', fontweight='bold')
ax4.set_xlabel('Date')
ax4.set_ylabel('Real Interest Rate (%)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, 'pih_preliminary_analysis.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"  Visualization saved: {plot_path}")
print()

# ==============================================================================
# STEP 15: EXPORT CLEANED DATASET
# ==============================================================================
print("="*80)
print("STEP 15: EXPORTING FINAL DATASET")
print("="*80)
print()

# Select key variables for econometric analysis
export_vars = [
    'date',
    'C_real', 'Y_real',  # Real levels
    'c', 'y',  # Log levels
    'delta_c', 'delta_y',  # Growth rates (first differences)
    'delta_c_lag1', 'delta_y_lag1',  # Lagged growth rates
    'r_real_filled',  # Real interest rate
    'pi_expected_filled',  # Expected inflation
]

data_export = data_clean[export_vars].copy()

# Save to CSV
csv_path = os.path.join(OUTPUT_DIR, 'pih_data_cleaned.csv')
data_export.to_csv(csv_path, index=False)
print(f"  Dataset exported: {csv_path}")
print(f"  Number of observations: {len(data_export)}")
print(f"  Time period: {data_export['date'].min()} to {data_export['date'].max()}")
print()

# Also save to Excel for easier viewing
excel_path = os.path.join(OUTPUT_DIR, 'pih_data_cleaned.xlsx')
try:
    data_export.to_excel(excel_path, index=False, sheet_name='PIH_Data')
    print(f"  Dataset also saved as: {excel_path}")
except ImportError:
    print("  Note: Excel export requires 'openpyxl'. Install with: pip install openpyxl")
    print("  CSV file is still available for analysis.")
print()

# ==============================================================================
# STEP 16: SUMMARY REPORT
# ==============================================================================
print("="*80)
print("SUMMARY REPORT")
print("="*80)
print()

print("DATA PREPARATION COMPLETE!")
print()
print("Key Findings from Preliminary Analysis:")
print()

print("1. UNIT ROOT TESTS:")
if adf_c_level['p_value'] >= 0.05:
    print("   - Consumption (level): Has unit root → Non-stationary ✓")
else:
    print("   - Consumption (level): Stationary (unexpected)")
    
if adf_y_level['p_value'] >= 0.05:
    print("   - Income (level): Has unit root → Non-stationary ✓")
else:
    print("   - Income (level): Stationary (unexpected)")
    
if adf_dc['p_value'] < 0.05:
    print("   - Consumption growth (Δc): Stationary → Valid for regression ✓")
else:
    print("   - Consumption growth (Δc): Still has unit root (problematic)")
    
if adf_dy['p_value'] < 0.05:
    print("   - Income growth (Δy): Stationary → Valid for regression ✓")
else:
    print("   - Income growth (Δy): Still has unit root (problematic)")
print()

print("2. COINTEGRATION:")
if coint_result['adf_stat'] < -3.5:
    print(f"   - Evidence of cointegration between c and y (ADF={coint_result['adf_stat']:.4f})")
    print(f"   - Long-run consumption/income ratio: {beta_coint[1]:.4f}")
else:
    print(f"   - Weak evidence of cointegration (ADF={coint_result['adf_stat']:.4f})")
print()

print("3. PIH PRELIMINARY INDICATORS:")
corr_dc_dy_lag = correlation_matrix.loc['delta_c', 'delta_y_lag1']
corr_dc_dc_lag = correlation_matrix.loc['delta_c', 'delta_c_lag1']

if abs(corr_dc_dy_lag) < 0.1:
    print(f"   - Corr(Δc_t, Δy_{{t-1}}) = {corr_dc_dy_lag:.4f} → Weak (consistent with PIH) ✓")
else:
    print(f"   - Corr(Δc_t, Δy_{{t-1}}) = {corr_dc_dy_lag:.4f} → Significant (excess sensitivity)")
    
if abs(corr_dc_dc_lag) < 0.1:
    print(f"   - Corr(Δc_t, Δc_{{t-1}}) = {corr_dc_dc_lag:.4f} → Weak (consistent with martingale) ✓")
else:
    print(f"   - Corr(Δc_t, Δc_{{t-1}}) = {corr_dc_dc_lag:.4f} → Significant autocorrelation")
print()

print("4. NEXT STEPS:")
print("   - Proceed with Hall regression tests")
print("   - Estimate Campbell-Mankiw model for λ (rule-of-thumb fraction)")
print("   - Decompose income into permanent/transitory components using VAR")
print("   - Consider instrumental variables for endogeneity")
print("   - Conduct subsample analysis (pre/post Great Recession, etc.)")
print()

print("="*80)
print("ANALYSIS COMPLETE - Ready for econometric testing!")
print("="*80)
print()
print(f"All output files saved in: ./{OUTPUT_DIR}/")
print("  - pih_data_cleaned.csv (cleaned dataset)")
print("  - pih_data_cleaned.xlsx (cleaned dataset in Excel)")
print("  - pih_preliminary_analysis.png (visualization)")
print()
