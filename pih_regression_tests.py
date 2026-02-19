"""
Permanent Income Hypothesis - Econometric Testing
==================================================

This script implements the main empirical tests of the Permanent Income Hypothesis:

1. Hall (1978) Regression Test
2. Excess Sensitivity Tests
3. Campbell-Mankiw (1989) Model
4. Instrumental Variables Estimation
5. Subsample Analysis

Theoretical Foundation:
-----------------------
Under PIH with quadratic utility and rational expectations, consumption follows
a martingale. This implies:

    E_t[C_{t+1}] = C_t
    
Therefore: ΔC_{t+1} = ε_{t+1} where E_t[ε_{t+1}] = 0

Any variable in the information set Ω_t should have zero predictive power for
future consumption changes.

Author: [Your Name]
Date: 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import t as t_dist, f as f_dist
import warnings
import os
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)
np.set_printoptions(precision=4, suppress=True)

print("="*80)
print("PERMANENT INCOME HYPOTHESIS - ECONOMETRIC TESTING")
print("="*80)
print()

# ==============================================================================
# LOAD CLEANED DATA
# ==============================================================================
print("Loading cleaned dataset...")

# Try to load from pih_output directory first, then current directory
data_paths = ['pih_output/pih_data_cleaned.csv', 'pih_data_cleaned.csv']
data = None

for path in data_paths:
    try:
        data = pd.read_csv(path)
        print(f"  ✓ Data loaded from: {path}")
        break
    except FileNotFoundError:
        continue

if data is None:
    print("ERROR: Could not find pih_data_cleaned.csv")
    print("Please run pih_data_preparation.py first to generate the cleaned dataset.")
    exit(1)

print(f"  Sample: {len(data)} observations")
print()

# Create output directory for results
OUTPUT_DIR = 'pih_regression_results'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==============================================================================
# HELPER FUNCTIONS FOR REGRESSION ANALYSIS
# ==============================================================================

def ols_regression(y, X, var_names=None):
    """
    Perform OLS regression with robust standard errors.
    
    Model: y = X*β + ε
    
    Returns dictionary with:
    - coefficients
    - standard errors (Newey-West HAC)
    - t-statistics
    - p-values
    - R-squared
    - F-statistic
    """
    n = len(y)
    k = X.shape[1]
    
    # OLS estimation
    from numpy.linalg import inv
    beta = inv(X.T @ X) @ (X.T @ y)
    residuals = y - X @ beta
    
    # Calculate R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k)
    
    # Newey-West HAC standard errors (lag = 4 for quarterly data)
    # This accounts for heteroskedasticity and autocorrelation
    lags = 4
    omega = np.zeros((k, k))
    
    # Initial variance
    X_weighted = X * residuals[:, np.newaxis]
    omega = (X_weighted.T @ X_weighted) / n
    
    # Add autocovariance terms with Bartlett weights
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)  # Bartlett kernel
        gamma = np.zeros((k, k))
        
        for t in range(lag, n):
            gamma += np.outer(X[t] * residuals[t], X[t-lag] * residuals[t-lag])
        
        gamma = gamma / n
        omega += weight * (gamma + gamma.T)
    
    # Variance-covariance matrix
    var_cov = inv(X.T @ X / n) @ omega @ inv(X.T @ X / n) / n
    std_errors = np.sqrt(np.diag(var_cov))
    
    # t-statistics and p-values
    t_stats = beta / std_errors
    p_values = 2 * (1 - t_dist.cdf(np.abs(t_stats), n - k))
    
    # F-statistic for joint significance (excluding constant)
    # H0: all slope coefficients = 0
    if k > 1:
        R = np.eye(k)[1:]  # Restriction matrix (exclude constant)
        r = np.zeros(k-1)  # Zero restrictions
        
        # Wald test statistic
        restriction = R @ beta - r
        middle = inv(R @ var_cov @ R.T)
        f_stat = (restriction.T @ middle @ restriction) / (k - 1)
        f_pval = 1 - f_dist.cdf(f_stat, k-1, n-k)
    else:
        f_stat = np.nan
        f_pval = np.nan
    
    results = {
        'coefficients': beta,
        'std_errors': std_errors,
        't_stats': t_stats,
        'p_values': p_values,
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        'f_stat': f_stat,
        'f_pval': f_pval,
        'residuals': residuals,
        'n_obs': n,
        'n_vars': k,
        'var_names': var_names if var_names else [f'X{i}' for i in range(k)]
    }
    
    return results


def print_regression_table(results, title="Regression Results"):
    """Print formatted regression results table"""
    print()
    print("="*80)
    print(title)
    print("="*80)
    print()
    
    # Coefficients table
    print(f"{'Variable':<20} {'Coef':>10} {'Std Err':>10} {'t-stat':>10} {'P>|t|':>10} {'Signif':>8}")
    print("-"*80)
    
    for i, var in enumerate(results['var_names']):
        coef = results['coefficients'][i]
        se = results['std_errors'][i]
        t = results['t_stats'][i]
        p = results['p_values'][i]
        
        # Significance stars
        if p < 0.01:
            sig = "***"
        elif p < 0.05:
            sig = "**"
        elif p < 0.10:
            sig = "*"
        else:
            sig = ""
        
        print(f"{var:<20} {coef:>10.6f} {se:>10.6f} {t:>10.3f} {p:>10.4f} {sig:>8}")
    
    print("-"*80)
    print(f"{'N':<20} {results['n_obs']:>10}")
    
    if 'r_squared' in results:
        print(f"{'R-squared':<20} {results['r_squared']:>10.4f}")
        print(f"{'Adj. R-squared':<20} {results['adj_r_squared']:>10.4f}")
    
    if not np.isnan(results.get('f_stat', np.nan)):
        print(f"{'F-statistic':<20} {results['f_stat']:>10.4f}")
        print(f"{'Prob(F)':<20} {results['f_pval']:>10.4f}")
    
    print()
    print("Significance: *** p<0.01, ** p<0.05, * p<0.10")
    print("Standard errors: Newey-West HAC (4 lags)")
    print()


def instrumental_variables(y, X, Z, var_names=None, instrument_names=None):
    """
    Two-stage least squares (2SLS) estimation.
    
    Stage 1: X = Z*π + v
    Stage 2: y = X*β + ε
    
    Where Z are instruments that satisfy:
    - Relevance: Corr(Z, X) ≠ 0
    - Exogeneity: Corr(Z, ε) = 0
    """
    n = len(y)
    
    # Stage 1: Regress endogenous variables on instruments
    from numpy.linalg import inv
    pi_hat = inv(Z.T @ Z) @ (Z.T @ X)
    X_fitted = Z @ pi_hat
    
    # Stage 2: Use fitted values
    beta_2sls = inv(X_fitted.T @ X_fitted) @ (X_fitted.T @ y)
    residuals = y - X @ beta_2sls
    
    # Standard errors for 2SLS
    sigma2 = np.sum(residuals**2) / (n - X.shape[1])
    var_beta = sigma2 * inv(X_fitted.T @ X_fitted)
    std_errors = np.sqrt(np.diag(var_beta))
    
    # Test statistics
    t_stats = beta_2sls / std_errors
    p_values = 2 * (1 - t_dist.cdf(np.abs(t_stats), n - X.shape[1]))
    
    # First-stage F-statistic (instrument strength)
    # For each endogenous variable, test if instruments are relevant
    first_stage_f = []
    for j in range(X.shape[1]):
        X_j = X[:, j]
        resid_j = X_j - Z @ (inv(Z.T @ Z) @ (Z.T @ X_j))
        ss_res = np.sum(resid_j**2)
        ss_tot = np.sum((X_j - np.mean(X_j))**2)
        r2 = 1 - ss_res/ss_tot
        f = (r2 / (Z.shape[1] - 1)) / ((1 - r2) / (n - Z.shape[1]))
        first_stage_f.append(f)
    
    results = {
        'coefficients': beta_2sls,
        'std_errors': std_errors,
        't_stats': t_stats,
        'p_values': p_values,
        'residuals': residuals,
        'first_stage_f': first_stage_f,
        'n_obs': n,
        'n_vars': X.shape[1],
        'var_names': var_names if var_names else [f'X{i}' for i in range(X.shape[1])],
        'instrument_names': instrument_names
    }
    
    return results


# ==============================================================================
# TEST 1: HALL (1978) BASIC REGRESSION
# ==============================================================================
print("="*80)
print("TEST 1: HALL (1978) REGRESSION")
print("="*80)
print()

print("Theoretical Model:")
print("  Under PIH with rational expectations:")
print("    ΔC_{t+1} = α + ε_{t+1}")
print()
print("  Extended version testing for predictability:")
print("    ΔC_{t+1} = α + β₁·ΔC_t + β₂·ΔY_t + β₃·r_t + ε_{t+1}")
print()
print("  Null Hypothesis (PIH): β₁ = β₂ = β₃ = 0")
print("  Alternative: At least one β ≠ 0 (PIH violation)")
print()

# Prepare variables (need to lead consumption growth by one period)
data['delta_c_lead'] = data['delta_c'].shift(-1)

# Remove last observation (since we need future consumption)
hall_data = data[:-1].dropna(subset=['delta_c_lead', 'delta_c', 'delta_y', 'r_real_filled']).copy()

# Dependent variable: ΔC_{t+1}
y_hall = hall_data['delta_c_lead'].values

# Independent variables: constant, ΔC_t, ΔY_t, r_t
X_hall = np.column_stack([
    np.ones(len(hall_data)),
    hall_data['delta_c'].values,
    hall_data['delta_y'].values,
    hall_data['r_real_filled'].values
])

var_names_hall = ['Constant', 'ΔC_t', 'ΔY_t', 'r_t']

# Run OLS
results_hall = ols_regression(y_hall, X_hall, var_names_hall)
print_regression_table(results_hall, "Hall (1978) Test - OLS Estimation")

# Interpretation
print("INTERPRETATION:")
print("-" * 80)

significant_vars = []
for i in range(1, len(results_hall['var_names'])):
    if results_hall['p_values'][i] < 0.10:
        significant_vars.append(results_hall['var_names'][i])

if significant_vars:
    print(f"✗ PIH REJECTED: Variables {', '.join(significant_vars)} are significant predictors")
    print(f"  This indicates EXCESS SENSITIVITY - consumption responds to predictable")
    print(f"  income changes, violating the random walk hypothesis.")
else:
    print("✓ PIH NOT REJECTED: No variables significantly predict future consumption growth")
    print("  Consumption changes appear unpredictable, consistent with PIH.")

print()

# Joint test of all slope coefficients
if results_hall['f_pval'] < 0.05:
    print(f"Joint F-test: F = {results_hall['f_stat']:.3f}, p = {results_hall['f_pval']:.4f}")
    print("  ✗ Reject H0: Variables jointly predict consumption (PIH violation)")
else:
    print(f"Joint F-test: F = {results_hall['f_stat']:.3f}, p = {results_hall['f_pval']:.4f}")
    print("  ✓ Cannot reject H0: Variables don't jointly predict consumption")

print()

# ==============================================================================
# TEST 2: EXCESS SENSITIVITY TO LAGGED INCOME
# ==============================================================================
print("="*80)
print("TEST 2: EXCESS SENSITIVITY TEST")
print("="*80)
print()

print("Theoretical Model:")
print("  PIH predicts that only UNEXPECTED income changes affect consumption.")
print("  If consumers are forward-looking, lagged income growth should not predict")
print("  current consumption growth.")
print()
print("  Test regression:")
print("    ΔC_t = α + β₁·ΔY_{t-1} + β₂·ΔY_{t-2} + ε_t")
print()
print("  H0 (PIH): β₁ = β₂ = 0")
print()

# Create additional lags
data['delta_y_lag2'] = data['delta_y'].shift(2)

# Prepare data
excess_data = data.dropna(subset=['delta_c', 'delta_y_lag1', 'delta_y_lag2']).copy()

y_excess = excess_data['delta_c'].values
X_excess = np.column_stack([
    np.ones(len(excess_data)),
    excess_data['delta_y_lag1'].values,
    excess_data['delta_y_lag2'].values
])

var_names_excess = ['Constant', 'ΔY_{t-1}', 'ΔY_{t-2}']

# Run OLS
results_excess = ols_regression(y_excess, X_excess, var_names_excess)
print_regression_table(results_excess, "Excess Sensitivity Test - Lagged Income")

print("INTERPRETATION:")
print("-" * 80)

if results_excess['f_pval'] < 0.05:
    print(f"✗ EXCESS SENSITIVITY DETECTED (F = {results_excess['f_stat']:.3f}, p = {results_excess['f_pval']:.4f})")
    print("  Past income changes significantly predict current consumption.")
    print("  Evidence against pure PIH - suggests liquidity constraints or rule-of-thumb behavior.")
else:
    print(f"✓ No excess sensitivity (F = {results_excess['f_stat']:.3f}, p = {results_excess['f_pval']:.4f})")
    print("  Lagged income doesn't predict consumption, consistent with PIH.")

print()

# ==============================================================================
# TEST 3: CAMPBELL-MANKIW (1989) MODEL
# ==============================================================================
print("="*80)
print("TEST 3: CAMPBELL-MANKIW (1989) RULE-OF-THUMB MODEL")
print("="*80)
print()

print("Theoretical Model:")
print("  Assume fraction λ of consumers are 'rule-of-thumb' (consume current income)")
print("  and fraction (1-λ) follow PIH.")
print()
print("  Aggregate consumption growth:")
print("    ΔC_t = λ·ΔY_t + (1-λ)·[consumption growth from PIH consumers]")
print()
print("  Under rational expectations for PIH consumers:")
print("    ΔC_t = λ·ΔY_t + ε_t")
print()
print("  where λ measures the fraction of liquidity-constrained consumers.")
print()
print("  H0: λ = 0 (all consumers follow PIH)")
print("  H1: λ > 0 (some consumers are rule-of-thumb)")
print()

# Prepare data
cm_data = data.dropna(subset=['delta_c', 'delta_y']).copy()

y_cm = cm_data['delta_c'].values
X_cm = np.column_stack([
    np.ones(len(cm_data)),
    cm_data['delta_y'].values
])

var_names_cm = ['Constant', 'λ (ΔY_t)']

# Run OLS
results_cm = ols_regression(y_cm, X_cm, var_names_cm)
print_regression_table(results_cm, "Campbell-Mankiw Model - Basic Specification")

# Extract lambda estimate
lambda_est = results_cm['coefficients'][1]
lambda_se = results_cm['std_errors'][1]

print("INTERPRETATION:")
print("-" * 80)
print(f"Estimated λ (rule-of-thumb fraction): {lambda_est:.4f}")
print(f"Standard error: {lambda_se:.4f}")
print(f"95% Confidence interval: [{lambda_est - 1.96*lambda_se:.4f}, {lambda_est + 1.96*lambda_se:.4f}]")
print()

if results_cm['p_values'][1] < 0.05 and lambda_est > 0:
    pct = lambda_est * 100
    print(f"✗ SIGNIFICANT RULE-OF-THUMB BEHAVIOR")
    print(f"  Approximately {pct:.1f}% of aggregate consumption appears to be")
    print(f"  from liquidity-constrained consumers who consume current income.")
    print(f"  This represents a significant violation of the pure PIH.")
elif results_cm['p_values'][1] < 0.05 and lambda_est < 0:
    print(f"⚠ NEGATIVE λ (unexpected result)")
    print(f"  This suggests consumption growth is NEGATIVELY related to income growth.")
    print(f"  Possible explanations: precautionary saving, measurement error, or structural breaks.")
else:
    print(f"✓ NO SIGNIFICANT RULE-OF-THUMB BEHAVIOR")
    print(f"  The data are consistent with the pure PIH (λ ≈ 0).")

print()

# ==============================================================================
# TEST 4: INSTRUMENTAL VARIABLES ESTIMATION
# ==============================================================================
print("="*80)
print("TEST 4: INSTRUMENTAL VARIABLES (IV) ESTIMATION")
print("="*80)
print()

print("Motivation:")
print("  Income and consumption are determined simultaneously, causing endogeneity bias.")
print("  OLS estimates of λ may be biased upward (overestimate excess sensitivity).")
print()
print("  Solution: Use instrumental variables that are:")
print("    1. Correlated with income growth (relevance)")
print("    2. Uncorrelated with consumption shocks (exogeneity)")
print()
print("  Instruments used:")
print("    - ΔY_{t-2}, ΔY_{t-3}: Lagged income growth")
print("    - r_{t-1}: Lagged real interest rate")
print()

# Create additional lags for instruments
data['delta_y_lag3'] = data['delta_y'].shift(3)
data['r_lag1'] = data['r_real_filled'].shift(1)

# Prepare IV data
iv_data = data.dropna(subset=['delta_c', 'delta_y', 'delta_y_lag2', 
                               'delta_y_lag3', 'r_lag1']).copy()

# Dependent variable
y_iv = iv_data['delta_c'].values

# Endogenous variable (with constant)
X_iv = np.column_stack([
    np.ones(len(iv_data)),
    iv_data['delta_y'].values
])

# Instruments (including constant)
Z_iv = np.column_stack([
    np.ones(len(iv_data)),
    iv_data['delta_y_lag2'].values,
    iv_data['delta_y_lag3'].values,
    iv_data['r_lag1'].values
])

var_names_iv = ['Constant', 'λ (ΔY_t)']
instrument_names = ['Constant', 'ΔY_{t-2}', 'ΔY_{t-3}', 'r_{t-1}']

# Run 2SLS
results_iv = instrumental_variables(y_iv, X_iv, Z_iv, var_names_iv, instrument_names)

print_regression_table(results_iv, "Campbell-Mankiw Model - IV Estimation (2SLS)")

print("First-Stage F-Statistics (Instrument Strength):")
print("-" * 80)
for i, var in enumerate(var_names_iv):
    if i > 0:  # Skip constant
        print(f"  {var}: F = {results_iv['first_stage_f'][i]:.2f}")
        if results_iv['first_stage_f'][i] > 10:
            print(f"    ✓ Strong instruments (F > 10)")
        else:
            print(f"    ⚠ Weak instruments (F < 10) - IV estimates may be unreliable")
print()

# Compare OLS vs IV
lambda_ols = results_cm['coefficients'][1]
lambda_iv = results_iv['coefficients'][1]

print("COMPARISON: OLS vs IV")
print("-" * 80)
print(f"λ (OLS):  {lambda_ols:.4f} (SE: {results_cm['std_errors'][1]:.4f})")
print(f"λ (IV):   {lambda_iv:.4f} (SE: {results_iv['std_errors'][1]:.4f})")
print(f"Difference: {lambda_ols - lambda_iv:.4f}")
print()

if abs(lambda_ols - lambda_iv) > 0.1:
    print("⚠ SUBSTANTIAL DIFFERENCE between OLS and IV")
    print("  This suggests significant endogeneity bias in OLS.")
    print("  IV estimate is more reliable for causal inference.")
else:
    print("✓ OLS and IV estimates are similar")
    print("  Endogeneity bias appears to be small.")

print()

# ==============================================================================
# TEST 5: SUBSAMPLE ANALYSIS
# ==============================================================================
print("="*80)
print("TEST 5: SUBSAMPLE ANALYSIS - STRUCTURAL STABILITY")
print("="*80)
print()

print("Testing whether PIH violations have changed over time.")
print("Periods analyzed:")
print("  1. Pre-Great Recession (1980-2007)")
print("  2. Post-Great Recession (2008-2025)")
print()

# Split data
data['year'] = pd.to_datetime(data['date']).dt.year
pre_crisis = data[data['year'] < 2008].copy()
post_crisis = data[data['year'] >= 2008].copy()

# Run Campbell-Mankiw for each subsample
subsample_results = {}

for name, subsample in [('Pre-2008', pre_crisis), ('Post-2008', post_crisis)]:
    sub_clean = subsample.dropna(subset=['delta_c', 'delta_y'])
    
    if len(sub_clean) < 30:  # Need reasonable sample size
        print(f"⚠ {name}: Insufficient observations ({len(sub_clean)})")
        continue
    
    y_sub = sub_clean['delta_c'].values
    X_sub = np.column_stack([
        np.ones(len(sub_clean)),
        sub_clean['delta_y'].values
    ])
    
    results_sub = ols_regression(y_sub, X_sub, ['Constant', 'λ'])
    subsample_results[name] = results_sub
    
    print(f"\n{name} Period:")
    print(f"  λ = {results_sub['coefficients'][1]:.4f} (SE: {results_sub['std_errors'][1]:.4f})")
    print(f"  t-stat = {results_sub['t_stats'][1]:.3f}, p-value = {results_sub['p_values'][1]:.4f}")
    print(f"  N = {results_sub['n_obs']}, R² = {results_sub['r_squared']:.4f}")

if len(subsample_results) == 2:
    lambda_pre = subsample_results['Pre-2008']['coefficients'][1]
    lambda_post = subsample_results['Post-2008']['coefficients'][1]
    
    print()
    print("TEMPORAL COMPARISON:")
    print("-" * 80)
    print(f"Change in λ: {lambda_post - lambda_pre:+.4f}")
    
    if lambda_post > lambda_pre + 0.1:
        print("  → INCREASE in rule-of-thumb behavior post-2008")
        print("  Possible explanation: Financial crisis increased liquidity constraints")
    elif lambda_post < lambda_pre - 0.1:
        print("  → DECREASE in rule-of-thumb behavior post-2008")
        print("  Possible explanation: Policy interventions or structural changes")
    else:
        print("  → STABLE rule-of-thumb behavior across periods")

print()

# ==============================================================================
# SAVE ALL RESULTS
# ==============================================================================
print("="*80)
print("SAVING RESULTS")
print("="*80)
print()

# Create summary dataframe
summary_data = {
    'Test': ['Hall (1978)', 'Excess Sensitivity', 'Campbell-Mankiw (OLS)', 
             'Campbell-Mankiw (IV)', 'Pre-2008', 'Post-2008'],
    'Lambda': [np.nan, np.nan, lambda_est, lambda_iv, 
               lambda_pre if 'Pre-2008' in subsample_results else np.nan,
               lambda_post if 'Post-2008' in subsample_results else np.nan],
    'F_Stat': [results_hall['f_stat'], results_excess['f_stat'], 
               np.nan, np.nan, np.nan, np.nan],
    'P_Value': [results_hall['f_pval'], results_excess['f_pval'],
                results_cm['p_values'][1], results_iv['p_values'][1],
                subsample_results.get('Pre-2008', {}).get('p_values', [np.nan, np.nan])[1],
                subsample_results.get('Post-2008', {}).get('p_values', [np.nan, np.nan])[1]],
    'N_Obs': [results_hall['n_obs'], results_excess['n_obs'], 
              results_cm['n_obs'], results_iv['n_obs'],
              subsample_results.get('Pre-2008', {}).get('n_obs', np.nan),
              subsample_results.get('Post-2008', {}).get('n_obs', np.nan)]
}

summary_df = pd.DataFrame(summary_data)
summary_path = os.path.join(OUTPUT_DIR, 'pih_test_summary.csv')
summary_df.to_csv(summary_path, index=False)

print(f"Summary results saved: {summary_path}")
print()

# ==============================================================================
# CREATE VISUALIZATION
# ==============================================================================
print("Creating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('PIH Testing Results - Visual Summary', fontsize=16, fontweight='bold')

# Plot 1: Actual vs Fitted (Campbell-Mankiw)
ax1 = axes[0, 0]
fitted_cm = X_cm @ results_cm['coefficients']
ax1.scatter(cm_data['delta_y'], cm_data['delta_c'], alpha=0.5, s=30, label='Actual')
sort_idx = np.argsort(cm_data['delta_y'])
ax1.plot(cm_data['delta_y'].values[sort_idx], fitted_cm[sort_idx], 
         'r-', linewidth=2, label=f'Fitted (λ={lambda_est:.3f})')
ax1.set_xlabel('Income Growth (ΔY)')
ax1.set_ylabel('Consumption Growth (ΔC)')
ax1.set_title('Campbell-Mankiw Model Fit')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Residuals over time
ax2 = axes[0, 1]
ax2.plot(cm_data['date'], results_cm['residuals'], linewidth=1)
ax2.axhline(y=0, color='r', linestyle='--', linewidth=1)
ax2.set_xlabel('Date')
ax2.set_ylabel('Residuals')
ax2.set_title('Campbell-Mankiw Residuals')
ax2.grid(True, alpha=0.3)

# Plot 3: Lambda estimates across tests
ax3 = axes[1, 0]
lambda_estimates = [lambda_est, lambda_iv]
lambda_ses = [results_cm['std_errors'][1], results_iv['std_errors'][1]]
labels = ['OLS', 'IV']

if 'Pre-2008' in subsample_results and 'Post-2008' in subsample_results:
    lambda_estimates.extend([lambda_pre, lambda_post])
    lambda_ses.extend([subsample_results['Pre-2008']['std_errors'][1],
                       subsample_results['Post-2008']['std_errors'][1]])
    labels.extend(['Pre-2008', 'Post-2008'])

x_pos = np.arange(len(labels))
ax3.bar(x_pos, lambda_estimates, yerr=[1.96*se for se in lambda_ses], 
        capsize=5, alpha=0.7, color=['blue', 'green', 'orange', 'red'][:len(labels)])
ax3.set_xticks(x_pos)
ax3.set_xticklabels(labels)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax3.set_ylabel('λ (Rule-of-Thumb Fraction)')
ax3.set_title('Estimated λ Across Specifications')
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: QQ plot of residuals (normality check)
ax4 = axes[1, 1]
residuals_standardized = results_cm['residuals'] / np.std(results_cm['residuals'])
stats.probplot(residuals_standardized, dist="norm", plot=ax4)
ax4.set_title('Q-Q Plot: Residuals Normality Check')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, 'pih_regression_results.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Visualization saved: {plot_path}")
print()

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print("="*80)
print("FINAL SUMMARY - PIH EMPIRICAL TESTING")
print("="*80)
print()

print("KEY FINDINGS:")
print("-" * 80)
print()

# Hall test
print("1. HALL (1978) TEST:")
if results_hall['f_pval'] < 0.05:
    print("   ✗ REJECTED - Variables in info set predict consumption")
else:
    print("   ✓ NOT REJECTED - Random walk property holds")
print()

# Excess sensitivity
print("2. EXCESS SENSITIVITY:")
if results_excess['f_pval'] < 0.05:
    print("   ✗ DETECTED - Past income predicts current consumption")
else:
    print("   ✓ NOT DETECTED - No sensitivity to past income")
print()

# Campbell-Mankiw
print("3. RULE-OF-THUMB CONSUMERS:")
print(f"   Estimated fraction (λ): {lambda_est:.1%}")
if results_cm['p_values'][1] < 0.05 and lambda_est > 0:
    print("   ✗ SIGNIFICANT - Evidence of liquidity constraints")
else:
    print("   ✓ INSIGNIFICANT - Pure PIH cannot be rejected")
print()

# IV vs OLS
print("4. ENDOGENEITY:")
if abs(lambda_ols - lambda_iv) > 0.1:
    print(f"   ⚠ PRESENT - IV estimate ({lambda_iv:.3f}) differs from OLS")
else:
    print(f"   ✓ MINIMAL - IV and OLS estimates similar")
print()

print("-" * 80)
print("POLICY IMPLICATIONS:")
print()
if lambda_est > 0.3:
    print("  • Large fraction of rule-of-thumb consumers suggests fiscal stimulus")
    print("    (tax cuts, transfers) will have substantial effects on consumption")
    print("  • Liquidity constraints are a significant friction in the economy")
elif lambda_est > 0 and lambda_est <= 0.3:
    print("  • Moderate rule-of-thumb behavior suggests some effectiveness of")
    print("    short-term fiscal policy, but limited compared to permanent changes")
else:
    print("  • Consumption largely follows PIH - only permanent income changes")
    print("    significantly affect consumption")
    print("  • Temporary fiscal stimulus likely to have limited effects")

print()
print("="*80)
print(f"All results saved in: ./{OUTPUT_DIR}/")
print("  - pih_test_summary.csv")
print("  - pih_regression_results.png")
print("="*80)
