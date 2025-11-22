import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.weightstats import DescrStatsW

# Load your data — set your actual filename/path
df = pd.read_csv('data/observational_dataset.csv')

# -------- 1. Data Preprocessing --------
# Impute missing values
df = df.fillna(df.median(numeric_only=True))
df = df.fillna({col: df[col].mode()[0] for col in df.select_dtypes('object')})

# Define core columns
treatment_col = 'treatment'   # CHANGE to your column name
outcome_col = 'outcome'       # CHANGE to your column name
covariate_cols = [c for c in df.columns if c not in [treatment_col, outcome_col]]

# Scale covariates
scaler = StandardScaler()
df[covariate_cols] = scaler.fit_transform(df[covariate_cols])

X = df[covariate_cols]
T = df[treatment_col]
Y = df[outcome_col]

# -------- 2. Propensity Score Model --------
ps_model = LogisticRegression(max_iter=1000)
ps_model.fit(X, T)
ps_scores = ps_model.predict_proba(X)[:, 1]

roc_auc = roc_auc_score(T, ps_scores)

import matplotlib.pyplot as plt
plt.hist(ps_scores[T==1], bins=20, alpha=0.5, label='Treated')
plt.hist(ps_scores[T==0], bins=20, alpha=0.5, label='Control')
plt.legend()
plt.title('Propensity Score Distribution')
plt.savefig('propensity_scores.png')
plt.close()

# -------- 2. Outcome Model --------
treated_X, treated_Y = X[T == 1], Y[T == 1]
control_X, control_Y = X[T == 0], Y[T == 0]
outcome_model_treated = LinearRegression().fit(treated_X, treated_Y)
outcome_model_control = LinearRegression().fit(control_X, control_Y)

mu1_hat = outcome_model_treated.predict(X)
mu0_hat = outcome_model_control.predict(X)

# -------- 3. Doubly Robust Estimator (AIPW) --------
aipw_term1 = T * (Y - mu1_hat) / ps_scores
aipw_term2 = (1 - T) * (Y - mu0_hat) / (1 - ps_scores)
dr_ate_estimates = aipw_term1 - aipw_term2 + mu1_hat - mu0_hat
dr_ate = np.mean(dr_ate_estimates)

def bootstrap_ate(vals, n_boot=1000):
    n = len(vals)
    samples = [np.mean(np.random.choice(vals, n, replace=True)) for _ in range(n_boot)]
    return np.percentile(samples, 2.5), np.percentile(samples, 97.5)

dr_ci = bootstrap_ate(dr_ate_estimates)

# -------- 3B. OLS Regression --------
ols_model = LinearRegression()
ols_model.fit(pd.concat([X, T], axis=1), Y)
ols_ate = ols_model.coef_[-1]

# -------- 3C. IPW --------
ipw_ate = np.mean((T * Y / ps_scores) - ((1-T) * Y / (1-ps_scores)))

# -------- 4. Sensitivity Analysis: Random Forest Outcome Model --------
rf_treated = RandomForestRegressor(n_estimators=100, random_state=42).fit(treated_X, treated_Y)
rf_control = RandomForestRegressor(n_estimators=100, random_state=42).fit(control_X, control_Y)
rf_mu1_hat = rf_treated.predict(X)
rf_mu0_hat = rf_control.predict(X)
rf_dr_estimates = T * (Y - rf_mu1_hat) / ps_scores - (1 - T) * (Y - rf_mu0_hat) / (1 - ps_scores) + rf_mu1_hat - rf_mu0_hat
rf_dr_ate = np.mean(rf_dr_estimates)
rf_dr_ci = bootstrap_ate(rf_dr_estimates)

# -------- Results --------
results = {
    "Doubly Robust (AIPW) ATE": dr_ate,
    "95% CI": dr_ci,
    "OLS Regression Estimate": ols_ate,
    "IPW Estimate": ipw_ate,
    "Sensitivity (DR with RF Outcome)": rf_dr_ate,
    "95% CI (RF)": rf_dr_ci,
    "Propensity Model ROC AUC": roc_auc
}
for k, v in results.items():
    print(f"{k}: {v}")
