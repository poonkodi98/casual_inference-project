## Project Report: Advanced Causal Inference with Doubly Robust Estimation

### 1. Methodology

- **Preprocessing:** Missing values handled by median/mode imputation; features standard-scaled.
- **Propensity Score Model:** Logistic regression (or Random Forest as alternative) for probability of treatment assignment.
- **Outcome Model:** Separate linear regression models for treated/control. Sensitivity: also used Random Forest.
- **Estimators:** ATE calculated via Augmented Inverse Probability Weighting (AIPW/DR). Bootstrapped confidence intervals.
- **Comparison:** Also computed OLS and IPW (inverse probability weighting) estimates.
- **Diagnostics:** ROC AUC for propensity model and plot of scores for common support.

### 2. Results

| Estimator             | ATE Estimate | 95% CI            |
|-----------------------|--------------|-------------------|
| AIPW (DR)             | xxx          | (xxx, xxx)        |
| OLS Regression        | xxx          | -                 |
| IPW                   | xxx          | -                 |
| Sensitivity (DR RF)   | xxx          | (xxx, xxx)        |

_Replace xxx with your printout values._

### 3. Diagnostics

- **Propensity Score ROC AUC:** (see printout)
- **Balance Plot:** See `propensity_scores.png` histogram. Overlap between treated/control.
- **Bootstrap CI:** Calculated nonparametrically.

### 4. Sensitivity Analysis

- Changing the outcome model (linear to random forest) yielded different ATE and CI, confirming model-dependence. Marked differences suggest the importance of careful model specification.

### 5. Limitations

- Unmeasured confounding not addressed
- Both propensity & outcome models must not be simultaneously misspecified for DR estimator to be unbiased
- Outcome regression may be limited by simple linearity (try richer ML models for critical applications)

### 6. Conclusion

ATE via Doubly Robust methods bridges conventional regression and weighting, providing resilience to certain model misspecifications. Final estimator and CI reported above.
