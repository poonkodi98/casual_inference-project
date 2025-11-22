# Advanced Causal Inference: Doubly Robust ATE Estimation

Estimate treatment effects in observational data with advanced causal inference, using Augmented Inverse Probability Weighting (AIPW) and multiple benchmarks.


## Usage

1. Place your data as `data/observational_dataset.csv`.
2. Update column names (`treatment_col`, `outcome_col`) in `causal_inference_dr.py`.
3. Install dependencies:
    ```
    pip install pandas numpy scikit-learn matplotlib statsmodels
    ```
4. Run:
    ```
    python causal_inference_dr.py
    ```
5. Review results in the terminal and diagnostics in `propensity_scores.png`.

## Output

- Main results and confidence intervals print in console.
- Diagnostics in PNG/terminal.
- See `report.md` for structure, explanation, and where to paste output for submission.

## Notes

- Code is modular for easy adjustment of models.
- Add or preprocess categorical/special variables as needed.
- Sensitivity via ML models (Random Forest) is provided.

---

This repository demonstrates causal inference skills for non-randomized (“real world”) settings in health, economics, or other impact evaluations.
