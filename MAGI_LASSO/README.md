# Side-by-Side Benchmarks: MAGI vs LASSO

This README explains how to run **apples-to-apples** comparisons between **MAGI** (Dependent Bayes with Temporal Ordering) and **LASSO logistic regression**. The goal is to isolate the modeling effect—not differences in cohorts, features, or splits.

---

## 1) Objectives

- Use the **same cohort**, **same outcome definition** (including SNOMED expansion if enabled), and **same data splits**.
- Apply **identical feature governance** (duplicates/constants removal, EPV limits).
- Optionally **lock both models to the same final feature set** (e.g., top-|β| or top-prevalence) to isolate the learning rule.

---

## 2) Inputs & Configuration

- **Cohort & counts**: SQLite or CSV with edge counts for k→T pairs  
  Required columns (or equivalents):  
  `n_code_target, n_code_no_target, n_target_no_code, n_no_target_no_code`
- **Outcome T**: A target code (SNOMED, RxNorm, CPT, etc.).  
  Optional: **descendants (+/- parents)** expansion for SNOMED targets.
- **Splits**: fixed CV folds or train/valid/test with a shared random seed.
- **EPV settings**: events-per-variable thresholds (e.g., target EPV≈9, min EPV≥5).
- **Ranking mode** (pick one):
  - `total_effect` (MAGI-derived ranking)
  - `abs_beta` (LASSO |β| from a pilot fit)
  - `prevalence` (most frequent predictors)
- **Bootstrap** (optional): .632 or .632+ for small samples.
