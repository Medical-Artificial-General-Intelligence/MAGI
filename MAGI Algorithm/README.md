# MAGI Algorithm (INT-based)

This notebook (`MAGI ALGORITHM.ipynb`) contains a single **core MAGI implementation** in Python:

- `analyze_causal_sequence_py(...)` — an **integer-coded** ( `_int` ) version of the MAGI causal sequence analysis.

The code is designed to run on a **pairwise “edge” table** (MAGI DB) where each row represents a relationship between:
- **LEFT / target event:** `target_concept_code_int` (written as “j” or “Y”)
- **RIGHT / predictor event:** `concept_code_int` (written as “k”)

It computes:
1. A **temporal ordering** of events (concepts) based on temporal count differences
2. A **total-effect** term `T_{kY}` (from DB `total_effects` when present)
3. A **λ (lambda)** adjacency/mixture term `λ_{k,j}`
4. A backward-recursed **D** term (adjusted effect)
5. A logistic-style set of coefficients `β` and a `predict_proba` helper

---

## Requirements

Python packages used:

- `numpy`
- `pandas`
- `typing` (standard library)

---

## Quick start

### 1) Prepare input data (DataFrame or CSV)

You can pass either:

- a **CSV filepath** (`df_in="path/to/edges.csv"`) or
- a **pandas DataFrame** (`df_in=df_edges`)

### 2) Run MAGI

```python
out = analyze_causal_sequence_py(
    df_in=df_edges,               # or a CSV path
    events=None,                  # auto-detect from *_int columns
    force_outcome=None,           # optionally force a specific final node
    lambda_min_count=15           # minimum support for λ terms
)
```
### 3) Inspect outputs

```python
out["temporal_order"]     # the computed event order (final element is outcome)
out["T_val"]              # T_{kY} for each non-outcome node k
out["lambda_l"]           # dict: k -> Series of λ_{k,j} for downstream j
out["D_val"]              # D_k values after backward recursion
out["coef_df"]            # table of β coefficients + intercept
out["predict_proba"]      # function to compute P(Y=1 | Z)
```
### 4) Predict probability for a new feature vector

`predict_proba` accepts multiple formats:

A) Dict keyed by concept_code_int: 
```python
Z = { 1001: 1, 1002: 0, 1003: 1 }   # keys are concept_code_int predictors
p = out["predict_proba"](Z)
```
B) DataFrame with columns matching predictors:
```python
X = pd.DataFrame([{1001: 1, 1002: 0, 1003: 1}]).fillna(0)
p = out["predict_proba"](X)   # returns array
```
C) Numpy array in predictor order:
```python
predictors = out["logit_predictors"]
arr = np.zeros(len(predictors))
arr[0] = 1
p = out["predict_proba"](arr)
```

---
## Input schema (required columns)
`analyze_causal_sequence_py` expects the following columns in the input edge table:

#### Identity columns (integer-coded)
- `target_concept_code_int` (LEFT node; “target” / “j” / sometimes “Y”)
- `concept_code_int` (RIGHT node; “code” / “k” predictor)
   - Note: `name_map` is accepted for compatibility but is ignored in this `_int` version.

#### Count columns (required)
These must exist, and will be coerced to numeric:
- `n_code_target`
- `n_code_no_target`
- `n_target`
- `n_no_target`
- `n_target_no_code`
- `n_code`
- `n_code_before_target`
- `n_target_before_code`

#### Optional column
- `total_effects`

If `total_effects` is present, it is used to compute `T_{kY}`; otherwise the code falls back to `T_{kY} = 1`.

### Column meaning (practical interpretation)

The function treats each `(target_concept_code_int = j, concept_code_int = k)` pair as an “edge row” with aggregated counts.

Common interpretation of the key fields:

- `n_code_target`: count of cases with **k and j**
- `n_code`: count of cases with **k** (any j status)
- `n_target`: count of cases with **j** (any k status)
- `n_no_target`: count of cases without **j**
- `n_target_before_code`: count where **j occurs before k**
- `n_code_before_target`: count where **k occurs before j**
- `total_effects` (if present): an externally estimated effect signal for the pair (used directly for `T`)

---
