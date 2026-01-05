
# 2025-02-12, YL
import sqlite3
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union

# ======================================================================
# MAGI core: analyze_causal_sequence_py (INT-BASED)
# ======================================================================

# RIGHT = k (predictor) and LEFT = Y/j
def analyze_causal_sequence_py(
    df_in: Union[str, pd.DataFrame],
    *,
    name_map: Dict[str, str] = None,     # kept for compatibility but IGNORED in _int version
    events: List[int] = None,            # event IDs to KEEP; if None: auto-detect from *_int cols
    force_outcome: int = None,           # if provided and present, force this to be the FINAL node (Y)
    lambda_min_count: int = 15           # L-threshold for λ: if n_code < L ⇒ λ_{k,j}=0
) -> Dict[str, Any]:
    """
    MAGI (Python, INT-BASED) – uses total_effects from DB for T, no 2×2 fallback.
    All computations are done on target_concept_code_int / concept_code_int.

    Rules:
      • T_{kY}:
            If `total_effects` column exists:
                T_{kY} = mean(total_effects) from row(s) with (target=Y, code=k).
                If those rows are missing or total_effects is NaN/≤0/±inf → T_{kY} = 1.
            If `total_effects` column does NOT exist at all → T_{kY} = 1 for all k.
      • Temporal score:
            For each Zi:
                score(Zi) = Σ_{Zj≠Zi} [ n_target_before_code(Zi,Zj) - n_code_before_target(Zi,Zj) ]
            This is computed from the same counts as your original code,
            just via a MultiIndex instead of repeated scans.
      • λ_{k,j}:
            λ_{k,j} = n_code_target(j,k) / n_code(j,k),
            read from rows with (target=j, code=k), with L-threshold on n_code.
    """
    # ── 0) Ingest & basic checks ───────────────────────────────────────────────
    df = pd.read_csv(df_in) if isinstance(df_in, str) else df_in.copy()

    need_cols = [
        "target_concept_code_int", "concept_code_int",
        "n_code_target", "n_code_no_target",
        "n_target", "n_no_target",
        "n_target_no_code",
        "n_code",
        "n_code_before_target", "n_target_before_code",
    ]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required count columns for MAGI: {', '.join(missing)}")

    has_total_effects = "total_effects" in df.columns

    # Ensure *_int are numeric / nullable ints
    df["target_concept_code_int"] = pd.to_numeric(df["target_concept_code_int"], errors="coerce").astype("Int64")
    df["concept_code_int"]        = pd.to_numeric(df["concept_code_int"],        errors="coerce").astype("Int64")

    # name_map intentionally ignored in _int version

    # Limit to selected events
    if events is None:
        arr_t = df["target_concept_code_int"].dropna().unique()
        arr_c = df["concept_code_int"].dropna().unique()
        # IntegerArray -> list of Python ints
        ev_t = [int(x) for x in arr_t]
        ev_c = [int(x) for x in arr_c]
        events = sorted(set(ev_t) | set(ev_c))
    else:
        events = [int(e) for e in events]

    if len(events) < 2:
        raise ValueError("Need at least two events.")

    df = df[
        df["target_concept_code_int"].isin(events)
        & df["concept_code_int"].isin(events)
    ].copy()

    # Coerce numerics
    num_cols = [
        "n_code_target", "n_code_no_target",
        "n_target", "n_no_target", "n_target_no_code",
        "n_code",
        "n_code_before_target", "n_target_before_code",
    ]
    if has_total_effects:
        num_cols.append("total_effects")

    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ── 1) Build an indexed edge table ────────────────────────────────────────
    edge = (
        df.groupby(["target_concept_code_int", "concept_code_int"], as_index=True)[
            ["n_target_before_code", "n_code_before_target",
             "n_code_target", "n_code"]
        ].sum()
    )

    edge_targets = edge.index.get_level_values(0)

    # ── 2) Temporal scores ────────────────────────────────────────────────────
    scores: Dict[int, float] = {}
    for zi in events:
        if zi not in edge_targets:
            scores[zi] = 0.0
            continue

        try:
            sub = edge.xs(zi, level="target_concept_code_int")  # index = concept_code_int
        except KeyError:
            scores[zi] = 0.0
            continue

        s = float(
            (sub["n_target_before_code"].fillna(0.0) -
             sub["n_code_before_target"].fillna(0.0)).sum()
        )
        scores[zi] = s

    sorted_scores = pd.Series(scores).sort_values(ascending=False)

    # Choose outcome Y
    if (force_outcome is not None) and (force_outcome in sorted_scores.index):
        outcome = int(force_outcome)
        temporal_order = [ev for ev in sorted_scores.index if ev != outcome] + [outcome]
    else:
        outcome = int(sorted_scores.index[0])
        temporal_order = [ev for ev in sorted_scores.index if ev != outcome] + [outcome]

    events_order = temporal_order
    nodes = events_order[:-1]

    pos_by_event = {ev: i for i, ev in enumerate(events_order)}

    # Pre-filter rows where LEFT == outcome for T_{kY}
    if has_total_effects:
        dfY = df[df["target_concept_code_int"] == outcome].copy()
        dfY["total_effects"] = pd.to_numeric(dfY["total_effects"], errors="coerce")
    else:
        dfY = None

    # ── 3) T and λ ─────────────────────────────────────────────────────────────
    T_val = pd.Series(0.0, index=nodes, dtype=float)
    D_val = pd.Series(np.nan, index=nodes, dtype=float)
    lambda_l: Dict[int, pd.Series] = {}

    for k in nodes:
        # T_{kY}
        if has_total_effects:
            row_Yk = dfY[dfY["concept_code_int"] == k]
            if row_Yk.empty:
                T_val.loc[k] = 1.0
            else:
                T_col = pd.to_numeric(row_Yk["total_effects"], errors="coerce")
                T_col = T_col.replace([np.inf, -np.inf], np.nan)
                T_clean = T_col.dropna()
                if T_clean.empty:
                    T_val.loc[k] = 1.0
                else:
                    T_db = float(T_clean.mean())
                    if (not np.isfinite(T_db)) or (T_db <= 0):
                        T_db = 1.0
                    T_val.loc[k] = T_db
        else:
            T_val.loc[k] = 1.0

        # λ_{k,j}
        pos_k = pos_by_event[k]
        js = events_order[pos_k + 1 : -1] if pos_k < len(events_order) - 1 else []

        lam_pairs = {}
        for j in js:
            key = (j, k)
            if key not in edge.index:
                lam_pairs[j] = 0.0
                continue

            row_jk = edge.loc[key]
            num = float(row_jk["n_code_target"])
            den = float(row_jk["n_code"])

            if (den <= 0) or (den < lambda_min_count):
                lam_pairs[j] = 0.0
                continue

            lam = num / den
            if not np.isfinite(lam):
                lam = 0.0
            lam_pairs[j] = float(min(max(lam, 0.0), 1.0))

        lambda_l[k] = pd.Series(lam_pairs, dtype=float)

    # ── 4) Backward recursion for D ────────────────────────────────────────────
    if len(nodes) >= 1:
        last_anc = nodes[-1]
        D_val.loc[last_anc] = T_val.loc[last_anc]

    if len(nodes) > 1:
        for k in list(reversed(nodes[:-1])):
            lam_vec = lambda_l.get(k, pd.Series(dtype=float))
            downstream = list(lam_vec.index)
            lam_vals = lam_vec.reindex(downstream).fillna(0.0).to_numpy()
            D_down  = pd.to_numeric(D_val.reindex(downstream),
                                    errors="coerce").fillna(0.0).to_numpy()

            num = T_val.loc[k] - float(np.nansum(lam_vals * D_down))
            den = 1.0 - float(np.nansum(lam_vals))

            if (not np.isfinite(den)) or den == 0.0:
                D_val.loc[k] = T_val.loc[k]
            else:
                tmp = num / den
                D_val.loc[k] = tmp if np.isfinite(tmp) else T_val.loc[k]

    # ── 5) Logistic link (β) and predict_proba ─────────────────────────────────
    resp_rows = df[df["target_concept_code_int"] == outcome]
    n_t = float(pd.to_numeric(resp_rows["n_target"],      errors="coerce").max()) if not resp_rows.empty else np.nan
    n_n = float(pd.to_numeric(resp_rows["n_no_target"],   errors="coerce").max()) if not resp_rows.empty else np.nan
    denom = n_t + n_n
    p_y = 0.5 if (not np.isfinite(denom) or denom <= 0) else (n_t / denom)
    p_y = min(max(p_y, 1e-12), 1 - 1e-12)
    beta_0 = float(np.log(p_y / (1 - p_y)))

    D_clean = pd.to_numeric(D_val, errors="coerce").astype(float)
    beta_vals = np.log(D_clean.where(D_clean > 0.0)) \
                    .replace([np.inf, -np.inf], np.nan).fillna(0.0)

    coef_df = pd.DataFrame({
        "predictor": list(beta_vals.index) + ["(intercept)"],
        "beta":      list(beta_vals.values) + [beta_0],
    })

    predictors = list(beta_vals.index)
    beta_vec = beta_vals.values

    def predict_proba(Z):
        """
        Compute P(Y=1|Z) using: logit P = β0 + Σ_k β_k Z_k.
        Here Z keys should be concept_code_int IDs.
        """
        def sigmoid(x):
            x = np.clip(x, -700, 700)
            return 1.0 / (1.0 + np.exp(-x))

        if isinstance(Z, pd.DataFrame):
            M = Z.reindex(columns=predictors, fill_value=0.0) \
                 .astype(float).to_numpy()
            return sigmoid(beta_0 + M @ beta_vec)

        if isinstance(Z, (dict, pd.Series)):
            v = np.array([float(Z.get(p, 0.0)) for p in predictors], dtype=float)
            return float(sigmoid(beta_0 + float(v @ beta_vec)))

        arr = np.asarray(Z, dtype=float)
        if arr.ndim == 1:
            if arr.size != len(predictors):
                raise ValueError(f"Expected {len(predictors)} features in order: {predictors}")
            return float(sigmoid(beta_0 + float(arr @ beta_vec)))
        if arr.ndim == 2:
            if arr.shape[1] != len(predictors):
                raise ValueError(f"Expected shape (*,{len(predictors)}), got {arr.shape}")
            return sigmoid(beta_0 + arr @ beta_vec)

        raise ValueError("Unsupported input for predict_proba")

    return {
        "sorted_scores": sorted_scores,
        "temporal_order": events_order,
        "order_used": events_order,
        "T_val": T_val,
        "D_val": D_val,
        "lambda_l": lambda_l,
        "coef_df": coef_df,
        "beta_0": beta_0,
        "beta": pd.Series(beta_vec, index=predictors, dtype=float),
        "logit_predictors": predictors,
        "predict_proba": predict_proba,
    }

def fetch_magi_subgraph_for_events(
    db_path: str,
    events: List[str],
    edge_table: str = "magi_counts_published",
    concept_table: str = "concept_names",
) -> pd.DataFrame:
    """
    Fetch a MAGI subgraph from `magi_counts_published` for a given set of events.

    Parameters
    ----------
    db_path : str
        Path to the SQLite DB (e.g., '/projects/.../magiv2.db').
    events : list of str
        Concept codes to keep on BOTH sides:
        - LEFT  (target_concept_code)
        - RIGHT (concept_code)
        The resulting df will only include rows where both codes are in this set.
    edge_table : str, default 'magi_counts_published'
        Name of the edge/counts table in the DB.
    concept_table : str, default 'concept_names'
        Name of the concept-name lookup table in the DB.

    Returns
    -------
    pd.DataFrame
        DataFrame with all required columns for analyze_causal_sequence_py:
        - target_concept_code
        - concept_code
        - n_code_target, n_code_no_target, n_target, n_no_target,
          n_target_no_code, n_code, n_code_before_target, n_target_before_code,
          and any additional columns in the edge table (e.g., total_effects).
    """
    # basic safety
    if not events:
        raise ValueError("fetch_magi_subgraph_for_events: `events` list is empty.")

    # ensure unique, sorted string codes
    events = sorted({str(e) for e in events})

    # read-only URI for SQLite
    uri = f"file:{db_path}?mode=ro"

    placeholders = ",".join(["?"] * len(events))

    q = f"""
      SELECT
          m.*,
          tcn.concept_code AS target_concept_code,
          ccn.concept_code AS concept_code
      FROM {edge_table} m
      JOIN {concept_table} tcn
        ON m.target_concept_code_int = tcn.concept_code_int
      JOIN {concept_table} ccn
        ON m.concept_code_int        = ccn.concept_code_int
      WHERE tcn.concept_code_int IN ({placeholders})
        AND ccn.concept_code_int IN ({placeholders})
    """

    params = events + events

    with sqlite3.connect(uri, uri=True) as conn:
        df = pd.read_sql_query(q, conn, params=params)

    return df


