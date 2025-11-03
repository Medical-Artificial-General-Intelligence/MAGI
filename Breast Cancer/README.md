# MAGI vs LASSO for Breast-Cancer Identification (EHR)

This fold accompanies the manuscript **“Head-to-Head Comparison of a Knowledgebase Causal-Network Model (MAGI) and LASSO for Breast-Cancer Identification Using EHR Concepts.”** It contains the code and artifacts needed to reproduce the screening pipeline, model fitting, and main tables/figures.

> **What this fold gives you:** the full feature lists actually used by MAGI and LASSO (after anti-leakage rules), the harmonized EPV configuration, bootstrap evaluation code, and example configs for running the two models on an OMOP-style EHR extract.

---

## Overview

We compare two approaches on the **same** breast-cancer case–control cohort (N = 3,360; 672 cases, 2,688 controls):

1. **LASSO logistic regression**  
   - Patient-level, tabular design matrix  
   - Binary present/absent indicators for EHR concepts  
   - Feature selection via the regularization path

2. **MAGI (knowledgebase-driven causal-network model)**  
   - Operates on a **registry of concept–concept relationships** learned from medical records  
   - Encodes **temporal order** and estimates **direct (de-mediated) effects**  
   - **Does not require patient-level data at inference** (uses the registry)

Both approaches were forced through the **same screening pipeline** (SAFE-family + anti-leakage rules) and the **same EPV target** so that differences reflect the modeling strategy, not looser feature access.

