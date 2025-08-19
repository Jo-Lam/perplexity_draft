# Record-level Perplexity, Matchability & Pre-linkage Ambiguity for Splink

## Overview
- **Goal:** quantify *how certain* a linkage decision is **per record**, and summarise uncertainty for **sub-populations** and the **whole dataset**.  

- **Outputs per record:**  
  - `p_null` = probability of **no match**; `matchability = 1 − p_null`  
  - `perp_cond` = **effective number of plausible candidates** *given it links*  
  - optional: `H_cond` (bits), `top1`, `margin`, candidate-level `p_i`, `p_cond`
- **Pre-linkage ambiguity:** count how many records fall into the **same evidence neighbourhood** (built from your comparison levels); report `Perp_pre = n_pre` and `H_pre = log2(n_pre)`.  
- **Nice combo:** **Resolution Gain** = `H_pre − H_cond` shows how much our linker resolved the crowding.

---

## 1) Why this is needed (pairwise ≠ record-level)

Splink’s `predict()` returns **pairwise** scores (`match_weight`, `match_probability`) for each **left record × candidate** edge. For record-level decisions you need a **single distribution** over mutually exclusive outcomes for that record:

> {match to candidate 1, 2, …, **no match**}.

From this we derive:
- **`p_null`** (unlinkable probability)  
- **`matchability = 1 − p_null`**  
- **`perp_cond`** (ambiguity among the candidates *if* it does link)

These drive **clerical review** and are easy to aggregate by **sub-group** and **population**.

---

## 2) The math (weights *or* probabilities)

For one left record with candidates \(i=1..K\):

### 2.1 Convert to odds vs no-match
- From Splink **weights** \(Mi\) (base-2 log-odds):  
  <img width="227" height="72" alt="image" src="https://github.com/user-attachments/assets/147af06d-294c-4621-84a4-6c443348a66b" />

- From pairwise **probabilities** \(qi\):  
  <img width="230" height="82" alt="image" src="https://github.com/user-attachments/assets/8d2c56f8-061c-4311-a0ef-e2fef3545e69" />

> **Note on prior:** Splink’s *linkable prior* is already inside \(Mi\) / \(qi\).  
> When forming the record-level distribution, set the no-match odds to **1** (neutral), unless you intentionally re-prior.

### 2.2 Normalise over {candidates + no-match}
<img width="609" height="119" alt="image" src="https://github.com/user-attachments/assets/20d8cd23-affc-4a55-b011-e6b9581469bc" />

### 2.3 Record-level metrics
- **Matchability: <img width="190" height="51" alt="image" src="https://github.com/user-attachments/assets/99e79966-7380-4b20-9676-7cbaca6bfdaf" />
- **Conditional candidate probabilities (given it links):**  
  <img width="274" height="64" alt="image" src="https://github.com/user-attachments/assets/e7049868-672e-4db1-9bf1-5f9b51ce6927" />

- **Conditional entropy (bits) & perplexity:**  
  <img width="274" height="73" alt="image" src="https://github.com/user-attachments/assets/26170d43-8afd-42f5-8fef-56f0f2eb98ff" />
  (≈ **effective number of plausible candidates**; continuous, so 1.03 means “almost unique”.)
- (Optional) **Unconditional** entropy/perplexity (include `p_null`) for dataset QA.

---

## 3) Pre-linkage ambiguity (multivariable cardinality)

Measure how “crowded” a record’s identifier **combination** is *before* the model weighs evidence:

- Use the **same comparison levels** you trust for linkage (e.g., “first name exact or Jaro≥0.9”, “surname exact”, “DOB exact or ±1”, “postcode sector exact”).  
- For each left record, count spine records that satisfy this **profile** ⇒ Npre
- Report:  Perp_pre = Npre, Hpre = log2 Npre
- Resolution Gain: RG = H_pre - Hcond (bits)
- Resolution Factor: RF = Perp_pre / Perp_cond
       potentially useful - Large RG/RF shows that our linker resolved a crowded neighbourhood. Small RG/RG shows that there are lingering ambiguity or duplication in the spine.

---

## 4) Calibration & priors (short)

- **Calibration matters:** if pairwise scores are over/under-confident, entropy/perplexity will be biased. 
- **Priors:** keep the default \(O_0=1\) (Splink’s prior already applied). Override only if you intentionally re-prior for a specific dataset/source.

---

## 5) Implementing

### 5.1 Compute record-level metrics from `predict()`

```python
import numpy as np
import pandas as pd

def _entropy_bits(p, eps=1e-12):
    p = np.asarray(p, float)
    s = p.sum()
    if s <= 0: return 0.0
    p = np.clip(p / s, eps, 1 - eps)
    p /= p.sum()
    H = -(p * (np.log(p) / np.log(2.0))).sum()
    return float(max(H, 0.0))

def compute_perplexity_metrics_from_splink(
    df_edges: pd.DataFrame,
    *,
    source_col: str = "unique_id_l",
    candidate_col: str = "unique_id_r",
    weight_col: str = "match_weight",
    prob_col: str | None = None,
    # prior for no-match; keep neutral unless you deliberately re-prior
    prior_default_odds: float = 1.0,
    # optional trimming (robustness to dust tails):
    topk: int | None = None,
    min_cum_mass: float | None = None,  # e.g., 0.99
    return_edge_probs: bool = True,
    all_source_ids: list | pd.Series | None = None,  # to include zero-candidate records
    eps: float = 1e-12,
):
    df = df_edges.copy()
    rec_rows, edge_rows = [], []

    # count raw candidates per record
    n_cand_map = df.groupby(source_col, sort=False)[candidate_col].size() if len(df) else pd.Series(dtype=int)

    for sid, g in df.groupby(source_col, sort=False):
        g = g.copy()
        # odds vs null
        if prob_col is not None:
            q = g[prob_col].astype(float).clip(eps, 1-eps)
            Oi = (q / (1-q)).to_numpy(float)
            O0_scaled = prior_default_odds
            order_strength = Oi  # for trimming
        else:
            M = g[weight_col].astype(float).to_numpy()
            mmax = np.max(M) if M.size else 0.0
            Oi = np.power(2.0, M - mmax)
            O0_scaled = prior_default_odds * np.power(2.0, -mmax)
            order_strength = M

        # optional trimming
        keep_idx = np.arange(len(Oi))
        if len(Oi) and (topk is not None or min_cum_mass is not None):
            order = np.argsort(-order_strength)  # desc
            Oi_sorted = Oi[order]
            k = len(Oi_sorted)
            if min_cum_mass is not None:
                cs = np.cumsum(Oi_sorted)
                k = min(k, np.searchsorted(cs / cs[-1], min_cum_mass, side="left") + 1)
            if topk is not None:
                k = min(k, int(topk))
            keep_ord = order[:k]
            keep_idx = np.sort(keep_ord)
            Oi = Oi[keep_idx]

        Z = O0_scaled + Oi.sum()
        p_null = O0_scaled / Z if Z > 0 else 1.0
        matchability = 1.0 - p_null
        p_i = Oi / Z if Z > 0 else np.zeros_like(Oi)

        # conditional over candidates
        if matchability > eps and p_i.size:
            p_cond = p_i / matchability
            H_cond = _entropy_bits(p_cond, eps)
            perp_cond = float(2.0 ** H_cond)
            ord_pc = np.sort(p_cond)[::-1]
            top1 = float(ord_pc[0])
            margin = float(ord_pc[0] - ord_pc[1]) if ord_pc.size >= 2 else float("nan")
        else:
            p_cond = np.zeros_like(p_i)
            H_cond, perp_cond, top1, margin = 0.0, 1.0, 0.0, float("nan")

        H_all = _entropy_bits(np.r_[p_i, p_null], eps) if p_i.size else _entropy_bits([p_null], eps)
        perp_all = float(2.0 ** H_all)
        rec_rows.append({
            source_col: sid,
            "p_null": float(p_null),
            "matchability": float(matchability),
            "H_cond": H_cond,
            "perp_cond": perp_cond,
            "H_all": H_all,
            "perp_all": perp_all,
            "top1": top1,
            "margin": margin,
            "n_candidates": int(n_cand_map.get(sid, 0)),
            "n_kept": int(len(Oi)),
            "trimmed": bool(len(Oi) != int(n_cand_map.get(sid, 0))),
        })

        if return_edge_probs:
            g = g.iloc[keep_idx].copy()
            g["p_i"] = p_i
            g["p_cond"] = p_cond
            edge_rows.append(g)

    record_metrics = pd.DataFrame(rec_rows)

    # include zero-candidate left records
    if all_source_ids is not None:
        all_ids = pd.Series(all_source_ids, name=source_col).drop_duplicates()
        record_metrics = all_ids.to_frame().merge(record_metrics, on=source_col, how="left")
        fills = {
            "p_null": 1.0, "matchability": 0.0,
            "H_cond": 0.0, "perp_cond": 1.0,
            "H_all": 0.0, "perp_all": 1.0,
            "top1": 0.0, "margin": np.nan,
            "n_candidates": 0, "n_kept": 0, "trimmed": False
        }
        for c, v in fills.items():
            record_metrics[c] = record_metrics[c].fillna(v)
        record_metrics["zero_candidate"] = record_metrics["n_candidates"].eq(0)
    else:
        record_metrics["zero_candidate"] = record_metrics["n_candidates"].eq(0)

    edges_enriched = pd.concat(edge_rows, ignore_index=True) if (return_edge_probs and edge_rows) else None
    return record_metrics, edges_enriched
```

### 5.2 Pre-linkage ambiguity via your comparison levels (to be populated)

```python
# After predict() with gamma/level columns available
def pre_profile(row):
    # replace LEVEL_* with the integers that correspond to your comparison levels
    ok_first = row['gamma_first_name'] >= LEVEL_NEAR_OR_EXACT
    ok_surn  = row['gamma_surname']    >= LEVEL_EXACT
    ok_dob   = row['gamma_dob']        >= LEVEL_EXACT_OR_ADJ
    ok_pc    = row['gamma_pc_sector']  >= LEVEL_EXACT
    return bool(ok_first and ok_surn and ok_dob and ok_pc)

df_pred["pre_ok"] = df_pred.apply(pre_profile, axis=1)
pre_counts = df_pred.groupby("unique_id_l")["pre_ok"].sum().rename("n_pre")
pre_stats = pre_counts.to_frame()
pre_stats["H_pre"] = np.where(pre_stats["n_pre"]>0, np.log2(pre_stats["n_pre"]), 0.0)
pre_stats["perp_pre"] = pre_stats["n_pre"].astype(float)

rm = record_metrics.merge(pre_stats, left_on="unique_id_l", right_index=True, how="left").fillna({"n_pre":0,"H_pre":0.0,"perp_pre":1.0})
rm["RG_bits"] = rm["H_pre"] - np.log2(np.clip(rm["perp_cond"], 1.0, None))
rm["RF_factor"] = rm["perp_pre"] / np.clip(rm["perp_cond"], 1e-12, None)
```

---

## 6) How to interpret

### 6.1 Single patient (record)
- **Likely unlinkable:** `p_null ≥ 0.8` (or `zero_candidate=True`).  
- **Uncertain:** `matchability ∈ [0.2, 0.8]` → review; use `perp_cond` to see if one candidate vs many.  
- **Likely link:** `matchability ≥ 0.8`.  
  - `perp_cond ≈ 1`, large `margin` → auto-accept.  
  - `perp_cond ≥ 3` → rival cluster or spine duplicates → review/dedupe.  
- **Context with pre-ambiguity:** high `Perp_pre` but low `perp_cond` → well resolved; high/high → genuine homonyms.

### 6.2 Sub-population (e.g., ethnicity, age, region)
Report per group:
- **Unlinkables-like rate:** share with `p_null > τ` (e.g., 0.8/0.9).  
- **Ambiguity burden:** median/p90 `perp_cond` among `matchability ≥ 0.5`.  
- **Intrinsic crowding:** median `H_pre`.  
- **Resolution gain:** median `RG_bits`.  

### 6.3 Population/dataset
Track over time:
- Unlinkables share, zero-candidate rate  
- Median matchability  
- p90 `perp_cond` (clerical load)  
- Median `H_pre` (cohort difficulty), median `RG_bits` (system resolving power)

---

## 7) Visuals (quick starters)

```python
import matplotlib.pyplot as plt

def plot_pnull_hist(record_metrics, bins=30, thresholds=(0.2, 0.8)):
    x = record_metrics["p_null"].to_numpy()
    fig, ax = plt.subplots()
    ax.hist(x, bins=bins)
    for t in thresholds:
        ax.axvline(t)
    ax.set_xlabel("p_null  (P(no match))")
    ax.set_ylabel("count")
    ax.set_title("Distribution of p_null")
    plt.show()
```

<img width="571" height="455" alt="cf403920-79ac-4b35-98b6-837936f12471" src="https://github.com/user-attachments/assets/a48a9bc0-4dbb-4630-9302-8c9fc6b0410f" />

```python
def plot_perp_cond_hist(record_metrics, bins=30, matchability_min=0.5):
    m = record_metrics["matchability"].to_numpy()
    perp = record_metrics["perp_cond"].to_numpy()
    keep = m >= matchability_min
    fig, ax = plt.subplots()
    ax.hist(perp[keep], bins=bins)
    ax.set_xlabel("perp_cond (effective # candidates | links)")
    ax.set_ylabel("count")
    ax.set_title(f"Perplexity (conditional) for matchable records (matchability ≥ {matchability_min})")
    plt.show()
```
<img width="600" height="455" alt="8c9e1160-cf3d-4f65-8b91-e48679dc30f0" src="https://github.com/user-attachments/assets/6b9dc18d-eaae-40fa-ad7d-2b8fc8207939" />



```python
def plot_matchability_vs_perp(record_metrics, m_band=(0.2, 0.8), perp_thresh=3):
    m = record_metrics["matchability"].to_numpy()
    perp = record_metrics["perp_cond"].to_numpy()
    fig, ax = plt.subplots()
    ax.scatter(m, perp, s=8)
    ax.axvline(m_band[0])
    ax.axvline(m_band[1])
    ax.axhline(perp_thresh)
    ax.set_xlabel("matchability = 1 - p_null")
    ax.set_ylabel("perp_cond")
    ax.set_title("Matchability vs Conditional Perplexity (triage bands)")
    plt.show()
```

<img width="554" height="455" alt="2afa4b36-9a59-4b09-9f5d-e231f3d8564b" src="https://github.com/user-attachments/assets/0d500f58-449d-4c46-82a0-263c31addf3a" />

### What the axes & lines mean
x (matchability = 1 − p_null): how likely the record links to someone.
y (perp_cond): “effective # of plausible candidates given it links.”
1 ≈ one clear winner; 2 ≈ two near-equals; 3+ = genuine ambiguity.

Vertical lines (0.2, 0.8): rough bands → left = likely unlinkable, middle = uncertain, right = likely link.
Horizontal line (y=3): flag if there are ~3+ viable candidates.

(Note this is done without setting a threshold for linkage model analysis.)

### How to interpret the clusters  - to drive clerical review (use thresholded version)
- Pile at x≈0, y≈1
Zero-candidate or only vanishingly weak candidates. These are the unlinkables (or blocking missed the true match). Not worth clerical time—fix blocking/backoffs & data quality.

- Left side (x≈0) but y>1 (some up to ~9)
Many weak lookalikes, none convincing → ambiguous and unlikely to link. Treat as unlinkable for this run; investigate blocking/recall for common-name blocks.

- Middle band (0.2 ≤ x ≤ 0.8)
Uncertain existence of a link.
Low y (~1–1.5): one lukewarm candidate vs no-match → good for review (often data-quality issues or harsh prior), but not priority.
High y (≥3): several contenders and overall uncertainty → strong review candidates.

- Right edge (x≈1), y≈1–2
Confident links. Most have one dominant candidate (y≈1); some have two close ones (y≈2). Generally auto-accept. But depends on use case, may warrant reviews 

- Right edge (x≈1), y≥3 (if any)
Likely duplicates in the spine or multiple near-identical candidates. Send to review for duplication or add tie-breaker features.


