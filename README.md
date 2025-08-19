# README — Record-level Perplexity, Matchability & Pre-linkage Ambiguity for Splink

## TL;DR
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
- From Splink **weights** \(M_i\) (base-2 log-odds):  
  \(O_i = 2^{M_i}\)
- From pairwise **probabilities** \(q_i\):  
  \(O_i = \dfrac{q_i}{1 - q_i}\)

> **Note on prior:** Splink’s *linkable prior* is already inside \(M_i\) / \(q_i\).  
> When forming the record-level distribution, set the no-match odds to **1** (neutral), unless you intentionally re-prior.

### 2.2 Normalise over {candidates + no-match}
Let \(Z = 1 + \sum_i O_i\). Then
\[
p_{\text{null}} = \frac{1}{Z},\qquad p_i = \frac{O_i}{Z}.
\]

### 2.3 Record-level metrics
- **Matchability:** \(1 - p_{\text{null}} = \frac{\sum_i O_i}{1+\sum_i O_i}\).
- **Conditional candidate probabilities (given it links):**  
  \(\tilde p_i = \dfrac{p_i}{1 - p_{\text{null}}} = \dfrac{O_i}{\sum_j O_j}\).
- **Conditional entropy (bits) & perplexity:**  
  \(H_{\text{cond}} = -\sum_i \tilde p_i \log_2 \tilde p_i\),  
  \(\text{Perp}_{\text{cond}} = 2^{H_{\text{cond}}}\)  
  (≈ **effective number of plausible candidates**; continuous, so 1.03 means “almost unique”.)
- (Optional) **Unconditional** entropy/perplexity (include `p_null`) for dataset QA.

> **Numerical stability:** use the “max-trick” when exponentiating weights: let \(b=\max(0,M_1,\dots,M_K)\), use \(2^{M_i-b}\) and scale the null by \(2^{-b}\).

---

## 3) Pre-linkage ambiguity (multivariable cardinality)

Measure how “crowded” a record’s identifier **combination** is *before* the model weighs evidence:

- Use the **same comparison levels** you trust for linkage (e.g., “first name exact or Jaro≥0.9”, “surname exact”, “DOB exact or ±1”, “postcode sector exact”).  
- For each left record, count spine records that satisfy this **profile** ⇒ \(n_{\text{pre}}\).  
- Report:  
  \( \text{Perp}_{\text{pre}} = n_{\text{pre}}, \qquad H_{\text{pre}} = \log_2 n_{\text{pre}}.\)

**Resolution Gain:** \(RG = H_{\text{pre}} - H_{\text{cond}}\) (bits)  
**Resolution Factor:** \(RF = \text{Perp}_{\text{pre}} / \text{Perp}_{\text{cond}}\).

Large \(RG\)/\(RF\) = the matcher resolved a crowded neighbourhood; small or negative = lingering ambiguity (or duplicates in the spine).

---

## 4) Thresholding: before vs after threshold selection 

- **Best practice FOR MODEL BIAS ANALYSIS:** compute metrics on the **unthresholded** predictions; trim only for display (e.g., top-k or 99% mass).  
- **If production must threshold:** keep **pre-threshold aggregates** per record:  
  \(S_{\text{all}}=\sum_i 2^{M_i}\), raw candidate count. Compute `p_null` using \(S_{\text{all}}\), and, if rivals are dropped, add an **“other” bucket** with mass \(S_{\text{drop}} = S_{\text{all}} - S_{\text{kept}}\) when forming \(\tilde p\).  
- **Why:** thresholding **inflates `p_null`** and **deflates `perp_cond`** by hiding mass and rivals.

---

## 5) Calibration & priors (short)

- **Calibration matters:** if pairwise scores are over/under-confident, entropy/perplexity will be biased. 
- **Priors:** keep the default \(O_0=1\) (Splink’s prior already applied). Override only if you intentionally re-prior for a specific dataset/source.

---

## 6) Implementing

### 6.1 Compute record-level metrics from `predict()`

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

### 6.2 Pre-linkage ambiguity via your comparison levels (to be populated)

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

## 7) How to interpret

### 7.1 Single patient (record)
- **Likely unlinkable:** `p_null ≥ 0.8` (or `zero_candidate=True`).  
- **Uncertain:** `matchability ∈ [0.2, 0.8]` → review; use `perp_cond` to see if one candidate vs many.  
- **Likely link:** `matchability ≥ 0.8`.  
  - `perp_cond ≈ 1`, large `margin` → auto-accept.  
  - `perp_cond ≥ 3` → rival cluster or spine duplicates → review/dedupe.  
- **Context with pre-ambiguity:** high `Perp_pre` but low `perp_cond` → well resolved; high/high → genuine homonyms.

### 7.2 Sub-population (e.g., ethnicity, age, region)
Report per group:
- **Unlinkables-like rate:** share with `p_null > τ` (e.g., 0.8/0.9).  
- **Ambiguity burden:** median/p90 `perp_cond` among `matchability ≥ 0.5`.  
- **Intrinsic crowding:** median `H_pre`.  
- **Resolution gain:** median `RG_bits`.  

### 7.3 Population/dataset
Track over time:
- Unlinkables share, zero-candidate rate  
- Median matchability  
- p90 `perp_cond` (clerical load)  
- Median `H_pre` (cohort difficulty), median `RG_bits` (system resolving power)

---

## 8) Visuals (quick starters)

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
---

## 9) Glossary
- **`match_weight`**: Splink’s base-2 log-odds for a pair (includes prior).  
- **Odds vs no-match**: \(O_i = 2^{M_i}\) or \(q_i/(1-q_i)\).  
- **`p_null`**: probability of no match for the record.  
- **`matchability`**: \(1-p_{\text{null}}\).  
- **`p_i` / `p_cond`**: per-candidate probability mass (unconditional / conditional on linking).  
- **`H_cond` / `perp_cond`**: entropy & perplexity over `p_cond` (ambiguity among candidates).  
- **`H_pre` / `Perp_pre`**: pre-linkage combined cardinality using your comparison levels.  
- **`RG_bits` / `RF_factor`**: resolution gain/factor from pre to post.

---
