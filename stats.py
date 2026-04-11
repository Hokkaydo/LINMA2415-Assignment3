"""
statistics.py  —  Cross-instance statistics for Assignment 3 analysis.

Usage
─────
    python statistics.py path/to/results.json
    python statistics.py --run path/to/instances/  # solve all and collect

The script expects a results dict (produced by run_all or loaded from JSON)
with the following structure:

    results = {
        "2015-01-01_hw": {
            "cost_uc":        float,
            "cost_ed":        float,
            "cost_lp":        float,          # LP relaxation (ACHP)
            "lam_mp":         list[float],    # length T
            "lam_achp":       list[float],
            "mu_mp":          list[float],
            "mu_achp":        list[float],
            "loc_mp":         list[float],    # one per generator
            "loc_achp":       list[float],
            "profits_uc_mp":  list[float],
            "profits_uc_achp":list[float],
            "u_uc":           list[list[int]], # (G, T)
            "demand":         list[float],
            "reserves":       list[float],
            "G":              int,
            "W":              int,
            "T":              int,
        },
        "2015-01-01_lw": { ... },
        "2015-02-01_hw": { ... },
        ...
    }

Call collect_results() to build this dict by running your homework.py
on each instance, then call compute_statistics(results) to get the
full statistical report.
"""

import os, json, re
import numpy as np
import pandas as pd
from pathlib import Path
import threading

def worker(path, homework_module, T_max, results):
    # run data collection for each instance with threading to speed up
    hw = homework_module
    key = path.stem          # e.g.  "2015-01-01_hw"
    print(f"{key}    ... processing {path.name}")
    try:
        data = hw.parse_json(str(path), T_max=T_max)
        T, G, W = data["T"], data["G"], data["W"]

        # UC (MILP)
        cost_uc, u_uc, pg_uc, pw_uc, _, _ = hw.solve_uc(data, is_binary=True)

        # ED (MP prices)
        cost_ed, pg_ed, pw_ed, lam_mp, mu_mp = hw.solve_economic_dispatch(data, u_uc)

        # LP relaxation (ACHP prices)
        cost_lp, _, _, _, lam_achp, mu_achp = hw.solve_uc(data, is_binary=False)

        # LOC under both pricing methods (always evaluated on binary u_uc)
        loc_mp,   profits_uc_mp,   _, _ = hw.compute_loc(data, u_uc, pg_uc, lam_mp)
        loc_achp, profits_uc_achp, _, _ = hw.compute_loc(data, u_uc, pg_uc, lam_achp)

        results[key] = dict(
            cost_uc        = float(cost_uc),
            cost_ed        = float(cost_ed),
            cost_lp        = float(cost_lp),
            lam_mp         = lam_mp.tolist(),
            lam_achp       = lam_achp.tolist(),
            mu_mp          = mu_mp.tolist(),
            mu_achp        = mu_achp.tolist(),
            loc_mp         = loc_mp,
            loc_achp       = loc_achp,
            profits_uc_mp  = profits_uc_mp,
            profits_uc_achp= profits_uc_achp,
            u_uc           = [u.tolist() for u in u_uc],
            demand         = data["demand"].tolist(),
            reserves       = data["reserves"].tolist(),
            G=G, W=W, T=T,
        )
        print(f"{key}    ✓  UC=${cost_uc:,.0f}  ED=${cost_ed:,.0f}  "
                f"LOC_MP=${sum(loc_mp):,.0f}  LOC_ACHP=${sum(loc_achp):,.0f}")

    except Exception as e:
        print(f"{key}    ✗  FAILED: {e}")
        raise e

# ══════════════════════════════════════════════════════════════════════════════
#  1.  COLLECT RESULTS  (runs homework.py on each instance)
# ══════════════════════════════════════════════════════════════════════════════

def collect_results(instances_dir, homework_module, T_max=12, save_path=None):
    """
    Solve every hw/lw instance in instances_dir and collect results.

    Parameters
    ----------
    instances_dir  : str   directory containing the .json instance files
    homework_module: module   your imported homework module (already has
                              parse_json, solve_uc, solve_economic_dispatch,
                              solve_lp_relaxation, compute_loc, etc.)
    T_max          : int   cap on number of periods (12 for this assignment)
    save_path      : str   if given, dump results to this JSON file

    Returns
    -------
    results : dict   structured as described in the module docstring
    """
    hw = homework_module
    results = {}

    # Find all JSON files, group by date and suffix
    jsons = sorted(Path(instances_dir).glob("*.json"))
    print(f"Found {len(jsons)} instance files.")

    threads = []
    for path in jsons:
        # launch a thread per instance
        t = threading.Thread(target=worker, args=(path, hw, T_max, results))
        t.start()
        threads.append(t)

    # Wait for all threads to complete
    for t in threads:
        t.join()

    if save_path:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {save_path}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  2.  STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_statistics(results, print_report=True):
    """
    Compute all statistics useful for the theoretical analysis.

    Returns a dict with keys:
      per_instance   : DataFrame  — one row per instance
      hw_vs_lw       : DataFrame  — hw vs lw comparison per month
      monthly        : DataFrame  — aggregated per month
      pricing        : dict       — pricing-related statistics
      loc            : dict       — LOC statistics
      generators     : dict       — generator-level statistics
    """
    keys = sorted(results.keys())

    # ── Parse key names ───────────────────────────────────────────────────────
    def parse_key(k):
        m = re.match(r"(\d{4}-\d{2}-\d{2})_(hw|lw)", k)
        if m:
            return m.group(1), m.group(2)
        return k, "unknown"

    rows = []
    for key in keys:
        r         = results[key]
        date, tag = parse_key(key)
        month     = date[:7]
        lam_mp    = np.array(r["lam_mp"])
        lam_achp  = np.array(r["lam_achp"])
        mu_mp     = np.array(r["mu_mp"])
        mu_achp   = np.array(r["mu_achp"])
        demand    = np.array(r["demand"])
        loc_mp    = np.array(r["loc_mp"])
        loc_achp  = np.array(r["loc_achp"])
        p_mp      = np.array(r["profits_uc_mp"])
        p_achp    = np.array(r["profits_uc_achp"])
        u_uc      = np.array(r["u_uc"])   # (G, T)
        G, T      = r["G"], r["T"]

        # Demand-weighted average price
        wm = lambda lam: float(np.dot(lam, demand) / demand.sum()) if demand.sum() > 0 else 0.

        # Integrality gap = (UC cost - LP relax cost) / UC cost
        ig = (r["cost_uc"] - r["cost_lp"]) / r["cost_uc"] if r["cost_uc"] > 0 else 0.

        # UC vs ED cost difference
        delta_cost = r["cost_uc"] - r["cost_ed"]

        # Price volatility (std / mean)
        def cv(x): return float(x.std() / x.mean()) if x.mean() > 0 else 0.

        # Number of generators needing uplift
        n_need_mp   = int((p_mp   < 0).sum())
        n_need_achp = int((p_achp < 0).sum())

        # Average commitment rate (fraction of time each generator is on)
        avg_commitment = float(u_uc.mean())
        n_always_on    = int((u_uc.sum(axis=1) == T).sum())
        n_never_on     = int((u_uc.sum(axis=1) == 0).sum())

        # Peak demand and reserve margin
        peak_demand   = float(demand.max())
        avg_demand    = float(demand.mean())
        demand_range  = float(demand.max() - demand.min())

        rows.append(dict(
            key                = key,
            date               = date,
            month              = month,
            tag                = tag,
            G                  = G,
            W                  = r["W"],
            T                  = T,

            # Costs
            cost_uc            = r["cost_uc"],
            cost_ed            = r["cost_ed"],
            cost_lp            = r["cost_lp"],
            delta_cost_uc_ed   = delta_cost,
            delta_cost_pct     = 100. * delta_cost / r["cost_ed"] if r["cost_ed"] > 0 else 0.,
            integrality_gap    = 100. * ig,

            # Demand
            peak_demand        = peak_demand,
            avg_demand         = avg_demand,
            demand_range       = demand_range,
            demand_cv          = cv(demand),

            # Energy prices
            lam_mp_mean        = float(lam_mp.mean()),
            lam_mp_max         = float(lam_mp.max()),
            lam_mp_cv          = cv(lam_mp),
            lam_achp_mean      = float(lam_achp.mean()),
            lam_achp_max       = float(lam_achp.max()),
            lam_achp_cv        = cv(lam_achp),
            lam_achp_premium   = float((lam_achp - lam_mp).mean()),  # avg ACHP - MP gap
            lam_wm_mp          = wm(lam_mp),
            lam_wm_achp        = wm(lam_achp),

            # Reserve prices
            mu_mp_mean         = float(mu_mp.mean()),
            mu_mp_nonzero_frac = float((mu_mp > 0.01).mean()),
            mu_achp_mean       = float(mu_achp.mean()),
            mu_achp_nonzero_frac = float((mu_achp > 0.01).mean()),

            # LOC
            total_loc_mp       = float(loc_mp.sum()),
            total_loc_achp     = float(loc_achp.sum()),
            loc_reduction_pct  = 100. * (loc_mp.sum() - loc_achp.sum()) / loc_mp.sum()
                                  if loc_mp.sum() > 0 else 0.,
            n_generators_loc_mp   = int((loc_mp   > 1.).sum()),
            n_generators_loc_achp = int((loc_achp > 1.).sum()),

            # Revenue adequacy
            n_need_uplift_mp   = n_need_mp,
            n_need_uplift_achp = n_need_achp,
            pct_need_uplift_mp = 100. * n_need_mp   / G,
            pct_need_uplift_achp = 100. * n_need_achp / G,

            # Commitment
            avg_commitment_rate = 100. * avg_commitment,
            n_always_on         = n_always_on,
            n_never_on          = n_never_on,
            n_sometimes_on      = G - n_always_on - n_never_on,
        ))

    df = pd.DataFrame(rows).set_index("key")

    # ── hw vs lw comparison ───────────────────────────────────────────────────
    hw_lw_rows = []
    months = df["month"].unique()
    for month in sorted(months):
        subset = df[df["month"] == month]
        hw = subset[subset["tag"] == "hw"]
        lw = subset[subset["tag"] == "lw"]
        if len(hw) == 0 or len(lw) == 0:
            continue
        hw = hw.iloc[0]; lw = lw.iloc[0]
        hw_lw_rows.append(dict(
            month                     = month,
            hw_peak_demand            = hw["peak_demand"],
            lw_peak_demand            = lw["peak_demand"],
            demand_ratio              = hw["peak_demand"] / lw["peak_demand"] if lw["peak_demand"] > 0 else np.nan,

            hw_cost_uc                = hw["cost_uc"],
            lw_cost_uc                = lw["cost_uc"],
            cost_ratio                = hw["cost_uc"] / lw["cost_uc"] if lw["cost_uc"] > 0 else np.nan,

            hw_integrality_gap        = hw["integrality_gap"],
            lw_integrality_gap        = lw["integrality_gap"],

            hw_lam_mp_mean            = hw["lam_mp_mean"],
            lw_lam_mp_mean            = lw["lam_mp_mean"],
            hw_lam_achp_mean          = hw["lam_achp_mean"],
            lw_lam_achp_mean          = lw["lam_achp_mean"],
            hw_lam_achp_premium       = hw["lam_achp_premium"],
            lw_lam_achp_premium       = lw["lam_achp_premium"],

            hw_total_loc_mp           = hw["total_loc_mp"],
            lw_total_loc_mp           = lw["total_loc_mp"],
            hw_total_loc_achp         = hw["total_loc_achp"],
            lw_total_loc_achp         = lw["total_loc_achp"],
            hw_loc_reduction_pct      = hw["loc_reduction_pct"],
            lw_loc_reduction_pct      = lw["loc_reduction_pct"],

            hw_pct_need_uplift_mp     = hw["pct_need_uplift_mp"],
            lw_pct_need_uplift_mp     = lw["pct_need_uplift_mp"],
            hw_avg_commitment_rate    = hw["avg_commitment_rate"],
            lw_avg_commitment_rate    = lw["avg_commitment_rate"],
        ))
    df_hw_lw = pd.DataFrame(hw_lw_rows).set_index("month") if hw_lw_rows else pd.DataFrame()

    # ── Summary statistics ────────────────────────────────────────────────────
    def summary(series, label):
        return {
            f"{label}_mean":   float(series.mean()),
            f"{label}_std":    float(series.std()),
            f"{label}_min":    float(series.min()),
            f"{label}_median": float(series.median()),
            f"{label}_max":    float(series.max()),
        }

    pricing_stats = {}
    pricing_stats.update(summary(df["lam_mp_mean"],       "lam_mp"))
    pricing_stats.update(summary(df["lam_achp_mean"],     "lam_achp"))
    pricing_stats.update(summary(df["lam_achp_premium"],  "achp_premium"))
    pricing_stats.update(summary(df["lam_mp_cv"],         "lam_mp_cv"))
    pricing_stats.update(summary(df["lam_achp_cv"],       "lam_achp_cv"))
    pricing_stats.update(summary(df["mu_mp_mean"],        "mu_mp"))
    pricing_stats.update(summary(df["mu_mp_nonzero_frac"],"mu_nonzero_frac"))
    pricing_stats["pct_instances_reserve_binding"] = float(
        (df["mu_mp_nonzero_frac"] > 0).mean() * 100
    )

    loc_stats = {}
    loc_stats.update(summary(df["total_loc_mp"],         "total_loc_mp"))
    loc_stats.update(summary(df["total_loc_achp"],       "total_loc_achp"))
    loc_stats.update(summary(df["loc_reduction_pct"],    "loc_reduction_pct"))
    loc_stats.update(summary(df["pct_need_uplift_mp"],   "pct_need_uplift_mp"))
    loc_stats.update(summary(df["pct_need_uplift_achp"], "pct_need_uplift_achp"))
    loc_stats.update(summary(df["integrality_gap"],      "integrality_gap"))

    gen_stats = {}
    gen_stats.update(summary(df["avg_commitment_rate"],  "avg_commitment"))
    gen_stats.update(summary(df["n_always_on"],          "n_always_on"))
    gen_stats.update(summary(df["n_never_on"],           "n_never_on"))
    gen_stats.update(summary(df["n_sometimes_on"],       "n_sometimes_on"))

    out = dict(
        per_instance = df,
        hw_vs_lw     = df_hw_lw,
        pricing      = pricing_stats,
        loc          = loc_stats,
        generators   = gen_stats,
    )

    if print_report:
        _print_report(df, df_hw_lw, pricing_stats, loc_stats, gen_stats)

    return out


# ══════════════════════════════════════════════════════════════════════════════
#  3.  PRINT REPORT
# ══════════════════════════════════════════════════════════════════════════════

def _print_report(df, df_hw_lw, pricing, loc, gen):
    N = len(df)
    n_hw = (df["tag"] == "hw").sum()
    n_lw = (df["tag"] == "lw").sum()

    print("\n" + "═"*70)
    print("  STATISTICAL REPORT  —  Assignment 3 Cross-Instance Analysis")
    print("═"*70)
    print(f"  Instances analysed: {N}  ({n_hw} hw, {n_lw} lw)")
    print(f"  Months covered:     {df['month'].nunique()}")
    print(f"  Generators (G):     {df['G'].iloc[0] if len(df)>0 else '?'}")
    print(f"  Periods  (T):       {df['T'].iloc[0] if len(df)>0 else '?'}")

    # ── Section 1: Costs ──────────────────────────────────────────────────────
    print("\n" + "─"*70)
    print("  1. COSTS")
    print("─"*70)
    fmt = "    {:45s}: {:>10.2f}  ±{:.2f}"
    print(fmt.format("UC total cost (mean ± std)",
                     df["cost_uc"].mean(), df["cost_uc"].std()))
    print(fmt.format("ED total cost (mean ± std)",
                     df["cost_ed"].mean(), df["cost_ed"].std()))
    print(fmt.format("LP relaxation cost (mean ± std)",
                     df["cost_lp"].mean(), df["cost_lp"].std()))
    print(fmt.format("Integrality gap % (mean ± std)",
                     df["integrality_gap"].mean(), df["integrality_gap"].std()))
    print(fmt.format("UC−ED cost difference % (mean ± std)",
                     df["delta_cost_pct"].mean(), df["delta_cost_pct"].std()))
    print()
    print("    hw instances:")
    hw = df[df["tag"]=="hw"]
    lw = df[df["tag"]=="lw"]
    print(fmt.format("      UC cost",  hw["cost_uc"].mean(), hw["cost_uc"].std()))
    print("    lw instances:")
    print(fmt.format("      UC cost",  lw["cost_uc"].mean(), lw["cost_uc"].std()))

    # ── Section 2: Prices ─────────────────────────────────────────────────────
    print("\n" + "─"*70)
    print("  2. ENERGY PRICES")
    print("─"*70)
    print(fmt.format("λ_MP  mean (mean over instances ± std)",
                     pricing["lam_mp_mean"],  pricing["lam_mp_std"]))
    print(fmt.format("λ_MP  max  (mean over instances ± std)",
                     df["lam_mp_max"].mean(), df["lam_mp_max"].std()))
    print(fmt.format("λ_MP  coeff. of variation (mean ± std)",
                     pricing["lam_mp_cv_mean"], pricing["lam_mp_cv_std"]))
    print(fmt.format("λ_ACHP mean (mean ± std)",
                     pricing["lam_achp_mean"], pricing["lam_achp_std"]))
    print(fmt.format("ACHP premium over MP  (mean ± std)",
                     pricing["achp_premium_mean"], pricing["achp_premium_std"]))
    print()
    print(fmt.format("hw: λ_MP mean",  hw["lam_mp_mean"].mean(),  hw["lam_mp_mean"].std()))
    print(fmt.format("lw: λ_MP mean",  lw["lam_mp_mean"].mean(),  lw["lam_mp_mean"].std()))
    print(fmt.format("hw: λ_ACHP mean",hw["lam_achp_mean"].mean(),hw["lam_achp_mean"].std()))
    print(fmt.format("lw: λ_ACHP mean",lw["lam_achp_mean"].mean(),lw["lam_achp_mean"].std()))

    # ── Section 3: Reserve prices ─────────────────────────────────────────────
    print("\n" + "─"*70)
    print("  3. RESERVE PRICES")
    print("─"*70)
    print(fmt.format("μ_MP  mean (mean ± std)",
                     pricing["mu_mp_mean"], pricing["mu_mp_std"]))
    print(f"    {'Fraction of periods with binding reserve (MP):':45s}"
          f"  {pricing['mu_nonzero_frac_mean']*100:>6.1f}%")
    print(f"    {'Instances where any reserve period is binding:':45s}"
          f"  {pricing['pct_instances_reserve_binding']:>6.1f}%")

    # ── Section 4: LOC ────────────────────────────────────────────────────────
    print("\n" + "─"*70)
    print("  4. LOSS OPPORTUNITY COST (LOC)")
    print("─"*70)
    print(fmt.format("Total LOC under MP   (mean ± std)",
                     loc["total_loc_mp_mean"],   loc["total_loc_mp_std"]))
    print(fmt.format("Total LOC under ACHP (mean ± std)",
                     loc["total_loc_achp_mean"], loc["total_loc_achp_std"]))
    print(fmt.format("LOC reduction MP→ACHP %  (mean ± std)",
                     loc["loc_reduction_pct_mean"], loc["loc_reduction_pct_std"]))
    print(fmt.format("% generators needing uplift under MP  (mean)",
                     loc["pct_need_uplift_mp_mean"], loc["pct_need_uplift_mp_std"]))
    print(fmt.format("% generators needing uplift under ACHP (mean)",
                     loc["pct_need_uplift_achp_mean"], loc["pct_need_uplift_achp_std"]))
    print()
    print(fmt.format("hw: total LOC MP",  hw["total_loc_mp"].mean(),   hw["total_loc_mp"].std()))
    print(fmt.format("lw: total LOC MP",  lw["total_loc_mp"].mean(),   lw["total_loc_mp"].std()))
    print(fmt.format("hw: total LOC ACHP",hw["total_loc_achp"].mean(), hw["total_loc_achp"].std()))
    print(fmt.format("lw: total LOC ACHP",lw["total_loc_achp"].mean(), lw["total_loc_achp"].std()))

    # ── Section 5: Commitment ─────────────────────────────────────────────────
    print("\n" + "─"*70)
    print("  5. COMMITMENT PATTERNS")
    print("─"*70)
    print(fmt.format("Avg commitment rate %  (mean ± std)",
                     gen["avg_commitment_mean"], gen["avg_commitment_std"]))
    print(fmt.format("Always-on generators  (mean ± std)",
                     gen["n_always_on_mean"],    gen["n_always_on_std"]))
    print(fmt.format("Never-on  generators  (mean ± std)",
                     gen["n_never_on_mean"],     gen["n_never_on_std"]))
    print(fmt.format("Flexible  generators  (mean ± std)",
                     gen["n_sometimes_on_mean"], gen["n_sometimes_on_std"]))
    print()
    print(fmt.format("hw: avg commitment rate",
                     hw["avg_commitment_rate"].mean(), hw["avg_commitment_rate"].std()))
    print(fmt.format("lw: avg commitment rate",
                     lw["avg_commitment_rate"].mean(), lw["avg_commitment_rate"].std()))

    # ── Section 6: hw vs lw summary table ────────────────────────────────────
    if len(df_hw_lw) > 0:
        print("\n" + "─"*70)
        print("  6. HW vs LW — MONTH-BY-MONTH SUMMARY")
        print("─"*70)
        cols = ["hw_peak_demand","lw_peak_demand","hw_cost_uc","lw_cost_uc",
                "hw_total_loc_mp","lw_total_loc_mp","hw_loc_reduction_pct","lw_loc_reduction_pct"]
        print(df_hw_lw[cols].to_string(float_format=lambda x: f"{x:,.1f}"))

    print("\n" + "═"*70)
    print("  KEY FINDINGS FOR THEORETICAL ANALYSIS")
    print("═"*70)

    ig_mean = loc["integrality_gap_mean"]
    red_mean = loc["loc_reduction_pct_mean"]
    prem_mean = pricing["achp_premium_mean"]
    up_mp  = loc["pct_need_uplift_mp_mean"]
    up_ach = loc["pct_need_uplift_achp_mean"]

    print(f"""
  1. Integrality gap averages {ig_mean:.2f}%, indicating the true cost of
     integer constraints (startup costs, min-up/down) above the LP lower bound.
     Higher gaps in hw instances reflect tighter capacity utilisation.

  2. ACHP prices are on average {prem_mean:.2f} $/MWh higher than MP prices,
     because the LP relaxation internalises fixed and startup costs into the
     dual — MP only reflects marginal variable costs.

  3. ACHP reduces total LOC by {red_mean:.1f}% on average vs MP, confirming
     the theoretical result that ACHP minimises aggregate opportunity costs
     over the convex hull relaxation.

  4. Under MP prices, {up_mp:.1f}% of generators have negative profit and
     require make-whole uplift.  Under ACHP this falls to {up_ach:.1f}%.
     hw instances show larger uplift requirements (tighter system, more
     peakers committed at startup costs not covered by marginal prices).

  5. hw vs lw: high-load days have higher prices, higher LOC, and more
     peakers committed. Low-load days have near-zero reserve prices and
     almost all LOC concentrated in base-load generators forced to run
     below minimum output.
""")


# ══════════════════════════════════════════════════════════════════════════════
#  4.  WHICH INSTANCES TO USE
# ══════════════════════════════════════════════════════════════════════════════

def recommend_instances(results):
    """
    Print a recommendation on which hw/lw instances are most informative
    for the theoretical analysis.
    """
    stats = compute_statistics(results, print_report=False)
    df    = stats["per_instance"]
    hw    = df[df["tag"] == "hw"]
    lw    = df[df["tag"] == "lw"]

    print("\n" + "═"*70)
    print("  INSTANCE SELECTION RECOMMENDATION")
    print("═"*70)

    print("""
  HW vs LW: use BOTH for a complete analysis.

  HW (high-water / high-load) instances
  ───────────────────────────────────────
  • Represent stressed system conditions (summer peak, winter peak).
  • More peakers are committed → larger startup costs → larger integrality
    gap → larger difference between MP and ACHP prices.
  • LOC is concentrated in peakers and mid-merit generators that are
    committed but only partially loaded — these show the clearest
    nonconvexity effects.
  • Reserve constraints are more likely to be binding → μ_t > 0.
  • Best instance for illustrating WHY nonconvex pricing matters.
  • Recommended: month with highest peak demand (typically July or August
    for PJM) for the most dramatic LOC and price spread.
""")

    if len(hw) > 0:
        best_hw = hw.loc[hw["total_loc_mp"].idxmax()]
        print(f"  → Largest LOC hw instance: {best_hw.name}")
        print(f"    peak demand={best_hw['peak_demand']:.0f} MW, "
              f"LOC_MP=${best_hw['total_loc_mp']:,.0f}, "
              f"integrality gap={best_hw['integrality_gap']:.2f}%\n")

    print("""
  LW (low-water / low-load) instances
  ─────────────────────────────────────
  • Represent off-peak conditions (weekday spring/autumn night).
  • Fewer generators committed → but must_run base-load units may be forced
    to produce at minimum output, creating LOC for them at low prices.
  • MP and ACHP prices are close (small integrality gap).
  • Reserve constraints rarely binding → μ_t ≈ 0.
  • Useful for showing that LOC is NOT always about peakers; in low-load
    conditions, must-run base-load generators face LOC because they cannot
    turn off even if prices fall below their cost.
  • Recommended: month with lowest demand (typically April or November).
""")

    if len(lw) > 0:
        best_lw = lw.loc[lw["total_loc_mp"].idxmax()]
        print(f"  → Most interesting lw instance: {best_lw.name}")
        print(f"    peak demand={best_lw['peak_demand']:.0f} MW, "
              f"LOC_MP=${best_lw['total_loc_mp']:,.0f}, "
              f"integrality gap={best_lw['integrality_gap']:.2f}%\n")

    print("""
  Summary recommendation
  ──────────────────────
  For the theoretical write-up, report results for ALL 24 instances
  (12 months × 2 load levels) as cross-instance statistics, and pick
  one hw + one lw instance as illustrative case studies:
    • hw case study: the month with the highest peak demand
      (maximises price spread and LOC → clearest nonconvexity effects)
    • lw case study: the month with the largest LOC among lw instances
      (shows that nonconvexity issues are not exclusive to peak conditions)
  This provides both breadth (all 24 in tables) and depth (2 detailed).
""")


# ══════════════════════════════════════════════════════════════════════════════
#  5.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python statistics.py results.json             # load and analyse")
        print("  python statistics.py --run instances_dir/     # solve all then analyse")
        sys.exit(0)

    if sys.argv[1] == "--run":
        if len(sys.argv) < 3:
            print("Provide the instances directory path.")
            sys.exit(1)
        import importlib.util
        spec = importlib.util.spec_from_file_location("homework",
               sys.argv[3] if len(sys.argv) > 3 else "homework.py")
        hw_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hw_mod)

        results = collect_results(sys.argv[2], hw_mod,
                                   save_path="results.json")
    else:
        with open(sys.argv[1]) as f:
            results = json.load(f)

    stats = compute_statistics(results)
    recommend_instances(results)