#!/usr/bin/env python3
"""Reviewer-requested analyses computable from the existing full-run data:
 (1) A vs {G, |G|, L_val(thetaK), I_val} correlations, pooled per n, with
     hierarchical bootstrap 95% CIs (resample (L,seed) configs, then tasks).
 (2) Sensitivity partials: rho(A,G|I_val), rho(A,G|I_val,Lval0),
     rho(A,G|I_tr,I_val), rho(A,|G||I_val).
 (3) Seed-level spread of the key correlations (stability).
 (4) Null model: observed CV(||g_tr||) vs Gaussian null 1/sqrt(2d), d=2nL.
 (5) Barren-plateau variance: exponential fit slope (ln BPvar vs n) with CI.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata

pt = pd.read_csv("results_pertask.csv")
bar = pd.read_csv("results_barren.csv")
pt["absG"] = pt["G"].abs()
ns = sorted(pt["n"].unique())
rng = np.random.default_rng(0)
B = 2000


def rho(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return np.nan
    return spearmanr(x[m], y[m])[0]


def partial(x, y, Zcols, df):
    """Partial Spearman rho(x,y | Z): rank-transform, regress out Z (multiple
    linear), correlate residuals."""
    d = df[[x, y] + Zcols].dropna()
    if len(d) < 5:
        return np.nan
    R = {c: rankdata(d[c].values) for c in [x, y] + Zcols}
    Z = np.column_stack([R[c] for c in Zcols])
    Z = np.column_stack([np.ones(len(Z)), Z])  # intercept
    def resid(v):
        beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ beta
    ex, ey = resid(R[x]), resid(R[y])
    if np.std(ex) == 0 or np.std(ey) == 0:
        return np.nan
    return np.corrcoef(ex, ey)[0, 1]


def hier_boot_ci(dfn, xcol, ycol, partial_ctrl=None, B=B):
    """Hierarchical bootstrap: resample (L,seed) configs w/ replacement, then
    tasks within each, pool, recompute correlation. Returns (est, lo, hi)."""
    configs = [g for _, g in dfn.groupby(["L", "seed"])]
    def stat(frame):
        if partial_ctrl is None:
            return rho(frame[xcol], frame[ycol])
        return partial(xcol, ycol, partial_ctrl, frame)
    est = stat(dfn)
    boot = np.empty(B)
    for b in range(B):
        chosen = [configs[i] for i in rng.integers(0, len(configs), len(configs))]
        parts = []
        for c in chosen:
            idx = rng.integers(0, len(c), len(c))
            parts.append(c.iloc[idx])
        frame = pd.concat(parts, ignore_index=True)
        boot[b] = stat(frame)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return est, lo, hi


print("="*96)
print("(1)+(3) POOLED CORRELATIONS PER n WITH HIERARCHICAL BOOTSTRAP 95% CI  (N=450/n)")
print("="*96)
targets = [("I_val", "rho(A, I_val)"), ("G", "rho(A, G)"),
           ("absG", "rho(A, |G|)"), ("LvalK", "rho(A, L_val(thetaK))")]
for lbl_pair in targets:
    col, name = lbl_pair
    print(f"\n{name}")
    print(f"  {'n':>3} {'est':>8} {'95% CI':>20}   seed-means (L-pooled)")
    for n in ns:
        dfn = pt[pt["n"] == n]
        est, lo, hi = hier_boot_ci(dfn, "align", col)
        seedvals = [rho(dfn[dfn.seed == s]["align"], dfn[dfn.seed == s][col])
                    for s in sorted(dfn.seed.unique())]
        sv = " ".join(f"{v:+.2f}" for v in seedvals)
        print(f"  {n:>3} {est:>+8.3f} [{lo:>+6.3f}, {hi:>+6.3f}]   [{sv}]")

print("\n" + "="*96)
print("(1) PARTIAL rho(A, G | I_val) PER n WITH HIERARCHICAL BOOTSTRAP 95% CI")
print("="*96)
print(f"  {'n':>3} {'est':>8} {'95% CI':>20}")
for n in ns:
    dfn = pt[pt["n"] == n]
    est, lo, hi = hier_boot_ci(dfn, "align", "G", partial_ctrl=["I_val"])
    print(f"  {n:>3} {est:>+8.3f} [{lo:>+6.3f}, {hi:>+6.3f}]")

print("\n" + "="*96)
print("(2) SENSITIVITY PARTIALS (point estimates, pooled per n)")
print("="*96)
print(f"  {'n':>3} {'A,G|Ival':>10} {'A,G|Ival,Lval0':>16} {'A,G|Itr,Ival':>14} {'A,|G||Ival':>12}")
for n in ns:
    dfn = pt[pt["n"] == n]
    p1 = partial("align", "G", ["I_val"], dfn)
    p2 = partial("align", "G", ["I_val", "Lval0"], dfn)
    p3 = partial("align", "G", ["I_tr", "I_val"], dfn)
    p4 = partial("align", "absG", ["I_val"], dfn)
    print(f"  {n:>3} {p1:>+10.3f} {p2:>+16.3f} {p3:>+14.3f} {p4:>+12.3f}")

print("\n" + "="*96)
print("(4) NULL MODEL: observed CV(||g_tr||) vs Gaussian null 1/sqrt(2d), d=2nL")
print("="*96)
print(f"  {'n':>3} {'L':>3} {'d=2nL':>6} {'CV_obs':>8} {'CV_null':>8} {'ratio':>7}")
for n in ns:
    for L in sorted(pt[pt.n == n]["L"].unique()):
        sub = pt[(pt.n == n) & (pt.L == L)]["norm_gtr"].values
        cv_obs = np.std(sub) / np.mean(sub)
        d = 2 * n * L
        cv_null = 1.0 / np.sqrt(2 * d)
        print(f"  {n:>3} {L:>3} {d:>6} {cv_obs:>8.3f} {cv_null:>8.3f} {cv_obs/cv_null:>7.2f}")

print("\n" + "="*96)
print("(5) BARREN-PLATEAU VARIANCE: exponential fit ln(Var[dL/dtheta_1]) = a - b*n")
print("="*96)
print(f"  {'L':>3} {'slope b (per qubit)':>20} {'95% CI (boot)':>22}")
for L in sorted(bar["L"].unique()):
    sub = bar[bar["L"] == L].groupby("n")["grad_comp_var"].mean()
    x = np.array(sub.index, float); y = np.log(sub.values)
    b_hat = -np.polyfit(x, y, 1)[0]
    # bootstrap over seeds
    raw = bar[bar["L"] == L]
    slopes = []
    for _ in range(1000):
        samp = raw.groupby("n").apply(
            lambda g: g.iloc[rng.integers(0, len(g), len(g))], include_groups=False)
        s = samp.groupby("n")["grad_comp_var"].mean()
        xx = np.array(s.index, float); yy = np.log(s.values)
        slopes.append(-np.polyfit(xx, yy, 1)[0])
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    print(f"  {L:>3} {b_hat:>20.3f} [{lo:>+.3f}, {hi:>+.3f}]")
