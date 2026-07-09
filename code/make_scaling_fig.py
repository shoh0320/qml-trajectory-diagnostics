import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata

plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.linewidth": 0.8,
                     "mathtext.fontset": "dejavuserif", "savefig.dpi": 300})

s  = pd.read_csv("results_summary.csv")
bd = pd.read_csv("results_barren.csv")
pt = pd.read_csv("results_pertask.csv"); pt["absG"] = pt["G"].abs()
ns = sorted(s["n"].unique()); Ls = sorted(s["L"].unique())
colors = {2: "#1f77b4", 3: "#ff7f0e", 5: "#2ca02c"}
rng = np.random.default_rng(1); Bn = 1500

def rho(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.std(x[m]) == 0 or np.std(y[m]) == 0: return np.nan
    return spearmanr(x[m], y[m])[0]

def partial(x, y, Z, df):
    d = df[[x, y] + Z].dropna()
    R = {c: rankdata(d[c].values) for c in [x, y] + Z}
    M = np.column_stack([np.ones(len(d))] + [R[c] for c in Z])
    res = lambda v: v - M @ np.linalg.lstsq(M, v, rcond=None)[0]
    ex, ey = res(R[x]), res(R[y])
    return np.corrcoef(ex, ey)[0, 1]

def boot_ci(dfn, fn):
    cfgs = [g for _, g in dfn.groupby(["L", "seed"])]
    out = np.empty(Bn)
    for k in range(Bn):
        ch = [cfgs[i] for i in rng.integers(0, len(cfgs), len(cfgs))]
        fr = pd.concat([c.iloc[rng.integers(0, len(c), len(c))] for c in ch], ignore_index=True)
        out[k] = fn(fr)
    return np.nanpercentile(out, [2.5, 97.5])

fig, axs = plt.subplots(2, 2, figsize=(7.1, 5.8))

# ---- (a) CV vs n per L + Gaussian dimensional null ----
for L in Ls:
    g = s[s["L"] == L].groupby("n")["cv_gtr"].mean()
    axs[0, 0].plot(g.index, g.values, "o-", color=colors[L], label=f"L={L}", ms=4.5)
nn = np.linspace(min(ns), max(ns), 50)
for L, ls in zip(Ls, [":", "--", "-."]):
    axs[0, 0].plot(nn, 1/np.sqrt(2*2*nn*L), ls, color=colors[L], lw=0.9, alpha=0.7)
axs[0, 0].plot([], [], "k--", lw=0.9, label=r"i.i.d. null $1/\sqrt{2d}$")
axs[0, 0].set_xlabel("qubits $n$"); axs[0, 0].set_ylabel(r"$\mathrm{CV}(\|g_{\mathrm{tr}}\|)$")
axs[0, 0].set_title("(a) gradient-norm concentration", fontsize=9)
axs[0, 0].legend(frameon=False, fontsize=7); axs[0, 0].set_xticks(ns)

# ---- (b) BPvar vs n per L (log): markers+guide line through points; b from exp fit ----
for L in Ls:
    bb = bd[bd["L"] == L].groupby("n")["grad_comp_var"].mean()
    x = np.array(bb.index, float)
    b_rate = -np.polyfit(x, np.log(bb.values), 1)[0]          # exponential-fit decay rate
    axs[0, 1].semilogy(bb.index, bb.values, "s-", color=colors[L], ms=5, lw=1.1,
                       label=f"L={L}: $b$={b_rate:.2f}")       # line passes through the points
axs[0, 1].set_xlabel("qubits $n$"); axs[0, 1].set_ylabel(r"$\mathrm{Var}[\partial L/\partial\theta_1]$")
axs[0, 1].set_title(r"(b) barren-plateau probe: $\propto e^{-bn}$", fontsize=9)
axs[0, 1].legend(frameon=False, fontsize=7); axs[0, 1].set_xticks(ns)

# ---- (c) pooled rho(A,Ival) vs n with bootstrap CI band -- POOLED CURVE ONLY ----
est_c, lo_c, hi_c = [], [], []
for n in ns:
    dfn = pt[pt["n"] == n]
    est_c.append(rho(dfn["align"], dfn["I_val"]))
    lo, hi = boot_ci(dfn, lambda f: rho(f["align"], f["I_val"])); lo_c.append(lo); hi_c.append(hi)
axs[1, 0].fill_between(ns, lo_c, hi_c, color="#333333", alpha=0.15)
axs[1, 0].plot(ns, est_c, "ko-", ms=5, label="pooled (95% CI)")
axs[1, 0].axhline(0, color="grey", lw=0.7)
axs[1, 0].set_xlabel("qubits $n$"); axs[1, 0].set_ylabel(r"$\rho(A,\,I_{\mathrm{val}})$")
axs[1, 0].set_title("(c) identity link strengthens with scale", fontsize=9)
axs[1, 0].legend(frameon=False, fontsize=7.5); axs[1, 0].set_xticks(ns); axs[1, 0].set_ylim(0, 1)

# ---- (d) raw / |Ival / |Ival,Lval0 pooled per n -> joint control collapses to 0 ----
raw, p1, p2, p2lo, p2hi = [], [], [], [], []
for n in ns:
    dfn = pt[pt["n"] == n]
    raw.append(rho(dfn["align"], dfn["G"]))
    p1.append(partial("align", "G", ["I_val"], dfn))
    p2.append(partial("align", "G", ["I_val", "Lval0"], dfn))
    lo, hi = boot_ci(dfn, lambda f: partial("align", "G", ["I_val", "Lval0"], f))
    p2lo.append(lo); p2hi.append(hi)
axs[1, 1].plot(ns, raw, "x--", c="#d62728", ms=8, mew=2, label=r"$\rho(A,G)$ raw")
axs[1, 1].plot(ns, p1, "o-", c="#7a4fb0", ms=5, label=r"$\rho(A,G\,|\,I_{\mathrm{val}})$")
axs[1, 1].fill_between(ns, p2lo, p2hi, color="#17375e", alpha=0.15)
axs[1, 1].plot(ns, p2, "s-", c="#17375e", ms=5,
               label=r"$\rho(A,G\,|\,I_{\mathrm{val}},L_{\mathrm{val}}(\theta_0))$")
axs[1, 1].axhline(0, color="grey", lw=0.7)
axs[1, 1].set_xlabel("qubits $n$"); axs[1, 1].set_ylabel(r"Spearman $\rho$")
axs[1, 1].set_title("(d) gap association vanishes under joint control", fontsize=9)
_ymin = min(min(raw), min(p1), min(p2lo)) - 0.05
axs[1, 1].set_ylim(_ymin, 0.34)
axs[1, 1].legend(frameon=False, fontsize=6.4, loc="upper left",
                 handlelength=1.6, borderaxespad=0.3, labelspacing=0.3)
axs[1, 1].set_xticks(ns)

plt.tight_layout()
plt.savefig("figs/fig_scaling.pdf", bbox_inches="tight")
plt.close()
print("wrote figs/fig_scaling.pdf")
print("(b) b per L:", {L: round(-np.polyfit(np.array(bd[bd.L==L].groupby('n')['grad_comp_var'].mean().index,float), np.log(bd[bd.L==L].groupby('n')['grad_comp_var'].mean().values),1)[0],3) for L in Ls})
print("(c) rho(A,Ival) pooled:", [round(x,3) for x in est_c])
print("(d) raw:", [round(x,3) for x in raw], "| joint:", [round(x,3) for x in p2])
