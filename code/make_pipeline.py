import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "font.size": 7.4,
                     "mathtext.fontset": "cm", "savefig.dpi": 300})

def box(ax, cx, cy, w, h, text, fc, ec="#3a3a3a", fs=7.4):
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.025",
                 linewidth=0.9, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            zorder=3, linespacing=1.4)

def arrow(ax, p0, p1, color="#555", dashed=False):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12,
                 lw=1.05, color=color, zorder=1, shrinkA=0, shrinkB=0,
                 linestyle=(0, (4, 2)) if dashed else "solid"))

def elbow(ax, p0, corner, p1, color="#555", dashed=False):
    """Two-segment orthogonal connector p0 -> corner -> p1, arrowhead at p1."""
    ax.add_patch(FancyArrowPatch(p0, corner, arrowstyle="-", mutation_scale=12,
                 lw=1.05, color=color, zorder=1, shrinkA=0, shrinkB=0,
                 linestyle=(0, (4, 2)) if dashed else "solid"))
    ax.add_patch(FancyArrowPatch(corner, p1, arrowstyle="-|>", mutation_scale=12,
                 lw=1.05, color=color, zorder=1, shrinkA=0, shrinkB=0,
                 linestyle=(0, (4, 2)) if dashed else "solid"))

def lbl(ax, x, y, t, color="#333", ha="center", va="center"):
    ax.text(x, y, t, fontsize=6.5, style="italic", color=color,
            ha=ha, va=va, zorder=4,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))

fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.set_xlim(0, 12); ax.set_ylim(0, 8.2); ax.axis("off")

c_setup="#dbe7f3"; c_diag="#fbe5d6"; c_corr="#fde3c8"; c_ctrl="#e6e0ef"
c_target="#e2efda"; c_phys="#fff2cc"; c_verdict="#f2dede"

BW, BH = 3.5, 1.15
xL, xM, xR = 2.15, 6.0, 9.85
yTop, yMid, yBot = 6.7, 4.3, 1.8

# ---- top spine ----
box(ax, xL, yTop, BW, BH,
    "Variational quantum\nmeta-adaptation\n" + r"$\theta_0 \to \theta_K$", c_setup)
box(ax, xM, yTop, BW, BH,
    "Scalar diagnostics\n" + r"drift, $V_{\mathrm{tr}}$, $\cos(g_{\mathrm{tr}},g_{\mathrm{val}})$", c_diag)
box(ax, xR, yTop, BW, BH,
    "Apparent correlation\n(diagnostic vs. target)", c_corr)
# ---- controls (under correlation) ----
box(ax, xR, yMid, BW, BH + 0.15,
    "Artifact controls\n" + r"partial corr. incl. baseline" + "\n" +
    r"shuffle $\cdot$ linearization $\cdot$ seeds", c_ctrl)
# ---- verdict (under controls) ----
box(ax, xR, yBot, BW, BH,
    "Verdict\n" + r"collapse $\Rightarrow$ artifact" + "\n" + r"survive $\Rightarrow$ candidate", c_verdict)
# ---- pre-specified target (left of correlation row-ish, bottom-mid) ----
box(ax, xM, yBot, BW, BH,
    "Pre-specified target\n" + r"$G=L_{\mathrm{val}}(\theta_K)-L_{\mathrm{tr}}(\theta_K)$", c_target)
# ---- physical activators (bottom-left) ----
box(ax, xL, yBot, BW, BH + 0.15,
    "Physical activators\n" + r"drift $\cdot$ grad. concentration" + "\n" +
    r"$\cdot$ observable decomp.", c_phys)

# ================= connectors, each labelled =================
# 1) adaptation -> diagnostics (the trajectory produces the scalars)
arrow(ax, (xL + BW/2, yTop), (xM - BW/2, yTop))
lbl(ax, (xL + xM)/2, yTop + 0.34, "trajectory")
# 2) diagnostics -> apparent correlation
arrow(ax, (xM + BW/2, yTop), (xR - BW/2, yTop))
lbl(ax, (xM + xR)/2, yTop + 0.34, "correlate")
# 3) pre-specified target -> apparent correlation (the target it is correlated against)
#    3-segment route entering the correlation box bottom-left, offset from arrow 4
xEnter = xR - 1.0
yRun = 5.35
ax.add_patch(FancyArrowPatch((xM, yBot + BH/2), (xM, yRun), arrowstyle="-",
             mutation_scale=12, lw=1.05, color="#555", zorder=1, shrinkA=0, shrinkB=0))
ax.add_patch(FancyArrowPatch((xM, yRun), (xEnter, yRun), arrowstyle="-",
             mutation_scale=12, lw=1.05, color="#555", zorder=1, shrinkA=0, shrinkB=0))
ax.add_patch(FancyArrowPatch((xEnter, yRun), (xEnter, yTop - BH/2), arrowstyle="-|>",
             mutation_scale=12, lw=1.05, color="#555", zorder=1, shrinkA=0, shrinkB=0))
lbl(ax, xM, (yBot + yRun)/2 + 0.2, "compared\nagainst")
# 4) apparent correlation -> controls
arrow(ax, (xR, yTop - BH/2), (xR, yMid + (BH+0.15)/2))
lbl(ax, xR + 0.95, (yTop - BH/2 + yMid + (BH+0.15)/2)/2, "test", ha="left")
# 5) controls -> verdict
arrow(ax, (xR, yMid - (BH+0.15)/2), (xR, yBot + BH/2))
lbl(ax, xR + 0.95, (yMid - (BH+0.15)/2 + yBot + BH/2)/2, "collapse /\nsurvive", ha="left")
# 6) physical activators -> verdict (they explain a collapse); dashed = interpretive
#    headless down + horizontal, single arrowhead only where it reaches the verdict box
dash = (0, (4, 2)); gold = "#a08a3a"
ax.add_patch(FancyArrowPatch((xL, yBot - BH/2 - 0.075), (xL, 0.75), arrowstyle="-",
             mutation_scale=12, lw=1.05, color=gold, zorder=1, shrinkA=0, shrinkB=0, linestyle=dash))
ax.add_patch(FancyArrowPatch((xL, 0.75), (xR, 0.75), arrowstyle="-",
             mutation_scale=12, lw=1.05, color=gold, zorder=1, shrinkA=0, shrinkB=0, linestyle=dash))
ax.add_patch(FancyArrowPatch((xR, 0.75), (xR, yBot - BH/2), arrowstyle="-|>",
             mutation_scale=12, lw=1.05, color=gold, zorder=1, shrinkA=0, shrinkB=0, linestyle=dash))
lbl(ax, (xL + xR)/2, 0.75, "explain a collapse", color="#8a7420")

plt.savefig("figs/fig1_pipeline.pdf", bbox_inches="tight")
plt.savefig("figs/_pipeline_preview.png", bbox_inches="tight", dpi=150)
plt.close()
print("wrote figs/fig1_pipeline.pdf")
