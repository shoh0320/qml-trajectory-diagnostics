import argparse
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import pennylane as qml
from pennylane import numpy as pnp


# --------------------------------------------------------------------------- #
# Task family : random anisotropic Heisenberg/XYZ with local Z fields
# --------------------------------------------------------------------------- #
def build_ops(n):
    """Fixed Pauli operators for an n-qubit open XYZ chain + local Z fields.

    Order: [XX_i, YY_i, ZZ_i for i=0..n-2] then [Z_i for i=0..n-1].
    Length M = 3*(n-1) + n = 4n - 3.  (n=6 -> 21, matching the paper.)
    """
    ops = []
    for i in range(n - 1):
        ops.append(qml.PauliX(i) @ qml.PauliX(i + 1))
        ops.append(qml.PauliY(i) @ qml.PauliY(i + 1))
        ops.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))
    for i in range(n):
        ops.append(qml.PauliZ(i))
    return ops


def draw_coeffs(rng, n, coupling_lo=0.5, coupling_hi=1.5, field_scale=0.5):
    """Draw one task's coefficients.

    Couplings (XX/YY/ZZ, each bond drawn independently -> anisotropic) and
    random local Z fields. REPLACE with your v17 generator for exact parity.
    """
    M = 4 * n - 3
    n_coupl = 3 * (n - 1)
    c = np.empty(M)
    signs = rng.choice([-1.0, 1.0], size=n_coupl)
    c[:n_coupl] = rng.uniform(coupling_lo, coupling_hi, size=n_coupl) * signs
    c[n_coupl:] = rng.normal(0.0, field_scale, size=n)
    return c


def split_indices(rng, M, support_frac=2.0 / 3.0):
    """Disjoint observable support/query split (fixed once per task)."""
    m_tr = int(round(support_frac * M))
    m_tr = max(1, min(M - 1, m_tr))
    perm = rng.permutation(M)
    return perm[:m_tr], perm[m_tr:]


# --------------------------------------------------------------------------- #
# Ansatz and loss qnodes
# --------------------------------------------------------------------------- #
def ansatz(theta, n, L):
    """Hardware-efficient ansatz: L layers of (RY,RZ) per qubit + linear CNOTs.

    theta has shape (L, n, 2).
    """
    for l in range(L):
        for w in range(n):
            qml.RY(theta[l, w, 0], wires=w)
            qml.RZ(theta[l, w, 1], wires=w)
        for w in range(n - 1):
            qml.CNOT(wires=[w, w + 1])


def make_loss(dev, n, L, coeffs_subset, ops_subset):
    """Bounded subset loss <H_S>/sum|c|, differentiable via backprop.

    coeffs_subset are the (raw) task coefficients on the chosen subset;
    they are L1-normalized here so the loss lies in [-1, 1].
    """
    denom = float(np.sum(np.abs(coeffs_subset)))
    if denom == 0.0:
        denom = 1.0
    norm_coeffs = [float(c) / denom for c in coeffs_subset]
    H = qml.Hamiltonian(norm_coeffs, list(ops_subset))

    @qml.qnode(dev, diff_method="backprop")
    def qn(theta):
        ansatz(theta, n, L)
        return qml.expval(H)

    return qn


def flat_grad(qn, theta):
    """Flattened gradient of a scalar qnode at theta."""
    g = qml.grad(qn)(theta)
    return np.asarray(g, dtype=float).reshape(-1)


# --------------------------------------------------------------------------- #
# Reptile meta-initialization
# --------------------------------------------------------------------------- #
def reptile_meta_init(dev, n, L, ops, rng, n_outer, K, inner_lr, outer_lr,
                      init_scale, coeff_kw):
    """Return a Reptile-style meta-initialization theta_0 (pnp array)."""
    theta = pnp.array(rng.normal(0.0, init_scale, size=(L, n, 2)),
                      requires_grad=True)
    M = len(ops)
    for _ in range(n_outer):
        c = draw_coeffs(rng, n, **coeff_kw)
        tr_idx, _ = split_indices(rng, M)
        qn_tr = make_loss(dev, n, L, c[tr_idx], [ops[j] for j in tr_idx])
        phi = pnp.array(theta, requires_grad=True)
        for _ in range(K):
            g = qml.grad(qn_tr)(phi)
            phi = pnp.array(phi - inner_lr * g, requires_grad=True)
        theta = pnp.array(theta + outer_lr * (phi - theta), requires_grad=True)
    return theta


def adapt(dev, n, L, theta0, ops, tr_idx, c, K, inner_lr):
    """K-step inner adaptation on the support loss; return adapted params."""
    qn_tr = make_loss(dev, n, L, c[tr_idx], [ops[j] for j in tr_idx])
    phi = pnp.array(theta0, requires_grad=True)
    for _ in range(K):
        g = qml.grad(qn_tr)(phi)
        phi = pnp.array(phi - inner_lr * g, requires_grad=True)
    return phi


# --------------------------------------------------------------------------- #
# (A) Barren-plateau probe at random initializations
# --------------------------------------------------------------------------- #
def barren_probe(dev, n, L, ops, rng, R, coeff_kw):
    """Var and mean of a single gradient component + norm stats at random init.

    Canonical barren-plateau signature: Var[ dL/dtheta_1 ] should decay (roughly
    exponentially) with n for expressive hardware-efficient ansaetze.
    """
    M = len(ops)
    comp = []
    norms = []
    for _ in range(R):
        c = draw_coeffs(rng, n, **coeff_kw)
        tr_idx, _ = split_indices(rng, M)
        qn_tr = make_loss(dev, n, L, c[tr_idx], [ops[j] for j in tr_idx])
        th = pnp.array(rng.uniform(0.0, 2 * np.pi, size=(L, n, 2)),
                       requires_grad=True)
        g = flat_grad(qn_tr, th)
        comp.append(g[0])
        norms.append(np.linalg.norm(g))
    comp = np.asarray(comp)
    norms = np.asarray(norms)
    mean_norm = float(np.mean(norms))
    return {
        "grad_comp_var": float(np.var(comp)),
        "grad_comp_absmean": float(np.mean(np.abs(comp))),
        "norm_mean": mean_norm,
        "norm_cv": float(np.std(norms) / mean_norm) if mean_norm > 0 else np.nan,
        "R": R,
    }


# --------------------------------------------------------------------------- #
# (B) Meta-learning diagnostic experiment
# --------------------------------------------------------------------------- #
def meta_eval(dev, n, L, ops, theta0, rng, n_eval, K, inner_lr, coeff_kw):
    """Per-task diagnostics at theta_0 and after adaptation. Returns a DataFrame."""
    M = len(ops)
    rows = []
    for t in range(n_eval):
        c = draw_coeffs(rng, n, **coeff_kw)
        tr_idx, val_idx = split_indices(rng, M)

        qn_tr = make_loss(dev, n, L, c[tr_idx], [ops[j] for j in tr_idx])
        qn_val = make_loss(dev, n, L, c[val_idx], [ops[j] for j in val_idx])

        # gradients at the meta-initialization -> non-circular alignment
        g_tr = flat_grad(qn_tr, theta0)
        g_val = flat_grad(qn_val, theta0)
        ntr = float(np.linalg.norm(g_tr))
        nval = float(np.linalg.norm(g_val))
        dot = float(np.dot(g_tr, g_val))
        A = dot / (ntr * nval) if (ntr > 0 and nval > 0) else np.nan
        IP = dot  # identity predictor = cos * ||g_tr|| * ||g_val||

        # losses at theta_0
        Ltr0 = float(qn_tr(theta0))
        Lval0 = float(qn_val(theta0))

        # adapt on support, evaluate
        phi = adapt(dev, n, L, theta0, ops, tr_idx, c, K, inner_lr)
        LtrK = float(qn_tr(phi))
        LvalK = float(qn_val(phi))

        I_tr = Ltr0 - LtrK          # support improvement
        I_val = Lval0 - LvalK       # validation improvement (local)
        G = LvalK - LtrK            # held-out observable gap (primary target)
        drift = float(np.linalg.norm(
            np.asarray(phi, dtype=float).reshape(-1)
            - np.asarray(theta0, dtype=float).reshape(-1)))

        rows.append(dict(task=t, norm_gtr=ntr, norm_gval=nval, align=A, IP=IP,
                         Ltr0=Ltr0, LtrK=LtrK, Lval0=Lval0, LvalK=LvalK,
                         I_tr=I_tr, I_val=I_val, G=G, drift=drift))
    return pd.DataFrame(rows)


def safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return np.nan, np.nan
    r, p = spearmanr(x[m], y[m])
    return float(r), float(p)


def summarize(df, n, L, seed):
    cv = lambda a: float(np.std(a) / np.mean(a)) if np.mean(a) > 0 else np.nan
    rA_Ival, pA_Ival = safe_spearman(df["align"], df["I_val"])
    rIP_Ival, _ = safe_spearman(df["IP"], df["I_val"])
    rA_G, pA_G = safe_spearman(df["align"], df["G"])
    rdrift_Itr, _ = safe_spearman(df["drift"], df["I_tr"])
    return dict(
        n=n, L=L, seed=seed, n_eval=len(df),
        cv_gtr=cv(df["norm_gtr"].values),
        cv_gval=cv(df["norm_gval"].values),
        mean_norm_gtr=float(np.mean(df["norm_gtr"])),
        mean_norm_gval=float(np.mean(df["norm_gval"])),
        rho_A_Ival=rA_Ival, p_A_Ival=pA_Ival,
        rho_IP_Ival=rIP_Ival,
        identity_gap=(rIP_Ival - rA_Ival)
        if np.isfinite(rIP_Ival) and np.isfinite(rA_Ival) else np.nan,
        rho_A_G=rA_G, p_A_G=pA_G,
        rho_drift_Itr=rdrift_Itr,
    )


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(args):
    coeff_kw = dict(coupling_lo=args.coupling_lo, coupling_hi=args.coupling_hi,
                    field_scale=args.field_scale)

    pertask_frames, summary_rows, barren_rows = [], [], []

    for n in args.n:
        dev = qml.device("default.qubit", wires=n)
        ops = build_ops(n)
        M = len(ops)
        for L in args.L:
            for seed in range(args.seeds):
                t0 = time.time()
                rng = np.random.default_rng(
                    args.base_seed + 1000 * n + 100 * L + seed)

                # (A) barren-plateau probe (independent random-init statistics)
                bp = barren_probe(dev, n, L, ops, rng, args.R, coeff_kw)
                bp.update(dict(n=n, L=L, seed=seed,
                               params=2 * n * L, M=M))
                barren_rows.append(bp)

                # (B) meta-init then diagnostic evaluation
                theta0 = reptile_meta_init(
                    dev, n, L, ops, rng, args.n_outer, args.K,
                    args.inner_lr, args.outer_lr, args.init_scale, coeff_kw)
                df = meta_eval(dev, n, L, ops, theta0, rng, args.n_eval,
                               args.K, args.inner_lr, coeff_kw)
                df.insert(0, "seed", seed)
                df.insert(0, "L", L)
                df.insert(0, "n", n)
                pertask_frames.append(df)
                summary_rows.append(summarize(df, n, L, seed))

                dt = time.time() - t0
                s = summary_rows[-1]
                print(f"[n={n} L={L} seed={seed}] "
                      f"params={2*n*L:3d}  M={M:3d}  "
                      f"BPvar={bp['grad_comp_var']:.2e}  "
                      f"|g|={s['mean_norm_gtr']:.3e}  "
                      f"CV(gtr)={s['cv_gtr']:.3f}  "
                      f"rho(A,Ival)={s['rho_A_Ival']:+.3f}  "
                      f"rho(A,G)={s['rho_A_G']:+.3f}  "
                      f"idgap={s['identity_gap']:+.3f}  "
                      f"({dt:.1f}s)", flush=True)

    pertask = pd.concat(pertask_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    barren = pd.DataFrame(barren_rows)

    pertask.to_csv(args.out_pertask, index=False)
    summary.to_csv(args.out_summary, index=False)
    barren.to_csv(args.out_barren, index=False)

    print_report(summary, barren)
    print(f"\nWrote:\n  {args.out_pertask}\n  {args.out_summary}\n  {args.out_barren}")


def print_report(summary, barren):
    print("\n" + "=" * 78)
    print("PREDICTION CHECK  (aggregated across seeds; rho(A,G) shown PER seed)")
    print("=" * 78)

    # aggregate over seeds for concentration + identity metrics
    agg = (summary
           .groupby(["n", "L"])
           .agg(cv_gtr=("cv_gtr", "mean"),
                mean_norm=("mean_norm_gtr", "mean"),
                rho_A_Ival=("rho_A_Ival", "mean"),
                identity_gap=("identity_gap", "mean"))
           .reset_index())
    bp_agg = (barren
              .groupby(["n", "L"])
              .agg(BPvar=("grad_comp_var", "mean"))
              .reset_index())
    agg = agg.merge(bp_agg, on=["n", "L"], how="left")

    print("\n[Concentration + identity dominance] "
          "expect: as n,L up -> BPvar down, mean|g| down, CV down, "
          "rho(A,Ival) high, identity_gap -> 0")
    hdr = f"{'n':>3} {'L':>3} {'BPvar':>10} {'mean|g|':>10} " \
          f"{'CV(gtr)':>8} {'rho(A,Ival)':>12} {'id_gap':>8}"
    print(hdr); print("-" * len(hdr))
    for _, r in agg.sort_values(["L", "n"]).iterrows():
        print(f"{int(r['n']):>3} {int(r['L']):>3} {r['BPvar']:>10.2e} "
              f"{r['mean_norm']:>10.3e} {r['cv_gtr']:>8.3f} "
              f"{r['rho_A_Ival']:>+12.3f} {r['identity_gap']:>+8.3f}")

    print("\n[Held-out gap is NOT predicted] "
          "expect: rho(A,G) weak and sign/magnitude UNSTABLE across seeds")
    hdr2 = f"{'n':>3} {'L':>3} " + " ".join(
        f"{'seed'+str(s):>9}" for s in sorted(summary['seed'].unique()))
    print(hdr2); print("-" * len(hdr2))
    piv = summary.pivot_table(index=["n", "L"], columns="seed",
                              values="rho_A_G")
    for (n, L), row in piv.iterrows():
        cells = " ".join(f"{row[s]:>+9.3f}" if np.isfinite(row[s]) else f"{'nan':>9}"
                         for s in piv.columns)
        print(f"{int(n):>3} {int(L):>3} {cells}")

    print("\nVerdict guide:")
    print("  * BPvar / mean|g| / CV decreasing down each L-block  => gradient concentration (BP onset).")
    print("  * rho(A,Ival) staying high while identity_gap -> 0    => identity dominance strengthens.")
    print("  * rho(A,G) small and flipping sign across seeds       => alignment does NOT track held-out gap.")
    print("  If all three hold, the paper's physical prediction is demonstrated across scale.")


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, nargs="+", default=[4, 6, 8, 10],
                   help="qubit counts to sweep (add 12 if runtime allows)")
    p.add_argument("--L", type=int, nargs="+", default=[2, 3, 5],
                   help="ansatz depths to sweep")
    p.add_argument("--seeds", type=int, default=3, help="seeds per (n,L)")
    p.add_argument("--n_outer", type=int, default=200,
                   help="Reptile outer steps for the meta-initialization")
    p.add_argument("--n_eval", type=int, default=50,
                   help="evaluation tasks per (n,L,seed)")
    p.add_argument("--R", type=int, default=200,
                   help="random-init samples for the barren-plateau probe")
    p.add_argument("--K", type=int, default=10, help="inner adaptation steps")
    p.add_argument("--inner_lr", type=float, default=0.5)
    p.add_argument("--outer_lr", type=float, default=0.3)
    p.add_argument("--init_scale", type=float, default=0.1,
                   help="std of the small random pre-Reptile initialization")
    p.add_argument("--coupling_lo", type=float, default=0.5)
    p.add_argument("--coupling_hi", type=float, default=1.5)
    p.add_argument("--field_scale", type=float, default=0.5)
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--out_pertask", default="results_pertask.csv")
    p.add_argument("--out_summary", default="results_summary.csv")
    p.add_argument("--out_barren", default="results_barren.csv")
    p.add_argument("--quick", action="store_true",
                   help="fast smoke test (small n,L, few tasks/steps)")
    return p


def apply_quick(args):
    args.n = [4, 6]
    args.L = [2, 3]
    args.seeds = 2
    args.n_outer = 30
    args.n_eval = 15
    args.R = 40
    return args


if __name__ == "__main__":
    args = build_argparser().parse_args()
    if args.quick:
        args = apply_quick(args)
    print("Configuration:")
    for k, v in sorted(vars(args).items()):
        print(f"  {k} = {v}")
    print()
    t0 = time.time()
    run(args)
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")
