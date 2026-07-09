# Gradient concentration constrains trajectory diagnostics in variational quantum meta-learning

Code, data, and manuscript for the paper

> **Gradient concentration constrains trajectory diagnostics in variational quantum meta-learning**
> Sang Ho Oh (Department of Computer Engineering and Artificial Intelligence, Pukyong National University, Busan, Korea)

This repository contains everything needed to reproduce the figures, tables, and statistical
analyses in the main text and Supplemental Material.

---

## What is in this repository

```
code/          available upon reasonable request
data/          authentic per-task result tables (CSV) used for every figure and table
supplements/  compiled manuscript, Supplemental Material, LaTeX sources, and figures
```


### `data/`

All CSVs are authentic outputs of the scripts above — no values were hand-edited.

| File | Used for |
|---|---|
| `results_pertask.csv` | Master per-task table for the scaling sweep (1800 rows: 4 sizes × 3 depths × 3 seeds × 50 tasks). Source for Fig. 1 and most main-text correlations. |
| `results_summary.csv` | Per-configuration summary statistics. |
| `results_barren.csv` | Barren-plateau probe: gradient-component variance and mean norm vs (n, L). |
| `results_prefactor_CV.csv` | Coefficient of variation of ‖g_tr‖, ‖g_val‖, and the prefactor q (SM Table SI). |
| `results_A_vs_heldout_loss.csv` | Alignment vs held-out loss, raw and controlled (SM Table SIII). |
| `results_item8_baseline_corrected_gap.csv` | Baseline-corrected gap ΔG analysis. |
| `results_item8_transfer_signal.csv` | Near-zero-mean-transfer statistics (SM Table SIV). |
| `results_item5_stratified.csv` | Term-type-stratified split (SM Table SV). |
| `results_item6_families.csv` | Additional Hamiltonian families and topologies (SM Table SVI). |
| `results_item7_finiteshot.csv` | Finite-shot variance proxy check. |
| `results_item13_randominit.csv` | Random-initialization baseline (per-task). |
| `results_v17_pertask.csv`, `results_v17_tableI.csv`, `results_v17_tableII.csv` | Corollary run per-task table and the two summary tables. |
| `results_positive_control_*.csv` | Synthetic positive control (channels-only vs genuine-channel per-task tables, summary, and injected-strength sensitivity; SM Sec. S6). |

### `supplements/`

`manuscript.pdf` / `manuscript.tex` (main text), `supplement.pdf` / `supplement.tex`
(Supplemental Material), and `figs/` (all figures as PDF).

---

## Reproducing the results

### Environment

Python 3.11+ with the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

The experiments were run with PennyLane 0.45, NumPy 2.4, SciPy 1.17, pandas 3.0, and
Matplotlib 3.10. All simulations are noise-free state-vector (`default.qubit`) with
backpropagation gradients; randomness is fixed by explicit per-run seeds.

### Regenerate the data

```bash
# main scaling sweep (writes results_pertask.csv, results_summary.csv, results_barren.csv)
python code/scaling_sweep.py

# corollary run (writes the v17 tables)
python code/v17_fast.py

# robustness / control analyses (reads the CSVs in data/)
python code/reviewer_analysis.py
```

### Regenerate the figures

```bash
python code/make_scaling_fig.py   # Fig. 1
python code/make_pipeline.py      # Fig. 4
```

Figures 2 and 3 are produced from the corollary run and its synthetic control; see
`v17_corollary.ipynb`.

---

## Notes on scope

The primary task family is a **near-zero-mean observable-transfer regime by construction**: support
adaptation improves the support loss but the mean validation improvement is ≈ 0 at every system
size. This is intentional and is discussed in the paper; the contribution is a physical explanation
of why trajectory diagnostics can still show apparent correlations with the held-out gap in this
regime. See the manuscript for the full scope statement.

## Citation

If you use this code or data, please cite the paper (citation details to be added upon publication).

## License

Code is released under the MIT License (`LICENSE`). The manuscript and figures are subject to the
copyright terms of the published article.
