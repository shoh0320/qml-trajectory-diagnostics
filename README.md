# Auditing Trajectory-Based Transfer Diagnostics in Variational Quantum Learning

Code, per-task data, and Supplemental Material for the manuscript
*"Auditing Trajectory-Based Transfer Diagnostics in Variational Quantum Learning"* (submitted to *Electronics*, Special Issue *Advanced Computer Science and Intelligent Systems Innovations*).

## What this repository contains

This is a **metric-audit study**: it examines what trajectory-derived scalars (parameter drift, an observable-variance proxy, and support–query gradient alignment) actually measure during few-shot adaptation of variational quantum circuits on Heisenberg/XYZ task families (4–10 qubits, depths 2/3/5). The analysis separates first-order optimization coupling, target-construction overlap, and configuration-level instability, and states explicitly which claims are and are not supported without an independent support–validation–test split.

- `code/` — simulation and analysis scripts (PennyLane, statevector). `scaling_sweep.py` reproduces the main sweep; `run_v17_cond.py`/`v17_core.py` reproduce the corollary configuration; `make_pipeline.py` regenerates the audit-workflow figure; `analysis.py` reproduces the reported statistics from the CSVs.
- `data/` — per-task CSVs (`results_pertask.csv`, `results_barren.csv`, `results_v17_*.csv`). Every statistic reported in the manuscript, including the configuration-level attribution and target-decomposition checks, can be recomputed from these files.
- `supplements/` — Supplemental Material source and PDF (Tables SI–SVII).

## Reproducing reported numbers

All correlations use Spearman rank statistics; partial correlations are rank-transformed linear residualizations as defined in the manuscript. The ratio diagnostic in Fig. 2(b) uses the stabilizer value stated in the text. Uncertainty is estimated by hierarchical bootstrap: (L, seed) configurations are resampled first, then tasks within each resampled configuration.

## Scope note

The reported adjusted nulls are pooled-by-size statements in a same-query-set, empirically near-zero-mean-transfer, noise-free setting; configuration-level heterogeneity and the algebraic target-decomposition reading are documented in the manuscript and are part of the audit itself.
