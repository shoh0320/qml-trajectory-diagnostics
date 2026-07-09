import os
import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

try:
    import pennylane as qml
except ImportError as e:
    raise ImportError("PennyLane is required. Run: !pip install pennylane numpy scipy pandas matplotlib tqdm") from e

warnings.filterwarnings("ignore")
print("PennyLane version:", qml.__version__)


# ============================================================
# Hamiltonian utilities
# ============================================================

@dataclass
class PauliTerm:
    coeff: float
    pauli: str

def heisenberg_xyz_open_fields_terms(
    n_qubits: int,
    rng: np.random.Generator,
    jx_range=(0.5, 2.0),
    jy_range=(0.5, 2.0),
    jz_range=(0.5, 2.0),
    hz_range=(-1.0, 1.0),
):
    terms = []

    for i in range(n_qubits - 1):
        jx = float(rng.uniform(*jx_range))
        jy = float(rng.uniform(*jy_range))
        jz = float(rng.uniform(*jz_range))

        p = ["I"] * n_qubits
        p[i], p[i+1] = "X", "X"
        terms.append(PauliTerm(jx, "".join(p)))

        p = ["I"] * n_qubits
        p[i], p[i+1] = "Y", "Y"
        terms.append(PauliTerm(jy, "".join(p)))

        p = ["I"] * n_qubits
        p[i], p[i+1] = "Z", "Z"
        terms.append(PauliTerm(jz, "".join(p)))

    for i in range(n_qubits):
        hz = float(rng.uniform(*hz_range))
        p = ["I"] * n_qubits
        p[i] = "Z"
        terms.append(PauliTerm(hz, "".join(p)))

    return terms

def pauli_string_to_observable(pauli: str):
    obs = None
    for wire, char in enumerate(pauli):
        if char == "I":
            continue
        if char == "X":
            o = qml.PauliX(wire)
        elif char == "Y":
            o = qml.PauliY(wire)
        elif char == "Z":
            o = qml.PauliZ(wire)
        else:
            raise ValueError(f"Unknown Pauli char: {char}")
        obs = o if obs is None else obs @ o
    return obs if obs is not None else qml.Identity(0)

def split_terms_uniform(terms, rng, m_tr, m_val):
    M = len(terms)
    assert m_tr + m_val <= M, f"m_tr+m_val={m_tr+m_val} exceeds M={M}"
    idx = rng.permutation(M)
    return list(idx[:m_tr]), list(idx[m_tr:m_tr+m_val])



# ============================================================
# Quantum evaluator
# ============================================================

_qnode_cache = {}

def make_qnode(n_qubits: int, observable, key):
    cache_key = (n_qubits, key)
    if cache_key in _qnode_cache:
        return _qnode_cache[cache_key]

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(theta):
        n_layers = theta.shape[0]
        for l in range(n_layers):
            for q in range(n_qubits):
                qml.RY(theta[l, q, 0], wires=q)
                qml.RZ(theta[l, q, 1], wires=q)
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
        return qml.expval(observable)

    _qnode_cache[cache_key] = circuit
    return circuit

class QuantumEvaluator:
    def __init__(self, n_qubits, terms):
        self.n_qubits = n_qubits
        self.terms = terms
        self.M = len(terms)
        self.coeffs = np.array([t.coeff for t in terms], dtype=float)
        self.coeff_l1 = float(np.sum(np.abs(self.coeffs)))
        self.observables = [pauli_string_to_observable(t.pauli) for t in terms]
        self.qnodes = [
            make_qnode(n_qubits, obs, f"{j}_{terms[j].pauli}") 
            for j, obs in enumerate(self.observables)
        ]

    def expvals(self, theta, indices=None):
        if indices is None:
            indices = range(self.M)
        return np.array([float(self.qnodes[j](theta)) for j in indices], dtype=float)

    def subset_energy(self, theta, indices):
        vals = self.expvals(theta, indices)
        coeffs = self.coeffs[np.array(indices)]
        return float(self.M / len(indices) * np.sum(coeffs * vals))

    def full_energy(self, theta):
        vals = self.expvals(theta, range(self.M))
        return float(np.sum(self.coeffs * vals))

    def normalized_subset_loss(self, theta, indices):
        E = self.subset_energy(theta, indices)
        denom = max(2 * self.coeff_l1, 1e-12)
        z = (E + self.coeff_l1) / denom
        return float(np.clip(z, 0.0, 1.0))

    def measurement_variance_proxy(self, theta, indices, shots=1000):
        vals = self.expvals(theta, indices)
        coeffs = np.abs(self.coeffs[np.array(indices)])
        variances = np.maximum(0.0, 1.0 - vals**2)
        return float(np.mean(coeffs * np.sqrt(variances / max(shots, 1))))



# ============================================================
# Gradient and adaptation
# ============================================================

def finite_diff_grad(fn, theta, eps=1e-4):
    grad = np.zeros_like(theta, dtype=float)
    it = np.nditer(theta, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = theta[idx]
        theta[idx] = old + eps
        f_plus = fn(theta)
        theta[idx] = old - eps
        f_minus = fn(theta)
        theta[idx] = old
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return grad

def adapt_task(
    mu,
    evaluator,
    train_idx,
    condition,
    inner_steps,
    inner_lr,
    beta_drift,
    beta_var,
    sigma_sq,
    shots_for_variance,
):
    theta = np.array(mu, dtype=float).copy()

    if condition == "reptile":
        bd, bv = 0.0, 0.0
    elif condition == "drift_only":
        bd, bv = beta_drift, 0.0
    elif condition == "variance_only":
        bd, bv = 0.0, beta_var
    elif condition == "full_bqml":
        bd, bv = beta_drift, beta_var
    else:
        raise ValueError(f"Unknown condition: {condition}")

    def objective(th):
        loss = evaluator.normalized_subset_loss(th, train_idx)
        drift_pen = bd * np.sum((th - mu) ** 2) / (2.0 * sigma_sq)
        var_pen = bv * evaluator.measurement_variance_proxy(th, train_idx, shots=shots_for_variance)
        return loss + drift_pen + var_pen

    for _ in range(inner_steps):
        grad = finite_diff_grad(objective, theta)
        theta = theta - inner_lr * grad
        if not np.all(np.isfinite(theta)):
            raise FloatingPointError("Non-finite theta during adaptation.")
    return theta

def create_task(seed, n_qubits):
    return heisenberg_xyz_open_fields_terms(n_qubits, np.random.default_rng(seed))

def evaluate_task_row(mu, task_seed, split_seed, condition, config):
    terms = create_task(task_seed, config["n_qubits"])
    evaluator = QuantumEvaluator(config["n_qubits"], terms)
    train_idx, val_idx = split_terms_uniform(
        terms, np.random.default_rng(split_seed), config["m_tr"], config["m_val"]
    )

    theta_init = np.array(mu, dtype=float).copy()
    train_loss_init = evaluator.normalized_subset_loss(theta_init, train_idx)
    val_loss_init = evaluator.normalized_subset_loss(theta_init, val_idx)

    theta_adapted = adapt_task(
        mu=mu,
        evaluator=evaluator,
        train_idx=train_idx,
        condition=condition,
        inner_steps=config["inner_steps"],
        inner_lr=config["inner_lr"],
        beta_drift=config["beta_drift"],
        beta_var=config["beta_var"],
        sigma_sq=config["sigma_sq"],
        shots_for_variance=config["shots_for_variance"],
    )

    train_loss = evaluator.normalized_subset_loss(theta_adapted, train_idx)
    val_loss = evaluator.normalized_subset_loss(theta_adapted, val_idx)

    gap = val_loss - train_loss
    abs_gap = abs(gap)

    train_improvement = train_loss_init - train_loss
    val_improvement = val_loss_init - val_loss
    improvement_gap = train_improvement - val_improvement
    abs_improvement_gap = abs(improvement_gap)

    drift = float(np.linalg.norm(theta_adapted - mu))
    v_tr = evaluator.measurement_variance_proxy(theta_adapted, train_idx, config["shots_for_variance"])
    v_val = evaluator.measurement_variance_proxy(theta_adapted, val_idx, config["shots_for_variance"])

    return {
        "task_seed": int(task_seed),
        "split_seed": int(split_seed),
        "condition": condition,
        "train_loss_init": train_loss_init,
        "val_loss_init": val_loss_init,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_improvement": train_improvement,
        "val_improvement": val_improvement,
        "improvement_gap": improvement_gap,
        "abs_improvement_gap": abs_improvement_gap,
        "gap": gap,
        "abs_gap": abs_gap,
        "drift": drift,
        "v_tr": v_tr,
        "v_val": v_val,
    }

def meta_train_condition(condition, seed, config):
    rng = np.random.default_rng(seed)
    mu = rng.normal(
        0, config["init_scale"], 
        size=(config["n_layers"], config["n_qubits"], 2)
    )

    task_seeds = rng.integers(10_000, 999_999, size=config["n_train_tasks"])
    split_seeds = rng.integers(10_000, 999_999, size=config["n_train_tasks"])

    for t_seed, s_seed in zip(task_seeds, split_seeds):
        terms = create_task(int(t_seed), config["n_qubits"])
        evaluator = QuantumEvaluator(config["n_qubits"], terms)
        train_idx, _ = split_terms_uniform(
            terms, np.random.default_rng(int(s_seed)), 
            config["m_tr"], config["m_val"]
        )

        theta_adapted = adapt_task(
            mu=mu,
            evaluator=evaluator,
            train_idx=train_idx,
            condition=condition,
            inner_steps=config["inner_steps"],
            inner_lr=config["inner_lr"],
            beta_drift=config["beta_drift"],
            beta_var=config["beta_var"],
            sigma_sq=config["sigma_sq"],
            shots_for_variance=config["shots_for_variance"],
        )
        mu = mu + config["outer_lr"] * (theta_adapted - mu)

    return mu

def meta_test_condition(mu, condition, seed, config):
    rng = np.random.default_rng(seed + 123456)
    task_seeds = rng.integers(10_000, 999_999, size=config["n_test_tasks"])
    split_seeds = rng.integers(10_000, 999_999, size=config["n_test_tasks"])

    rows = []
    for t_seed, s_seed in zip(task_seeds, split_seeds):
        row = evaluate_task_row(mu, int(t_seed), int(s_seed), condition, config)
        row["meta_seed"] = int(seed)
        rows.append(row)
    return rows

def run_condition_seed(condition, seed, config):
    start = time.time()
    mu = meta_train_condition(condition, seed, config)
    rows = meta_test_condition(mu, condition, seed, config)
    elapsed = time.time() - start
    for r in rows:
        r["elapsed_sec_condition_seed"] = elapsed
    return rows
