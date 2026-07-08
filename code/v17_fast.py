"""Fast, faithful v17 core. Batched qnode + analytic (backprop) inner gradient.
Reproduces the original finite-diff results to ~1e-5 (finite-diff eps=1e-4 approximates
the exact gradient to O(eps^2)). Task/split seed logic is byte-identical to the original."""
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from dataclasses import dataclass


@dataclass
class PauliTerm:
    coeff: float
    pauli: str


def heisenberg_xyz_open_fields_terms(n_qubits, rng, jx_range=(0.5, 2.0),
                                     jy_range=(0.5, 2.0), jz_range=(0.5, 2.0),
                                     hz_range=(-1.0, 1.0)):
    terms = []
    for i in range(n_qubits - 1):
        for (a, b), rr in [(("X", "X"), jx_range), (("Y", "Y"), jy_range), (("Z", "Z"), jz_range)]:
            p = ["I"] * n_qubits; p[i], p[i + 1] = a, b
            terms.append(PauliTerm(float(rng.uniform(*rr)), "".join(p)))
    for i in range(n_qubits):
        p = ["I"] * n_qubits; p[i] = "Z"
        terms.append(PauliTerm(float(rng.uniform(*hz_range)), "".join(p)))
    return terms


def pauli_string_to_observable(pauli):
    obs = None
    for wire, char in enumerate(pauli):
        if char == "I":
            continue
        o = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}[char](wire)
        obs = o if obs is None else obs @ o
    return obs if obs is not None else qml.Identity(0)


def split_terms_uniform(terms, rng, m_tr, m_val):
    M = len(terms); assert m_tr + m_val <= M
    idx = rng.permutation(M)
    return list(idx[:m_tr]), list(idx[m_tr:m_tr + m_val])


_dev_cache = {}
_qn_cache = {}


def _batched_qnode(n_qubits, observables):
    key = ("batch", n_qubits)
    if key in _qn_cache:
        return _qn_cache[key]
    if n_qubits not in _dev_cache:
        _dev_cache[n_qubits] = qml.device("default.qubit", wires=n_qubits)
    dev = _dev_cache[n_qubits]

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(theta):
        nL = theta.shape[0]
        for l in range(nL):
            for q in range(n_qubits):
                qml.RY(theta[l, q, 0], wires=q); qml.RZ(theta[l, q, 1], wires=q)
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
        return [qml.expval(o) for o in observables]
    _qn_cache[key] = circuit
    return circuit


class FastEvaluator:
    def __init__(self, n_qubits, terms):
        self.n_qubits = n_qubits; self.terms = terms; self.M = len(terms)
        self.coeffs = np.array([t.coeff for t in terms], float)
        self.coeff_l1 = float(np.sum(np.abs(self.coeffs)))
        self.observables = [pauli_string_to_observable(t.pauli) for t in terms]
        self._qn = _batched_qnode(n_qubits, self.observables)

    def _all(self, theta):
        return self._qn(theta)

    def expvals(self, theta, indices=None):
        allv = np.array([float(x) for x in self._qn(theta)], float)
        return allv if indices is None else allv[np.array(list(indices))]

    def subset_energy(self, theta, indices):
        vals = self.expvals(theta, indices)
        return float(self.M / len(indices) * np.sum(self.coeffs[np.array(indices)] * vals))

    def normalized_subset_loss(self, theta, indices):
        E = self.subset_energy(theta, indices)
        z = (E + self.coeff_l1) / max(2 * self.coeff_l1, 1e-12)
        return float(np.clip(z, 0.0, 1.0))

    def measurement_variance_proxy(self, theta, indices, shots=1000):
        vals = self.expvals(theta, indices)
        coeffs = np.abs(self.coeffs[np.array(indices)])
        variances = np.maximum(0.0, 1.0 - vals ** 2)
        return float(np.mean(coeffs * np.sqrt(variances / max(shots, 1))))

    def loss_diff(self, indices, allv):
        idx = np.array(indices)
        E = self.M / len(indices) * pnp.sum(pnp.stack([self.coeffs[j] * allv[j] for j in idx]))
        z = (E + self.coeff_l1) / max(2 * self.coeff_l1, 1e-12)
        return pnp.clip(z, 0.0, 1.0)

    def vtr_diff(self, indices, allv, shots=1000):
        idx = np.array(indices)
        t = [pnp.abs(self.coeffs[j]) * pnp.sqrt(pnp.maximum(0.0, 1.0 - allv[j] ** 2) / max(shots, 1))
             for j in idx]
        return pnp.mean(pnp.stack(t))


def adapt_task_fast(mu, evaluator, train_idx, condition, inner_steps, inner_lr,
                    beta_drift, beta_var, sigma_sq, shots_for_variance):
    if condition == "reptile":
        bd, bv = 0.0, 0.0
    elif condition == "drift_only":
        bd, bv = beta_drift, 0.0
    elif condition == "variance_only":
        bd, bv = 0.0, beta_var
    elif condition == "full_bqml":
        bd, bv = beta_drift, beta_var
    else:
        raise ValueError(condition)
    mu_p = pnp.array(np.array(mu, float), requires_grad=False)
    theta = pnp.array(np.array(mu, float), requires_grad=True)

    def objective(th):
        allv = evaluator._all(th)
        loss = evaluator.loss_diff(train_idx, allv)
        drift_pen = bd * pnp.sum((th - mu_p) ** 2) / (2.0 * sigma_sq)
        var_pen = bv * evaluator.vtr_diff(train_idx, allv, shots_for_variance) if bv > 0 else 0.0
        return loss + drift_pen + var_pen

    for _ in range(inner_steps):
        g = qml.grad(objective)(theta)
        theta = pnp.array(theta - inner_lr * g, requires_grad=True)
    return np.array(theta, dtype=float)


def create_task(seed, n_qubits):
    return heisenberg_xyz_open_fields_terms(n_qubits, np.random.default_rng(seed))


def evaluate_task_row(mu, task_seed, split_seed, condition, config):
    terms = create_task(task_seed, config["n_qubits"])
    ev = FastEvaluator(config["n_qubits"], terms)
    train_idx, val_idx = split_terms_uniform(terms, np.random.default_rng(split_seed),
                                             config["m_tr"], config["m_val"])
    theta_init = np.array(mu, float)
    tr0 = ev.normalized_subset_loss(theta_init, train_idx)
    vl0 = ev.normalized_subset_loss(theta_init, val_idx)
    theta = adapt_task_fast(mu, ev, train_idx, condition, config["inner_steps"],
                            config["inner_lr"], config["beta_drift"], config["beta_var"],
                            config["sigma_sq"], config["shots_for_variance"])
    trL = ev.normalized_subset_loss(theta, train_idx)
    vlL = ev.normalized_subset_loss(theta, val_idx)
    gap = vlL - trL
    tr_imp = tr0 - trL; vl_imp = vl0 - vlL
    return {
        "task_seed": int(task_seed), "split_seed": int(split_seed), "condition": condition,
        "train_loss_init": tr0, "val_loss_init": vl0, "train_loss": trL, "val_loss": vlL,
        "train_improvement": tr_imp, "val_improvement": vl_imp,
        "improvement_gap": tr_imp - vl_imp, "abs_improvement_gap": abs(tr_imp - vl_imp),
        "gap": gap, "abs_gap": abs(gap),
        "drift": float(np.linalg.norm(theta - np.array(mu, float))),
        "v_tr": ev.measurement_variance_proxy(theta, train_idx, config["shots_for_variance"]),
        "v_val": ev.measurement_variance_proxy(theta, val_idx, config["shots_for_variance"]),
    }


def meta_train_condition(condition, seed, config):
    rng = np.random.default_rng(seed)
    mu = rng.normal(0, config["init_scale"], size=(config["n_layers"], config["n_qubits"], 2))
    task_seeds = rng.integers(10_000, 999_999, size=config["n_train_tasks"])
    split_seeds = rng.integers(10_000, 999_999, size=config["n_train_tasks"])
    for t_seed, s_seed in zip(task_seeds, split_seeds):
        terms = create_task(int(t_seed), config["n_qubits"])
        ev = FastEvaluator(config["n_qubits"], terms)
        train_idx, _ = split_terms_uniform(terms, np.random.default_rng(int(s_seed)),
                                           config["m_tr"], config["m_val"])
        theta = adapt_task_fast(mu, ev, train_idx, condition, config["inner_steps"],
                                config["inner_lr"], config["beta_drift"], config["beta_var"],
                                config["sigma_sq"], config["shots_for_variance"])
        mu = mu + config["outer_lr"] * (theta - mu)
    return mu


def meta_test_condition(mu, condition, seed, config):
    rng = np.random.default_rng(seed + 123456)
    task_seeds = rng.integers(10_000, 999_999, size=config["n_test_tasks"])
    split_seeds = rng.integers(10_000, 999_999, size=config["n_test_tasks"])
    rows = []
    for t_seed, s_seed in zip(task_seeds, split_seeds):
        r = evaluate_task_row(mu, int(t_seed), int(s_seed), condition, config)
        r["meta_seed"] = int(seed); rows.append(r)
    return rows


def run_condition_seed(condition, seed, config):
    mu = meta_train_condition(condition, seed, config)
    return meta_test_condition(mu, condition, seed, config)


V17_CONDITIONS = {
    "reptile":          {"beta_drift": 0.0, "beta_var": 0.0},
    "drift_light":      {"beta_drift": 0.2, "beta_var": 0.0},
    "variance_light":   {"beta_drift": 0.0, "beta_var": 5.0},
    "full_bqml_light":  {"beta_drift": 0.2, "beta_var": 5.0},
    "full_bqml_strong": {"beta_drift": 1.0, "beta_var": 10.0},
}

V17_CONFIG = dict(n_qubits=6, n_layers=3, n_train_tasks=20, n_test_tasks=50,
                  seeds=[1, 2, 3], inner_steps=10, inner_lr=0.2, outer_lr=0.2,
                  init_scale=0.5, sigma_sq=1.0, m_tr=14, m_val=7, shots_for_variance=1000)
