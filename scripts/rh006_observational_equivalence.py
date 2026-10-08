#!/usr/bin/env python3
import json
import math
import random
from dataclasses import dataclass

T = 0.5
VY = 1.5
VF = 2.0
COV_TOL = 1e-12
S_MIN = T * T / VY

@dataclass(frozen=True)
class DGP:
    name: str
    s: float
    k: float
    b: float
    vare: float
    vv: float

    @property
    def xi(self):
        return self.k

    @property
    def K(self):
        return self.k / self.s

    @property
    def C(self):
        denom = self.vare * self.s
        return self.k / math.sqrt(denom) if denom > 0 else (1.0 if self.k > 0 else -1.0)

    def observable_covariance(self):
        var_f = self.s + self.vv
        cov_fy = self.b * self.s + self.k
        var_y = self.b * self.b * self.s + self.vare + 2.0 * self.b * self.k
        return (var_f, cov_fy, var_y)

    def latent_psd_min_eigen(self):
        a, d, b = self.s, self.vare, self.k
        trace = a + d
        disc = max(0.0, (a - d) ** 2 + 4.0 * b * b)
        return 0.5 * (trace - math.sqrt(disc))

def dgp_for_c(c: float) -> DGP:
    if abs(c) < 1.0:
        s = 1.0
        tau2 = VY - T * T / s
        k = c / math.sqrt(1.0 - c * c) * math.sqrt(s * tau2)
        b = (T - k) / s
        vare = k * k / s + tau2
        return DGP(f"c={c:+.6f}", s, k, b, vare, VF - s)

    s = S_MIN
    k = 1.0 if c > 0 else -1.0
    b = (T - k) / s
    vare = k * k / s
    return DGP(f"c={c:+.6f}", s, k, b, vare, VF - s)

def simulate(dgp: DGP, n: int = 100_000, seed: int = 20261008):
    rng = random.Random(seed)
    sq = math.sqrt(dgp.s)
    rem = max(0.0, dgp.vare - dgp.k * dgp.k / dgp.s)
    se = math.sqrt(rem)
    sv = math.sqrt(max(0.0, dgp.vv))
    vals_f, vals_y = [], []

    for _ in range(n):
        z1, z2, z3 = (rng.gauss(0.0, 1.0) for _ in range(3))
        q = sq * z1
        e = (dgp.k / sq) * z1 + se * z2
        v = sv * z3
        vals_f.append(q + v)
        vals_y.append(dgp.b * q + e)

    mf = sum(vals_f) / n
    my = sum(vals_y) / n
    vf = sum((x - mf) ** 2 for x in vals_f) / n
    vy = sum((y - my) ** 2 for y in vals_y) / n
    cov = sum((x - mf) * (y - my) for x, y in zip(vals_f, vals_y)) / n
    return (vf, cov, vy)

def check_close(actual, expected, tol=COV_TOL):
    return max(abs(a - b) for a, b in zip(actual, expected)) <= tol

observable_target = (VF, T, VY)
model_a = DGP("A", 1.0, 0.4, 0.1, 1.41, 1.0)
model_b = DGP("B", 0.5, -0.4, 1.8, 1.32, 1.5)
models = [model_a, model_b]

assert all(check_close(d.observable_covariance(), observable_target) for d in models)
assert all(d.latent_psd_min_eigen() >= -1e-12 for d in models)
assert model_a.K != model_b.K
assert abs(model_a.C - 0.4 / math.sqrt(1.41)) < 1e-12
assert abs(model_b.C + 0.4 / math.sqrt(1.32 * 0.5)) < 1e-12

simulated = {d.name: simulate(d) for d in models}

cs = [-0.999, -0.99, -0.85, -0.5, 0.0, 0.5, 0.85, 0.99, 0.999]
continuum = []
for c in cs:
    d = dgp_for_c(c)
    assert d.latent_psd_min_eigen() >= -1e-12
    assert check_close(d.observable_covariance(), observable_target)
    assert abs(d.C - c) < 1e-12
    continuum.append({
        "c": c,
        "K": d.K,
        "Xi": d.xi,
        "observable_covariance": d.observable_covariance(),
    })

for c in (-1.0, 1.0):
    d = dgp_for_c(c)
    assert check_close(d.observable_covariance(), observable_target)
    assert abs(abs(d.C) - 1.0) < 1e-12
    assert d.latent_psd_min_eigen() >= -1e-12

for c in (0.99, 0.999, 0.9999, 0.99999):
    assert check_close(dgp_for_c(c).observable_covariance(), observable_target)

def branch_discriminant(c):
    return 0.25 - c * c

assert branch_discriminant(0.0) > 0
assert abs(branch_discriminant(0.5)) < 1e-15
assert branch_discriminant(0.99) < 0

# Identification rescue:
# q = Z + eta, Var(Z)=Var(eta)=0.5; e = 0.8 eta + eps, with Z independent of e.
z_var, eta_var = 0.5, 0.5
s = z_var + eta_var
k_true, b_true = 0.4, 0.1
var_eps = 1.41 - (0.8 ** 2) * eta_var
assert var_eps > 0
cov_z_f = z_var
cov_z_y = b_true * z_var
b_iv = cov_z_y / cov_z_f
k_recovered = T - b_iv * s
var_e_recovered = VY - b_iv * b_iv * s - 2.0 * b_iv * k_recovered
c_recovered = k_recovered / math.sqrt(var_e_recovered * s)
assert abs(b_iv - b_true) < 1e-12
assert abs(k_recovered - k_true) < 1e-12
assert abs(c_recovered - model_a.C) < 1e-12

result = {
    "schema": "rh006-observational-equivalence-identification/v1",
    "status": "research-diagnostic-only",
    "selection": "stop-assumption-failure",
    "formal_inference": False,
    "assumption_family": "F=q+v; Y=bq+e; jointly Gaussian; v independent of (q,e)",
    "observable_target": {"var_F": VF, "cov_FY": T, "var_Y": VY},
    "witness_models": [
        {
            "name": d.name,
            "var_q": d.s,
            "cov_e_q": d.k,
            "K": d.K,
            "C": d.C,
            "b": d.b,
            "var_e": d.vare,
            "var_v": d.vv,
            "observable_covariance": d.observable_covariance(),
            "latent_psd_min_eigen": d.latent_psd_min_eigen(),
            "simulated_observable_covariance": simulated[d.name],
        }
        for d in models
    ],
    "identified_sets_under_minimal_model": {
        "Var(F)": "{2}",
        "Cov(F,Y)": "{0.5}",
        "Var(Y)": "{1.5}",
        "Var(q)": "[1/6, 2] subject to chosen nondegeneracy convention",
        "Var(v)": "[0, 11/6] subject to chosen nondegeneracy convention",
        "Cov(e,q)": "R",
        "K": "R",
        "C": "[-1, 1]",
        "b": "R",
        "branch_discriminant_0.25_minus_C2": "[-0.75, 0.25]",
    },
    "classification": {
        "Var(F)": "O",
        "Cov(F,Y)": "O",
        "Var(Y)": "O",
        "Var(q)": "PI",
        "Var(v)": "PI",
        "Cov(e,q)": "NI",
        "K": "NI",
        "C": "NI",
        "b": "NI",
        "Var(e)": "NI",
        "branch_discriminant": "NI",
        "branch_existence": "NI",
    },
    "continuum_witness": continuum,
    "branch_examples": {
        "C=0": branch_discriminant(0.0),
        "C=0.5": branch_discriminant(0.5),
        "C=0.99": branch_discriminant(0.99),
    },
    "identification_rescue": {
        "structure": [
            "two calibrated independent indicators of q (unit loading, independent measurement errors)",
            "one valid excluded instrument Z with Cov(Z,e)=0 and Cov(Z,q)!=0",
            "joint Gaussian/finite-second-moment linear model maintained for the target",
        ],
        "recovered_s": s,
        "recovered_b": b_iv,
        "recovered_k": k_recovered,
        "recovered_C": c_recovered,
        "status": "point-identified-under-explicit-structure",
    },
    "nonclaims": [
        "finite-sample simulation is calibration only, not a proof of equal laws",
        "does not validate the RH-006 estimator",
        "does not establish a valid instrument exists in the present experiment",
        "does not enable formal inference or bootstrap selection",
    ],
}

print(json.dumps(result, indent=2, sort_keys=True))
