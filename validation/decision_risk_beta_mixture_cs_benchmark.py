#!/usr/bin/env python3
"""Matched sequential audit benchmark with beta-mixture e-process confidence sequences.

The beta-binomial mixture likelihood ratio is a nonnegative martingale under each
candidate Bernoulli mean. Inverting it with Ville's inequality yields a time-uniform
confidence sequence for each predeclared risk cell. Familywise error is allocated
across cells. Synthetic only; no claim about arbitrary bounded non-Bernoulli audits.
"""
from __future__ import annotations
import hashlib, json, math, platform, time
from pathlib import Path
import numpy as np

SCHEMA = "symthaea-decision-risk-beta-mixture-cs-v1"
SEED = 20261029
DELTA = 0.10
RADIUS_BITS = 0.06664248830771191
REGRET_BUDGET_BITS = 0.20
EPSILON_ALLOWANCE_BITS = REGRET_BUDGET_BITS / 2.0 - RADIUS_BITS
SAMPLE_SIZES = (4096, 8192, 16384, 32768, 65536)
N_REPEATS = 1000
BETA_A = 0.5
BETA_B = 0.5
TOL = 1e-12


def empirical_bernstein_radius(variance: np.ndarray, n: int, log_term: float, span: float = 1.0) -> np.ndarray:
    if n < 2 or np.any(~np.isfinite(variance)) or np.any(variance < 0) or log_term <= 0 or span <= 0:
        raise ValueError("invalid empirical-Bernstein inputs")
    return np.sqrt(2.0 * variance * log_term / n) + 7.0 * span * log_term / (3.0 * (n - 1))


def fixed_time_radius(variance: np.ndarray, n: int, n_cells: int, delta: float = DELTA) -> np.ndarray:
    if not (0 < delta < 1) or n_cells < 1:
        raise ValueError("invalid familywise delta or cell count")
    return empirical_bernstein_radius(variance, n, math.log(4.0 * n_cells / delta))


def finite_look_radius(variance: np.ndarray, n: int, n_cells: int, n_looks: int, delta: float = DELTA) -> np.ndarray:
    if n_looks < 1:
        raise ValueError("number of predeclared looks must be positive")
    return fixed_time_radius(variance, n, n_cells, delta / n_looks)


def anytime_eb_radius(variance: np.ndarray, n: int, n_cells: int, delta: float = DELTA) -> np.ndarray:
    if not (0 < delta < 1) or n_cells < 1:
        raise ValueError("invalid familywise delta or cell count")
    delta_n = delta * 6.0 / (math.pi ** 2 * n ** 2)
    return empirical_bernstein_radius(variance, n, math.log(4.0 * n_cells / delta_n))


def beta_mixture_log_evalue(mu: float, successes: int, n: int, a: float = BETA_A, b: float = BETA_B) -> float:
    """Log of Beta(a,b)-mixture likelihood ratio against Bernoulli(mu)."""
    if not 0.0 <= mu <= 1.0 or not 0 <= successes <= n or n < 1 or a <= 0 or b <= 0:
        raise ValueError("invalid beta-mixture e-value inputs")
    failures = n - successes
    log_marginal = (math.lgamma(successes + a) + math.lgamma(failures + b)
                    - math.lgamma(n + a + b)
                    - math.lgamma(a) - math.lgamma(b) + math.lgamma(a + b))
    if mu == 0.0:
        return log_marginal if successes == 0 else math.inf
    if mu == 1.0:
        return log_marginal if failures == 0 else math.inf
    return log_marginal - successes * math.log(mu) - failures * math.log1p(-mu)


def beta_mixture_cs(successes: int, n: int, *, alpha_cell: float) -> tuple[float, float]:
    """Two-sided time-uniform CS by inverting the beta-mixture e-process.

    For every candidate true mean mu, E_n(mu) is a nonnegative martingale under
    iid Bernoulli(mu). Ville: P_mu(sup_n E_n(mu) >= 1/alpha_cell) <= alpha_cell.
    The confidence set {mu: log E_n(mu) < log(1/alpha_cell)} is an interval.
    """
    if n < 1 or not 0 <= successes <= n or not (0 < alpha_cell < 1):
        raise ValueError("invalid confidence-sequence inputs")
    threshold = math.log(1.0 / alpha_cell)
    phat = successes / n
    if successes == 0:
        lower = 0.0
    else:
        outside, inside = 0.0, phat
        for _ in range(64):
            mid = (outside + inside) / 2.0
            if beta_mixture_log_evalue(mid, successes, n) >= threshold:
                outside = mid
            else:
                inside = mid
        lower = inside
    if successes == n:
        upper = 1.0
    else:
        inside, outside = phat, 1.0
        for _ in range(64):
            mid = (inside + outside) / 2.0
            if beta_mixture_log_evalue(mid, successes, n) < threshold:
                inside = mid
            else:
                outside = mid
        upper = inside
    return max(0.0, lower), min(1.0, upper)


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    p = successes / trials
    den = 1 + z*z/trials
    center = (p + z*z/(2*trials)) / den
    half = z * math.sqrt(p*(1-p)/trials + z*z/(4*trials*trials)) / den
    return [max(0.0, center-half), min(1.0, center+half)]


def run(*, repeats: int = N_REPEATS, sample_sizes: tuple[int, ...] = SAMPLE_SIZES) -> dict:
    if repeats < 100:
        raise ValueError("use at least 100 repeats for sequential false-admission estimation")
    sizes = tuple(sorted(set(int(n) for n in sample_sizes)))
    if not sizes or sizes[0] < 2:
        raise ValueError("sample sizes must be >= 2")
    truth = np.array([[0.50, 0.52], [0.48, 0.55]], dtype=float)
    allowance = EPSILON_ALLOWANCE_BITS
    models = {"correct_model": truth.copy(), "just_over_regret_budget": truth.copy(), "clearly_over_regret_budget": truth.copy()}
    models["just_over_regret_budget"][0, 0] += allowance + 1e-5
    models["clearly_over_regret_budget"][0, 0] += allowance + 0.01
    errors = {name: float(np.max(np.abs(model-truth))) for name, model in models.items()}
    methods = ("fixed_time_repeated_peek", "finite_look_alpha_spending", "anytime_eb_alpha_spending", "beta_mixture_e_process_cs")
    stats = {name: {method: {"false_admissions": 0, "any_admissions": 0, "sample_sum": 0,
                              "stopped": 0, "admitted_by_look": {str(n): 0 for n in sizes}}
                    for method in methods} for name in models}
    rng = np.random.default_rng(SEED)
    k = truth.size
    for _ in range(repeats):
        draws = rng.binomial(1, truth[None, :, :], size=(sizes[-1], *truth.shape)).astype(np.int8)
        csum = np.cumsum(draws, axis=0, dtype=np.int64)
        for name, model in models.items():
            for method in methods:
                rec = stats[name][method]
                stopped = False
                for n in sizes:
                    totals = csum[n-1]
                    means = totals / n
                    variances = np.maximum((totals - totals*totals/n) / (n-1), 0.0)
                    if method == "fixed_time_repeated_peek":
                        radii = fixed_time_radius(variances, n, k, DELTA)
                        epsilon_upper = float(np.max(np.abs(model - means) + radii))
                    elif method == "finite_look_alpha_spending":
                        radii = finite_look_radius(variances, n, k, len(sizes), DELTA)
                        epsilon_upper = float(np.max(np.abs(model - means) + radii))
                    elif method == "anytime_eb_alpha_spending":
                        radii = anytime_eb_radius(variances, n, k, DELTA)
                        epsilon_upper = float(np.max(np.abs(model - means) + radii))
                    else:
                        alpha_cell = DELTA / k
                        cell_upper = []
                        for i in range(truth.shape[0]):
                            for j in range(truth.shape[1]):
                                low, high = beta_mixture_cs(int(totals[i,j]), n, alpha_cell=alpha_cell)
                                cell_upper.append(max(abs(float(model[i,j])-low), abs(float(model[i,j])-high)))
                        epsilon_upper = max(cell_upper)
                    if epsilon_upper <= allowance + TOL:
                        rec["any_admissions"] += 1
                        rec["admitted_by_look"][str(n)] += 1
                        rec["sample_sum"] += n
                        rec["stopped"] += 1
                        if errors[name] > allowance + TOL:
                            rec["false_admissions"] += 1
                        stopped = True
                        break
                if not stopped:
                    rec["sample_sum"] += sizes[-1]
    results = {}
    for name in models:
        results[name] = {}
        for method in methods:
            rec = stats[name][method]
            f = rec["false_admissions"]
            results[name][method] = {
                "true_max_cell_error_bits_scoring_only": errors[name],
                "false_admissions": f, "false_admission_rate": f/repeats,
                "false_admission_rate_wilson95": wilson_interval(f, repeats),
                "admission_rate": rec["any_admissions"]/repeats,
                "mean_audit_observations_per_cell_until_stop_or_horizon": rec["sample_sum"]/repeats,
                "stopped_before_horizon_count": rec["stopped"],
                "admitted_by_look_rate": {n: c/repeats for n,c in rec["admitted_by_look"].items()},
            }
    return {
        "schema": SCHEMA, "status": "SYNTHETIC_BERNOULLI_MIXTURE_CS_BENCHMARK_NOT_PRODUCTION_CERTIFICATION",
        "protocol": {"seed":SEED,"repeats":repeats,"looks":list(sizes),"cells":k,"loss_range":[0.0,1.0],
                     "truth_risk_matrix":truth.tolist(),"familywise_delta":DELTA,
                     "alpha_cell_for_mixture_eprocess":DELTA/k,"mixture_prior":"Beta(1/2,1/2)",
                     "decision_radius_bits":RADIUS_BITS,"regret_budget_bits":REGRET_BUDGET_BITS,
                     "maximum_admissible_epsilon_bits":allowance,"true_max_cell_error_bits_scoring_only":errors},
        "methods": {"fixed_time_repeated_peek":"Single-look empirical Bernstein reused at multiple looks without correction; negative control.",
                    "finite_look_alpha_spending":"Predeclared L looks with delta/L per look, plus cell familywise correction.",
                    "anytime_eb_alpha_spending":"Infinite horizon summable alpha allocation across n, plus cell familywise correction.",
                    "beta_mixture_e_process_cs":"Exact Bernoulli beta-binomial mixture likelihood-ratio e-process, inverted into a time-uniform confidence sequence; familywise delta/K across the fixed cells."},
        "results":results,
        "limitations":["Synthetic Bernoulli cell observations only; not arbitrary bounded losses without a valid Bernoulli/conditional-mean reduction.",
                       "Fixed finite four-cell family; no protection from shared audit bias, support omission, identity mismatch or model changes.",
                       "The beta-mixture sequence is a principled betting/e-process comparator, not the newer Gaussian-efficient betting method.",
                       "No published package or external baseline implementation run; no SOTA claim.",
                       "Not integrated into native Rust decision/dispatch path; no Rust or remote CI pass claimed."],
        "environment":{"python":platform.python_version(),"numpy":np.__version__},
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main() -> None:
    import argparse
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True); p.add_argument('--repeats',type=int,default=N_REPEATS); a=p.parse_args()
    start=time.perf_counter(); out=run(repeats=a.repeats); raw=(json.dumps(out,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode(); a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_bytes(raw)
    print(json.dumps({"path":str(a.output),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"runtime_seconds":round(time.perf_counter()-start,3)},indent=2))

if __name__ == '__main__': main()
