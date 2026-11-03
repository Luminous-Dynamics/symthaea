#!/usr/bin/env python3
"""HIST-CAUSAL-001 v0.11 clustered/persistent-confounder stress test.

Treatment is randomized at the cluster/family level. Repeated observations
within a family share latent cluster and participant effects. The invalid
participant-level Welch test is compared with cluster-mean Welch and a
Monte-Carlo cluster randomization test. Synthetic design check only.
"""
from __future__ import annotations
import json
import numpy as np
from scipy.stats import t as student_t

DEFAULT = {
    "replicates": 1000,
    "permutations": 999,
    "clusters": 64,
    "treated_clusters": 32,
    "participants_per_cluster": 16,
    "repeated_probes_per_participant": 3,
    "cluster_sd": 1.0,
    "participant_sd": 0.5,
    "measurement_sd": 1.0,
    "treatment_effect": 0.0,
    "seed": 20261103,
}


def _welch_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = a.size, b.size
    ma, mb = float(a.mean()), float(b.mean())
    va, vb = float(a.var(ddof=1)), float(b.var(ddof=1))
    se2 = va/na + vb/nb
    if se2 <= 0:
        return 1.0
    stat = (ma-mb) / np.sqrt(se2)
    df = se2**2 / ((va/na)**2/(na-1) + (vb/nb)**2/(nb-1))
    return float(2*student_t.sf(abs(stat), df))


def _cluster_randomization_pvalue(cluster_means: np.ndarray, treated: np.ndarray,
                                  rng: np.random.Generator, permutations: int) -> float:
    n = cluster_means.size
    n_treated = int(treated.sum())
    observed = float(cluster_means[treated].mean() - cluster_means[~treated].mean())
    keys = rng.random((permutations, n))
    order = np.argpartition(keys, n_treated-1, axis=1)[:, :n_treated]
    mask = np.zeros((permutations,n), dtype=np.int8)
    np.put_along_axis(mask, order, 1, axis=1)
    sums = mask @ cluster_means
    perm_diff = sums/n_treated - (cluster_means.sum()-sums)/(n-n_treated)
    return float((1 + np.count_nonzero(np.abs(perm_diff) >= abs(observed))) / (permutations+1))


def run(config: dict | None = None) -> dict:
    cfg = {**DEFAULT, **(config or {})}
    if cfg["clusters"] % 2 or cfg["treated_clusters"] != cfg["clusters"]//2:
        raise ValueError("this reference requires balanced 1:1 cluster randomization")
    if min(cfg["replicates"], cfg["permutations"], cfg["participants_per_cluster"],
           cfg["repeated_probes_per_participant"]) <= 0:
        raise ValueError("replicates, permutations and within-cluster sizes must be positive")
    rng = np.random.default_rng(cfg["seed"])
    c = cfg["clusters"]
    treated_n = cfg["treated_clusters"]
    people = cfg["participants_per_cluster"]
    probes = cfg["repeated_probes_per_participant"]
    total = cfg["replicates"]
    p_naive = np.empty(total)
    p_cluster_t = np.empty(total)
    p_cluster_ri = np.empty(total)
    effects = np.empty(total)

    for rep in range(total):
        treatment = np.zeros(c, dtype=bool)
        treatment[rng.permutation(c)[:treated_n]] = True
        cluster_u = rng.normal(0, cfg["cluster_sd"], size=c)
        person_u = rng.normal(0, cfg["participant_sd"], size=(c,people))
        noise = rng.normal(0, cfg["measurement_sd"], size=(c,people,probes))
        outcomes = (cluster_u[:,None,None] + person_u[:,:,None] + noise
                    + cfg["treatment_effect"]*treatment[:,None,None])
        flat_y = outcomes.reshape(c,-1)
        treated_y = outcomes[treatment].reshape(-1)
        control_y = outcomes[~treatment].reshape(-1)
        p_naive[rep] = _welch_pvalue(treated_y, control_y)
        means = flat_y.mean(axis=1)
        effects[rep] = means[treatment].mean() - means[~treatment].mean()
        p_cluster_t[rep] = _welch_pvalue(means[treatment], means[~treatment])
        p_cluster_ri[rep] = _cluster_randomization_pvalue(means,treatment,rng,cfg["permutations"])

    def summary(p):
        return {"rejection_rate_at_0_05":float(np.mean(p<=0.05)),
                "mean_p_value":float(p.mean()),
                "p_value_quantiles":{str(q):float(np.quantile(p,q)) for q in (0.01,0.05,0.5,0.95,0.99)}}

    return {
        "benchmark":"HIST-CAUSAL-001","version":"0.11",
        "design":cfg,
        "independent_assignment_unit":"cluster/family",
        "within_cluster_dependence":"shared persistent cluster intercept and participant intercept across repeated probes",
        "null_hypothesis":"zero treatment effect",
        "methods":{
            "invalid_participant_level_welch":summary(p_naive),
            "cluster_mean_welch":summary(p_cluster_t),
            "cluster_level_randomization_inference":summary(p_cluster_ri),
        },
        "mean_observed_cluster_mean_difference":float(effects.mean()),
        "claim_ceiling":"synthetic design stress test under declared Gaussian random-intercept model; not a universal validity guarantee or empirical effect",
    }


if __name__=="__main__":
    print(json.dumps(run(),indent=2,sort_keys=True))
