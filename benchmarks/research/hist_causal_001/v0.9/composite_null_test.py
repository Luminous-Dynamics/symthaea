#!/usr/bin/env python3
"""HIST-CAUSAL-001 v0.9: finite-sample composite-null adequacy test.

Uses candidate-specific Monte Carlo lower-tail p-values for a common statistic:
maximum joint log likelihood over the declared candidate catalogue. A dataset
is flagged OOD only when every candidate model is rejected. Under known
candidate parameters and independent exchangeable units, this intersection-
union construction controls false OOD at alpha for every in-family candidate.
"""
from __future__ import annotations
from fractions import Fraction as F
from hashlib import sha256
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v0.6"))
sys.path.insert(0, str(ROOT / "v0.5"))
import ood_falsification as ood
import generator as base

DEFAULT_N = (1, 2, 5, 10, 20)
DEFAULT_CALIBRATION_REPS = 5000
DEFAULT_EVALUATION_REPS = 2000
DEFAULT_ALPHA = 0.05

def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], "big")

def _prepare(params: dict) -> dict:
    vocabularies, candidate_prob, ood_prob, candidate_log = {}, {}, {}, {}
    for probe in base.PROBES:
        dists = {m: base._dist(m, probe, params) for m in base.MODELS}
        d_ood = ood.ood_distribution(probe, params)
        outcomes = sorted(set(d_ood).union(*(set(d) for d in dists.values())))
        vocabularies[probe] = outcomes
        pmat = np.asarray([[float(d.get(x, F(0))) for x in outcomes] for d in dists.values()], dtype=np.float64)
        po = np.asarray([float(d_ood.get(x, F(0))) for x in outcomes], dtype=np.float64)
        if not np.allclose(pmat.sum(axis=1), 1.0, atol=1e-14) or not np.isclose(po.sum(), 1.0, atol=1e-14):
            raise ArithmeticError(f"probability mass invalid for probe {probe}")
        with np.errstate(divide="ignore"):
            logmat = np.log(pmat)
        candidate_prob[probe], ood_prob[probe], candidate_log[probe] = pmat, po, logmat
    return {"vocabularies":vocabularies,"candidate_prob":candidate_prob,"ood_prob":ood_prob,"candidate_log":candidate_log}

def _max_joint_loglik(counts_by_probe: list[np.ndarray], logmats_by_probe: list[np.ndarray]) -> np.ndarray:
    """Maximum joint log-likelihood, selecting one candidate for all probes."""
    if len(counts_by_probe) != len(logmats_by_probe) or not counts_by_probe:
        raise ValueError("counts/log-matrix probe lists must be non-empty and aligned")
    reps = counts_by_probe[0].shape[0]
    joint = np.zeros((reps, logmats_by_probe[0].shape[0]), dtype=np.float64)
    for counts, logmat in zip(counts_by_probe, logmats_by_probe):
        if counts.shape[0] != reps or counts.shape[1] != logmat.shape[1]:
            raise ValueError("incompatible count and log-likelihood shapes")
        impossible = ~np.isfinite(logmat)
        ll = counts @ np.where(impossible, 0.0, logmat).T
        ll[(counts @ impossible.astype(np.int8).T) > 0] = float("-inf")
        joint += ll
    return np.max(joint, axis=1)

def _simulate_scores(n: int, reps: int, truth: str, model_index: int | None,
                     prepared: dict, rng: np.random.Generator) -> np.ndarray:
    if n <= 0 or reps <= 0:
        raise ValueError("n and reps must be positive")
    if truth not in ("ood","candidate"):
        raise ValueError("truth must be ood or candidate")
    if truth == "candidate" and (model_index is None or not 0 <= model_index < len(base.MODELS)):
        raise ValueError("candidate truth requires a valid model index")
    counts, logmats = [], []
    for probe in base.PROBES:
        probs = prepared["ood_prob"][probe] if truth == "ood" else prepared["candidate_prob"][probe][model_index]
        counts.append(rng.multinomial(n, probs, size=reps))
        logmats.append(prepared["candidate_log"][probe])
    return _max_joint_loglik(counts, logmats)

def lower_tail_mc_pvalues(calibration_scores: np.ndarray, evaluation_scores: np.ndarray) -> np.ndarray:
    """p=(1 + #{calibration score <= evaluation score})/(B+1); ties are conservative."""
    if calibration_scores.ndim != 1 or evaluation_scores.ndim != 1 or calibration_scores.size == 0:
        raise ValueError("scores must be non-empty one-dimensional arrays")
    return (1.0 + np.searchsorted(np.sort(calibration_scores), evaluation_scores, side="right")) / (calibration_scores.size + 1.0)

def classify_from_model_pvalues(model_pvalues: np.ndarray, alpha: float = DEFAULT_ALPHA) -> str:
    if model_pvalues.ndim != 1 or model_pvalues.size != len(base.MODELS):
        raise ValueError("one p-value is required for each declared candidate")
    if not 0 < alpha < 0.5 or np.any((model_pvalues < 0) | (model_pvalues > 1)):
        raise ValueError("invalid alpha or p-value")
    return "model_misspecified" if np.all(model_pvalues <= alpha) else "catalogue_not_rejected"

def evaluate_seed(seed: int, n_values=DEFAULT_N, calibration_reps: int = DEFAULT_CALIBRATION_REPS,
                  evaluation_reps: int = DEFAULT_EVALUATION_REPS, alpha: float = DEFAULT_ALPHA) -> dict:
    if calibration_reps < 1 or evaluation_reps < 1 or not 0 < alpha < 0.5:
        raise ValueError("invalid calibration/evaluation counts or alpha")
    params = base.generate_recipe(seed)["_params"]
    prepared = _prepare(params)
    rows = {}
    for n in n_values:
        calibration = {}
        for i, model in enumerate(base.MODELS):
            rng = np.random.default_rng(_stable_seed("HIST-CAUSAL-001","v0.9",seed,n,"cal",model))
            calibration[model] = _simulate_scores(n,calibration_reps,"candidate",i,prepared,rng)
        rng_ood = np.random.default_rng(_stable_seed("HIST-CAUSAL-001","v0.9",seed,n,"eval","ood"))
        ood_scores = _simulate_scores(n,evaluation_reps,"ood",None,prepared,rng_ood)
        candidate_scores = {}
        for i, model in enumerate(base.MODELS):
            rng = np.random.default_rng(_stable_seed("HIST-CAUSAL-001","v0.9",seed,n,"eval",model))
            candidate_scores[model] = _simulate_scores(n,evaluation_reps,"candidate",i,prepared,rng)
        ood_pm = np.vstack([lower_tail_mc_pvalues(calibration[m],ood_scores) for m in base.MODELS]).T
        ood_flagged = sum(classify_from_model_pvalues(p,alpha)=="model_misspecified" for p in ood_pm)
        false_by_model = {}
        for model in base.MODELS:
            pm = np.vstack([lower_tail_mc_pvalues(calibration[m],candidate_scores[model]) for m in base.MODELS]).T
            false_by_model[model] = sum(classify_from_model_pvalues(p,alpha)=="model_misspecified" for p in pm)/evaluation_reps
        rows[str(n)] = {"ood_detection_rate":ood_flagged/evaluation_reps,
            "in_family_false_ood_by_model":false_by_model,
            "in_family_false_ood_pooled":sum(false_by_model.values())/len(base.MODELS),
            "calibration_replicates_per_candidate":calibration_reps,
            "evaluation_replicates_per_truth":evaluation_reps,
            "minimum_attainable_mc_pvalue":1.0/(calibration_reps+1.0)}
    return {"seed":seed,"results":rows}

def evaluate(seed_count: int = 20, n_values=DEFAULT_N, calibration_reps: int = DEFAULT_CALIBRATION_REPS,
             evaluation_reps: int = DEFAULT_EVALUATION_REPS, alpha: float = DEFAULT_ALPHA) -> dict:
    if not isinstance(seed_count,int) or seed_count<=0 or not n_values or any(not isinstance(n,int) or n<=0 for n in n_values) or len(set(n_values))!=len(n_values):
        raise ValueError("invalid seed count or sample sizes")
    seed_results=[evaluate_seed(s,n_values,calibration_reps,evaluation_reps,alpha) for s in range(seed_count)]
    summary, per_recipe = {}, {}
    for n in n_values:
        rows=[r["results"][str(n)] for r in seed_results]
        by_model_values={m:np.asarray([r["in_family_false_ood_by_model"][m] for r in rows]) for m in base.MODELS}
        by_model={m:float(v.mean()) for m,v in by_model_values.items()}
        od=np.asarray([r["ood_detection_rate"] for r in rows])
        po=np.asarray([r["in_family_false_ood_pooled"] for r in rows])
        summary[str(n)]={"ood_detection_rate_mean_over_recipes":float(od.mean()),
            "ood_detection_rate_sd_between_recipes":float(od.std(ddof=1)),
            "in_family_false_ood_pooled_mean_over_recipes":float(po.mean()),
            "in_family_false_ood_pooled_sd_between_recipes":float(po.std(ddof=1)),
            "in_family_false_ood_by_model_mean_over_recipes":by_model,
            "in_family_false_ood_by_model_sd_between_recipes":{m:float(v.std(ddof=1)) for m,v in by_model_values.items()},
            "worst_candidate_false_ood_rate":max(by_model.values())}
        per_recipe[str(n)]=[{"seed":sr["seed"],"ood_detection_rate":row["ood_detection_rate"],
            "in_family_false_ood_pooled":row["in_family_false_ood_pooled"],
            "in_family_false_ood_by_model":row["in_family_false_ood_by_model"]} for sr,row in zip(seed_results,rows)]
    return {"benchmark":"HIST-CAUSAL-001","version":"0.9",
        "method":"candidate-specific Monte Carlo lower-tail p-values + intersection-union model-adequacy test",
        "known_to_detector":["candidate catalogue","point parameters of each generated recipe"],
        "hidden_from_detector":["OOD mixture likelihood","OOD component identity"],
        "alpha":alpha,"calibration_replicates_per_candidate_per_recipe":calibration_reps,
        "evaluation_replicates_per_truth_per_recipe":evaluation_reps,"recipe_count":len(seed_results),
        "samples_per_probe":list(n_values),"probe_count":len(base.PROBES),
        "aggregate":summary,"per_recipe_results":per_recipe,
        "claim_ceiling":"finite-sample composite-null calibration conditional on known point parameters and independent exchangeable units; not general OOD validity"}

if __name__=="__main__":
    import argparse,json
    parser=argparse.ArgumentParser()
    parser.add_argument("--seeds",type=int,default=20)
    parser.add_argument("--calibration",type=int,default=5000)
    parser.add_argument("--evaluation",type=int,default=2000)
    parser.add_argument("--n",type=int,nargs="+",default=list(DEFAULT_N))
    parser.add_argument("--alpha",type=float,default=DEFAULT_ALPHA)
    parser.add_argument("--output")
    args=parser.parse_args()
    result=evaluate(args.seeds,tuple(args.n),args.calibration,args.evaluation,args.alpha)
    payload=json.dumps(result,indent=2,sort_keys=True)
    if args.output: Path(args.output).write_text(payload+"\n",encoding="utf-8")
    print(payload)
