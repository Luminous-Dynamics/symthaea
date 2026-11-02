# HIST-CAUSAL-001 v0.3: finite-sample leakage audit sensitivity

**Status:** exact binomial design sensitivity only; no empirical audit calibration.

The v0.2 result establishes that the candidate sealed-history model and hidden-leak model have different distributions only when an independent leakage audit is included. This v0.3 layer asks how many independent audit units are needed to distinguish them under known audit performance.

Assumptions:
- two hypotheses: no conventional leakage (history model) versus conventional leakage;
- sensitivity and false-positive rate are fixed and known;
- audit outcomes are independent across units;
- posterior threshold is 95% for either confident classification;
- minimum correct-confident classification probability is 90% under either true model;
- maximum wrong-confident probability is 1%;
- priors over leakage are 0.1, 0.5 and 0.9;
- exact rational binomial probabilities; no Monte Carlo error.

| Audit quality | Sensitivity | False alarm | Prior leak 0.1 | Prior leak 0.5 | Prior leak 0.9 |
|---|---:|---:|---:|---:|---:|
| High | 95% | 1% | 2 units | 2 units | 3 units |
| Moderate | 80% | 5% | 5 units | 5 units | 7 units |
| Weak | 70% | 10% | 11 units | 9 units | 11 units |
| Poor | 60% | 20% | 30 units | 23 units | 31 units |

A unit is an independently audited experimental family, not a participant, repeated trial or sensor read treated as IID. The assumption of fixed, known calibration is optimistic. Shared detector drift, unknown sensitivity, correlated leakage, site effects and prior misspecification can increase the required sample substantially or make the mechanism non-identifiable.

The result shows why a positive leakage monitor or a nominally sealed setup is not enough. The audit itself needs blinded positive/negative calibration, drift checks, independence analysis and an adversarial leakage model. An underpowered or poorly calibrated audit must lead to abstention.

## Reproducibility

Run `python audit_power.py` and `python -m unittest -v test_audit_power.py` using Python's standard library only. The script searches n=1..256 under exact rational arithmetic.

## Claim ceiling

These are mathematical sensitivity values for a stylized pair of Bernoulli models. They are not real-world sample-size recommendations and do not support or refute any cross-system historical influence.