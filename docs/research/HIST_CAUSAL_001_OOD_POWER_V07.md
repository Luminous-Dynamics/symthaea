# HIST-CAUSAL-001 v0.7: finite-sample open-world detection sensitivity

## Objective

v0.6 established exact signature separation between the out-of-family population mixture and six candidate mechanisms. v0.7 estimates how many independent observations are needed to distinguish that mixture from its nearest in-family alternative at a specified confidence.

## Oracle reference

This is deliberately an optimistic binary comparator. It knows:
- the exact out-of-family mixture likelihood;
- the recipe's parameter values;
- the nearest in-family candidate chosen by average Jensen–Shannon divergence;
- equal prior odds between those two hypotheses.

It accumulates a log Bayes factor over six probe contexts. It declares a confident answer only when the likelihood ratio is at least 19 or at most 1/19, corresponding to posterior confidence at least 95% under equal priors and the two-model oracle. Otherwise it abstains.

This is not the performance of a real solver: a practical agent does not know the hidden OOD likelihood or necessarily the nearest candidate.

## Run design

- 20 public development seeds.
- 250 Monte Carlo replicates per seed, truth and sample-count condition.
- 5,000 trials per truth per sample-count condition.
- Six probes per trial; n independent observations per probe (so 6n observations per case).
- Equal prior odds and LR threshold 19.
- 95% marginal Wilson intervals reported for each outcome rate.

Nearest in-family model counts: actuator non-compliance for 18 seeds; primary sensor artefact for 2. Mean nearest-pair Jensen–Shannon divergence was 0.03030 nats.

## Results

| n per probe | OOD correct confident | OOD wrong confident | OOD abstain | Candidate correct confident | Candidate falsely called OOD | Candidate abstain |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 26.94% | 0.00% | 73.06% | 0.00% | 0.68% | 99.32% |
| 2 | 41.84% | 0.12% | 58.04% | 0.96% | 0.86% | 98.18% |
| 5 | 73.28% | 1.62% | 25.10% | 65.72% | 0.48% | 33.80% |
| 10 | 92.36% | 0.34% | 7.30% | 90.22% | 0.12% | 9.66% |
| 20 | 99.26% | 0.02% | 0.72% | 99.22% | 0.00% | 0.78% |
| 50 | 100.00% | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% |

At n=10, each case contains 60 independent probe observations; the OOD oracle confidently classified 92.36% correctly and abstained on 7.30%. At n=20 (120 observations), correct confident classification exceeded 99% in both directions. These figures are Monte Carlo performance of an oracle with known likelihoods over 20 development recipes, not general OOD detection performance or sample-size recommendations.

The Wilson intervals in results.json quantify Monte Carlo uncertainty for each reported proportion. They do not account for variation outside the 20 selected recipes, model misspecification beyond the declared mixture, or uncertainty in the real audit/measurement process.

## Paradigm shift

A model-misspecification benchmark needs three separate lanes:
1. closed-world identification, where a candidate is correct;
2. observational non-identifiability, where multiple candidates remain equivalent;
3. open-world detection, where none of the candidates is correct.

It must also measure correct abstention, false confident claims, false OOD alarms, and sample complexity separately. The v0.7 oracle is a ceiling/reference for one known OOD mixture; the real challenge is to build a calibrated solver that detects unknown misspecification without being given its likelihood.

## Claim boundary

The results are synthetic, finite and oracle-assisted. They do not establish physical discovery, morphic resonance, Chain Theory's physical interpretation, or an SOTA win.