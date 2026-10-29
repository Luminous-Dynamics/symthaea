# Decision-frontier deficit scheduling: matched audit allocation experiment

**Date:** 2026-10-29. **Status:** exploratory synthetic result; not production certification or a direct SOTA result.

## Hypothesis and method

The previous baseline allocates the next audit batch to the active risk cell with the largest upper bound on model error. This experiment tests a decision-aware scheduler that estimates which unresolved constraint is likely to produce a valid admission or rejection certificate with the least additional sampling.

The new frontier-deficit scheduler estimates a sample target from Bernoulli KL divergence between the empirical mean and the nearest admissible error boundary. If any active empirical mean is outside the allowed error band, it selects the projected cheapest rejection candidate. Otherwise, it works on the largest estimated admission deficit. This is only a sampling heuristic, not a confidence bound, optimality proof, or formal sample-complexity guarantee.

Crucially, the scheduler does not change the certificate. The only stopping rule remains the existing per-cell beta-mixture anytime confidence sequence, with familywise delta 0.10 divided across eight fixed cells.

## Matched protocol

- Eight IID Bernoulli risk cells with means [0.015, 0.035, 0.075, 0.14, 0.27, 0.50, 0.73, 0.94].
- Loss range [0,1]; pilot 128 observations/cell; batches of 128; cap 8,192 observations/cell.
- All methods reuse identical streams within each repeat/scenario. Allocation is based only on past observations.
- Five methods: uniform, maximum error upper bound, widest active confidence interval, marginal CI-width reduction, and frontier deficit.
- Four model cases: correct, safe offset 0.02, near violation 0.051, and clear violation 0.07.
- Primary panel: 100 repeats across all five methods. Secondary paired stress panel: 500 repeats on the two violating cases comparing frontier deficit with the current maximum-error-upper baseline.

## Primary 100-repeat results

Mean total observations; outcomes shown as admit / reject / unresolved.

| Case | Uniform | Max error upper | Frontier deficit | Frontier outcomes |
|---|---:|---:|---:|---|
| Correct model | 21,760 | 9,545 | 9,545 | 100 / 0 / 0 |
| Safe offset (0.02) | 35,707 | 11,415 | 11,401 | 93 / 0 / 7 |
| Near violation (0.051) | 65,536 | 11,909 | 9,549 | 0 / 0 / 100 |
| Clear violation (0.07) | 59,709 | 9,919 | 8,548 | 0 / 28 / 72 |

On the correct model, frontier deficit ties the current max-error-upper policy. The 14-observation gain on safe offset is negligible. In the clear-violation case, it uses about 13.8% fewer observations with the same reject/unresolved counts. Near the boundary it is cheaper but every run remains unresolved: that is cheaper refusal, not improved detection.

## Larger paired stress panel: 500 repeats

| Case | Method | Mean observations | P95 | Reject | Unresolved | Unsafe admission |
|---|---|---:|---:|---:|---:|---:|
| Near (0.051) | Max error upper | 11,799 | 12,934 | 0 | 500 | 0/500 |
| Near (0.051) | Frontier deficit | 9,560 | 10,880 | 0 | 500 | 0/500 |
| Clear (0.07) | Max error upper | 9,723 | 11,264 | 163 | 337 | 0/500 |
| Clear (0.07) | Frontier deficit | 8,517 | 10,240 | 163 | 337 | 0/500 |

The mean observation savings are 19.0% in the near-boundary unresolved case and 12.4% in the clear-violation case. In the clear case decisions are identical: 163 rejects and 337 unresolved results, but the new policy spends less to reach those outcomes. This is a promising cost result on this generator, not evidence of a lower false-admission rate. Zero unsafe admissions in 500 trials still leaves a nonzero two-sided Wilson 95% upper limit of approximately 0.76%.

## Interpretation and limitations

The narrow finding is that a KL-based decision-frontier proxy reduced audit consumption while preserving observed outcomes on matched IID Bernoulli scenarios. It did not win in all regimes: correct-model performance is tied, safe-offset gain is negligible, and the near-boundary case trades sooner unresolved outcomes for cost. Do not promote it over the simpler current baseline based on this alone.

The test assumes fixed support and IID Bernoulli streams. It does not establish validity under dependent audits, shared systematic bias, missing support, policy identity mismatch, concept drift, or adversarial sampling. The KL proxy never controls admit/reject, but its efficiency may degrade badly when its statistical model is wrong.

## Current research and SOTA comparators

- Cai & Kandasamy, *Constrained Best Arm Identification with Tests for Feasibility*, AAAI 2026: chooses whether to test objective performance or feasibility constraints and provides sample-complexity analysis. https://ojs.aaai.org/index.php/AAAI/article/view/39063
- Jourdan et al., *An Anytime Algorithm for Good Arm Identification* (APGAI), JMLR 2026: anytime threshold identification, theory and empirical evaluations. Official code: https://github.com/clreda/apgai. Paper: https://www.jmlr.org/papers/v27/24-0680.html
- Komiyama, Jang & Honda, *Rate-optimal Design for Anytime Best Arm Identification*, AISTATS 2026: an anytime approach with minimax-rate guarantees. https://proceedings.mlr.press/v300/komiyama26a.html
- Harding & Kandasamy, *Balancing Performance and Costs in Best Arm Identification* (DBCARE), NeurIPS 2025: explicitly balances observations against recommendation error. https://papers.nips.cc/paper_files/paper/2025/hash/b0dfe8236981ba84b4d98a7ed0426f0b-Abstract-Conference.html

These are related but not directly interchangeable with multi-constraint risk-certificate allocation. No published baseline implementation was run here; this report makes no claim of beating state of the art or industry.

## Next benchmark gates

1. Reproduce APGAI and the AAAI constrained-BAI method where usable code/data are available, and define a matched objective/feasibility task so every baseline solves the same problem.
2. Run 1,000+ independent seeds with 8, 32, 128 and 512 cells, varied threshold gaps, rare-event risks, heteroskedasticity and unequal audit costs.
3. Measure mean, P95/P99 audit cost, unsafe admission, false rejection, unresolved rate, anytime calibration, wall time, memory and worst-group outcomes.
4. Add dependent/clustered data and explicit bias/support-shift negative controls. IID Bernoulli confidence sequences are not enough for those cases; use a justified model or fail closed.
5. Keep audit-integrity preflight, support identity and independent verifier separate from the scheduler. Never let its KL proxy alter certificate thresholds.

## Reproduction and provenance

The reproducibility bundle includes both Python baseline modules, the new implementation and test, both raw JSON result files, test log, summary and SHA256 checksums. Run the test with:

    python -m unittest discover -s . -p 'test_decision_risk_frontier_deficit_allocation.py' -v

Run the 100-repeat comparison with:

    python decision_risk_frontier_deficit_allocation.py --repeats 100 --output DECISION_RISK_FRONTIER_DEFICIT_ALLOCATION_RESULTS_2026-10-29_100.json

Python research lane only; not integrated into the Rust planner or production dispatch gate. Local test success is not remote CI success.
