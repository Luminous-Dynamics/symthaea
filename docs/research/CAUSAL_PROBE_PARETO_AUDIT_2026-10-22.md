# Causal Probe Pareto Audit — 2026-10-22

**Status:** exact local path enumeration; embedded self-test and 8/8 unit tests passed locally. This is a small research pilot, not an ACDB/CausalArena reproduction, not a benchmark ranking, and not an SOTA claim.

## Question

Can active causal discovery do more than identify a graph inside its current candidate set? This experiment adds an explicit second objective: select actions that both reduce uncertainty between candidate graphs and separate the candidate predictive distribution from a reference family representing model-set misspecification.

The prior intervention-reliability experiment showed that assuming perfect compliance can make valid in-class observations impossible under the learner. Here compliance is therefore varied explicitly, with both a q-aware learner and a deliberately misspecified q=1 learner.

## Protocol

Implementation: validation/causal_probe_pareto_benchmark.py; tests: validation/test_causal_probe_pareto_benchmark.py.

- Candidate graphs: a three-node chain and fork, with fixed Boolean mechanisms.
- OOD reference family for action selection: collider with AND and OR mechanisms.
- Held-out test world: collider with XOR mechanism. It shares the broad collider topology with the reference family but holds out its mechanism; this is **not** open-world graph-family evaluation.
- Each policy can choose at most two distinct requested actions from six do(X_i=v) candidates. Outcomes and adaptive paths are enumerated exactly.
- Actual compliance q is 0.50, 0.75 or 1.00; mechanism noise is 0.10 or 0.25. The learner either receives the correct q or incorrectly assumes q=1.
- Policies maximize EIG + beta * JSD(P(candidate outcome), P(OOD-reference outcome)), with beta in {0, 0.5, 2.0}. EIG is expected reduction in graph entropy within the chain/fork candidates. JSD is Jensen–Shannon divergence in bits between the one-step predictive distributions of candidate and reference families.
- Metrics: exact weighted ROC AUC of cumulative predictive surprise for held-out XOR versus equal-prior chain/fork worlds; expected usable graph-information reduction; in-class zero-support probability mass; and expected number of probes.

The OOD-reference models are provided to the acquisition policy. That is a deliberately informed falsification policy, not a deployable detector without a reference bank. A key next test is cross-topology holdout and a learned/uncertain reference mixture.

## Results at mechanism noise 0.10

AUC 0.5 is chance ranking. Graph information is measured in bits, with zero-support paths given zero utility credit.

| Actual q | Assumed q | Policy | Graph information (bits) | OOD surprise AUC | In-class zero-support mass |
|---:|---:|---|---:|---:|---:|
| 0.50 | 0.50 | EIG only | 0.302606 | 0.519851 | 0.0000 |
| 0.50 | 0.50 | EIG + 0.5 JSD | 0.302606 | 0.773408 | 0.0000 |
| 0.75 | 0.75 | EIG only | 0.402409 | 0.515279 | 0.0000 |
| 0.75 | 0.75 | EIG + 0.5 JSD | 0.402409 | 0.881847 | 0.0000 |
| 1.00 | 1.00 | EIG only | 0.502211 | 0.513675 | 0.0000 |
| 1.00 | 1.00 | EIG + 0.5 JSD | 0.502211 | 0.951640 | 0.0000 |
| 0.50 | 1.00 | EIG only | 0.264423 | 0.516907 | 0.4395 |
| 0.50 | 1.00 | EIG + 0.5 JSD | 0.153087 | 0.554337 | 0.6755 |
| 0.75 | 1.00 | EIG only | 0.380519 | 0.513234 | 0.2249 |
| 0.75 | 1.00 | EIG + 0.5 JSD | 0.302464 | 0.673489 | 0.3839 |

Across the 30 policy/noise/compliance cells, beta=0.5 and beta=2.0 often pick the same action path, so the pilot does not establish a stable dose-response to the new objective. At mechanism noise 0.25, the OOD AUC benefit is small or absent when q is correctly modeled in most cells; for q-misspecified cells, it improves modestly while zero-support mass increases.

## Interpretation

1. **The dual objective can expose model-set mismatch that EIG ignores.** In matched-compliance toy cases at noise 0.10, the balanced policy raises OOD AUC from roughly 0.51 to 0.77–0.95 without lowering this benchmark's graph-information metric.
2. **It is not robust to bad intervention assumptions by itself.** When actual q=0.50 but the learner assumes q=1, adding the OOD objective raises AUC only from 0.5169 to 0.5543 while zero-support mass worsens from 43.95% to 67.55%. At q=0.75 it rises from 22.49% to 38.39%. The same failure appears under the noisier condition. **Action-reliability modeling must be a safety constraint or uncertainty dimension, not an optional score.**
3. **The AUC gain is benchmark-conditional.** The policy knows AND/OR collider reference worlds, and XOR is an unseen mechanism on that same broad graph family. The result is not evidence that the policy detects arbitrary omitted variables, confounding or graph families.
4. **The correct product is a Pareto policy with a hard validity gate.** Rank actions by graph EIG and model-falsification information, but exclude actions whose benefit is conditional on an unverified actuator model; when the action model is unknown, acquire compliance evidence or propagate uncertainty over q.

## Paradigm shift

Replace the single objective “reduce posterior entropy over candidate graphs” with three separately observable goals:

- **Identify:** reduce uncertainty within the current candidate set.
- **Falsify:** distinguish candidate predictions from a preregistered reference bank of plausible alternative mechanisms and structures.
- **Validate the experiment:** detect compliance failure, off-target action or measurement/assumption mismatch.

The output should be a Pareto frontier across all three, with explicit abstention/NOT_IDENTIFIED when no action has adequate validity. It should never collapse these into a single confidence number.

## Industry comparison and next gates

- **ACDB (September 2026)** separates skeleton recovery, directed-graph recovery and intervention efficiency under a common observe/intervene/submit contract. Its reported linear-Gaussian scores are not comparable to this three-node Boolean test. The next external gate is a verified ACDB protocol adapter plus matched-budget baselines.
- **CausalArena (September 2026)** includes synthetic, semantic-operational and formula-grounded worlds, and documents ranking shifts across families and protocols. That breadth is a useful target: split by topology and mechanism, then add scientific and operational SCMs and public real-world data.
- **Falsifying Causal Graphs With Outlier Events (UAI 2026)** is relevant prior art for candidate falsification with explicit false-positive control and power claims under assumptions. Symthaea should compare to such formal tests rather than treating raw surprise AUC as a substitute.
- **Since Faithfulness Fails (ICML 2025)** makes violations of standard discovery assumptions a core evaluation axis. Our suite must extend beyond only AND/OR/XOR mechanisms and include cancellation/non-faithfulness, latent confounding, omitted variables and nonstationarity.

Proposed, not achieved qualification targets:

- At least 1,000 independently seeded SCMs across in-distribution, topology-OOD, mechanism-OOD and combined-OOD splits.
- Separate action reliability, measurement noise, structural misspecification and mechanism shift; include weak, delayed, off-target and unknown-compliance interventions.
- Report SHD, skeleton and directed precision/recall, OOD AUROC and AUPRC, 5%-false-alarm detection power, calibration/risk–coverage, zero-support mass, intervention cost, and paired confidence intervals by world.
- Compare at equal total sample/intervention budgets against PC/FCI/GES or other assumption-appropriate baselines, random/predeclared actions, greedy EIG and a verified benchmark reference method.
- Require an improvement on a preregistered primary endpoint with paired uncertainty excluding zero **and** no regression in the actuator-validity/safety endpoint.
- Evaluate 50/100/500/1000-node scale only after small-graph correctness and matched benchmark semantics are established.

## Reproduction and provenance

From the repository root:

    cd validation
    python3 causal_probe_pareto_benchmark.py --self-test
    python3 -m unittest -v test_causal_probe_pareto_benchmark.py
    python3 causal_probe_pareto_benchmark.py --source-git-blob df1d8b69f2c8e761b0e406f33e52d39757fcf603 --output /tmp/causal-probe-pareto.json

- Implementation blob: df1d8b69f2c8e761b0e406f33e52d39757fcf603; SHA-256 8882fcd0e4b320f5d12394559548821db1f11372b912e14cee3b55b06ea426e8; 11,967 bytes.
- Test blob: 21355949cd138e0406ec022a335aa232408e47d8; SHA-256 341dd38ee7541cc1b699368e959ecb651069a327050ff91a26a18b5e92ac3e4e; eight tests.
- Embedded self-test passed; 8/8 unit tests passed locally using the above exact source/test blobs.
- Exact local full-run JSON: 12,235 bytes, SHA-256 cbaf6b305926b21099be106a1dd868a24a4d224682bfef0d8b221056bb2e734a; source blob bound in the JSON.
- The repository also contains a rounded selected-results summary. Remote Actions were not claimed as passed; this is local evidence only.
