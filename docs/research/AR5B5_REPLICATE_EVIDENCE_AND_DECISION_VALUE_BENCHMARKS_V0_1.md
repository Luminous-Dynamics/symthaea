# AR-5B5: Replicate Evidence Binding and Decision-Value Benchmark Protocol v0.1

**Status:** implementation draft; tests authored, not executed.  
**Tracking:** [#9692](https://github.com/Luminous-Dynamics/symthaea/issues/9692), [#9663](https://github.com/Luminous-Dynamics/symthaea/issues/9663), [#9696](https://github.com/Luminous-Dynamics/symthaea/issues/9696).  
**Claim ceiling:** byte/identity binding for one canonical observation envelope. No evaluator-validity, remote-store authenticity, signer-authenticity, statistical-independence, optimizer-quality, or SOTA claim.

## 1. Why this tranche exists

A noisy Pareto sample is a measurement, not merely a scalar score or a seed. The history already retains the objective vector and an evidence digest, but a digest alone does not establish that the referenced bytes exist or that they encode the vector used by the confidence analysis.

This tranche closes one narrow integrity gap:

- one replicate has one manifest;
- the manifest binds run, source commit, candidate, decision digest, replicate ID, sample seed, frozen noisy-plan digest, ordered objective-vector digest, and artifact reference;
- the canonical JSON observation envelope repeats those identities and objective IDs/values;
- verification checks exact byte length and SHA-256, parses a deny-unknown-fields record, requires canonical serialization, and compares the complete vector to the sample retained by history;
- the receipt binds the manifest, artifact bytes, objective vector, and verifier ID/build digest.

The verifier API accepts already-resolved bytes. It does not perform I/O. A future resolver adapter must enforce namespace and byte ceilings using the existing content-addressed artifact resolver contract.

## 2. Data contract

The new Rust module is **noisy_replicate_evidence.rs**.

### 2.1 Canonical observation envelope

Schema: **symthaea.architecture-engineering.noisy-replicate-observation v0.1.0**.

Required fields: run_id, candidate_id, source_commit, decision_sha256, replicate_id, sample_seed, plan_sha256, and objective_values preserving the preregistered objective ID order.

Each objective value must be finite and inside the observation bounds in the frozen plan. Objective IDs and directions are not inferred from the artifact: they are read from the frozen plan.

### 2.2 Replicate manifest

Schema: **symthaea.architecture-engineering.noisy-replicate-evidence v0.1.0**.

The manifest is replicate-scoped, not candidate-scoped. It binds the canonical JSON artifact as an evaluation_output, plus a content-addressed locator, byte length, and SHA-256. The artifact SHA must equal the NoisyReplicate.evidence_sha256 retained by history.

The objective-vector digest is over the ordered (objective_id, value) vector. Reordering objectives therefore changes the digest even when the numeric values happen to be the same.

### 2.3 Verification receipt

Schema: **symthaea.architecture-engineering.noisy-replicate-byte-receipt v0.1.0**.

The receipt is independently reconstructable from the manifest, frozen plan, expected history sample, verifier ID/build digest, and supplied bytes. Altering any identity, vector value, artifact byte, manifest field, or receipt field must fail closed.

**Interpretation limit:** verifier ID/build are recorded and digest-bound, not authenticated by a signature in this tranche. Byte correspondence is not proof that the measurement process was honest or correct. Keep raw instrument/evaluator logs and verifier attestations as separate artifacts.

## 3. Replay and integration boundary

This is a composable verifier primitive, not yet the end-to-end EventTape implementation requested by #9692.

Still required:

1. Add native EventTape vector/noisy actions under a separately versioned campaign schema.
2. Record all replicate attempts, including failures, invalid samples, timeouts, retries, and verifier rejections, with complete cost receipts.
3. Retain one replicate manifest and raw artifact for every successful observation.
4. Resolve bytes through a bounded resolver and reconstruct every per-replicate receipt.
5. Reconstruct the noisy Pareto report from EventTape alone, then compare its digest with the AnytimeHistory reconstruction.
6. Require exact campaign run, source commit, objective ordering/directions/bounds, plan, sampling, verifier, and history-tip agreement.
7. Keep failed observations charged but exclude them from means and confidence intervals.
8. Distinguish summed worker time from elapsed campaign latency under parallel execution.

No candidate becomes ArtifactVerified merely because an artifact hash is present. Promotion requires exact byte resolution and receipt reconstruction through an independently identified verifier path.

## 4. Scientific assumptions and adversarial controls

The current confidence analyzer uses bounded-observation Hoeffding intervals with summable alpha spending. Its coverage claim is conditional on the declared sampling model; unique seed identifiers alone do not establish independence.

The next benchmark matrix must include both in-model and deliberately out-of-model data:

| Regime | Construction | Required behavior |
|---|---|---|
| Bounded IID | Frozen support bounds; independent counter-based streams | Confidence coverage at the preregistered family-wise level |
| Heteroskedastic | Candidate- and objective-dependent variance | No false certainty; compare cost to decision |
| Cross-objective correlation | Shared latent noise and shared physical samples | Preserve sample linkage; do not pretend coordinates are independent |
| Serial correlation | AR-like dependence / block resampling | Flag unsupported independence assumptions |
| Heavy tails / outliers | Frozen contamination and outlier sweeps | Expose model failure; compare robust alternatives |
| Regime drift | Change-point and gradual drift | Detect/invalidate stale sampling assumptions |
| Informative missingness | Objective-specific timeout / missingness correlated with outcome | Never silently treat missing as zero or IID |
| Seed collision | Repeated or aliased seed streams | Reject or explicitly mark dependence |
| Verifier corruption | Wrong vector, swapped objective IDs, fabricated receipt | Reject before statistical aggregation |
| Artifact corruption | Missing bytes, truncation, substitution, length/hash mismatch | Reject before evidence-level promotion |

Future confidence methods should be separate, versioned estimators: bounded Hoeffding, empirical-Bernstein where assumptions apply, robust bounded/contaminated estimators, and dependence-aware methods. Do not silently change the coverage interpretation of the current method.

## 5. Benchmark programme and industry comparators

### Public testbeds

- **COCO:** use bbob-biobj (55 bi-objective functions), bbob-noisy (24 noisy single-objective functions), and bbob-constrained (54 constrained functions) as complementary—not interchangeable—suites. Add bbob-largescale when testing representation/scaling, with dimensions reaching 640. The official archive provides datasets, source links, and programmatic metadata: https://coco-platform.org/data-archive/index.html
- **BBOB noise sweeps:** include frozen-noise outlier models and parameter sweeps. The 2025 COCO workshop explicitly called for underexplored noisy, constrained, mixed-integer and multi-objective regimes, including parametrized outlier noise: https://coco-platform.org/workshops/bbob2025.html
- **PlatEMO:** use a pinned release and select appropriate algorithms/problems by regime; do not treat the whole catalogue as one baseline. The project is a broad multi-objective optimization platform: https://github.com/BIMK/PlatEMO
- **BoTorch/Ax:** use qLogNEHVI/qNEHVI and qLogNParEGO/qNParEGO for noisy and/or batched Bayesian MOO when their surrogate assumptions are suitable. BoTorch's tutorial explicitly recommends qNEHVI for noisy or batched settings: https://botorch.org/docs/v0.17.2/tutorials/multi_objective_bo
- **Classical evolutionary baselines:** retain NSGA-II, MOEA/D-DE and SMS-EMOA, with exact pinned configs, seeds, stopping budgets and hypervolume reference points.
- **Cheap exploration controls:** Sobol, random search, uniform replication, random replicate allocation, interval-width-per-cost, racing/successive-halving and best-arm identification controls.

### Benchmark layers

1. **Algorithmic Pareto search:** ZDT, DTLZ, WFG, COCO bi-objective and selected many-objective problems.
2. **Noisy expensive optimization:** Branin-Currin, noisy DTLZ, COCO noisy/outlier models, heteroskedastic and correlated objectives.
3. **Constrained/feasibility search:** constrained DTLZ and COCO constrained families; report safety/feasibility separately from objective trade-offs.
4. **Real system optimization:** held-out Rust/Nix build, test, runtime, memory, energy and reliability tasks. Freeze task recipes and validate resulting artifacts, not just predicted objective values.

Use public suites for comparability, plus sealed held-out task families for generalization. Never tune against the same problem instances used for final claims.

## 6. Metrics: score the decision, not only the front

Report all of the following by problem family and by independent landscape/task cluster:

- **Quality:** hypervolume with a frozen reference point, IGD+, additive epsilon, and where suitable R2/attainment indicators.
- **Anytime:** target-attainment runtime and empirical cumulative distributions over target levels, not only final-budget quality. COCO's performance-assessment methodology uses runtime-to-target and ECDF-style aggregation: https://numbbo.github.io/coco-doc/perf-assessment/
- **Decision safety:** false-certain-dominance, false-feasibility, missed target, and premature-stop rates against known ground truth.
- **Uncertainty:** empirical interval coverage, width/sharpness, calibration by replicate count, and frequency of detecting a violated sampling assumption.
- **Economics:** sample count and all-in host CPU, wall latency, worker time, accelerator, memory, energy, evaluator, verifier, setup, acquisition, retries, and artifact-resolution cost.
- **Evidence quality:** unresolved references, digest failures, wrong-vector substitution, receipt reconstruction failures, and time from measurement to independently verified result.
- **Controller burden:** decision overhead and selection-policy computation cost, including model fitting and acquisition optimization.

COCO's anytime performance work is a useful standard for target/runtime reporting; it does not replace replicate-level evidence accounting.

## 7. Cost-aware decision-value research (issue #9696)

The next optimizer should choose the next measurement by expected reduction in decision loss per all-in cost—not merely the widest interval.

Start with a tractable exact oracle:

- enumerate a small candidate set, two objectives, and discrete bounded noise distributions;
- define a decision loss over false-dominance, missed feasibility, missed target, and delayed decision;
- exactly integrate the predictive outcome of each possible candidate/objective/replicate action;
- subtract acquisition, evaluation, setup, verification, and retry costs;
- compare the exact EVSI ranking to uniform, random, width-only, width-per-cost and racing policies.

Only call a policy EVSI when it uses an explicit predictive distribution and decision loss. Interval width is an uncertainty heuristic, not EVSI.

Then add a scalable approximation and compare it with the exact oracle on the same small instances before evaluating larger problems. Report approximation error and decision regret, not just rank correlation.

## 8. Proposed acceptance gates

These are proposed research gates, not measured results:

- **Artifact integrity:** 100% of successful replicate rows resolve to exact bytes and reconstruct an exact receipt; zero accepted wrong-vector or wrong-identity substitutions in adversarial controls.
- **Statistical calibration:** nominal 95% family-wise coverage is empirically checked across many independent generated campaigns; report confidence bounds for coverage and false-certification rates.
- **Fail-closed behavior:** all deliberately invalid dependence, drift, missingness, timeout, and verifier-corruption controls are detected or cause an explicit downgrade to an inapplicable/unqualified state.
- **Decision value:** on the exact small-instance suite, scalable EVSI's decision regret is within a preregistered tolerance of the exact oracle; at equal all-in cost it must beat uniform replication and width-only allocation on held-out families before it can be called an improvement.
- **Generalization:** pilot results may guide engineering; confirmatory claims require at least 100 independent landscape/task clusters with paired seeds and a locked analysis plan.
- **SOTA:** no global SOTA language from a narrow synthetic suite. State the exact suite, dimensions, noise model, constraints, cost definition, baseline versions, confidence intervals, and held-out split for every claimed win.

## 9. Exact-head validation commands

Run on the PR head (not a parent or local stale branch):

    cargo fmt --all -- --check
    cargo test -p symthaea-architecture-engineering noisy_replicate_evidence -- --nocapture
    cargo test -p symthaea-architecture-engineering
    cargo clippy -p symthaea-architecture-engineering --all-targets -- -D warnings

The current execution environment did not provide cargo or rustc; therefore the tests in this tranche are authored but not compiled or executed. CI skipped is not a pass.
