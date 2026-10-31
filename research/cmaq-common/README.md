# CMAQ common frozen-run manifest

**Status:** v0.1 schema and CMAQ-018 development example; no benchmark result is included.

## Why this exists

CMAQ-011, -017 and -018 now have separate scientific estimands and their own reports. A local schema pass in each track does not prove their runs used compatible candidate versions, data splits, action semantics, compute budgets, baselines or statistical units.

This directory defines the shared **pre-run freeze contract**. It complements, rather than replaces:

- CMAQ-015 signed benchmark evidence receipts, which attest to evidence after execution;
- each track's native metric schema, which preserves its scientific meaning;
- CMAQ-011 external baseline locks, which pin outside repositories, dataset and wrapper semantics.

The intended lifecycle is:

1. Freeze candidate source/config/adapter and build identity.
2. Freeze generator, environment, action and reward semantics.
3. Freeze dataset revisions, split families and seed commitment.
4. Freeze eligible baselines and the exact comparison claims.
5. Freeze resource limits, cost-measurement method, metric bundle and statistical plan.
6. Hash the manifest and run only the frozen candidate/evaluator pair.
7. After execution, emit the report and a digest-bound CMAQ-015 run receipt.

The manifest is not a run result and does not attest that a run actually happened.

## Contract

The strict schema is `frozen-run-manifest.schema.json`. The example validator recomputes SHA-256 from repository-relative candidate, generator, metric bundle, data manifest, observation/action/reward schema and wrapper-stack artifacts; hash-shaped strings alone are not accepted. It rejects traversal, symlink and out-of-repository artifact paths. Its top-level fields bind:

- **Track and claim:** one CMAQ lane, protocol version and maximum scientific claim.
- **Repository and candidate:** immutable source commit, candidate artifact digest (nullable only when a source-only development fixture is explicitly marked), config/weights/adapter digests and build state.
- **Generator and evaluator:** their source revisions, repository-relative configuration/metric-bundle paths and digests, plus external judge identity/version where relevant.
- **Data sources:** exact revision, split, manifest digest, license, candidate visibility and permitted candidate-facing fields.
- **Task semantics:** observation/action schemas, reward definition, wrapper stack, action repeat, simulator-step cadence, controller horizon and terminal behavior.
- **Sample and split custody:** episode count, seed-schedule commitment, analysis unit, mechanism-family split and paired-seed declaration.
- **Budgets:** action/probe/candidate budgets, forward calls, model-transition evaluations, wall time, memory and energy. Unmeasured values are `null`, never silently zero.
- **Baseline registry:** baseline role, implementation/config identity, eligibility and whether cost is measured, instrumented, estimated, candidate-count-only or unavailable.
- **Comparisons:** treatment/control IDs, explicitly matched dimensions, status and scientific claim.
- **Statistics and qualification:** cluster unit, bootstrap or sequential inference method, freeze order, truth separation and whether the qualification split is sealed.
- **Claim ceiling:** explicit prohibition on pooled cross-track scores and unsupported claims.

## Important semantics

### Cost matching

The schema distinguishes `candidate_budget_only`, `estimated_compute_proxy`, `measured_flops`, `measured_wallclock` and `matched_multi_resource`. They are not synonyms.

For example, CMAQ-017's `direct_search_control` matches candidate-sequence count but has no world-model calls and is not hardware/FLOP/latency matched. It must remain `candidate_count_only`; it cannot be named or cited as the D1 compute-matched direct baseline. The true D1 contrast remains unqualified until compute is instrumented and matched.

### Statistical units

CMAQ-018 has 21 action pairs per episode history. Those pairs are correlated. The sample unit is the episode history, and bootstrap resampling must resample whole episode histories rather than individual pairs.

CMAQ-017 resamples paired episode seeds. Its lower-decile result from 16 episodes is a tail diagnostic, not a stable CVaR95 estimate.

### No cross-track scalar

Visual fidelity (WorldOracle/WBench/WorldSimProbe), interventional branch ranking (CMAQ-018), planner exploitability (CMAQ-017), and closed-loop decision value (CMAQ-002/004) remain separate score families. The manifest requires `cross_track_pooling=false`.

## Current track migration

| Track | Manifest binding | Important limitation |
| --- | --- | --- |
| CMAQ-011 | External baseline locks, native-vs-normalized wrapper mode, task/action cadence and reward semantics | A pinned baseline is not a reproduction |
| CMAQ-017 | Scenario/horizon/execution block/candidate budget, candidate-count-only control, transition call counts | No true measured-compute D1 yet; hardware cost not measured |
| CMAQ-018 | Same-history action bank, evaluator-only branch identity, per-episode metrics and episode-cluster bootstrap | Synthetic predictor arms only; no physical counterfactual |
| CMAQ-019 | WorldOracle code/dataset revision, CC BY-NC-SA 4.0 use constraints, judge version and per-episode identity | Public validation is non-commercial/share-alike; test inputs remain private and test GT withheld |
| CMAQ-015 | Post-run evidence receipt and signature/purpose/freshness/replay verification | Receipt validity is not empirical validity |

The example manifest is for CMAQ-018's frozen **development** panel. It explicitly marks the candidate as source-only/not compiled, reports no hardware cost, marks all synthetic predictor comparisons diagnostic-only and includes no performance result. It is a schema/migration fixture—not a qualification claim.

## Reproduction and validation

From this directory:

```sh
python3 -m pip install -r requirements-validation.txt
python3 scripts/validate_manifest.py
```

The GitHub Actions workflow validates the example against the schema, checks cross-reference integrity, and runs hostile mutations that must be rejected.

This first PR establishes a declarative contract and CI cross-checks. A dependency-light Rust runtime validator and per-track native-report adapters remain follow-on work; the Python script is CI-only and must not be treated as production runtime code.

## Claim ceiling

Passing schema and hostile fixtures proves only that this manifest shape is parsed and its declared cross-field rules are enforced. It does not prove baseline comparability, actual cost matching, data rights, split secrecy, evaluator correctness, model performance, physical safety or SOTA. Those require independent artifacts and exact-head runs.
