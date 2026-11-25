# SDCB hidden evaluation and reproducibility protocol v1

**Status:** design-only governance protocol  
**Parent:** [SDCB v1](SPECTRAL_DYNAMICAL_COGNITION_BRIDGE_V1.md)  
**Purpose:** prevent benchmark leakage, adaptive overfitting, cherry-picked spectral statistics, and non-reproducible model comparisons.

## 1. Public / hidden split boundary

The public repository may contain:
- task generator source and version;
- task families and parameter domains;
- public development examples;
- validation examples and expected dispositions;
- schema, scoring code, negative-control definitions, and analytic oracles;
- commitments to a future hidden evaluation snapshot.

The public repository must not contain before a round closes:
- hidden task seeds or task-specific parameters;
- hidden task payloads, target outputs or proof answers;
- hidden labels or scoring annotations;
- private evaluation credentials;
- secret salts/nonces used for dataset commitments.

The current 34-task manifest is design-only and contains development/validation specifications. It is not a hidden test set. The hidden split must be generated and held by an independent steward before confirmatory evaluation.

## 2. Commitment and reveal

Before model submissions are accepted, the steward freezes:
- generator repository and commit;
- task-family/parameter distribution;
- source and toolchain revisions;
- hidden seed material;
- task count and exclusion rules;
- scoring code and version;
- allowed compute/tool budget;
- evaluation window and disclosure schedule.

For each hidden snapshot, compute a domain-separated commitment:

    SHA256("SDCB-HIDDEN-SNAPSHOT-v1\0" || canonical_manifest || secret_nonce)

The secret nonce must be at least 256 random bits and retained by the steward. Do not use an unsalted hash of a low-entropy task seed: it can be brute-forced. The public commitment binds the sealed snapshot but does not authenticate its author, prove custody, or establish unbiased task selection.

The steward signs a run manifest and the post-run receipt with a separately governed signing key. Publish the key identity, validity interval, revocation procedure, and signature verification instructions. A signature establishes a statement by the identified key, not truth of the measured outcome.

After the evaluation window closes, reveal the nonce and eligible snapshot data (respecting upstream licensing and contamination constraints). Independent auditors recompute the commitment and replay the task generator and scoring code. A future round uses fresh hidden entropy.

## 3. Anti-leakage design

- Split by **task family and generating parameter regime**, not merely by random rows.
- Separate nearby arithmetic ranges, dynamical parameter regions, graph topologies, proof families, and source lineages where they could leak structural templates.
- Keep synthetic generator code public while withholding independently sampled seeds/parameters until the round is frozen.
- Randomize task IDs and evaluation order; preserve a private mapping for audit.
- Include duplicate/canary items and semantically equivalent metamorphic variants to detect memorization and order effects.
- Record known overlap with training corpora, public benchmarks, upstream repositories, and previously exposed evaluation tasks.
- Exclude tasks whose provenance, license, oracle, or target statement is ambiguous.
- Do not treat a hidden benchmark as secret forever: publish a delayed reveal and rotate the next round.

## 4. Exact system freeze

Every submission must bind:
- model/checkpoint digest and model family;
- tokenizer/vocabulary digest, if relevant;
- architecture and parameter-count report;
- inference engine, runtime, dependency lock and container/environment digest;
- prompt/template, decoding settings, seed, sampling count and retry policy;
- retrieval index and corpus digests;
- tool definitions, permissions, network policy and sandbox image digest;
- hardware model, accelerator, precision, batch size and resource limits;
- benchmark harness, scoring and generator commits;
- task snapshot commitment and run identifier.

A model label alone is insufficient. If any binding changes, it is a different system submission.

## 5. Matched comparisons

For each comparator, report three non-interchangeable views:
1. **Capability-matched:** same task, task information, data exposure and allowable context.
2. **Compute-matched:** comparable measured accelerator/CPU time, memory and inference budget.
3. **Deployment-matched:** same latency, memory, energy/power envelope and tool permissions.

Where exact matching is impossible, show the mismatch and report a sensitivity analysis. Do not normalize away a decisive mismatch with a single score.

For API-backed systems, record timestamped model/version identifiers, token usage, request count, retries, rate limits and billed cost. For local systems, record wall-clock, CPU/GPU telemetry and measured energy only when the sensor/calibration is documented. If energy measurement is unavailable, mark it unavailable.

## 6. Evaluation outcome vector

Publish these separately:
- task success / exact correctness;
- calibrated confidence and selective abstention;
- OOD and family-level generalization;
- metamorphic and adversarial robustness;
- safety/authority violations;
- invalid-evidence, stale-evidence and provenance failures;
- recovery after induced fault or unknown outcome;
- trajectory quality and unnecessary action/tool count;
- latency, peak memory, token/API cost and measured energy;
- reproducibility across seeds and independent implementations.

A critical gate failure is non-compensatory. High accuracy cannot offset an unauthorized action, a false proof acceptance, or corrupted evidence. A valid abstention is distinct from a failed attempt and from an evaluator outage.

## 7. Statistical treatment

- Use paired task instances across systems wherever possible.
- Report raw per-task and per-seed outcomes where licensing and privacy permit.
- Cluster uncertainty by task family / source lineage; do not treat correlated examples as independent.
- Report confidence intervals, effect sizes and all preregistered primary outcomes.
- Correct confirmatory multiple comparisons.
- Publish excluded-task IDs and reasons, not just the retained denominator.
- Use an explicit inconclusive result when sample size, telemetry, source status, or evaluator integrity is insufficient.
- Preserve failures and negative results. Do not delete benchmark versions after saturation; mark them and publish the successor protocol.

## 8. Evaluator independence

The evaluator must not be the system under test. For stronger evidence:
- isolate the evaluator from model credentials and mutable candidate state;
- use separate roles, profiles and signing keys for submission, scoring and attestation;
- independently capture task inputs and effect/resource telemetry;
- retain an externally witnessed checkpoint of run records;
- allow a second party to replay selected tasks from the sealed snapshot;
- report whether evaluator identity, signing, timestamp, checkpoint inclusion, and independent reproduction were actually verified.

A locally consistent event chain is not proof that the full history was never rewritten. A signed report is not proof of source truth. A second evaluator is not independent merely because it has a different process name.

## 9. SDCB-specific leakage and confound controls

**Zeta/prime track:** pin the zero-ordinate source and precision, exact arithmetic range, truncation depth and explicit-formula implementation. Keep entire arithmetic intervals and moduli out of training/tuning. Do not let training labels encode prime targets through features or file names.

**RMT track:** choose null ensembles and spectral statistics before test-set access. Keep matrix dimensions, aspect ratios, normalization, finite-size correction, and weight/activation extraction policy fixed. Report both singular tails and bulk. Random orthogonal rotations and spectrum-preserving eigenvector shuffles are mandatory controls.

**Criticality track:** preregister all diagnostics and scaling ranges. Use multiple system sizes and held-out topologies. Do not infer criticality from one log-log slope. Include off-critical controls and test whether the estimator distinguishes known synthetic regimes.

**Koopman / system-identification track:** report persistent excitation, observability, sampling, noise, stationarity/regime assumptions, sequence length and approximation error. Include adversarial trajectories where the available observations do not identify the underlying dynamics. The correct outcome may be “not identifiable from this evidence.”

**Memory track:** freeze codebook generators, item identities, distractor schedule and recall timing. Use no-op, stale-memory and version-swap controls. Separate storage capacity from retrieval accuracy and recovery.

**Proof track:** pin exact theorem statements and Lean environments. Prevent theorem-family and source-lineage leakage. Keep premise retrieval, proof-plan accounting, kernel acceptance, checker diversity, theorem explanation and human transfer as separate outcomes.

## 10. Claim ladder

Promote claims only through these levels:

1. **Reference validated:** analytic controls and mutation checks match expected results.
2. **Exploratory observation:** a signal appears on the development set; no generalization claim.
3. **Held-out replicated result:** signal survives validation and preregistered family-level testing.
4. **Causal mechanism evidence:** controlled intervention and restoration support the proposed mechanism.
5. **Independent reproduction:** another group reproduces the effect and controls on frozen data.
6. **Comparative advantage:** the system outperforms matched external baselines on preregistered outcomes and budgets.

No step may be skipped. A large effect on a public development set cannot be promoted directly to a field-level claim.

## 11. Current implementation status

- Public generator/task specifications: present in SDCB v1.
- Public hidden-set commitment implementation: not yet implemented.
- Independent benchmark steward: not yet appointed.
- Signing/checkpoint infrastructure for this benchmark: not yet implemented.
- Dataset licensing and human/EEG ethics: biological data are excluded from v1.
- No hidden corpus, model run, benchmark result, or independent evaluation is claimed.

The immediate next implementation should be a deterministic canonical snapshot/receipt format. Signing and external checkpoint witness integration should reuse the existing Luminous evidence architecture; do not create a competing transparency log or duplicate authority subsystem.
