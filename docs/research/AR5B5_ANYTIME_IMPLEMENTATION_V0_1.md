# AR-5B5 Anytime Evaluation History Implementation v0.1

**Status:** Rust implementation committed; compilation, formatting, tests, clippy, and external parity checks not yet demonstrated  
**Module:** \`crates/domains/symthaea-architecture-engineering/src/pareto_anytime.rs\`  
**Export:** \`pub mod pareto_anytime;\` in the architecture-engineering crate  
**Protocol:** \`AR5B5_ANYTIME_EVIDENCE_COST_ATTAINMENT_V0_1.md\`

## 1. Implemented surface

The first code tranche is optimizer-agnostic. It provides a shared event-history and target-attainment interface that can be emitted by NSGA-II, MOEA/D-DE, SMS-EMOA, or a future optimizer without making their internal algorithms depend on one another.

### Frozen header and provenance

\`AnytimeHistoryHeader\` binds:

- run, algorithm, evaluator, and verifier identities;
- algorithm, evaluator, and verifier build SHA-256 digests;
- execution-environment digest;
- problem ID and seed;
- ordered objective specifications and scale anchors;
- hypervolume reference point;
- canonicalized target plan and reference Pareto front;
- cost-model ID/digest;
- objective-evaluation attempt budget;
- required resource meters.

The header itself is SHA-256 bound. The target plan is independently digestible.

### Hash-chained event history

\`AnytimeHistory::append\` emits contiguous one-based event ordinals and a SHA-256 chain rooted in the header digest. Each event binds its time interval, resource vector, action payload, and previous-event digest.

Supported action types:

- successful, failed, invalid, and timed-out candidate evaluations;
- candidate verification promotions;
- explicit candidate verification rejections;
- proposal/measurement/tool/other auxiliary actions.

A failed attempt is retained and charged. Candidate retries must have contiguous attempt numbers and retain the same decision digest. A decision digest cannot be admitted under two different candidate IDs, preventing duplicate bytes from inflating archive size. A canonical candidate may be successfully admitted only once in v0.1; repeated measurements must use a separate run/replication identity until an explicit replicate-aggregation contract is implemented.

### Evidence levels

The recorded levels are:

- Recorded
- ReplayVerified
- ArtifactVerified
- HeldOutVerified
- Confirmatory

A target declares the minimum level required for inclusion in its target-specific frontier. This allows ordinary optimizer-quality targets and held-out evidence targets to be reported separately.

Important boundary: the history module validates the structure and digest references of verification events. It does **not** retrieve candidate artifact bytes, validate cryptographic verifier signatures, or independently prove that a named verifier performed the declared check. Evidence levels are assertions recorded in the chain until an artifact/receipt resolver and verifier-authenticity layer independently validate the underlying evidence. Do not use a HeldOutVerified label alone as a scientific superiority claim.

### Resource accounting

\`ResourceCost\` stores twelve integer meter fields. Additive values (CPU, worker-time, accelerator-time, energy, tokens, calls, and normalized charge) sum with checked arithmetic. Peak-memory fields aggregate by maximum. Unknown values propagate as unknown and disqualify a target's strict resource-cost qualification whenever the meter is declared required.

Elapsed wall time is the monotonic completion timestamp of the run/event prefix, not the sum of action durations. Worker-time remains separate from elapsed time.

### Archive-prefix metrics and target receipts

For each event prefix and each declared evidence level, the module reconstructs the candidate set and calls the shared Pareto analysis implementation. It calculates:

- hypervolume against the frozen reference point;
- IGD+ against the frozen reference front;
- additive epsilon against the frozen reference front.

An indicator is unavailable when the eligible feasible archive is empty; unavailable IGD+/epsilon does not satisfy a target. Hypervolume is zero for an empty feasible archive.

Each target has one final outcome:

- Reached, with the first crossing event ordinal, objective-evaluation-attempt count, elapsed time, cumulative resource vector, archive digest, and resource-qualification flag;
- NotReachedByBudget, with final metric, final archive digest, final resource vector, elapsed time, and resource-qualification flag.

The receipt binds the header digest, target-plan digest, final history-chain tip, all target results, and total resources. \`verify_anytime_run_receipt\` reconstructs the receipt from the exact event history and rejects substituted fields.

## 2. Initial authored tests

The module contains tests for:

1. exact target-prefix crossing and HeldOutVerified gating;
2. failed retry charging and unknown required meter disqualification;
3. a dominated candidate not increasing hypervolume;
4. verifier rejection being retained and charged without promoting evidence level;
5. duplicate decision bytes being rejected under a second candidate ID;
6. overlapping event intervals and mismatched verifier evidence being rejected;
7. receipt tampering being rejected by deterministic reconstruction.

These tests are authored only. They have not been compiled or executed in this environment.

## 3. Explicit v0.1 limitations

- Sequential accounting only. Any overlapping event intervals are rejected; parallel, batched, and asynchronous work is not qualified.
- The three optimizer runners do not yet emit this event history. The module API exists, but evaluation-history instrumentation is a separate integration task.
- No candidate artifact-byte resolver, immutable object-store retrieval, verifier signature check, or independent evidence-bundle validator is present yet.
- One successful measurement per candidate ID per run. Replicate aggregation is not yet defined.
- No multi-run empirical attainment surface or cluster-level target-cost inference yet.
- Exact hypervolume target analysis inherits the shared 2–4 objective limit.
- No measured optimization result, benchmark win, PlatEMO parity, or SOTA claim is established.

## 4. Next integration issue

The next engineering task is to add a deterministic evaluation-history hook to NSGA-II, MOEA/D-DE, and SMS-EMOA without changing their proposal/selection behavior. Each optimizer must retain every evaluated decision/objective vector and emit the same candidate IDs, decision digests, objective-evaluation ordinals, and resource/action events.

The resulting histories should then be passed through \`analyze_anytime_history\` and \`verify_anytime_run_receipt\` under the same frozen target plan. Do not retrofit a trace from final_population; it cannot reconstruct candidates discarded before the end of the run.

## 5. Exact-head acceptance commands

\`\`\`bash
cargo fmt --all -- --check
cargo test -p symthaea-architecture-engineering pareto_anytime -- --nocapture
cargo test -p symthaea-architecture-engineering pareto_objectives -- --nocapture
cargo test -p symthaea-architecture-engineering
cargo clippy -p symthaea-architecture-engineering --all-targets -- -D warnings
\`\`\`

Then perform deterministic two-run replay from clean state and cross-check target metrics/reference-front behavior against the pinned benchmark implementation. GitHub workflow conclusions marked skipped are not passes.
