# MACE Decision-Sufficient Memory reference evaluator v0.1

**Status:** Rust reference implementation authored; exact-head compilation and test execution are pending. The included responses are synthetic conformance fixtures, not model runs or benchmark scores.

This package adds an executable bridge for the episode contract in the parent DSM proposal. It does not replace the existing causal-transfer harness, paired intervention runner, dependency-knockout audit, or decision-sufficiency receipt verifier.

## Commands

Generate deterministic finite-world episodes from the first four confirmatory families:

    cargo run --manifest-path research/mace-decision-sufficient-memory-bench-v0-1/reference-evaluator/Cargo.toml -- generate 4100 256 /tmp/dsm-episodes.jsonl

Split combined episode records into candidate-visible and evaluator-private JSONL streams:

    cargo run --manifest-path research/mace-decision-sufficient-memory-bench-v0-1/reference-evaluator/Cargo.toml -- prepare /tmp/dsm-episodes.jsonl /tmp/dsm-candidate.jsonl /secure/dsm-private.jsonl

A candidate adapter consumes only the candidate stream, conforming to schemas/candidate-input-v0.1.schema.json, and emits one response per episode and arm, conforming to schemas/response-v0.1.schema.json. The response contains a selected action, resolved evidence identifiers, observation-request count and candidate-reported resource counters.

Score a balanced set of candidate responses:

    cargo run --manifest-path research/mace-decision-sufficient-memory-bench-v0-1/reference-evaluator/Cargo.toml -- score /tmp/dsm-candidate.jsonl /secure/dsm-private.jsonl /tmp/dsm-responses.jsonl /tmp/dsm-report.json

Run the six synthetic positive-control records against the six source episodes already in the parent fixture:

    cargo run --manifest-path research/mace-decision-sufficient-memory-bench-v0-1/reference-evaluator/Cargo.toml -- prepare research/mace-decision-sufficient-memory-bench-v0-1/fixtures/seed-v0.1.jsonl /tmp/dsm-candidate.jsonl /secure/dsm-private.jsonl

    cargo run --manifest-path research/mace-decision-sufficient-memory-bench-v0-1/reference-evaluator/Cargo.toml -- score /tmp/dsm-candidate.jsonl /secure/dsm-private.jsonl research/mace-decision-sufficient-memory-bench-v0-1/reference-evaluator/fixtures/reference-qualified-responses-v0.1.jsonl /tmp/dsm-fixture-report.json

## Evaluator semantics

- Every candidate response must resolve to a known episode and an allowed action.
- The evidence and observation budgets are enforced before scoring.
- Each episode has exactly one response per declared arm. Missing and duplicate episode/arm pairs are rejected; no arm can silently drop hard episodes.
- Candidate-visible and evaluator-private inventories must match exactly. The private record commits to a domain-separated SHA-256 digest of the exact typed candidate projection; a modified candidate input is rejected.
- Valid time is half-open: valid-from is inclusive and valid-until is exclusive.
- Superseded or retracted parent records are stale at query time. A contradiction edge does not itself erase its parent; it is measured as a conflict requiring policy handling.
- Untrusted and evaluator-labelled forbidden records are inadmissible. The evaluator does not interpret similarity, authority tiers or uncertainty as proof of truth.
- Required-evidence recall, valid-evidence precision, stale/forbidden/untrusted use, unresolved contradiction co-retrieval, regret, memory gain, memory-induced harm, decision sufficiency and resource counters are reported independently.
- The no-memory action is an evaluator-defined paired baseline. Positive memory gain means lower loss than that baseline; negative gain means memory worsened the decision.
- Action loss is compared to the evaluator's minimum-loss action. Matching the no-memory action or full-context action is not itself evidence of correctness.

## Trust and execution boundary

The prepare command creates two files but does not enforce a process boundary. For any candidate evaluation, the candidate process must run under a separate OS identity or container with read access only to the candidate-visible file and its own outputs. The sealed private file and scoring process must remain inaccessible to the candidate. Do not put the private stream in a shared working directory, candidate prompt, tool mount, debug log or telemetry collector.

The SHA-256 binding checks content consistency; it is not a signature, custody proof, independent authority attestation or evaluator-secrecy guarantee. A serious external challenge should move the scorer, private labels, seed derivation and scoring key material to evaluator-controlled custody before reporting an official result.

Cost fields are candidate-reported counters. The evaluator does not independently measure CPU, memory, energy, storage, network, latency or monetary cost. Latency is aggregated only when present. Do not describe absent latency as zero or convert operation counters into measured cost.

## Known limitations

- The first deterministic generator covers temporal drift, contradiction, revocation and insufficient evidence. The original six seed episodes additionally contain poisoning and compression examples; the current generator does not yet generate those families.
- The current episode schema has one scope per episode, not per source event. Cross-tenant isolation and event-level authorization require a later schema revision before they can be qualified.
- Source authority, required/forbidden evidence and loss tables are synthetic benchmark labels. Real-world qualification needs the independent trust and custody chain.
- Uncertainty ppm is preserved but not calibrated and is not used as a probability.
- Observation requests are scored as actions and budget use; no physical or external probe is executed.
- This version is a response scorer, not a fully autonomous memory-system runner. It does not ingest candidate events into a production store or drive a downstream action environment.
- There is no schema-validation dependency in this package. Use a Draft 2020-12 validator on the episode, candidate-input and response schemas as a separate release check.

## Tests

Authored unit tests cover:
- half-open valid-time boundaries;
- the distinction between contradiction and supersession/retraction;
- fail-closed handling of untrusted evidence;
- seeded deterministic generation and first-four-family coverage;
- separation of regret from memory-induced harm.

No Rust compilation, Cargo test, rustfmt, Clippy, CI pass or benchmark result is claimed until those checks have executed on the exact commit head.

## Composition with the MACE programme

- Parent DSM protocol and seed fixtures: research/mace-decision-sufficient-memory-bench-v0-1/
- Paired causal-memory interventions and phase-attributed costs: PRs #13836 and #13869. They remain distinct scenario and adapter contracts; this package does not silently claim interoperability with their JSONL outputs.
- Dependency-aware evidence knockouts and cluster-level reduction: existing counterfactual-evidence research branch. Reuse it through a versioned adapter rather than implementing another Shapley or knockout engine.
- Claim-relative invalidation, projection rebuild and store metamemory summaries: PRs #13849, #13876 and #13878. These provide complementary lifecycle primitives, not a DSM score.
