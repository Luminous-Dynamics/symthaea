# MACE Decision-Sufficient Memory reference evaluator v0.1

**Status:** Rust reference implementation authored; exact-head compilation and test execution are pending. The included responses are synthetic conformance fixtures, not model runs or benchmark scores.

This package adds an executable bridge for the episode contract in the parent DSM proposal. Its Cargo manifest declares a nested workspace so it remains isolated from the repository's root workspace globs. It does not replace the existing causal-transfer harness, paired intervention runner, dependency-knockout audit, or decision-sufficiency receipt verifier.

## Commands

Generate deterministic finite-world episodes from the first four confirmatory families:

    cargo run --manifest-path research/mace-decision-sufficient-memory-bench-v0-1/reference-evaluator/Cargo.toml -- generate 4100 256 /tmp/dsm-episodes.jsonl

Split combined episode records into candidate-visible and evaluator-private JSONL streams:

    cargo run --manifest-path research/mace-decision-sufficient-memory-bench-v0-1/reference-evaluator/Cargo.toml -- prepare /tmp/dsm-episodes.jsonl /tmp/dsm-candidate.jsonl /secure/dsm-private.jsonl

A candidate adapter consumes only the candidate stream, conforming to schemas/candidate-input-v0.1.schema.json, and emits one response per episode and arm, conforming to schemas/response-v0.1.schema.json. The response echoes the candidate row's candidate_sha256, carries an arm_binding_sha256 derived from that digest and the declared arm ID, and contains a selected action, separate retrieved_evidence_ids and selected_evidence_ids (the latter must be a subset of the former), observation-request count and candidate-reported resource counters. Detailed phase-attributed cost receipts preserve nanoseconds and measured-versus-estimated storage labels; unavailable top-level counters are nullable rather than silently set to zero. The scorer verifies both bindings against the candidate row, evaluator-private commitment and preregistered arm ID.

Run the Rust unit tests when a Rust 2024 toolchain is available:

    cargo test --manifest-path research/mace-decision-sufficient-memory-bench-v0-1/reference-evaluator/Cargo.toml

Score a balanced set of candidate responses:

    cargo run --manifest-path research/mace-decision-sufficient-memory-bench-v0-1/reference-evaluator/Cargo.toml -- score /tmp/dsm-candidate.jsonl /secure/dsm-private.jsonl /tmp/dsm-responses.jsonl research/mace-decision-sufficient-memory-bench-v0-1/reference-evaluator/fixtures/arms-v0.1.json /secure/dsm-pairs.json /tmp/dsm-report.json

Run the six synthetic positive-control records against the six source episodes already in the parent fixture. The evaluator-private pair manifest binds every episode to a pair block, independent world cluster, query ID and candidate digest, and commits to the arm manifest:

    cargo run --manifest-path research/mace-decision-sufficient-memory-bench-v0-1/reference-evaluator/Cargo.toml -- prepare research/mace-decision-sufficient-memory-bench-v0-1/fixtures/seed-v0.1.jsonl /tmp/dsm-candidate.jsonl /secure/dsm-private.jsonl

    cargo run --manifest-path research/mace-decision-sufficient-memory-bench-v0-1/reference-evaluator/Cargo.toml -- score /tmp/dsm-candidate.jsonl /secure/dsm-private.jsonl research/mace-decision-sufficient-memory-bench-v0-1/reference-evaluator/fixtures/reference-qualified-responses-v0.1.jsonl research/mace-decision-sufficient-memory-bench-v0-1/reference-evaluator/fixtures/arms-v0.1.json research/mace-decision-sufficient-memory-bench-v0-1/reference-evaluator/fixtures/pairs-v0.1.json /tmp/dsm-fixture-report.json

## Evaluator semantics

- Every candidate response must resolve to a known episode and an allowed action.
- The evidence and observation budgets are enforced before scoring. Retrieved IDs are validated and scored separately from decision-support IDs; selected IDs must be drawn from the retrieved set. The response arm IDs must match the preregistered arm manifest; an arm cannot disappear because its output is missing. Every episode must also appear exactly once in an evaluator-private pair/cluster manifest whose candidate digest matches the candidate stream and whose arm-manifest digest matches the exact arm manifest bytes.
- Each episode has exactly one response per arm declared in a separate preregistered arm manifest. Undeclared arms, missing and duplicate episode/arm pairs are rejected; no arm can silently drop hard episodes.
- Candidate-visible and evaluator-private inventories must match exactly. The candidate row and private record commit to a domain-separated SHA-256 digest of the exact identity-bound typed candidate projection; the response must echo the same digest. The private stream has its own evaluator-only schema at schemas/evaluator-private-v0.1.schema.json. Modified or mismatched candidate/response inputs are rejected. This binds the response record to the presented input and declared arm to prevent accidental cross-arm relabelling; it is not a signature, response-payload attestation or proof that the adapter reasoned from the input.
- Knowledge time is explicit: an event is not available before observed_at_ms. Valid time is half-open: valid-from is inclusive and valid-until is exclusive. A supersession/retraction only invalidates its parent once that change has itself been observed and reached its valid-from time; later observations cannot leak into earlier decisions.
- Superseded or retracted parent records are stale at query time. A contradiction edge does not itself erase its parent; retrieving contradictory records is measured separately from selecting evidence to support an action, and v0.1 fails decision-sufficiency closed when a conflicting pair appears anywhere in retrieved context because the response protocol does not yet carry an explicit conflict-resolution receipt.
- Untrusted and evaluator-labelled forbidden records are inadmissible. Source-family and upstream-root identities are reported separately from event counts; repeated events that share one root contribute one distinct root. In v0.1 these are benchmark-supplied identifiers, not authenticated attestations or proof of statistical independence. The evaluator does not interpret similarity, authority tiers or uncertainty as proof of truth.
- Retrieval precision and selected-evidence precision are separate. Required-evidence recall, stale/forbidden/untrusted retrieval, invalid selected evidence, unresolved contradiction in retrieved context, regret, gain/worsening versus the declared reference action, decision sufficiency and resource receipts are reported independently.
- The episode's no_memory_action is a task-authored reference action, not a measured memory-system baseline. The report labels this comparison as a declared reference only. To measure true paired system impact, execute a real no-memory arm and list its ID in the arm manifest's optional no_memory_arm_id field; the report then emits per-arm matched gain and harm rates against that executed no-memory arm.

For a real paired comparison, the arm manifest can declare, for example:

    {
      "schema_version": "mace_dsm_arm_manifest_v0_1",
      "expected_arm_ids": ["no_memory", "mace_memory"],
      "no_memory_arm_id": "no_memory"
    }

The scorer rejects a baseline ID not present in expected_arm_ids and reports matched per-episode gain, worsening rate, harm count, improvement count and ties for each other arm. The included fixture manifest intentionally omits this field because its sole "synthetic-qualified-control" row is not a no-memory agent execution.

The evaluator-private pair manifest is required for every scoring run. It maps each episode to a unique pair block and query ID, groups episodes by independent world cluster, binds each row to its candidate digest, and commits to the exact arm-manifest bytes. The report emits per-arm/per-cluster means and cluster-weighted paired gains, cluster harm rate and worst-cluster gain when a real no-memory arm is declared. A cluster ID is an evaluator-provided grouping label, not cryptographic proof of independent sampling; confirmatory inference must verify disjoint world-generation roots and use the existing cluster bootstrap. The six-case fixture has six distinct synthetic clusters; these demonstrate the report shape only and do not provide a statistically meaningful confidence interval. Reuse the existing cluster-bootstrap implementation for confirmatory inference rather than treating these summaries as inferential tests.
- Action loss is compared to the evaluator's minimum-loss action. Matching the no-memory action or full-context action is not itself evidence of correctness.

## Trust and execution boundary

The prepare command creates two files but does not enforce a process boundary. For any candidate evaluation, the candidate process must run under a separate OS identity or container with read access only to the candidate-visible file and its own outputs. The sealed private file and scoring process must remain inaccessible to the candidate. Do not put the private stream in a shared working directory, candidate prompt, tool mount, debug log or telemetry collector.

The SHA-256 binding checks content consistency; it is not a signature, custody proof, independent authority attestation or evaluator-secrecy guarantee. A serious external challenge should move the scorer, private labels, seed derivation and scoring key material to evaluator-controlled custody before reporting an official result.

Cost fields are candidate-reported receipts. Unknown operation counts, bytes or latency remain null; aggregation reports both the number of rows with a reported value and the sum of reported values, never treating missing values as zero. A detailed receipt can retain phase timings (memory selection, index build, retrieval, evidence resolution/verification, policy application, unattributed coordinator, total wall), token/model/tool-call counts, measured-or-estimated index/sidecar byte labels and explicit unmeasured stages. The evaluator does not independently measure CPU, memory, energy, storage, network, latency or monetary cost. Do not describe absent latency as zero or convert operation counters into measured cost.

## Known limitations

- The first deterministic generator covers temporal drift, contradiction, revocation and insufficient evidence. The original six seed episodes additionally contain poisoning and compression examples; the current generator does not yet generate those families.
- The current episode schema has one scope per episode, not per source event. Cross-tenant isolation and event-level authorization require a later schema revision before they can be qualified. The paired runner's richer `Experience` source-root and causal qualification metadata cannot be reconstructed from retrieved event IDs alone; an adapter must receive the candidate-visible source-event projection, not invent these fields.
- Source authority, required/forbidden evidence and loss tables are synthetic benchmark labels. Real-world qualification needs the independent trust and custody chain.
- Uncertainty ppm is preserved but not calibrated and is not used as a probability.
- Observation requests are scored as actions and budget use; no physical or external probe is executed.
- This version is a response scorer, not a fully autonomous memory-system runner. It does not ingest candidate events into a production store or drive a downstream action environment.
- There is no schema-validation dependency in this package. Use a Draft 2020-12 validator on the episode, candidate-input, evaluator-private, response, arm-manifest and pair-manifest schemas as a separate release check.

## Tests

Eight authored unit tests cover:
- half-open valid-time windows and future-observation exclusion;
- observed supersession/retraction and rejection of untrusted lifecycle events;
- untrusted source evidence rejection;
- cyclic and duplicate source ancestry rejection;
- deterministic generation across the four confirmatory families;
- contradiction in retrieved context cannot silently pass decision sufficiency;
- stale retrieval separated from selected decision support, including selected-subset enforcement;
- candidate/arm digest mismatch rejection and the distinction between regret and a task-authored reference-action comparison.

No Rust compilation, Cargo test, rustfmt, Clippy, CI pass or benchmark result is claimed until those checks have executed on the exact commit head.

## Composition with the MACE programme

- Parent DSM protocol and seed fixtures: research/mace-decision-sufficient-memory-bench-v0-1/
- Paired causal-memory interventions and phase-attributed costs: PRs #13836 and #13869. Their output contract is mapped in [MACE DSM paired-intervention adapter contract v0.1](../../../../docs/research/MACE_DSM_PAIRED_INTERVENTION_ADAPTER_V0_1.md). This document specifies a fail-closed adapter path but does not claim an executable adapter or full interoperability.
- Dependency-aware evidence knockouts and cluster-level reduction: existing counterfactual-evidence research branch. Reuse it through a versioned adapter rather than implementing another Shapley or knockout engine.
- Claim-relative invalidation, projection rebuild and store metamemory summaries: PRs #13849, #13876 and #13878. These provide complementary lifecycle primitives, not a DSM score.
