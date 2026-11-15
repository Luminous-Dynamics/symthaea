# sympoetic-bench

Rust validation and measurement core for SympoeticBench v0.1, including a proposal-only agent decision protocol guard.

This package is intentionally small and deterministic. It does not call a model, run a human-subject study, or assign creativity scores from an LLM judge. It validates benchmark inputs and evidence records, and exposes metrics whose assumptions are explicit. The 17-scenario development corpus includes partner coordination, adaptive intervention, capability amplification, epistemic resilience, and persistent-team scenarios.

## Commands

From the Symthaea repository root:

```bash
# Validate every JSONL scenario (typed validation + corpus digest)
cargo run -p sympoetic-bench -- validate-corpus \
  --input docs/benchmarks/sympoetic/SEED_CORPUS.v0.1.jsonl

# Validate a multi-cycle team trajectory
cargo run -p sympoetic-bench -- validate-trajectory --input path/to/team-trajectory.json

# Validate a recorded interaction trace
cargo run -p sympoetic-bench -- validate-trace --input path/to/trace.json

# Validate scorecard ranges, F1 consistency, evidence digests, and hard gates
cargo run -p sympoetic-bench -- validate-scorecard --input path/to/scorecard.json

# Digest a file without modifying it
cargo run -p sympoetic-bench -- digest --input path/to/artifact
```

## Validation rules

- Scenario IDs, domains, difficulty, splits, events, baseline conditions, consent scopes, rubric anchors, and required arrays are checked.
- Trace event IDs are unique, sequence numbers are strictly increasing, elapsed time is non-decreasing, and causal parents must refer to earlier events.
- Participant consent grant/revocation events must be authored by the human participant.
- Raw/consented content references cannot be stored under `none` or `revoked` consent scope.
- A tool result with an unknown outcome blocks subsequent tool invocation until an independently observed reconciliation event references it and includes a digest of the reconciled state.
- A state-changing tool result with an observed success/failure must include a postcondition digest.
- Agent decisions are checked against the current run/turn, role-filtered tool catalogue, granted action set, and remaining tool budget.
- Every proposed tool call must receive its matching observation before another decision is accepted.
- An unknown tool outcome blocks the agent decision loop until a matching, independently observed reconciliation event arrives.
- Scorecard dimensions are range-checked; attribution F1 must agree with precision and recall.
- Hard-gate failures are non-compensatory. A scorecard cannot mark hard gates as passing while any failure counter is non-zero.

## Digest conventions

- File and payload digests use SHA-256 over the exact file or payload bytes supplied by the recorder.
- `trace_digest`, when present, is SHA-256 over the trace JSON parsed into a `serde_json::Value`, with the `trace_digest` key removed, then serialized by `serde_json` (deterministic map-key ordering). This is the v0.1 canonicalization rule; it is not claimed to be RFC 8785.
- Scenario, trace, artifact, prompt, and environment digests in a scorecard identify the exact inputs used. The scorer revision must identify the scoring implementation.

The epistemic resilience track also reports confidence calibration (Brier/ECE), metacognitive sensitivity (AUC), selective accuracy, human-first attempts, AI core-step substitution, and human verification rates. These measures are separate; no single cognitive-sovereignty aggregate is asserted.

## Agent decision bridge

The typed protocol contract lives in `agent_protocol.rs`; the normative schemas and design are in `docs/benchmarks/agent-bridge/` and `docs/benchmarks/AGENT_DECISION_PROTOCOL.md`. `AgentProtocolGuard` validates proposals and observations but deliberately does not execute tools. The benchmark adapter remains the authority boundary.

## Persistent team trajectories

The `trajectory` module validates multi-cycle team receipts, prior-cycle digests, team-state continuity, explicit membership/role/authority changes, memory-policy changes, and per-member consent. The CLI command is `validate-trajectory`. This verifies recorded continuity claims; it is not a collective-intelligence estimator.

## Metrics

- `synergy_delta`: joint quality minus the best matched solo quality.
- `collaboration_gap`: best matched solo quality minus joint quality. It is the negative of `synergy_delta` when both come from the same matched comparison.
- `partner_order_sensitivity`: absolute difference between counterbalanced A→B and B→A outcomes.
- `handoff_recovery_rate`: successful independently adjudicated handoffs divided by valid attempts; unavailable when no handoff was attempted.
- `marginal_contribution_impact`: full artifact quality minus quality from an identifiable counterfactual replay without the contribution. Non-identifiable replays return unavailable rather than an invented score.
- `exact_shapley_values`: allocates a bounded outcome across up to 12 contributors using a complete, identifiable (2^n) coalition table; missing coalitions are rejected.
- `participant_capability_amplification`: independent-arm, participant-cluster bootstrap of pre/post learning gains versus a matched control. Delayed retention and transfer use the same estimator with their respective held-out assessment scores.
- `paired_cluster_bootstrap_difference`: seeded percentile bootstrap over one pre-aggregated value per participant cluster. It must not be used to treat conversation turns as independent samples.

## Status and limitations

This is a first executable core, not a complete human-AI benchmark runner. It does not yet provide:

- A model adapter or live agent orchestration.
- The new `agent_turn` protocol is not yet wired into the Symthaea service daemon; this crate validates the contract only.
- Human-subject recruitment, survey delivery, consent management UI, or data retention infrastructure.
- Blind expert artifact rating or inter-rater reliability calculations.
- Full task-level hierarchical models or multi-level cluster bootstrap.
- General counterfactual replay for arbitrary dependent interaction traces.
- A validated psychometric agency scale.

The crate and its focused CI job are authored in this research branch. Compilation and test results remain pending until a Rust runner executes them. Do not report this branch as tested or green until exact-head CI evidence is available.
