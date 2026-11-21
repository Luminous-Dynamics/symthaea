# MEL-BENCH-002 — matched external comparison run contract v1

**Date:** 21 November 2026  
**Parent programme:** [#10417 — reproducible external SOTA and industry comparison ledger](https://github.com/Luminous-Dynamics/symthaea/issues/10417)  
**Capability snapshot:** [PR #13088 — dated external capability ledger](https://github.com/Luminous-Dynamics/symthaea/pull/13088)  
**Schema:** \`benchmarks/mel/schemas/matched-comparison-run-v1.schema.json\`  
**Example:** \`benchmarks/mel/matched-comparison-run-v1.example.json\`  
**Validator:** \`tools/mel-matched-comparison-validator\`

## Objective

A competitive claim is only meaningful when the systems have been evaluated on the same task, input, split, budget, controls and scorer. This contract freezes the comparison conditions and preserves every attempt and judgment so that a reported win cannot be manufactured by changing denominators, retry counts, prompts, renderers or evaluation scope.

This is an evidence format and validator. The checked-in example is a **frozen plan**, not an executed study or a Muse result.

## Core design

### 1. Benchmark identity is immutable

A run binds:
- benchmark family and exact version;
- task-definition digest and scorer digest;
- official split identity/digest where applicable;
- rights-cleared dataset split manifest and usage receipt;
- comparability-group label.

Do not compare:
- instrumental audio preference Elo to vocal Elo;
- symbolic 0–5 human ratings to audio Elo;
- source/target edit preservation to generation preference;
- a live MIDI agent to an audio-only stream API on note-level capabilities it does not expose;
- synthetic reciprocity to human reciprocal influence.

Each comparator remains in its own native benchmark family.

### 2. Model and adapter identity is not a display name

Each arm records:
- system and model version;
- adapter version;
- modality and execution surface;
- provider-declared or immutable model identity state;
- evidence digest supporting that identity;
- immutable artifact digest where the system can actually expose one;
- control-surface digest;
- per-arm time/call/token/cost budget.

A provider-declared version is not equivalent to an immutable checkpoint. Do not silently claim immutable identity for a closed API. Mark it provider-declared or unavailable, and record the replay limits.

### 3. Every case is paired on one frozen input

Each case has one exact shared input digest, task-instance digest, prompt/instruction digest and seed. All comparison arms receive that same case input under the frozen task contract. A changed prompt, seed schedule, source artifact, renderer, duration target, or control set creates a new case or run version rather than being hidden inside an existing row.

### 4. Attempts are the denominator

For every arm and case, store each attempt separately:
- attempt index;
- completed, timeout, error, abstained, unavailable, or invalid-output status;
- wall time;
- exact output bytes digest and length where output exists;
- cost and provider request identity where available;
- structured error code and optional detail digest.

Failed, partial, timed-out, invalid and abstained attempts consume the declared attempt budget. A retry is a new attempt record. A completed output is not a substitute for a failed prior attempt, and an arm with no observed attempt may not disappear from the denominator.

The frozen retry policy requires \`failed_attempts_consume_budget=true\`.

### 5. Human judgments stay blinded

Judgment rows use:
- exact left/right output digests;
- pseudonymized judge digest;
- randomized display-order seed;
- judgment protocol digest;
- left, right, tie, neither, invalid or skip outcome.

The judgment record does not include model/arm names. The evaluator can join outputs to arms after unblinding using their output digests. Keep participant identity and sensitive raw data outside public benchmark artifacts; the pseudonym digest is a join key under the study's approved data-governance plan, not authentication or consent proof.

### 6. Aggregate metrics expose their denominator

Every aggregate stores metric name, value, unit, numerator, denominator and cohort scope. Optional uncertainty bounds must be supplied as a pair, and a declared confidence level must be in (0,1). Metric denominators are not silently inferred from the number of successful outputs.

## Semantic invariants enforced by the Rust validator

- Strict top-level and nested field sets.
- At least two unique comparison arms and at least one unique case.
- SHA-256 syntax and exact 40-character source revision.
- Exact model identity state; \`immutable_verified\` requires an immutable artifact digest.
- All case attempts refer to declared arms.
- Attempt identities are unique per arm/case and bounded by the frozen retry policy.
- Executed/analyzed cases have at least one attempt for every comparison arm.
- Frozen plans contain no execution attempts, judgments or aggregate metrics.
- Failed/unavailable/abstained/invalid attempts require a reason code; they are not treated as omitted observations.
- Pairwise judgments reference two distinct, completed output digests from that exact case.
- Metric IDs are unique, denominators are positive, and numerators cannot exceed denominators.
- An analyzed run must contain aggregate metrics and an analysis timestamp.

The JSON Schema validates structural shape. The Rust validator enforces cross-record references and state-dependent invariants that are not sufficiently expressed by local JSON Schema constraints.

## Run locally

This validator is a standalone Rust 2024 package and does not load the monorepo workspace:

\`\`\`bash
cargo generate-lockfile --manifest-path tools/mel-matched-comparison-validator/Cargo.toml
cargo test --locked --manifest-path tools/mel-matched-comparison-validator/Cargo.toml
cargo run --locked --manifest-path tools/mel-matched-comparison-validator/Cargo.toml
\`\`\`

The default invocation validates the frozen example under \`benchmarks/mel/matched-comparison-run-v1.example.json\`. It deliberately reports that run as a plan, not an executed comparison.

## First matched experiments to build on this contract

1. **Instrumental audio preference:** a common rights-cleared prompt set, equal requested duration, explicit generation and retry budgets, fixed loudness/format normalization, blind pairwise listening, and exact failure retention. Compare only within the AA instrumental task family when a sufficiently matched access path exists.
2. **Symbolic continuation:** identical source Score, continuation window, constraints, seed policy, renderer, and candidate count. Score symbolic validity and human-rated coherence/structure/creativity separately.
3. **Music editing:** paired source/target audio and score, exact operation intent, target occurrence, protected content and source lineage. Report target agreement, preservation, text compatibility, rights and cost independently.
4. **Live improvisation:** same MIDI instrument/controller, device, clock calibration, tempo/meter/task schedule and response deadline. Report actual p50/p95/p99 and beat-relative phase error; distinguish system processing from MIDI/audio/physical output.
5. **Reciprocal interaction:** pair the state-isolated RMI-Bench intervention records with real policy identity and blinded musical relevance evidence. Synthetic positive-control results are only a validator check.

## Claim ceiling

Passing the validator means the run record is structurally and semantically complete under this v1 contract. It does not prove that:
- the provider actually received the declared input or budget;
- a provider's self-reported model version is immutable;
- an output is musically good, rights-cleared or instruction-compliant;
- a judge has the declared expertise or has not colluded;
- a preference result proves editing correctness, reciprocity, authorship or agency;
- one benchmark result generalizes outside its declared dataset, split and task.

The next external-comparison milestone is to implement capability-matched adapters and collect an actual rights-cleared pilot run. No system ranking is claimed by this contract PR.
