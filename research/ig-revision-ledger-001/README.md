# IG-REVISION-LEDGER-001 — evidence-carrying revision events

**Status:** synthetic fixture design / research proposal; no model simulation executed  
**Owner:** [Symthaea #11340](https://github.com/Luminous-Dynamics/symthaea/issues/11340)  
**Parent experiment:** [#11320](https://github.com/Luminous-Dynamics/symthaea/issues/11320)

## Purpose

A final answer does not show how a position changed. The revision ledger records the observable transition, evidence delta, delivered-message exposure, agent-reported explanation and any independently estimated matched-pair effect as separate information.

It does **not** inspect hidden model states or assert human-like beliefs.

## Four evidence layers

1. **Observed transition:** before/after elicited output-state digests and structured claim/stance deltas.
2. **Evidence availability:** exact source/evidence IDs available before and newly exposed, unavailable, or invalidated before the next response.
3. **Agent self-report:** an optional reason code and explanation digest. This is descriptive only; it does not establish cause.
4. **Experimentally estimated effect:** a separately bound matched treatment/control run pair, equal commitments for ten shared input dimensions, a distinct declared intervention factor, pair-validator identity/receipt, analysis-plan digest and effect-estimate receipt. A single run cannot establish a causal effect.

The Rust qualifier recomputes each state commitment as SHA-256 over canonical compact JSON (UTF-8, sorted object keys, with only `state_sha256` removed). A state cannot be changed while retaining a plausible-looking digest. `Changed` requires an actual semantic state change; `NoChange` requires identical state content as well as identical commitments. Delivered-message records bind sender, recipient, channel event, communication graph, payload and ordered delivery sequence; the recipient must match the AgentInstanceId whose revision is recorded.

The Rust qualifier refuses a `MatchedPairEstimate` without a pair ID, distinct treatment/control run identities, equal treatment/control hashes for task, model, both prompts, evidence, communication graph, budget, decoding, RNG and evaluator, a distinct registered varied-factor identity, an independent pair-validator identity/receipt, and analysis/effect digests. The receipt is a required commitment, not self-authenticating proof; its verifier and underlying run manifests remain external evidence. It never derives a causal label from the agent's own reason code.

## Typed event

The event schema binds:
- revision, run, scenario, AgentInstanceId and optional PrincipalId references;
- stage and transition class (Changed, NoChange, Abstained, Missing, Invalid);
- previous/next state hashes, stance, uncertainty, claims and evidence IDs;
- claim delta (added, removed, retained, contradicted, retracted);
- evidence delta (available before, newly available/unavailable, invalidated);
- claim-to-evidence references;
- delivered message IDs, sender/recipient AgentInstanceIds, CommunicationGraph/channel-event hashes, payload hashes and strictly ordered delivery sequence;
- agent-reported reason code, separate from external oracle outputs;
- task truth profile (measurement-only, synthetic known-answer, open-world, normative pluralism, social claim integrity);
- causal-attribution state and matched-run identities.

Delegation graph semantics remain separate: #11248 owns mandate/delegation edges; #11337 owns AgentInstanceId message channels and delivery. Neither is inferred from the other.

## Fixtures

The v0.1 corpus contains 22 synthetic cases:
- Ten valid fixtures: repeat/no-change, evidence-supported revision, social exposure without evidence, abstention under uncertainty, matched-pair estimate, normative pluralism, missing measurement, source correction, high self-reported confidence without new evidence, and a confidence value bound to external calibration references.
- Twelve negative controls: false Changed with identical state hashes, Missing with a post-state, causal estimate without a pair, overlapping claim deltas, delivered message without channel event, missing answer-space identity, open-world correctness score, same-run treatment/control, a mismatched shared-input digest in a causal pair, calibrated confidence without profile/receipt, and out-of-range probability confidence, and state-content tampering under a valid-looking digest.

Truth profiles are assigned by fixture semantics: only tasks with a frozen synthetic answer space use `SyntheticKnownAnswer`; measurement, social-claim-integrity, open-world and normative-pluralism controls do not inherit a correctness oracle by default.

## Confidence boundary

The state schema optionally records confidence as `NotProvided`, `SelfReported` or `Calibrated`. Self-reported values require an explicit scale and remain distinct from evidence. A Calibrated value must reference an external calibration-profile digest and receipt digest; the receipt is a pointer/commitment, not proof by itself. The qualifier checks only structural presence and [0,1] probability bounds. It does not calculate Brier score, ECE, calibration slope, held-out performance or transfer validity. Those are owned by [IG-CONFIDENCE-CALIBRATION-001 #11367](https://github.com/Luminous-Dynamics/symthaea/issues/11367).

Confidence does not grant vote weight, authority, eligibility, moderation priority or credibility.

Negative controls are expected to be rejected; they are not failed empirical experiments.

## Run the qualifier

```bash
cargo fmt --manifest-path research/ig-revision-ledger-001/qualifier/Cargo.toml --check
cargo test --manifest-path research/ig-revision-ledger-001/qualifier/Cargo.toml
cargo run --manifest-path research/ig-revision-ledger-001/qualifier/Cargo.toml -- \
  research/ig-revision-ledger-001/REVISION_LEDGER_FIXTURES_V0_1.json \
  research/ig-revision-ledger-001/REVISION_LEDGER_FIXTURES_V0_1.schema.json \
  research/ig-revision-ledger-001/REVISION_LEDGER_RECORD_V0_1.schema.json
```

The qualifier recursively audits the complete schema definitions for unsupported keywords and applies the declared bounded schema subset to the manifest and every event record. It recomputes canonical state-content SHA-256 commitments, enforces semantic transition consistency, claim-set deltas, evidence availability and invalidation, claim-to-evidence state binding, unique event IDs, sender/recipient/channel delivery provenance and order, confidence-report boundaries, and matched-pair common-input / registered varied-factor identities. SHA-256 fields require canonical lowercase hexadecimal. This is a deliberately bounded JSON-Schema subset, not a complete JSON Schema implementation; it is not a model runner, causal estimator, confidence calibrator, or human-subject measurement tool.

## Claim ceiling

A structural PASS means only that the checked-in synthetic event fixtures conform to the declared ledger contract. It does not show that a model revised its stance, that an intervention caused a change, or that human groups behave similarly. Human studies require separate ethics review, informed consent, privacy protections, preregistration and independent evaluation.
