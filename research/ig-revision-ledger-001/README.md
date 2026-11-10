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

The Rust qualifier refuses a `MatchedPairEstimate` without a pair ID, distinct treatment/control run identities, equal treatment/control hashes for task, model, both prompts, evidence, communication graph, budget, decoding, RNG and evaluator, a distinct varied-factor digest, an independent pair-validator identity/receipt, and analysis/effect digests. The receipt is a required commitment, not self-authenticating proof; its verifier and underlying run manifests remain external evidence. It never derives a causal label from the agent's own reason code.

## Typed event

The event schema binds:
- revision, run, scenario, AgentInstanceId and optional PrincipalId references;
- stage and transition class (Changed, NoChange, Abstained, Missing, Invalid);
- previous/next state hashes, stance, uncertainty, claims and evidence IDs;
- claim delta (added, removed, retained, contradicted, retracted);
- evidence delta (available before, newly available/unavailable, invalidated);
- claim-to-evidence references;
- delivered message IDs, CommunicationGraph channel-event IDs and payload hashes;
- agent-reported reason code, separate from external oracle outputs;
- task truth profile (measurement-only, synthetic known-answer, open-world, normative pluralism, social claim integrity);
- causal-attribution state and matched-run identities.

Delegation graph semantics remain separate: #11248 owns mandate/delegation edges; #11337 owns AgentInstanceId message channels and delivery. Neither is inferred from the other.

## Fixtures

The v0.1 corpus contains 17 synthetic cases:
- Eight valid fixtures: repeat/no-change, evidence-supported revision, social exposure without evidence, abstention under uncertainty, matched-pair estimate, normative pluralism, missing measurement and source correction.
- Nine negative controls: false Changed with identical state hashes, Missing with a post-state, causal estimate without a pair, overlapping claim deltas, delivered message without channel event, missing answer-space identity, open-world correctness score, same-run treatment/control, and a mismatched shared-input digest in a causal pair.

Truth profiles are assigned by fixture semantics: only tasks with a frozen synthetic answer space use `SyntheticKnownAnswer`; measurement, social-claim-integrity, open-world and normative-pluralism controls do not inherit a correctness oracle by default.

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

The qualifier checks a bounded event/fixture contract and expected violation oracles. It is not a complete JSON Schema validator, model runner, causal estimator, or human-subject measurement tool.

## Claim ceiling

A structural PASS means only that the checked-in synthetic event fixtures conform to the declared ledger contract. It does not show that a model revised its stance, that an intervention caused a change, or that human groups behave similarly. Human studies require separate ethics review, informed consent, privacy protections, preregistration and independent evaluation.
