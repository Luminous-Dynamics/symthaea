# IG-PHANTOM-CONSENSUS-001 v0.1

**Status:** synthetic fixture design / structural qualification only  
**Owning issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11682  
**Research reference:** https://arxiv.org/abs/2611.08971

## Research question

Does a system's estimate of what peers privately judged match the independently recorded private baseline, and do ambiguous public signals cause public decisions to diverge from that baseline?

This is distinct from:
- self-confidence calibration (#11367);
- an agent's own stance transition (#11320);
- evidence-carrying revision provenance (#11340);
- communication topology and actual message delivery (#11337).

This artifact composes with those owners; it does not implement a new agent runtime or graph engine.

## Files

- PHANTOM_CONSENSUS_PROTOCOL_V0_1.json and its schema: four-stage experiment contract, intervention arms, metrics and claim ceiling.
- PHANTOM_CONSENSUS_FIXTURES_V0_1.json and its schema: 12 deterministic synthetic controls, including a phantom-consensus case, transparent recovery controls, fabricated tally, missing record, duplicate identity, abstention/tie and normative-pluralism case.
- qualifier/: standalone Rust structural and oracle validator.
- .github/workflows/ig-phantom-consensus-qualifier.yml: path-filtered exact-revision workflow.

## Local qualification

From the repository root:

cargo test --manifest-path research/ig-phantom-consensus-001/qualifier/Cargo.toml

cargo run --manifest-path research/ig-phantom-consensus-001/qualifier/Cargo.toml -- \
  research/ig-phantom-consensus-001/PHANTOM_CONSENSUS_PROTOCOL_V0_1.json \
  research/ig-phantom-consensus-001/PHANTOM_CONSENSUS_PROTOCOL_V0_1.schema.json \
  research/ig-phantom-consensus-001/PHANTOM_CONSENSUS_FIXTURES_V0_1.json \
  research/ig-phantom-consensus-001/PHANTOM_CONSENSUS_FIXTURES_V0_1.schema.json

The qualifier checks a bounded schema header and explicit structural invariants. It does not claim complete JSON Schema implementation. It computes peer distributions from the recorded baseline, checks probability-vector completeness, calculates multiclass Brier scores, and verifies the expected phantom-consensus oracle. Baseline and intervention payload digests are emitted for reproducible run receipts.

## Measurement semantics

- Each private baseline row is a recorded elicited output, not hidden model state.
- Peer estimates exclude the estimating agent; the denominator includes missing records as UNKNOWN and abstentions as ABSTAIN.
- ABSTAIN and UNKNOWN are retained as explicit categories but are not treated as substantive preferences when identifying a unique plurality.
- A phantom-consensus oracle is true only when an agent's estimated unique peer plurality differs from the unique plurality in the other agents' recorded private baseline. Ties / no unique plurality do not become an artificial majority.
- Verifiable task accuracy is separate from peer-belief accuracy.
- Normative-pluralism tasks have no gold preference. The benchmark checks representation, not agreement with a preferred policy.
- Transparency arms reveal genuine recorded private judgments but never the answer key.

## Claim ceiling

A successful synthetic qualification establishes only that these 12 fixture contracts and bounded oracles behave as declared. No LLM agents were executed by this corpus. No PHANTOM-Bench reproduction, human effect, democratic legitimacy, public-opinion simulation or deployment claim is made. Model evaluation requires exact artifact/method review, matched baselines, pinned prompts/models, randomized treatment blocks, multiple seeds, held-out tasks and independent analysis.
