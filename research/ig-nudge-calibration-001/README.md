# IG-NUDGE-CALIBRATION-001 v0.1

**Owner:** Symthaea #11704  
**Status:** deterministic synthetic fixture qualification only

## Research question

Can an agent use accurate high-value information, avoid overreacting to accurate low-value information or presentation-only cues, and reject unsupported factual claims?

Robustness means **selective responsiveness**, not blanket resistance to influence.

## External comparator

Cherep, Maes and Singh, *AI Agents Are Sensitive to Nudges*, PNAS 123(25), e2537030123 (June 23, 2026): https://pmc.ncbi.nlm.nih.gov/articles/PMC13291645/

The PNAS study separates information value from generic nudge compliance in a multiattribute sequential-choice task. This synthetic benchmark is not a reproduction of that task.

## Workload

- 42 scenarios in six matched task blocks.
- Seven cue arms per block: no cue, verified high-value preview, verified low-value preview, unverified recommendation, default preselection, salience highlight, unsupported fact claim.
- Rust qualifier validates candidate/oracle separation, evidence provenance, high/low preview semantics, matched blocks and regret calculations.
- No LLM, human or model API was executed.

High-value previews are accurate and change the best option under the candidate-visible decision state. Low-value previews are also accurate but leave the preferred option unchanged. Presentation-only cues add no evidence. Unsupported claims are false under the oracle and carry no evidence reference.

## Local run

```bash
cargo test --manifest-path research/ig-nudge-calibration-001/qualifier/Cargo.toml
cargo run --manifest-path research/ig-nudge-calibration-001/qualifier/Cargo.toml -- \
  research/ig-nudge-calibration-001/IG_NUDGE_CALIBRATION_PROTOCOL_V0_1.json \
  research/ig-nudge-calibration-001/IG_NUDGE_CALIBRATION_PROTOCOL_V0_1.schema.json \
  research/ig-nudge-calibration-001/IG_NUDGE_CALIBRATION_FIXTURES_V0_1.json \
  research/ig-nudge-calibration-001/IG_NUDGE_CALIBRATION_FIXTURES_V0_1.schema.json
```

The validator implements bounded invariants, not full JSON Schema.

## Next stage

Audit and reproduce a subset of PNAS arms where exact source artifacts and terms permit. Then compare pinned models under matched task states, prompts, tool interfaces, information and compute budgets. Report useful-cue uptake, low-value-preview restraint, presentation-only displacement, unsupported-claim acceptance, regret, information cost, latency and compute separately.

## Claim ceiling

Synthetic qualification validates the fixtures and deterministic payoff calculations only. No model behavior, human effects or production safety are established.
