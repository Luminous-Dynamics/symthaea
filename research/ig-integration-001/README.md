# IG-INTEGRATION-001 — evidence integration fidelity v0.1

**Owner:** [Symthaea #12775](https://github.com/Luminous-Dynamics/symthaea/issues/12775)  
**Composes with:** [IG-HIDDEN-INFO-001 #12762](https://github.com/Luminous-Dynamics/symthaea/issues/12762)  
**Status:** deterministic synthetic qualifier; no LLM model campaign has been run.

## Why this track exists

The HiddenBench/ICML 2026 result motivates collective information discovery. A separate CSCW Companion '26 exploratory study demonstrates that information surfacing and information integration are distinct capabilities: GPT-OSS and Gemini groups in the tested hidden-profile cells often surfaced most unique facts but still selected the wrong candidate. The paper proposes an explicit candidate-by-attribute synthesis matrix and integration audit.

Paper: Toxtli & Delaflor, *Hidden Profile Decision Making in Multi-Agent LLM Groups*, DOI https://doi.org/10.1145/3785651.3831525

This benchmark separates:
1. **Surfacing:** did the evidence become available to the group?
2. **Integration:** did the synthesizer preserve source, entity, attribute, polarity and diagnostic contribution?
3. **Decision:** did the chosen option follow from the integrated evidence, or should the system abstain?

A transcript containing all relevant information is not proof that the system used it correctly.

## Included

- `INTEGRATION_PROTOCOL_V0_1.md`: experimental protocol and metrics.
- `INTEGRATION_FIXTURES_V0_1.json`: 12 deterministic known-truth integration fixtures, including surfaced-but-not-integrated, duplicate rows, polarity reversal, source tampering, fabricated evidence, candidate misattribution, omitted negative evidence, tie abstention, minority evidence and incomplete-information abstention.
- `qualifier/`: standalone Rust qualifier with unit and negative-control tests.
- `.github/workflows/ig-integration-qualifier.yml`: path-filtered exact-revision workflow.

These fixtures are synthetic structural oracles, not LLM trials, not a reproduction of the CSCW study, and not HiddenBench results.

## Run

```bash
cargo fmt --manifest-path research/ig-integration-001/qualifier/Cargo.toml --check
cargo test --manifest-path research/ig-integration-001/qualifier/Cargo.toml
cargo run --manifest-path research/ig-integration-001/qualifier/Cargo.toml -- \
  research/ig-integration-001/INTEGRATION_FIXTURES_V0_1.json
```

## Measurement contract

- Surfacing is measured from the evaluator's set of unique evidence IDs recorded as delivered. Repeating an atom never increases coverage.
- A matrix row can only reference an evidence atom that was surfaced.
- Source identity must match the source attached to the atom. Invalid source claims or fabricated/unavailable atoms reject the record.
- Candidate, attribute and polarity mapping errors are scored as integration failures, not silently repaired.
- Duplicate matrix rows are counted but only the first row contributes to coverage or ranking.
- Model-supplied signed contributions are compared with evaluator-only synthetic contributions.
- The matrix-implied ranking and selected decision are checked separately from gold-answer correctness.
- If the fixture says decisive evidence is incomplete, the correct behaviour is abstention. A forced choice is not counted as success.
- A valid tied oracle requires abstention rather than arbitrary tie-breaking.

## Existing architecture boundaries

This work does not create a message graph, evidence ledger, debate controller or runtime. It composes with:
- #12762 — hidden-information task and actual evidence discovery.
- #11337/#11360 — communication topology and delivered-message identity.
- #11350 — evidence-carrying revision events.
- #11383 — deliberation action selection and stopping.
- #9431 — independent evaluator and correlated-judge controls.
- #3095 — experiment manifests, receipts and replay.

## Claim ceiling

A passing local qualifier establishes only that the synthetic matrices and their declared negative controls behave as specified. It does not show that any model integrates evidence correctly, that asynchronous communication improves performance, or that Symthaea outperforms an industry baseline. Those claims require actual matched-budget model experiments, pinned artifacts, held-out tasks and independent analysis.
