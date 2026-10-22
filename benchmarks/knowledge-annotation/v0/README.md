# Knowledge annotation contract v0

**Status:** schema and synthetic qualification fixtures only. No expert annotation or model score is claimed.

This contract represents evaluator judgments at the **claim and interpretation level**, not a single answer string. It is intended to support blinded independent reviews, explicit adjudication, disagreement preservation, calibration and error analysis across evidence-grounded knowledge tasks.

## What it records

- **Answerability:** whether the supplied evidence can answer the question, partially answer it, cannot answer it, cannot be assessed, or is prohibited for the requested use.
- **Abstention:** whether a model's abstention is appropriate, excessive, insufficient, or not applicable.
- **Valid alternatives:** zero or more independently defensible, source-scoped interpretations. More than one answer can be acceptable without implying that all interpretations are interchangeable or equally evidenced.
- **Claim-level support:** supported, partly supported, unsupported, contradicted, unverifiable from the pack, or tied to the wrong source.
- **Epistemic lane:** source-attested, tradition-internal, scholarly interpretation, comparative hypothesis, empirical claim, procedural statement, or unknown.
- **Criticality and failure tags:** source/witness mismatch, false lineage, variant fusion, universalization, unsupported confidence, causal overreach, rights violation, and other explicit failure modes.
- **Confidence:** model-stated confidence is distinct from reviewer confidence. Neither substitutes for an evidence-based label.
- **Review trail:** independent annotations, source-family scope, and separate disagreement/adjudication notes.

## Annotation protocol

1. Freeze the case, evidence-pack digest, model output and rubric version before annotation.
2. Reviewers independently label the answer without seeing other reviewers' judgments. Use pseudonymous reviewer identifiers, not real names or contact details.
3. Mark each atomic factual or interpretive claim separately. Split a compound claim when one clause is supportable and another is not.
4. Bind each support label to evidence/source locators. If evidence is unavailable, distinguish unsupported from contradicted and not-verifiable.
5. Record every defensible alternative with its scope and supporting sources; do not use exact-match as the only grading rule.
6. Keep disagreements. Report raw agreement and a suitable chance-corrected reliability measure; adjudicate only with a written rationale and preserve the original independent labels.
7. Mark rights-prohibited cases explicitly. A public URL or visible metadata is not a content-use grant.
8. Keep synthetic contract fixtures out of expert gold sets and final test splits.

## Synthetic qualification fixtures

The schema and these controls are also run through the focused Rust harness as `--kind annotation`. Contract checks reject positive source-grounded labels without evidence references, duplicate claim/alternative IDs, rights violations below release-blocker severity, invalid independent-review trails, and content leakage in rights-prohibited abstentions. The separate `score-annotation-agreement` binary computes raw exact agreement and nominal Krippendorff alpha for answerability, abstention, support label, epistemic lane, and severity. Single-rater units are excluded and counted; alpha is left undefined where category variation is absent. The CI fixture must remain marked `qualification_only` and its reliability numbers are not expert-reliability results.

`synthetic-fixtures.jsonl` contains five controls covering multi-answer interpretation, two intentionally disagreeing independent labels, appropriate abstention when the evidence pack is insufficient, and rights-based abstention. The source IDs used by those fixtures are synthetic placeholders, not catalog entries.

The schema validates record shape; the Rust harness adds identity and per-record integrity checks. This does not assess scholarly truth or reviewer competence. Real annotations should be added only through a documented expert workflow with rights-reviewed evidence.
