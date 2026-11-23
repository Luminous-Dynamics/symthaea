# TheoremSandwichReceiptV1 — finite control pilot

**Status:** executable synthetic control; no model evaluation and no Lean kernel proof.

## Contract

The candidate partition contains source/task identity, model identity, visible assumptions, and the formal statement the candidate proposes. The evaluator-only partition contains a hidden target statement, stronger obligations and required corollaries. The evaluator-only partition is post-hoc only; the model receives candidate_view and never receives hidden gold or witnesses.

Before classifying a candidate, the evaluator checks the integrity of its hidden controls: each stronger obligation S_i must imply the hidden target P, and P must imply each selected corollary C_j. A false hidden bracket is classified as invalid_evaluator_controls; an atom-budget overflow remains unknown. Only after these controls pass does the evaluator test S_i implies candidate_statement and candidate_statement implies C_j.

Classification (after hidden-control validation):
- consistent_with_selected_controls: every finite upper and lower obligation passes.
- candidate_may_be_too_strong: an upper obligation fails while lower controls pass.
- candidate_may_be_too_weak: upper controls pass while a lower corollary fails.
- incomparable_or_translation_mismatch: both sides fail.
- unknown: a finite truth-table atom budget is exceeded.
- invalid_evaluator_controls: the hidden stronger/corollary obligations do not actually bracket their hidden target.

These labels are diagnostic classes over selected obligations, not proof of complete equivalence. A candidate that passes is only consistent with this finite control set; expert semantic review and Lean kernel replay remain separate authorities.

## Synthetic task

The target statement is p implies q. The hidden stronger control is (p implies q) and r, and the hidden corollaries are (p and r) implies q and (p and not r) implies q. The exact candidate p implies q passes all selected obligations. A deliberately over-strengthened candidate (p implies q) and s fails the upper obligation. A deliberately weakened candidate r implies (p implies q) passes the upper control but fails the corollary for not-r.

## Commands

    cargo test --locked --manifest-path scripts/research/proof-plan-contract/Cargo.toml
    cargo run --manifest-path scripts/research/proof-plan-contract/Cargo.toml --bin theorem-sandwich -- docs/research/MATH_PROOF_COMPREHENSION_THEOREM_SANDWICH_V1.json

Successful execution establishes structural validity and finite truth-table outcomes for the synthetic Boolean fixture only. It does not establish model accuracy, general mathematical semantics, LPBench performance, theorem equivalence, or human comprehension.

## Context-indexed theorem semantics and translation review

The theorem-sandwich contract now compares **sequents**, not isolated propositions. A candidate statement is represented as:

\`Γ ⊢ P\`

where \`statement_context\` is the formal list of assumptions and \`statement_formula\` is the conclusion. The finite propositional evaluator gives this sequent the meaning \`(∧ Γ) → P\`. The hidden target has its own \`target_statement_context\`; it is not inferred from the candidate's free-text assumption descriptions.

This prevents two common formalization errors from disappearing into metadata:
- dropping a required assumption can make the candidate proposition too strong;
- adding an unnecessary assumption can make the candidate theorem too weak.

The free-text \`visible_assumptions\` field is explanatory metadata only. It is not used as a logical premise. Each assumption that should affect theorem meaning must appear in the formal context AST.

### Translation review is a separate gate

A machine can check the supplied Boolean AST exactly and still evaluate the wrong translation of the natural-language statement. The evaluator-only partition now records \`translation_assessment\`, \`translation_assessor\` and \`translation_review_refs\`.

- \`aligned\`: eligible for selected-control classification, subject to all other checks.
- \`misaligned\`: classification is \`incomparable_or_translation_mismatch\`, never accepted as a semantic match.
- \`partially_aligned\`, \`unknown\`, or \`not_reviewed\`: classification is \`unknown\`, not a pass.

The translation assessment must be held out of the candidate-visible projection. The synthetic fixture stipulates an exact text/formula pairing only to test the contract; it is not independent mathematical review or model evaluation.

### New adversarial controls

Six tests cover moving an assumption between context and implication, adding an extra assumption, removing a required assumption, explicit text/formula mismatch, unreviewed translation abstention, and malformed context formulas. These demonstrate the new scope semantics on finite propositional examples only.

