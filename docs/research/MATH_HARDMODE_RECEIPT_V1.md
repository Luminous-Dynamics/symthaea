# HardModeReceiptV1 — staged answer-to-proof evaluation

**Status:** first executable synthetic contract. No external benchmark task or model result is claimed.

## Purpose

Hard-mode theorem proving requires a system to discover the answer before constructing a formal proof. A single pass/fail label hides several failure modes. This receipt records answer discovery, candidate formalization, finite semantic checking, candidate countermodel validity, optional formal proof submission and optional ProofPlanV1 artifact as separate outcomes.

The candidate partition is model-visible. The evaluator-only partition is post-hoc and must not be included in prompts, retrieval contexts or candidate artifacts. A candidate-only digest is computed over the exact visible payload.

## Answer-provided versus no-answer

Task variants are explicit:

- answer_provided: visible_answer must be populated and is included in the prompt digest.
- no_answer: visible_answer must be null; the answer must be discovered from the problem text.

Problem and prompt digests are recomputed from canonical preimages, preventing a receipt from relabelling an answer-provided task as no-answer while leaving its prompt unchanged.

## Independent outcomes

- answer_correct evaluates the discovered answer only.
- formalization_matches_target evaluates statement fidelity only.
- finite_semantics is computed for supported propositional sequents (maximum 16 atoms): countermodel, no countermodel found, unknown, or not applicable.
- candidate_countermodel is scored against the exact candidate sequent. An invalid candidate witness remains visible as an invalid response rather than invalidating the whole run record.
- proof_status records an independent proof-checker result. not_run carries no checker evidence. A completed proof check binds the task, source revision, statement digest, proof digest, toolchain, environment manifest, result artifact and evidence references.
- proof_plan_submission is only an artifact receipt; it does not imply reviewer acceptance or human comprehension.

## Synthetic example and claim ceiling

The fixture asks whether p → q entails q → p. The candidate answers no and supplies p=false, q=true. The finite semantic oracle validates the witness. The evaluator records answer and formalization adjudication as true but Lean proof verification as not_run.

This is not an LLM evaluation, Lean proof, external benchmark result or measured CIRR score. It tests receipt structure, candidate/evaluator separation, exact prompt variants, canonical digests and finite propositional semantics.

## Commands

    cargo test --locked --manifest-path scripts/research/proof-plan-contract/Cargo.toml
    cargo run --locked --manifest-path scripts/research/proof-plan-contract/Cargo.toml --bin hard-mode -- docs/research/MATH_HARDMODE_SYNTHETIC_V1.json

A successful run validates the contract and finite semantic layer only. It does not verify an external Lean proof or demonstrate model capability.
