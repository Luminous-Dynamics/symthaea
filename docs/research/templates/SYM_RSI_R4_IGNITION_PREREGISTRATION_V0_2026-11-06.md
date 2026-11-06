# Symthaea R4 ignition preregistration template V0

**Status:** Unfilled template. It is not a preregistration, receipt, test result or approval.  
**Research owner:** [#10497 — R4 meta-improvement](https://github.com/Luminous-Dynamics/symthaea/issues/10497)  
**Template work:** [#10749 — operational R4 preregistration](https://github.com/Luminous-Dynamics/symthaea/issues/10749)

## Purpose

Freeze the exact same-start, matched-total-budget comparison between an evolved outer-loop researcher and a fixed/incumbent or human-authored reference researcher before confirmatory outcomes are accessed.

The JSON file is intentionally populated with `__REQUIRED_BEFORE_PREREGISTRATION__` placeholders. The presence of a syntactically valid JSON file does not mean the experiment is ready. The template must be completed, independently reviewed and committed before execution or outcome observation.

## Structural schema

The companion file
`SYM_RSI_R4_IGNITION_PREREGISTRATION_SCHEMA_V0_2026-11-06.json`
defines the JSON shape, required keys, primitive types, two-arm structure,
minimum pilot/rerun counts, control count and several fail-closed authority
flags.

The JSON Schema is deliberately a **structural** check only. It does not
compare the two arms' referenced artifact digests, calculate or verify equal
total-cost budgets, verify split disjointness, authenticate a custodian, prove
that confirmation material is inaccessible, or decide whether the selected
claim is supported. A later semantic validator under #10749 must reject
unresolved placeholders when a document is marked preregistered and verify
cross-field invariants. Human review and independent infrastructure evidence
remain necessary.

## How to use

1. Copy the JSON template to a new immutable experiment-specific path.
2. Replace every required placeholder with a concrete identity, digest, budget, split, seed plan, outcome, analysis, or governance reference.
3. Run a structural completeness validator once it is implemented under #10749.
4. Obtain independent review of identical starting inner-agent artifacts, the declared outer-loop treatment difference, the total-cost accounting boundary, and confirmation/transfer isolation.
5. Commit and freeze the completed preregistration. Any later amendment must be versioned, justified and made before the affected outcomes are observed.
6. Bind actual execution, build, evaluation and review evidence through the existing canonical receipt and attestation owners; do not use this JSON as a replacement receipt.

## Key scientific distinction

This protocol separates:
- **Product improvement:** the inner/task agent is better.
- **Accumulation:** independently audited gains persist across rounds.
- **Repeatability:** the full loop reproduces over independent seeds.
- **Better improver:** the evolved outer-loop procedure beats the earlier/reference procedure from the same start at matched total cost.
- **Ignition:** that successor researcher itself produces better successors.
- **Recursive progress:** the improvement is inherited and repeated over further full outer cycles.
- **Transfer:** the improver works on fresh task families or domains without retuning.

None of these claims follows automatically from another.

## Evidence obligations

For an R4 claim, report selection-independent confirmation, round-by-round development and independent audit scores, at least two complete-loop reruns, a same-start matched-total-budget improver comparison, null controls, noise-aware promotion, full cost accounting, and best-so-far vs final capability.

A five-seed paired pilot is a floor for estimating variance where feasible; it is not a universal confirmatory sample size. The confirmatory count must be fixed from a preregistered precision/power analysis before sealed confirmation is accessed.

## Authority boundaries

- #10497 owns the R4 scientific design and acceptance criteria.
- #10573 retains independent evaluator and promotion authority.
- #10708 retains controlled evaluator epoch evolution.
- #10569 retains candidate/source/build lineage binding.
- #10314/#10370/#10533 retain canonical run receipts and attestation/regrade.
- #8438/#9433/#10513 retain statistical and population-aware inference.
- #10717 owns causal attribution of specific change units and interactions.

This template is not itself an evaluator, signer, receipt, archive, statistical authority or promotion mechanism.

## Semantic readiness contract (v0 extension)

The companion Rust reference validator checks invariants that JSON Schema cannot express by itself:

- The two arms must bind to the exact same `starting_state.shared_inner_agent` reference; their researcher references must resolve to the declared evolved/reference artifacts.
- The backend version and frozen experimental context must match across arms. The treatment difference must remain the declared outer-loop researcher.
- Both arms must bind to one identical, digest-identified inclusive budget envelope. Failure, invalid/rejected retries, human review/rework and confirmation/transfer costs remain inside the boundary.
- Discovery, validation, sealed confirmation and fresh-family transfer require distinct manifest digests. A separately identified, independently reviewed disjointness audit must declare zero task-instance and task-family overlap.
- Evaluator implementation, configuration, task contract and epoch schedule are digest-bound and shared across arms. Candidate write access to evaluator or confirmation material is forbidden.
- Paired pilot count remains at least five, full-loop reruns at least two, and recursive-progress claims require at least two complete outer cycles.
- All required controls, per-round development/audit fields, analysis fields and non-compensatory authority flags are checked. R4 claim dependencies are enforced.

Run the offline reference checker:

```sh
cargo fmt --manifest-path tools/sym-rsi-r4-prereg-reference/Cargo.toml -- --check
cargo clippy --manifest-path tools/sym-rsi-r4-prereg-reference/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path tools/sym-rsi-r4-prereg-reference/Cargo.toml
cargo run --manifest-path tools/sym-rsi-r4-prereg-reference/Cargo.toml -- docs/research/templates/SYM_RSI_R4_IGNITION_PREREGISTRATION_V0_2026-11-06.json
```

The current repository artifact is still `TEMPLATE_NOT_PREREGISTERED`; the checker must reject it as not ready. A passing check on a future completed document means only that its declared structure and semantic invariants are internally consistent. The checker does not authenticate referenced hashes/signatures, inspect split contents, prove custody or independence, execute a model, validate scientific truth, or authorize a run/promotion. The custodian's disjointness evidence and existing independent evaluator/receipt authorities remain authoritative.

