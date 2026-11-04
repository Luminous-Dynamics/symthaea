# CEB causal ledger (standalone research crate)

This crate is the first pure-Rust implementation slice for CEB's causal-investigation ledger.

## Scope

It validates inert records for:
- exact task/source/configuration binding;
- unique record identifiers and append-order sequence numbers;
- evidence-parent ordering and duplicate evidence-origin detection;
- prospective forecast → action → outcome ordering;
- typed action categories and explicit authorization for side-effecting/test actions;
- action-kind-specific evidence authority (source/compiler facts are not runtime outcomes; tests require runtime observations; interventions require intervention observations); 
- nonblank task, record, evidence-origin, evaluator and hypothesis identities; 
- unique evidence/hypothesis reference lists and rejection when one evidence item is assigned both supporting and contradicting roles;
- nonblank identification assumptions and abstention/escalation requirements;
- explicit decision-evidence linkage to considered hypotheses and fail-closed rejection of any unqualified evidence cited by a repair decision;
- matched ablation-group validation requiring an intact arm and comparator with identical policy and budget digests;
- explicit missingness rather than converting unobserved outcomes to numeric zero;
- explicit proceed / abstain / escalate decisions, with missing requirements recorded;
- repairs bound to the current hypothesis revision and an explicit proceed decision;
- version-linked hypothesis revisions, explicit proceed/abstain/escalate decisions, decision-time evidence checks, and repair/primary-grade/pristine-regrade identity links.

## Non-goals

This crate does **not**:
- execute experiments, tests, builds, shell commands or candidate code;
- read or write files, use the network, spawn processes or access credentials;
- compute or authenticate digests;
- verify DSSE signatures, trust roots, runner identity, isolation or grader independence;
- prove that an observation is true or that a causal claim is correct;
- emit or replace a CEB run receipt.

The `Digest` type is opaque. The caller supplies it; this crate only checks equality and linkage. Evidence payloads, forecast policy and action parameters are digest-bound for substitution detection, but this crate does not compute or authenticate those digests. The revision chain is validated when the complete record history is supplied, but the crate has no persistent append-only store and does not prevent a caller from replacing its input data. A `PristineRegrade` label is an identity field, not proof that the grading environment was actually independent.

## Run the pure unit tests

From the repository root:

```sh
cargo test --manifest-path tools/ceb-causal-ledger/Cargo.toml --locked
```

The crate has its own `[workspace]` boundary and is intentionally not a member or dependency of the production workspace. Forty-eight unit tests are authored in the current source revision (48 `#[test]` attributes counted by static source inspection). The latest seven adversarial tests cover blank identities, blank evidence origins, blank assumptions, mixed blank/nonblank abstention requirements, conflicting support/contradiction roles, and blank evaluator revisions. Integrate it only after exact-head compilation, independent review, and reconciliation with the existing CEB/general receipt and DSSE work.

## Status

Research prototype. The source and unit tests are committed but have **not** been compiled or executed in this environment. The PR remains draft. CI jobs that were skipped for the draft PR are not evidence of passing compilation or tests.
