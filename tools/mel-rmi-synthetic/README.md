# Melothaea RMI-Bench v1 — synthetic Track A harness

This isolated, dependency-minimal Rust crate implements the first offline milestone from the preregistered RMI-Bench v1 protocol.

It is a benchmark-integrity oracle, not a Melothaea evaluation. Its positive control is a rule-based policy whose expected transformation is declared by a small synthetic task fixture. It does not establish human musical relevance, collaborative value, consciousness, or performance against a commercial model.

## What it exercises

- Paired partner observations with identical task, session, seed and frozen history identities.
- A targeted treatment, sham/no-op control, and unrelated-feature negative control; only the declared feature may differ in the targeted arm.
- Fixed-response, imitation-only, one-way-prompt and cross-session yoked controls, plus a rule-based positive control.
- Separate targeted, sham and unrelated-feature action contrasts.
- A task-specific synthetic response-relation check. Imitation can be input-sensitive while still failing this complementarity oracle.
- A four-condition controlled-partner response path with separate targeted/sham/unrelated contrasts and a declared reverse-direction response-rule oracle.
- Exact event-ID, seeded condition-order execution, policy-version and Git-revision binding.
- SHA-256 binding of exact supplied source/output bytes and serialized transaction-receipt bytes, plus a second-call digest comparison over caller-supplied payload sets. The helper itself does not fetch durable Studio storage or prove that a read-back occurred.
- A typed `CommittedArtifactReadbackV1` boundary that asks an adapter to fetch each of the eight event IDs from committed storage before digest comparison. Two tests check exact event lookup and post-commit byte substitution rejection using a fake read-back provider. The fake provider tests the contract only; it is not proof that Studio storage has been reopened.
- Pinned `sha2 = 0.10.9` dependency and minimal committed lockfile; all test/run commands use `--locked`.
- A fresh policy fork restored from one frozen pre-intervention checkpoint for every condition; mutable policy state cannot leak between counterfactual arms. The same explicit common-random-number seed is passed to every arm so stochastic adapters can pair random draws.
- Explicit NotApplicable metric state for unsupported modalities.
- Multi-valid negotiated response contracts that distinguish contract validity, partner vetoes, explicit partner choice, unresolved ambiguity, and clarification/branch/abstention.

## Run

From this directory, run:

```sh
cargo fmt --check
cargo test --locked
MEL_RMI_CODE_REVISION="$(git rev-parse HEAD)" \
  MEL_RMI_ARTIFACT_REPORT="./mel-rmi-artifact-digests.csv" \
  cargo run --locked --example canonical_suite
```

The example emits a CSV over 160 paired fixtures (32 deterministic variants per
feature dimension) for fixed, imitation, one-way, yoked, unrelated-feature-
shortcut and rule-based policies, followed by 160 four-condition reciprocal
controlled-partner trials. Every one of the 960 policy trials and 160 reciprocal
trials receives four input and four output synthetic Score JSON/MIDI payload
sets plus synthetic receipt bytes. The example hashes and rechecks the supplied
in-memory payload sets before reporting `artifact_verified,1120`; this is
synthetic byte-equality evidence, not proof of a separate durable-storage
read-back. A real Studio adapter must independently fetch committed bytes and
authenticated transaction receipts before making a production claim. The
example also emits dimension-specific agent and partner summaries to stderr:
fixture count, means, targeted-effect dispersion, sham/unrelated contrasts,
and control/oracle qualification counts. These are descriptive summaries over
constructed fixtures, not statistical estimates from sampled musicians; no
confidence interval or cross-dimension composite is reported. The separately
written artifact-digest CSV contains one row per input/output artifact,
including event ID, session/task/assignment seeds, policy/version, code
revision, metric version, caller-supplied verifier identity/version,
transaction-receipt SHA-256, artifact SHA-256, format and byte length
(17,920 rows for the canonical sweep).

The repository workflow stages this crate outside the root Cargo workspace, tests the exact pull-request head using pinned Rust 1.96.0 and a committed `--locked` dependency graph, gates all expected positive/negative-control counts, all five dimensions' summaries, and 1,120 successful artifact re-verifications, then uploads the raw trial CSV, per-dimension summaries and digest CSV as a 90-day workflow artifact.

The same executable emits eleven `negotiation` rows. Six are expected to qualify: complete branch within budget, clarification, veto-compliant branch, following explicit partner selection, abstention after all options are vetoed, and clarification when the viable set exceeds the branch budget. Five are expected to fail: premature single commitment, a branch containing a vetoed value, fabricated commitment after all options are removed, incomplete branching, and attempted branch explosion above the configured fan-out limit. The workflow gates these counts separately from causal-contingency and artifact-integrity outcomes.

The Rust unit suite additionally exhaustively checks 3,464 dispositions across all non-empty accepted-offset sets for cyclic domains of size 2–5, every source value, every veto subset, and unresolved/explicit partner choices. It checks both expected-valid and expected-invalid outcomes and pins the exact enumeration count.

The runner also emits seven scoped-authority cases: one valid event and six rejected cases covering wrong source event, stale sequence, wrong clock origin, not-yet-effective event, expiry, and missing receipt reference. These checks bind the decision state to the current session/task/observation and time context. They do **not** authenticate the referenced actor or signed receipt. Production use must first verify the opaque receipt, actor and purpose-specific authorization through the MEL-013A trust boundary.

## Interpretation boundary

Passing these tests means only that this finite synthetic harness detects its known positive and negative controls. The manifest stores exact event identities and a caller-supplied code revision; it does not cryptographically hash arbitrary audio or MIDI payloads. Real artifact digests and Studio transaction receipts must be supplied by the MEL-013A adapter before external or human-session claims.

Track A's deterministic fixture set is a census of those fixtures, not a statistical sample of musicians. Human/live tracks still require a pilot, preregistered power analysis, appropriate ethics review, and blinded musician judgments. No human data collection is included here.


## Multi-valid response negotiation

The crate also provides an interaction-level `NegotiatedResponseContractV1`.
A contract can admit several symbolic alternatives; it does not rank them or
claim they are perceptually interchangeable. Partner authority is represented
separately as an optional explicit choice plus vetoes. Under unresolved
multi-option preference, the agent must either preserve the complete viable
branch set or ask for clarification. It may not silently commit to one valid
option. Once the partner has selected an option, a different valid option is
not preference-compliant. If vetoes eliminate every option, abstention is the
qualified outcome.

The assessment reports contract validity, veto compliance, explicit-choice
compliance, ambiguity preservation, action shape and qualification separately.
These are synthetic interaction/authority checks—not human judgments,
authorship rulings, or score/audio perceptual-equivalence claims. The response
contract composes with the existing symbolic realization-equivalence work in
PR #9325; it does not reimplement its Score/event matching logic.
