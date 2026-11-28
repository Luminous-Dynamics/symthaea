# MACE Interaction Execution Receipt V0.1

## Purpose

This crate fills a narrow gap exposed by MemPABench: a personal agent can select the correct interaction preference yet fail to carry it out. The public benchmark reports 96% correct setting selection under its Oracle condition but only 59% reply adherence on five difficult settings. Memory selection and behavior realization must therefore be scored separately.

This is an execution-evidence contract, not a preference store, selector, consent authority, tool authorizer, semantic judge, or memory backend. It consumes a decision from #13153 and composes with #13388's on-policy interaction evaluation, #9016's subject/scope/audience boundaries, and the existing Mycelix/tool execution authority.

## Protocol

1. The existing policy selects one or more currently applicable interaction preferences. The selection binds subject, context, decision-policy digest, source evidence receipts, scope, purpose, audience and bitemporal validity.
2. The executor emits an ordered event stream containing applied controls, committed actions, response commitment and—when a prohibited action must not occur—a closed no-action observation window.
3. The execution adapter creates a receipt binding request, selected decision, exact response bytes, ordered event manifest and executor build.
4. An independent host verifier checks source/authority receipts and every executor event. Required control/action events and no-action-window closure must occur no later than the exact response-commit event; a later event cannot retroactively qualify an already emitted response. For a no-action requirement the host must prove that the action journal is complete through the requested observation window; absence from an incomplete log is not evidence that the action did not occur.
5. The verifier returns an in-process VerifiedInteractionExecutionV1 capability. Its constructor and fields are private; a JSON receipt and re-computed SHA-256 cannot recreate the capability.

## Supported requirements

- control_equals: the executor must emit an independently verified control receipt with the exact selected setting digest (for example verbosity mode or autonomy mode).
- action_must_occur: the executor must show a verified action event (for example an authorized calendar create).
- action_must_not_occur: the executor must close a precisely defined time window and prove its action ledger is complete. Any matching forbidden action in the half-open interval causes rejection.

No-action coverage is a first-class event. It is not inferred from a missing tool call or lack of a database row.

## What this proves—and does not

- The contract proves that the injected verifier accepted the selection and executor evidence, and that the observed events satisfy the structured requirements.
- It binds the exact response bytes and tool/action event IDs to the selected preference, but a digest alone is not an authenticated signature.
- control_equals proves that the trusted renderer/executor applied a particular control setting. It does not prove the resulting prose semantically followed a preference such as "be warm" or "be concise"; BRS/PRS, an independent evaluator and human studies remain separate.
- A no-action result is only as strong as the injected verifier's ability to establish complete event-journal coverage and the executor's trust boundary.
- The fixture verifier is synthetic test scaffolding and is not suitable for production.
- This contract does not decide whether a preference is true, durable, consented, or applicable. It checks the upstream selection's scope/time and lets the host verify its authority. This version also does not prove that an Ignore decision was withheld from the model's prompt: that requires a complete exact provider-context exposure receipt from #13973. Conversely, Support/Dominate require positive exposure evidence. The cross-benchmark integration and this hard gate are tracked in #14264; do not treat a selected Ignore label or missing retrieval row as proof of non-exposure.

## Tests

Eighteen deterministic tests cover control realization, required actions, prohibited action detection, complete and incomplete no-action windows, context/audience mismatch, stale/future-known preferences, wrong subject, exact response binding, tampered receipts, duplicate/noncanonical events, untrusted event receipts, missing control evidence, multiple independent preferences, and control/action events that occur too late to govern the committed response.

No MemPABench/OP-Bench score, production integration, semantic-quality guarantee or external authentication is claimed.
