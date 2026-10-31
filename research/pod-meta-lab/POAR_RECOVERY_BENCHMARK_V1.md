# POAR-RX v1: Recovery Under Fault Injection

**Date:** 2026-10-31. **Status:** benchmark specification plus a first in-memory Rust lifecycle model; no Symthaea recovery score is claimed.

## Thesis

Evaluate agents on whether they recover after a stateful operation has an uncertain outcome, not only whether they eventually produce a plausible answer. A lost acknowledgement after dispatch does not prove that a write failed. Retrying a non-idempotent action without checking current state can duplicate effects.

The proposed protocol is: intent, precondition check, prepare, dispatch, reconcile observed state, verify postconditions, then issue a completion receipt. When the outcome is unknown, inspect authoritative state before another attempt. If state cannot be verified, preserve the unknown outcome and report it accurately.

## Fault suite

- Timeout before dispatch.
- Timeout after a committed operation but before acknowledgement.
- Stale read with a concurrent writer.
- Permission change during a multi-step task.
- Partial success across multiple records.
- New conflicting information during execution.
- Tool contract or response-schema drift.
- Failed compensation.
- Duplicate delivery or replay.
- Missing authoritative read-back.

Each case declares initial state/version, goal predicates, allowed effects, principal and scope, tool contract, idempotency support, action pre/postconditions, compensation policy, hidden fault schedule, root cause, and recoverability. Pair faulted scenarios with no-fault controls. Hold out tool/action families and fault combinations, not only random task instances.

## Execution journal

Each action records a run ID, monotonic sequence, parent action, pre-state digest/version, intended postcondition, principal and capability, tool/schema revision, parameter digest, idempotency key (or explicit unsupported status), dispatch outcome, authoritative read-back, state classification (not dispatched / committed / not committed / unknown), recovery decision, and verifier receipt.

The journal is append-only. A transport acknowledgement is not proof of a state transition. Completion requires independent observation of all required goal predicates and a check that forbidden effects did not occur.

## Matched systems

Compare reactive tool use, plan-then-execute, retry-only, policy-guarded agent, saga/compensation controller, generic COG-META, POD-Meta without fault reconciliation, POAR-RX with durable journal and proof-gated recovery, and an oracle diagnostic upper bound. Match model, tools, policy and budget where possible.

## Metrics

Report final-state predicates, partial completion, forbidden effects, truthful terminal status, safe-abort accuracy, first critical failure localization, root-cause macro-F1, detection delay, recovery rate by fault recoverability, duplicate effects, compensation accuracy/harm, residual state, unaffected-state stability, postcondition read-back, journal integrity, scope/authority/replay violations, tool calls, tokens, CPU/GPU, memory, latency, energy and cost.

Publish per-family metrics and success/safety/cost Pareto curves. Never average hard safety failures into an overall score.

## Hypothesis gates (not results)

Aim for zero forbidden effects and zero false completion in the deterministic contract suite; zero blind duplicate non-idempotent actions over a large acknowledgement-loss trial set; at least 95% recovery on declared recoverable single-fault cases; at least 99% correct safe-abort on unrecoverable cases; at least 90% top-1 critical-failure localization; and a 20% all-in cost reduction at matched success and safety as a stretch goal. Freeze these targets before confirmatory evaluation and report uncertainty and family clustering.

## External suites

Use official evaluators independently: [Open Agent Leaderboard / Exgentic](https://www.exgentic.ai/blog/open-leaderboard/) for full-system success and cost; [NOWAI-Bench](https://github.com/ServiceNow/NOWAI-Bench) for enterprise workflows and voice tasks; [OSWorld 2.0](https://osworld-v2.xlang.ai/) for long-horizon computer use; [AgentRx](https://aclanthology.org/2026.findings-emnlp.598/) for failure localization; [TEMPO](https://aclanthology.org/2026.emnlp-main.291/) for temporal retrieval; and [GRACE](https://aclanthology.org/2026.emnlp-main.1582/) / [FaithCoT-Bench](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6c7154e394e24c69409256ccf8bf0804-Abstract-Conference.html) for trace faithfulness. Do not average incompatible scores.

## Symthaea hypothesis

Symthaea proposes transitions; the obligation graph defines pre/postconditions; the journal binds actions to state, authority and idempotency; the recovery controller chooses reconcile, retry, compensate, seek approval or stop; a separate verifier checks observed outcomes. Mycelix provenance can later attest authority, but an identifier or hash alone does not prove real-world identity.

The lab now contains an in-memory action lifecycle in research/pod-meta-lab/src/recovery.rs and a run-level journal in research/pod-meta-lab/src/recovery_journal.rs. The action model exercises ambiguous acknowledgement, non-idempotent retry blocking, retry after observed non-commit, declared idempotency-key replay, unknown read-back, safe-abort versus unknown-stop, and compensation authorization plus restored-state read-back. The run journal enforces run identity, monotonic sequence, unique action IDs, and ordered event history. Thirteen tests cover these lifecycle and run-journal invariants. These types do not execute external tools and are not durable storage. Next implement a multi-action append-only run journal and deterministic fault-state runner, add the remaining fault families, conduct no-journal/no-reconcile/no-proof-gate ablations, then adapt a reversible sandbox with live writes disabled by default.

**Claim ceiling:** any result is limited to named tool contracts, state observers, fault distributions and policies. It does not prove universal reliability or exactly-once execution.
