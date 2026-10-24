# Nixward one-shot execution authorization: contract and evaluation plan

Date: 2026-10-24  
Status: **partially integrated for typed built-in commands; legacy bypass paths remain**

## Why this boundary matters

A scalar confidence, Phi value, model verdict, caller boolean, or log entry is not an authorization artifact. A privileged dispatcher needs evidence for the *specific action it is about to execute*. The minimum binding is the exact executable plus ordered argument vector, principal, action scope, authority kind, unique decision ID, freshness, revocation state, and one-shot consumption.

This implementation adds a process-local typed ledger in `crates/core/nixward/src/action/authorization.rs` and a permit-checked `NixOSExecutor::execute_with_permit` path for typed built-in commands. The executor now supports an injectable process runner, enabling tests to assert that denied permits never invoke the process boundary and valid permits invoke it exactly once. This is still partial integration: ordinary `execute`, public `execute_confirmed`, generation capture, the daemon, rollback and the legacy executor remain separate paths. The next stage must route every governed process invocation through an enforced common boundary and remove or narrow unguarded confirmed-execution APIs only after every caller has explicit authority semantics.

## Implemented primitive

- Exact `executable + argv[]` equality. Arguments are not flattened into a shell string, so argument-boundary changes are detectable.
- Principal, action-scope, and authority-kind checks are independent.
- Monotonic-clock expiry and a configurable maximum permit lifetime; default is 30 seconds.
- Decision IDs can be issued only once per in-memory ledger.
- A failed first dispatch attempt burns the permit, including a wrong-command attempt.
- Revocation and per-ledger identity; permits are opaque, non-cloneable, and not serializable.
- Rollback and recovery have separate authority kinds rather than masquerading as a human approval.

## Trust model and limits

This is deliberately a **same-process, in-memory primitive**, not a complete security boundary.

1. The caller of `issue` remains trusted to call it only after a real approval or policy decision. The type does not authenticate its own issuer.
2. The new `execute_with_permit` method checks a capability before calling the shared injectable process runner, but ordinary `execute`, public `execute_confirmed`, generation capture, daemon execution, rollback and the legacy executor remain separate paths. This is not a repository-wide fail-closed guarantee.
3. The handle and receipt are not signed and must not cross a process boundary as trusted evidence. A future daemon/worker split needs authenticated, replay-resistant receipts and a well-defined key/identity lifecycle.
4. The ledger's IDs and replay state are process-local, not durable across restart. Persistence and crash consistency belong in the later enforcement design.
5. Command equality does not by itself prove that a command is safe, that the principal is correctly authenticated, or that policy was appropriate. It only binds the permit to the attempt that consumes it.

## Deterministic invariants

The unit-test contract is:

| Invariant | Required result |
|---|---|
| Same exact executable and ordered argv, principal, scope, authority, fresh permit | consume once |
| Same permit consumed again | reject replay |
| Different executable or one changed/reordered argument | reject and burn permit |
| Different principal | reject and burn permit |
| Different scope or authority class | reject and burn permit |
| Expired permit | reject and burn permit |
| Revoked decision | reject |
| Reuse decision ID to issue another permit | reject |
| Present permit to another ledger instance | reject |
| Whitespace-bearing arguments versus multiple argv entries | remain distinguishable |
| Empty executable, embedded NUL, zero lifetime | reject |

The authored unit tests cover the in-memory model and exercise `execute_with_permit` against a counting mock runner, asserting one invocation for a valid permit and zero invocations for replay, command substitution, unsupported custom commands, and malformed identity. They do not prove that every existing production path is gated, and they have not been executed in this environment.

## Integration acceptance gates

Before treating this as production enforcement:

- Inventory and route every real process dispatch through a common point-of-use authorization check; no legacy escape hatch may remain for governed modifying actions.
- Consume authorization immediately before dispatch, after command construction and after all plan/approval updates, closing the approval-to-dispatch TOCTOU window.
- Check the authority required by the action at the dispatcher, not merely the authority string supplied by the caller.
- Treat rollback and recovery as explicit bounded policies with their own permitted commands and reason codes.
- Use the injectable process runner to assert **zero calls** for denied, mismatched, stale, revoked, replayed, unauthenticated, custom-command, and policy-error cases.
- Assert exactly one call for valid positive controls, including a valid typed built-in command. A deny-everything implementation must fail the benchmark.
- Exercise concurrent consumption and cancellation; if a permit is consumed before a process spawn fails, it remains consumed and needs a new decision.
- Record exact source SHA, toolchain, test command, test output, and independent review. Queued/skipped checks are not passes.

## Runnable primitive microbenchmark

The repository now includes `crates/core/nixward/examples/one_shot_authorization_bench.rs`.

Run it with:

```sh
cargo run -p nixward --example one_shot_authorization_bench --release
```

It generates 10,000 deterministic scenario trials across valid authorization, command substitution, principal mismatch, scope mismatch, authority mismatch, revocation, and replay. It reports p50/p95/p99 for `ledger.consume`, plus false permits, false blocks, and per-scenario counts; the process exits nonzero if the decision contract is violated. Scenario setup and permit issuance are outside the timed region.

This is intentionally a *microbenchmark of the ledger*, not an end-to-end agent benchmark. It does not measure process launch, policy-generation latency, real system side effects, expiry under a production clock, or the repository-wide bypass surface. The authored harness has not been compiled or run in this environment; no latency number is claimed yet.

## Benchmark plan: attack resistance *and* utility

Use frozen paired cases with an identical command/policy context across a baseline path and the gated path. Separate these events; do not collapse them into one “safety score”:

1. request generated;
2. authorization decision;
3. dispatch attempted;
4. process actually invoked;
5. protected side effect occurred;
6. decision was attributable to a current, correctly scoped authority.

Proposed initial deterministic suite: 10,000 generated attempts, stratified across command mismatch, argv-boundary mutation, replay, expiry, revocation, principal mismatch, scope mismatch, authority mismatch, duplicate decision issuance, and legitimate allowlisted actions. Add property-based fuzzing around argument vectors, concurrent replay tests, and long-running restart/crash tests in later stages.

Proposed targets—not current results:

- **100%** rejection of all deterministic invalid/replayed capabilities in the frozen suite.
- **0** mock-runner invocations (and, in a later OS-level sandbox harness, zero protected side effects) on denied paths.
- **100%** valid positive-control dispatches for policies defined as allowed.
- **0** duplicate process invocations from concurrent consumption of one permit.
- Report false permits, false blocks, request rate, dispatch rate, side effects, and p50/p95/p99 overhead separately.
- Target **less than 1 ms p95** for in-process ledger verification on the reference machine, but only accept this after measuring with a reproducible benchmark and recording the hardware/toolchain. Do not trade away enforcement correctness for latency.

For an agent-level evaluation, include noisy natural-language contexts, explicit opt-outs, quoted/reported/conditional instructions, adversarial prompt variations, benign workflows, and culturally/domain-specific acceptable responses. Moral competence should not be reduced to one verdict-accuracy number; moral sensitivity, context counterfactuals, reasoning quality, steerability, and pluralism need distinct metrics.

## External baselines and reading

These are external reports, not results for Symthaea or Nixward:

- [Open Agent Passport: Before the Tool Call (March 2026)](https://arxiv.org/abs/2603.20953) describes synchronous pre-action policy checks and signed audit records; its abstract reports a 53 ms median enforcement latency for its particular 1,000-request test.
- [APort Vault (September 2026)](https://arxiv.org/abs/2609.22076) reports 4,371 human-authored attacks replayed across 14 models and 225,964 completed evaluations. It distinguishes unpermitted successful transfers from requests and policy decisions, and reports that many allowed payments still executed—an important guard against “deny everything” scoring. Its [released dataset](https://huggingface.co/datasets/aporthq/vault-benchmark-v1) discloses 5,395 error rows against a planned 244,776-evaluation grid, which is a reminder to make failed/missing runs first-class benchmark outcomes.
- [A roadmap for evaluating moral competence in LLMs (Nature, February 2026)](https://www.nature.com/articles/s41586-025-10021-1) argues for adversarial and confirmatory evaluation, sensitivity to relevant context, and pluralism rather than binary verdict accuracy alone.
- [Beyond Verdicts: Evaluating Language Model Moral Competence (AAAI, March 2026)](https://ojs.aaai.org/index.php/AAAI/article/view/41131) identifies weaknesses in curated scenarios and overreliance on verdict accuracy, and advocates testing moral sensitivity, reasoning quality, and steerability.

A fair comparison must pin versions, policies, prompts, hardware, retries, and failure rows, publish the raw per-case outcomes, and compare the same task/policy conditions. External headline numbers are not transferable to this repository without running the same evaluation.
