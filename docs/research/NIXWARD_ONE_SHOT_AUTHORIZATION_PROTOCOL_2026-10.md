# Nixward one-shot execution authorization: contract and evaluation plan

Date: 2026-10-24  
Status: **partially integrated for typed built-in commands; legacy bypass paths remain**

## Why this boundary matters

A scalar confidence, Phi value, model verdict, caller boolean, or log entry is not an authorization artifact. A privileged dispatcher needs evidence for the *specific action it is about to execute*. The minimum binding is the exact executable plus ordered argument vector, principal, action scope, authority kind, unique decision ID, freshness, revocation state, and one-shot consumption.

This implementation adds a process-local typed ledger in `crates/core/nixward/src/action/authorization.rs` and a permit-checked `NixOSExecutor::execute_with_permit` path for typed built-in commands. The executor uses an injectable process runner for ordinary execution, confirmed execution, and generation capture; `capture_generation` also rejects nonzero exit status before accepting a parsed current-generation row. Authored mock tests target the runner boundary and failure-shaped generation output. These changes are source-reviewed only until exact-head tests execute. This is still partial integration: ordinary `execute` retains its legacy authorization semantics, public `execute_confirmed`, the daemon, plan rollback, and the legacy executor remain separate paths. The next stage must route every governed process invocation through an enforced common boundary and remove or narrow unguarded confirmed-execution APIs only after every caller has explicit authority semantics.

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

The current example harness now stratifies 10,000 consumption attempts across nine scenarios: valid authorization, command mismatch, principal mismatch, scope mismatch, authority mismatch, ordered-argv boundary mismatch, foreign-ledger use, revocation, and replay. It reports aggregate and per-scenario p50/p95/p99. Because 10,000 is not evenly divisible by nine, per-scenario counts differ by one; raw counts must be preserved when interpreting the distributions. This is deliberately a *consume-path microbenchmark*, not the complete deterministic conformance suite: expiry, duplicate decision issuance, malformed requests, concurrency, restart/crash recovery and process-side-effect tests remain separate required tests. Add property-based fuzzing around argument vectors, concurrent replay tests, and long-running restart/crash tests in later stages.

Proposed targets—not current results:

- **100%** rejection of all deterministic invalid/replayed capabilities in the frozen suite.
- **0** mock-runner invocations (and, in a later OS-level sandbox harness, zero protected side effects) on denied paths.
- **100%** valid positive-control dispatches for policies defined as allowed.
- **0** duplicate process invocations from concurrent consumption of one permit.
- Report false permits, false blocks, request rate, dispatch rate, side effects, and p50/p95/p99 overhead separately.
- Target **less than 1 ms p95** for in-process ledger verification on the reference machine, but only accept this after measuring with a reproducible benchmark and recording the hardware/toolchain. Do not trade away enforcement correctness for latency.

For an agent-level evaluation, include noisy natural-language contexts, explicit opt-outs, quoted/reported/conditional instructions, adversarial prompt variations, benign workflows, and culturally/domain-specific acceptable responses. Moral competence should not be reduced to one verdict-accuracy number; moral sensitivity, context counterfactuals, reasoning quality, steerability, and pluralism need distinct metrics.

## Composition with the repository-wide action runtime

Nixward's local ledger is a useful narrow guard, but it must not become a parallel authority system. The repository already has a broader action-runtime theorem and migration line:

- [ACTION-RUNTIME-V2E-R1 #4556](https://github.com/Luminous-Dynamics/symthaea/issues/4556) defines the repaired effect-entry firewall using an affine `DispatchPermitV2`, exact `CapabilityGrant`, fresh `VerifiedAuthorityStateV2` / `VerifiedAuthorityTime`, and executor/dispatch-incarnation binding.
- [ACTION-RUNTIME-V2F #4545](https://github.com/Luminous-Dynamics/symthaea/issues/4545) addresses the stronger reconciliation theorem for proving both non-application and absence of any live/delayed dispatch before authority may be released.
- [ACTION-LEGACY-PEP-001 #4593](https://github.com/Luminous-Dynamics/symthaea/issues/4593) owns the legacy `SimpleExecutor` filesystem/process effect plane.
- [NIXWARD-AUTH-CROSS-046 #6849](https://github.com/Luminous-Dynamics/symthaea/issues/6849) owns unmanaged real I/O in `MoleculeExecutor`.

The intended composition is:

```text
verified current authority + exact grant + durable use reservation
        -> affine dispatch permit / repaired effect-entry firewall
        -> NixOS, file, command, service, robot or connector effect adapter
        -> normalized outcome evidence + postcondition evidence
```

The one-shot ledger from this PR should either become a documented local adapter to the qualified runtime contract or be superseded at integration time by the existing permit lineage. It must not independently grant new authority simply because a process-local request has an `AuthorizationAuthority` enum value. Nixward's domain-specific checks can further restrict an already-authorized action, but must not bypass or replace the shared authority theorem.

A source search on default-branch `main` also surfaced direct command execution through `SimpleExecutor::execute(ActionIR::RunCommand)`, `MoleculeExecutor::execute_atom_real(Atom::Exec)`, `src/bin/symthaea.rs::run_structured_command`, the root NixOS executor, plus separate daemon/watchdog paths. These code paths are not all introduced by this PR and are not all governed mutations; their presence means a Nixward-only green test cannot substantiate a repository-wide no-bypass claim. Track their migration through the linked work items and use an explicit authority-surface registry in the eventual qualification gate. The cross-surface registry and qualification gate are now tracked explicitly in [ACTION-SURFACE-QUAL-001 #8656](https://github.com/Luminous-Dynamics/symthaea/issues/8656), which complements the generic assurance campaign in #2622 and per-surface migrations rather than replacing them.

## Additional assurance boundary: issuance is not verification

The ledger enforces a capability once one has been issued, but the enum `AuthorizationAuthority` and the public `issue(request)` operation do not authenticate the issuer. The current profile therefore assumes that code with mutable ledger access is inside the trusted computing base and calls `issue` only after a qualified decision. This is a stated trust assumption, not an enforcement property of the type.

Before using this across an agent/daemon boundary, explicitly separate:

```text
untrusted planner / model
        -> typed proposed intent
trusted approval/policy component
        -> authenticated, audience-bound authority
executor / reference monitor
        -> independently verify issuer + audience + intent + epoch + expiry
        -> reserve one-shot/replay state at a defined linearization point
        -> process dispatch
        -> outcome receipt
```

A caller-selected `HumanOperator` or `WatchdogOperator` label is not evidence that the corresponding authority actually approved an action. A process-local opaque handle is not a signed cross-process credential. Key custody, audience restriction, delegation attenuation, revocation freshness, durable nonce reservation, and crash behavior must be specified before receipts are accepted across a boundary. This is being tracked separately in [NIXWARD-AUTH-ISSUER-001](https://github.com/Luminous-Dynamics/symthaea/issues/8624), distinct from [NIXWARD-AUTH-INTENT-001](https://github.com/Luminous-Dynamics/symthaea/issues/8615), which concerns action-context and pre-state binding.

## Fair performance comparison: compare like with like

Published performance figures are useful reference points, not portable targets unless the measured layer and environment match.

- [Agent Passport System's published benchmarks](https://agent-passport.org/benchmarks.html) report 420 ns p50 / 470 ns p99 for its raw verifier core on an AMD EPYC 7313P reference machine, and 822 µs p50 / 6 ms p99 for its gateway round-trip. The core, SDK, durable-record, and gateway layers use different workloads and iteration counts; these are project-published results and have not been independently reproduced by Nixward.
- [Open Agent Passport's pre-action authorization paper](https://arxiv.org/abs/2603.20953) reports a 53 ms median in its 1,000-request setup. This includes a different enforcement configuration and must not be compared directly with a local in-memory ledger.
- [APort Vault](https://arxiv.org/abs/2609.22076) reports a large paired authorization benchmark and publishes its evaluation data. Its separation of requested tool calls, policy decisions, payments, and unpermitted successful transfers is a useful model for event accounting. Its own disclosed missing/error rows reinforce that non-completions must remain visible.

Nixward should report at least four distinct latency layers when those paths exist: (1) in-memory ledger verification/consume, (2) authenticated receipt + durable replay reservation, (3) complete authorization gateway round-trip, and (4) end-to-end dispatch through observed postcondition. For every layer, freeze the workload, policy, machine, toolchain, warm/cold-cache state, iteration count, concurrency and percentile definition. Never claim that a faster primitive makes the full authorization system faster or safer without a matched end-to-end measurement.

The current example now prints p50/p95/p99 consumption latency **per scenario** as well as the aggregate. This is source-reviewed, not an observed result until the exact-head benchmark has been executed and its output archived.

## External baselines and reading

These are external reports, not results for Symthaea or Nixward:

- [Open Agent Passport: Before the Tool Call (March 2026)](https://arxiv.org/abs/2603.20953) describes synchronous pre-action policy checks and signed audit records; its abstract reports a 53 ms median enforcement latency for its particular 1,000-request test.
- [Agent Passport System (APS) published benchmarks](https://agent-passport.org/benchmarks.html) report a 420 ns p50 / 470 ns p99 raw verifier core, 7.6 µs p50 / 13.2 µs p99 for a queued durable-record layer, and 822 µs p50 / 6 ms p99 for a gateway round-trip on an AMD EPYC 7313P reference machine. These are distinct layers, have different iteration counts, and are project-published results; they are not direct measurements of Nixward.
- [Microsoft Agent Governance Toolkit (AGT) benchmark](https://microsoft.github.io/agent-governance-toolkit/BENCHMARKS/) reports 0.030 ms p50 / 0.108 ms p99 for 100-rule policy evaluation and 0.103 ms p50 for kernel enforcement on its documented Windows 11 development workstation. This is another useful reference point, but hardware, language, policy and benchmark path differ.
- [Mergen Agent Runtime Security Benchmark v1.0.0](https://mergen.app/docs/agent-security-benchmark) describes a 551-case corpus (322 attack, 229 benign) processed through its live decision gate in decide-only mode. It reports 100% attack detection and 0% benign blocks for that fixed corpus, and p50 2.935 ms / p95 5.069 ms / p99 10.169 ms across 2,755 evaluations. The stated corpus and the absence of real downstream execution are important boundaries; fixed-corpus performance is not a guarantee against unseen attacks.
- [NIST NCCoE's Software and Agentic AI Identity and Authorization initiative](https://www.nist.gov/news-events/news/2026/09/comments-software-and-agentic-ai-identity-concept-paper) states that its first implementation use case will address agent identity, authentication and authorization in the software-development lifecycle. Nixward's conformance suite should therefore include developer agents, repository/tool identity, least-privilege delegation and auditable build/deploy actions as a concrete interoperability profile, without claiming NIST endorsement.
- [APort Vault (September 2026)](https://arxiv.org/abs/2609.22076) reports 4,371 human-authored attacks replayed across 14 models and 225,964 completed evaluations. It distinguishes unpermitted successful transfers from requests and policy decisions, and reports that many allowed payments still executed—an important guard against “deny everything” scoring. Its [released dataset](https://huggingface.co/datasets/aporthq/vault-benchmark-v1) discloses 5,395 error rows against a planned 244,776-evaluation grid, which is a reminder to make failed/missing runs first-class benchmark outcomes.
- [A roadmap for evaluating moral competence in LLMs (Nature, February 2026)](https://www.nature.com/articles/s41586-025-10021-1) argues for adversarial and confirmatory evaluation, sensitivity to relevant context, and pluralism rather than binary verdict accuracy alone.
- [Beyond Verdicts: Evaluating Language Model Moral Competence (AAAI, March 2026)](https://ojs.aaai.org/index.php/AAAI/article/view/41131) identifies weaknesses in curated scenarios and overreliance on verdict accuracy, and advocates testing moral sensitivity, reasoning quality, and steerability.

A fair comparison must pin versions, policies, prompts, hardware, retries, and failure rows, publish the raw per-case outcomes, and compare the same task/policy conditions. External headline numbers are not transferable to this repository without running the same evaluation.
