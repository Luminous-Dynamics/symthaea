# Effect Surface Registry v1

The machine-readable inventory lives at [effect-surface-registry.v1.json](effect-surface-registry.v1.json).

## Run the integrity check

From the repository root:

```sh
cargo run -p xtask -- effect-surface-registry
```

To make a local pipeline reject this starter registry until every record is marked qualified and carries a syntactically complete evidence record:

```sh
cargo run -p xtask -- effect-surface-registry --require-evidence-records
```

The strict invocation is expected to fail today because none of the entries has exact qualification evidence. This is intentional. The flag checks that records claim qualification and carry evidence-shaped identities; it does not independently verify receipts, so its success is not a security qualification result.


## Rust effect-primitive discovery (candidate-only)

Run the AST discovery command from the repository root:

```sh
cargo run -p xtask -- effect-primitive-scan --out reports/effect-primitive-scan.json
```

It recursively scans Rust files across the repository, excluding generated/cache directories (.git, target, node_modules, .direnv, .cache, .venv, .devenv). The deterministic JSON report records:

- repository HEAD when Git can identify it;
- number of source files discovered and parsed;
- parse failures (any parse failure makes the command exit nonzero);
- each candidate's repository-relative source path, line/column, syntactic call path, heuristic class, source-file SHA-256, and site-identity SHA-256.

The first classifier recognizes direct AST call expressions that look like process command construction, direct `fs` reads/writes/mutations, direct `File::open/create` calls, common socket connect/bind calls, and `libc::system`. It intentionally emits **candidate sites**, not confirmed effects. A `Command::new` call does not prove that a process was spawned, and a path matching `fs` does not prove a particular file was changed.

The scanner is not a sound whole-program effect analysis. It does not resolve aliases, types, control flow or reachability; analyze macro token bodies, FFI internals, generated sources in excluded build directories, scripts, dynamic loading, trait-object dispatch, or arbitrary indirect calls; nor prove the absence of unlisted effect families. It may produce both false positives and false negatives.

Do not claim “all effect sites registered” from a scan alone. The intended next iteration is to adjudicate candidates against the registry, add explicit classifications/exclusions with human-reviewed reasons, and build exact-call-path positive/negative controls with side-effect witnesses. The scan report is a review input and a discovery ratchet, not a qualification receipt.

## What the checker establishes

The Rust xtask command validates the v1 schema, uniqueness of surface IDs, repository-relative source paths, required claim-boundary fields, qualification-evidence shape, and whether every recorded source anchor occurs in its recorded file. It emits the inspected Git HEAD when available. Its registry identity digest (printed as `registry_identity_sha256`) is SHA-256 over compact serialized parsed JSON after removing the `qualification_evidence` field from every entry; this prevents a qualification record from having to contain a digest of a document that includes that very digest. The digest still binds all other registry fields, including the source baseline and entry descriptions.

It does **not** establish that:

- the inventory includes every process, filesystem, network, service, device, FFI, generated, macro-expanded, plugin, shell, or indirect effect;
- the recorded symbol is reachable from an untrusted entrypoint;
- a listed gate is enforced at the true effect point;
- positive or negative controls were compiled or executed;
- an authorization was authentic, current, correctly scoped, or durable;
- the runtime is secure or the repository is free of bypasses.


## Research alignment and benchmark boundary

NIST's October 27, 2026 article, [Design Tenets for Agent Identity Management Systems](https://www.nist.gov/blogs/cybersecurity-insights/design-tenets-agent-identity-management-systems), proposes ten design tenets for agent identity, authentication and authorization. They are guidance from an ongoing NCCoE project, not a conformance certificate or endorsement of this repository. The registry should evolve to make each relevant property testable:

| NIST design theme | Registry/qualification consequence |
| --- | --- |
| Distinct agent identity tied to workload/configuration | Bind principal, workload, environment, approved configuration and executable identity to the action-intent digest. |
| Delegation and attenuation | Bind the parent mandate/delegator and ensure downstream scopes cannot widen parent authority. |
| Least/zero standing privilege | Prefer just-in-time, one-shot scoped permission over ambient or persistent authority. |
| Short-lived, dynamic and revocable grants | Test expiry, revocation, current state, configuration drift and the exact point-of-use revalidation boundary. |
| Human-in-the-loop triggers | Include escalation thresholds and measure benign interruption/consent-fatigue costs alongside security. |
| Mandates and constraints | Bind signed task intent, resource/operation scope, preconditions and required postconditions. |
| Separation of policy from reasoning | Ensure prompts, retrieved content, model confidence, Phi and planner state cannot mint authority. |
| Deterministic access decisions | Use pinned, reproducible policy and test exact authority outcomes independently of model behavior. |
| Continuous monitoring | Test drift detection, anomalous sequences, policy/credential revocation and response latency. |
| Immutable evidence | Bind source, policy, grant, effect adapter, outcome and postcondition to independently verifiable tamper-evident receipts. |

The current registry has only begun this mapping; the seven entries do not satisfy those ten themes by virtue of being recorded.

Benchmark comparisons must also distinguish **decision-only** security from end-to-end effect enforcement. For example, [Mergen's October 12, 2026 benchmark](https://mergen.app/docs/agent-security-benchmark) runs a fixed 551-case attack/benign corpus through its deterministic gate in decide-only mode and reports gate latency. That is useful evidence for the narrow decision path, but it does not by itself establish how many real downstream side effects occurred. Nixward should report request, decision, adapter invocation, actual side effect, and postcondition as separate denominators and publish missing/error rows. Do not compare a local registry check or ledger lookup to a complete gateway and do not rank systems using attack detection alone.

Two useful external reference points reinforce why the benchmark must be layered:

- The [Agent Passport System benchmark](https://agent-passport.org/benchmarks.html) reports, on its canonical AMD EPYC 7313P host, 420 ns p50 / 470 ns p99 for the hot-cache verifier, 7.6 µs p50 / 13.2 µs p99 for a queued durable-record layer, and 822 µs p50 / 6.0 ms p99 for its gateway round-trip. These are vendor-published measurements across distinct layers and iteration counts, not a performance result for Nixward. Aim to compare equivalent layers on a pinned machine, including durability and tail latency rather than selecting only the fastest number.
- The [APort Vault study, published September 18, 2026](https://arxiv.org/abs/2609.22076), replayed 4,371 human-authored attacks across 14 models and reports 225,964 completed evaluations out of 244,776 planned. It distinguishes requests, authorization decisions and successful protected transfers; for Levels 2–4 it reports 140 unpermitted transfers across 76,842 model-alone evaluations versus zero across 69,297 evaluations behind the authorization layer, while 25,370 payments still executed. Its disclosed incomplete rows and positive successful-payment count make the evaluation methodology useful: missing trials remain visible, and a deny-all gate cannot win solely by eliminating utility.

Nixward's proposed benchmark should emulate that denominator discipline while testing its own different threat surface: every case should record intent, decision, adapter invocation, actual side-effect witness, postcondition, error/timeout, and execution/receipt identity. Attack-detection and false-block metrics must be separately reported, and the benchmark should publish raw rows plus the frozen corpus identity.

A stretch research result for Nixward would be an open, reproducible cross-surface conformance suite that combines a versioned effect registry, frozen adversarial and benign controls, actual instrumented effects, exact source/toolchain identity, and independently verified outcome receipts. This could be more useful than chasing a single leaderboard score because it exposes both bypasses and overblocking while making the evidence auditable.


## Federated identity composition is a separate follow-up

The cross-repository integration is tracked in [NIXWARD-FEDERATED-IDENTITY-001 / Symthaea #8717](https://github.com/Luminous-Dynamics/symthaea/issues/8717), with coordination comments on [Mycelix LEX-NET-025 / #1149](https://github.com/Luminous-Dynamics/mycelix/issues/1149), [Mycelix LEX-NET-028 / #1215](https://github.com/Luminous-Dynamics/mycelix/issues/1215), and [Symthaea NIXWARD-AUTH-ISSUER-001 / #8624](https://github.com/Luminous-Dynamics/symthaea/issues/8624).

That follow-up should reuse Symthaea's existing executor-identity composition lineage (#4519, #4585, #4618), Nixward's semantic adapter and generic Action Runtime convergence (#5197, #6385), and Mycelix's local recognition/lease, typed receipt and revocation/delegation contracts. It must not create a parallel identity provider or execution-authority kernel.

The key boundary is:

- Mycelix can contribute authenticated, provenance-bound claims and a purpose-scoped local recognition result.
- Symthaea's trusted composer determines whether the required workload/software/device/profile evidence identifies the same live executor subject.
- A local authority owner—not the foreign issuer, reputation system or identity composer—issues any local bounded grant.
- Nixward binds that grant to the exact target, operation, pre-state and postcondition.
- The existing qualified Action Runtime remains responsible for current authority and the final one-shot dispatch boundary.

W3C Verifiable Credentials Data Model v2.0 is a W3C Recommendation for portable cryptographically secured, privacy-respecting, machine-verifiable claims, while SPIFFE SVIDs provide workload identity under a trust domain and are not local authorization. These standards can inform adapters; neither implies the cross-repository composition has been implemented or qualified. See [W3C VC Data Model 2.0](https://www.w3.org/TR/vc-data-model/) and [SPIFFE SVID concepts](https://spiffe.io/docs/latest/spiffe/concepts/).

## Current scope

This is an initial known-surface baseline audited against main commit c6355cc6ffe01843b841f6b9a8a09eabd2556deb on 2026-10-28, informed by #8656 and the linked per-surface issues. The entries are source-review records with explicit limitations. The inventory is intentionally incomplete and all listed surfaces remain unqualified or only partially guarded.

## Next qualification stage

1. Expand discovery beyond the initial known paths using Rust AST and effect-primitive scans, while reviewing macros, traits, FFI, subprocess wrappers, scripts, and generated code.
2. Split compound records (especially SimpleExecutor) until each independently governable effect family has a unique adapter/test seam.
3. For every surface, add instrumented positive and negative controls with actual side-effect witnesses.
4. Pin source tree, registry digest, toolchain, test plan, and raw result digests in an immutable receipt.
5. Fail a qualification gate when a new effect site is unregistered or a registered entrypoint changes without reviewed inventory treatment.
6. Treat an exact-head skipped, queued, missing, or non-executed lane as unavailable evidence, never as PASS.

Output fields distinguish records marked qualified from independently verified qualification. Even if `--require-evidence-records` succeeds, the command still reports `security_qualification=NOT_EVALUATED_BY_THIS_COMMAND`. The registry is a discovery and assurance input, not an authority source. Adding a row cannot authorize an effect.


## Frozen effect-authorization conformance contract

The companion [`effect-authorization-conformance.v1.json`](effect-authorization-conformance.v1.json) defines 41 versioned test vectors spanning positive utility controls, credential and issuer validation, purpose/audience/domain/resource/action binding, delegation attenuation and revocation, same-live-subject composition, reputation/model-signal non-authority, one-shot replay/concurrency/restart, effect-surface bypasses, pre-state drift, unknown outcomes, and independent postcondition/receipt failures.

This file is a **frozen design contract, not an executable harness, test result, or security qualification**. The baseline fixture and named mutation paths must be mapped to concrete typed test inputs by a later harness. A harness must fail on unrecognized mutation paths; silently ignoring a vector is a corpus execution failure. Per-case setup is explicit for already-consumed and concurrent-use cases.

The scoring contract separates the authorization decision from actual effect-adapter invocation, observed side effect, terminal outcome class, and independent postcondition. It includes positive controls specifically so deny-all cannot score as secure. It requires zero adapter invocations and zero unauthorized side effects for denied/indeterminate cases, exactly one side effect for a valid one-shot action, atomic single-use under a 32-request race, no replay after the declared restart model, and no automatic retry after an unknown outcome. Lifecycle cases where an effect may already have occurred are not incorrectly scored as “zero side effects”; they instead require quarantine/reconciliation and prohibit a success claim without evidence.

Performance campaigns are specified separately for credential verification, recognition, policy decision, durable reservation, dispatch admission, effect handling, postcondition verification, and end-to-end. They must publish p50/p95/p99/p99.9, warm/cold semantics, occupancy points of 1/1K/10K/100K, exact source/toolchain/hardware identity, and complete error/timeout/missing-row counts. These are measurement requirements, not invented performance results or an unmeasured SLO. Never compare an in-memory decision to a durable gateway round trip, and never relax a security gate to improve latency.


### Contract integrity command

Run `cargo run -p xtask -- effect-authorization-conformance` to validate the frozen v1 corpus contract before a behavioral harness consumes it. The checker uses strict typed parsing (unknown fields fail), validates unique case IDs and mutation paths, enforces class-specific decisions/effect counts/terminal dispositions, and guards the baseline security invariants and benchmark-stage requirements. It prints a SHA-256 of the exact corpus bytes.

Its `contract_integrity=PASS` means only that the specification is structurally consistent. The command deliberately emits `execution_status=NOT_EXECUTED_BY_THIS_COMMAND` and `security_qualification=NOT_EVALUATED_BY_THIS_COMMAND`. Unit-test source covers positive-control utility, missing negative mutations, duplicate/invalid mutation paths, forbidden effect counts, and lifecycle terminal-class regressions. These Rust tests and the command have not been compiled or executed in this environment because the Rust toolchain is unavailable.


## Competitive benchmark roadmap: targets, not claims

The current v1 corpus contains 41 frozen contract vectors: 4 positive controls, 32 negative controls, 1 indeterminate control, and 4 lifecycle/evidence controls. This is a useful conformance seed, **not yet a representative industry-scale adversarial dataset and not an executed security benchmark**.

The path to a credible comparison is staged:

1. **Contract integrity (this PR):** fail closed on unknown JSON fields, case removal/addition, duplicate IDs or mutation paths, weakened one-shot/restart gates, missing outcome denominators, and invalid expected dispositions. The committed Rust checker and unit-test source establish this stage, pending actual compilation/execution.
2. **Instrumented effect harness:** turn each case into typed test input and count calls at the true effect adapter. Add positive and negative filesystem/process sentinels plus an outcome-unknown fixture. An ignored mutation or uninstrumented effect must fail the harness; merely obtaining a policy decision is insufficient.
3. **Broaden the fixed corpus:** target at least 500 independently reviewed adversarial cases and 1,000 legitimate/benign operations, with explicit coverage across credential/issuer, audience/purpose, delegation/revocation, identity joins, path/argv binding, concurrency/restart, and outcome evidence. Report category-level confidence intervals, false blocks, missing trials and all raw rows.
4. **Model-mediated track:** where prompt-to-tool selection is under test, replay a fixed attack/benign prompt set across multiple models and replay configurations, while holding the enforcement layer and policy constant. A stretch scale target is comparable to APort Vault's published 4,371 human-authored attacks and 14 models; do not imply equivalence before the corpus, task distribution, and replay protocol match. The deterministic conformance track must remain separable from the model's ability to request a tool.
5. **Independent reproduction:** publish a versioned dataset snapshot, exact code/source SHA, toolchain, host capture, policy profile, runner instrumentation, per-trial event rows, receipt digests, and an independent replay script. Changes to case inventory or policy profiles require a new version, not retroactive scoring changes.

### Proposed performance gates

These are engineering targets, not measured results:

- Local identity/recognition/policy-decision path: target p99 below 1 ms on a pinned reference host.
- Complete durable authorization-admission round trip: first reproduce an equivalent APS L4-style baseline (APS publishes 822 µs p50 / 6.0 ms p99 on its canonical EPYC host), then target p99 below that baseline with the same durability, revocation, receipt, and retry semantics. Never benchmark a hot in-memory cache against a durable round trip.
- Positive utility: zero false blocks on frozen mandatory positive controls; as benign coverage grows, target a false-block rate below 1% on at least 1,000 reviewed legitimate operations, with confidence intervals disclosed.
- Safety correctness is a hard gate, not a weighted score: zero unauthorized effects, zero effects after deny/strict-indeterminate, exactly one successful use under a shared-capability concurrency race, and zero successful replay after the declared restart model.
- Every report separately counts requests, decisions, adapter calls, side effects, independently confirmed postconditions, unknown outcomes, receipt failures, timeouts and missing/incomplete rows. A crash after possible effect is not reclassified as “no effect,” and an exit code alone is not a postcondition.

External context: APort Vault reports 0 unpermitted transfers across 69,297 authorization-layer evaluations at policy Levels 2–4, with 25,370 payments still executed; its dataset had 225,964 completed out of 244,776 planned evaluations, so its own missing rows remain part of the comparison. Mergen's published 551-case suite is a decision-only gate measurement, not an end-to-end side-effect guarantee. APS reports separate verifier, SDK, queued-durable and gateway layers. These are useful methodological baselines; unlike-for-like replication is required before claiming to outperform any of them.


The checker now pins each v1 vector's **full semantic tuple** in Rust: category, setup state, exact mutation paths and JSON scalar values, authorization decision, adapter-invocation count, effect witness, and terminal class. Retaining an ID while changing `signature_valid=false` to `true`, swapping a denial into a permissive outcome, or stripping the setup state from a replay case is a contract-integrity failure. To change a vector deliberately, add a reviewed versioned corpus and checker profile rather than modifying v1 in place.


The next implementation stage is tracked in [ACTION-EFFECT-HARNESS-001 / #8769](https://github.com/Luminous-Dynamics/symthaea/issues/8769): execute the corpus against instrumented effect boundaries and independently witnessed postconditions, reusing the existing Action Runtime rather than adding a parallel authority kernel.
