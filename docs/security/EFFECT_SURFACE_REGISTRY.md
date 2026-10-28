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
