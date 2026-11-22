# Frontend Engineering Benchmarks

This directory contains the **Symthaea Frontend Engineering Benchmark (SFEB)**.

## Current status

- `sfeb-v0.json` contains 11 staged task specifications; the end-to-end SFEB benchmark remains `specification_only`.
- One local service-dashboard reference fixture and its task contract are checked in as a **public pilot**. This is not a hidden holdout.
- `contract-check/` validates interface/task schemas, source anchors, fixture digests, provenance and retrieval-query references. Its Rust source and tests are authored but have not been compiled or executed here.
- `retrieval-eval/` contains keyword, BM25, TF-IDF, deterministic bipolar-HDC and hybrid-RRF retrieval methods. It has not been compiled or run; no retrieval score exists yet.
- `symthaea-browser` contains the CDP evidence-capture primitive, but the isolated full browser runner and task-level independent oracles remain pending.
- No benchmark score or SOTA claim is implied by these artifacts.

See [the research programme](../../docs/research/SYTHAEA_FRONTEND_ENGINEERING_INTELLIGENCE_2026-11-21.md) for rationale, repository audit, external comparisons, metrics, targets and the staged implementation plan.

## Why a separate benchmark?

Frontend engineering combines multiple capabilities which should not be collapsed into one score:

1. Requirement and interaction-contract understanding.
2. Static visual implementation.
3. Responsive behavior across real and intermediate viewport sizes.
4. Multi-page functionality and state transitions.
5. Accessibility and keyboard operation.
6. Runtime/browser debugging.
7. Service integration, performance and security.
8. Maintenance without regression.

SFEB separates these dimensions and requires independent browser evidence. It complements external benchmarks such as Vision2Web and Design2Code; it does not replace or impersonate them.

## Observation harness

The first Chromium observation primitive reuses the existing
`crates/domains/symthaea-browser` crate and its `chromiumoxide` CDP session.
Do not add a parallel browser-control dependency for the initial Chromium lane.

`CdpSession::capture_frontend_evidence(FrontendEvidenceContext)` produces:

- a redacted accessibility-tree JSON artifact;
- a bounded layout JSON artifact with viewport, document overflow and semantic
  element rectangles;
- a viewport PNG screenshot whose IHDR pixel dimensions are checked against the
  recorded CSS viewport multiplied by `window.devicePixelRatio`;
- a receipt with source/task/fixture identity, environment-manifest digest,
  capture time, final redacted URL, viewport and SHA-256 artifact digests.

Important limitations:

- This is a browser-library capture primitive, not the SFEB runner.
- Capture steps are sequential, not an atomic snapshot.
- The bundle deliberately keeps binary screenshot bytes outside JSON
  serialization.
- The accessibility tree is sanitized, but screenshots and page-derived
  artifacts must still be treated as potentially sensitive.
- The module does not create OS-level network isolation or make an arbitrary
  CDP endpoint trustworthy. Run benchmark browsers in disposable, fixture-only
  network sandboxes; do not expose a host browser debugging port.
- An ignored, no-network Chromium smoke test is provided at
  `crates/domains/symthaea-browser/tests/frontend_capture_smoke.rs`. It runs
  against `about:blank` and must be invoked explicitly on a host with Chrome or
  Chromium installed:
  `cargo test -p symthaea-browser --test frontend_capture_smoke -- --ignored --nocapture`.
  It is not an SFEB task score and has not been executed in this environment.
- The current layout probe is intentionally bounded to semantic elements.
  Reference fixtures should mark important component regions with stable
  `data-sfeb-id` attributes to connect rendered geometry to source components.

## Interface contracts

The **Interface Contract Compiler** starts with a versioned JSON contract:

- `interface-contract-v1.schema.json` — JSON Schema 2020-12 for product intent,
  Leptos/runtime, routes, semantic regions, state machines, data sources,
  responsive constraints, accessibility, security and acceptance journeys.
- `contracts/symthaea-ui-v0.json` — first source-observed contract for the
  existing Converse/Observe interface, tied to an exact source commit.

Contract status is explicit: `source_observed` is not `browser_verified`.
A contract may describe current implementation, proposed behavior or reviewed
requirements, but browser-verified status requires independent runner receipts.
The sample records six journeys, 17 named source regions and the current
telemetry handshake/retry behavior. It deliberately calls out that changing
the gateway URL does not terminate an already-open WebSocket immediately.

## Task and oracle contract

`task-oracle-v1.schema.json` defines a typed task format with a product brief,
implementation constraints, pinned environment requirements, reference-fixture
digest, semantic regions, viewports, journeys, machine-readable assertions,
hard gates, evidence obligations and a bounded task budget.

The first original task is:

- `tasks/sfeb-l1-service-dashboard-v1.json`
- Reference assets: `fixtures/service-dashboard/reference.html` and
  `fixtures/service-dashboard/style.css`
- Eight required semantic regions, eleven assertions, five fixed viewports and
  two acceptance journeys.
- The fixture digest is SHA-256 over the reference HTML bytes, one NUL byte, then
  the CSS bytes. The task validator recomputes it before accepting the fixture.
- The task remains `specification_only` in the SFEB index because Chromium
  screenshots, independent browser oracles and a baseline agent run are not
  available yet. The source fixture assets themselves are checked in.

### Standalone Rust validator

`contract-check` is intentionally outside the main Symthaea workspace and
does not add dependencies to production crates. It validates JSON Schema
conformance, duplicate/missing source anchors, task fixture digest, fixture
region mapping, duplicate assertion IDs, fixture-only resource policy, Git
blob provenance for knowledge examples, and retrieval-query gold-ID integrity.

From the repository root:

```bash
# Unit tests validate the checked-in contract, task definition and fixtures.
cargo test --manifest-path benchmarks/frontend/contract-check/Cargo.toml

# Validate the Leptos source-observed interface contract.
cargo run --manifest-path benchmarks/frontend/contract-check/Cargo.toml

# Validate the SFEB task schema, fixture digest and required region anchors.
cargo run --manifest-path benchmarks/frontend/contract-check/Cargo.toml -- --task

# Validate the Leptos pattern retrieval query set and gold pattern IDs.
cargo run --manifest-path benchmarks/frontend/contract-check/Cargo.toml -- --retrieval
```

The commands are authored but have not been executed in this environment
because the Rust toolchain is unavailable. JSON documents were parsed and the
fixture digest was independently confirmed with `sha256sum`.

## Leptos knowledge corpus

The expanded corpus is pinned to Leptos 0.8.20 (the Cargo.lock resolution). The latest stable 0.8.23 compatibility experiment is kept separate. The corpus is in
`knowledge/leptos-0.8/patterns-v0.json`, with schema
`knowledge/leptos-0.8/pattern-v1.schema.json`. It currently records seventeen
source-grounded patterns for Leptos 0.8 CSR and exact Git blob SHA-1 provenance
for every example:

- component composition, `RwSignal`, tracked/untracked reads and Effects;
- client-side async work, keyed `For` collections and typed `NodeRef`;
- HTTP/WebSocket service boundaries;
- Router / Routes / Route composition and active links;
- typed shared context, derived `Memo` state and programmatic navigation;
- pure reducers, epoch-safe browser event adapters and persistent DOM resources;
- fail-closed authorization and asynchronous initialization states.

The current checked-in `Cargo.lock` resolves Leptos to 0.8.20. Although
upstream has since published stable 0.8.23 (with 0.9.0-rc1 still prerelease),
SFEB's baseline remains pinned to the lockfile version. A 0.8.23 upgrade is
a separate compatibility experiment, not a silent benchmark dependency change.

Patterns have explicit promotion states:
`source_observed_not_compiled`,
`source_observed_reference_checked`, `compiled`, `unit_tested`, and
`browser_verified`. Each promotion requires stronger evidence; source
presence or generated examples do not imply compilation.

Validate the pattern schema and exact Git blob provenance with:

```bash
cargo run --manifest-path benchmarks/frontend/contract-check/Cargo.toml -- --patterns
```

The validator rejects duplicate pattern IDs, unsafe paths, stale source blobs,
and invalid corpus structure. It is authored but not yet executed because the
Rust toolchain is unavailable in this environment.

## Knowledge retrieval evaluation

The first Leptos 0.8 retrieval pilot is defined in
`knowledge/leptos-0.8/retrieval-queries-v0.json`:

- 34 positive queries cover the seventeen source-grounded patterns (two paraphrases each).
- Eight hard negatives represent concepts not yet covered by the corpus.
- The gold answers are exact versioned pattern IDs.
- The public query set is development data, not a hidden holdout.

Validate schema and gold-ID references with:

```bash
cargo run --manifest-path benchmarks/frontend/contract-check/Cargo.toml -- --retrieval
```

The intended controlled arms are keyword exact-match, BM25, TF-IDF, dense
embeddings, HDC associative retrieval and a hybrid. Report recall@1/3, MRR,
nDCG@3, precision@3, negative abstention accuracy, p50/p95 latency, memory
and cost under the same corpus and hardware. Do not claim a Symthaea retrieval
advantage until a private holdout query set has been evaluated.

The first Leptos target (`symthaea-ui`) now exposes four separate
telemetry status labels—disconnected, connecting, connected and reconnecting.
The live status region reports the same state that colors the indicator.
The connected state is entered only after the browser WebSocket reaches
`State::Open`; a handshake timeout or closed stream returns to the
bounded retry path.

## Framework version compatibility

The current checked-in `Cargo.lock` resolves Leptos to **0.8.20**. Upstream has
published **0.8.23** as the latest stable 0.8 release (2026-10-31); **0.9.0-rc1**
is a prerelease. The benchmark baseline remains pinned to the repository's
0.8.20 lock resolution.

A separate, machine-readable seven-gate compatibility experiment is defined in
`compatibility/leptos-0.8.20-to-0.8.23-v1.json`. It covers dependency
resolution, lockfile diff, fmt/Clippy, package tests, WASM builds, browser
regression and performance. It must run in an isolated branch/worktree; do not
change the baseline lockfile or mix a framework update into an agent-quality
comparison.

## Task lifecycle

A task moves through these states:

`inventory -> fixtures-authored -> oracle-reviewed -> runnable -> baseline-measured -> holdout-qualified`

Each transition must be supported by a checked-in artifact or immutable run receipt. A task cannot be called runnable until it has deterministic inputs, declared environment pins, independently reviewed acceptance oracles and a documented cleanup procedure.

## Required environment pins

Every run must record at least:

- source commit and Cargo.lock digest;
- Rust toolchain and resolved Leptos version;
- browser build and OS image;
- installed fonts, viewport and device scale factor;
- fixture service revision and network policy;
- evaluator revision and artifact digests.

Changing any of these creates a distinct benchmark environment. Do not pool results from different environments without an explicit compatibility analysis.

## Runnable retrieval evaluator

The public Leptos retrieval pilot now has an offline Rust evaluator at
`retrieval-eval/`. It implements keyword overlap, BM25, TF-IDF,
deterministic 16,384D bipolar HDC, and reciprocal-rank fusion over BM25,
TF-IDF and HDC. It reports recall@1/3, MRR, nDCG@3, precision@3, per-query
rankings, top-score distributions for hard negatives, and p50/p95 latency.

Run it from the repository root:

```bash
cargo test --manifest-path benchmarks/frontend/retrieval-eval/Cargo.toml
cargo run --release --manifest-path benchmarks/frontend/retrieval-eval/Cargo.toml
```

The result should be retained and validated against
`retrieval-eval/result-v1.schema.json`, together with exact source commit,
Cargo.lock, Rust toolchain and hardware metadata. The HDC encoder is a
transparent research baseline—not Symthaea's full learned HDC subsystem.
Dense embeddings still require an explicitly pinned model adapter. The tool
does not select an abstention threshold; score distributions are emitted so
threshold calibration can occur on a separate development set.

The runner source and tests are authored but have not been compiled or run in
this environment.

## Fixture exposure and benchmark integrity

Every task declares `fixture_exposure`:

- `public_pilot`: task and reference source are public. Useful for developing
  the runner and comparing iterative improvements, but never a hidden holdout.
- `controller_only`: fixture source is outside the agent workspace; the
  evaluator mounts it into the controller and provides only the allowed
  screenshots/brief to the agent.
- `hidden_holdout`: task, fixture, reference captures and detailed oracle
  remain withheld until an evaluation run. Hidden tasks must not be committed
  to a public repository.

The service-dashboard task is explicitly `public_pilot`: its HTML/CSS fixture
is checked into this public branch. It must not be counted as blind performance
or anti-contamination evidence. Future holdouts require separate private
storage, sparse or isolated agent workspaces, and blocked external fixture
retrieval. Public-pilot and hidden-holdout scores must always be reported
separately.

## Scoring

Publish the quality vector and the hard-gate outcomes per task. Report repeated trials, confidence intervals, cost and wall time. A failed required journey, critical accessibility blocker, security violation, missing receipt or omitted task is a failure regardless of visual score.

Never let the tested agent be the only evaluator of its own output. Use deterministic assertions, independent browser observation, blinded human review where visual judgment is required, and hidden holdout tasks.

## Next implementation steps

1. Author original reference briefs and screenshot assets with clear provenance.
2. Add fixture-backed routes and service behavior for the selected tasks.
3. Build the Rust-owned runner on the existing `symthaea-browser` CDP substrate; evaluate WebDriver BiDi only as a later cross-browser adapter.
4. Implement the runner and immutable receipt schema.
5. Run the existing `symthaea-ui` as the baseline.
6. Run controlled ablations: base model, retrieval, interface contract, browser observation/repair, and full Symthaea integration.

Until those steps are complete, SFEB remains a research specification, not a measured capability.


## Benchmark-oracle qualification

Before comparing agents, qualify the evaluator itself. The seeded mutation campaign in
[`qualification/README.md`](qualification/README.md) defines 20 negative controls,
5 semantics-preserving positive controls and 5 metamorphic relations. It is a
qualification plan, not an executed result. Critical security and evidence-integrity
mutants must be rejected; harmless controls must remain accepted. Do not publish SFEB
leaderboard claims until the independent oracles pass this campaign on a pinned runner.

## Validate the oracle-qualification plan

The standalone Rust contract checker now has a `--mutations` mode:

```bash
cargo test --manifest-path benchmarks/frontend/contract-check/Cargo.toml
cargo run --manifest-path benchmarks/frontend/contract-check/Cargo.toml -- --mutations
```

This validates the qualification campaign schema, unique IDs, expected dispositions,
minimum case counts and non-weakenable 100% critical/security/evidence-integrity
targets. It validates the plan only; it does not execute seeded mutants, run a browser,
or certify oracle quality. These commands are authored but remain unexecuted here.


## Paired qualification report adjudicator

The standalone Rust contract checker now also has a
`--qualification-report` mode. It validates a completed 30-case report against
the frozen campaign and report schema, verifies exact receipt bytes and the
screenshot/accessibility/layout artifact bytes, checks receipt identity and
fixture/environment pairing, then emits mutation kill rates with 95% Wilson
intervals and positive/metamorphic-control outcomes.

See [the execution protocol](qualification/EXECUTION.md). This is an evidence
adjudicator, not the browser mutation executor. It does not run Chromium, create
mutations, or cryptographically authenticate the runner's reported oracle IDs.
The current SFEB benchmark remains `specification_only`.


## Comparative evaluation protocol

The machine-readable study design is in
[`comparative-evaluation-v1.json`](comparative-evaluation-v1.json) with a
matching schema. It defines matched arms A0–A4 plus a human reference arm,
task-block randomization, task-cluster bootstrap, exact McNemar testing,
Wilson intervals, multiple-comparison correction, judge blinding and
publication/holdout scale targets. It is a protocol, not an executed result.


## Reference-fixture static audit

The standalone task validator now checks a static, independently-auditable subset
of the public service-dashboard reference: exactly one `main` landmark and one
named `h1`, the skip-link target and focus CSS, named workspace/summary/activity/
schedule regions, exactly three metric-card source anchors, declared desktop and
mobile grid rules, and absence of scripts/external asset imports. Tests also seed
a missing heading and a breakpoint regression while recomputing the fixture digest.

This audit qualifies the **reference fixture**, not the evaluated agent's
rendered page. Runtime reflow, keyboard reachability, accessibility-tree
behavior, network egress and actual screenshot geometry still require browser
execution.


The emitted qualification summary is validated against
[`qualification-summary-v1.schema.json`](qualification/qualification-summary-v1.schema.json).


## Trusted execution and causal repair

The next assurance layer is specified in [attestation/](attestation/README.md):
in-toto run statements, hash-chained browser observation traces, isolated
execution requirements, independent oracle derivation and evidence-linked
causal repair. These are authored schemas and design protocols, not an
implemented executor or measured benchmark result.


## External comparator ledger and independent oracle replay

- [External comparator ledger](external-comparator-ledger-v1.json) records benchmark scope, dated snapshots and non-comparability constraints.
- [Independent oracle verification predicate](attestation/sfeb-oracle-verification-predicate-v1.schema.json) defines the separately signed verifier result; it is a contract, not an implemented replay engine.
- See [the oracle replay and benchmark frontier note](../../docs/research/SFEB_INDEPENDENT_ORACLE_REPLAY_AND_BENCHMARK_FRONTIER_2026-11-22.md).


### New external comparisons (2026-11-22)

The external comparator ledger now includes Arena Code Arena's WebDev, Image-to-WebDev and Fullstack preference boards, and Interaction-as-a-Judge research. These are distinct human-preference / interaction-judge dimensions and must not be blended with SFEB's deterministic oracle qualification or Vision2Web scores. See `external-comparator-ledger-v1.json` and the linked research note.


## Evaluator and judge qualification

- [Judge qualification plan](judge-qualification-v1.json) and [schema](judge-qualification-v1.schema.json) compare deterministic, code-only, visual, interaction-grounded and bounded-ensemble evaluators.
- [Judge qualification research note](../../docs/research/SFEB_JUDGE_QUALIFICATION_2026-11-22.md) defines authority separation, calibration, adversarial testing and Arena dataset rights constraints.
