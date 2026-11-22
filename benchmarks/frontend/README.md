# Frontend Engineering Benchmarks

This directory contains the **Symthaea Frontend Engineering Benchmark (SFEB)**.

## Current status

- `sfeb-v0.json` is a machine-readable task inventory.
- The manifest status is `specification_only`.
- There is no runnable SFEB evaluator yet.
- Reference assets, fixture services, browser automation, independent oracles and result receipts must be added before any score can be reported.
- No benchmark result or SOTA claim is implied by the existence of the inventory.

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

The versioned starter corpus is in
`knowledge/leptos-0.8/patterns-v0.json`, with schema
`knowledge/leptos-0.8/pattern-v1.schema.json`. It currently records eight
source-grounded patterns for Leptos 0.8 CSR and exact Git blob SHA-1 provenance
for every example:

- function components and view composition;
- `RwSignal` local state;
- tracked reads versus `get_untracked()`;
- Effects and external side effects;
- client-side async work with `spawn_local`;
- keyed list rendering with `For`;
- typed `NodeRef`;
- the HTTP/WebSocket service boundary.

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

- 16 positive queries paraphrase the eight source-grounded patterns.
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
