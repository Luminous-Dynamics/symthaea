# Symthaea Frontend Engineering Intelligence

**Programme:** SFEB — Symthaea Frontend Engineering Benchmark  
**Version:** 0.1.0  
**Status:** Research specification; baseline not yet measured  
**Date:** 2026-11-21  
**Repository baseline:** `36eb3adc1dc2d5ad1ae643863fee828d594472d4`

## Executive decision

Use the existing Leptos applications as the first learning and evaluation substrate. Keep the current `0.8` dependency family for the initial baseline; do not upgrade the framework as a side effect of this programme. Build a benchmark and an evidence pipeline before claiming that Symthaea can independently engineer production-quality interfaces.

The research target is not "generate more Leptos code." It is:

> Given a product brief and constraints, Symthaea can construct an explicit interface contract, design and implement the application, run it in an isolated browser, observe the actual result, diagnose failures, repair them, and provide reproducible evidence that the accepted result satisfies the contract.

This specification introduces a staged programme and a task inventory. It does **not** claim that the benchmark runner, browser observation pipeline, or target scores are implemented or passed.

## 1. Repository audit at the baseline commit

The audit was performed against the public `main` head recorded above.

| Surface | Verified state | Consequence |
|---|---|---|
| `crates/bridges/symthaea-web` | Leptos `0.8` CSR; `leptos_router`; Trunk; WASM worker and application state | Existing portal is the first source of framework patterns. |
| `crates/bridges/symthaea-ui` | Leptos `0.8` CSR; service gateway boundary; Converse and Observe scope | Smaller, isolated candidate for the first full engineering exercise. |
| `crates/domains/symthaea-muse-ui` | Leptos application with separate Studio, Listen and Research design specifications | Existing product-design material can seed interaction and information-architecture tasks. |
| `crates/bridges/symthaea-ui/src/app.rs` | Typed Rust state and gateway parsing; telemetry, status, chat, visual projections | Useful code-understanding, state-flow and regression tasks. |
| `crates/bridges/symthaea-ui/style.css` | Responsive auto-fit readouts, a focused visual language and basic focus styling | A useful baseline, not proof of comprehensive responsive/accessibility behavior. |
| `crates/bridges/symthaea-web/style/main.css` | Existing palette tokens and component styling | Candidate source for design-token extraction and consistency checks. |

Two concrete scope observations matter:

1. The `symthaea-ui` README says its v0 surface is Converse + Observe and explicitly identifies Steward and Bench panes as not yet built. This provides a bounded product extension task.
2. The `symthaea-ui` root app documents gateway reconnection on URL change as a future nicety. This provides a genuine state-lifecycle and failure-recovery task, rather than an invented toy bug.

The two bridge applications currently declare Leptos `0.8` and CSR features in their Cargo manifests. The baseline is therefore a Leptos learning programme, not an introduction of a new framework.

### Framework version policy

The upstream release page consulted on 2026-11-21 lists `0.8.22` and `0.9.0-rc0`. The latter is a release candidate. The repository declares the `0.8` semver family, with exact dependency resolution governed by the lockfile. Keep the initial benchmark on the checked-in lockfile and record the resolved versions in every run receipt. Treat 0.9 migration as a separate compatibility experiment; never mix framework migration with a frontend-quality comparison.

## 2. Paradigm shifts to investigate

These are testable hypotheses, not established advantages of HDC, CfC, or any other Symthaea subsystem.

### 2.1 Interface Contract Compiler (ICC)

Replace the prompt-to-code-only model with a structured, intermediate interface representation.

A contract should explicitly represent:

- product goal, user roles and user journeys;
- routes, page regions and semantic content hierarchy;
- components, properties and composition boundaries;
- interaction state machines and transition preconditions;
- design tokens, typography, spacing and visual hierarchy;
- responsive constraints and supported viewport ranges;
- accessibility semantics, keyboard behavior and focus order;
- data contracts, loading states, empty states and error behavior;
- security boundaries and actions that require user approval;
- measurable acceptance criteria and their independent oracles.

The same contract should drive implementation planning and test generation. Code and tests derived from one contract can still share a mistake, so the evaluator must also include independent checks and human review.

**Research hypothesis:** reducing ambiguity before implementation will improve first-pass correctness and reduce repair cost on multi-step tasks.

### 2.2 Browser-grounded causal debugging

A screenshot mismatch is an observation, not a diagnosis. Build a repair loop that gathers:

1. screenshot at a declared viewport and device scale;
2. DOM structure and computed geometry for target regions;
3. accessibility tree and keyboard/focus observations;
4. console errors and failed network requests;
5. route and interaction traces;
6. performance measurements;
7. build/test output tied to the exact source revision.

Each benchmark element should have a stable semantic identifier that can be mapped back to a component or source region. A repair should name the observed failure, the hypothesized cause, the source change, and the post-change evidence. Where feasible, use controlled interventions (change one variable, replay the same trace) to distinguish root cause from correlation.

**Research hypothesis:** structured observations and controlled repair will outperform screenshot-only critique on defect localization and regression rate.

### 2.3 Accessibility by construction

Accessibility should be encoded in the component contracts rather than appended during final polish. Prefer semantic HTML, explicit labels, keyboard-operable controls, visible focus, reduced-motion support, status announcements and predictable focus management. Custom widgets need an explicit keyboard and assistive-technology interaction contract.

Automated checks are necessary but insufficient: a passing scanner does not prove that a complete user journey is usable with a keyboard or screen reader.

### 2.4 Responsive design as constraints, not device templates

Represent layout behavior using constraints: minimum and maximum widths, intrinsic content size, wrapping rules, grid/flex relationships, readable line lengths and interaction target dimensions. Viewports are test samples; they are not the whole responsive design.

Test at fixed reference widths and at randomized widths between them to find discontinuities, overflow, collisions and brittle breakpoint assumptions.

### 2.5 Evidence-linked design memory

Store patterns as versioned, provenance-carrying knowledge records:

- framework and dependency versions;
- pattern or anti-pattern;
- source example and license/provenance;
- intended context and preconditions;
- known failure modes;
- verification tasks and results;
- confidence and last revalidation date.

HDC may be useful for associative retrieval across design intent, component role and failure signatures. CfC may be useful for modelling event sequences and temporal interaction patterns. These are experimental retrieval/modeling backends: compare them with sparse keyword search, embeddings and conventional typed indexes on retrieval precision, latency, memory and downstream task success. Do not grant architectural novelty as evidence of performance.

### 2.6 Human-governed engineering

Symthaea may propose, implement and test changes in an isolated branch/worktree. She must not silently deploy, alter production data, change credentials, introduce dependencies, weaken gates, or commit destructive behavior. A user-visible review boundary remains before merge or deployment. Unknown execution outcomes must be reconciled before retrying mutations.

## 3. Proposed architecture

```text
Product brief + constraints
          |
          v
Interface Contract (typed, versioned, reviewable)
          |
          v
Design / component / state plan
          |
          v
Leptos implementation + generated test obligations
          |
          v
Static gates: fmt, clippy, native tests, WASM build
          |
          v
Isolated browser execution (fixed environment and fixtures)
          |
          v
Observation bundle: screenshots, DOM, AX, traces, console, network, perf
          |
          v
Independent evaluators + hard acceptance gates
          |
          +---- failure evidence ----> repair hypothesis / patch
          |                                  |
          +<----------- replay --------------+
          |
          v
Human review + immutable evidence receipt
```

### Trust boundaries

- Treat requirements, reference assets, remote API responses and generated code as untrusted input.
- Run browser workloads in a disposable, network-restricted environment with explicit fixture allowlists.
- Do not expose host credentials, local filesystem contents or deployment tokens to generated code.
- Pin browser version, fonts, viewport, device scale, OS image, dependencies and network fixtures.
- Keep benchmark fixtures deterministic and versioned.
- Do not let the agent be the sole judge of its own output.
- Store artifact digests, source SHA, environment manifest, command lines, test results and evaluator versions in each receipt.

### Suggested initial integration boundary

Start with `symthaea-ui`, not a rewrite of `symthaea-web`. It is the smaller service-gateway application and its README already defines a narrow v0 product. Use it to prove the loop. Keep the Leptos/WASM UI independent from the large Symthaea engine, preserving the current gateway boundary.

### Existing browser substrate discovered during implementation

The repository already contains `crates/domains/symthaea-browser`, a CDP browser agent built on `chromiumoxide`. It is not necessary to introduce a second browser automation dependency for the first SFEB lane.

Verified existing capabilities include:

- Chromium launch or attachment through `CdpSession`;
- bounded accessibility-tree observations with redaction and URL sanitization;
- full-page PNG screenshots;
- capability- and policy-governed browser mutations through `BrowserExecutor`;
- URL scheme, origin, credential and local-network checks (with an explicit warning that a production network boundary must additionally pin DNS resolution and re-check redirects).

This programme extends that substrate with `symthaea_browser::frontend`:

- a fixed-script, bounded layout probe for semantic elements;
- optional `data-sfeb-id` identifiers linking reference regions to rendered components;
- viewport and document overflow observations;
- a capture bundle containing separate PNG, accessibility JSON and layout JSON artifacts;
- SHA-256 artifact digests, byte lengths, a receipt digest and tamper verification;
- sensitive-field and URL-secret redaction before serializing the accessibility artifact.

The binary bundle deliberately does not implement JSON serialization: screenshot bytes remain a separate artifact instead of becoming an accidental unbounded JSON byte array.

**Capture limitation:** accessibility, layout and screenshot observations are sequential, not an atomic browser transaction. SFEB fixtures must freeze animations, use deterministic fixture services, wait for fonts/layout readiness, and capture under a pinned browser environment. The capture primitive does not by itself enforce OS-level network isolation or make an arbitrary existing CDP endpoint trustworthy.

For local fixture applications, use a disposable browser/network namespace and a fixture-only allowlist. Do not expose the host browser's debugging port or enable private-network access on a general-purpose browser session merely to make localhost tests work.

Start with `symthaea-ui`, not a rewrite of `symthaea-web`. It is the smaller service-gateway application and its README already defines a narrow v0 product. Use it to prove the loop. Keep the Leptos/WASM UI independent from the large Symthaea engine, preserving the current gateway boundary.

## 4. SFEB benchmark design

SFEB is an internal, versioned benchmark. It complements external leaderboards; it must not be described as an independent public leaderboard until its tasks, fixtures, scoring, and submissions are published with reproducibility instructions.

### Levels

| Level | Task class | Required evidence |
|---|---|---|
| L0 — Understand | Requirement extraction, design critique, component/state architecture, version-aware API selection | Contract completeness, correct dependency/version references, human rubric |
| L1 — Static | Single-page interface from a brief or reference image | Screenshot captures at desktop/tablet/mobile, visual metrics, semantic DOM checks |
| L2 — Interactive | Multi-page flows, forms, reactive state, loading/error/empty states | Independent browser action traces and expected-state assertions |
| L3 — Integrated | Complete frontend connected to a fixture-backed service | End-to-end journeys, contract checks, security boundary checks, performance and regression evidence |
| L4 — Maintain | Diagnose seeded defects, add features without regressions, migrate framework/API versions | Defect localization, patch quality, full replay, compatibility and regression reports |

### Task families

The initial suite inventory is in `benchmarks/frontend/sfeb-v0.json`. It includes:

- screenshot/reference reconstruction;
- responsive layout under intermediate viewport widths;
- reusable component extraction;
- form validation and accessible errors;
- navigation and state persistence;
- keyboard and reduced-motion behavior;
- visual and interaction defect repair;
- gateway reconnect and failure recovery;
- observability / status visualization;
- full-stack feature extension with fixture-backed APIs.

The manifest is a **specification-only inventory**. Reference screenshots, deterministic browser fixtures, independent oracles and a runner still need to be authored. A task is not runnable merely because it appears in the JSON inventory.

### Scoring and anti-gaming policy

Report a vector, not a single unexplained score:

- **Visual fidelity:** region geometry, text/layout alignment, image similarity and human preference.
- **Functional correctness:** passed required journeys divided by required journeys; report each flow.
- **Responsive robustness:** pass rate across fixed and randomized viewports; overflow and overlap counts.
- **Accessibility:** automated findings by severity plus manual keyboard/task checks.
- **Performance:** LCP, INP, CLS, transfer size, WASM size and interaction latency, in a controlled environment.
- **Engineering quality:** compile/build, clippy, tests, dependency changes, code complexity, maintainability review.
- **Reliability:** task completion across repeated seeds, flaky test rate, repair success and regression rate.
- **Efficiency:** wall time, model/tool calls, token use, compute and cost per accepted task.
- **Calibration:** confidence versus actual task success and correctness of self-reported evidence state.

Hard gates override weighted averages. A broken required journey, critical accessibility blocker, security violation or missing evidence receipt cannot be compensated for by a beautiful screenshot. Publish raw per-task outcomes and confidence intervals, not only aggregates.

For image-based evaluation, combine deterministic geometric/region metrics, image metrics and blinded human assessment. Do not optimize only for a single SSIM/CLIP/VLM score; those measures can reward superficial similarity while missing semantic or interaction defects. Freeze evaluation assets before submission and keep hidden holdout tasks to detect overfitting.

### Executable contract and first fixture-backed task

This programme now has a first machine-readable interface contract and a first
original design task, rather than only a benchmark outline.

- `benchmarks/frontend/interface-contract-v1.schema.json` defines JSON Schema
  2020-12 for product intent, Leptos runtime, routes, named regions, state
  machines, data sources, responsive constraints, accessibility, security and
  independent acceptance journeys.
- `benchmarks/frontend/contracts/symthaea-ui-v0.json` describes the source
  observed Converse/Observe application: 16 named regions and six journeys.
  It remains `source_observed`, not `browser_verified`.
- `benchmarks/frontend/task-oracle-v1.schema.json` defines the SFEB task and
  assertion contract.
- `benchmarks/frontend/tasks/sfeb-l1-service-dashboard-v1.json` defines the
  first Leptos design task: an operational dashboard with 11 assertions, eight
  required regions, five fixed viewports and two acceptance journeys.
- `benchmarks/frontend/fixtures/service-dashboard/` contains an original,
  static HTML/CSS fixture with no remote fonts, scripts or images. Its canonical
  digest is SHA-256(HTML bytes || NUL || CSS bytes), stored in the task file.

A standalone Rust utility at `benchmarks/frontend/contract-check` validates
the JSON Schemas, the source-observed contract's exact `data-sfeb-id` mapping,
the task's fixture digest, required fixture regions, unique assertion IDs and
the fixture-only resource policy. Tests for the checked-in contract, task and
tamper/mismatch negative controls are authored.

The standalone validator is intentionally outside the primary workspace and
does not add dependencies to production crates. It is not yet executed because
this environment has no Rust toolchain. The HTML/CSS digest was independently
confirmed using `sha256sum`. A Chromium headless screenshot attempt in the
available container timed out before producing a capture; the browser-render
gate therefore remains blocked, not passed. The task stays
`specification_only` in the SFEB index until its screenshot baselines,
independent oracles and browser runs exist.

The first Leptos implementation also received 16 stable `data-sfeb-id` anchors,
semantic live/status/alert regions, responsive adjustments, visible focus and
reduced-motion behavior. Its telemetry indicator now waits for the actual
gloo-net 0.5 `WebSocket::state() == State::Open` state, has a bounded 10-second
handshake wait and retries sequentially with capped backoff. The backoff is
deterministic (1, 2, 4, 8, 16, then 30 seconds), making it testable. The current
gateway URL is picked up on the next retry; an already-live connection is not
forcibly replaced.

## 5. External comparison and state of the field

As rechecked on 2026-11-22:

- **Vision2Web** is the closest external end-to-end comparator found in this research. Its official benchmark describes 193 tasks across 16 categories, 918 prototype images and 1,255 test cases, with static, interactive and full-stack levels and verification combining workflow-driven GUI testing with VLM visual judging. The search-indexed S4-2026 leaderboard snapshot (refreshed 2026-10-01) showed 82.3 overall for Claude-Fable-5 + Claude Code on 2026-08-28, with level averages 87.4 static, 83.2 interactive and 76.3 full-stack. The live page's selected season displayed no rows when rechecked on 2026-11-22. These are dated historical reference scores, not a verified current leader or directly transferable SFEB targets.
- **Design2Code** (2024/NAACL 2025 publication track) contains 484 real-world webpages and evaluates screenshot-to-code rendering with high-level CLIP similarity, block matching, text, position and color metrics, complemented by human pairwise judgments. Its analysis identifies visual-element recall and layout correctness as recurring weaknesses. Use it as a visual-reconstruction reference, subject to task/data license and reproducibility conditions.
- **WebdriverIO's browser-agent benchmark** is a controlled tool comparison on a fixed random sample of Online-Mind2Web live-site tasks. It holds model, harness and prompt constant while swapping browser tools and uses an independent judge; its October 2026 page snapshot reported 62% for WebdriverIO session. This measures browser-operation performance, not frontend code generation. The methodology warns of roughly ±10-point intervals on its small sample, site instability and judge disagreement on about one in seven runs, so treat close scores as ties rather than a definitive SOTA ranking.
- **Leptos upstream** is the API authority. Pin and record the actual lockfile version rather than treating a semver requirement as a resolved version.

External benchmarks measure different capabilities. Do not collapse visual generation, browser operation, software engineering, full-stack completion and framework expertise into one cross-benchmark ranking.

### Fair comparison protocol

1. Fix the task set, prompt, time budget, tool access and browser environment.
2. Compare the same base model with and without Symthaea's frontend knowledge/retrieval and engineering loop.
3. Include a strong general coding agent and a human Leptos engineer as reference arms.
4. Run multiple trials per task; report paired differences and confidence intervals.
5. Record cost and wall time as well as quality.
6. Keep a hidden holdout split and preregister task acceptance criteria.
7. Release exact commits, fixtures, environment manifests and raw result receipts where licensing permits.

The key question is not merely "does Symthaea beat model X?" It is whether the Symthaea architecture creates a reproducible improvement over the same model and tools: better success, fewer regressions, lower repair cost, better accessibility, or more reliable evidence.

## 6. Target milestones (goals, not achieved results)

These are proposed programme targets. They must be calibrated after the first baseline run.

| Milestone | Proposed acceptance target |
|---|---|
| M0 — Reproducible baseline | 100% of task definitions versioned; 100% of runs record source/environment/evaluator digests; no unreported task omissions |
| M1 — Static competence | At least 90/100 aggregate visual score on the held-out internal L1 set; no target viewport below 85; zero critical semantic errors |
| M2 — Interactive competence | At least 88% required journey pass rate; no critical keyboard trap; all required form errors and status transitions observable |
| M3 — Full application | At least 82% L3 end-to-end completion; all security and evidence hard gates pass |
| M4 — Maintenance | At least 80% seeded-defect repair success; regression rate below 5%; every patch tied to reproduced failure evidence |
| M5 — Competitive claim | Statistically supported improvement over the same base model/harness on a preregistered holdout, without worse critical safety/accessibility outcomes |

Do not compare these internal scores numerically with Vision2Web's public score unless we reproduce its exact scoring pipeline. The internal targets are directional gates, not claims of SOTA.

## 7. Knowledge curriculum: Leptos first

The first knowledge corpus should be versioned and tested against the repository's actual dependency set.

1. Rust/WASM build model, Trunk, wasm-bindgen and browser APIs.
2. Leptos component composition, props, children and ownership.
3. Signals, memos, effects, derived state and lifecycle hazards.
4. Routing, navigation, URL state and error boundaries.
5. Forms, validation, async resources, loading/error/empty states.
6. SSR, hydration and server functions as a separate deployment track (the current bridge apps are CSR).
7. CSS layout, design tokens, responsive constraints, typography and motion.
8. HTML semantics, keyboard interaction, focus management and WCAG 2.2.
9. Browser rendering, paint/layout, performance profiling and image/asset optimization.
10. Web security, CSP, untrusted content, supply-chain and sandboxing.
11. Testing: unit/property tests, browser acceptance, visual regression and seeded fault injection.
12. Maintenance: API compatibility, framework migration, refactoring and regression control.

Each lesson/example must have: version, runnable code, expected output, negative test, provenance and a last-verified commit. Examples that fail on the pinned baseline are quarantined, not silently treated as knowledge.

## 8. First engineering increments

### Increment A — now

- Establish this specification and the task inventory.
- Keep the current Leptos 0.8 dependency family.
- Choose `symthaea-ui` as the first bounded engineering target.
- Draft original, license-clean UI briefs and deterministic service fixtures.
- Decide the browser adapter after a small Rust-first proof of concept; compare CDP and WebDriver integrations before adding a dependency.

### Increment B — runner and fixtures

- Reuse `symthaea-browser::CdpSession` and `capture_frontend_evidence`; do not add another browser-control framework for the initial Chromium lane.
- Run the ignored `frontend_capture_smoke` test on a host with Chromium installed, then build the Rust-owned SFEB runner/orchestrator around the verified capture API and a pinned disposable Chromium environment.
- Add fixed viewport and intermediate-width capture. The current layout probe records viewport, document overflow and bounded semantic element rectangles; `data-sfeb-id` anchors should be added to original fixture components.
- Extend the receipt pipeline with exact source SHA, Cargo.lock digest, resolved Leptos version, browser build, OS image, font set, fixture revision, network policy and evaluator revision.
- Record route/action traces, browser console, failed requests and performance observations alongside the three current capture artifacts.
- Create independent assertions for each task and at least one hidden holdout partition.

### Increment C — baseline and ablations

- Run current repository code as the baseline.
- Run the same model/harness with documentation retrieval only.
- Run with the structured interface contract.
- Run with browser observation and repair.
- Run with HDC/CfC memory versus conventional retrieval.
- Report paired outcomes, ablations, costs and confidence intervals.

### Increment D — first production-relevant task

Extend `symthaea-ui` with one bounded, reviewed feature from its README's deferred scope (Steward or Bench), only after its user journey and service contract are made explicit. Include loading, error, empty, reconnect and accessibility behavior. Do not add a visual feature without browser acceptance evidence.

## 9. Evidence status for this increment

| Item | Status |
|---|---|
| Public repository baseline inspected | **Verified** at `36eb3ad` |
| Existing Leptos application surfaces inspected | **Verified** by source/manifests |
| Existing `symthaea-browser` CDP substrate inspected | **Verified** by source |
| Frontend layout/AX/screenshot evidence bundle | **Implemented in source**; compile/tests not run |
| Source anchors and first UI accessibility/responsive refinements | **Implemented in source**; browser validation not run |
| Bundle unit tests and ignored Chromium smoke test | **Authored**; not executed |
| External benchmark landscape researched | **Sources reviewed**; benchmark protocols differ |
| Programme, task/oracle schemas and source-observed contract | **Authored** in this branch |
| First static design fixture and task definition | **Authored**; canonical digest confirmed with `sha256sum` |
| Standalone Rust contract/task validator | **Implemented in source with tests authored**; compilation/tests not run |
| Runnable SFEB browser evaluator and screenshot baselines | **Not implemented**; Chromium capture timed out in this container |
| Browser execution on SFEB fixtures | **Not run** |
| Symthaea frontend competence baseline | **Not measured** |
| SOTA superiority | **Not claimed** |
| CI | **Not awaited; no CI pass claimed** |

## 10. Decision rule

The programme advances only when the next increment adds one of:

- a runnable benchmark task with an independent oracle;
- a reusable, version-checked Leptos capability;
- a browser-observed defect with a reproduced repair;
- a statistically supported improvement over a matched baseline;
- a security, accessibility or reliability gate that prevents a real failure.

More architecture, more generated components or more benchmark names are not progress by themselves. The unit of progress is an independently verifiable improvement in a real interface-engineering task.
