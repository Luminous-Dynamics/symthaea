# SFEB-L2-005 candidate and oracle sensitivity report

**Date:** 2026-11-22  
**Status:** Leptos candidate authored; independent model preflights and static-reference Chromium preflight performed; Rust/WASM candidate and isolated evaluator qualification pending.

## Candidate

A Leptos 0.8.20 CSR implementation now exists at `benchmarks/frontend/solutions/sfeb-l2-search-filter-reference/`. It imports the same pure Rust evaluator and query-state URL codec used by the contract checker, operates on the local synthetic dataset, and implements search, conjunctive filters, inclusive integer-cent prices, deterministic sort/ties, pagination and deep-link state. The UI source includes labelled controls, skip navigation, focus styling, semantic table headers and responsive rules.

The candidate has not been compiled, unit-tested, formatted, linted, built to WASM or opened in Chromium. Rust, Cargo, rustfmt and the WASM target are absent here. Source presence does not qualify the task as runnable.

## Benchmark-the-benchmark

Independent JavaScript cross-checking against the exact branch JSON—not Rust execution—produced:
- 11/11 golden cases matched.
- 5,000/5,000 generated query states passed pagination reconstruction, uniqueness, page-count, availability-subset and URL-state round-trip properties.
- 9/9 semantic fault simulations were detected: case-sensitive search, name-only search, OR filters, exclusive price bounds, input-order tie handling, paginate-before-filter, swapped reversed bounds, omitted URL fields and page-only restoration.
- Each of the eight query-state fields was independently omitted and detected.
- Metamorphic checks passed for ASCII case invariance, stable ID ties after dataset permutation and exact inclusive price boundaries.

These are evaluator/model sensitivity simulations, not a mutation campaign against a compiled candidate. The synthetic data now carries an inert `<img onerror>` text canary; static inspection confirms Leptos text interpolation and no raw-HTML APIs. The actual browser DOM assertion remains unexecuted, so XSS mutation detection is not included in the 9/9 result.

## Local browser preflight

The fetched static reference HTML/CSS was loaded into an in-memory Chromium/Playwright page. An extended integer-width sweep covered every CSS width from 320 through 1440 (1,121 widths), in addition to the five named matrix widths. Every sampled width retained one main landmark, one H1, visible form controls and a table constrained to its overflow wrapper. The first run found a real 390px horizontal-overflow defect: numeric inputs retained intrinsic width and expanded the document to 453px. The checked-in stylesheet is constrained with `.field input,.field select{width:100%;min-width:0}`. A repeat sweep showed document width equal to viewport at all five widths, with intentional table scrolling confined to the table wrapper. Keyboard Tab focused the visible skip link with a solid outline; reduced-motion media emulation was detected; no external requests occurred.

This is a browser preflight of the static reference fixture, not the Leptos/WASM candidate or the isolated production evaluator. It does not prove full keyboard journey completion, accessibility-tree compliance, network egress enforcement or browser-based injection safety.

## Industry frontier

- Vision2Web's ICML 2026 paper describes 193 tasks across 16 categories, 918 prototype images and 1,255 test cases, spanning static, interactive and full-stack website engineering. Its leaderboard snapshot last updated 2026-10-01 showed an 82.3 overall example dated 2026-08-28; S4-2026 had no results at that update. Treat the score as a dated comparator, not a current SFEB baseline.
- Code Arena WebDev's 2026-11-22 page reports 1,044,803 votes over 164 models; Claude Fable 5.2 (xHigh) is shown at 1,822. This is human preference, not deterministic correctness.
- WebdriverIO's 2026-10-09 browser-agent snapshot reports 62% success for its session tool, median $0.013 and 156 seconds/task. It swaps browser tools under a fixed model/harness on live-web tasks; it is not a website-generation benchmark.
- Design2Code provides a 484-real-webpage screenshot-to-code comparison, useful for visual reconstruction but not sufficient for stateful engineering.
- Interaction-as-a-Judge reports preference prediction 65.6→69.1%, requirement verification 77.0→81.6%, and repair within five attempts 55.4→75.5%. Interaction judges should provide diagnostics and repair feedback, never override critical independent oracles.

Scores are on different protocols and scales. Publish any Vision2Web comparison only after matching task category, model, tools, budget, browser and evaluation rules.

## Paradigm shifts and targets

SFEB should measure verified engineering change rather than screenshots alone: compile product intent into an executable interface contract; diagnose causally from source anchors, DOM geometry, accessibility tree, screenshots, traces and network observations; treat every fix as a counterfactual regression experiment; bind raw evidence to exact source/browser/fixture identities; and continuously qualify the benchmark using negative, positive and metamorphic controls.

Proposed targets, not achieved results: 100% critical/security/evidence fault detection; ≥95% high-severity and ≥90% overall seeded fault detection; zero positive-control false rejects; 100% metamorphic pass; ≥90/100 visual quality; ≥90% interactive journeys; ≥85% full-stack pass; ≥95% keyboard completion; ≥85% repair within three bounded attempts; <5% unrelated-journey regressions; ≤70% matched-baseline cost per accepted task; ECE ≤0.10.

The 30-task corpus remains a plan: one fixture is fixture-ready and 29 briefs remain planned. The task-family manifest correctly retains `runnable_task_count=0` until Rust/WASM and trusted browser/oracle qualification pass.


## Reproducible independent oracle command

Added `benchmarks/frontend/independent-oracle/search-filter-reference.mjs`
with no third-party packages. Run from the repository root:

```sh
node benchmarks/frontend/independent-oracle/search-filter-reference.mjs
```

Executed locally with Node.js v22.16.0 against the fixture JSON copied from
the repository branch. Output: 11/11 golden cases, 5,000/5,000 deterministic
generated property states, 9/9 semantic fault models, and 8/8 independent
URL-field omission controls detected. The tool exits non-zero on an
assertion mismatch.

This improves reproducibility of the JavaScript model cross-check, but it is
not a third independent oracle implementation of the Rust module (it is a
separate-language model), does not execute the Leptos candidate and does not
include the DOM-based XSS assertion or responsive CSS browser mutation. Node
script output must not be conflated with the Rust CLI or CI workflows.
