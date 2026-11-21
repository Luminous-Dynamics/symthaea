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

## Scoring

Publish the quality vector and the hard-gate outcomes per task. Report repeated trials, confidence intervals, cost and wall time. A failed required journey, critical accessibility blocker, security violation, missing receipt or omitted task is a failure regardless of visual score.

Never let the tested agent be the only evaluator of its own output. Use deterministic assertions, independent browser observation, blinded human review where visual judgment is required, and hidden holdout tasks.

## Next implementation steps

1. Author original reference briefs and screenshot assets with clear provenance.
2. Add fixture-backed routes and service behavior for the selected tasks.
3. Choose a Rust-first browser adapter after a small CDP/WebDriver proof of concept.
4. Implement the runner and immutable receipt schema.
5. Run the existing `symthaea-ui` as the baseline.
6. Run controlled ablations: base model, retrieval, interface contract, browser observation/repair, and full Symthaea integration.

Until those steps are complete, SFEB remains a research specification, not a measured capability.
