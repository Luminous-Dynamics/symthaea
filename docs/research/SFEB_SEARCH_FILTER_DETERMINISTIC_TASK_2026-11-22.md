# SFEB-L2-005: deterministic search/filter task slice

**Date:** 2026-11-22  
**Status:** reference fixture authored; Leptos candidate source added; Rust/WASM and full browser qualification pending

The task defines trimmed ASCII case-insensitive search over name/SKU/description, conjunctive category and stock filters, inclusive integer-cent price boundaries, deterministic ID tie-breaking, filter-sort-pagination semantics and complete query-state URL restoration. Its fixture contains 13 synthetic assets, eleven golden cases, ten mutation hypotheses, five positive controls and five metamorphic relations. One description contains an inert HTML-injection canary for a future browser text-node assertion.

A Leptos 0.8.20 CSR candidate is present under `benchmarks/frontend/solutions/sfeb-l2-search-filter-reference/`. It uses the same pure Rust evaluator/query codec as the standalone contract checker. It is source-authored but has not been compiled or browser-run.

An independent JavaScript evaluator matched all 11 golden cases and passed 5,000 deterministic generated query states. Nine of nine simulated semantic faults were detected. This is a model/oracle sensitivity preflight, not a compiled-candidate mutation campaign. The canary has only a source-level escaping audit.

A local Chromium/Playwright preflight of the static reference fixture exposed a 390px input-width overflow. The input sizing rule `.field input,.field select{width:100%;min-width:0}` removes document-level horizontal overflow across 320/390/768/1280/1440 widths. Keyboard skip-link focus and reduced-motion media emulation were also checked. These checks concern the static reference fixture, not the Leptos candidate.

Rust compilation/tests/format/Clippy, WASM build, candidate-browser tests, complete keyboard/accessibility-tree checks, enforced egress policy and independent browser oracle replay remain outstanding. The task is not yet runnable.
