# SFEB-L2-005: deterministic search/filter task slice

**Date:** 2026-11-22  
**Status:** original reference fixture and golden contract authored; Rust execution and browser qualification pending

The task fixes search over name/SKU/description, category and availability conjunction, inclusive integer-cent bounds, deterministic ID tie-breaking for all sorts, filter-then-sort-then-paginate semantics, and a complete state-URL round trip.

The public-pilot fixture contains 13 synthetic asset records, ten golden positive/negative cases, ten mutation hypotheses, five positive controls and five metamorphic relations. The data has no credentials or network dependencies.

The standalone checker now has a `--search-filter-model` path backed by a pure Rust module. Unit tests cover the ten golden cases, case-fold invariance, filter-subset monotonicity, stable ties, page concatenation, URL round-trip and reversed-range rejection.

This reference model is not a Leptos implementation and does not establish browser behavior. Keyboard access, actual candidate semantics, accessibility-tree output, responsive layout, network policy and visual quality remain browser qualification gates. The mutation campaign is specified but not executed; no mutation kill rate or oracle qualification is claimed.
