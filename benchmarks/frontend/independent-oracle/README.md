# SFEB-L2-005 independent oracle preflight

This is a dependency-free Node.js reference implementation used to cross-check
the public-pilot dataset and frozen golden cases independently of the Rust
contract checker.

Run from any working directory:

```sh
node benchmarks/frontend/independent-oracle/search-filter-reference.mjs
```

It reads the checked-in fixture files directly, validates record identity and
integer fields, evaluates all golden cases, runs 5,000 deterministic generated
query states, and tests nine seeded semantic fault models. The expected output
is a JSON summary, with any mismatch causing a non-zero exit.

Requires Node.js 20 or newer; no package install is required. This tool is
benchmark/evaluation code only and is not a runtime dependency of the Leptos
candidate or Symthaea.

## Evidence boundary

A pass establishes agreement between this independent JavaScript model and
the checked-in public-pilot oracle fixtures. It does **not** establish that the
Rust evaluator compiles or passes, that the Leptos candidate runs, that the
browser DOM is safe, or that the complete mutation campaign is qualified.
