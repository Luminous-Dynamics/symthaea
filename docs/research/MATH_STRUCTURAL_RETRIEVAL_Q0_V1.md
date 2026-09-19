# MATH-EXP-001B — Canonical Structural Baseline and Frozen Q0 Set

Status: **experimental / non-authoritative**

This tranche closes a key causal confound in MATH-EXP-001:

> If structural HDC beats lexical retrieval, did HDC help, or did *any* structural representation help?

The correct control is therefore not merely random/token retrieval. Structural HDC must also be compared against a conventional non-HDC representation of the same mathematical AST.

## Frozen identities

HDC intervention inherited unchanged from the parent #4087 head:

`symthaea-math-structural-hdc-v1`

Conventional structural baseline:

`canonical-ast-sparse-v1`

Frozen Q0 challenge manifest:

`data/benchmarks/math_structural_q0_v1.json`

The manifest is `MeasurementOnly`. It labels intended structural neighbors; it carries no theorem truth, proof validity, novelty, or formal authority.

## Conventional baseline

`canonical-ast-sparse-v1` produces a deterministic sparse feature vector over `FolFormulaExt` containing:

- AST node-kind counts;
- depth-bucketed node kinds;
- directional role/node features;
- commutative member roles;
- quantifier kind and numeric type;
- bound-variable de-Bruijn-like distance features;
- canonical free-variable co-reference IDs;
- canonical proposition-atom IDs;
- coarse literal classes;
- exponent classes.

Retrieval similarity is ordinary cosine similarity over the sparse vector. There is no HDC operation in this baseline.

The feature grammar deliberately mirrors the *information available* to the HDC intervention while changing the representation and similarity mechanism. This makes the comparison substantially more causal.

## Frozen Q0 cases

The first set contains five synthetic families:

1. additive linear shape under alpha renaming / literal-class changes;
2. implication direction;
3. repeated-variable co-reference;
4. quantifier nesting and numeric type;
5. commutative child order.

Each case has exactly one frozen intended positive and multiple distractors chosen along named structural axes.

This is intentionally easy. Q0 asks only whether the intervention activates in the intended direction. It is not intended to establish practical theorem-search performance.

## Exact HDC reuse

The comparison harness does not copy the HDC encoder into a second implementation. It uses Rust `include!` on the parent tranche's `math_structural_hdc_q0.rs` and exposes only a local wrapper around its private encoder.

That means the comparison is against the exact parent implementation rather than a hand-maintained duplicate that could drift.

## Measurements

For every frozen case the harness records:

```text
HDC positive similarity
HDC best-negative similarity
HDC shuffled-vector similarity
HDC positive margin
HDC top-1 correctness

canonical-AST positive similarity
canonical-AST best-negative similarity
canonical-AST positive margin
canonical-AST top-1 correctness
```

No assertion requires HDC to outperform canonical AST.

That is deliberate.

The Q0 execution gate only requires both representations to demonstrate that their structural intervention is functioning. Whether HDC adds value beyond the conventional baseline is an empirical result to preserve, positive or negative.

## Interpretation law

```text
HDC > lexical, HDC ~= canonical AST
    => mathematical structure helps
    => HDC-specific advantage NOT established

HDC > lexical, HDC > canonical AST
    => evidence consistent with HDC-specific retrieval value
    => still requires held-out replication and equal-budget search tests

canonical AST > HDC
    => preserve the negative HDC result
    => use the conventional representation unless later evidence changes the conclusion
```

Q0 alone cannot support any of those practical-search claims; it only prepares the representations for the later held-out tests.

## Reproduction

From repository root after the parent structural-HDC tranche is present:

```bash
cargo test -p symthaea-core --example math_structural_retrieval_q0
cargo run -p symthaea-core --example math_structural_retrieval_q0
```

The executable prints CSV-like measurements with both frozen representation IDs.

## Required next evidence

After Q0 compiles and executes, the next tranche should add the remaining retrieval controls over the same frozen challenge set:

- lexical/token retrieval;
- random retrieval;
- shuffled HDC vectors;
- permuted challenge associations.

Then move to a fresh held-out Q0 set rather than tuning against these five cases.

Only after structural retrieval itself qualifies should Q1 measure blind solved-problem transfer under identical proof/search budgets.

## Authority boundary

Nothing in this tranche may alter:

- theorem truth;
- evidence state;
- formal authority;
- epistemic confidence;
- proof verification;
- ResultMemory/SearchMemory retention.

These are retrieval measurements only.
