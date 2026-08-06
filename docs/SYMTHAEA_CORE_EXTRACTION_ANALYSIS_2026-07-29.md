# `symthaea-core` extraction: measured dependency analysis

> ## ⚠ CORRECTION (same day, before any work was based on this)
>
> **Finding 5 of the original version of this document was wrong, and it was the headline
> recommendation.** It claimed 240 of 350 files in `hdc/` contain a cross-module `use super::*`
> glob, that the seam's footprint was therefore unknowable, and that de-globbing must be step 1.
>
> All 240 of those globs are `use super::*` **inside `#[cfg(test)] mod tests`** — the idiomatic
> Rust pattern where a test module imports its own parent file. `super` there is the file's own
> module, not `hdc/mod.rs`. **Top-level cross-module globs in `hdc/`: zero.**
>
> The error came from a grep (`^\s*use (crate|super)::`) whose leading `\s*` matched indented
> test-module imports, which I then read as top-level ones without checking the context.
>
> Corrected findings are in **Finding 5 (revised)** and **Finding 7** below. The net effect is
> that the extraction case is *substantially stronger* than the original document said: there is
> no glob blocker, the ~11K-line figure is accurate rather than a lower bound, and the seam turns
> out to be fully self-contained.

**Status: analysis only. No extraction performed, none authorized.**
Produced 2026-07-29 so that the first seam is *chosen from measured dependency edges* rather
than from a conceptual taxonomy. Everything below is a measurement with the command that
produced it; the interpretation is separated from the data.

## Why this document exists rather than a proposed crate split

An earlier external review recommended splitting `symthaea-core` into six crates
(`hdc-primitives`, `hdc-runtime`, `consciousness-metrics`, `physics-research`,
`symbolic-discovery`, plus a facade). That is a plausible taxonomy, but a taxonomy is a
hypothesis about what is shared. This measures what is *actually* shared, which turns out to
be both narrower and sharper than the taxonomy predicts.

## Method and its limits

For all 86 workspace packages depending on `symthaea-core` (via `cargo metadata`), every
`use symthaea_core::…` clause and inline `symthaea_core::…` path in `src/`, `examples/`,
`tests/` and `benches/` was parsed, with nested brace groups expanded to full paths.

**Limits, stated up front:**
- Counts are *crates referencing an item*, not call-site frequency. A crate touching
  `ContinuousHV` once counts the same as one using it everywhere.
- Re-exports are followed only as written. A dependent importing something `symthaea-core`
  re-exports from elsewhere is attributed to `symthaea-core`.
- `use symthaea_core as X` aliasing would be missed. **Zero** such aliases exist (checked).
- Feature-gated code is counted whether or not the feature is enabled.

## Finding 1 — the consumed surface is tiny and extremely top-heavy

Only **13 distinct first-level items** of a ~371K-line crate are referenced by any dependent:

| item | crates | share of 86 |
|---|---:|---:|
| `hdc` | 78 | 90% |
| `genesis` | 30 | 34% |
| `embodiment` | 22 | 25% |
| `temporal` | 7 | 8% |
| `math` | 5 | 5% |
| `physics` | 4 | 4% |
| `consciousness_metrics` | 3 | 3% |
| `phi_engine` | 3 | 3% |
| `observation` | 2 | 2% |
| `observability`, `core`, `synthesis_trait`, `ConsciousnessContext` | 1 each | 1% |

## Finding 2 — within `hdc/`, nine items carry almost all the weight

181 distinct `hdc::` sub-items are referenced somewhere, but the head is sharp:

| item | crates |
|---|---:|
| `ContinuousHV` | 47 |
| `HDC_DIMENSION` | 34 |
| `unified_hv` | 26 |
| `UnifiedConfig` | 21 |
| `HdcLtcUnifiedNetwork` | 20 |
| `UnifiedNetworkConfig` | 20 |
| `hdc_ltc_unified` | 19 |
| `binary_hv` | 15 |
| `BinaryHV` | 9 |
| *(tail: `logic_engine` 6, then ≤4 each)* | |

## Finding 3 — the candidate seam is ~11K lines serving 90% of edges

The files defining that head:

| file | lines |
|---|---:|
| `binary_hv.rs` | 3,135 |
| `hdc_ltc_unified.rs` | 2,847 |
| `simd_continuous.rs` | 1,533 |
| `unified_hv.rs` | 1,508 |
| `simd_ops.rs` | 1,422 |
| `simd_detect.rs` | 350 |
| **total** | **~10,795** |

Against **295,048** lines in `hdc/` and ~371K in the crate. If the seam is genuinely
separable, ~3.7% of `hdc/` serves 90% of the dependency edges — roughly a **27× reduction** in
what a typical dependent must compile and trust.

The original document flagged this as a lower bound pending a glob audit. That caveat is
**withdrawn** — see Finding 5 (revised) and Finding 7. The figure is real.

## Finding 4 — migration cost is concentrated in one crate

How much of `symthaea-core` each dependent needs:

| needs at most | crates | share |
|---|---:|---:|
| `hdc` only | 35 | 40% |
| `hdc` + `genesis` | 50 | 58% |
| `hdc` + `genesis` + `embodiment` | 70 | **81%** |
| more than those three | 16 | 19% |

Of the 16 outliers, 14 need only one or two additional modules — mostly `temporal`, `math`,
`physics`, `phi_engine`, `consciousness_metrics`. The genuine outlier is the main `symthaea`
crate itself, which needs nine (`consciousness_metrics`, `core`, `math`, `observability`,
`observation`, `phi_engine`, `physics`, `synthesis_trait`, `temporal`).

This matters for sequencing: a seam plus `genesis` and `embodiment` would let **70 of 86**
dependents drop the monolith entirely, while `symthaea` keeps a full-fat dependency.

## Finding 5 (revised) — there is no glob blocker

**Top-level cross-module `use super::*` in `hdc/`: zero**, across all 350 files.

```
grep -rlc "use super::\*"  crates/core/symthaea-core/src/hdc/*.rs | wc -l   # 240
grep -rl  "^use super::\*" crates/core/symthaea-core/src/hdc/*.rs | wc -l   # 0
```

Every one of the 240 is inside a `#[cfg(test)] mod tests` block. That construct imports the
file's *own* module and creates no dependency on `hdc/mod.rs`. The original Finding 5 conflated
the two and recommended a de-globbing pass that is not needed.

Because there are no globs, each seam file's real dependencies **can** be read directly — which
makes Finding 7 possible.

## Finding 6 — incidental: `HDC_DIMENSION` is defined twice

`hdc/mod.rs:58` and `hdc/unified_hv.rs:73` each define `pub const HDC_DIMENSION: usize = 16_384`.
`mod.rs` re-exports only `{ContinuousHV, HV}` from `unified_hv`, so both are reachable at
different paths. **Values agree today, so this is not a live bug** — but 34 crates reference
`HDC_DIMENSION` and which one they get depends on the path they wrote. A future edit to one
would silently diverge. Cheap to unify; not done here (out of scope for an analysis).

## Finding 7 — the seam is self-contained but for a single external type

With no globs in the way, the six seam files' real (non-test) dependencies can be read exactly.
Both `use` clauses and fully-qualified `crate::…` paths were counted — the latter matters, since
`hdc_ltc_unified.rs` reaches `GenesisSeed` by fully-qualified path with no `use` at all, which an
import-only scan misses.

Internal to the seam:

| file | depends on |
|---|---|
| `unified_hv.rs` | `simd_continuous::{bundle_simd, similarity_simd, norm_simd, bind_simd}` |
| `hdc_ltc_unified.rs` | `unified_hv::{ContinuousHV, HDC_DIMENSION}`, `simd_detect::{has_avx, has_fma}` |
| `simd_continuous.rs` | `simd_detect::{has_avx, has_avx2, has_fma, has_neon, has_sse41}` |
| `simd_ops.rs` | `simd_detect::…`, `binary_hv::BinaryHV` |
| `binary_hv.rs` | *(nothing)* |
| `simd_detect.rs` | *(nothing)* |

**External to the seam: `crate::genesis::GenesisSeed` and nothing else** — used once in
`unified_hv.rs` and twice in `hdc_ltc_unified.rs`.

`genesis.rs` is **247 lines** and its own only internal dependency is `crate::hdc::unified_hv`,
which is inside the seam. So including it closes the graph completely:

> **seam (6 files, 10,795 lines) + `genesis.rs` (247) = 11,042 lines, fully self-contained**,
> with zero references to anything else in `symthaea-core`.

The `genesis` ↔ `unified_hv` mutual reference is a module cycle *within* one crate, which Rust
permits — it is not an obstacle.

This is a materially better result than the original document reported. `genesis` is also
independently the second most-used module (30 crates), so including it is not a compromise: it
serves the **50 of 86 dependents (58%) that need only `hdc` + `genesis`**, who could then depend
on the extracted crate *alone*.

## Finding 8 — this extraction has already been attempted twice, and both attempts are orphaned

**Discovered before creating any crate, by checking whether the target already existed.** It did.

Commit `9baa58af21`, *"feat(standalone): extract symthaea-hdc-crypto + symthaea-hdc-ltc as
independent crates"*, performed essentially the extraction this document recommends. The outcome:

| extracted crate | lines | dependents | original still in `symthaea-core`? |
|---|---:|---:|---|
| `symthaea-hdc-ltc` | 2,660 | **1** | yes — `hdc_ltc_unified.rs`, 103,710 bytes, changed as recently as 2026-07-18 |
| `symthaea-hdc-crypto` | 1,877 | **0** | yes — `hdc_crypto.rs`, 37,086 bytes |

`symthaea-hdc-ltc` is technically clean: standalone, depends only on `serde` and `rand`, carries
its own `ContinuousHV` and `HDC_DIMENSION = 16_384`, and states the same O(1) closed-form
temporal-evolution thesis as the in-core version. The extraction *as an engineering act*
succeeded.

**What did not happen was the migration.** The in-core original was left in place, 86 crates
went on using it, it kept being developed, and the extracted crate acquired one consumer
(`symthaea-probe-stream`) and then stopped. `symthaea-hdc-crypto` acquired none.

The two have since diverged into different APIs — of 58 public functions in-core and 45 in the
extracted crate, only **16 names are shared**. The extracted crate is therefore not a drop-in
replacement; adopting it now would be a porting job, not a re-pointing.

### Why this matters more than the size numbers

**Net effect of the previous attempt was more duplication, not less.** The workspace gained a
second implementation of its most-depended-on concept while keeping the first.

And the mechanism that allowed it is the one this document recommended in step 3: *"re-export
from `symthaea-core` as a facade so no dependent changes on day one."* Zero day-one change means
zero pressure to migrate, and the original remains the path of least resistance — permanently,
because it is also the one that keeps receiving fixes.

### Revised recommendation

**Do not perform a third extraction yet.** The binding constraint is not identifying a seam
(this document did that, and the trial crate compiled it). It is that this workspace has a
0-for-2 record on the part that actually removes duplication.

Before any new crate is created, the following should be settled:

1. **Decide the fate of the two existing orphans.** Either adopt `symthaea-hdc-ltc` as the
   migration target and delete the in-core implementation, or archive it and record why. Leaving
   both is the status quo that produced this finding.
2. **Make migration the deliverable, not crate creation.** A seam that 50 of 86 dependents have
   *actually moved to* is worth more than a technically perfect crate with one consumer.
3. **Drop the day-one facade, or time-box it.** If the facade stays indefinitely, nothing forces
   migration and the outcome above repeats. A facade with a deletion date, or migrating the 35
   `hdc`-only dependents in the same change, both create the pressure the previous attempt lacked.

Findings 1–7 remain valid and useful — the seam is real, measured, and now compiles. This finding
changes *when and how* to act on them, not whether they are true.

## What this implies (interpretation, not measurement)

The measured seam is narrower than the proposed taxonomy: `ContinuousHV`, `BinaryHV`,
`HDC_DIMENSION` and the unified HDC-LTC network types, with their SIMD backing — roughly the
review's `hdc-primitives` and `hdc-runtime` merged, since `simd_*` exists only to serve them
and no dependent references SIMD independently in meaningful numbers.

The other four proposed crates are **not supported by dependency evidence**:
`consciousness_metrics` (3 crates), `physics` (4), `math` (5) and `phi_engine` (3) are used by
so few dependents that extracting them changes almost nothing about what the other 80 compile.
They may still be worth separating for review-boundary reasons — but that is a different
argument from the one this data makes, and should be made explicitly rather than inherited.

## Recommended sequence, if extraction is authorized

**Superseded in part by Finding 8** — settle the two existing orphaned extractions first. The
original step 1 (de-glob) is **withdrawn**; there is nothing to de-glob.

1. **Unify the duplicate `HDC_DIMENSION`** (Finding 6). Cheap, independently correct, and
   removes an ambiguity that would otherwise be duplicated into the new crate.
2. **Create the crate from the seven files** (six seam + `genesis.rs`), which Finding 7 shows is
   a closed set. No other `symthaea-core` module needs to move.
3. **Re-export from `symthaea-core` as a facade** so no dependent changes on day one and the
   extraction is provably behaviour-preserving.
4. **Migrate the 35 `hdc`-only dependents first**, then the 15 more that need only
   `hdc` + `genesis` — 50 of 86 can drop the monolith with no other work.
5. `embodiment` (22 crates) is the natural second seam; it was not analysed here.

The remaining verification this document does NOT provide: whether the seven files compile as a
standalone crate. That is a single mechanical experiment and it is the honest next gate — every
number above is static analysis, and static analysis of a 371K-line crate is exactly the kind of
claim this project has learned to distrust until something executes.

## Reproducing

```bash
cargo metadata --no-deps --format-version 1 > meta.json   # 86 dependents
grep -rlc "use super::\*" crates/core/symthaea-core/src/hdc/*.rs | wc -l   # 240
find crates/core/symthaea-core/src/hdc -name '*.rs' | xargs cat | wc -l    # 295,048
```

The import-surface scan is a throwaway script, not checked in; the counts above are what it
produced and the method is described in full under *Method and its limits* so it can be
rebuilt and disagreed with.
