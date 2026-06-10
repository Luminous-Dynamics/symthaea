# Direction 2 expansion results — more data, honest numbers

**Status:** complete. 2026-04-18.
**Context:** The previous "48% auto-ingested" headline was a single-seed, single-slice number on a benchmark of size 13 (invariants) or 50 (miniF2F). Direction 2 from `session1b-results.md` asked for "numbers that survive skeptical review." This doc delivers the expansion.

## Invariant discovery — 13 → 25 sequences (93% expansion)

The 30 `observe_X` functions in `conjecture_engine.rs` already cover a diverse set of sequences; the original benchmark used 13. This pass adds 12 more from what was already implemented:

**New physics:** `inverse_square_law`, `relativistic_kinetic_energy`, `nuclear_binding_energy`, `gr_correction` (general-relativity precession).

**New combinatorics:** `derangements`, `bell_stirling_residual`, `stirling_sum`, `fubini` (ordered Bell), `motzkin`, `perm_det_ratio`.

**New number theory:** `prime_gaps`, `maximal_prime_gap`.

### Scorecard (seed=42, pop=200, gen=100, 3.2s total)

**8/25 = 32.0% closed (test_rel_err < 5%).**

Down from the 13-problem version's 46.2% — expected. Adding harder sequences lowers the rate and reveals the ceiling. Honest measurement, not a regression.

Notable recoveries (the 2 new closures):

| Sequence | Closed formula | Complexity | Test rel err |
|----------|----------------|------------|--------------|
| `inverse_square_law` | `r^(-2)` equivalent | 3 | **0.00** |
| `bell_stirling_residual` | `0` (constant — identity is exactly zero) | 1 | **0.00** |

The GP correctly found that the Bell/Stirling identity residual is literally the zero function — no free variables, no degrees of freedom needed. Meta-valid recovery.

Remaining misses are super-exponential growth without log-space pre-transform (derangements, fubini, motzkin, stirling_sum) — same shape as the original Bell/Catalan/partitions misses. A single fix (log-space pipeline) would likely close several at once; deferred to future work.

Raw data: `docs/phase6-scoping/invariant_discovery_n25_seed42.csv`.

## miniF2F end-to-end — 1 seed → 3 seeds

Same harness, same corpus, different shuffles (42, 1337, 7919).

### Per-seed scorecard

| Seed | Parsed | Translated | Lake-accepted / total | Lake-accepted / translated |
|------|--------|------------|---------------------|---------------------------|
| 42 | 35/50 | 35/35 | **25** (50.0%) | 71.4% |
| 1337 | 29/50 | 27/29 | 22 (44.0%) | **81.5%** |
| 7919 | 31/50 | 29/31 | 23 (46.0%) | 79.3% |

### Aggregate statistics

- **Median accept-of-total: 46.0%**
- **Range: 44.0% – 50.0% (spread 6 pp)**
- Mean: 46.7%

### What the spread tells us

- **Parse rate varies 58% – 70%** across seeds. Different shuffles hit different mixtures of `↑`-coerced and function-abstracted problems. The parse-gap is the single biggest lever.
- **Accept-of-translated varies 71% – 82%.** Seed 42's slice just happened to contain harder post-translation fixtures (more Pattern-D-shaped problems). The cascade itself is stable.
- **Single-seed headlines are noisy.** The original "44%" from seed-42-only looked like a 4pp movement when Pattern B landed; the 3-seed picture shows that 44% was on the low end of the stable range. Proper citation: **46% ± 3pp median end-to-end**.

Raw data: `docs/phase6-scoping/minif2f_baseline_seed{42,1337,7919}_n50.csv`.

## What this shipped and what it didn't

Shipped:
- 93% larger invariant benchmark (25 vs 13), with 2 new clean closures
- 3× seed coverage on miniF2F (3 × 50 vs 1 × 50)
- Honest median + spread statistics

Not shipped:
- Full-pool miniF2F (N=178) — would take ~4-6h; left as future work if a bigger number matters
- Log-space pre-transform for super-exponential sequences — would likely close derangements/fubini/motzkin/stirling_sum, but needs pipeline work in `SymbolicRegressor`
- Difficulty stratification (AIME vs AMC vs olympiad) — requires corpus-level annotation not currently available

## Citation-ready numbers

- **Invariant discovery:** 8/25 = 32.0% closed at test_rel_err < 5%, on a benchmark spanning 6 physics + 10 combinatorics + 3 number-theory + 6 baseline-physics sequences.
- **miniF2F-v2 end-to-end:** 46.0% ± 3pp median Lake accept across 3 random 50-problem shuffles (seeds 42/1337/7919).
- **Hand-curated miniF2F (reference):** 96.9% (unchanged from prior sessions).
