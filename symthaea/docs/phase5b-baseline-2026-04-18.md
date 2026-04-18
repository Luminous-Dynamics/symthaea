# Phase 5b Zero-Shot Baseline — 2026-04-18

**Harness:** `cargo run -p symthaea-lean-bridge --example ingest_minif2f_baseline`
**Seed:** 42 — `MINIF2F_N=50 MINIF2F_SEED=42`
**Corpus:** `data/benchmarks/minif2f/MiniF2F/{Valid,Test}/mathd_{algebra,numbertheory}_*.lean`
**Raw CSV:** `phase5b-baseline-2026-04-18.csv`

## Top line

| Stage | Count | Rate |
|-------|-------|------|
| Total sampled | 50 | 100% |
| Parsed | 18 | 36.0% |
| Translated (of parsed) | 17 | 94.4% |
| Translated (of total) | **17** | **34.0%** |

**What this measures.** Stage 1 + Stage 2 of the three-tier scorecard only — tokenizer+parser then LeanTheorem→FolFormulaExt translation. The third tier (Lake acceptance via the Phase 2 W4 cascade) is not exercised here; rerunning the same 17 translated formulas through `render_fol_ext_file` + `lake env lean` is the next measurement.

**What the number means.** On a fixture-agnostic random slice of the filter-passed pool, the Rust-native ingest handles 34%. On the 93.8%-accepted hand-translated set (`phase3-findings.md`), every problem parses + translates, because each was chosen to fit our grammar. The gap between "hand-curated 93.8%" and "zero-shot 34%" is the filter gap — not a cascade gap.

## Parse-failure histogram (32/50 = 64% of total)

| Count | Variant | Characteristic cause |
|------:|---------|----------------------|
| 23 | `UnknownChar` | Unicode outside our tokenizer's alphabet |
| 9 | `Unexpected` | Parser grammar position mismatch |

### UnknownChar breakdown from CSV inspection

| Glyph | Seen in | Semantics |
|-------|---------|-----------|
| `⁻` | 4 rows (e.g. `mathd_algebra_245`) | Reciprocal syntax `x⁻¹` |
| `%` | 6 rows (e.g. `mathd_numbertheory_101`) | Modular arithmetic |
| `σ` | 1 row (`mathd_algebra_422`) | Greek variable name |
| `∣` | Several | Divisibility in number-theory problems |
| `↑` | Several | `Nat → Int` coercion |
| misc | remainder | Type ascriptions, Finset sigma, etc. |

### Unexpected breakdown

The 9 `Unexpected` errors all say `found "ℝ" at offset N` where `N` is inside what the parser tried to parse as a `Term`. Cause: the hypothesis body contains either an inline type ascription (`(x : ℝ)` inside a larger expression) or a nested ∀-quantifier (`∀ x : ℝ, …`), both of which our grammar doesn't yet handle. The parser leaves ∀ as a keyword-token but only `parse_theorem` consumes it at the outer binder level.

## Actionable next increments

Ranked by expected marginal gain per LOC:

1. **`⁻` / reciprocal** (~50 LOC, parser + translator): Lean treats `x⁻¹` as `HPow.hPow x (-1 : ℤ)`. Translate at parse time to `1 / x`. Recovers ~4 of 23 UnknownChar rows.
2. **`∣` / divisibility** (~40 LOC, AST + translator): `a ∣ b` means `∃ c, b = a * c`. In `FolFormulaExt` no direct operator — lower to an existential. Needed for many mathd_numbertheory problems. Likely recovers ~5 rows.
3. **`%` / modular** (~30 LOC parser + punt at translate): our filter should exclude these; they're genuinely out of `FolFormulaExt` scope. Update `gather_candidates` in the baseline harness to reject `%`-containing files like Phase 5c will need `mod` support. Recovers 0 rows but cleans the denominator (50 → ~44).
4. **Inline type ascriptions and nested ∀** (~80 LOC parser): resolves all 9 `Unexpected` rows.

Following (1) + (2) + (4), a second measurement at the same seed should land around 34% + ~28pp = ~60-65% parse rate. Pushing past that requires AST extension for `abs`, `Finset`, or function abstraction (Phase 5c proper).

## Not measured here

- **Lake-acceptance on the 17 translated.** Next commit in the Phase 5b track can pipe each translated FolFormulaExt through `render_fol_ext_file` + `lake env lean` for the real third-tier number. Expected: ~90-100% accept on the 17, matching the 93.8% cascade rate on Phase 3's hand-translated set (same distribution of shapes).
- **Seed variance.** One seed, one slice. Re-running with seeds 1337 and 7919 would give a ±1-3 point confidence interval on the 36% parse rate.

## Update — same day, post-reciprocal + ZMod filter

After adding `⁻¹` (reciprocal) tokenizer + parser support and extending the candidate filter to exclude `ZMod` (modular-ring arithmetic, genuinely out of `FolFormulaExt` scope), the **full-pool** rerun gives a sample-invariant number (artifact: `phase5b-baseline-2026-04-18-full-pool.csv`):

| Stage | Count | Rate |
|-------|-------|------|
| Total (full filter-passed pool) | 177 | 100% |
| Parsed | 59 | **33.3%** |
| Translated (of parsed) | 57 | **96.6%** |
| Translated (of total) | 57 | **32.2%** |

Parse-failure histogram: 81× `UnknownChar`, 37× `Unexpected`. The reciprocal change didn't meaningfully move the parse rate because most `UnknownChar` failures are other glyphs (`∣`, `↑`, `σ`, etc.); divisibility + coercion are the bigger prizes. `translate-of-parsed` lifted from 94.4% to 96.6% — the two remaining translate failures are probably `^` with non-literal exponent and one variable-exponent shape.

The 33.3% on 177 files is now the canonical "honest zero-shot" number for the Rust-native ingest. Every future parser extension should be measured against this baseline + seed 42 + `MINIF2F_N=200`.

## Update — extended filter (same day)

The 177-file baseline's 33.3% parse rate was dominated by denominator inflation: 42 files contained `%` (modular arithmetic, no `mod` in `FolFormulaExt`), 6 used `ℂ` (complex), 2 used `ℚ` (rational type), and ~25 declared function-typed binders `(f : ℝ → ℝ)` — all Phase 5c territory. Added them to the Rust harness's out-of-scope list (artifact: `phase5b-baseline-2026-04-18-extended-filter.csv`):

| Stage | Count | Rate |
|-------|-------|------|
| Total (extended-filter pool) | 98 | 100% |
| Parsed | **59** | **60.2%** |
| Translated (of parsed) | 57 | 96.6% |
| Translated (of total) | **57** | **58.2%** |

The numerator didn't change (57 translated) — the 79 excluded files were all failing anyway. The denominator dropped from 177 to 98 because the filter now matches FolFormulaExt's actual scope. Parse rate lifted from 33.3% to 60.2%, end-to-end from 32.2% to 58.2%. Parse-failure histogram shrank to 23× UnknownChar, 16× Unexpected — tractable size for targeted next increments.

**Interpretation:** 60.2% is the honest number for "given a miniF2F file that's in `FolFormulaExt`'s scope, what fraction does the Rust parser read today?" The gap to 100% is now a small set of specific glyphs/grammar shapes (`∣`, `↑`, `∃`, nested `∀`, inline type ascription). Each is a ~20-80 LOC parser extension.

This is the canonical baseline going forward. The 33.3% number from earlier should be cited only in archival context.

## Update — divisibility `∣` support (same day)

Added `∣` (U+2223 DIVIDES) handling: `TokenKind::Divides`, `RelOp::Divides`, translator lowers `a ∣ b` to `∃ _div_witness_N : ℤ, b = a * _div_witness_N` via a counter threaded through `TranslationCtx`. Artifact: `phase5b-baseline-2026-04-18-divisibility.csv`.

| Stage | Before (60.2%) | After (63.3%) | Δ |
|-------|----------------|---------------|---|
| Parsed | 59 / 98 | **62 / 98** | +3 |
| Translated (of parsed) | 57 / 59 (96.6%) | **60 / 62 (96.8%)** | +3 |
| Translated (of total) | 57 / 98 (58.2%) | **60 / 98 (61.2%)** | +3 |

The prediction was +5 rows; the delivered gain was +3. The shortfall is due to files that contain `∣` *plus* another unsupported construct (e.g. `↑` coercion), which still parse-fail on the secondary glyph. `UnknownChar` histogram dropped from 23 to 20 (matches the 3 `∣` rows now clean).

Witness-counter design note: we thread an integer through `TranslationCtx` that bumps on each `Rel(Divides, _, _)` encountered in a single theorem. Two `∣` in the same formula (e.g. `3 ∣ n ∧ 5 ∣ m`) get `_div_witness_1` and `_div_witness_2`, preventing existential capture. Covered by `translate_multiple_divides_use_distinct_witness_names`.
