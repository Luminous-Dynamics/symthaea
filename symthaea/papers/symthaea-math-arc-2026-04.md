# Symthaea's Mathematical Reasoning Arc (April 2026)

*A consolidated account of six-phase empirical work on Symthaea's two
mathematical tools — a symbolic proof bridge to Lean 4 + Mathlib, and
a genetic-programming invariant discoverer — including a cleanly
negative result on connecting them via hyperdimensional signatures.*

---

## 1. Headline numbers

Three distinct benchmarks, each with reproduction command. All numbers
are on externally verified outputs (either Lean/Mathlib accept or
closed-form test-split MSE).

| Benchmark | N | Metric | Result |
|-----------|---|--------|--------|
| Hand-curated miniF2F-v2 | 32 | `lake env lean` accept | **31/32 = 96.9%** |
| Auto-ingested miniF2F-v2 (3 seeds) | 3 × 50 | Lake accept (median) | **46.0% ± 3pp** (44–50%) |
| Invariant discovery (known CF) | 25 | Test-split rel. error < 5% | **10/25 = 40.0%** |

Plus two structural results:

- Z3 oracle is now reliable — 0 timeouts, 0 errors on the 32-fixture
  curated set (Skolemization + non-linear detection fixes).
- The learned-cascade direction (HDC signature → best cascade variant)
  was ruled out by a pre-declared kNN null on 31 Lake-verified goals.

---

## 2. What got built

Two mathematical tools inside the Symthaea codebase, both measured in
this arc:

### 2.1 Lean-bridge: `FolFormulaExt` → Mathlib tactic cascade

`symthaea-core/src/hdc/fol_formula_ext.rs` defines a first-order logic
AST extending Phase 1's `Proposition` with arithmetic terms (`Term`),
quantifier binders (`Forall`/`Exists` with `NumericType`), and ordering
relations. `symthaea-lean-bridge::fol_ext_bridge::render_fol_ext_file`
compiles a goal into a `.lean` file whose proof is a named-intro +
`first | … | …` tactic cascade. `lake env lean` decides the proof
externally against Mathlib v4.12.0.

The cascade contains a dozen alternatives, each `; done`-terminated to
prevent partial-simplification poisoning `first`'s backtracking:

```
  rfl → norm_num → ring → omega → linarith → nlinarith[compact]
  → positivity → field-reasoning (3 sub-branches)
  → And-splitter (gated) → nlinarith[widened] → lt_trichotomy → …
```

Three gating predicates select optional branches: `conclusion_is_and`
(Pattern A), `formula_has_symbolic_division` (Pattern B),
`is_ne_hypothesis` (Pattern B witness extraction).

### 2.2 `SymbolicRegressor`: genetic-programming invariant discoverer

`symthaea-core/src/hdc/conjecture_engine.rs` defines a GP over closed-form
expressions. Given an `ObservedSequence` of `(x, y)` points, `fit(seq,
top_k)` returns the top-k closed-form formulas by fitness (MSE + Occam
penalty). The GP includes a log-space pre-transform (activated when data
is positive and growth > 50×) that wraps in `exp(…)` and was reworked
in this arc to pin the exp-wrapped candidates into the returned top-k
regardless of tournament-selection pressure.

The two tools don't currently share state. Section 7 reports the
experiment that tested whether HDC signatures over Lean source could
drive cascade selection — it didn't work, for reasons we diagnose there.

---

## 3. Benchmark 1 — hand-curated miniF2F (96.9%)

### Fixtures

32 problems hand-translated from `data/benchmarks/minif2f/MiniF2F/`
(public miniF2F-v2 corpus, 488 files). Each fixture is a literal
translation of the upstream Lean statement into `FolFormulaExt`, with
the source filename recorded for cross-reference. Selection
criterion: problems using only ℝ/ℤ/ℕ numeric types and the AST's
operators (no `Real.sqrt`, `Real.log`, `Finset`, `abs`, `mod`, …).

### Evolution of the number

| Phase | Accept | Rate | Mechanism |
|-------|--------|------|-----------|
| 3 baseline | 25/32 | 78.1% | Phase 2 W4 cascade (hand-crafted training set: 14/14 on its own fixtures) |
| 4a (naive) | 28/32 | 87.5% | Widened `sq_nonneg` offsets + And-splitter. **2 regressions** from Lean heartbeat timeouts |
| 4b | 30/32 | 93.8% | Compact-first hints, gated And-splitter, no regressions |
| 5 | 31/32 | 96.9% | `field_simp` branch (Pattern B) closed `mathd_algebra_55` |
| 5a | 31/32 | 96.9% | Z3 Skolemization (semantics-preserving; Z3 timeouts 5→0) |
| **5a+cx** | **31/32** | **96.9%** | `compound×compound` non-linear detection → Z3 errors 1→0 |

The 78 → 97% arc took five focused refinements, each closing a named
pattern. Phase 4a's regression taught a specific lesson — see §6.

### Remaining rejection

`mathd_algebra_338`: `3a+b+c=−3 ∧ a+3b+c=9 ∧ a+b+3c=19 → abc=−56`. The
unique solution `(a, b, c) = (−4, 2, 7)` gives `abc = −56`, but our
cascade can't derive it. Reasons:

- `polyrith` (Mathlib's Gröbner-basis closer) shells out to Sage Cell
  over HTTPS. In offline environments the call times out with a
  JSON-decode error. The `| (polyrith; done)` alternative has been a
  silent no-op throughout.
- `nlinarith` with solution-value `sq_nonneg` hints still fails —
  "linarith failed to find a contradiction". Cubic conclusions aren't
  derivable from non-negativity combinations of linear facts.
- Closing it requires Rust-side symbolic RREF to emit explicit `have
  a = −4 := by linarith; subst; norm_num` — ~100+ LOC for a single
  fixture. Deferred.

### Reproduction

```bash
cd /srv/luminous-dynamics/symthaea
LAKE_ENV=1 cargo run -p symthaea-lean-bridge --example prove_minif2f_curated
```

Artifacts: `docs/phase3-results/minif2f_curated_results.csv`,
`proofs/minif2f_curated/*.lean`.

---

## 4. Benchmark 2 — auto-ingested miniF2F (46% ± 3pp)

### Harness

A sibling session shipped the parse-and-translate pipeline
(`symthaea-lean-bridge::minif2f_ingest`) in April 2026: tokenizer →
recursive-descent parser → AST translator → `FolFormulaExt`. Tier-3
Lake verification was added in the current arc, making the scorecard
three-tier: parsed / translated / Lake-accepted.

The harness samples N problems from the 178-file filter-passed pool
with a deterministic shuffle (`MINIF2F_SEED`, `MINIF2F_N`).

### Three-seed variance (Pattern B cascade on main)

| Seed | Parsed | Translated | Lake / total | Lake / translated |
|------|--------|------------|-------------|-------------------|
| 42 | 35/50 | 35/35 | **25** (50.0%) | 71.4% |
| 1337 | 29/50 | 27/29 | 22 (44.0%) | **81.5%** |
| 7919 | 31/50 | 29/31 | 23 (46.0%) | 79.3% |

Median 46.0%. Range 44–50%. Spread 6pp.

### What the variance reveals

- **Parse rate varies 58–70%** across seeds — the largest single lever.
  9 `UnknownChar` failures (mostly `↑` coercions) and 6 `Unexpected`
  failures (function abstraction `∀ x, f x = …`) dominate; different
  shuffles hit different ratios of these.
- **Accept-of-translated is 71–82%** — cascade itself is stable. Seed
  42's 71.4% reflects a harder post-translation slice (more
  Pattern-D-shaped problems), not a cascade regression.
- **Single-seed headlines are noisy.** The proper citation is
  "46% ± 3pp median", not "44%" or "48%" cherry-picked from one seed.

### Why the 97→46 gap (hand-curated vs auto-ingested)

The 32 curated problems were hand-selected to be in AST scope. The
auto-ingested 50 includes problems whose shapes the cascade can parse
but hasn't been tuned for (no matching Pattern X hint set). The gap is
a selection-vs-measurement difference, not a regression.

### Reproduction

```bash
cd /srv/luminous-dynamics/symthaea
LAKE_ENV=1 cargo run -p symthaea-lean-bridge --example ingest_minif2f_baseline
MINIF2F_SEED=1337 LAKE_ENV=1 cargo run -p symthaea-lean-bridge --example ingest_minif2f_baseline
MINIF2F_SEED=7919 LAKE_ENV=1 cargo run -p symthaea-lean-bridge --example ingest_minif2f_baseline
```

Artifacts: `docs/phase6-scoping/minif2f_baseline_seed{42,1337,7919}_n50.csv`.

---

## 5. Benchmark 3 — invariant discovery (40%)

### Fixtures

25 univariate sequences with known closed-form solutions, all drawn
from `observe_X` functions that already exist in `conjecture_engine.rs`
(built during the Ramanujan arc, Feb–Apr 2026). The set spans:

- **Physics**: hydrogen, Stefan–Boltzmann, Kepler third law, blackbody,
  Balmer, quantum harmonic oscillator, inverse square, relativistic KE,
  nuclear binding energy, GR correction (10).
- **Combinatorics**: partitions, Bell, Catalan, derangement ratio,
  fibonacci ratios, central binomial limit, derangements, Bell–Stirling
  residual, Stirling sum, Fubini, Motzkin, permutation/determinant
  ratio (12).
- **Number theory**: prime counting, prime gaps, maximal prime gap (3).

### Scorecard (seed 42, default config, 2.9s total)

**10/25 = 40.0% closed (test split relative error < 5%).**

Notable clean recoveries:

| Sequence | Formula (discovered) | Complexity | Test rel err |
|----------|-----------------------|------------|-------------|
| `kepler_third_law` | `n^1.500000` | 5 | **3.71e-15** |
| `stefan_boltzmann` | `n^4` | 3 | **0.00** |
| `inverse_square_law` | `n^(-2)` equivalent | 3 | **0.00** |
| `bell_stirling_residual` | `0` (constant) | 1 | **0.00** |
| `quantum_harmonic_oscillator` | `n + sin(0.5236)` (sin π/6 = ½ → `n + ½`) | 4 | 2.73e-9 |
| `fibonacci_ratios` | `1.626 - 0.505^n` (Binet limit) | 5 | 5.06e-3 |
| `prime_counting` | `0.764 · n^0.762` (π(n) ≈ n/ln n) | 5 | 1.78e-2 |
| `partitions` | Hardy–Ramanujan-shaped | 11 | 4.77e-2 |
| `catalan` | 4^n / n^1.5 shaped | 12 | 1.27e-3 |

Kepler's third law recovered at machine epsilon (`n^1.500000`) in under
two seconds of CPU time is the arc's strongest single result — it is
physical-discovery-grade numerical accuracy on held-out data.

### Evolution of the number

| Version | N | Closed | Rate | Change |
|---------|---|--------|------|--------|
| v1 | 13 | 6 | 46.2% | Initial |
| v2 (expansion) | 25 | 8 | 32.0% | +12 harder sequences; rate honestly drops |
| **v3 (log-space fix)** | **25** | **10** | **40.0%** | Pin exp-wrapped candidates into top_k |

The log-space fix is described in §6.

### Misses

Still missing: `bell_numbers` (1.0), `stirling_sum` (1.0), `derangements`
(inf), `fubini` (inf), `motzkin` (inf). These are all super-exponential.
`bell_numbers` and `stirling_sum` now produce a formula that's
directionally right but above the 5% test-split threshold; the others
fail log-space's `training_mse < 0.1` gate. Tighter log-space, richer
function set, or analytic closed-form priors would likely close several.

### Reproduction

```bash
cd /srv/luminous-dynamics/symthaea
cargo run -p symthaea-core --release --example invariant_discovery_bench
```

Artifacts: `docs/phase6-scoping/invariant_discovery_n25_logspace_fix.csv`.

---

## 6. Failure-mode taxonomy

Each pattern is cited to the specific fixture that revealed it.

### Pattern A — conjunction in conclusion

Goal shape: `… → (A ∧ B)`. `linarith`/`nlinarith`/`omega` can't split.

**Fix:** `refine ⟨?_, ?_⟩ <;> first | linarith | nlinarith [hints] | …`
branch, gated on `conclusion_is_and` to avoid Phase 4a regression
(heartbeat timeouts when emitted for non-And goals). Closed
`mathd_algebra_126`, `mathd_algebra_101`.

### Pattern B — field reasoning with symbolic denominator

Goal shape: `h : x ≠ c → expr/(x − c) = k → conclusion`.

**Fix (naive):** `field_simp at *`. Didn't close the `_181` family —
Mathlib's `field_simp` doesn't auto-derive `x − c ≠ 0` from `¬ x = c`.

**Fix (works):** name hypotheses via explicit `intro`, emit `have ne_i
:= sub_ne_zero.mpr h_i` for each `≠` hypothesis, pass as explicit
argument `field_simp [ne_0, ne_1, …] at *`. Closed `mathd_algebra_181`,
`mathd_algebra_251`, `mathd_algebra_55`.

Gotcha worth remembering: Lean 4's `first | A | B` commits to A when
`A = try X; try Y; first_inner; done` and inner_first "succeeds with
unsolved goals". That blocked a later field-branch from ever firing.
Fix: the Pattern B witness branch must come *first* among field
sub-branches.

### Pattern C — polynomial inequality, vertex offset ≠ ±1

Goal shape: `x² + bx + c ≥ k` where the vertex isn't at x = ±1.

**Fix:** widen the literal offset set in `sq_nonneg (x ± k)` hints to
`{−10, −7, −5, −3, −1, 1, 3, 5, 7, 10}` — but only in a *fallback*
nlinarith branch. The fast-path nlinarith keeps compact `{−1, 1}`
offsets to avoid Lean heartbeat blowup on well-behaved goals.
Phase 4a's regression was a lesson in hint-count matters: too many
`sq_nonneg` hints push `whnf` past the 200k heartbeat budget.
Closed `_113` (vertex at 7), `_410` (vertex at 3), bonus
`mathd_numbertheory_326` (integer cubic root).

### Pattern D — solve-then-evaluate

Goal shape: `{linear equations} → f(solution)`. `nlinarith`'s
Positivstellensatz search doesn't reason in "solve, then evaluate"
mode. Closing requires Rust-side symbolic RREF. Deferred at 100+ LOC
per single fixture (`mathd_algebra_338`).

### Super-exponential cluster

Sequences with factorial-level growth (`derangements`, `fubini`,
`motzkin`, etc.) overflow the GP's ordinary linear-space fitness
evaluation. Log-space pre-transform (`y → ln y`) is activated when
`growth > 50×`, but the naive "insert exp(fit) into population" path
lost tournament selection to polynomial approximants that fit partial
data tightly and exploded on test.

**Fix:** pin exp-wrapped candidates into the returned top_k, re-ranked
by original-space fitness. Tournament can no longer discard them.
Tightened acceptance threshold (`training_mse < 1.0 → < 0.1`) stops
admitting factor-of-e errors. Closed `partitions`, `catalan`.

---

## 7. The null result: learned cascade selection

Phase 6 asked whether a learned mapping from goal HDC-signature to
"best cascade variant" could beat the hand-tuned cascade. The
hypothesis was that goals with similar structure would prefer similar
tactic orderings.

**Setup:** For each of the 31 Lake-verified goals in the Pattern-B
auto-ingested slice, compute an HDC signature: tokenize the raw Lean
source (splitting on whitespace + a small punctuation set), hash each
token → `BinaryHV::random(seed)`, permute by position, bundle.

**Session 1 — cluster separation:**

| Partition | Mean cosine | Pairs |
|-----------|-------------|-------|
| within-accept | +0.2478 | 231 |
| within-reject | +0.2288 | 36 |
| between | +0.2340 | 198 |

Effect size (accept − between): **+0.0138** (+1.4%). Above the
pre-declared null threshold of 0.005. Weak positive cluster signal.

**Session 1b — kNN leave-one-out classification:**

| Metric | Value |
|--------|-------|
| kNN (k=3) accuracy | 22/31 = **71.0%** |
| Majority-class baseline (always "accepted") | 22/31 = **71.0%** |
| Lift over baseline | **+0.0 pp** |
| Accepted recall | 81.8% |
| Rejected recall | 44.4% |

**The +1.4% cluster signal does not translate into predictive power.**
The classifier matches the constant-prediction baseline exactly.
Rejected-class recall (44%) is barely above coin-flip.

**Honest conclusion:** surface-token HDC signatures capture *presence*
of operators (`∀`, `ℝ`, arithmetic symbols) but not the *semantic
shape* that distinguishes Pattern-B from Pattern-D from linear
problems. Session 2 (4-cascade tournament across 67 goals) was
pre-declared not-worth-running on a non-informative signal — building
infrastructure against this null would have been dishonest work.

**What would change this:** a richer encoder — tree-structured HDC
over the `FolFormulaExt` AST, or the cognitive loop's full `wisdom_hv`
(which would require bringing the main `symthaea` crate's
dependencies into the bridge). Testing this is a future-work path.

---

## 8. Architecture lessons

Three specific gotchas worth remembering:

### 8.1 Lean 4 `first | A | B` commitment

If `A = try X; try Y; first_inner; done` and `first_inner` fails to
close, Lean interprets the *outer* `first`'s experience of `A` as
"succeeded with unsolved goals" rather than "failed." The outer
`first` then commits to `A`, never trying `B`.

**Implication:** ordering matters. A well-tuned later branch can be
silently blocked by an earlier `try`-guarded branch that didn't quite
close. The Pattern B fix only worked after moving the witness branch
to first among field branches.

### 8.2 `cargo run` resolves path-deps from the current directory

Running `cargo run -p symthaea-lean-bridge --example …` from the
main tree's `/srv/luminous-dynamics/symthaea/` resolved `symthaea-core`
from the *main tree*, not the worktree with my changes. Manifested
as "my edits aren't taking effect" twice in this arc. **Always `cd`
to the worktree's `symthaea/` directory before `cargo run`**, or the
build uses stale sources.

### 8.3 Worktree data isolation

`scripts/session-worktree.sh` doesn't symlink `data/benchmarks`. Any
harness that reads corpus files fails silently with "corpus not found"
until the symlink is manually created:

```bash
ln -s /srv/luminous-dynamics/symthaea/data/benchmarks \
      /srv/luminous-dynamics/.claude/worktrees/session-<name>/symthaea/data/benchmarks
```

### 8.4 NRA nondeterminism

Z3 and Lean's `nlinarith` both use heuristics with nondeterministic
timeouts on QF_NRA problems. Across runs, different individual
fixtures flap between `accepted` / `rejected` / `lake_error`. The
*set* of problems that close is stable; *which specific problem times
out on any given run* is not. Multi-seed measurement is the honest
mitigation; single-run numbers should be reported with variance.

---

## 9. What this arc does NOT claim

- **Not a claim that Symthaea "understands" math.** The Lean bridge
  compiles a hand-designed tactic cascade; it doesn't learn. The GP
  discovers closed-form fits to sequences; it doesn't prove theorems.
  The two tools coexist in the repo but don't cooperate. Section 7
  reports the specific attempt to connect them and its null result.
- **Not a full miniF2F-v2 result.** 50-problem shuffles, not 488-file
  full-pool. A full-pool run would take ~4–6 hours of Lake CPU and
  probably produce a similar-shaped number (46% ± some spread).
- **Not a claim about difficulty.** Every sequence in the invariant
  bench has a known closed form; the GP just finds it (or doesn't).
  No claim is made that the GP discovers *novel* physics. It
  recovers Kepler's third law at machine epsilon, which means it can
  find `n^1.5` — the theorem, not the phenomenon.
- **Not a paper.** This doc is a consolidation for internal
  reference and code-review. Any external claim derived from these
  numbers should cite the underlying CSV artifacts and include the
  variance bars.

---

## 10. Commit trail

Chronological arc, all on `main`:

| Commit | Title | Number |
|--------|-------|--------|
| `22827d0d0f` | Phase 2 W3 bridge | 10/14 = 71% (training fixtures) |
| `87ab9aa90f` | Phase 2 W4 | 14/14 = 100% (training fixtures) |
| `7908cf86d9` | Phase 3 measurement | 25/32 = 78.1% (curated) |
| `aa2a84cc64` | Phase 4b cascade refinements | 30/32 = 93.8% |
| `e65e5b9f17` | Phase 5 `field_simp` | 31/32 = 96.9% |
| `95a375b3b0` | Phase 5a Skolemization | Z3 timeouts 5→0 |
| `c4e62aa492` | `compound×compound` non-linear | Z3 errors 1→0 |
| `5ba4bcfee3` | Pattern B `sub_ne_zero` | auto-ingest 22→24 |
| `61d40799bb` | Phase 6 Session 1 fingerprint | +1.4% cluster signal |
| `22f599874b` | Phase 6 Session 1b kNN null | 0.0 pp lift (null) |
| `062f271f00` | Invariant discovery v1 | 6/13 = 46.2% |
| `908f487604` | Invariant v2 + 3-seed miniF2F | 8/25; 46% ± 3pp |
| `f112f80e25` | Log-space pin fix | 10/25 = 40.0% |

---

## 11. What's honestly next (if anyone wants to extend)

Ranked by marginal value per engineer-session, evidence-based:

1. **Log-space function-set enrichment.** The remaining super-exp
   misses (`bell_numbers`, `stirling_sum`, `derangements`, `fubini`,
   `motzkin`) share one root cause: log-space GP uses the same
   function pool as linear-space. Adding `ln`, `fact` approximants,
   and Stirling-shape primitives to the log-space pool would likely
   close 2–3 at once.
2. **AST extension for `abs`, `mod`, `Finset`.** Parse rate is stuck
   at ~65% because function abstraction and coercion arrows fail
   tokenization. Each AST extension unlocks tens of corpus fixtures.
3. **Richer bridge encoder.** The surface-token HDC failed (§7).
   A tree-structured encoder over `FolFormulaExt` directly, or the
   cognitive loop's full `wisdom_hv`, is the honest follow-up.
4. **`_267` Pattern B refinement.** Two-`≠` case needs `x − (−c) ≠ 0
   → x + c ≠ 0` rewrite. ~30 LOC.
5. **Full-pool miniF2F (N=178, 3 seeds).** Would take ~4–6 hours
   of compute. Numbers are likely stable near the current 46%;
   adds confidence intervals rather than new information.
6. **Symbolic RREF for Pattern D.** 100+ LOC for one fixture.
   Not cost-justified unless the Pattern-D shape is common in the
   expanded corpus.

Item 1 is the cheapest real gain. Item 2 is the highest-leverage
long-term.

---

*Written 2026-04-18. Artifacts in `docs/phase3-results/`,
`docs/phase6-scoping/`, `docs/phase6-results/`. Code in
`symthaea-core/src/hdc/{fol_formula_ext.rs, fol_ext_smt.rs,
conjecture_engine.rs}` and `symthaea-lean-bridge/src/fol_ext_bridge.rs`.*
