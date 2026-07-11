# symthaea-galactic

Galactic rotation-curve model comparison on the SPARC catalog: Newtonian
baryonic-only, NFW dark-matter halo, MOND, and conformal-gravity (Mannheim)
predictions, compared with honest χ²/AIC/BIC statistics, plus an HDC
residual-learnability diagnostic.

## Verified results

Ran end-to-end against the real SPARC catalog (175 galaxies, 153 after the
Q≤2/inc≥30° quality cut); see `benchmark_provenance/sparc_benchmark.json`
for the full numbers with git-SHA provenance.

| model | k/galaxy | χ²/dof | AIC | BIC |
|---|---|---|---|---|
| Newtonian (baryonic-only) | 0 | 616.2 | 1,952,253 | 1,952,253 |
| MOND (RAR) | 0 | 51.5 | 163,049 | 163,049 |
| Conformal gravity (Mannheim) | 0 | 150.3 | 476,088 | 476,088 |
| NFW halo | 2 | 6.7 | 19,829 | 21,684 |

Sane and expected: pure Newtonian gravity fails badly (this failure **is**
the rotation-curve problem), MOND recovers most of the gap with zero free
parameters, conformal gravity helps over Newtonian but substantially
underperforms MOND at fixed Υ, and NFW — unsurprisingly, given its two
free parameters per galaxy — fits best of all four. 15/153 NFW fits did not
converge (flagged, not silently dropped).

**External validation**: as a hard, citable check independent of our own
statistics, `rar_scatter_is_in_published_ballpark` (in `gravity_models/mond.rs`,
`#[ignore]`d, needs real data) computes the actual scatter of our RAR
implementation against 3,166 real SPARC points: **mean bias −0.026 dex,
scatter 0.177 dex**. McGaugh, Lelli & Schombert (2016) report ~0.11–0.13 dex
intrinsic scatter under their tighter point selection (they additionally
exclude beam-smeared and inner-radius points we don't filter out); our
looser cut landing close to and just above their floor is exactly the
expected relationship if the implementation is correct — a real formula bug
(wrong exponent, wrong acceleration conversion) would blow this out to
multiple dex, not a few tenths.

**Residual-learnability diagnostic — a genuine null result, investigated,
not just observed.** Ran at both `epochs=1` (fast smoke-test) and the
default `epochs=5`; the numbers barely move between them (e.g. Newtonian
main-diagnostic R²: −0.0221 → −0.0137). All four models show near-zero or
negative held-out R² at both settings — the regressor isn't beating the
mean-predictor baseline for *any* model, including ones we know are wrong
(Newtonian, conformal). That rules out "just needs more epochs."

We went further: `loss_curve_probe_newtonian_20_epochs` (in
`hdc_residual.rs`, `#[ignore]`d) trains Newtonian's residuals for 20 epochs
and prints the per-epoch RMS. Training loss converges cleanly (20.9 → 17.6,
plateauing by epoch ~14) — it's not a stuck optimizer. But held-out R² stays
negative, which is the textbook overfitting fingerprint: the GLU readout has
2×16,384 = 32,768 trainable weights against splits with only ~130
independent training galaxies (points within one galaxy share the same
luminosity/distance/inclination/gas-fraction inputs, so the real degrees of
freedom are closer to galaxy count than point count). We added L2 weight
decay (`WEIGHT_DECAY` in `hdc_residual.rs`) targeting exactly this — and it
did **not** fix it: held-out R² with regularization was −0.0711, no better
than (arguably within noise of) the unregularized baseline. Kept anyway as
sound practice for an overparameterized model, but it should not be read as
"the fix." The honest conclusion after this investigation: the negative
held-out R² isn't simple underfitting or unregularized overfitting — it
likely needs either real architectural capacity (multi-head GLU, richer
features) or should stand as a documented limitation rather than something
a training-hyperparameter tweak resolves. Flagged as a known limitation,
not hidden — see `benchmark_provenance/sparc_benchmark.json` for the full
epochs=5 numbers. The classical χ²/AIC/BIC comparison above is unaffected by
this; it doesn't depend on the HDC regressor at all.

## Scope — what this crate does and does not claim

This crate tests rotation-curve **phenomenology only**. It fits ~175 real
galaxy rotation curves and asks which model predicts them best, with correct
accounting for each model's free parameters.

It says **nothing** about:
- Quantization, ghosts (Ostrogradsky instability), or unitarity of any
  gravity theory. Conformal gravity's fourth-order field equations raise
  serious quantum-consistency questions (Mannheim & Bender's PT-symmetric
  proposal, Maldacena's boundary-condition argument) that are entirely
  outside what a rotation-curve fit can address.
- The full dark-matter case. Rotation curves are the *easiest* piece of
  dark-matter evidence to explain away — CMB acoustic peaks, the Bullet
  Cluster's lensing/baryon separation, and large-scale structure formation
  are not addressed here at all. A strong result for any alternative model
  in this benchmark is roughly 20% of the dark-matter problem, not all of it.

## Models compared

| Model | Free params/galaxy | Formula |
|---|---|---|
| Newtonian (baryonic-only) | 0 | V² = V_bar² |
| MOND (RAR) | 0 | g_obs = g_bar/(1−e^(−√(g_bar/a₀))), a₀=1.2e-10 m/s² |
| Conformal gravity (Mannheim) | 0 | V² = V_bar² + γ★N★c²r/2 + γ₀c²r/2 − κc²r² |
| NFW halo | 2 (V200, c) | standard NFW circular-velocity profile |

Mass-to-light ratios (Υ_disk=0.5, Υ_bulge=0.7 at [3.6]μm) are **fixed across
all four models** for fairness. Published NFW/SPARC fits often free Υ per
galaxy, which would improve its numbers further — we deliberately don't do
that here, since giving one model an extra tunable knob the others lack
would make the comparison meaningless.

## Known criticisms of conformal gravity (cited, not resolved here)

- Flanagan (2006), *PRD* 74, 023002 — argues the non-relativistic limit
  Mannheim uses does not follow once matter couplings are handled
  consistently.
- Hobson & Lasenby (2021), *PRD* 104, 064014 — argues the published fits
  depend on an inconsistent choice of conformal frame.

This crate implements and tests the published phenomenological formula
(Mannheim & O'Brien 2012); it does not adjudicate this dispute, which is a
field-theory question, not a curve-fitting question.

## The residual-learnability diagnostic — and its asymmetry caveat

Beyond the classical statistics, this crate trains one HDC+CfC+GLU regressor
per model on that model's normalized residuals `(v_obs−v_model)/e_v_obs`
(architecture: `symthaea-nuclear`'s `HdcMassPredictor`, adapted). The idea:
a correct model leaves residuals with no learnable structure; a wrong model
leaves structure a flexible learner can extract.

**This is not apples-to-apples across models.** NFW's two per-galaxy
parameters (V200, c) absorb galaxy-level structure *by construction* — its
residuals are less learnable independent of whether NFW is "true." The fair
learnability comparison is among the three 0-parameter models (Newtonian,
MOND, conformal); NFW is included as the flexible reference point, not a
peer in this specific comparison. Every results artifact this crate
produces states this caveat next to the numbers.

## Running

```bash
# One-time data fetch (~0.3 MB)
bash scripts/download_sparc.sh

# Unit tests that don't need data
cargo test -p symthaea-galactic

# Tests that need the real SPARC sample (175 galaxies): full load/parse test
# plus the RAR-scatter external-validation gate (rar_scatter_is_in_published_ballpark)
cargo test -p symthaea-galactic -- --ignored

# Full benchmark: classical fits + residual diagnostic + LSB extrapolation holdout
cargo run --release -p symthaea-galactic --example benchmark_sparc
```

Override the data location with `SYMTHAEA_SPARC_DATA_DIR`. Override the HDC
residual regressor's training depth with `SYMTHAEA_SPARC_RESIDUAL_EPOCHS`
(default 5) — the diagnostic's 16,384-D vector ops scale linearly with this,
so it's the main lever for trading wall-clock time on slower/contended
machines. Note this only trades speed, not diagnostic quality: measured
results at `epochs=1` and `epochs=5` are nearly identical (see "Verified
results" above) — this is not currently a knob that produces more
informative numbers, just a faster or slower path to the same (currently
uninformative) ones.

Results are written to two places:
- `data/benchmarks/sparc/results.json` — ephemeral (this path is gitignored)
- `benchmark_provenance/sparc_benchmark.json` (in this crate) — **committed**
  (note: a directory literally named `results/` is blanket-gitignored at the
  symthaea workspace root, so this artifact deliberately lives elsewhere),
  carries git SHA + dataset provenance + quality cuts + full per-model numbers

## Known approximations

- Sign-preserving baryonic quadrature (`v·|v|` not `v²`) to correctly handle
  SPARC's negative-Vgas central gas depressions.
- Stellar mass for the conformal-gravity N★ term uses `Υ_disk × L[3.6]`
  applied to *total* luminosity (disk+bulge), which slightly underweights
  bulge-heavy systems — SPARC's per-component luminosity split isn't in the
  public table.
- No statistical error floor is applied to SPARC's published velocity
  uncertainties; a tiny numerical floor (1e-3 km/s) exists purely to guard
  against division by zero.

## References

- Lelli, F., McGaugh, S. S., & Schombert, J. M. (2016). SPARC: Mass models
  for 175 disk galaxies. *AJ*, 152, 157.
- McGaugh, S. S., Lelli, F., & Schombert, J. M. (2016). Radial acceleration
  relation in rotationally supported galaxies. *PRL*, 117, 201101.
- Navarro, J. F., Frenk, C. S., & White, S. D. M. (1996). The structure of
  cold dark matter halos. *ApJ*, 462, 563.
- Mannheim, P. D., & O'Brien, J. G. (2012). Fitting galactic rotation curves
  with conformal gravity. *PRD*, 85, 124020.
- Flanagan, É. É. (2006). Fourth-order Weyl gravity. *PRD*, 74, 023002.
- Hobson, M. P., & Lasenby, A. N. (2021). *PRD*, 104, 064014.
