# Spark Engine Technical Report: LCF Anomaly Investigation

**Version:** 0.1.0
**Original date:** 2026-02-05
**Revised:** 2026-07-07
**Status:** Physics-Honest Assessment
**Package:** [`spark-engine`](https://crates.io/crates/spark-engine) on crates.io ·
[github.com/Luminous-Dynamics/spark-engine](https://github.com/Luminous-Dynamics/spark-engine)

> **Revision note (2026-07-07):** this report was updated to reflect the current
> codebase (99 tests, up from 42) and two capabilities added since the original
> draft — a Bayesian belief model over the five candidate hypotheses
> (`bayesian.rs`) and an expected-information-gain experiment ranker
> (`optimal_experiment.rs`). Section 5's static priority table is now superseded
> by a regenerable, decision-theoretic ranking — see §5.1. One numeric table from
> the original draft (§2.3) is flagged as illustrative rather than verified; see
> the caveat there. No conclusion about the physics itself has changed.

---

## Executive Summary

This report presents a rigorous, physics-honest analysis of Lattice Confinement Fusion (LCF) using the Spark Engine simulation framework. Our key findings:

1. **Standard Gamow physics predicts Q << 1** at room temperature. LCF cannot produce net energy under known physics.

2. **A 40-order-of-magnitude gap exists** between NASA's observed neutron rate (~10³ n/s) and Gamow predictions (~10⁻³⁸ n/s).

3. **The most plausible explanation** is transient hot spots from X-ray absorption, which could locally achieve temperatures of ~3,500K.

4. **LCF may still be valuable** as a compact neutron source, even without energy gain.

5. **(New, 2026-07-07) The single best next experiment, by expected information gain per dollar, is the Hydrogen Control** (~$10K, 1 month) — not the Temperature Mapping study this report originally emphasized. See §5.1.

---

## 1. Standard Physics Baseline

### 1.1 Gamow Integration

The D-D fusion rate is calculated using the full Gamow integral:

```
<σv> = √(8/πμ) · (kT)^(-3/2) · ∫ σ(E) · E · exp(-E/kT - E_G/√E + U_e/kT) dE
```

Where:
- μ = reduced mass (≈ 1 amu for D-D)
- E_G = Gamow energy = 31.38 keV^0.5 for D-D
- U_e = electron screening energy

### 1.2 Room Temperature Results

| Parameter | Value | Unit |
|-----------|-------|------|
| Temperature | 300 | K |
| Screening (Pd) | 309 | eV |
| <σv> | 5.3 × 10⁻⁹⁰ | cm³/s |
| Gamow peak | 5.0 | eV |
| Q factor | 1.8 × 10⁻⁵⁷ | - |

**Conclusion:** At room temperature, <σv> is negligible. Standard physics predicts essentially zero fusion.

### 1.3 Temperature Dependence

| Temperature (K) | <σv> (cm³/s) | Q Factor |
|-----------------|--------------|----------|
| 300 | 5.3 × 10⁻⁹⁰ | 10⁻⁵⁷ |
| 1,000 | 2.9 × 10⁻⁶⁴ | 10⁻³¹ |
| 10,000 | 2.6 × 10⁻²⁹ | 10⁺⁴ |
| 100,000 | 1.2 × 10⁻¹⁶ | 10¹⁷ |

The rate increases by ~25 orders of magnitude per factor of 10 in temperature. Q > 1 requires T > 10⁷ K under standard physics.

---

## 2. The NASA Anomaly

### 2.1 Observation Summary

NASA Glenn Research Center (Steinetz et al., 2020, Phys. Rev. C) reported:

- **Neutron rate:** ~10³ n/s
- **Neutron energy:** 2.45 MeV (consistent with D-D)
- **Host material:** ErD₂ on TiD₂
- **Trigger:** 12 keV bremsstrahlung X-rays
- **Controls:** H control showed no signal

### 2.2 Gap Analysis

| Metric | Value |
|--------|-------|
| Observed rate | 10³ n/s |
| Gamow predicted | 4.3 × 10⁻³⁸ n/s |
| **Gap factor** | **2.3 × 10⁴⁰** |
| Gap orders | 40 |
| Classification | Extraordinary |

### 2.3 What Could Explain the Gap?

| Enhancement Source | Factor | Cumulative |
|--------------------|--------|------------|
| Electron screening (309 eV) | 10²⁰⁰ | 10²⁰⁰ |
| Phonon enhancement (2 modes) | 10² | 10²⁰² |
| Hot spot (10× T) | 10¹⁰ | 10²¹² |
| **Unexplained** | **10⁻¹⁷²** | 10⁴⁰ |

The combined known enhancements far exceed the gap, but each is applied at its theoretical maximum. In practice, these enhancements may not combine multiplicatively.

> **Honesty caveat (added 2026-07-07):** this table's numbers could not be reproduced
> from the current `screening_enhancement()` implementation (`physics.rs`) at any
> physically relevant energy for this system — the function returns the trivial
> value 1.0 below 10 eV collision energy by design, and is capped at 10⁶ above
> that (added later, per `clippy::manual_clamp` cleanup 2026-07-07). The 10²⁰⁰
> screening figure appears to be from an earlier, uncapped, illustrative
> back-of-envelope exercise evaluated outside the function's intended domain,
> not a live-computed result. Treat this table as **narrative intuition about
> orders of magnitude, not a verified calculation** — which is consistent with
> the original prose above already flagging the multiplicative-combination
> assumption as questionable. Recomputing it rigorously (correct energy
> convention, no artificial floor/ceiling) is open work, not done here.

---

## 3. Hypothesis Evaluation

### 3.1 Hot Spots (Most Plausible)

**Mechanism:** X-ray absorption deposits energy in ~10 nm radius, creating transient regions with T >> T_bulk for ~0.1 ps.

**Predictions:**
- Rate scales with X-ray intensity
- Weak bulk temperature dependence
- Burst structure following trigger pulses

**Required conditions:**
- Peak temperature: ~3,500 K (12× room temp)
- Cooling time: ~0.1 ps
- Plausibility: **HIGH** - thermally achievable

### 3.2 Phonon Cascade (Possible)

**Mechanism:** Multiple optical phonons coherently add their energy (~56 meV each) to the D-D center-of-mass.

**Predictions:**
- Rate increases with acoustic drive
- Resonances at phonon frequencies
- Strong temperature dependence

**Required conditions:**
- Coherent phonons needed: >50
- Plausibility: **LOW** - maintaining coherence of 50+ modes is difficult

### 3.3 Super-Screening (Unlikely)

**Mechanism:** Electron screening energy exceeds measured values due to collective effects at high loading.

**Predictions:**
- Rate correlates with D/M loading
- Should be observable in accelerator experiments

**Required conditions:**
- Screening needed: ~100,000 eV (vs. measured 309 eV)
- Plausibility: **VERY LOW** - 300× higher than any measurement

### 3.4 Measurement Error (Cannot Rule Out)

**Mechanism:** Signal is not from D-D fusion but from background, cosmic rays, or electronic noise.

**Predictions:**
- Signal persists with H instead of D
- No correlation with trigger

**Required conditions:**
- NASA's H control passed
- Plausibility: **LOW** but independent replication needed

---

## 4. Literature Survey

### 4.1 Database Summary

| Statistic | Value |
|-----------|-------|
| Total observations | 11 |
| Average credibility | 6.3/10 |
| High-credibility (≥7) | 4 |
| With neutron detection | 2 |
| With excess heat claims | 5 |

### 4.2 Most Credible Observations

1. **Raiola 2004** (9.0/10) - Screening measurement, confirmed by multiple labs
2. **Steinetz 2020** (8.5/10) - NASA neutron observation with controls
3. **Czerski 2004** (8.5/10) - Screening in Ti, confirmed
4. **Huizenga 1993** (7.0/10) - Skeptical review, documents failures

### 4.3 Screening Energy Measurements

| Material | U_e (eV) | Reference |
|----------|----------|-----------|
| Pd | 309 ± 12 | Raiola 2004 |
| Pt | 320 ± 15 | Raiola 2004 |
| Ti | 363 ± 25 | Czerski 2004 |
| Er | ~800 | Steinetz 2020 (estimated) |

These are anomalously large compared to the Debye model prediction (~25 eV) but are well-established by accelerator measurements.

---

## 5. Recommended Experimental Program

### 5.1 Superseding this section: the live, decision-theoretic ranking (added 2026-07-07)

The phase/priority tables below (§5.1–5.3 as originally written) reflect **hand-set
priorities** from the initial design. Since then, two modules make this
decision-theoretic and regenerable rather than fixed:

- `bayesian.rs` — a probability distribution (`HypothesisBelief`) over the five
  candidate explanations (hot spots, phonon cascade, super-screening,
  lattice-modified tunneling, measurement error), with a deliberately skeptical
  default prior (measurement error weighted highest, 0.40).
- `optimal_experiment.rs` — ranks every candidate experiment by **expected
  information gain (EIG) per dollar** against that belief, and derives a
  budget-constrained sequence.

Regenerate the live ranking with:

```bash
cargo run --example optimal_experiment_proposal 500000   # budget in USD
```

As of this writing (see
[`docs/DISCRIMINATING_EXPERIMENT_PROPOSAL.md`](DISCRIMINATING_EXPERIMENT_PROPOSAL.md)),
the ranking **differs from the static table below in its top pick**: the **Hydrogen
Control** experiment — $10K, 1 month — ranks #1 at 9.71 bits of information gain per
$100K, ahead of the NASA Replication and Temperature Mapping experiments the original
static priorities emphasized. This makes physical sense in hindsight: it is the
cheapest experiment that cleanly separates "real fusion" from "measurement artifact,"
the single largest source of belief uncertainty under the skeptical prior. The static
tables below are kept for historical/narrative context but the generated proposal is
now the source of truth for sequencing.

### Phase 1: Validation ($200K, 6 months)

**Goal:** Confirm the anomaly is real and from D-D fusion.

| Experiment | Cost | Priority |
|------------|------|----------|
| NASA Replication | $75K | 100% |
| H Control | $10K | 100% |
| Neutron Spectroscopy | $50K | 95% |

**Go/No-Go:** All three criteria must pass:
- Observe >100 n/s with D
- H control shows <1 n/s
- Neutron energy = 2.45 ± 0.3 MeV

### Phase 2: Mechanism ($300K, 12 months)

**Goal:** Discriminate between hypotheses.

| Experiment | Cost | Priority |
|------------|------|----------|
| Temperature Mapping | $80K | 90% |
| Acoustic Drive | $40K | 80% |
| Host Material Survey | $60K | 85% |

**Go/No-Go:** Identify plausible mechanism.

### Phase 3: Optimization ($200K, 12 months)

**Goal:** Maximize neutron output for applications.

| Experiment | Cost | Priority |
|------------|------|----------|
| Scale-Up | $50K | 70% |
| Lifetime | $30K | 65% |
| Efficiency | $40K | 60% |

**Go/No-Go:** Achieve >10⁶ n/s sustained.

---

## 6. Path Forward: Compact Neutron Source

Even with Q << 1, LCF may be valuable as a neutron source.

### 6.1 Application Requirements

| Application | Rate (n/s) | LCF Viable? |
|-------------|------------|-------------|
| Research | 10⁴ | Yes (if anomaly real) |
| NAA | 10⁶ | Possible with scale-up |
| BNCT | 10⁹ | Challenging |
| Security | 10⁸ | Challenging |

### 6.2 Competitive Comparison

| Source | Rate (n/s) | Cost | Portability |
|--------|------------|------|-------------|
| Cf-252 | 2.3×10⁶/μg | $50K/μg | Portable |
| D-D accelerator | 10⁸ | $200K | Transportable |
| D-T accelerator | 10¹⁰ | $300K | Transportable |
| LCF (if anomaly) | 10⁵/cm³ | ~$20K/cm³ | Portable |

LCF advantage: No high voltage, no tritium, potentially very compact.

---

## 7. Conclusions

1. **Physics is honest:** Standard Gamow gives Q ~ 10⁻⁵⁷ at room temperature. No energy gain is possible.

2. **The anomaly is extraordinary:** A 40-order-of-magnitude gap requires either new physics, unrecognized enhancements, or systematic error.

3. **Hot spots are the most plausible explanation:** X-ray-induced local heating to ~3,500K could explain the observations without invoking new physics.

4. **Independent replication is critical:** The NASA result must be reproduced with proper controls before investing in scale-up.

5. **Pivot to neutron source:** If the anomaly is real, LCF's value is as a compact neutron generator, not an energy source.

6. **(New, 2026-07-07) The recommended next step, by expected information gain per
   dollar, is the Hydrogen Control experiment** — cheaper and more discriminating
   than the NASA Replication or Temperature Mapping studies emphasized above. See §5.1.

---

## Appendix A: Spark Engine Modules

| Module | Description |
|--------|-------------|
| `physics` | Gamow integration, Q factor, screening |
| `lattice` | DPA tracking, lifetime model |
| `anomaly` | Gap analysis framework |
| `literature` | 11 canonical observations |
| `hypothesis_models` | Hot spot, phonon, screening models |
| `rate_gap` | Quantitative gap calculator |
| `experimental_design` | 3-phase, 9-experiment program |
| `uncertainty` | Monte Carlo propagation |
| `neutron_source` | Compact source design |
| `shielding` | Dose calculations |
| `bayesian` *(added 2026-07-07)* | Probability distribution over the 5 hypotheses, Bayes updates from observations |
| `optimal_experiment` *(added 2026-07-07)* | Expected-information-gain ranking, budget-constrained experiment sequencing |

## Appendix B: Running the Analysis

```bash
# Full analysis report
cargo run --example analysis_runner

# Live discriminating-experiment ranking (added 2026-07-07)
cargo run --example optimal_experiment_proposal 500000

# CLI tool
cargo run --bin spark -- analyze --rate 1000 --material Er
cargo run --bin spark -- hypotheses
cargo run --bin spark -- experiments --phase 1
cargo run --bin spark -- gamow --temp 1000 --screening 309
```

## Appendix C: Test Coverage

- 99 unit tests (up from 42 at original draft), 1 doc test — verified 2026-07-07
- All passing; CI (GitHub Actions) enforces `cargo fmt --check` and
  `cargo clippy -- -D warnings` on every push

## Appendix D: References

- Steinetz, B. M. et al. (2020). "Novel nuclear reactions observed in bremsstrahlung-irradiated deuterated metals." *Physical Review C*, 101(4), 044610.
- Raiola, F. et al. (2004). "Enhanced electron screening in d(d,p)t for deuterated metals." *European Physical Journal A*, 19(2), 283–287.
- Czerski, K. et al. (2004). "Enhancement of the electron screening effect for d + d fusion reactions in metallic environments." *Europhysics Letters*, 68(3), 363.
- Huizenga, J. R. (1993). *Cold Fusion: The Scientific Fiasco of the Century*. University of Rochester Press. (Skeptical review cited in §4.2 as the field's documented failure modes.)
- Brown, D. A. et al. (2018). "ENDF/B-VIII.0: The 8th major release of the nuclear reaction data library." *Nuclear Data Sheets*, 148, 1–142. (Neutron cross-section validation source, per `literature_validation.rs`.)
- Miracle, D. B. & Senkov, O. N. (2017). "A critical review of high entropy alloys and related concepts." *Acta Materialia*, 122, 448–511. (HEA property validation source.)

---

*Report generated by Spark Engine v0.1.0. Original draft 2026-02-05; revised 2026-07-07 alongside the crate's first public release on [crates.io](https://crates.io/crates/spark-engine) and [GitHub](https://github.com/Luminous-Dynamics/spark-engine).*
