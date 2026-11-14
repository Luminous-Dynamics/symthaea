# Plasma equivalence-design evidence — November 14, 2026

**Issue:** [#12064](https://github.com/Luminous-Dynamics/symthaea/issues/12064)  
**Branch:** `research/plasma-equivalence-design-v0-1`  
**Dependency:** [draft PR #12051](https://github.com/Luminous-Dynamics/symthaea/pull/12051)

## Why this increment exists

A transfer interval is not self-justifying. A confidence interval plus coupon IDs does not show that a study had the right independent unit, sufficient independent production clusters, a frozen equivalence margin, a preregistered analysis, or a defensible attrition plan.

This increment moves from **interval-shaped evidence** to **design-and-result-linked equivalence evidence**. It binds the frozen design, independent-unit grouping, specimens, method-qualified observations and analysis result. It does not create a new statistics engine.

## Research and industry position

The repository's Statistics crate already provides TOST equivalence tests, ICC(1,1)/ICC(2,1)/ICC(3,1), cluster-robust covariance, bootstrap/permutation methods, and approximate power/minimum-sample-size calculations. Its 0.4 migration note states that the power routines are large-sample normal approximations for planning/screening, and that exact noncentral-t or simulation methods may be preferable near minimum sample sizes or under complex dependence. The existing power-planning issue #9841 also correctly distinguishes many within-pair task observations from independent training replications.

The plasma analogue is material lot, adhesive batch, plasma head, production day, line or site. Ten coupons from one lot are not ten independent lots.

ASTM D3163-01R23 is the active rigid-plastic single-lap comparative method; ASTM D5868-01R23 covers FRP-to-FRP and FRP-to-metal single-lap comparisons. ASTM cautions against treating single-lap measurements as design allowables for structurally different joints without further analysis. NASA PRC-1001 Rev G is an example of process-specific adhesive bonding qualification, including requalification when procedures or materials change. This project does not replace those methods or specifications.

Official references:
- ASTM D3163-01R23: https://store.astm.org/standards/d3163
- ASTM D5868-01R23: https://store.astm.org/standards/d5868
- NASA PRC-1001 Rev G: https://www.nasa.gov/wp-content/uploads/2023/03/prc-1001-current.pdf
- Symthaea Statistics numerical scope: `crates/domains/symthaea-statistics`
- Independent model-pair power design: [#9841](https://github.com/Luminous-Dynamics/symthaea/issues/9841)

The differentiator to prove is not a better lap-shear method or more accurate generic power calculation. It is the vendor-neutral evidence architecture connecting method/edition, material/geometry, independent lot, process, calibration, frozen design, sample manifest and exact result—and later invalidating dependent claims when those inputs change.

## Implemented design receipt

Rust module: `xtask/src/plasma_equivalence_design.rs`  
Fixture: `data/research/plasma_equivalence_design_fixture_v0_1.json`  
Command:

```bash
cargo run -p xtask -- plasma-equivalence-design-audit
```

The typed receipt binds:
- immutable design ID and plan digest;
- endpoint, estimand, quantity unit and registered TOST analysis profile;
- independent-unit type and grouping scheme (`independent_arms` or `paired_by_cluster`);
- equivalence margins, alpha and matching two-sided CI confidence;
- freeze time, first-outcome-available time and post-outcome-change flag;
- planned sample and independent-cluster minima;
- specimen and cluster attrition allowances;
- assumed effect and low/central/high variance scenarios;
- power/precision receipt, matching assumptions digest and `planning_only` claim ceiling;
- randomization, analysis, exclusion and time-source references;
- actual included specimen/cluster manifest and excluded specimens with reason references;
- exact method-qualified baseline/candidate receipts;
- analysis result and interval, bound to the same plan, endpoint, profile and sample-manifest digest.

Independent arms cannot share specimen or cluster IDs. Paired designs must have exact baseline/candidate cluster-set equality and count each paired cluster once as the independent unit. The analysis receipt must report the actual included specimen and cluster counts.

### Design quality is separate from the measured outcome

A valid study design may still fail to establish equivalence. The audit returns:
- `PASS / EQUIVALENCE_ESTABLISHED` when the linked interval is inside the frozen margins;
- `PASS / EQUIVALENCE_NOT_ESTABLISHED` when the design/result linkage is valid but the interval crosses a margin;
- `BLOCKED` for missing planning, lineage, independent-unit or sample evidence;
- `REJECTED` for known contradictions, post-hoc changes, cluster reuse or mismatched plan/result.

This prevents a structurally valid preregistration audit from being misrepresented as a positive material result.

## Transfer composition integration

The existing transfer-composition audit in `xtask/src/plasma_evidence_composition.rs` now requires an `EquivalenceDesignReceipt` for every endpoint. It checks that the design embeds the **same exact baseline and candidate method receipts** used by transfer composition, comparing method ID, edition, evidence kind, material class, geometry, endpoint, claim, protocol, raw-data digest and specimen IDs.

The integrated fixture is `data/research/plasma_evidence_composition_fixture_v0_2.json`. It contains 15 transfer cases and 16 endpoint receipts. The composition command now defaults to v0.2.

Commands:

```bash
cargo run -p xtask -- plasma-equivalence-design-audit
cargo run -p xtask -- plasma-evidence-composition-audit
```

## Frozen challenge results

Latest exact-branch source and fixtures were fetched back and independently replayed in JavaScript:

| Fixture | Cases | PASS | BLOCKED | REJECTED | Replay |
|---|---:|---:|---:|---:|---:|
| Standalone equivalence design | 23 | 3 | 6 | 14 | **23/23** |
| Integrated transfer composition | 15 | 2 | 3 | 10 | **15/15** |

Adversarial design cases cover:
- coupon pseudo-replication (many coupons but too few lots);
- overlapping independent-arm lots and invalid paired-cluster maps;
- post-outcome design freeze or explicit post-outcome changes;
- post-hoc margin/alpha changes;
- mismatched plan, assumptions and sample-manifest digests;
- missing power calculation or randomization references;
- missing exclusion reasons and excess attrition;
- unknown independent unit;
- invalid equivalence confidence/alpha convention;
- method-receipt/specimen mismatch;
- analysis interval outside equivalence margins;
- promotion of a planning receipt into a certified-power claim.

These are deterministic software-contract fixtures, not statistically powered tests of real plasma bonding.

## Benchmark programme

Compare interval-only review, coupon-count-only review, independent-cluster-count review without preregistration, and the full design/result receipt. Report false-PASS, pseudo-replication detection, post-hoc-change detection, cluster-map accuracy, attrition handling, false BLOCKED, equivalence-outcome labeling and reviewer burden separately.

Prospective targets (not current results):
- zero PASS for post-outcome plan freezes;
- zero acceptance of coupon count as independent cluster count;
- 100% detection of specimen and cluster overlap / paired-map errors;
- 100% detection of changes to margin, alpha, estimand, method profile and planning assumptions;
- 100% detection of actual independent-cluster counts below frozen minima;
- zero transfer PASS without a design receipt bound to the exact endpoint observation receipts;
- deterministic replay under input-array permutation when identities and memberships are unchanged.

## Ownership boundaries

- #9841 / Statistics owns numerical power and precision validity.
- #9215 owns generic experiment preregistration and protocol compilation.
- #11907 owns plasma OED / measurement selection.
- #11784 owns generic round-robin and interlaboratory metrology.
- #12026 owns method-to-claim scope.
- #12033 owns method-qualified transfer observation composition.
- #12002 owns qualification transfer and equivalence decisions.
- #11971 owns bond aging and survival evidence.
- #6414 owns generic evidence-DAG and invalidation closure.

No duplicate TOST, power, bootstrap, ICC, DOE, randomization, MSA, material registry or evidence-DAG engine is introduced.

## Validation boundary

- Independent synthetic replay: 23/23 design cases and 15/15 composition cases.
- Seven design Rust tests and seven composition Rust tests are authored but **not executed**.
- Rust formatting, compilation, Clippy and exact-head hosted CI are **not verified**.
- This module does not calculate power, TOST, confidence intervals, ICC or bootstrap results.
- SHA fields are shape-checked and cross-bound; referenced plan/sample/artifact bytes and signatures are not authenticated.
- The actual independent-lot identity and statistical assumptions are not authenticated.
- No physical bond equivalence, process qualification, design allowable, service life, product release or production authority is claimed.
