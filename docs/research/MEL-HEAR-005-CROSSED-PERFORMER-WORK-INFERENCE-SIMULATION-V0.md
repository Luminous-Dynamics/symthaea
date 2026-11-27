# MEL-HEAR-005 — Crossed performer × source-work inference simulation v0

**Status:** research-only simulation result; candidate method, not qualified inference  
**Date:** 27 November 2026  
**Owner:** issue #14026  
**Composes:** #13947 musical identity reducer and #14019 paired comparison layer  
**Annotation dependency:** #14030

## Executive result

The source-work-only interval in #14019 is appropriate only when source-work clusters are defensibly independent. In a controlled crossed random-effects simulation where the same performers contribute to multiple works, work-only resampling substantially undercovers a nominal 95% interval. A two-way pigeonhole bootstrap is promising in the balanced 30-work × 20-performer design, but still undercovers when the performer axis is small (30 works × 8 performers).

This is a design warning and a candidate-method result, not a production implementation or evidence about real musician cohorts.

## Why this matters

For a paired model comparison, define the cell-level candidate-minus-baseline effect as:

    d[w,p] = delta + u[w] + v[p] + e[w,p]

where source-work effects u and performer effects v are crossed, and e is a work-performer residual. In a complete balanced W × P design, the grand-mean variance is:

    Var(mean(d)) = tau_work^2 / W + tau_performer^2 / P
                   + sigma_cell^2 / (W * P)

When the same performers recur across works, a work-only bootstrap resamples work means but does not resample the shared performer axis. Its uncertainty can therefore be much too small. This is precisely the dependence condition that #14019 already excludes from its v1 claim.

## Simulation protocol

The research-only oracle is in research/mel-hear-005/simulate_crossed_bootstrap.py. It was executed with Python 3.13.5 and NumPy 2.3.5. The retained machine-readable output is simulation-v0.json; the NumPy version is part of the evidence identity.

- 1,000 independent Monte Carlo datasets per scenario.
- Nominal 95% percentile intervals.
- 399 bootstrap replicates for the work-disjoint control; 499 for each crossed scenario.
- Work-only bootstrap resamples W source-work clusters.
- Two-way pigeonhole bootstrap independently resamples W works and P performers; each observed cell receives the product of its work and performer multiplicities.
- The same paired cell effect is used throughout; this is a paired contrast, not an independent-arm comparison.
- Fixed seeds are committed in the script. Coverage Monte Carlo standard errors are reported.

### Results

| Design | Work-only coverage | Two-way coverage | Mean interval width: work-only | Mean interval width: two-way |
|---|---:|---:|---:|---:|
| 30 work-disjoint performer/work clusters; 4 within-work repeats | 93.8% | — | 0.2261 | — |
| 30 works × 20 recurring performers | 69.5% (MCSE 1.46 pp) | 93.7% (MCSE 0.77 pp) | 0.1332 | 0.2378 |
| 30 works × 8 recurring performers | 47.3% (MCSE 1.58 pp) | 90.8% (MCSE 0.91 pp) | 0.1444 | 0.4139 |

The crossed 30 × 20 estimator's empirical standard deviation was 0.0623 versus theoretical 0.0604. In the 30 × 8 case it was 0.1142 versus theoretical 0.1127. The candidate two-way method's interval coverage is close to nominal for the larger performer axis but materially low for only eight performers.

The disjoint control is consistent with retaining #14019's one-way work bootstrap when performers are nested within independently sampled work clusters. The crossed controls show why the same method must not be used when performers recur across works.

## Interpretation and limitations

1. **Do not generalize the 30 × 20 result into a universal method claim.** The design is balanced and complete, each cell has one performer identity, outcomes follow a Gaussian random-effects oracle, and there is no missingness, ensemble membership, or high-leverage cluster.
2. **Small performer axes remain a failure mode.** At eight performers, the pigeonhole percentile interval's 90.8% empirical coverage is not acceptable evidence for a nominal 95% interval. More clusters or a separately validated finite-sample method are required.
3. **Multi-performer cells are not solved.** Do not duplicate an ensemble trial once per member and count those copies as independent. The membership-set estimand and weighting must be explicit; otherwise classify the panel as descriptive-only.
4. **Nested sessions are not crossed performers.** Preserve work → performer/session hierarchy where appropriate; collapse repeated recordings only under a preregistered aggregation rule or model the nested random effect explicitly.
5. **Simulation is not authentication.** It says nothing about whether performer pseudonyms represent distinct people, whether audio labels are true, whether a cohort is representative, or whether rights/consent are valid.
6. **Percentile bootstrap is only one candidate.** CR2/small-sample corrections and multiway wild cluster bootstrap should be compared using frozen known-answer designs. Do not choose whichever method yields the narrowest interval.

## Candidate fail-closed design classifier

- **Work-disjoint independent clusters:** keep #14019's source-work bootstrap, only when the sampling manifest supports performer/session independence across works.
- **Crossed work × performer:** require stable performer commitments and an explicit work/performer incidence matrix; compare two-way pigeonhole bootstrap with multiway wild/CR2 alternatives. Retain both axes and their cluster counts in every report.
- **Nested sessions:** use an explicit hierarchy-aware profile; session IDs alone do not establish independent clusters.
- **Ensemble / multiple performers per trial:** use an explicit ensemble/team or set-membership design and estimand. Until qualified, emit descriptive estimates only.
- **Unknown or inconsistent identity links:** descriptive-only; no population interval.
- **Small axis:** suppress confirmatory intervals or mark exploratory until preregistered simulation demonstrates acceptable coverage and power for that axis size.

There is no automatic fallback to item-wise or source-work-only bootstrap when a crossed design is declared.

## Proposed qualification gates (not yet measured)

Before this candidate enters a production inference profile:

- Run at least 2,000 Monte Carlo datasets per preregistered DGP and enough bootstrap replicates to make Monte Carlo error negligible relative to the coverage tolerance.
- For nominal 95% intervals, target 93–97% empirical coverage in every supported design, with coverage Monte Carlo uncertainty reported.
- Null type-I error, bias, interval width, and paired-effect recovery must be reported by design, not pooled.
- Identity-linkage, incidence-matrix, row-order, item-order, and paired-equality negative controls must reject or remain exactly invariant as specified.
- No interval may be emitted for missing/ambiguous cluster axes, unresolved multi-performer cells, or a cluster count below the method-specific preregistered support floor.
- Independently reproduce at least one small design against an analytical or separately implemented oracle.

These are proposed gates, not achieved results. The 30-work floor in #14019 must not be reused as a performer-axis power calculation.

## Next implementation sequence

1. Freeze a versioned sampling-design manifest and design-classification report under #14026.
2. Add balanced crossed reference vectors and an independently implemented oracle; then implement a separately versioned Rust profile.
3. Compare one-way work bootstrap, performer bootstrap, two-way pigeonhole, CR2, and multiway wild bootstrap across balanced, unbalanced, sparse, nested, and small-axis designs.
4. Add property/factor strata only after design support is adequate for each stratum.
5. Join independently annotated audio evidence through #14030 / #14005; do not treat digest shape as authentication.
6. Keep #14019's historical v1 reports unchanged.

## Reproduction

Install the pinned research dependency from the repository root, then run:

    python -m pip install -r research/mel-hear-005/requirements.txt

Then execute:

    python research/mel-hear-005/simulate_crossed_bootstrap.py

The committed JSON output should be byte-identical with the pinned Python/NumPy pair. This Python utility is an offline research oracle only; no production Melothaea code or runtime path depends on it.
