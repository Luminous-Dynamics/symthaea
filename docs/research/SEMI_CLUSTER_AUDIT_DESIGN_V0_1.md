# SEMI-CLUSTER-AUDIT-001 — design-based finite-population audit v0.1

**Issue:** #14051  
**Implementation:** \`symthaea-semiconductor::benchmark::cluster_audit_design\`  
**Stack:** joint observation-process ledger #13955; population partial-identification / Bernoulli audit #13650; stage-factorized observation-risk estimator #13984; common-cause context registry/decomposition #13910.  
**Status:** Rust estimator, replayable randomization and tests authored; exact-head validation pending.

## Why this is a separate estimator

Qualified-only VM coverage is not automatically population coverage. The existing independent Bernoulli unit audit (#13650) and stage-factorized HT estimator (#13984) handle valid unit-level randomized designs under their assumptions. This work adds a different design for a manufacturing system where common-cause events naturally group wafers into lots, tool/chamber/recipe epochs, shared-reference cohorts or other predeclared strata.

The sampling unit is a **whole cohort**, not an individual wafer. The estimator does not require losses within the selected cohort to be independent; it uses the observed cohort mean as one bounded sampling unit.

It remains distinct from:
- physical metrology acquisition/selection;
- K1 calibration and value qualification;
- CA-OCRC feedback-slot selection and controller update tracking;
- online conformal coverage inference;
- contextual residual decomposition.

None of those probabilities are automatically multiplied into the independent audit design.

## Design V1: stratified SRSWOR of equal-size cohorts, census within selected cohorts

The population frame is partitioned into strata. Within each stratum:
- every cluster has equal size (put unequal cluster sizes into separate strata);
- select \(C_h\) of \(G_h\) clusters by simple random sampling without replacement;
- audit **all** population units within each selected cluster;
- enforce a hard unit-level reference-metrology capacity.

The cluster is a predeclared unit of common-cause membership (e.g. lot × tool/chamber × recipe epoch). Its membership receipt is frozen before randomization. The cluster must not be defined after seeing audit losses.

For stratum \(h\):
\[
\pi_g = \frac{C_h}{G_h},\qquad
\pi_{gq} = \frac{C_h(C_h-1)}{G_h(G_h-1)}
\]
for distinct clusters \(g\ne q\), with \(\pi_{gq}=0\) when \(C_h=1<G_h\).

Let \(\bar L_g\in[0,1]\) be the complete audit mean for cluster \(g\), and \(M_h\) its common unit count. Estimate stratum risk by:
\[
\widehat R_h=\frac1{C_h}\sum_{g\in s_h}\bar L_g.
\]
Combine the strata using finite-population unit weights \(W_h=G_hM_h/N\):
\[
\widehat R=\sum_h W_h\widehat R_h.
\]

The unbiased SRSWOR variance estimator for each stratum mean is:
\[
\widehat{\mathrm{Var}}(\widehat R_h)
=\left(1-\frac{C_h}{G_h}\right)\frac{s_h^2}{C_h},
\qquad
s_h^2=\frac1{C_h-1}\sum_{g\in s_h}(\bar L_g-\widehat R_h)^2.
\]
The combined design variance is \(\sum_h W_h^2\widehat{\mathrm{Var}}(\widehat R_h)\). If \(C_h=1<G_h\), the point estimator remains design-unbiased, but the within-stratum sample variance cannot be estimated; the report therefore withholds the combined design standard error rather than inventing it.

## Conservative finite-population bound

Because each cluster mean is in \([0,1]\), emit a simultaneous Hoeffding bound over the \(H\) strata with a union-bound allocation \(\delta/H\):
\[
r_h=\begin{cases}
0,& C_h=G_h,\\
\sqrt{\frac{\log(2H/\delta)}{2C_h}},& C_h<G_h.
\end{cases}
\]
Combine the stratum radii with population weights:
\[
r=\sum_h W_h r_h,\qquad
R\in[\widehat R-r,\widehat R+r]\cap[0,1].
\]
This is conservative. It does not assume independent wafer losses inside a cluster or a Gaussian cluster-mean distribution. The design still requires actual randomized selection of clusters within strata and complete auditing of every selected cluster.

If all clusters are audited, the report returns an exact finite-population census with zero sampling variance and zero Hoeffding radius.

## Randomization commitment and replay

The v1 chronology is:

1. Freeze the complete population frame, including unit IDs, stratum IDs, cohort membership receipts and frame digest.
2. Commit to the 32-byte randomization seed **after frame freeze and before the first selection**. The commitment binds the seed, design ID, population-manifest digest, frozen frame digest and exact algorithm ID.
3. Perform the declared stratified SRSWOR sample.
4. Audit all units in the selected clusters using an independent reference method.
5. Close the audit and seal all reference records.
6. Reveal the seed and independently replay the selected cluster IDs.

The random stream is a pinned SHA-256 counter construction with design/stratum domain separation. The partial Fisher–Yates shuffle uses rejection sampling to avoid modulo bias. The evaluator reconstructs the selected cluster set and requires exact agreement with the submitted audit records.

The commitment/replay establishes deterministic ordering and prevents a post-outcome change of sample from passing the verifier. It does **not** prove that the seed came from an unpredictable or independent entropy source. The randomization provenance receipt and its issuer require a separate trust assessment.

## Independent reference evidence

Each selected unit outcome binds:
- population unit ID;
- reference label ID and revision;
- reference label payload digest;
- independent reference measurement event ID;
- reference measurement receipt;
- reference qualification report and receipt;
- bounded loss in \([0,1]\), only when reference-qualified.

The reference method identity must differ from the operational VM/metrology method. UNKNOWN/BLOCKED/non-qualified reference outcomes invalidate the population point estimate; the report returns UNKNOWN and does not replace missing losses with zero.

K1 operational qualification is not multiplied into the audit inclusion probability. The independent audit has its own reference qualification contract and its own randomized sample.

## Implementation checks

The implementation validates:
- frame and unit/cluster identities, uniqueness and content digests;
- common-cause membership receipt for every cohort;
- equal cohort size within each stratum;
- unique, complete stratum sample plan;
- frame/design/randomization-algorithm compatibility;
- seed commitment chronology and commitment digest;
- exact replayed selected cluster IDs;
- independent reference method identity;
- one complete audit record per selected cluster and every unit within it;
- per-unit reference-label revision, payload, event and qualification receipt fields;
- loss finite and in \([0,1]\);
- hard reference-audit unit capacity;
- canonical order-independent frame, design, input, audit-record and report digests.

No controller-update propensities, operational metrology-selection probabilities or K1 qualification rates are used as cluster-audit weights.

## Tests

Unit tests cover:
1. two size-stratified cohorts with weighted population risk, design variance and hard capacity;
2. full census exact result and zero uncertainty;
3. perfect within-cohort loss correlation without IID-wafer precision;
4. seed commitment mismatch;
5. \(C_h=1<G_h\) — no fabricated design variance;
6. non-qualified reference evidence — UNKNOWN, no point estimate;
7. selected cohort set differs from replay — reject;
8. hard reference capacity exceeded — reject;
9. frozen frame digest and order-invariant report replay;
10. exact pairwise inclusion probability \(0\) for \(C_h=1<G_h\).

Independent integration oracles reconstruct the counter-based random stream, Fisher–Yates selection, cluster mean, finite-population correction, pairwise inclusion probabilities and full-census finite-population mean without calling the production randomization or estimator code.

## Benchmark tournament

Compare on independently simulated finite populations:
- unit-level independent Bernoulli audit (#13650);
- stage-factorized HT under declared ignorability/positivity assumptions (#13984);
- stratified whole-cohort audit (this design);
- cluster bootstrap as a separately named resampling comparator;
- naive IID-wafer variance as a negative control;
- worst-case partial-identification bounds when audit evidence is absent or invalid.

Stress:
- within-cohort dependence from perfectly shared loss to heterogeneous losses;
- rare catastrophic clusters;
- unequal cohort sizes (must stratify by size);
- high and low between-cluster heterogeneity;
- common-reference and common-chamber shocks;
- budget from 1% to 20% of population units;
- 1 selected cluster through full census;
- reference non-response and qualification failure;
- incorrect common-cause membership as a measurement/design sensitivity.

Report:
- finite-population bias and RMSE;
- empirical coverage of the Hoeffding/design intervals;
- interval width and design-variance calibration;
- number of audited clusters and physical units;
- probability of detecting rare high-risk cohorts;
- randomization replay success;
- sensitivity to strata/cohort misclassification.

Proposed target hypotheses are 100% exact replay of committed samples, no undercoverage in the frozen Monte Carlo corpus at declared \(\delta\), no misuse of operational propensities as audit weights, and UNKNOWN for incomplete frame/randomization/reference evidence. At equal unit budget, measure when this design outperforms per-unit Bernoulli auditing; do not assume cluster audit always dominates.

## Claim ceiling

This estimates a declared finite-population mean loss under a replayable randomized cluster audit. It does not prove physical common-cause grouping, reference-method traceability or entropy quality; it does not generalize automatically to future fabs or a superpopulation. It does not authorize equipment commands, APC/R2R, wafer disposition or manufacturing intervention.
