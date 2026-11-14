# PLASMA-EQUIV-DESIGN-001 — preregistered cluster-aware equivalence design

**Date:** 2026-11-14  
**Status:** S0 design/result receipt audit; integrated into transfer evidence composition  
**Issue:** [#12064](https://github.com/Luminous-Dynamics/symthaea/issues/12064)  
**Stack:** draft #12028 → #12051 → this branch  
**Composes with:** #9841 statistical power planning, #9215 experiment DSL, #11907 plasma OED, #11784 interlaboratory metrology, #12026 method-to-claim scope, #12033 transfer observation composition and #12002 qualification transfer.

## Executive thesis

A transfer interval is not self-justifying. A confidence interval plus a pile of coupon identifiers does not show that the experimental design had the correct independent unit, enough independent clusters, a frozen equivalence margin, an adequate preregistered precision/power plan, or a non-leaking analysis path.

This increment moves from **interval-shaped evidence** to **design-and-result-linked equivalence evidence**.

It binds the immutable pre-outcome design, independent-unit grouping, specimen manifest, method-qualified baseline/candidate measurements and analysis result into one structural receipt. The design audit and transfer-composition audit remain distinct so each can be reviewed independently; the latter now requires the design receipt.

## Research and industry comparison

### Existing statistical capability

Symthaea Statistics already has:
- one- and two-sample TOST equivalence testing;
- ICC(1,1), ICC(2,1) and ICC(3,1);
- cluster-robust CR0/CR1 covariance and Newey-West HAC;
- deterministic bootstrap/permutation procedures;
- approximate prospective power and minimum sample-size calculations for several common estimands.

The Statistics 0.4 migration note explicitly identifies its power functions as large-sample normal approximations for planning/screening and points toward exact noncentral-t or simulation approaches for small samples, unequal allocation and complex dependence. The design receipt therefore references shared numerical work rather than reimplementing it. A planning receipt carries the claim ceiling `planning_only`; this auditor does not certify power.

Related internal owner #9841 makes the crucial general distinction that many observations within one independently generated model pair do not create many independent training replicates. Plasma transfer has the same issue: many coupons cut from one material lot do not become many independent production lots.

### Standards and aerospace practice

ASTM D3163-01R23 is an active rigid-plastic single-lap method for comparative shear testing. ASTM explicitly cautions that single-lap strength on a small specimen is not, without further analysis, a design allowable for a structurally different joint. ASTM D5868-01R23 extends comparative lap-shear testing to FRP-to-FRP and FRP-to-metal joints. Neither standard makes a confidence interval or one material-pair result a universal transfer certificate.

NASA PRC-1001 Rev G provides an example of mature process qualification: detailed bonding instructions are qualified for each material combination, and procedure/material changes require requalification. It prescribes a minimum ten room-temperature lap-shear specimens for a specified qualification path when using its stated assumptions. This branch does not criticize or replace that procedure. The engineering opportunity is to make the complete evidence chain explicit across *independent production groups, material/process changes, endpoint methods, calibration, and the transfer result* rather than using coupon count as a proxy for study quality.

Official references:
- ASTM D3163-01R23: https://store.astm.org/standards/d3163
- ASTM D5868-01R23: https://store.astm.org/standards/d5868
- NASA PRC-1001 Rev G: https://www.nasa.gov/wp-content/uploads/2023/03/prc-1001-current.pdf
- Symthaea Statistics numerical scope: `crates/domains/symthaea-statistics` in this repository.
- Power-planning design precedent: [#9841](https://github.com/Luminous-Dynamics/symthaea/issues/9841).

### What we can do better

Do not compete with ASTM by inventing a new lap-shear procedure, with NASA by claiming a superior process specification, or with the Statistics crate by cloning TOST and power calculations.

The Luminous differentiator to demonstrate is a vendor-neutral evidence composition layer:
- exact test-method/edition/material/geometry/endpoint scope;
- explicit lot/batch/head/day/line/site as the independent unit;
- paired-cluster semantics when pairing is legitimate;
- an immutable design digest frozen before outcome availability;
- explicit margin, alpha, CI convention, assumed effect and variance sensitivity;
- exact included/excluded specimen identities and reasons;
- a source-bound power/precision planning receipt;
- method-qualified observation receipts;
- an analysis result bound to the same design and sample manifest;
- lifecycle invalidation when the relevant upstream calibration, lot, process or method evidence changes.

That is a stronger qualification *evidence architecture*, not a claim that its statistics are more accurate than a validated analysis package.

## S0 design receipt

The Rust auditor lives in `xtask/src/plasma_equivalence_design.rs`. It consumes `EquivalenceDesignReceipt`, which requires:

### Frozen design identity
- design ID, plan SHA-256, endpoint, estimand, quantity unit;
- registered analysis profile;
- independent-unit type and grouping scheme;
- equivalence lower/upper margins, alpha and corresponding TOST confidence level;
- design-freeze time, first-outcome-available time and explicit post-outcome-change flag;
- randomization, analysis, exclusion and time-source references.

### Planning assumptions
- target power and target precision half-width;
- effect under the alternative;
- low / central / high positive variance scenarios;
- assumptions digest;
- power/precision receipt ID, calculation digest, plan digest, endpoint, grouping and unit;
- matching assumptions and target values;
- `planning_only` claim ceiling.

The receipt verifies consistency of these references and values. It does **not** re-run or authenticate the power calculation, infer the true variance, or assert a powered experiment.

### Actual analysis cohort
Every included baseline/candidate specimen binds a specimen ID to an independent cluster ID. Excluded specimens bind their arm, specimen ID, cluster ID and reason reference. The auditor checks:
- no duplicate specimen identity within an arm;
- no specimen identity overlap across arms;
- no cluster overlap for `independent_arms`;
- exact baseline/candidate cluster-set equality for `paired_by_cluster`;
- paired clusters counted once as independent units, not twice as two arms;
- actual specimen and independent-cluster counts meet the frozen minima;
- attrition remains within separate specimen and cluster allowances;
- method-qualified receipt specimen identities exactly match included analysis specimens;
- analysis result binds the same plan, endpoint, analysis profile, sample-manifest digest, sample counts, cluster counts, equivalence margins, alpha and confidence level.

The result stores the observed confidence interval and a distinct outcome field. A structurally valid study can yield `EQUIVALENCE_NOT_ESTABLISHED`; the audit must not convert that negative outcome into an invalid-study disposition or suppress it.

### TOST convention

For the two-one-sided-test profile used by this S0 contract, alpha is checked against the corresponding two-sided interval confidence level:

[
  	ext{CI confidence}=1-2\alpha
]

For example, alpha 0.05 corresponds to a 90% two-sided interval for the conventional TOST decision. A different method/profile needs a separate registered contract; it must not quietly reuse this profile.

## Integrated transfer composition

The composition audit now consumes `EquivalenceDesignReceipt` for every endpoint in `EndpointEvidence`.

It verifies the design receipt's exact baseline and candidate method receipts against the receipts used by transfer composition, field-by-field—including method ID, edition, evidence kind, material class, geometry, endpoint, claim type, protocol IDs/digests, raw-data digests and specimen IDs.

Therefore a valid design for a different set of observations cannot be substituted into the transfer bridge. The transfer audit still retains all its existing endpoint, material, geometry, protocol, cohort and claim-scope checks.

New composition fixture: `data/research/plasma_evidence_composition_fixture_v0_2.json` (15 cases / 16 endpoint receipts). The composition command now defaults to v0.2.

## Frozen synthetic challenge

Standalone fixture: `data/research/plasma_equivalence_design_fixture_v0_1.json`.

The current 23 cases are:
- **3 PASS** — independent-lot design, paired-by-lot design, and valid design with an interval that does not establish equivalence;
- **6 BLOCKED** — too few independent lots despite many coupons, missing power-calculation digest, missing exclusion reason, attrition above allowance, unknown independent unit and missing randomization protocol reference;
- **14 REJECTED** — arm/cluster overlap, bad paired cluster map, post-outcome freeze, changed margin, plan/power digest mismatch, analysis-manifest mismatch, method receipt/sample mismatch, confidence/alpha mismatch, planning-claim escalation, endpoint mismatch, invalid pairing unit, variance envelope disorder, power assumptions digest mismatch and zero effect alternative.

Exact latest branch files were fetched back and independently replayed in JavaScript:
- standalone design fixture: **23/23** expected dispositions and equivalence outcomes;
- integrated transfer-composition fixture: **15/15** expected dispositions across 16 endpoint receipts.

Synthetic PASS is a contract result only. It is not a power result, a TOST computation, a calibrated statistical conclusion or evidence that a material is equivalent.

## Benchmark programme

Compare under identical frozen studies:
1. interval-only transfer review;
2. coupon-count threshold without independent cluster lineage;
3. independent cluster count only, without preregistration or manifest binding;
4. full design receipt and method-qualified observation composition.

Report separate metrics:
- false-PASS rate on pseudo-replicated designs;
- post-outcome-change detection;
- plan/result digest and sample-manifest mismatch detection;
- cluster-map mismatch and independent-unit shortfall detection;
- attrition and exclusion-reason audit coverage;
- baseline/candidate method-receipt mismatch detection;
- false BLOCKED rate on valid known-answer designs;
- percentage of intervals correctly labeled EQUIVALENCE_ESTABLISHED versus NOT_ESTABLISHED;
- reviewer time and missing-evidence rate.

Prospective software-contract targets:
- zero PASS with a post-outcome plan freeze;
- zero acceptance of coupon count as independent cluster count;
- 100% detection of known specimen/cluster overlap and paired-map errors;
- 100% detection of changes to margin, alpha, estimand, analysis profile or planning assumptions;
- 100% detection of actual clusters below frozen minima;
- zero transfer PASS without design receipts that point to the exact endpoint observation receipts;
- deterministic replay across JSON array permutation when specimen/cluster membership and sequence identity are unchanged.

All figures are targets, not measured results. A challenge suite with 23 cases is an engineering regression corpus, not a statistically powered benchmark of the method's real-world performance.

## Ownership and integration

- #9841 / Symthaea Statistics: numerical power/precision methods, TOST, ICC, cluster covariance, independent numerical references.
- #9215: generic preregistered experiment protocol and compiler.
- #11907: plasma measurement selection / OED.
- #11784: generic round-robin and interlaboratory metrology.
- #12026: exact method-to-claim admissibility.
- #12033: method-qualified endpoint composition.
- #12002: qualification transfer and equivalence bridge decision.
- #11971: bond aging/survival analysis.
- #6414: generic evidence-DAG resolution and invalidation closure.

The design auditor checks that a closure/plan/reference is declared and linked. It does not resolve/authenticate the generic evidence DAG, plan bytes, sample data bytes, calibration certificates, event signatures or source artifacts. The digest fields are shape-checked and cross-bound; this is not cryptographic artifact verification.

## Commands

```bash
cargo run -p xtask -- plasma-equivalence-design-audit
cargo run -p xtask -- plasma-evidence-composition-audit
```

The composition command now consumes fixture v0.2, which carries design receipts for every transfer endpoint.

## Validation boundary

- Standalone synthetic replay: **23/23**.
- Integrated composition synthetic replay: **15/15** across 16 endpoint receipts.
- Seven design Rust tests and six composition Rust tests are authored but **not executed**.
- Rust formatting, compilation, Clippy and exact-head hosted CI are **not verified**.
- No power calculation is executed here.
- No confidence interval is calculated here.
- No independent lot authenticity is established.
- No statistical assumptions, variance scenario, lab execution or calibration is authenticated.
- No physical bond performance, process equivalence, design allowable, service lifetime, product qualification, safety or production release authority is claimed.
