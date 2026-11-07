# Phase-response confirmatory family and anytime-FWER V1

**Tracking:** #10948 — SCI-CONNECTOMICS-100  
**Composes:** #10919 / PR #10922, #10897 / PR #10898 and #9110.  
**Guarantee:** finite-family anytime familywise error rate (FWER), conditional on valid component e-processes in one shared global filtration.  
**Status:** exact synthetic family-allocation and persistent-decision auditor; no biological result is claimed.

## 1. Why a family layer is needed

The phase-response tensor has many potential inferential contrasts: directed sender/recipient pairs, contexts, phase and magnitude bins, response delays, and endpoints. These contrasts are dependent and may be observed asynchronously.

A per-claim anytime-valid monitor does not automatically control the probability of one or more false claims across the entire programme. Nor is a running maximum of an e-process itself an e-value for ordinary e-BH.

## 2. V1 theorem and exact scope

Let the finite family be frozen before confirmatory outcomes. For each null claim i, let E_i,t be a nonnegative test supermartingale in the **same global filtration** F_t, including all adaptive selection, other-stream observations, stopping and missingness. A dormant claim is held constant.

Choose ex-ante nonnegative weights w_i with sum_i w_i <= 1. For family level alpha, reject claim i at its first stage t satisfying:

    sup_{s <= t} E_i,s >= 1 / (alpha * w_i)

For every true null H_i, Ville's inequality gives:

    P_Hi(ever reject i) <= alpha * w_i

No independence between claims is required. By the union bound:

    FWER = P(any true null is ever rejected)
         <= sum_{i in true nulls} alpha * w_i
         <= alpha.

V1 is intentionally the conservative, dependence-robust weighted anytime-Bonferroni construction. The running supremum is used only for this per-claim crossing event. It is not treated as an e-value, averaged, or passed to e-BH.

## 3. Critical filtration condition

Each component process must be valid in the common global filtration, not merely in its local observation stream. If claim i is sampled adaptively after observing another claim, the conditional e-process guarantee for i must still hold given the complete global past. A local-filtration e-process cannot be upgraded by storing a digest or changing a status string.

V1 now consumes the typed `GlobalLedger` and `FiltrationAdmission` from the as-of filtration auditor (#10982 / PR #10997) through a direct Rust path dependency. It calls `audit_ledger` and `audit_admission` for every claim. Only `ADMITTED_DIRECT_GLOBAL` and `ADMITTED_AFTER_LIFT` are combinable; local-only, incomparable, malformed, stale or mismatched receipts fail closed.

Each claim binds:
- campaign and process ID;
- exact null specification digest;
- input-process digest and output-process digest;
- source and target filtration identity/digest;
- observed global event IDs and process-output availability sequence;
- independently reviewed proof receipt and assumption digest;
- exact direct-global or adjuster-lifted admission mode.

The output-process digest is recomputed from the exact rational path with this canonical encoding:

    symthaea-eprocess-path-v1\\n
    <reduced rational E_0>\\n
    <reduced rational E_1>\\n
    ...

The path is hashed as UTF-8 bytes with SHA-256. The admission's output digest and the claim's path digest must both equal this recomputed value. The proof-receipt digest must also match the claim's validity receipt.

This closes the prior status-string gap: a caller cannot turn a local stream into a combinable stream merely by setting `VALIDATED_GLOBAL_FILTRATION_EPROCESS`. It still does **not** prove the upstream supermartingale or adjuster theorem. The as-of auditor verifies chronology and receipt binding only; independent statistical review must inspect the proof referenced by the receipt.

## 4. Query identity and precommitment

Each claim key retains the complete directional contrast:
- sender leg and recipient leg;
- probe class and descending neuron identity;
- context;
- perturbation phase bin and magnitude bin;
- delay index and endpoint.

Reverse sender/recipient directions, other contexts, distinct DNg labels, doses or endpoints are distinct claims. The confirmatory family, query/null digests and risk weights are fixed before opening the confirmation split.

Exploration can create candidate claims. It cannot retroactively add confirmatory claims using the same observations. A new candidate needs a new, independent confirmation family with its own precommitted budget.

## 5. Exact synthetic known answer

Family alpha = 1/10; weights = (1/2, 1/4, 1/4); total weight = 1.

| Claim | Weight | E-process path | Threshold | First rejection |
|---|---:|---|---:|---:|
| C1 local reset | 1/2 | 1, 5, 25 | 20 | stage 2 |
| C2 off-diagonal transfer | 1/4 | 1, 10, 50 | 40 | stage 2 |
| C3 contact recovery | 1/4 | 1, 2, 12 | 40 | none |

The family FWER bound is alpha * sum(weight) = 1/10. This arithmetic is a software known answer, not an empirical error-rate estimate or biological outcome.

## 6. What V1 deliberately does not do

- No post-outcome weight changes, alpha recycling or re-normalization.
- No FDR, FDR-sup or power claim.
- No e-BH on running maxima.
- No countably infinite or adaptively discovered claim family.
- No authentication of upstream signatures or source evidence.
- No biological model validation or mechanism attribution.

## 7. Baselines and next tranche

Keep assurance targets on separate leaderboards:
- weighted anytime e-Bonferroni / V1 (FWER);
- terminal Holm/Bonferroni (fixed-horizon FWER);
- asymptotic online FWER under dependent test statistics using consistent weights / low-intensity bootstrap (Jankovic, Fischer & Brannath, 2 July 2026, DOI: 10.1177/09622802261461776). Report as a separate **asymptotic** guarantee, not as finite-sample anytime-valid FWER;
- naive current-value e-BH and naive running-max e-BH (negative controls, FDR/FDR-sup);
- A1/A2 adjuster-based running-sup e-BH from the Carefree multiple-testing work and PR #9110 (FDR-sup);
- e-LOND / e-GAI and e-PS only under their documented feedback/arrival assumptions;
- projective SupFDR closure from #9196 / #9777 only after their theorem and implementation gates qualify.

The next power-oriented tranche should compose the proved A1/A2 adjusters with the phase-response claim family. It should not replace the simple FWER backstop and must preserve the common-global-filtration requirement.

## 8. Claim ceiling

A passing V1 contract establishes exact rational allocation, per-claim thresholding and persistent family decisions for the finite synthetic family, after replaying and structurally admitting each typed as-of filtration receipt and recomputing each e-process path digest. It still assumes the upstream direct-global proof or filtration-lifting theorem is mathematically valid; this software does not prove it. It does not establish FDR-sup, identify a biological mechanism, or demonstrate SOTA superiority.
