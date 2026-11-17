# Optical ISRU common-cause uncertainty and bitemporal qualification reference v0.1

**Status:** standalone factor model authored; exact-head Rust 1.96 format/Clippy/tests pending  
**Parent:** Symthaea #12582 — ISRU-BENCH-003  
**Related:** B1 twin #12574 / PR #12578; stage workflow #12549; Symtropy bitemporal calibration and portability work (#5682 / #5734); Mycelix common-cause cut sets (#5762)

## Paradigm shift

A laboratory with four instruments does not necessarily have four independent sources of evidence. Multiple sensors may share an optical lamp, wavelength standard, reference artefact, environmental chamber, batch-preparation history, common clock, software pipeline or authority/certificate chain.

Treating those measurements as independent can make a Bayesian optimizer overconfident, cause EIG/EVSI to overvalue repeat measurements, and incorrectly present repeated readings as independent corroboration. The system therefore needs a causal dependency graph and covariance accounting, not merely more sensors or more repeated experiments.

## Factor model

The initial reference uses an additive factor model:

[
y_i = f_i(x) + sum_k L_{ik} z_k(t) + epsilon_i
]

where:
- (y_i) is a recorded instrument result;
- (f_i(x)) is the instrument's measured response to the candidate;
- (z_k(t)) is a shared latent cause, such as a lamp/reference/environment factor;
- (L_{ik}) is the loading of cause (k) on measurement (i); and
- (epsilon_i) is instrument-specific independent noise.

For an observation, the model produces:
- a common-cause bias correction;
- the variance contribution from each shared factor;
- independent instrument-noise variance;
- total variance;
- the cause identities and qualification receipts supporting the factor contributions; and
- an eligibility result with explicit findings.

For two eligible observations, shared-cause covariance is:

[
operatorname{Cov}(y_i,y_j) =
sum_{k in C_i cap C_j} L_{ik} L_{jk} sigma_k^2
]

Instrument-specific random noise is treated as independent across records. Any instrument drift or same-channel common effect must be represented as its own explicit cause node rather than hidden in the independent-noise term.

## Bitemporal cause qualification

Each shared cause has a qualification history with:
- effective time (when the condition applies physically/evidentially);
- recorded time (when the system learned and recorded it);
- disposition (Qualified, Unknown, Revoked, Superseded or ScopeMismatch); and
- receipt identity.

An evaluation uses both the measurement's observed time and the assessment's as-of recorded time. This supports retrospective revocation without rewriting history. A measurement evaluated at an earlier as-of time can still show what the system knew then; a later as-of evaluation applies a retroactive revocation when its effective time covers that measurement.

A cause with no applicable qualification record is Unknown, not zero and not implicitly qualified. A measurement before the cause's reference epoch is not extrapolated backward. A measurement cannot be evaluated at an as-of time before its own recorded time.

Receipt IDs must be non-empty and unique in this reference, but are not authenticated signatures. Production must resolve them through Mycelix authority verification and Symtropy bitemporal calibration records.

## Instrument dependency inventory

Each instrument declares whether its dependency inventory is complete. An instrument with an incomplete inventory is ineligible for a claim that requires common-cause independence. This avoids turning "not listed" into "independent."

For a complete inventory, every relevant shared factor is represented through an instrument-to-cause loading. Two instruments can be treated as independent under this model only when the declared complete inventories contain no shared factor. The inventory itself still requires independent governance and audit; this code cannot prove that an omitted dependency does not exist.

## Effective number of independent measurements

The implementation provides a Kish-style redundancy diagnostic for equally weighted measurements:

[
n_{mathrm{eff}} = rac{n^2}{sum_{i,j} ho_{ij}}
]

where (ho_{ij}) is the pairwise correlation implied by shared factors and independent noise. This can show why ten readings sharing one unstable lamp may contribute substantially less information than ten genuinely independent measurements.

This scalar is only a diagnostic. Downstream Bayesian inference, EIG/EVSI and decision analysis should consume the full covariance structure; (n_{mathrm{eff}}) is not a replacement for the covariance matrix and should not be used as a standalone proof of independence.

## Current Rust implementation

`research/asteroid-optical-isru-v0.2/common_cause.rs` implements:
- typed shared-cause and instrument specifications;
- explicit cause loadings and a completeness flag for instrument dependency inventory;
- bitemporal qualification events, including later-recorded retroactive revocation;
- factorized measurement evaluation with bias correction and decomposed variance;
- shared-cause covariance and pairwise correlation;
- an effective-independent-count diagnostic; and
- negative controls for unknown/revoked causes, incomplete dependency inventories, non-monotonic recorded time, measurement/cause epoch inversion and evaluations before record time.

### Reproduce locally

    rustc --edition=2021 --test research/asteroid-optical-isru-v0.2/common_cause.rs -o /tmp/isru-common-cause-tests
    /tmp/isru-common-cause-tests

## Limits

This v0.1 model assumes additive factors whose latent terms are mutually independent. It does not yet estimate factor loadings, represent covariance between distinct shared factors, model nonlinear interactions, learn drift from live calibration series, handle interval/censored measurements, authenticate receipts or integrate directly with Symtropy's full provenance graph.

It is a reference contract for the structure of common-cause accounting, not a demonstrated uncertainty improvement. The next research tranche should estimate factor loading/covariance on held-out source/reference/batch groups and integrate the resulting covariance into Symthaea likelihood EIG/EVSI, with Symtropy retaining bitemporal receipt and revocation authority.

## Benchmark plan

Compare:
- independent-noise GP/BO;
- independent sensor ensemble with no factor graph;
- fixed covariance model;
- factor-aware hierarchical/linear model;
- Symthaea EIG/EVSI with shared-factor covariance; and
- stage-aware EIG/EVSI with evidence qualification and common-cause cuts.

Freeze candidate pools, source/reference/batch factors, observations, measurement budgets and holdout groups. Report interval coverage, covariance recovery, false-independent-evidence rate, EIG optimism due to duplicated measurements, common-cause detection/localization and decision regret.

Frozen adversarial worlds should include:
1. two spectrometers sharing a defective lamp;
2. an independent QCM plus a shared environmental chamber;
3. retroactively revoked calibration artefact;
4. missing cause identity and incomplete dependency inventory;
5. a cause affecting only one wavelength/temperature regime; and
6. shared clock or data-transformation error.

Initial target: zero false independent-corroboration claims on the frozen controls and nominal interval coverage within the preregistered tolerance on held-out cause groups. These are research hypotheses, not current results.