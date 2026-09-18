# DE-001A likelihood provenance correction

**Date:** 2026-09-18  
**Status:** frozen provenance correction.  
**Scientific authority:** none.

## Correction

Earlier DE-001A planning described the public Cobaya DR2 BAO lane as an "independent public likelihood" because the DESI published best-fit files identify a historical external likelihood package named:

`desi_y3_cosmo_bindings.cobaya_likelihoods.bao_likelihoods_v1p2.desi_bao_all`

while the executable Symthaea lane is being built around:

`bao.desi_dr2.desi_bao_all`.

An official DESI Data Q&A subsequently clarifies the relationship. A DESI collaborator states that the historical `desi_bao_all` likelihood is identical to the released Cobaya `bao.desi_dr2.desi_bao_all` likelihood and that the measurement and covariance files are byte-identical.

Source:

https://help.desi.lbl.gov/index.php?qa=182&qa_1=provenance-external-cobaya-likelihood-public-cosmology-chains

## Consequence

DE-001A should therefore be described as a **public released likelihood reproduction through a separately packaged execution path**, not as an independent likelihood implementation.

The distinction is important:

- separate packaging/environment can reveal dependency, configuration, solver, and execution drift;
- it does not provide independence from the mathematical likelihood implementation when DESI attests that the released implementation is identical;
- agreement therefore qualifies reproduction/plumbing, not implementation independence;
- disagreement remains useful because it localizes environment/configuration/theory-backend differences;
- true implementation independence remains reserved for the later independent-replication gate, DE-001I.

## Claim boundary

This correction does not change the scientific status of DE-001A. A PASS still licenses reproduction only and cannot establish Lambda-CDM, an anomaly, evolving dark energy, or a physical mechanism.

The correction is intentionally additive rather than rewriting the earlier frozen evidence subjects. Historical PR text may retain the earlier wording, but all subsequent DE-001 execution and interpretation should use this provenance record.
