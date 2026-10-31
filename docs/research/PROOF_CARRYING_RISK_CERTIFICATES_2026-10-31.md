# Proof-carrying risk certificate v1

**Date:** 2026-10-31  
**Status:** 12/12 local tests passed on an executable copy matching the committed validation logic. The example certificate also passed independent JSON Schema structural validation and Python semantic validation. No Rust integration or remote CI pass is claimed.

## Paradigm

Every risk-controlled decision should carry a machine-verifiable contract, not just a number labelled "confidence" or "risk controlled". The certificate binds:
- the risk type, metric and target level;
- the exact statistical assumptions and scope;
- the method and version;
- digests of the decision table, loss, state space and relevant data summaries;
- a receipt describing which verifier ran and what it checked.

The certificate intentionally stores digests rather than raw calibration labels, features or prediction records. A verifier can match its evidence bundle against the digests without publishing private data.

## Keep guarantees distinct

Version 1 uses distinct risk kinds and proof lanes:

| Risk kind | Valid proof lane | What it means |
|---|---|---|
| expected risk | exact finite exchangeability | Exact augmented-multiset constraints for CRC-style expected risk |
| high-probability risk | high-probability bound | Confidence-level guarantee; bound and sampling assumption required |
| FDR | analytical FDR theorem or exact intersection closure | Fixed-family guarantee; arbitrary dependence scope explicitly recorded |
| SupFDR | theorem-backed e-process | Online time-uniform guarantee; common global filtration, predictable arrivals and optional stopping required |
| FWER | exact intersection closure or high-probability bound | Family-wise guarantee, never silently translated into FDR |

An offline FDR certificate cannot be relabelled as SupFDR. A simulated result cannot produce a passing certificate. For online SupFDR, v1 requires the explicit one-global-filtration and optional-stopping contract.

## Exact CRC certificate example

The included JSON binds the 500-round common-depth envelope CRC controller at alpha=1/10 and n=50. It has:
- 1,326 controller entries and 1,378 augmented count vectors;
- controller-table SHA-256 b987c413fb029fb06348605945498a44a95b84004db5ad35fc1f74c9263b8ecc;
- canonical manifest SHA-256 33116ade91b9c26e87db744774b0146d72b997151c33a265c07078ee676a37c5;
- 1,009 full-table verification passes / 1,390,402 exact inequalities;
- worst augmented state (0,0,51), total loss 51/10, budget 51/10, excess 0/1.

This is an exact finite-exchangeability CRC receipt for the common-depth family. It is not a time-uniform guarantee and says nothing about arbitrary distribution shift.

## Validator checks

The standard-library Python prototype:
- canonicalizes the manifest and hashes it;
- binds the receipt to the exact manifest digest;
- requires receipt evidence digests to equal manifest evidence;
- rejects raw calibration payloads;
- parses risk levels and proof receipts as exact rationals;
- reconciles the CRC budget to (n+1)*alpha;
- reconciles the total inequality count to table passes times augmented-state count;
- rejects positive budget excess;
- rejects offline-FDR-to-SupFDR relabelling;
- requires global filtration and optional stopping for a passing SupFDR receipt;
- refuses simulation-only passes while preserving simulation-only outputs as explicitly inconclusive research records.

The JSON Schema checks the portable envelope structure; guarantee-specific conditionals are enforced by the Python validator. The validator checks manifest structure and proof-receipt consistency. It does not prove external assumptions such as marginal e-value validity, the validity of a claimed theorem application, or that an input digest came from an independent trusted source. Those require independent provenance and assurance.

## Local tests

11/11 local tests passed, including:
- valid exact CRC certificate round-trip;
- manifest tampering;
- evidence digest mismatch;
- positive CRC budget excess;
- receipt-budget mismatch;
- exact check-count mismatch;
- offline FDR/SupFDR relabelling;
- missing global filtration;
- simulation-only pass rejection and inconclusive-record acceptance;
- forbidden raw calibration payload.

## Industry comparison and next engineering gates

MAPIE provides broad operational controllers including ConditionalExpectedRiskController, binary/multilabel CRC/RCPS/LTT and multiclass LTT. Its adaptive controller learns an input-dependent parameter; its documentation warns that clipping the parameter range does not strictly retain the theoretical guarantee. Symthaea's opportunity is a narrower but more auditable contract: exact finite-state verification, digest-bound evidence and distinct expected-risk / high-probability / FDR / SupFDR proof lanes.

Next gates:
1. Port the certificate data types to the active pure-std symthaea-statistics crate, without creating a duplicate statistics crate.
2. Implement a Rust canonical manifest encoder/digest check.
3. Differential-test Rust vs Python on this CRC certificate and all negative controls.
4. Build theorem registries and proof references for offline FDR and online SupFDR separately.
5. Add Mycelix provenance receipts only after source identity, issuer authority and revocation rules are explicit.

References:
- Bai & Ramdas (2026), CRC and randomized e-BH: https://arxiv.org/abs/2610.35821
- MAPIE risk-control API: https://mapie.readthedocs.io/en/stable/api/risk-control/
