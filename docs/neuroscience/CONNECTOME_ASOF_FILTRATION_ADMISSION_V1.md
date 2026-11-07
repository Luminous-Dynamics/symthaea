# As-of evidence and filtration admission V1

**Tracking:** #10982 — SCI-CONNECTOMICS-102  
**Composes:** #8732, #9848, #9862, #10897 / #10898, #10919 / #10922, #10948 / #10959, #10960 / #10961.  
**Guarantee:** operational chronology, prefix and proof-receipt admission audit only.  
**Status:** synthetic known-answer replay; no arbitrary e-process theorem is proved by this crate.

## 1. Event time is not filtration time

The filtration is the information actually available to the policy. It is ordered by the driver-owned global availability sequence.

For every evidence artifact retain biological event, acquisition, annotation/processing, QC/adjudication and availability times separately. A late tracking correction is a new immutable event, never an edit to the old record. Sorting on biological event time can create lookahead when a result is processed or adjudicated after the adaptive action.

## 2. Prefix contract

A decision at global sequence t supplies:
- the exact ordered evidence IDs delivered before t;
- the subset it actually consumed;
- immutable policy, action, history and driver receipt commitments.

The verifier checks that the available list exactly equals the ledger prefix, every consumed event belongs to it, and every consumed state is eligible. A decision must not consume its own outcome or any later-arriving observation. Confirmation decisions require a prior family freeze; discovery decisions cannot consume confirmation-split evidence.

## 3. Filtration admission contract

A local/coarser process is not automatically valid in the global filtration. V1 supports two positive *receipt modes*:
- direct-global supermartingale proof receipt, bound to the exact process, null, assumptions and global filtration;
- adjuster-lifted proof receipt, binding the exact source process/filtration, target global filtration, adjuster parameters, theorem assumptions and lifted output path.

Local-only and incomparable/unknown inputs are explicitly NOT_COMBINABLE. A digest is not a proof, and event-set inclusion alone does not establish the conditional expectation inequality. The tool validates receipt structure/identity; independent statistical review must inspect the referenced proof and exact Choe–Ramdas-compatible adjuster before any lifted stream is used by #10948/#10960.

## 4. Exact synthetic result

T7 has biological event time 90 but is not processed/QC-approved and delivered until global sequence 3. D1 at sequence 2 can use S1 only; D2 at sequence 4 can use S1 and T7. T8 is retained as a typed FAILED event.

P-S has a direct-global receipt and is structurally admitted. P-T-local remains NOT_COMBINABLE. P-T-lifted is structurally admitted only because its proof receipt and adjuster receipt bind the source/local filtration, target/global filtration and exact output process. The verifier does not claim to have proved that adjuster theorem.

## 5. Claim ceiling

A passing contract establishes exact synthetic as-of replay, prefix membership, lockbox timing and filtration-admission receipt semantics. It does not authenticate artifacts, prove the upstream e-processes, implement an adjuster, or authorize biological action.
