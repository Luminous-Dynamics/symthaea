# LQCD-021J — cross-receipt campaign evidence closure oracle

Independent standard-library execution for the cross-receipt composition semantics required by #2748 / LQCD-021J.

Exact executed subject SHA-256:

`393fcee4252ae963243048a1ec4e6a571fbb6654b7d4b6bc90aaf1b0f48e523b`

Canonical result SHA-256:

`8da73d27b2b63df96fabcf59788b4388c98c7aa554157b7aa87efd53ee8d3070`

Valid synthetic evidence-closure SHA-256:

`7fd8bde265bab11a677b38f993b38533296ea21edd3e7eedf7098c1d39c38836`

Valid synthetic comparison-closure SHA-256:

`81a08df12233352fd85c142da8c875618e9f87d1a9fe7e6dcd665c09eccffc76`

## Governing theorem

Individually valid receipts are not sufficient. Before a beta=6.0 result can become benchmark-comparable, every production/evidence receipt must bind the same frozen final-campaign subject and satisfy the authorized chronology and qualification state.

The synthetic valid fixture requires all of the following:

- one exact final-campaign subject and predecessor;
- `RustQualified` numerical-profile, RNG/restart, checkpoint-authority, and ConfigId/measurement-provenance implementations;
- exactly 4,000 retained configurations;
- an `Admissible` final-sample receipt bound to that exact configuration-set commitment;
- exactly 192 semantic measurement keys per configuration, hence exactly 768,000 final measurement keys;
- a qualified measurement-provenance receipt;
- an operator-robustness receipt;
- the historical-fidelity ledger and its exact-volume/declaration-limited claim ceiling;
- a target-isolated sealed analysis bound to the exact admitted configuration and complete measurement sets;
- no benchmark digest or target value in the sealed-analysis authority surface;
- a systematic ledger bound to the same sealed result, operator receipt, and historical ledger;
- only then may the external benchmark enter the comparison closure.

## Adversarial composition checks

The oracle rejects:

1. a checkpoint receipt from a different campaign (`CampaignSplice`);
2. a source-reviewed-only checkpoint implementation (`ImplementationNotQualified`);
3. a non-admissible final sample;
4. 3,999 retained configurations;
5. 767,999 measurement keys;
6. benchmark leakage into sealed analysis;
7. a stale/wrong final-campaign predecessor;
8. a missing historical-fidelity ledger;
9. a missing operator-robustness receipt;
10. a systematic ledger bound to the wrong sealed analysis.

A benchmark mutation changes the comparison-closure digest while leaving the upstream evidence-closure digest unchanged.

Operator sensitivity or an inconclusive operator bridge does not disappear. The evidence graph may remain structurally complete, but its claim ceiling is mechanically reduced to `InconclusiveOnly`.

## Current real-program status

The oracle deliberately records:

`real_beta6_comparison_capable = false`

The synthetic closure semantics are executable, but the real program still lacks production-qualified checkpoint authority, real final-sample admission, a complete real measurement set, and a real sealed analysis. In particular, #2952 remains a source-reviewed Rust candidate until actual execution evidence exists.

## Scientific boundary

This subject qualifies cross-receipt composition and claim-ceiling semantics only. It does not authorize the real beta=6.0 campaign, establish equilibration, qualify queued Rust implementations, or establish agreement with EHK.
