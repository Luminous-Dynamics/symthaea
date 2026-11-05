# Execution-route observability V0.1 fixture

This is a synthetic contract fixture. It contains no live route inventory, production execution trace, biological evidence, annotation, or real custodian attestation.

## Files

- `CONNECTOME_EXECUTION_ROUTE_OBSERVABILITY_V0.1.md`: research protocol and claim ceiling.
- `connectome-execution-route-observability-input-v0.1.schema.json`: strict versioned route/build/canary schema.
- `fixtures/connectome-execution-route-observability-missing-receipt-v0.1.json`: three synthetic routes and two canaries.

## Known-answer behavior

The synthetic inventory declares:
- one ACTIVE fusion API route with both expected observer receipts;
- one ACTIVE batch certificate-issuance route with only the execution-ledger receipt, omitting the claim/certificate-ledger receipt;
- one disabled legacy administrative route with synthetic exclusion evidence.

The first route is canary-covered. The second route is **not** covered because one expected observer receipt is absent. The disabled route is not an active route and requires its exclusion evidence to be checked by the eventual verifier.

Expected overall result: `OBSERVATION_MISSING` (or a stricter `NOT_ESTIMABLE` if the discovery receipt / disabled-route proof is not independently qualified). It must not receive a global or query-specific coverage pass.

The synthetic fault-injection record shows a dropped observer was detected. This does not compensate for the missing positive canary receipt on the batch certificate route.

## Identity and limitations

All digests, route names, custodians, event IDs, namespaces and source identities are synthetic placeholders. A 64-character value in this fixture is not a real signature, source attestation or production build digest.

A future verifier must:
- recompute canonical digests rather than trust supplied summary fields;
- reject duplicate route/observer/canary identities and unknown properties;
- bind every route and canary to the exact build artifact;
- require exact route-specific handler/trace identity;
- compare receipt fields to the canary's expected nonce, evidence, policy and verifier;
- enforce all expected observers and fault-injection outcomes;
- reject unresolved dynamic targets and incomplete discovery;
- reject any canary that can write annotations or export to training/evaluation corpora;
- invalidate all affected route receipts after build, route, instrumentation, observer or retention changes.

This starter artifact is protocol/schema/fixture only. It does not claim that a live source inventory or canary has been executed.
