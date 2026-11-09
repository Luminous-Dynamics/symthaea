# Sequential Financial Pilot v0 — Supplier Correction Stream

**Status:** public synthetic development fixture; deliberately non-qualifying  
**Parent:** [SYM-FIN-SEQ-001 #11290](https://github.com/Luminous-Dynamics/symthaea/issues/11290)  
**Consumer:** [Financial Research Agent #11233](https://github.com/Luminous-Dynamics/symthaea/pull/11233)  
**Temporal contract:** [#11241](https://github.com/Luminous-Dynamics/symthaea/issues/11241), [precision extension #11256](https://github.com/Luminous-Dynamics/symthaea/issues/11256), [schema/proof bridge #11264](https://github.com/Luminous-Dynamics/symthaea/issues/11264)

## Files

- `supplier-correction.stream.json`: ordered evidence versions and six decision frontiers.
- `supplier-correction.oracle.json`: expected admitted evidence and permitted conclusions at every frontier.

The fixture uses only fictional IDs, values and sources. It carries exact synthetic timestamps for development convenience. Its proof/rights declarations are explicitly synthetic placeholders, **not** independently authenticated availability or entitlement proofs. Do not use it to qualify historical data, rights or the production evidence schema.

## Event sequence

| Frontier | Decision time (UTC) | New information that becomes admissible |
|---|---|---|
| F0 | 1 Oct 2026, 12:00 | Original supplier disruption report |
| F1 | 2 Oct 2026, 09:00 | Correction: the reported suspension was misclassified; expected delay was 24 hours |
| F2 | 3 Oct 2026, 09:00 | Buyer-specific purchase share for the critical component |
| F3 | 3 Oct 2026, 11:00 | News mirror copied from the same authority source family |
| F4 | 4 Oct 2026, 09:00 | Buyer inventory coverage for the same component |
| F5 | 5 Oct 2026, 12:30 | Shipment receipt: actual arrival was 5 Oct 11:30, recorded as known at 12:15 |

The first five frontiers are frozen before the shipment outcome is known. F5 is an outcome-resolution frontier.

## Required behavior

1. Emit a complete, immutable research snapshot at each frontier.
2. F0 must not contain the F1 correction, buyer purchase share, inventory or delivery outcome.
3. At F1, use the corrected version for the then-current view while retaining the original version and its F0 result.
4. At F2, identify 35% purchase share only for the specified critical component. Do not reinterpret it as 35% of total procurement spend, revenue or all supplier purchases.
5. At F3, the mirror is another document but not another independent source family. Do not increase independent corroboration.
6. At F4, compare the 24-hour reported delay with 14 business days of reported component coverage. This supports a conditional buffer comparison; it does not prove zero production risk or financial impact.
7. At F5, admit the receipt only from 12:15 onwards. Earlier snapshots must remain byte-identical and must not gain the arrival outcome.
8. Forecast whether shipment S-001 arrives by 5 Oct 2026 12:00 UTC only if the candidate's probability was timestamped and retained before the outcome became known. Do not score a retrospectively generated forecast.

## Deliberate non-claims

The fixture does not establish company revenue impact, margin loss, supplier default, customer harm, causality, fraud, or investment action. It is a compact conformance test for sequential information updates, not a real-world supply-chain benchmark.
