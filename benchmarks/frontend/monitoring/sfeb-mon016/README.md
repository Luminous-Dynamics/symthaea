# MON-016 bounded capture and negative evidence

**Status:** standard-library semantic reference and adversarial tests. It consumes normalized source/manifest/witness verification states. It does not verify WAL, SCITT, RFC 9162, COSE, timestamps or source signatures.

## The core rule

Absence is not inferred from missing logs. MON-016 qualifies a *bounded source enumeration claim* only for an exact:

- source, stream and partition;
- stable snapshot token;
- canonical query digest and its declared digest context;
- exclusive starting and inclusive ending result position;
- terminal cursor/page and source watermark;
- source control root and signed manifest digest.

The source attestation must bind the same boundary identity, source, stream, partition, snapshot and query digest that MON verifies. A source response that is nonterminal, has an insufficient watermark, lacks a verified query binding or has a sequence gap cannot be called complete.

`Position` is a contiguous **query-result ordinal** within this declared enumeration, not a raw WAL LSN, timestamp, Kafka offset or arbitrary application ID. Adapters for PostgreSQL logical decoding, outbox scans, telemetry streams and API audit logs must map their source-specific cursor semantics into a snapshot-stable ordinal or supply a more specific profile. Do not assume raw LSNs are contiguous.

## Outcomes

- `source_attested_complete_within_boundary`: the normalized source attestation binds the exact query/snapshot/range, confirms terminal enumeration through the declared watermark, and the listed result ordinals have no gaps or conflicts.
- `incomplete`: the source did not reach terminal state or its watermark does not cover the declared end.
- `indeterminate`: source, query, watermark, manifest, or required trust evidence is missing/unverified.
- `contradicted`: invalid source proof, boundary substitution, duplicate position with changed identity, conflicting event identity, out-of-range item, or a gap in a terminal source-attested contiguous enumeration.

A zero-item, complete enumeration yields `zero_events_observed_within_boundary`. A distinct witness that anchors the exact manifest under a distinct declared control root yields `zero_events_observed_in_independently_anchored_manifest`. The source-attested watermark and control-root identifiers must match their verified attested values exactly. This witness establishes that the manifest was anchored; it does **not** independently prove that a dishonest source did not omit events.

MON-016 deliberately has no `nothing_happened_globally` or `absence_established` result. A stronger claim requires independently administered enumeration or cross-source reconciliation, with the distinct source's own query, snapshot, watermark, coverage and control-root-evidence qualified too.

## Duplicate and gap semantics

An at-least-once redelivery at the exact same source position is permitted only when event ID and typed digest are identical. It is counted as duplicate delivery, not a second event. A different binding at the same position is contradicted. Reuse of one event ID at multiple source positions is contradicted in this reference profile.

A gap in an attested terminal ordinal stream is a contradiction to the claimed complete enumeration. It is not silently filtered out or downgraded to a smaller successful denominator.

## Local preflight

Go 1.23.2 / linux-amd64:

    go test -race -count=5 ./...
    go vet ./...
    go test -count=1 -cover ./...
    go test -run '^$' -bench . -benchmem

Fuzz targets:
- `FuzzGapNeverQualifiesComplete`
- `FuzzSharedRootNeverProducesWitnessedNegative`
- `FuzzQuerySubstitutionNeverQualifies`

## Limitations

The `Verification=verified` values are normalized outputs from maintained upstream verifiers; this package does not authenticate them. A distinct witness root is only as independent as its real operator, credentials, cloud, storage, build pipeline, and administrative path. The test model cannot establish production source honesty or data capture completeness outside the explicitly declared source/query boundary.
