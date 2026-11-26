# MON-016 bounded capture contract v1

## Claim vocabulary

MON has to distinguish these statements:

1. **Observed in a report:** bytes/events appeared in a retrieved manifest.
2. **Source-attested complete within a boundary:** a verified source statement binds a specific query, snapshot and cursor interval; it attests terminal completion through a high watermark; the manifest's query-result ordinals are gap-free.
3. **Manifest independently anchored:** an additional witness in a declared different control root anchors the exact manifest digest and boundary.
4. **Cross-source corroborated:** a second separately administered source independently enumerates a semantically equivalent boundary and its results reconcile under an explicit mapping.
5. **Global absence:** not established by any of the above; MON does not assert this from finite logs, manifests or transparency receipts.

The witness in this reference implementation provides level 3 only. It does not elevate level 2 into proof of source honesty, and it does not establish level 4 or 5.

## Required boundary fields

- boundary ID, source ID, stream ID, partition ID;
- stable snapshot token or source-defined equivalent;
- query digest including filter, scope, ordering and projection semantics;
- digest algorithm, profile/version, payload type, canonicalization, encoding, purpose and subject;
- exclusive start and inclusive end positions;
- explicit position semantics (the reference profile uses a contiguous result ordinal);
- source-attested terminal condition and source watermark/high-water mark;
- source control-root identity;
- manifest digest that covers the boundary and enumerated items.

The source attestation must bind the exact same IDs, snapshot token and query digest as the submitted manifest. A proof from one query or snapshot cannot be replayed against another boundary.

## Completeness algorithm

1. Verify the manifest, source attestation, query-binding proof and watermark proof with the maintained profile verifier.
2. Require an exact match of source/stream/partition/snapshot/query binding.
3. Require terminal enumeration and watermark >= declared end.
4. Validate every event identity and digest.
5. Sort by query-result ordinal; exact duplicate deliveries at the same ordinal are counted and collapsed only if event ID and typed digest match.
6. Contradict conflicting duplicates, re-used event IDs at different positions, out-of-range positions, or missing ordinals in a source-attested terminal contiguous enumeration.
7. Mark source-attested completeness only for that exact boundary.
8. If no events are returned, report zero observed within that boundary, not global absence.
9. To report an independently anchored manifest, verify a witness over the exact manifest digest and boundary with an appropriately distinct control root.
10. To make a stronger absence/capture completeness claim, qualify a separate source enumeration and reconcile it; document common dependencies and coverage differences.

## Failure semantics

- Missing proof => indeterminate, never complete.
- Nonterminal cursor => incomplete.
- Watermark-below-end => incomplete.
- Query or snapshot substitution => contradicted.
- Gap despite verified terminal contiguous source attestation => contradicted.
- Changed duplicate at same ordinal => contradicted.
- Identical duplicate delivery => permitted and counted.
- Same control root for source and witness => not independently anchored.
- Witness not bound to exact boundary/manifest => mismatch.
- Source-attested empty manifest without an external witness => zero observed within boundary only.
- Independently anchored empty manifest => zero observed in independently anchored manifest, not proof of no omitted events.
