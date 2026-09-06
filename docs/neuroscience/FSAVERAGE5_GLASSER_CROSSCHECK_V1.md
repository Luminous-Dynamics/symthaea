# fsaverage5 → Glasser360 Independent-Lineage Cross-check v1

Status: **comparison mechanism implemented and synthetically qualified; no real independent-lineage comparison has qualified yet**

This document records the executable FMQ-010 comparison boundary for
`FsAverage5GlasserMapV1` artifacts.

The comparator answers a narrow question:

> Given two separately validated mapping artifacts, where do their vertex→parcel
> assignments agree and disagree?

It does **not** answer whether the two source lineages are genuinely independent.
Independence requires provenance review outside the comparison artifact.

## Inputs

Both lineages must first pass the `FsAverage5GlasserMapV1` validator against the
same canonical HCP-MMP1 area namespace. Invalid or digest-tampered mappings are
rejected before comparison.

The tool records source-metadata distinctness signals, but metadata differences
are not treated as proof of independence. The report permanently carries:

`independence_status = requires_external_provenance_review`

and:

`independence_established = false`

The report validator rejects attempts to upgrade that field to `true`.

## Comparison evidence

`FsAverage5GlasserCrosscheckV1` reports:

- exact same-parcel agreement
- both-unassigned agreement
- assigned-vs-unassigned disagreement
- assigned-parcel mismatch
- separate left/right hemisphere census
- all disagreement vertex indices and hemisphere-local indices
- the two parcel assignments at every disagreement
- all 360 parcel membership counts
- shared vertices per parcel
- lineage-A-only and lineage-B-only vertices per parcel
- per-parcel symmetric-difference counts
- immutable lineage artifact digests
- deterministic report content digest

The report is closed-world and internally revalidated. A recomputed hash does
not make a tampered census valid.

## Qualification modes

Normal comparison always emits the complete evidence report when both inputs are
valid.

`--require-exact` additionally returns non-zero when any vertex assignment
differs. This is intentionally strict and is useful before unexplained atlas
lineage differences have been reviewed.

`--reject-self-comparison` returns non-zero when both artifact content digests
are identical. Comparing an artifact to itself cannot satisfy an independent
cross-check requirement.

There is deliberately no built-in permissive percentage threshold such as
"95% agreement is good enough." A future scientific profile may define an
explicit, justified discrepancy-resolution policy, but the comparator itself
must not manufacture one.

## Qualification status

The implementation currently has 13 synthetic contract tests covering:

- exact mapping agreement from source-distinct artifacts
- explicit self-comparison detection
- parcel-to-parcel mismatch accounting
- assigned-vs-unassigned mismatch accounting
- hemisphere-specific disagreement census
- all-360-parcel reporting
- deterministic report bytes
- report digest tamper rejection
- input-map validation before comparison
- internally inconsistent census rejection even after re-hashing
- prohibition on claiming source independence
- CLI self-comparison failure mode
- CLI exact-agreement failure mode

These tests qualify the comparison mechanism only. They do not establish that
any two real atlas derivations are independent or scientifically correct.

## Relationship to FMQ-010

FMQ-010 requires the first real canonical mapping to be compared against at least
one independently sourced/generated HCP-MMP1→fsaverage lineage, with differences
reported by hemisphere, parcel, and vertex count.

This tool provides that deterministic disagreement evidence. Completing FMQ-010
still requires:

1. two genuinely independent, provenance-reviewed lineages;
2. validated `FsAverage5GlasserMapV1` artifacts from both;
3. this cross-check report;
4. scientific review of every nonzero disagreement class;
5. explicit resolution of unexplained differences;
6. retention of the report in the qualification evidence bundle.

## Non-authority invariant

Exact agreement between two mappings is evidence of reproducibility across the
represented lineages. It does not prove the atlas itself is correct, does not
establish empirical neural observation, and grants no substrate or consciousness
evidence authority.
