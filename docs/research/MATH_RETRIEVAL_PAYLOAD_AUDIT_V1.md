# MATH-RET-PAYLOAD-001A — Canonical Retrieval Payload Byte Audit v1

Status: materialized-byte audit for MATH-RET-TRACE-001A packing evidence.

Authority: `MeasurementOnly`.

## Purpose

MATH-RET-TRACE-001A proves that a recorded retrieval execution obeyed the qualified graph and that its **reported** canonical payload byte counts are internally consistent with the frozen packer budget.

This tranche independently reads the materialized canonical payload files and proves that:

```text
actual payload bytes
    -> actual payload SHA-256
    -> actual byte length
    -> exact agreement with trace packing accounting
```

It closes the byte-measurement gap without inventing a stronger source-identity claim than the existing contracts define.

## Two distinct identities

Each audit entry binds both:

```text
source_object_sha256
payload_sha256
```

They are deliberately separate.

`source_object_sha256` is the identity selected by retrieval and recorded in the qualified trace.

`payload_sha256` is SHA-256 of the exact materialized canonical UTF-8 bytes supplied to the downstream context packer, or of the first ranked payload rejected for not fitting.

v1 does **not** assume:

```text
source_object_sha256 == payload_sha256
```

because the repository has not yet frozen a universal contract stating that the source-object identity is exactly the hash of downstream serialized payload bytes.

## Audit root identity

The audit binds exact:

```text
trace_sha256
graph_report_sha256
context_packer_sha256
payload_serialization_sha256
```

Before reading payload files, the validator reruns MATH-RET-TRACE-001A against the graph bundle. The audit must match the qualified trace, graph report, packer and payload-serialization identities.

## Exact target set

The payload audit must contain exactly the trace's materialized packing targets, in trace rank order:

1. every `Delivered` output item;
2. the `FirstNonFitting` item when the trace stopped for that reason.

No extra payload artifacts are accepted.

No packed item may be omitted.

This prevents an evidence bundle from auditing only convenient payloads while leaving another delivered item unaudited.

## Entry shape

Each entry freezes:

```text
role = Delivered | FirstNonFitting
rank
source_object_sha256
payload_path
payload_sha256
canonical_payload_bytes
```

The path is only a local resolver under a caller-selected `--payload-root`. Scientific payload identity is the SHA-256 of the exact file bytes.

## Safe materialization paths

The validator rejects:

- absolute paths;
- empty paths;
- `.` or `..` path components;
- NUL-containing paths;
- paths that resolve outside `--payload-root`;
- missing/non-file targets.

Thus a payload audit cannot escape its declared materialization root.

## Exact byte verification

For every entry, the validator:

1. reads exact file bytes;
2. requires valid UTF-8;
3. recomputes SHA-256;
4. compares against `payload_sha256`;
5. computes actual byte length;
6. compares against both the audit entry and the corresponding runtime trace byte count.

For delivered items, actual byte lengths are summed and must equal:

```text
trace.packing.output_bytes_used
```

The delivered item count must already agree with:

```text
trace.packing.output_items_used
```

## First-nonfitting reconstruction

If the trace stopped at `FirstNonFittingItem`, the payload audit must materialize that exact next-ranked source object too.

After measuring its actual bytes, the validator rechecks that the real payload violates at least one frozen packer condition:

```text
actual_item_bytes > max_output_item_bytes
```

or:

```text
delivered_actual_bytes + actual_item_bytes > max_output_bytes
```

This converts MATH-RET-TRACE-001A's reported non-fit into an independently reconstructed byte-level non-fit.

## Source-object mapping boundary

This audit proves:

```text
this materialized file has digest P and byte length N,
and the trace associated it with source identity S
```

It does **not yet independently prove**:

```text
applying source_fetch_policy + payload_serialization_policy
to source identity S necessarily produces payload P
```

That is a separate source-materialization correctness claim.

The packer already freezes exact:

```text
source_object_contract_sha256
source_fetch_policy_sha256
payload_serialization_sha256
```

A future MATH-RET-SOURCE-001 executable source-materializer audit can close the S -> P transformation if needed.

Keeping this boundary explicit avoids turning a path resolver or audit declaration into false provenance authority.

## Validation report

Successful validation emits:

```text
math-retrieval-payload-audit-report-v1
```

recording:

- audit digest;
- trace digest;
- graph-report digest;
- context-packer digest;
- payload-serialization digest;
- audited entry count;
- independently reconstructed delivered payload bytes;
- whether a first-nonfitting payload was audited;
- `all_checks_passed = true`;
- `authority = MeasurementOnly`.

Schema:

`.github/schemas/math-retrieval-payload-audit-report-v1.schema.json`

## Pure self-test

```bash
python3 .github/scripts/validate-math-retrieval-payload-audit.py --self-test
```

The exact source used for this tranche passed its pure materialized-byte self-test before commit.

The self-test:

- materializes actual temporary UTF-8 payload files;
- recomputes their SHA-256 and byte lengths;
- reconciles delivered bytes;
- verifies a real first-nonfitting item;
- rejects a false byte-count claim;
- rejects path traversal outside the payload root.

## Full qualification command

Once real runtime evidence exists:

```bash
python3 .github/scripts/validate-math-retrieval-payload-audit.py \
  path/to/payload-audit.json \
  path/to/query-trace.json \
  path/to/retrieval-graph-bundle.json \
  --repo-root . \
  --payload-root path/to/materialized-payloads \
  --report payload-audit-report.json
```

The full path reruns the entire trace and graph qualification before accepting payload bytes.

## Deliberate nonclaims

A valid MATH-RET-PAYLOAD-001A audit does not establish that:

- `source_object_sha256` is the hash of the payload bytes;
- the source-fetch policy was implemented correctly;
- the index selected the mathematically best sources;
- the retrieved payload is relevant or equivalent to the query;
- retrieval improves proof search;
- HDC, normalization or fusion helps;
- any theorem is true, proved or novel.

It establishes that the runtime trace's canonical payload byte accounting is backed by exact materialized bytes rather than self-reported numbers alone.

## Resulting evidence chain

After this tranche, the intended chain is:

```text
MATH-EXP v2.1
    |
MATH-RET index/fusion/binding/packer contracts
    |
MATH-RET-001D content-addressed graph qualification
    |
MATH-RET-TRACE-001A runtime ranking/fusion/packing replay
    |
MATH-RET-PAYLOAD-001A exact materialized-byte audit
    |
experiment outcome analysis
    |
formal theorem/result authority remains separate
```

## Next high-value implementation

The architecture is now sufficiently specified that the next best step is no longer another paper contract. It is a runtime emitter seam:

```text
retrieval execution
  -> emit MATH-RET-TRACE-001A atomically
  -> materialize canonical payload audit targets
  -> emit MATH-RET-PAYLOAD-001A manifest
```

That emitter should live below theorem/research authority and should fail closed if it cannot bind the exact qualified arm/index/fusion/packer identities.
