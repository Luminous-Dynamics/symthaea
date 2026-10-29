# CASTING-0033: Pre-outcome forecast event journal (v1)

**Status:** executable provenance gate proposal; not an external timestamp authority and not a model-calibration certificate.

## Motivation

`forecast_validation.py` binds each scored outcome to the exact canonical forecast and binds both manifests into a scoring receipt. That detects artifact changes but does not establish that the forecast was emitted before the outcome was known. A valid retrospective forecast can otherwise be hashed perfectly. This module adds an explicit temporal contract without pretending that a self-reported timestamp is independently verified.

## Inputs and event schema

The CLI consumes the same forecast and outcome JSON arrays as `forecast_validation.py`, plus an event-journal JSON array:

```sh
python3 tools/casting-benchmark/forecast_event_journal.py \
  forecasts.json outcomes.json events.json --output chronology-receipt.json
```

Events are ordered from sequence zero, use UTC timestamps ending in `Z`, and form one SHA-256 chain from a zero-filled genesis digest. Exact schema by `kind`:

- `forecast_published`: `sequence`, `event_id`, `kind`, `forecast_id`, `occurred_at`, `artifact_sha256`, `previous_event_sha256`, `event_sha256`.
- `outcome_observed`: those fields plus `forecast_event_sha256`, which links to the exact publication event.

`artifact_sha256` is the canonical forecast digest on publication and the already-validated `outcome_sha256` on observation. `event_sha256` hashes canonical JSON of every event field except itself, including the prior-event link. Unknown fields and duplicate JSON keys are rejected.

## Admission invariants

1. Forecast/outcome schema, identity uniqueness, exact forecast digest binding, and outcome digest binding must pass the existing forecast scorer's validator.
2. Every forecast has exactly one publication event and every scored outcome exactly one observation event. Orphans and missing events fail closed.
3. Sequence is contiguous from zero, event IDs are unique, all event times are valid UTC `Z` timestamps and do not decrease in journal order.
4. Every event's predecessor hash equals the exact preceding event hash; every event hash recomputes correctly.
5. An observation must follow its publication in both sequence and declared time (strictly later timestamp), link to that exact publication-event digest, and bind the exact observed outcome digest.
6. Receipt output must not overwrite or alias an input manifest. Inputs are read as strict UTF-8 JSON, including duplicate-key rejection.

The output binds hashes of raw input bytes, the journal head, the evaluator source digest, and a fixed label: `FORECAST_EVENT_JOURNAL_VALIDATION_NOT_TRUSTED_TIMESTAMPING`.

## Threat model and claim ceiling

This gate detects accidental or partial tampering, broken links, retrospective/ambiguous timestamp order, missing coverage and many accidental leakage paths. It does **not** prove that an event time is true. An actor who controls the entire unsigned journal can backdate events and recompute the chain. A chain is tamper-evident only when there is a previously anchored digest or a trusted external observer; it is not a signature, trusted timestamp, non-repudiation mechanism, or proof that a holdout was kept secret.

For stronger evidence, the next layer must anchor publication digests before outcomes resolve: e.g. an independent append-only witness or timestamp service, authenticated signer identity, signed ingestion receipts, and a second verifier that receives the anchor before unsealing labels. Keep witness timestamps and outcome-resolution times distinct. Do not treat repository commit timestamps or wall-clock values supplied by the policy as trusted capture. A standards-based option to evaluate is the [IETF RFC 3161 Time-Stamp Protocol](https://www.rfc-editor.org/rfc/rfc3161.html): verify the signed timestamp token, the exact data imprint, timestamping certificate/policy, and response timeliness. Even a valid external token only evidences that the bound digest existed by the trusted timestamp; it does not prove the forecast was calibrated or that held-out outcomes were otherwise independent.

## Tests and controls

The standard-library test module covers valid known-answer chronology, byte tampering, reorder/sequence gaps, strict UTC syntax, outcome-before-publication, equal timestamps, bad artifact digests, incorrect publication links, missing/duplicate events, duplicate JSON keys, receipt provenance and overwrite protections. The focused casting CI job executes this module alongside the forecast scorer and action-ledger tests.

## Metrics for the next benchmark round

Report the number of records rejected by each failure code, leakage-injection detection rate, false rejection rate on valid journals, audit latency per million events, peak memory, and reproducibility across independent verifier implementations. Include controls where an actor modifies a forecast after publication, moves an outcome earlier, omits a failed trial, swaps events, recomputes only a suffix, and rebuilds the entire unsigned journal with backdated times. The last case should **not** be claimed detected by this gate; it demonstrates why external anchoring is necessary.

This validator is a provenance primitive, not an optimizer improvement or evidence of better casting outcomes. Benchmark policies on matched-cost tasks, preregistered shift families, held-out physical observations and decision-level metrics separately.
