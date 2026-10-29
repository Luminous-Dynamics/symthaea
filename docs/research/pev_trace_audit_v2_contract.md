# PEV trace auditor v2: reset epochs and canonical trace envelopes

**Status:** reference contract + synthetic verification only. This is not production Rust behavior, not a signed audit log, and not a task-utility benchmark.

## What changed

The v2 reference auditor wraps v1's semantic checks with a trace envelope that verifies:

- Stable trace-event IDs in the form run_id:e{epoch}:s{sequence}:{kind}.
- Strict, gap-free sequence numbers and one run_id per trace.
- Canonical JSON and a domain-separated SHA-256 digest over event metadata.
- A linked prev_digest hash chain.
- Explicit epoch_start and sequential reset records; event cycles may not regress within an epoch.
- No evidence reference across reset epochs and no PE-influenced consumer admission across a reset.
- Exact decision-event ID and original decision-epoch retention for posthoc outcomes.

An intentional nuance is tested as a positive control: a delayed outcome from an old decision may arrive after a reset when it preserves the original decision identity, epoch, and frozen PE/evidence lineage. That delayed calibration does not authorize a new PE-driven transition in the new epoch.

## Reproduce

From the repository root (the v1 auditor is a sibling dependency):

    python3 docs/research/pev_trace_audit_reference_v2.py --self-test --cases 100000 --seed 20261028 --output /tmp/pev_trace_audit_reference_v2_results.json

The checked run accepted 100,000 deterministic single-epoch synthetic cases (69,740 valid PE, 20,268 invalid/non-valid PE, 9,992 missing/no-reference), accepted the multi-epoch delayed-outcome positive control, and detected all six v2 mutations: digest tampering, event deletion/sequence gap, cross-epoch evidence reference, reset epoch jump, duplicate trace-event replay, and old-epoch PE admission after reset.

## Security and evidence limits

- A hash chain detects unsynchronized edits, deletion, reordering, and accidental corruption. It does **not** authenticate who produced a trace: an attacker able to rewrite the whole trace can recompute all hashes. Authenticity requires signing or anchoring the root digest in a trusted external store.
- Keep the canonical payload limited to metadata. Do not hash raw user prompts, private text, secrets, or high-cardinality content merely to establish provenance.
- The Python reference model does not import/execute production Symthaea Rust. It only establishes that this reference contract rejects the tested fixtures.
- No exact-head Rust compilation, formatter, property tests, checkpoint/restore test, or real emitted-trace audit is claimed. Keep the PR draft until these run on the exact head SHA.

## Next implementation gate

Port the minimum envelope to typed Rust structures and tests; include a checkpoint/restore fixture; emit raw test traces from real PredictionErrorEvidence decision/posthoc flows; run the independent auditor over those traces; sign or externally anchor a root digest only after schema and privacy review stabilize. Then test outcome quality separately with paired held-out scenarios, strongest matched baselines, and measured compute/memory cost.
