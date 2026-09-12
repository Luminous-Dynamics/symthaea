# symthaea-energy-evidence-envelope

Native candidate/campaign lineage for Tier-1 energy-material evidence receipts.

The existing evidence adapters can keep their domain-specific payload schemas. This crate wraps the exact JSON receipt text in a new outer envelope that binds the receipt to:

- candidate ID;
- immutable candidate-version SHA-256;
- frozen campaign-manifest SHA-256;
- exact campaign-lane SHA-256;
- evidence dimension;
- exact generic `Prediction` + prediction SHA-256;
- payload type;
- exact payload-text SHA-256.

## Exact prediction identity

The generic discovery prediction promoted into the dossier is stored directly in the envelope and receives its own domain-separated SHA-256.

Envelope validation revalidates the prediction and checks its metric/unit against the frozen screening contract for the bound dimension. Mutating the generic result while keeping the same adapter receipt therefore fails closed.

## Exact payload identity

The envelope preserves the exact UTF-8 JSON text supplied by the adapter/audit pipeline.

Whitespace and JSON key order are intentionally evidence-bearing. Reformatting a receipt therefore produces a different payload digest and envelope digest.

If semantic/canonical JSON normalization is useful, it should be emitted as a separately identified derived artifact rather than silently replacing the source representation.

## Campaign-lane identity

`campaign_lane_sha256(...)` binds the complete lane declaration, including:

- dimension;
- adapter name/version;
- expected model name/version;
- method parameters;
- source commitment;
- required evidence kinds.

Required-evidence-kind ordering is canonicalized for the lane hash.

Changing a hazard policy parameter, HHI aggregation mode, recovery-process parameter, source commitment, or similar lane field therefore changes the native lineage digest.

## What this improves

Older adapter receipts predate candidate/campaign digests and need #1963 compatibility review attestations.

A newly generated evidence envelope carries candidate + campaign + lane + generic-prediction lineage natively in the source receipt itself. Downstream dossier/campaign versions can therefore require the exact envelope prediction rather than accepting a separately supplied result.

## Important limitation

The envelope proves what candidate/campaign/lane/result the outer receipt claims to belong to and makes later mutation detectable.

It does **not** by itself prove that the inner adapter implementation actually obeyed every method parameter in the lane. That stronger theorem requires the adapter execution path to consume the frozen lane/config directly or emit separately qualified execution evidence.

## Network-free CLI

`energy-evidence-envelope <manifest.json> <dimension> <prediction.json> <payload-type> <receipt.json>`

The CLI reads the exact generic prediction and receipt JSON text, creates the native campaign binding and emits the envelope plus its SHA-256. Host-local file paths are not included in output.

## Authority boundary

A native envelope is provenance/integrity metadata. It is not scientific validation, experimental replication, safety certification, synthesis authorization, manufacturing approval, or deployment authority.
