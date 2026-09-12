# symthaea-energy-material-candidate-version

Binds a Tier-1 energy-material evidence dossier to one immutable domain-neutral `Candidate` version.

A candidate ID is not enough scientific identity. The same text ID can be reused after composition, phase, process, geometry, origin, or other specification fields change. This crate therefore content-addresses the complete `symthaea_discovery::Candidate`:

- candidate ID;
- candidate kind;
- sorted specification map;
- candidate origin/generator metadata.

The resulting candidate SHA-256 is domain-separated and deterministic.

## Compatibility review boundary

Existing Tier-1 adapter receipts predate candidate-version digests. V0 therefore requires one explicit `ReceiptVersionAttestation` per dossier contribution. Each attestation binds:

- evidence dimension;
- exact source-receipt SHA-256;
- exact candidate-version SHA-256;
- reviewer identifier;
- non-empty review note.

These are **review records, not signatures**. They make reuse of evidence across changed candidate specifications explicit and reviewable, but they do not cryptographically prove that an older source receipt was originally generated from that candidate version.

Future adapter schemas should carry the candidate-version digest natively. Once source receipts do that, the compatibility attestation can be replaced by direct receipt validation.

## One-to-one theorem

A valid bound dossier requires exactly one candidate-version attestation for every dossier contribution and no extras.

The attestation must bind the same:

- evidence dimension;
- source-receipt SHA-256;
- candidate-version SHA-256.

The underlying dossier is independently revalidated and its digest recomputed.

## Candidate validation

The anchor rejects:

- blank candidate IDs/kinds;
- blank specification keys/values;
- NUL-bearing specification fields;
- generated candidates with blank generator/version metadata;
- imported candidates with blank source metadata.

Because the specification uses `BTreeMap`, insertion order cannot change candidate identity.

## Network-free CLI

`energy-material-candidate-version <binding-input.json>`

Input JSON contains:

- `candidate`;
- assembled `dossier`;
- `receipt_version_attestations`.

Output contains the complete bound dossier and its SHA-256. The host-local input path is not emitted.

## Authority boundary

Candidate-version binding establishes content identity and review provenance only. It does not establish that an external mapping is scientifically correct, that source evidence is authentic, or that the candidate is safe, novel, synthesizable, manufacturable, certifiable, or deployable.
