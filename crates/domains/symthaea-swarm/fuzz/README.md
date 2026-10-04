# Symthaea Swarm COSE_Key fuzzing

This standalone cargo-fuzz package targets Rfc9942Es256CoseKey::from_cbor() with arbitrary raw CBOR input and no pre-validation.

Recommended bounded local run:

    cargo fuzz run fuzz_cose_key_es256 -- -max_len=4096 -timeout=2

The fuzz input cap is a run-level resource bound, not a change to the COSE wire format. A successful parse is not proof that the P-256 coordinates form a cryptographically valid point; that remains the responsibility of the ES256 verification boundary.


## Targets

- `fuzz_cose_key_es256` — COSE_Key structural decoding.
- `fuzz_rfc9942_receipt` — tagged RFC 9942 Receipt envelope decoding, including nested VDP/proof parsing.
- `fuzz_rfc9942_signature_with_receipts` — outer Signature_With_Receipt COSE_Sign1 decoding, including protected/unprotected header handling and nested receipt collections.
- `fuzz_rfc9942_proofs` — direct VDP plus RFC 9162 inclusion/consistency proof-content decoding.

The qualification workflow compile-checks all fuzz targets so newly added parser entry points cannot remain feature-dead or silently uncompilable.


## Qualification resource limits

The parser keeps RFC wire compatibility separate from implementation resource
limits. The current RFC 9942 qualification envelope is:

- up to 16 receipts per `receipts` collection;
- 4 MiB per encoded Receipt value;
- 32 MiB total encoded Receipt bytes, with a 33 MiB encoded collection scan budget;
- up to 256 VDP proof bstr members, with an 8 KiB per-proof limit and a 4 MiB encoded VDP scan budget;
- 8 MiB for an outer application payload;
- 4096 chunks for an indefinite-length bstr or tstr;
- recursion depth 16 for generic opaque-value scanning.

These are defensive decoder bounds, not changes to the RFC 9942 data model. The
dedicated qualification tests exercise the reachable boundaries and reject
oversized or over-deep hostile input before the corresponding allocation or
recursive scan can proceed beyond the configured ceiling.
