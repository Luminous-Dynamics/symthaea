# Symthaea Swarm COSE_Key fuzzing

This standalone cargo-fuzz package targets Rfc9942Es256CoseKey::from_cbor() with arbitrary raw CBOR input and no pre-validation.

Recommended bounded local run:

    cargo fuzz run fuzz_cose_key_es256 -- -max_len=4096 -timeout=2

The fuzz input cap is a run-level resource bound, not a change to the COSE wire format. A successful parse is not proof that the P-256 coordinates form a cryptographically valid point; that remains the responsibility of the ES256 verification boundary.
