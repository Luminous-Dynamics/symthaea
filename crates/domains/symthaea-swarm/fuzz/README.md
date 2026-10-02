# Symthaea Swarm COSE_Key fuzzing

This standalone cargo-fuzz package targets the untrusted CBOR parser
`Rfc9942Es256CoseKey::from_cbor` and intentionally performs no
pre-validation.

Recommended bounded local run:

```sh
cargo fuzz run fuzz_cose_key_es256 -- -max_len=4096 -timeout=2
```

The input cap is a fuzzing-run limit, not part of the COSE wire format.
The parser itself remains responsible for rejecting oversized map entries,
coordinate material, key-operation arrays, text/bstr values, and nested
unknown values.

A successful parse does not establish that the P-256 coordinates form a
cryptographically valid point. That remains the responsibility of the ES256
verification boundary.
