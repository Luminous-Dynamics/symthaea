# Symthaea Primality and Factorization Certificates

This document specifies the version-1 canonical byte encodings implemented in
`crates/core/symthaea-core/src/hdc/number_theory.rs`.

The formats carry mathematical evidence; they are not signatures, attestations,
or proof that a particular producer was trustworthy. Authenticity and transport
security are separate concerns. The current verifier is ordinary Rust code and
has not itself been formally verified.

## Common encoding rules

- All integers are fixed-width and big-endian.
- The eight-byte magic is also the format version. Unknown magic/version bytes
  are rejected rather than upgraded or normalized.
- No trailing bytes are permitted.
- Counts and lengths are checked before they drive allocation.
- The decoder verifies the decoded object and requires canonical re-encoding to
  reproduce the exact input bytes.
- The maximum encoded size of either top-level format is 1,048,576 bytes
  (1 MiB). Oversized input is rejected.

## Primality proof: `SYMPPRF1`

The byte sequence is:

| Field | Encoding |
| --- | --- |
| Magic/version | 8 bytes: ASCII `SYMPPRF1` |
| Claimed prime | unsigned 64-bit integer |
| Certificate count | unsigned 32-bit integer |
| Certificate nodes | exactly the number declared below |

Each certificate node is encoded in this order:

| Field | Encoding |
| --- | --- |
| Candidate prime | unsigned 64-bit integer |
| Predecessor-factor count | unsigned 32-bit integer |
| Predecessor factors | repeated `(factor: u64, exponent: u32)` |
| Witness count | unsigned 32-bit integer |
| Witnesses | repeated `(factor: u64, base: u64)` |

Certificate nodes must be strictly increasing by candidate prime. The node for
2 is the base case and must have no predecessor factors and no witnesses. For
every other candidate prime (p), the predecessor factors must be strictly
increasing, each exponent must be positive, their exact product must be
(p-1), and every distinct prime factor must have a corresponding witness in
the same order. The root candidate must have a node in the table. Nodes not
reachable from the root are rejected.

The verifier recursively validates every factor prime and checks the Pocklington
conditions for each distinct prime divisor (q) of (p-1), with witness (a):

`a^(p-1) mod p = 1` and `gcd((a^((p-1)/q) mod p) - 1, p) = 1`.

The modular residue form is used to compute the GCD without constructing a
potentially huge exponentiation result. The verifier also checks the theorem's
size requirement `(p - 1)^2 > p` explicitly. This implementation supplies the
complete factorization of (p-1), so the factored part (F) is (p-1).

Limits: at most 4,096 certificate nodes and at most 65,536 combined predecessor
factor entries plus witness entries.

## Proof-carrying factorization: `SYMFCER1`

The byte sequence is:

| Field | Encoding |
| --- | --- |
| Magic/version | 8 bytes: ASCII `SYMFCER1` |
| Input value | unsigned 64-bit integer |
| Factor count | unsigned 32-bit integer |
| Factor entries | repeated `(prime: u64, exponent: u32)` |
| Proof count | unsigned 32-bit integer |
| Nested proofs | repeated `(byte_length: u32, proof_bytes: byte_length bytes)` |

The factor entries must be strictly increasing by prime with positive exponents.
Their checked integer product must equal the input value. The value 1 is encoded
with an empty factor list and empty proof list. The value 0 is rejected. Proof
count must equal factor count; proof entry (i) certifies the prime in factor
entry (i). Each nested proof is a complete `SYMPPRF1` encoding and must pass
its own strict decoder and proof verifier.

Limits: at most 4,096 factor entries, at most 4,096 nested proofs, and 1 MiB for
the complete outer encoding. Each nested proof must also respect its own
`SYMPPRF1` limits.

## Verification and limitations

A verified `SYMPPRF1` bundle establishes primality through recursive
Pocklington certificates and integer checks. A verified `SYMFCER1` bundle
additionally reconstructs the exact input from the sorted prime powers.

Certificate generation can fail to find a witness within its current bounded
search, or can fail to factor a predecessor within the factorizer's bounded
search before its slower completeness fallback. Such a failure must be reported;
it is not evidence that the candidate is composite. External verifiers may
accept any valid certificate that meets this specification, regardless of which
generator produced it.

The on-wire encoding is a transport contract, not a formalization of the Rust
implementation. A future version should be introduced under a new magic/version
rather than changing the meaning of version 1.


## Standalone verifier

A second implementation is provided in `tools/verify_math_certificate.py`. It targets
Python 3.10 or later and uses only the Python standard library. It checks the
serialized Pocklington evidence directly; it does not import the Rust crate, HDC
code, or the factorization search engine. It supports both formats:

```sh
python3 tools/verify_math_certificate.py path/to/certificate.bin
python3 -m unittest discover -s tests -p 'test_verify_math_certificate.py' -v
```

On success, the command emits a JSON object with `"verified": true`, the input
value, and the verified format. Invalid certificates return a non-zero exit code
and a JSON error on standard error. The independent verifier checks the same
version-1 resource ceilings and demands byte-for-byte equality with its own
canonical re-encoding. The same hand-authored 80-byte proof-for-3 golden vector is asserted in both
`number_theory.rs` and the Python test suite; malformed-input tests and a subprocess
test of the actual CLI entry point additionally check rejection behavior and invocation.
This cross-language vector avoids relying only on an encoder/decoder round trip within
either implementation.

The focused GitHub Actions workflow
`.github/workflows/math-certificate-verifier.yml` runs Python bytecode
compilation, the standard-library unit tests, and the CLI help check on
`ubuntu-24.04`; it does not need Rust dependencies, network access, or secrets.
