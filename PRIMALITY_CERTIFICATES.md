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
factor entries plus witness entries. The Rust in-memory verifier applies these same
limits before constructing its lookup table or evaluating witness conditions, so a
caller cannot bypass the wire decoder's resource ceilings by constructing a proof
object directly.

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
`SYMPPRF1` limits. The Rust in-memory factorization verifiers enforce the same
factor/proof count ceilings before reconstruction or nested proof verification.

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
canonical re-encoding. Two hand-authored cross-language golden vectors are asserted in both
`number_theory.rs` and the Python test suite: the 80-byte `SYMPPRF1` proof-for-3
vector and a 76-byte `SYMFCER1` certificate for the factorization of 2, including
its length-prefixed nested base-case proof. Malformed-input tests and a subprocess
test of the actual CLI entry point additionally check rejection behavior and invocation.
These fixed vectors avoid relying only on an encoder/decoder round trip within
either implementation.

The Python verifier suite also differentially checks candidate claims for every
integer from 2 through 2,000 against a separate trial-division primality oracle.
For every prime in that interval it verifies and canonical-round-trips the
recursive Pocklington proof; for every composite it requires rejection. The
Carmichael number 561 receives a dedicated regression: base 2 satisfies
`2^560 mod 561 = 1`, so a Fermat-only test would accept it, but both the
Python and Rust certificate verifiers must reject the proof-shaped claim at the
Pocklington GCD condition. These are finite regression checks, not a formal
proof of either implementation's source code.

The focused GitHub Actions workflow
`.github/workflows/math-certificate-verifier.yml` runs Python bytecode
compilation, the standard-library unit tests, and the CLI help check on
`ubuntu-24.04`; it does not need Rust dependencies, network access, or secrets.


## Lean kernel qualification lane

`tools/generate_lean_math_certificate.py` provides a downstream proof bridge for either
wire format. It first decodes and verifies the exact input bytes with the independent
Python verifier. For a Pocklington node, it combines the per-prime-factor witnesses
into a common candidate root by isolating the prime-power components, then checks the
resulting modular-power and GCD conditions before emitting Lean source. This does
not trust the root-combination algorithm: the emitted claim is checked using the
pinned `PrimeCert` Pocklington tactic, and the resulting theorem is checked by Lean's
kernel.

`formal/math-certificate/` pins Lean 4.33.1, the `PrimeCert` source revision, and
its resolved Mathlib dependency graph. The workspace directly pins Mathlib tag
`v4.33.1` and its resolved commit so the kernel/runtime patch level is explicit
rather than inherited from the older `PrimeCert` snapshot. Lean's [official
4.33.1 release notes](https://lean-lang.org/doc/reference/latest/releases/v4.33.1/)
recommend upgrading from 4.33.0 or earlier and document kernel soundness fixes;
this pin is part of the proof-trust boundary, not a claim that Symthaea's Rust
or Python code is formally verified. The new focused workflow builds the project,
runs an axiom audit, generates Lean source from both cross-language golden vectors and a separate
152-byte multi-factor fixture for 13, and asks Lean to check all three generated
claims. The extra fixture exercises combining per-factor witnesses for
`12 = 2^2 * 3`; it is a proof-bridge regression, not a third Rust/Python
protocol golden vector. The source project also includes kernel-checked
claims for primality of 3 and 2 and exact reconstruction of the factorization of 2.
The Lean source now also defines
`pocklington_certify_independent_witnesses`: a generic theorem that derives
`Nat.Prime N` from the Pocklington size/divisibility conditions and a separate
verified witness for each prime divisor. A concrete theorem for 13 exercises
the two distinct factor obligations for (12 = 2^2 \cdot 3) without using the
Python common-root conversion. This theorem is currently an auxiliary formal
result; the generator still emits proofs through `PrimeCert`'s common-root
tactic path. Wiring generated records into the new theorem and formally decoding
the bytes remain separate qualification steps.


Each generated Lean module embeds the exact input bytes and the parsed
wire structure. Lean proves that its separate canonical encoder reproduces those
bytes exactly, checks field widths, resource ceilings and factor/proof alignment,
and only then presents the primality and reconstruction theorems. This binds each
generated mathematical claim to its byte-level input more strongly than emitting
a bare `Nat.Prime p` theorem.

**Formal boundary:** the generic Rust/Python decoders and the generator program
are not themselves formally verified for all inputs, and the Lean reference
encoder has not yet been proved equivalent to the prose specification as a
general theorem. The current workflow supplies a kernel-checked per-artifact
encoding equality plus kernel-checked mathematics; a fully formal decoder and
a theorem connecting the entire byte-language specification to sound verification
remain future work. This still does not make the Rust verifier formally verified.
The formal-generation utility deliberately caps one input at 65,536 bytes to
keep the generated numeral lists and reduction proofs bounded; the protocol
verifier itself retains its larger 1 MiB input ceiling.
