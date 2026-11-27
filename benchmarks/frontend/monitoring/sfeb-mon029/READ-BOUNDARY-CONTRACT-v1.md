# MON-029 — Verified Artifact Consumption Contract v1

**Profile:** mon029-verified-artifact-consumption-v1  
**Status:** read-boundary implementation and adversarial unit-test source added; exact-head local execution and independent review pending.

## 1. Problem

MON-026 verifies a signed manifest and the bytes found at its declared artifact paths. A downstream consumer must not assume that reopening a path yields the bytes it just verified. Between verification and consumption, a mutable workspace can change.

MON-028's scorer verified a custody bundle and then separately opened the suite, candidate-result and optional baseline JSON files without rechecking their signed size and digest. This left a verify-then-reopen gap at the point where evidence became a score.

## 2. Required invariant

**The exact byte sequence interpreted by a downstream consumer must be the byte sequence whose size and SHA-256 match the artifact entry in the exact manifest that passed independent custody verification.**

Verification and interpretation are separate operations, but consumption must be digest-bound to the verified manifest.

## 3. Read protocol

1. The caller independently verifies the complete MON-026 bundle with an out-of-band signer key, expected run ID, exact sequence, predecessor root, trusted clock and freshness policy.
2. The caller passes that same manifest and one exact relative artifact path to the MON-029 reader.
3. The reader requires exactly one matching manifest entry, validates the relative path, rejects symlinked roots/components and non-regular files, and bounds the structured artifact size.
4. The reader opens the file once, checks the opened file identity against the inspected regular file, verifies the declared size, reads a bounded byte buffer, checks post-read size, and hashes that buffer.
5. Only after the byte buffer matches the signed SHA-256 does the reader decode JSON from that same buffer.
6. Strict decoding rejects unknown fields and a second/trailing JSON document.

The scorer now applies this protocol to the benchmark suite, the candidate result and any selected baseline result. The reader package does not authenticate manifests; it explicitly requires a prior successful MON-026 verification.

## 4. Failure semantics

The read fails closed for:
- artifact not present exactly once in the signed inventory;
- invalid, absolute, traversing or symlinked paths;
- symlinked root or path components;
- non-regular files or file replacement between path inspection and open;
- size mismatch, concurrent size change or SHA-256 mismatch;
- structured JSON artifact larger than 256 MiB;
- invalid JSON, unknown fields or multiple top-level JSON values.

A failure prevents the score from being emitted. It is not converted into a skipped observation or a safe result.

## 5. Threat model and limits

The reader mitigates MON-028's path-based verify-then-reopen gap and binds parsing to the digest in the signed manifest. It does not authenticate the signer, prove that the artifact's underlying observation is true, or establish that two signer identities are administratively independent. Those remain MON-026/028 and operational trust responsibilities.

The selected JSON artifact is read into memory to guarantee that hashing and parsing consume identical bytes. The current limit is 256 MiB. Larger datasets should be represented as independently hashed raw artifacts plus bounded index/summary JSON, not by silently raising the limit.

Path checks and file identity checks reduce filesystem substitution risk. A hostile concurrent filesystem actor may still require a read-only snapshot, immutable mount or platform-specific descriptor-relative no-follow traversal. This package is not a general sandbox.

## 6. Adversarial known-answer tests

- matching signed-inventory bytes are parsed successfully;
- same-size replacement after prior custody verification fails digest matching;
- changed size fails;
- missing/duplicate inventory paths fail;
- traversal and symlink paths fail;
- symlinked root fails;
- oversized structured artifacts fail before reading;
- unknown fields and trailing JSON documents fail.

## 7. Qualification gates

Proposed acceptance criteria:
- 100% rejection of mutated, substituted, unlisted and malformed selected artifacts in the reader corpus;
- 100% acceptance of matching positive controls;
- zero score output when any selected artifact fails its manifest-bound read;
- deterministic output for fixed manifest and bytes;
- report the additional hashing/read cost separately from the initial whole-bundle custody verification.

These are test targets, not yet measured results.
