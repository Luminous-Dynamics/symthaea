# ADR-029: RSK Xenia witness trust context

**Status:** Proposed  
**Change Class**: A

## Context

RSK's external monotonic-anchor path may consume Xenia state-witness evidence.
Xenia can verify exact cryptographic keys, but a raw key quorum does not by itself
prove governed signer identity or independent administrative failure domains.

Accepting only a witness count would allow multiple keys controlled by one signer
or multiple signers controlled by one domain to masquerade as independent trust.

## Decision

RSK will treat the following as distinct evidence dimensions:

- verified cryptographic keys;
- governed signer identities;
- governed failure domains.

The RSK trust snapshot binds each exact Xenia `blake3-256` public-key fingerprint
to one key ID, signer ID, role, failure domain, signature profile, lifecycle, and
validity interval. The exact monotonic-anchor policy and trust-snapshot identity
are also bound.

RSK derives a domain-separated `TrustContextDigest` from that complete canonical
policy state. A Xenia state commitment is eligible for RSK external continuity
only when its `trust_context_digest` equals the independently reconstructed RSK
digest and the exact Xenia-verified keys satisfy the configured key, signer, and
failure-domain thresholds independently.

The trust-context verifier cannot grant replication authority or override any
negative safety state.

## Consequences

- multiple keys from one signer do not inflate signer quorum;
- multiple signers in one failure domain do not inflate failure-domain quorum;
- unknown, revoked, suspended, expired, aliased, or policy-incompatible keys fail
  closed;
- key identity remains cryptographic while signer/domain identity remains an
  independently governed policy fact;
- trust-policy changes cannot silently reuse old witnessed-state evidence under
  the same semantic context;
- Xenia and RSK retain separate responsibilities rather than sharing one broad
  authority object.

## Evidence status

The reference implementation, golden corpus, and adversarial tests are authored.
They are not production admission evidence until exact-head CI executes and the
real Rust/verified-trust boundaries are integrated.

Production admission remains **DENIED / NOT YET ELIGIBLE**.
