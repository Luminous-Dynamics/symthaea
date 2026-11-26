MON-018 authority-join semantic adapter

A standard-library Go reference implementation for a narrow MON-018 v0.1 profile. It checks exact principal/delegate/action/resource/destination/tenant/mode/epoch binding; root-to-leaf delegation continuity; exact non-amplifying scope; typed digest comparability; commit-time grant validity, expiry and revocation; bounded source capture; and normalized identity/issuer/grant verifier reports against pinned verifier IDs, versions, builds and trust-store digests. Any effective revocation timestamp also requires a distinct revocation event ID, digest and verifier report; a timestamp without that event evidence remains indeterminate.

The profile intentionally supports exact-match scopes only. Wildcards, numeric range delegation, cross-DNA resolution and multi-source joins are out of scope.

Cryptographic boundary: it does not verify credentials, signatures, COSE, AAC, OAR or SCITT proofs. It consumes normalized verifier reports and does not authenticate their transport. The verifier pin policy and MON-qualified commit/capture evidence must be supplied by separately controlled components.

Effect boundary: qualified means only that supplied authority evidence is semantically bound to the exact operation and provided commit time. It is not a dispatch command, destination readback, completeness proof outside the declared capture boundary, or physical-effect proof.

Preflight commands: go test -race -count=5 ./..., go vet ./..., go test -cover ./..., and go test -run=^$ -fuzz=FuzzDigestPurposeIsolation -fuzztime=2s.
