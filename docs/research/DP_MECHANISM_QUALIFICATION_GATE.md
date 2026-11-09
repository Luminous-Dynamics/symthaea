# Differential-Privacy Mechanism Qualification Gate for Energy Telemetry

**Status:** Research decision record. No noise mechanism is currently approved for production use in this telemetry stack. The durable privacy ledger accounts for declared releases, but does not make arbitrary bytes differentially private.

## Decision

Do not implement an ad-hoc noise sampler in the telemetry code and do not set a production “DP verified” bit from the ledger alone. A candidate mechanism must be selected from a maintained library, pinned to an exact release, and qualified with the exact input domain, privacy unit, adjacency definition, contribution bound, sensitivity proof, privacy parameters and output type.

The current ledger's discrete_laplace and gaussian strings are **policy declarations**, not evidence that those mechanisms ran. A reservation and immutable output digest are useful accounting artifacts, not a mathematical proof or proof that a caller used the claimed sampler.

A signed-attestation verification path is now implemented in `tools/energy_telemetry/qualified_output.py`. It validates canonical JSON, rejects duplicate keys, verifies Ed25519 using a public key pinned by deployment configuration, binds the receipt to the exact durable reservation and output digest, and checks the approved library/version, mechanism identifier, feature flags, privacy unit, adjacency, contribution-policy ID, privacy-policy ID, contribution ceiling and qualification reference. The output, canonical attestation bytes and signature are retained atomically in SQLite. On restart, `QualifiedOutputVerifier.load_committed_output` rechecks the stored-attestation digest and repeats signature and reservation-binding verification before returning the output. The low-level storage operations are marked internal; ordinary release code should use `verify_and_commit` and `load_committed_output`. This follows the maintained cryptography library's documented Ed25519 sign/verify interface: https://cryptography.io/en/50.0.2/hazmat/primitives/asymmetric/ed25519/

**This is a trust-boundary improvement, not a DP mechanism qualification.** The signature proves only that the corresponding signing key approved the receipt bytes. It does not prove the signer witnessed actual execution, that the clipper ran, or that the mathematical privacy map is sound. The tests use a fixed test-only signing key. No production key, qualified noise engine, trusted clipper, attester service or production release service is configured. Python's underscore-prefixed storage method is not a sandbox; the surrounding service still has to expose only the verifier-backed publication route.

## Dependency research

Research checked on 2026-10-13:

- **OpenDP 0.15.1** is the current stable version surfaced by the official PyPI project page. OpenDP describes a Rust core, Python and R bindings, a conceptual framework for privacy-aware computation, a process for independent review, and independent security audits. This makes it the first candidate for a small, audited mechanism adapter: https://pypi.org/project/opendp/ and https://docs.opendp.org/en/stable/
- **Important qualification caveat:** the current OpenDP Python API marks relevant low-level make_laplace and make_sum constructors as requiring its contrib feature. OpenDP explains that constructors not yet completing proof-writing and vetting are available only after explicitly enabling contrib. We must not silently turn that feature on and describe the result as production-qualified. Review the exact constructor/release and proof status first: https://docs.opendp.org/en/stable/api/python/opendp.measurements.html and https://docs.opendp.org/en/stable/api/python/opendp.transformations.html
- **IBM diffprivlib** describes its public release as intended for research and education and asks production users to contact the project. It is therefore useful for exploratory utility studies, not selected here as the production release gate: https://github.com/IBM/differential-privacy-library
- **Google Differential Privacy building blocks** documents secure noise generation for its Laplace and Gaussian mechanisms, but its published building-block table covers C++, Go and Java—not a ready-to-drop-in Python API for this current Python prototype. A Rust/C++ integration would require a separate portability, build and review decision: https://github.com/google/differential-privacy

This is a candidate assessment, not an assertion that any package is categorically secure or automatically fits this dataset.

## Exact release contract

For a single aggregate sum of energy, the proposed input contract is:

1. **Privacy unit:** one household or site—not one meter reading, packet, or time interval.
2. **Pre-processing:** use authorized local aggregation and bound the total contribution of each privacy unit per release. If one household can contribute many intervals, sum/clip the household's full contribution before summing across households. A per-row bound alone is insufficient.
3. **Bound:** for a one-sided energy sum, each unit's contribution must be clamped into a declared public interval such as [0, L Wh]. The service must enforce the bound; the ledger's contribution_limit_wh field is only a claim until a trusted implementation applies it.
4. **Adjacency:** declare whether one household's full contribution is added/removed or replaced. Derive sensitivity for this exact adjacency and aggregation—not from the cohort size alone.
5. **Mechanism:** instantiate a reviewed integer-valued discrete mechanism for integer Wh outputs, or a reviewed approximate-DP mechanism with a positive delta and an audited privacy map. Do not hand-code a sampler or infer Gaussian scale from a rule of thumb when the selected library can compute a qualified privacy map.
6. **Composition:** reserve cumulative epsilon/delta in DurablePrivacyLedger before sampling. Retries must reuse one committed output byte sequence and must not resample under the same reservation.
7. **Qualification receipt:** record exact library/version, constructor or mechanism ID, feature flags, policy/query version, adjacency, privacy unit, contribution bound, derived sensitivity, epsilon/delta, output digest and verifier/qualification reference. Sign or externally anchor this receipt if a malicious database operator is in scope.
8. **Fail closed:** if the selected library constructor requires an unqualified feature, the clipping step is missing, sensitivity is unknown, the budget ledger is unavailable, or the receipt does not match the reservation, do not release a research/public DP output.

## Implemented verification interface and remaining work

The current verifier requires a signed canonical receipt that binds:

- reservation receipt/query hashes, release ID, UTC window, recipient and exact output digest;
- declared mechanism, adjacency, integer contribution cap, sensitivity, epsilon/delta, cohort size and policy IDs;
- library name/version, constructor/mechanism ID, feature flags, privacy unit and qualification reference;
- execution ID and signer-key fingerprint.

It rejects unknown or missing fields, duplicate JSON keys, non-canonical JSON, invalid signatures, an unpinned signer, disallowed library/version/constructor/feature flags, mismatched reservation fields, mismatched output bytes, missing retained attestations and an energy-sum sensitivity not equal to the configured `[0, L]` contribution cap. Signed evidence and output bytes commit in one transaction and are rechecked after restart. The latter check is valid only for this narrow bounded integer sum profile under the recorded add/remove or replace-one-participant adjacency; other aggregation profiles require another reviewed contract.

The signed receipt is an attester assertion, not independently verified execution evidence by itself. Remaining work before a production DP claim includes a qualified noise/aggregation adapter; an independently verified clipper and privacy map; a service that generates and signs attestations only after checking those facts; protected signer provisioning/rotation/revocation; externally auditable qualification policy; fault-injection across reserve/sample/sign/commit/publish; and a deployment boundary that prevents ordinary callers from using the internal raw storage primitive. The published output must be exactly the committed bytes and must never be regenerated after restart.

## Adversarial qualification tests

Before any DP claim, add tests proving that:
- a household contributing thousands of intervals is clipped once at the household boundary;
- duplicate packets and duplicate release IDs do not spend budget twice or generate a fresh output;
- overlapping/correlated queries are accounted for by composition even if the extra interval-overlap rule is bypassed;
- the claimed adjacency and sensitivity correspond to the actual aggregator;
- an output with the wrong mechanism ID, bound, epsilon/delta, reservation, or output digest is rejected;
- malformed, duplicate-key, non-canonical and incorrectly signed attestations are rejected, including signatures from a key other than the deployment-pinned trust root;
- library version, mechanism ID, privacy unit, qualification reference and feature flags match deployment policy exactly;
- a reservation whose budget is exhausted or unavailable cannot publish a newly sampled value;
- a process crash between reserve, sample, output commit and publish never exposes an uncommitted output or resamples an existing reservation;
- no raw participant identifiers or fine-grained readings are copied into the public receipt.

## Qualification status

**Not qualified yet.** The durable ledger provides persistent reservations and immutable output binding. This is an accounting and workflow primitive—not a DP engine. Until the library constructors, contribution-clipping implementation, sensitivity proof and qualification receipt are reviewed and tested together, reports must state “DP mechanism not yet integrated” rather than “differentially private.”
