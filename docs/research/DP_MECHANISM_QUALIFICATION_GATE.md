# Differential-Privacy Mechanism Qualification Gate for Energy Telemetry

**Status:** Research decision record. No noise mechanism is currently approved for production use in this telemetry stack. The durable privacy ledger accounts for declared releases, but does not make arbitrary bytes differentially private.

## Decision

Do not implement an ad-hoc noise sampler in the telemetry code and do not set a production “DP verified” bit from the ledger alone. A candidate mechanism must be selected from a maintained library, pinned to an exact release, and qualified with the exact input domain, privacy unit, adjacency definition, contribution bound, sensitivity proof, privacy parameters and output type.

The current ledger's discrete_laplace and gaussian strings are **policy declarations**, not evidence that those mechanisms ran. A reservation and immutable output digest are useful accounting artifacts, not a mathematical proof or proof that a caller used the claimed sampler.

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

## Required interface before implementation

The future mechanism adapter should consume a validated vector of one value per privacy unit, not arbitrary meter intervals, and return a typed result:

- canonical noised output bytes;
- mechanism execution receipt with the exact bound and privacy parameters;
- qualification reference tied to the exact library version and constructor;
- sensitivity/privacy-map evidence from the selected library or an independently reviewed proof;
- an output digest matching those exact bytes.

Only after independent receipt validation should the release service call the durable output-binding operation. The published output must be exactly the stored output returned by committed_output; it must not be regenerated after a restart.

## Adversarial qualification tests

Before any DP claim, add tests proving that:
- a household contributing thousands of intervals is clipped once at the household boundary;
- duplicate packets and duplicate release IDs do not spend budget twice or generate a fresh output;
- overlapping/correlated queries are accounted for by composition even if the extra interval-overlap rule is bypassed;
- the claimed adjacency and sensitivity correspond to the actual aggregator;
- an output with the wrong mechanism ID, bound, epsilon/delta, reservation, or output digest is rejected;
- a reservation whose budget is exhausted or unavailable cannot publish a newly sampled value;
- a process crash between reserve, sample, output commit and publish never exposes an uncommitted output or resamples an existing reservation;
- no raw participant identifiers or fine-grained readings are copied into the public receipt.

## Qualification status

**Not qualified yet.** The durable ledger provides persistent reservations and immutable output binding. This is an accounting and workflow primitive—not a DP engine. Until the library constructors, contribution-clipping implementation, sensitivity proof and qualification receipt are reviewed and tested together, reports must state “DP mechanism not yet integrated” rather than “differentially private.”
