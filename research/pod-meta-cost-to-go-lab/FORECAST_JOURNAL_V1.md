# Immutable Forecast and Outcome Journal V1

**Status:** serialized Rust research journal. It does not provide signatures, external attestation, or distributed consensus.

## Invariant

A prediction is frozen before an operation or proof attempt. An observed result is a separate later artifact. The evaluator must never rewrite the probability or remaining-cost distribution that the scheduler actually used.

The journal has two append operations and read-only lookups:

- `append_forecast(ForecastReceipt)`
- `append_outcome(ForecastOutcomeReceipt)`
- immutable `forecast(id)` and `outcome(id)` accessors

There are intentionally no update or delete APIs.

## Forecast record

Each pre-execution forecast includes:

- unique forecast identity;
- run identity and a strictly increasing per-run decision sequence;
- task-family and operation-family labels;
- state, proof-frontier, model, and calibration digests through the forecast object;
- predicted completion probability, marginal remaining-resource quantiles, mandatory verification reserve, support count, and shift status.

Duplicate forecast IDs, duplicate `(run_id, decision_sequence)` pairs, and non-increasing decision sequences are rejected.

## Outcome record

An outcome is keyed to exactly one prior forecast and includes:

- observed sequence later than the forecast's decision sequence;
- realized resource vector from the forecast decision to the terminal observation;
- terminal class: sufficient proof, right-censoring, or a competing terminal event;
- evaluator / terminal-evidence digest;
- independent meter-receipt digest;
- verifier digest when sufficient proof is claimed.

Sufficient proof cannot be recorded without a nonzero verifier digest. Censored or competing outcomes may not carry a zero verifier digest. Every outcome needs nonzero evaluator and meter digests. Unknown forecast IDs, duplicate outcomes, and outcomes ordered before the forecast are rejected.

The journal permits exactly one terminal outcome per forecast. A censored observation records the observed resource lower bound and censor reason; it is not later rewritten as a completion failure or completion success. If an evaluation must be superseded, the correction must be a new linked record in a future schema version; silently replacing V1 data is prohibited.

## Trust boundary

The journal enforces local append semantics and structural linkage. The 32-byte digests are caller-supplied identity fields in this simulator; they are not recomputed, signed, or independently verified here. A production implementation requires canonical serialization, content-addressed digest recomputation, signature / attestation policy, durable append-only storage, access separation between planner and evaluator, and crash-consistent commit semantics.

## Qualification

Unit tests cover duplicate IDs and sequences, ordering, detached outcomes, missing verifier evidence, censored lower-bound preservation, single assignment, and invalid digests. The deterministic runner includes one verified-completion example and one budget-censored example.