# CMAQ-053 — Compositional fleet, calibration and representation risk budget

## Purpose

CMAQ-043, CMAQ-052 and CMAQ-051 address different layers of one assurance problem:

- CMAQ-043 assigns a countable fleet-wide anytime false-alarm budget to assets, evidence channels and deployments.
- CMAQ-052 reserves part of one deployment's allocation for reference-envelope coverage failure and gives the remainder to sequential testing.
- CMAQ-051 combines several representation-specific e-processes under one fixed prior mixture.

CMAQ-053 makes their risk arithmetic compositional and immutable. It is an allocation/receipt layer; it does not re-prove the child detector theorems.

## Composition theorem

Let the global target be alpha_global in (0,1). For one-based asset, channel and deployment indices i,j,k, define

    alpha_local(i,j,k) =
      alpha_global / [i(i+1) j(j+1) k(k+1)].

Since sum over n>=1 of 1/[n(n+1)] equals one, the sum of all local allocations over the countable three-level fleet is alpha_global. A union bound therefore controls the probability of at least one fleet false alarm by alpha_global, without requiring independence between monitor streams.

Within one local deployment allocation, reserve

    alpha_test + delta_reference <= alpha_local.

The reference envelope misses its target with probability at most delta_reference. Conditional on envelope coverage, the online test has false-alarm probability at most alpha_test. Therefore

    P(deployment false alarm ever)
      <= delta_reference + alpha_test
      <= alpha_local.

CMAQ-051's fixed representation portfolio is a convex mixture of valid e-processes for the same null. Its member weights sum to one and do not consume separate alpha allocations. The shared portfolio uses alpha_test once. There is no alpha_test/K split and no additional union bound over representation members.

Combining the two levels gives the countable-fleet bound alpha_global, subject to every child theorem and evidence obligation being valid.

## Immutable plan contract

A composed plan freezes:

- global alpha, fleet-plan digest and fleet-inventory digest;
- asset/channel/deployment identities and their one-based priority indices;
- exact local alpha, alpha_test and reference-failure delta;
- the complete representation-member inventory and exact rational prior weights;
- deployment-plan, reference-manifest, target-envelope, monitor-build, representation-inventory and calibration-method digests.

The implementation uses reduced positive u128 rational numbers. Authoritative allocation, sum-to-one checks and risk comparisons use checked integer arithmetic only. Overflow fails closed. Floating-point values are not used for policy decisions.

The in-memory ledger rejects duplicate stream IDs, duplicate scope triples, duplicate deployment-plan IDs and identity/index remapping. It commits a receipt only after every validation and exact arithmetic operation succeeds.

## Important compatibility boundary

The calibration-method digest must identify the exact theorem and implementation in use. In particular, the broad common-conditional-mean martingale contract described in the CMAQ-052 research note must not be inferred merely from a digest or from a reference sample mean. A calibration implementation that assumes iid stationary reference/test samples supports only that narrower claim unless its code and evidence establish the stronger conditional-mean assumptions.

Similarly, CMAQ-053 does not establish that:

- the fleet inventory is complete or its index-to-identity authority is authentic;
- source digests/signatures are genuine;
- CMAQ-052's reference coverage assumptions hold;
- every CMAQ-051 representation is a valid e-process under the same conditional null;
- representation features, targets or pair/covariate identities are truthful;
- an in-memory ledger provides cross-process or cross-host replay protection.

Those remain independent qualification requirements. The composition theorem is conditional on them; arithmetic composition cannot repair an invalid child guarantee.

## Implementation

The standalone, dependency-free Rust crate provides:

- normalized exact rational arithmetic with checked overflow;
- the three-level telescoping local-alpha formula;
- exact alpha_test + delta_reference comparison against the local allocation;
- fixed representation-weight validation (positive weights summing exactly to one);
- immutable deployment receipts carrying all plan and evidence bindings;
- a consume-once in-memory fleet ledger with one-to-one identity/index maps;
- transactional rejection: invalid plans do not mutate the ledger.

## Tests and probe

Deterministic tests cover:

- exact local allocation for known index triples;
- telescoping one-dimensional partial sums and bounded three-dimensional fleet sums;
- alpha split equality accepted and over-budget plans rejected;
- two or more representation members sharing one alpha_test;
- invalid, missing, duplicate and non-normalized representation inventories;
- duplicate plan, stream and scope rejection;
- index-to-identity reassignment rejection with unchanged ledger state;
- digest-shape failures and checked u128 overflow.

Run:

    cargo +1.100.0 fmt --manifest-path research/cmaq-053/Cargo.toml --all -- --check
    cargo +1.100.0 test --manifest-path research/cmaq-053/Cargo.toml --locked
    cargo +1.100.0 clippy --manifest-path research/cmaq-053/Cargo.toml --locked --all-targets -- -D warnings
    cargo +1.100.0 run --release --manifest-path research/cmaq-053/Cargo.toml --locked --bin composition_probe

The probe emits exact rational allocations for a small registered fleet. It is an arithmetic wiring check, not an empirical calibration or false-alarm study.

## Graduation criteria

Before deployment use, replace the in-memory ledger with a durable authenticated authority that provides atomic, cross-host consume-once registration and replay protection. Independently audit the canonical serialization/signatures and index-to-identity inventory. Then run the child-specific reference-coverage and e-process qualification, followed by a fleet-level simulation that reports reference-envelope failures and conditional betting alarms separately.

No physical safety, statistical source-authenticity, complete-inventory or deployment-level guarantee follows from this crate alone.