# ETK-3C — Currentness Freshness Policy V1 Reference Boundary

Status: **independent exact-byte reference theorem; no production Rust authority is introduced here**

## Core theorem

```text
finite currentness window
!= policy-bounded currentness
!= authorized freshness policy
!= trusted evaluation clock
!= current analytical authority
```

ETK-3C bounded-currentness V2 (#1999/#2211) closes the unbounded-time problem, but a caller could still choose a finite window that is arbitrarily long. This reference tranche adds a distinct freshness-policy semantic layer.

## Freshness policy

The fixture policy is content-addressed under:

```text
symthaea.etk-currentness-freshness-policy.v1
```

and commits:

```text
policy_key = native-analytical-currentness
clock_basis = unix_epoch_ms
max_validity_ms = 86400000
policy_record_digest = sha256:7777...7777
```

The policy-record digest is a fixture premise. Its presence binds identity; it does **not** authenticate who approved the policy.

Frozen policy revision:

```text
sha256:a65f087a26d73f4ae9cb95bf8fdf0da556bb646268eeeb22ff52f8bc55442896
```

## Policy-bound currentness V3

`currentness-assertion.v3` commits the freshness-policy revision in addition to attestation, observation time, expiry, twin and validity-domain identity.

Construction fails closed unless:

```text
valid_until_unix_ms > observed_at_unix_ms
(valid_until_unix_ms - observed_at_unix_ms) <= max_validity_ms
```

The positive fixture uses the exact policy limit:

```text
observed_at_unix_ms = 1789123456000
valid_until_unix_ms = 1789209856000
max_validity_ms = 86400000
```

Frozen V3 currentness revision:

```text
sha256:b1b18be74cdc9fcfb996687b93608d8a225c4c16e0756aac2de82ab4e1f68bb5
```

A window one millisecond longer than the policy limit is denied. A one-millisecond window is valid. A zero-width or negative-width window is denied.

## Analytical chain impact

The five upstream analytical semantics remain frozen and unchanged:

```text
requirement  sha256:10891514f6551c85b10674a3df04ab2c6d19f74f014adeacc671fa31deecd419
obligation   sha256:ea6e2f0f5e3ec37524325eda24b95ead75fa502f5d2a0b9410a407aed3a12a8c
method       sha256:73bb060e12b166cecfea9a73c1f274f4443f1a540a777e7ac8260c65808bedc3
input        sha256:3b9a62ea2ce031b73c188e8c0138fb1ebcda20e77d7f2401502ec493323d3c86
policy       sha256:7edc8fcbc03184413cc9c275537bdefb2d1c18f281406dc1adacd9a53a0a3bb5
```

The currentness-dependent chain re-keys deterministically:

```text
freshness policy  sha256:a65f087a26d73f4ae9cb95bf8fdf0da556bb646268eeeb22ff52f8bc55442896
currentness V3    sha256:b1b18be74cdc9fcfb996687b93608d8a225c4c16e0756aac2de82ab4e1f68bb5
plan              sha256:da8114d8431b171d185702157d8bbeab65f6be64fcac840051965ad4d3b80abf
admitted          sha256:9ebb6151f3cd99bbc34b252f399f981f20d9b78af99647862e05b8456fe2dd67
historical receipt sha256:419be46db68adf89d4581576b4fdefb35b230b296cfc0862a36a79fce001da3f
current fact V3   sha256:b651337d70e5c2c502639522caff6d9534743998a931da7a961dc52fd30947f8
```

The current-fact V3 preimage commits the freshness-policy revision, observation time, expiry and explicit evaluation time.

## Adversarial theorem

The exact self-test denies:

- zero/negative freshness-policy duration;
- zero-width currentness window;
- currentness window exceeding policy by one millisecond;
- evaluation before observation;
- evaluation after expiry.

It also proves:

- exact maximum policy duration is accepted;
- a shorter valid duration produces a distinct currentness identity;
- different policy-record identity changes both policy and currentness identity;
- different valid evaluation instants produce distinct current-fact identities;
- refreshed attestation changes currentness and composed plan identity.

## Exact-byte execution evidence

The checked-in oracle is exactly Git blob:

```text
27b4eb9daa07e7e7cc29767432916f36e3c008f5
```

Those exact bytes were executed locally before check-in:

```text
--self-test              PASS
python3 -m py_compile    PASS
raw SHA-256              4dcde3a9fce9a01f8fd35b6dfe6a18d328e2f9ed19f3edde0568672200ababe3
Git blob SHA-1           27b4eb9daa07e7e7cc29767432916f36e3c008f5
checked-in Git blob      27b4eb9daa07e7e7cc29767432916f36e3c008f5
```

This is exact-byte **reference execution evidence** only.

## Production migration rule

Do not weaken #2211 by simply adding another optional field. A production migration should make the policy-bound currentness type the only constructor capable of minting new present-tense analytical facts.

A safe sequence is:

```text
#2211 bounded currentness V2
    -> exact Rust execution
    -> typed freshness policy
    -> policy-bound currentness V3
    -> source ratchet forbidding policy-free currentness authority
    -> explicit freshness-policy authorization boundary
    -> trusted-time capability
```

## Deliberate nonclaims

This reference does not establish freshness-policy authorization, signer identity, organizational authority, attestation authenticity, trusted time, clock monotonicity, physical model validity, model qualification, evidence independence, requirement completeness, certification, manufacturing approval, deployment approval, or actuation authority.
