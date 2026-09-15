# IG-008A0 — Mycelix cross-repository conformance

Status: **MeasurementOnly / CrossImplementationConformance**

Tracks: #3229.

## Purpose

Before Symthaea uses Mycelix governance inside broader institutional experiments, it must prove that an independent Symthaea implementation reads the same frozen mechanism profile and derives the same canonical counterexamples as the Mycelix research oracle.

This is a narrow semantic-conformance gate:

```text
same exact profile
+ independent implementations
+ byte-identical canonical corpus
= cross-implementation agreement
```

It is **not** governance-safety evidence.

## Symthaea parent

Stacked on IG-006A0H / draft #3104 exact head:

`5d28ec2d9b2707010e9babb12bff716a55eec238`

That parent establishes content-bound external profile references for future institutional-lab receipts.

## Frozen Mycelix evidence lineage

Exact Mycelix research-evidence head:

`4b4e27910a98ad393e5a508e54e0f73c48c19107`

This head contains the additive Mycelix research lineage through IG-007A4 / draft #897.

The observed mechanism profile within that lineage binds production source subject:

`fca2c107a1ea5108823ce617ba4111b6f7f77230`

and exact Git blobs:

```text
coordinator 969b845e6186cbcad507c742a718060844f82eb2
integrity   658562c8dfaf6a2f1b97a7bfd5cf0fc8a5ab6e66
```

The research-evidence head is not claimed to be a deployed production release.

## Mechanism profile reference

The exact content-bound mechanism reference is:

```text
id             mycelix-voting-observed-fca2c107-v2
revision       2
content_sha256 680af4668889c299b7e0d74531f44894a64a384be71778bca54d3f21ca80ac01
authority      ObservedSourceBound
```

This maps naturally onto the IG-006A0H `id + revision + content_sha256` profile model.

## Counterexample lineage

Predecessor Mycelix corpus:

`bb1cdcfe2205bcf6e6d718b536cbb73ba29a21c9da7dfa664798d02f9f866d90`

Current A4 corpus:

`ee5e7649a773f564b443320689f465080d4641a0f4f09f13c0c49a7087d6dc10`

The independently reproduced fixtures are:

- CE-06 — repeated delegated records alter tally mass;
- CE-07 — overlapping full delegations duplicate represented source mass;
- CE-08 — closed-window direct/delegated admission differential;
- CE-09 — below-Phi-threshold direct/delegated admission differential.

## Independence boundary

`scripts/ig008a0_mycelix_adapter_oracle.py` is stdlib-only and does not import:

- Mycelix Python validators;
- Mycelix counterexample oracles;
- Mycelix Rust code;
- Symthaea game-theory implementation code.

It independently validates the minimum required profile/source fields and derives the canonical corpus.

The Mycelix oracle remains an external implementation used only by the qualification workflow for comparison.

## Exact-head cross-repository qualification

The workflow checks out two repositories simultaneously:

```text
Symthaea exact PR subject
Mycelix exact evidence head 4b4e279...
```

It then:

1. verifies both checkout SHAs;
2. verifies exact Mycelix production-source blob identities;
3. runs the Mycelix v2 profile validator;
4. runs the Mycelix A4 self-test and corpus generator;
5. runs the independent Symthaea self-test and corpus generator against the same profile;
6. requires byte equality of the complete corpus outputs;
7. requires exact frozen corpus commitment;
8. verifies Symthaea checkout immutability.

No `latest`, branch-tip rebinding, or normalization-after-disagreement is permitted.

## Authority ceiling

A PASS means only:

```text
CrossImplementationConformance
```

It does not promote the Mycelix profile beyond `ObservedSourceBound`, and it does not establish:

- deployment currentness;
- production exploitability;
- behavioral realism;
- fairness;
- constitutional legitimacy;
- mechanism safety.

## Next step

IG-008A1 may introduce a Rust Mycelix mechanism adapter only after this conformance gate passes. The Rust adapter's first requirement should be agreement with the same frozen golden fixtures before stochastic agents or broader institutional simulations are introduced.
