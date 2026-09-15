# IG-008M0 — Composed Mycelix observed mechanism manifest

## Purpose

IG-008M0 gives the Symthaea Institutional Laboratory one content-addressed reference for the **currently modeled portion** of the observed Mycelix governance mechanism.

It composes three independently source-bound surfaces without pretending they prove complete end-to-end institutional behavior.

## Manifest identity

```text
id        mycelix-governance-observed-fca2c107-bundle-v1
revision  1
SHA-256   a7bfe0a285dad6c83d2ac82a9edc7bba0c2a300532cba24a604411ca6c8e4e35
authority MeasurementOnly
class     PreparedComposition
```

All components bind production subject:

`Luminous-Dynamics/mycelix@fca2c107a1ea5108823ce617ba4111b6f7f77230`

## Components

### Voting / delegation

```text
profile  mycelix-voting-observed-fca2c107-v2
rev      2
SHA      680af4668889c299b7e0d74531f44894a64a384be71778bca54d3f21ca80ac01
corpus   ee5e7649a773f564b443320689f465080d4641a0f4f09f13c0c49a7087d6dc10
evidence 4b4e27910a98ad393e5a508e54e0f73c48c19107
adapter  IG-008A0
```

### Execution / timelock

```text
profile  mycelix-execution-observed-fca2c107-v1
rev      1
SHA      c977bdcef9e5faac83351050999451432b618d5cc523bece804eba5dd1ae81f6
corpus   0c6669e44d6d18396ede43324f5cf3abbb25ddd3a2a9f59abb2c8a3699ba5fd4
evidence 197714209c60503f0fba4143409da383bc9cbf83
adapter  IG-008E0
```

### Runtime governance configuration

```text
profile  mycelix-governance-config-observed-fca2c107-v1
rev      1
SHA      de4435a69356557b1812f8beb46d654c66b9c957be9d18c64bd0431f92546d5a
corpus   3009e97529934fa8f470769de5615dcfda17bde0e942683e939d2b733430a216
evidence 88922e7950ddf030026b5d4d2b06e0ed63727692
adapter  IG-008C0
```

## What the manifest means

A component reference means:

```text
these exact bytes / observations are the mechanism model input
```

It does not mean:

```text
this subsystem is safe
this subsystem is current in deployment
this subsystem's behavior is socially desirable
```

Composition means the three models are packaged under one identity and share the same frozen production subject. It does not automatically establish causal correctness between them.

## Source-bound coverage

The manifest currently covers:

- direct/delegated voting observations;
- tally, tier and delegation semantics represented by the voting profile;
- timelock construction/readiness/signature-branch/action-dispatch observations;
- runtime consciousness-config read and mutation-authority observations.

## Explicitly not established

The manifest identity includes these missing surfaces:

1. complete proposal lifecycle authorization;
2. threshold-signing cryptographic and committee correctness;
3. complete council and guardian authority;
4. constitution amendment process correctness;
5. actual finance/fund-allocation money movement;
6. downstream finance/commons/civic authorization;
7. identity/DID/Sybil system correctness;
8. cross-DNA/network failure semantics beyond frozen observations;
9. deployment currentness;
10. human and AI behavioral validity;
11. governance safety, fairness and legitimacy;
12. cross-component causal correctness beyond explicitly modeled bindings.

Removing or changing this list changes manifest identity.

## Why the negative coverage is first-class

A research model can become misleading by becoming more convenient to name.

Without an explicit negative coverage surface, a label such as `MycelixGovernance` can gradually acquire claims that were never tested. IG-008M0 makes the boundary content-addressed so scope expansion requires a visible successor.

## Qualification ceiling

The static manifest is:

`MeasurementOnly / PreparedComposition`

If the exact-head workflow independently requalifies all three component corpora against their exact Mycelix evidence heads, it may emit:

`MeasurementOnly / ComposedCrossImplementationConformance`

That still does not mean `EndToEndGovernanceQualified`.

## Institutional-lab use

An experiment that claims to use the observed Mycelix governance bundle should bind the manifest commitment:

`a7bfe0a285dad6c83d2ac82a9edc7bba0c2a300532cba24a604411ca6c8e4e35`

rather than only a label such as `mycelix`.

If an experiment substitutes one component—for example a repaired execution profile while retaining old voting semantics—it must use a new manifest identity.

This makes before/after governance experiments precise:

```text
old bundle
vs
successor bundle
```

rather than vaguely comparing “old Mycelix” and “new Mycelix.”

## Known open issues retained

The v1 manifest retains:

```text
#851 #855 #856 #876 #877 #892 #900 #904 #943 #944
```

This list is descriptive evidence metadata, not a severity aggregation score.

## Successor rule

A repaired production subject, changed component profile, changed evidence head, changed coverage boundary, or newly modeled subsystem creates a new manifest identity.

Never rewrite this manifest to make historical evidence appear more complete.

## Non-claims

No complete end-to-end governance qualification, deployment currentness, mechanism-safety verdict, fairness/legitimacy theorem, or human/AI behavioral-validity claim.