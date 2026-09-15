# IG-008CD0 — ConstitutionRuntimePolicyDownstream observed submanifest

Issue: #3390

Parent: IG-008CS0 / draft #3389.

## Purpose

Compose the exact observed **ConstitutionRuntimePolicyDownstreamV1** slice from three independently reproduced mechanisms without broadening the claim to the entire Constitution or charter-amendment system.

## Identity

```text
schema    symthaea-constitution-runtime-policy-downstream-manifest-v1
id        mycelix-constitution-runtime-policy-downstream-observed-fca2c107-v1
revision  1
authority ObservedCompositeSubslice
ceiling   ObservedConstitutionRuntimePolicySliceOnly
SHA-256   157402e76abffd3849dc6d2001a7c0bbde296e80ab6a7f5c77c8f3e288302183
```

Coverage semantics:

`SourceObservedMechanismRepresentedNotPropertySatisfied`.

## Mechanism 1 — ConstitutionParameter

```text
profile SHA       770552d12489df1d2cdf8b0af676b01ea9a3da21940f70ed8a71910deaa35009
corpus SHA        b37be9d2e3fd0cbec3696a19327c26dc4ba7062ec089ad92ad99bc28a11fea8e
Mycelix evidence  3232d611d8833b03eba9f5412f5d7cb0cf89d4e1
Symthaea evidence 3f6bd4210690b79734b71bd4168ce843aeda1036
Symthaea PR       #3349
issue             Mycelix #1002
```

This component preserves both the existing-parameter containment fact and the remaining mutation-authority gaps.

## Mechanism 2 — ConstitutionBridgeSync

```text
profile SHA       60daae86044098561fa8e41bcdf6f695ab41234760be4b7e235d2780271b681e
corpus SHA        2ca6d79212cfd0acae1e974c8390d564bab06821050d8eacbc32f437630ef60b
Mycelix evidence  abb277cc39418e43e80d48e8a4a57bc7c939d9e4
Symthaea evidence 594ac1a765131d242c01b71859ff28cb2fe71b1a
Symthaea PR       #3389
issues            Mycelix #943/#944
```

It preserves the absent-target, best-effort divergence, authorization-dependency and missing-reconciliation observations.

## Mechanism 3 — GovernanceConfig

```text
profile SHA       de4435a69356557b1812f8beb46d654c66b9c957be9d18c64bd0431f92546d5a
corpus SHA        3009e97529934fa8f470769de5615dcfda17bde0e942683e939d2b733430a216
Mycelix evidence  88922e7950ddf030026b5d4d2b06e0ed63727692
Symthaea evidence c66296a29e27633f470d18568c9a7de3fae5c624
Symthaea PR       #3269
issue             Mycelix #943
```

It preserves the distinction between proposal-record existence and actual Constitutional/approved mutation authority.

## Covered mechanism edges

Exactly:

```text
execution_update_parameter_to_constitution_parameter
constitution_parameter_storage_and_projection
constitution_parameter_to_runtime_config_sync_attempt
runtime_governance_config_mutation_entrypoint
runtime_governance_config_integrity_shape
```

## Explicitly unestablished properties

Representation does not establish:

- authorized ConstitutionParameter mutation;
- atomic ConstitutionParameter/runtime-config mutation;
- successful Constitution→runtime synchronization;
- content-bound reconciliation;
- authorized runtime GovernanceConfig mutation;
- authoritative runtime-config currentness;
- deployment currentness;
- governance safety.

These remain first-class fields in the manifest so future composition cannot make them disappear by omission.

## Scope boundary

This submanifest does **not** cover:

- charter creation/currentness;
- constitutional amendment proposal/ratification/application;
- enhanced immutable-core amendment requirements;
- treasury/credit mutation authority;
- deployment currentness.

A future charter/amendment assurance program must remain a separate component.

## Qualification architecture

The qualifier uses exact evidence heads rather than copying sibling-branch oracles:

1. checkout exact CP Mycelix + Symthaea conformance heads;
2. replay CP corpus cross-implementation;
3. checkout exact GovernanceConfig Mycelix + Symthaea conformance heads;
4. replay C1 corpus cross-implementation;
5. checkout exact Sync Mycelix evidence and use the current exact IG-008CS0 oracle;
6. replay CS1 corpus cross-implementation;
7. validate the submanifest twice byte-identically;
8. assert exact mechanism refs, issue set, unestablished-property set and commitment;
9. verify all checkouts immutable.

## Top-level correction

Composite-v4 / #3364 remains useful historical candidate evidence but its single ConstitutionParameter coverage is not sufficient to imply this full runtime-policy downstream slice.

A successor top-level manifest should bind this submanifest explicitly rather than silently treating ConstitutionParameter as the whole constitutional runtime-policy plane.

## Non-claims

No complete Constitution coverage, charter-amendment coverage, authorized parameter/config mutation, successful sync, live divergence, deployment currentness, governance safety, fairness, or constitutional legitimacy is established.
