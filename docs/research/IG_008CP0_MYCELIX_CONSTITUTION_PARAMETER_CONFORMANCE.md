# IG-008CP0 — Mycelix ConstitutionParameter cross-repository conformance

Issue: #3342

Parent: IG-008F2 / draft #3330

## Purpose

Independently reproduce the frozen Mycelix ConstitutionParameter authority corpus before that downstream stage can become a represented component of the composite governance manifest.

## Frozen Mycelix evidence

```text
evidence head      3232d611d8833b03eba9f5412f5d7cb0cf89d4e1
production subject fca2c107a1ea5108823ce617ba4111b6f7f77230
same-tree authoring 31ede2365b81365bb119cd9351b2739119974130
```

Source blobs:

```text
execution coordinator       3dbb8a8f69b377e494ccf24164c94bd80f54e0ef
constitution coordinator    923a1ce789c8319c79df7f33a9241af50804ec55
constitution integrity      f83a457a8ff40b0003c07dba9da598a478c5e6f6
```

Profile:

```text
mycelix-constitution-parameter-observed-fca2c107-v1
770552d12489df1d2cdf8b0af676b01ea9a3da21940f70ed8a71910deaa35009
ObservedSourceBound
```

Corpus:

```text
b37be9d2e3fd0cbec3696a19327c26dc4ba7062ec089ad92ad99bc28a11fea8e
MeasurementOnly
```

## Independent oracle

`scripts/ig008cp0_mycelix_constitution_parameter_oracle.py` is stdlib-only and imports none of Mycelix's CP0/CP1 code or Rust implementation.

It independently validates only the source-profile facts necessary to reproduce:

- CE-CP-01 — execution existing-parameter gate mismatch;
- CE-CP-02 — new parameter without proposal linkage;
- CE-CP-03 — proposal-ID presence without authority reconstruction;
- CE-CP-04 — integrity shape validity does not create mutation authority;
- CE-CP-05 — timestamp-selected projection is not authoritative fork resolution.

The complete canonical corpus bytes must equal the Mycelix CP1 output.

## Important positive containment

CE-CP-01 is preserved as a **negative execution result / positive containment fact**:

```text
legacy execution UpdateParameter
+ existing parameter
+ no proposal_id
-> observed downstream presence gate rejects
```

IG-008CP0 does not reinterpret that failure as proof that the rest of the parameter authority boundary is sound.

## Qualification

The exact-head workflow:

1. checks out exact Symthaea product subject;
2. checks out exact Mycelix CP1 evidence head;
3. binds all three production source blobs;
4. syntax-compiles Mycelix CP0/CP1 and independent Symthaea oracle;
5. revalidates Mycelix CP0 twice;
6. runs Mycelix CP1 twice and emits canonical corpus;
7. runs Symthaea oracle twice and emits canonical corpus;
8. requires byte-identical Mycelix/Symthaea corpus bytes;
9. asserts exact profile/corpus commitments, #1002, semantic production subject and separate same-tree authoring head;
10. verifies both checkouts remain immutable.

## Composite consequence

After qualification, a new composite revision may add `ConstitutionParameter` and remove only:

`constitution_parameter_authorization_downstream`

from the uncovered set.

Coverage remains representation, not authorization or safety.

## Non-claims

No live unauthorized mutation, exploit, constitutional invalidity, authoritative currentness, deployment currentness, governance safety, fairness, or constitutional legitimacy is established.
