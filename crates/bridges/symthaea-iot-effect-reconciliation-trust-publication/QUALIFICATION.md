# Effect Reconciliation Trust Publication v0.1

This crate atomically publishes the exact **outcome-verifier policy generation/digest + outcome-verifier trust head** authoritative for one device's terminal physical-effect reconciliation.

It is an atomicity meta-root, not an outcome verifier and not a journal-closing capability.

## Required terminal ordering

A later terminal reconciler must:

1. hold `CurrentEffectReconciliationTrustFence`;
2. require its published policy **generation and digest** to equal the owner-local current outcome verifier guard's exact policy generation and digest;
3. require its published outcome trust head to equal the owner-local current outcome verifier guard's exact trust head;
4. create `CurrentPhysicalEffectOutcomeFence` while publication is still held; that fence independently requires the historical proof's policy generation/digest and trust head to equal the same current guard;
5. re-read the rollback-protected unresolved effect-attempt journal and require its head to equal the challenge head retained by the verified outcome proof;
6. perform one terminal journal transition while both currentness conditions remain held.

This creates a three-way policy identity check:

```text
published policy generation/digest
    == current outcome guard generation/digest
    == historical proof generation/digest
```

A publication successor cannot become authoritative until the publication fence drops.

## Non-claims

This crate does not verify Ed25519, parse outcome evidence, decide a proof class, dispatch hardware, close an attempt journal, or mint actuation authority.

No qualification claim exists until the exact candidate head passes its focused hosted workflow and all required upstream evidence is green.
