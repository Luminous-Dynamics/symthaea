# Engineering Trust Clock Recursive Production V6

## Status

Draft production follow-on to #2529, with independent recursive reference #2644.

Rust qualification remains dependent on exact-head repository CI.

## Authority model

```text
opaque AcceptedClockBasisV6
        + one-time exact policy/snapshot witness bind
        ↓
ContinuousClockBasisV1
(runtime carrier only; wire identity remains V6)
        ↓ no caller policy/snapshot/time/candidate input
ContinuousClockSuccessorPermitV1
        ↓ original signed observations
retained quorum policy
        ↓
VerifiedClockWindow V1 + exact witness
        ↓ retained continuity policy
VerifiedClockContinuity V1
        ↓
ContinuousClockBasisV1
(new accepted-clock-basis.v6 wire digest)
        ↓ repeat
```

`ContinuousClockBasisV1` is not a V7 evidence protocol. It is a private runtime carrier for already-authenticated V6 authority and its exact witness records.

## Frozen recursive reference

Reference: #2644.

```text
basis epoch 43
6b386d75ac26802b5879e4ff820da5ceb6285ab9cc3d6d96729ffb665c422a6e

permit epoch 44
d154b7bc136371d3d0bd24389dcc139769a20fa36dcf34dd4d0b2162658345e7
window epoch 44
2304236bcb292476137284a25e943fe54bd0addac1e48ab917ba353bd25d1ffa
witness epoch 44
0bd33fe50684886e8cd5fea899ac73c57add275855778bde69f5e396e0f3db89
continuity 43 -> 44
194ce5eb8c491f02fe2d596801100abf3bd587513d1b6b569c692ae6e2cbfbff
basis epoch 44
e1d9035a7b938521e4339836fd48f74d54ce685795b8acdefe42e18b02c86825

permit epoch 45
6b8bf16525b315c9d20c0b5cad0a5c3f61c241cddc60ecc1fde7b5708d090b91
window epoch 45
2322df198ef985211b8c2af47e317b729926786be770b85c4777213e5dc4441e
witness epoch 45
8d4b6e26fd9f7a5f659938a6ab4fc617a32b9e45913d170c1e91356fd5837da0
continuity 44 -> 45
e2fa905fb5b1791bdd3dbbd240e468ab9352c6bdebc9f5977158110aebd0e4ec
basis epoch 45
825721c953d67cf60e7882a6c0547e00166906f330ed012f8ef4a5a8372a1686
```

## Fail-closed boundary

Production tests require:

- exact epoch-44 and epoch-45 vector parity with #2644;
- V6 wire schema/domain remains unchanged across both advances;
- initial policy witness substitution is rejected;
- initial trust-snapshot witness substitution is rejected;
- after binding, permit derivation takes no policy, snapshot, clock observation, or caller time;
- after binding, advance takes no policy, snapshot, lifecycle time, precomputed window, or precomputed witness;
- retained snapshot must remain valid for every future envelope;
- no `AcceptedClockBasisV7` / `accepted-clock-basis.v7` path exists;
- runtime basis/permit capabilities remain private-field and non-deserializable.

## Scope

The tranche adds one stable runtime module, public exports, a recursive parity/adversarial corpus, and this evidence note. Existing V4/V5/V6 protocol objects and #2529 one-step successor authority are left unchanged.

## Deliberate nonclaims

This does not establish policy migration, trust-snapshot rotation, persisted authenticated clock receipts, ETK evidence currentness, requirement satisfaction, design qualification, deployment approval, or physical actuation authority.
