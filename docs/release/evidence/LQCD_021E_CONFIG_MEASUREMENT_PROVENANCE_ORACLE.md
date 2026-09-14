# LQCD-021E — immutable ConfigId and measurement-sharding provenance oracle

Independent standard-library execution for the configuration/measurement provenance semantics tracked by #2714.

Exact executed subject SHA-256:

`b882f4d44a1e469fcf7c272161eb7826ae96052ae137d655fae7a4b93fff1464`

Canonical result SHA-256:

`873a1297a1666c857d7c2bd5687a16a3ac8e7ba59184247ed4f4159554d35ddf`

## Configuration identity

`ConfigId` is domain-separated from measurement identity and binds campaign subject, phase, chain ID, retained ordinal, transition ordinal, generation subject, canonical gauge-field digest, and checkpoint commitment.

The oracle mutates all eight authoritative fields independently and requires a different `ConfigId` every time. Synthetic pilot and final configurations with otherwise corresponding structure are mechanically disjoint. Duplicate `ConfigId` values are rejected from a configuration-set commitment.

Synthetic fixture commitments:

- final configuration set: `85b19b9908cbbd0f3b4528fb3b7c6f41626f6b94788b3486f2ed0827f7500b60`;
- pilot configuration set: `09365d0ab115ab01121abd3c59fa10a0da58fbfb84b6a9f8b874e75e2e731115`.

## Measurement identity

A measurement key binds the immutable `ConfigId`, measurement subject, APE identity/revision, operator identity/revision, displacement and temporal extent. A semantic measurement value separately binds the raw complex binary64 result, numerical profile, executable subject and environment profile.

Operational `shard_id` and `attempt_id` are intentionally outside the semantic measurement value and instead enter the receipt identity. Therefore a retry can produce a new receipt without pretending that a new gauge configuration or a second independent measurement sample was created.

## Exact frozen design shape

The #2528 vector set contains exactly 24 displacements and `T=1..8`, therefore exactly 192 required measurement keys per retained configuration. The real 4000-configuration final design therefore derives exactly:

`4000 × 24 × 8 = 768000`

expected semantic measurement keys before any final analysis may begin.

The executed synthetic fixture uses 16 final configurations and therefore 3072 expected keys.

## Sharding / retry theorem

The same 3072 semantic measurements were recombined under deliberately different shard partitions and completion orders. The canonical measurement-set commitment remains:

`7a781d7c1a7fdc4c3f898b9134349456a049879a968f3599cf7bfa13d2c51cc5`

An exact retry of one key changes its operational receipt digest but preserves the semantic value digest and leaves the measurement-set commitment unchanged. A conflicting retry with different raw complex content fails closed.

The oracle also rejects:

- one missing expected key;
- an unknown extra key;
- a wrong operator revision;
- a pilot `ConfigId` entering the final set;
- a duplicate configuration identity;
- conflicting duplicate measurement payloads.

This means retries and sharding cannot increase the effective configuration count or silently redefine the ensemble.

## Scientific boundary

This subject qualifies configuration identity, semantic measurement identity, exact expected-key derivation, retry idempotence, and canonical shard union only. It does not establish equilibration, operator robustness, statistical adequacy, final-sample admission, string tension, Sommer scales, or EHK agreement.
