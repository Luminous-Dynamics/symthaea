# LQCD-021L — deterministic keyed binary64 reduction oracle

Independent standard-library Python subject for the canonical reduction semantics required by the beta=6.0 numerical profile.

Exact executed subject SHA-256:

`88671f77f758717dd0a9d1c4cf0111a8e77cd1a8dd9a18278db47e013dfc5120`

Canonical result SHA-256:

`fe970141d8dcec56d2c669088c4b4cf342acc36d0afd926e990c755da848172a`

## Qualified contract

- scientific leaves are `(unique_key, finite binary64_value)` records;
- keys define the authoritative leaf order, not insertion or worker-completion order;
- leaves are sorted by ascending key;
- each reduction level adds adjacent pairs left-to-right;
- an odd unpaired leaf advances unchanged to the next level;
- duplicate keys, empty reductions, and non-finite inputs fail closed;
- shard partitions are permitted only when shards preserve atomic keyed leaves and canonical reduction occurs after union;
- arbitrary shard-local floating partial sums are **not** equivalent leaves and are not authorized by this oracle.

## Executed evidence

The frozen baseline reduces to:

- sum `4.375`, binary64 bits `0x4011800000000000`;
- mean `0.625`, binary64 bits `0x3fe4000000000000`.

All 64 deterministic input permutations reproduce the exact sum bits. Four distinct shard-union orders also reproduce the exact sum bits.

An odd-leaf fixture freezes sum `1.6875`, bits `0x3ffb000000000000`.

The negative control proves ordinary insertion-order summation is scientifically insufficient: the same mathematical values produce `0.0` (`0x0000000000000000`) in one naïve order and `1.0` (`0x3ff0000000000000`) in another.

## Intended production use

The Rust counterpart should expose a versioned numerical-profile/reduction ID, reproduce these exact fixtures, and be usable by Wilson origin/orientation aggregation, measurement-shard union, and later statistical reductions. Higher-level callers must supply canonical scientific keys appropriate to their domain.

This oracle does not yet change any Wilson measurement implementation.

## Scientific boundary

This subject qualifies deterministic keyed binary64 reduction semantics only. It does not establish cross-architecture bitwise identity, Markov-chain restart equivalence, equilibrium, statistical adequacy, or any physical lattice-QCD result. Those require separately bound numerical-profile and campaign evidence.
