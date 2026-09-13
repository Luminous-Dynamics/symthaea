# LQCD-021D — RNG namespace and checkpoint-lineage oracle

Independent standard-library Python subject for beta=6.0 phase/purpose stream separation and restart-checkpoint commitment semantics.

Exact executed subject SHA-256:

`d9c07b4b3b4f76c2b5409351ce512798d3e20b6a6a78b7fee43dfe8a739c7a33`

Canonical result SHA-256:

`2d293bdfeeb783143a1746cd476c7c40c4fde7fe5a2de5bf2f0bf491c6cc3d0b`

## Stream namespace contract

Version-1 successor coordinates reserve one 64-bit stream identifier as:

`[phase:4 | purpose:4 | ensemble_slot:24 | replica:16 | rank:16]`

Initial phase values are throughput, pilot, final and qualification. Initial purpose values are gauge transition, diagnostic, analysis and bootstrap. The executed oracle enumerates all 16 phase/purpose combinations at identical lower coordinates and proves all 16 stream IDs are distinct.

Frozen examples:

- throughput gauge transition: `0x1100021d00070000`;
- pilot gauge transition: `0x2100021d00070000`;
- final gauge transition: `0x3100021d00070000`.

This is a successor namespace and does not reinterpret existing V1 `LatticeStreamDomain` IDs.

## Seed commitment

A 32-byte seed may be preregistered by SHA-256 over the domain-separated encoding `symthaea.lqcd.seed.v1\0 || seed`. The checkpoint commits the resulting 32-byte value rather than allowing an unbound seed to appear at restart time.

## Checkpoint commitment

The canonical checkpoint byte encoding binds:

- phase / purpose / packed stream ID;
- ensemble slot / replica / rank;
- chain identity;
- campaign-subject digest;
- seed commitment;
- completed update ordinal;
- RNG word position;
- gauge-field digest;
- sampler-subject digest;
- numerical-profile digest;
- environment digest;
- exact predecessor checkpoint digest.

All integers use fixed-width big-endian encoding. Digest fields are exactly 32 bytes. The encoding begins with domain tag `symthaea.lqcd.checkpoint.v1\0`.

Frozen lineage digests:

- genesis: `872d5368c243e5e19fc9b9e0dcc8beeff2cd6ffc6a110ba4cdd5e1c56cce17b5`;
- second: `895a49ff92db11487934e89da7f87dd9b556dc6e67a279a3a926f2141ab32eea`;
- third: `5b1748f289e0e5e731f6fea3d6838d2176511f3a080416532c4d8e6aec286753`.

A stale predecessor is rejected. A same-ordinal/different-field fork produces a distinct checkpoint identity. Mutating every authoritative checkpoint field independently changes the commitment in the frozen fixture.

## Important claim boundary

This oracle establishes **namespace and checkpoint-commitment semantics only**. It deliberately does **not** implement ChaCha8 and therefore does not claim that restoring `(seed, stream_id, word_pos)` reproduces subsequent RNG bytes or gauge evolution. The production Rust child must separately prove exact ChaCha8 byte continuation and then prove `N+M` lattice transitions equal `N -> checkpoint/restore -> M` under the qualified numerical profile.

## Scientific boundary

No equilibrium, burn-in, stride, retained-configuration, topology, static-potential or benchmark claim follows from this subject.
