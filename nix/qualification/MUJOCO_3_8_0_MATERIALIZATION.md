# MuJoCo 3.8.0 qualification materialization

This document defines the pinned MuJoCo runtime input intended for QUAL-INFRA-003D and later hermetic MuJoCo qualification.

It is deliberately separate from the existing `mujoco337` flake derivation, which was introduced for the older `mujoco-rs 2.3.3 + MuJoCo 3.3.7` compatibility line. Current `symthaea-humanoid` and `symthaea-multirotor` both declare `mujoco-rs 4.0.1`, whose release line targets MuJoCo 3.8.0.

The source derivation in `mujoco-3.8.0.nix` pins the official Google DeepMind release assets directly and preserves the extracted upstream bytes without Nix fixup/patchelf mutation.

## Pinned upstream release

Release tag: `3.8.0`

Release commit: `34d69ad4cb1a21846b8297e2bc5e68a4938276c1`

Linux x86_64 asset:

- asset ID: `404891937`
- asset: `mujoco-3.8.0-linux-x86_64.tar.gz`
- size: `20812715`
- SHA-256: `2be88c6f92a06c3eaffdb47d3a6d3fbf159fbc057e9d272d592fb194e41fefab`

Linux aarch64 asset:

- asset ID: `404891905`
- asset: `mujoco-3.8.0-linux-aarch64.tar.gz`
- size: `20674425`
- SHA-256: `adc4a7856d2b8d42ba4e889b57cbceb13a329c869f410cf1ad110b153c4745e4`

## Hermetic build rule

The later MuJoCo qualification profile must set

`MUJOCO_DYNAMIC_LINK_DIR=/deps/mujoco-3.8.0/lib`

from the read-only 003D materialization. `mujoco-rs 4.0.1` checks that explicit dynamic-link directory before its pkg-config / `auto-download-mujoco` fallback, so a correctly materialized library prevents the download path from executing.

The materialization manifest must bind the upstream release identity, platform, asset ID/name/size/hash, extracted tree digest, and the exact `libmujoco.so` digest.

## Nonclaims

This source pin does not establish that the Nix derivation builds, that the extracted runtime is ABI-compatible with a particular host root, that Bubblewrap containment works, or that any HUM-DYN/HUM-WRENCH qualification has passed.
