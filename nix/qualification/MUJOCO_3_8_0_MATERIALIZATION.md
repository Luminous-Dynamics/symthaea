# MuJoCo 3.8.0 qualification materialization

Status: source pin only. No Nix build, hermetic materialization, MuJoCo ABI, source qualification, robotics authority, hardware observation, or physical-safety claim is established by this document.

## Why this exists

The repository's historical `flake.nix` MuJoCo derivation is still pinned to MuJoCo 3.3.7 and explicitly documents compatibility with the older `mujoco-rs 2.3.3 + MuJoCo 3.3.7` line.

Current `symthaea-humanoid` and `symthaea-multirotor` instead declare `mujoco-rs 4.0.1`. That release line targets MuJoCo 3.8.0. QUAL-INFRA-003D must therefore not reuse the 3.3.7 runtime as evidence for current humanoid/multirotor qualification.

`mujoco-3.8.0.nix` adds a separate version-correct qualification input without changing or removing the historical 3.3.7 derivation.

## Upstream release identity

Release tag: `3.8.0`

Release commit: `34d69ad4cb1a21846b8297e2bc5e68a4938276c1`

### Linux x86_64

- GitHub release asset ID: `404891937`
- asset: `mujoco-3.8.0-linux-x86_64.tar.gz`
- size: `20812715`
- SHA-256: `2be88c6f92a06c3eaffdb47d3a6d3fbf159fbc057e9d272d592fb194e41fefab`
- Nix SRI: `sha256-K+iMb5KgbD6v/bR9Om0/vxWfvAV+nSctWS+xlOQf76s=`

### Linux aarch64

- GitHub release asset ID: `404891905`
- asset: `mujoco-3.8.0-linux-aarch64.tar.gz`
- size: `20674425`
- SHA-256: `adc4a7856d2b8d42ba4e889b57cbceb13a329c869f410cf1ad110b153c4745e4`
- Nix SRI: `sha256-rcSnhW0rjUK6ToibV8vOsToynIafQQzxrRELFTxHReQ=`

The derivation disables normal Nix fixup so the qualification materializer can distinguish the exact official release payload from any later derived runtime closure.

## Hermetic mujoco-rs rule

`mujoco-rs 4.0.1` checks `MUJOCO_DYNAMIC_LINK_DIR` before its pkg-config / `auto-download-mujoco` fallback. Therefore the later hermetic MuJoCo profile must bind:

```text
MUJOCO_DYNAMIC_LINK_DIR=/deps/mujoco-3.8.0/lib
```

from a read-only QUAL-INFRA-003D materialization.

The materializer must not set `MUJOCO_DOWNLOAD_DIR`, and QUAL-INFRA-003C/003E should prove outbound network is unavailable during subject execution. A successful build under those conditions is stronger evidence that the explicit materialized runtime was used instead of the crate's automatic download path.

## Required 003D manifest fields

For MuJoCo-enabled profiles, the materialization manifest should bind at least:

- `mujoco-rs` package/version identity;
- MuJoCo release tag and upstream commit;
- platform architecture;
- release asset ID, name, size, URL and SHA-256;
- Nix derivation/source identity;
- extracted MuJoCo tree digest;
- exact `lib/libmujoco.so` digest;
- dependency closure/runtime libraries needed by that shared object;
- guest path `/deps/mujoco-3.8.0`;
- environment binding `MUJOCO_DYNAMIC_LINK_DIR=/deps/mujoco-3.8.0/lib`.

## Migration rule

This pin does not silently replace the historical 3.3.7 derivation. Existing consumers remain unchanged until separately reviewed.

A future compatibility repair may make 3.8.0 the default MuJoCo runtime for current `mujoco-rs 4.0.1` consumers, but that is a separate proposition from defining this qualification input.

## Nonclaims

```text
pinned upstream asset
    != Nix build PASS

Nix build PASS
    != complete 003D materialization PASS

003D materialization PASS
    != 003C sandbox-containment PASS

003C + 003D
    != exact-subject qualification automatically

source qualification
    != D4/D5C/QP authority
    != hardware observation
    != physical qualification
```
