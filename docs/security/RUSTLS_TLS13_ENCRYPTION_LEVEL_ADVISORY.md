# Rustls TLS 1.3 Encryption-Level Advisory

**Reviewed:** 2026-10-22  
**Upstream advisory:** GHSA-2mjx-qc3c-rqvc / RUSTSEC-2026-0285  
**Severity:** Moderate (upstream)  
**Fix:** rustls 0.23.45 or later; this branch locks 0.23.46

## Finding

The repository's base-branch lockfile resolved `rustls 0.23.40`, which is within the upstream affected range `0.23.13–0.23.44`. Rustls accepted some TLS 1.3 handshake messages at the wrong encryption level when they followed a key-changing message in the same record. The transcript remains authenticated; upstream describes the practical effect as a peer being able to send handshake messages in plaintext where encrypted ones should be expected, without rustls rejecting the connection. This is not described as an unauthenticated transcript modification.

Upstream references:
- https://github.com/rustls/rustls/security/advisories/GHSA-2mjx-qc3c-rqvc
- https://github.com/rustls/rustls/releases/tag/v%2F0.23.45
- https://rustsec.org/advisories/RUSTSEC-2026-0285.html

## Change in this PR

The only runtime dependency change is the lockfile resolution of `rustls` from `0.23.40` to `0.23.46`, with the upstream crate checksum and dependency list. This is above the patched minimum 0.23.45 and avoids an unnecessary application-layer TLS workaround.

## Scope

The pinned Iroh fork's manifests declare rustls in the 0.23 semver series, so the committed lockfile controls the resolved rustls release for repository builds using this lockfile. A separate check is still required for any downstream published library build that ignores the root lockfile or resolves its own dependency graph.

## Validation and acceptance

- Static inspection confirms the base branch locked the affected 0.23.40 and this branch locks 0.23.46.
- CI must verify Cargo lockfile resolution, affected transport build targets and tests on the exact commit.
- No runtime test or build pass is claimed by this document.
- Keep upstream advisory monitoring in the dependency audit; do not treat one patched lockfile as a permanent assertion that the entire transport stack is vulnerability-free.
