# Nixward CROSS-072 — Sealed NixOS Current Generation V1

## Purpose

CROSS-072 removes the last caller-supplied element from the service pre-state provenance chain: the NixOS generation number.

NixOS stores the system configuration history in the system profile `/nix/var/nix/profiles/system`, and generation links use the `system-N-link` form. The official NixOS documentation describes the system profile and generation switching through this profile. citeturn352939search0turn352939search5

## Observer

`NixVerifiedNixOSGenerationV1` is created only by the read-only generation observer.

The observer:

1. reads `/nix/var/nix/profiles/system` with `read_link`;
2. reads it again immediately;
3. requires both link targets to be identical;
4. accepts only a target whose filename is exactly `system-<nonzero-u64>-link`;
5. retains the profile path and observed target for provenance.

The value is not serializable and is not publicly constructible.

## Service pre-state binding

`NixVerifiedServicePreStateV1` now requires the verified generation token rather than a raw integer.

The token therefore binds:

- exact service state digest;
- exact canonical service unit;
- observer-originated NixOS generation;
- canonical generation-bound pre-state identity.

`NixServiceEffectAdmissionV1` consumes that sealed pre-state and no longer accepts raw pre-state identity, generation, or state-digest strings.

## Cross-checking

The systemd observer captures the NixOS generation both before and after the service-state D-Bus snapshot.

A generation change during the observation causes fail-closed rejection.

Likewise, the systemd manager unique owner is checked across the same observation.

Thus the trusted pre-state relation becomes:

`NixOS generation + systemd manager epoch + service Unit identity/state -> sealed pre-state`

## Claim ceiling

This is not an atomic cross-subsystem snapshot.

A generation can still change after the final read, just as systemd state can change after the observation. The token proves only that the two sampled generation reads agreed during the bounded observation.

It also does not cryptographically attest the NixOS profile symlink.

Those limits are intentional and remain explicit.

## Relationship to the hardening chain

- CROSS-060 binds the service effect context into authorization.
- CROSS-068 observes exact definition content.
- CROSS-069 binds content provenance into authorization identity.
- CROSS-070 fixes lifecycle transport sequencing.
- CROSS-071 seals the service pre-state itself.
- CROSS-072 removes caller-supplied NixOS generation from that pre-state token.

## Qualification

Exact-head GitHub Actions remain the qualification authority. Queued, pending, cancelled, mergeable, or static-only states are not PASS evidence.