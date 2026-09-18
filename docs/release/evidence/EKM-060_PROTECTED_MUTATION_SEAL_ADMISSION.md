# EKM-060 — Protected mutation-evidence-seal admission

## Status

Draft implementation. GitHub Actions remains the executable authority. No format, compile, Clippy, test, runtime, cryptographic-provider, or deployment PASS is inferred from static review or queued CI.

## Purpose

EKM-058 and EKM-059 intentionally prove different things:

- EKM-058 proves that a supplied EKM-057 sidecar is internally and cross-component consistent with one restart-v2 history, but cannot independently prove that historical non-basis evidence was not omitted.
- EKM-059 proves that an external deployment verifier accepted the exact original EKM-056 seal-capsule digest in the exact restart lineage, but does not prove that a later supplied sidecar reproduces that digest.

EKM-060 composes those two boundaries without constructing writable state.

The central invariant is:

> A supplied mutation-evidence sidecar is admitted as equivalent to the protected source census only when the exact restart receipt re-verifies, EKM-058 independently re-derives the complete sidecar capsule digest, and that digest exactly equals the EKM-059 externally protected source digest.

## Validation sequence

`ProtectedMutationSealAdmissionV1::validate_and_admit`:

1. re-runs the EKM-044 restart validation receipt against the exact restart-v2 snapshot;
2. re-verifies EKM-059's protected checkpoint internally;
3. requires observation at or after EKM-059 verification and before its expiry;
4. independently re-runs EKM-058 over the exact restart snapshot and exact EKM-057 sidecar;
5. requires EKM-058 to remain conservative on its own (`historical_census_completeness_independently_proven = false`);
6. rejects any unexpected construction, hydration, activation, or trusted-state authority;
7. requires exact restart/mutation/seal capture-epoch agreement;
8. requires mutation counts to agree across the restart history, sidecar, EKM-058 report, and protected EKM-059 statement;
9. requires the exact restart validation receipt digest to equal the protected EKM-059 receipt digest;
10. requires the exact restart-v2 digest to agree across the snapshot, receipt, and protected statement;
11. requires EKM-058's independently recomputed seal-capsule digest to equal the sidecar claim;
12. requires that independently recomputed digest to equal the externally protected EKM-059 seal-capsule digest.

## Admission receipt

The resulting immutable receipt binds:

- restart capture cycle;
- seal capture cycle;
- mutation and sealed-evidence counts;
- restart validation receipt digest;
- restart-v2 digest and outer checksum;
- seal wire checksum;
- independently recomputed EKM-056 seal-capsule digest;
- protected EKM-059 checkpoint statement digest and proof digest;
- admission observation cycle;
- all claim-boundary booleans.

The receipt is itself domain-separated and BLAKE3 hashed and can re-run the complete validation through `verify_against`.

## Three distinct claims

EKM-060 deliberately keeps three questions separate.

### 1. Cross-component consistency

`cross_component_consistent = true`

The supplied sidecar and restart history reproduce the same EKM-056 semantic capsule digest under EKM-058's independent validation rules.

### 2. Protected source equivalence

`protected_source_capsule_equivalent = true`

The independently recomputed semantic capsule digest exactly equals the seal-capsule digest accepted by EKM-059's external protection provider for the same restart lineage.

`historical_census_completeness_protected = true` is therefore a **conditional protected-source claim**: EKM-056's source capsule was complete by construction, EKM-059 protected its exact digest, and the supplied sidecar reproduces that exact digest. It is not a claim of metaphysical truth and remains subject to the configured external-provider and hash assumptions.

### 3. Rollback/currentness

`protected_checkpoint_currentness_independently_proven = false`

EKM-060 does not claim that the EKM-059 checkpoint is the latest deployment checkpoint. A valid old signature/checkpoint may still be authentic. Currentness requires the separate EKM-059 continuity/protected-state story or a deployment provider whose verified semantics explicitly include monotonic current-state protection.

This separation prevents an authentic historical checkpoint from being mislabeled as the current checkpoint.

## Authority boundary

Successful EKM-060 admission still reports:

- `capsule_construction_authorized = false`;
- `writable_hydration_authorized = false`;
- `activation_authorized = false`.

It does not:

- construct an EKM-056 capsule from wire data;
- reconstruct a mutation-time ledger;
- write an `EpistemicSupportStore` or `BeliefRevisionHistory`;
- swap live state;
- advance anchors/checkpoints;
- mutate legacy confidence;
- mutate causal/world-model/action state;
- perform file/network I/O or key custody.

## Next boundary

Before EKM-055 historical replay is widened, the next tranche should make **checkpoint currentness/freshness explicit**. A narrow option is an EKM-061 currentness receipt that accepts either:

- a verified EKM-059 continuity chain anchored in the deployment's trusted latest checkpoint; or
- an external provider profile that explicitly attests monotonic current-state semantics.

Only after protected source equivalence **and** currentness are independently represented should a later layer reconstruct mutation-time ledger projections for isolated firewall replay.
