# BASALT coverage envelope and profile contract — 2026-10-31

## Problem

The existing per-case result adjudicator checks whether observer receipts agree with an independently supplied scenario oracle. It does not establish that the observer's capture covered every relevant channel for the run. A producer can accurately sign or hash a partial receipt and still overstate what it observed.

The coverage envelope adds a separate, run-level claim whose scope is bound to a distinct profile file. The validator treats all receipt fields as untrusted claims. Hashes bind exact bytes; they do not prove that the observer emitted those bytes truthfully.

## New artifacts

- Profile: `benchmarks/basalt/coverage/linux-hermetic-candidate.v1.json`
- Profile schema: `benchmarks/basalt/schema/observer-coverage-profile.v1.schema.json`
- Receipt schema: `benchmarks/basalt/schema/observer-coverage-envelope.v1.schema.json`
- Auditor: `xtask/src/basalt_coverage.rs`
- CLI: `basalt-coverage-envelope-audit --profile PROFILE.json --receipt ENVELOPE.json`

The profile is explicitly `candidate_not_qualified`. It scopes Linux x86_64/aarch64 and names nine channels: filesystem, process, environment, generated artifacts, toolchain, cache, network, observer integrity, and snapshot consistency. It also lists the capture controls required before any complete-within-profile statement can be accepted.

## Coverage claim contract

Each required channel appears exactly once in the envelope. Each required event kind is classified exactly once as:

- `observed`: requires a SHA-256 commitment to retained observation evidence and no unresolved reason.
- `unresolved`: requires a reason and must not claim an observation digest.
- `excluded`: requires a reason and must not claim an observation digest.

Missing or duplicate channel records, missing or duplicate event classifications, unprofiled channels/event kinds, malformed digests, root-path traversal, and profile-byte mismatch make the envelope invalid or unbound.

A `complete_within_profile` claim is consistent only when all required event kinds are observed, all required controls are active, platform constraints match, kernel build identity and BTF digest are present when required, the capture interval is increasing, the process tree is closed, the snapshot is stable, monitored roots are explicit, and all event-loss/overflow/restart/unmonitored-child/root-escape counters are zero.

An honest `incomplete` or `unsupported` claim is reportable but never becomes complete. Unsupported platform claims remain explicit; they cannot satisfy the complete gate.

## Three separate conclusions

The JSON report intentionally separates:

1. **Receipt well-formedness** — strict structure and internally consistent classification.
2. **Coverage completeness within the candidate profile** — the supplied claim satisfies the pinned profile's mechanical conditions.
3. **Qualification and authenticity** — permanently false in this tool.

A successful audit does **not** prove the operating system emitted every event, that the evidence digest resolves to authentic kernel observations, that a signature belongs to a trusted observer, or that the profile itself has passed qualification. It only prevents an envelope from declaring complete when its own profile-bound classifications, controls, platform identity, capture health, and lifecycle fields contradict that claim.

The auditor is a policy consistency check, not a kernel observer, cryptographic signature verifier, event recorder, sandbox, or independent trust root.

## Fail-closed tests

The new module includes tests for: a self-consistent bounded complete claim; missing and duplicate event classifications; changed profile bytes with unchanged profile ID; event loss and ring-buffer overflow; observer restart; unmonitored children and root escape; unstable snapshots and open process trees; invalid capture intervals and path traversal; unknown channels; missing required controls; missing kernel BTF identity; and honest incomplete/unsupported downgrades.

These are authored unit tests only. They have not been compiled or executed in this environment.

## Industry positioning

This layer is designed to compose with existing systems rather than replace them:

- Cargo's build-script dependency declarations define its incremental invalidation inputs.
- Bazel's action execroot and sandbox provide action input isolation and are the strongest direct hermetic-execution comparison.
- Nix's store closure and Linux namespace sandbox provide dependency closure and isolation.
- SLSA/in-toto provide builder identity, provenance, authorized workflow and integrity semantics.
- Reproducible builds compare independently recreated artifacts.

BASALT's proposed addition is a falsifiable coverage envelope over a bounded execution profile, with explicit unresolved channels and loss state, joined to independently authored scenario oracles. The coverage envelope does not supersede those systems and must not claim their guarantees. The benchmark should compare composition arms under identical input snapshots, task lineages, toolchains, mutation schedules and resource limits.

## Next qualification steps

1. Implement a real Linux capture producer with process-tree supervision, immutable snapshot roots and kernel/runtime identity collection.
2. Independently attach an observer (candidate lanes include BPF LSM/audit-style hooks and a separate sandbox access log); record exact hook attach set, buffer sizes and loss counters.
3. Build a ground-truth capture lane that the observer cannot author or modify.
4. Cross-check filesystem/process/environment events against a separate replay and tool-native dependency metadata. Disagreement must downgrade the run to incomplete.
5. Sign the final envelope using an independent trust root and domain-separated, expiry-bound receipt format. Verify currentness/revocation separately.
6. Qualify only after adversarial, held-out task-lineage testing and all non-compensatory safety gates pass.

The environment used for this implementation does not have Cargo, rustc, rustfmt, strace, bpftrace, bpftool, or a usable BTF kernel file. Therefore no Rust compilation, test execution, live kernel observation, or observer qualification is claimed.
