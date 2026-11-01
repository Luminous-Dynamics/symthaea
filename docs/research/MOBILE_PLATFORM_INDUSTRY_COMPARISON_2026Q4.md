# Mobile Platform Industry Comparison — 2026 Q4

**Status:** Research and benchmark protocol; no head-to-head measurements are claimed.  
**Related:** [MOBILE-BENCH-001 / #9799](https://github.com/Luminous-Dynamics/symthaea/issues/9799), [MOBILE-OS-001 / #9772](https://github.com/Luminous-Dynamics/symthaea/issues/9772), [HARDSEC-029 / #9771](https://github.com/Luminous-Dynamics/symthaea/issues/9771).

## 1. Competitive landscape

### AOSP: the integration substrate

Current AOSP documentation supports native Rust OS components and a Rust AIDL backend. The Rust backend is built on `libbinder_ndk`; AIDL interfaces can generate Rust libraries used by Rust modules and APEX packaging. The backend was introduced in Android 12. This means a project can create Rust-based system-side services without rewriting the Android framework.

References:
- [Android Rust introduction](https://source.android.com/docs/setup/build/rust/building-rust-modules/overview)
- [AIDL backends](https://source.android.com/docs/core/architecture/aidl/aidl-backends)

### Google Android/Pixel: the production Rust and firmware reference

Google's 13 November 2025 Android Rust report states that memory-safety vulnerabilities fell below 20% of total Android vulnerabilities in its 2025 data. It reports approximately 1000x lower memory-safety vulnerability density in Rust changes than Android C/C++, 4x lower rollback rate, and 25% less code-review time in its measured population. These are vendor-reported figures using Google's own populations and methodology; they are not direct predictions for a small external project.

Reference: [Google Security Blog — Rust in Android: move fast and fix things](https://blog.google/security/rust-in-android-move-fast-fix-things/).

On 10 April 2026, Google described placing a Rust DNS parser based on Hickory-proto into Pixel 10 modem firmware. Google reports a combined parser/library/runtime code footprint of approximately 371 KB and discusses extending `no_std` support to the library and dependencies. This is an unusually relevant precedent for a staged migration of a small, exposed parser rather than a whole-system rewrite.

Reference: [Google Security Blog — Bringing Rust to the Pixel Baseband](https://blog.google/security/bringing-rust-to-the-pixel-baseband/).

### GrapheneOS: the direct hardening reference

GrapheneOS publishes Android 17-based builds and officially supports Pixel 8 Pro. Its release history includes kernel/GKI changes and security fixes, and its FAQ documents the role of Titan/Weaver and the device's hardware-backed key derivation and rate-limiting model.

References:
- [GrapheneOS releases](https://grapheneos.org/releases)
- [GrapheneOS FAQ](https://grapheneos.org/faq)
- [GrapheneOS features](https://grapheneos.org/features)

This means the Pixel 8 Pro is not merely a convenient test phone: it can support a controlled comparison of stock Pixel software, GrapheneOS and a future Luminous AOSP-derived build on the same hardware. Firmware, security patch level and boot configuration still need to be recorded for every run.

### Fuchsia: a component and capability architecture reference

Fuchsia is useful for studying component isolation, capability routing and driver boundaries. It should be treated as an architectural research reference, not as a ready-made Android replacement or as proof that its entire driver ABI is production-stable.

Reference: [Fuchsia driver framework](https://fuchsia.dev/fuchsia-src/development/drivers).

### Academic migration evidence

FSE Companion '26 published *An Empirical Analysis of Rust Integration in Android Open Source Project*. The authors analyzed 16 quarterly AOSP snapshots from January 2021 through October 2024, mapped Rust integration across 67 components, related security vulnerabilities to component history, and examined unsafe-code governance.

References:
- [ACM DOI: 10.1145/3803437.3805227](https://doi.org/10.1145/3803437.3805227)
- [FSE 2026 industry paper programme](https://conf.researchr.org/details/fse-2026/fse-2026-industry-papers/26/An-Empirical-Analysis-of-Rust-Integration-in-Android-Open-Source-Project)

The study is a strong source for migration methodology, but its source snapshots end in October 2024; later Android platform changes need separate verification.

## 2. What we should compete on

Do not try to beat Google at vendor hardware integration or GrapheneOS at mature Android hardening by announcing a Rust rewrite. Their advantages include long-lived engineering teams, production update pipelines, device-specific integration and mature test infrastructure.

Luminous should compete on a different, complementary axis:

**Declarative assurance with machine-checkable evidence obligations across the full device lifecycle.**

The proposed differentiators are:

1. **Evidence-carrying security policy:** each requirement maps to a platform enforcement point and an independent verification obligation.
2. **Proof of non-knowledge:** unsupported, unobservable, stale or missing inputs remain explicit and cannot create clean verdicts.
3. **Residual-risk allocation:** HARDSEC conditions detection on independently qualified prevention, instead of rewarding detection of attacks already blocked by platform controls.
4. **SOMA's user-controlled embodiment:** local-first sensing, explicit sharing modes, energy-aware operation and measurable privacy boundaries.
5. **Risk-weighted memory-safety migration:** target reachable untrusted-input surfaces in privileged components first, not the biggest C/C++ directories.
6. **Lifecycle re-attestation:** re-evaluate security posture after reboot, OTA, firmware change, key rotation, application update and instrumentation changes.

This is a proposed research contribution, not an established superiority claim.

## 3. Same-device benchmark matrix

Use the Pixel 8 Pro for the first controlled comparison:

| Configuration | What it establishes |
|---|---|
| Stock Google Pixel OS | OEM baseline, update path and default protections |
| Current GrapheneOS | Hardened AOSP baseline on same hardware |
| Luminous AOSP-derived build | Incremental platform changes, one verified component at a time |
| Luminous + SOMA | Additional runtime value and cost |
| Cuttlefish/emulator | Reproducible policy/service tests only; never pool performance with physical devices |

Pin:
- device model and SoC;
- bootloader and Verified Boot state;
- OS build fingerprint digest and security patch level;
- vendor image, boot/kernel/GKI and modem firmware;
- build manifest and source revisions;
- signer, build flags and artifact digests;
- temperature, battery health, charge state, display/radio conditions;
- test harness, policy and verifier versions.

Never imply that comparing different firmware, patch levels or device models isolates the operating system effect.

## 4. Separate scorecards

### Platform prevention

Measure Verified Boot, rollback resistance, KeyMint/StrongBox security level, MTE/PAC/BTI process coverage, allocator hardening, SELinux policy, Binder caller authorization, service isolation, USB and radio boundaries, OTA interruption/recovery, and patch latency.

### Memory-safe migration

Measure reachable untrusted-input paths by privilege and component, unsafe Rust block/LOC inventory, independent review coverage, fuzzing hours, unique crashes, differential mismatches, memory-safety defects, code size, latency, memory and power cost.

Prefer a risk-weighted attack-surface metric over Rust LOC percentage. A rewritten 350 KB modem DNS parser at an exposed trust boundary can be more valuable than a large rewrite of non-privileged code.

### Evidence assurance

Proposed gates:
- 100% of mandatory controls map to an enforcement point and verifier obligation, or explicit unsupported/unobservable.
- 0 false passes under missing, stale, replayed, source-substituted, scope-mismatched, version-skewed or digest-mismatched evidence.
- 100% of independently passed claims bind to exact device profile, evidence, policy and verifier identities.
- 0 claim upgrades from synthetic fixtures, vendor documentation, ADB properties or CPU feature strings alone.
- 0 missing measurements coerced to safe/zero/false.
- Re-attestation latency after reboot, OTA and runtime monitor changes is measured.

The current HARDSEC-029 compiler intentionally generates obligations but never declares a security control satisfied. A distinct verifier and authenticated receipt format are required before any such claim is allowed.

### Privacy and authority

Measure hardware-identifier access, sensor permission denial/revocation, background collection, network egress under offline/no-share mode, retention/export and IPC access. Every unauthorized operation is a hard failure, irrespective of average accuracy.

### User impact and reliability

Measure boot/startup time, p50/p95 latency, idle drain, energy per successful task, peak and steady memory, CPU, thermal throttling/recovery, process crash/restart, app compatibility, CTS/VTS, OTA success and update latency.

## 5. Initial qualification targets

Targets for research planning—not industry standards or current results:

- no CTS/VTS regression attributable to migrated components;
- zero policy false-passes in the frozen mutation suite;
- at least 20% fewer reachable memory-unsafe input-handling paths in the migrated scope, without increasing privilege or adding reachable attack surface;
- at least 1,000 CPU-hours of fuzzing per migrated parser before a production recommendation, with no unresolved high-severity crash;
- no more than 5% p95 latency regression and 2% measured incremental energy cost for SOMA security monitoring on the matched workload, unless a user-visible tradeoff is explicitly accepted;
- patch, OTA, recovery and rollback behavior no worse than the selected baseline;
- independent review and exact-head evidence for every release claim.

Do not collapse prevention, attestation, detector quality, measurement quality, privacy, maintenance and UX into one leaderboard score.

## 6. Immediate experiments

1. Run policy compiler mutation fixtures against the synthetic Pixel profile and prove it always emits abstention rather than a security pass.
2. Run the corrected read-only HARDSEC-028 acquisition on the physical Pixel 8 Pro, locally and without exporting stable identifiers.
3. Capture stock/GrapheneOS configuration metadata only after the user intentionally installs/boots each image and with a reversible recovery path.
4. Build a Cuttlefish image with one minimal Rust AIDL service and measure its caller-identity, SELinux, restart and compatibility contracts.
5. Pick one small, reachable parser in a Luminous component and run old/new differential, fuzz and performance comparison before considering Android system integration.

The project may ultimately deliver a more rigorous and transparent security assurance system than existing configuration-only approaches. It has not yet demonstrated better overall OS security than AOSP or GrapheneOS.
