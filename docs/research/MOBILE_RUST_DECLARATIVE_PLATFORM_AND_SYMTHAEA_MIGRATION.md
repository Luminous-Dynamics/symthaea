# Declarative Rust-First Mobile Platform: SOMA, HARDSEC, and Symthaea-Assisted AOSP Evolution

**Status:** Research architecture and staged implementation proposal  
**Repository baseline inspected:** `main` at `36b2105811200dd60b2d9eed1177f819e7ac9467` (2026-11-01)  
**Work items:** [HARDSEC-029 / #9771](https://github.com/Luminous-Dynamics/symthaea/issues/9771), [MOBILE-OS-001 / #9772](https://github.com/Luminous-Dynamics/symthaea/issues/9772), [SOMA-SEC-001 / #9773](https://github.com/Luminous-Dynamics/symthaea/issues/9773)  
**Related:** [MOBILE-DEVICE-001 / #8413](https://github.com/Luminous-Dynamics/symthaea/issues/8413), [HARDSEC-027 / #9736](https://github.com/Luminous-Dynamics/symthaea/issues/9736), [HARDSEC-028 / #9739](https://github.com/Luminous-Dynamics/symthaea/issues/9739)

## 1. Executive decision

Build a declarative, Rust-first **AOSP-derived** mobile platform in stages. Do not begin by rewriting all of Android, replacing Linux, or discarding the vendor hardware stack.

The near-term platform should combine:

1. **AOSP and its supported hardware interfaces** for the boot, kernel, vendor, modem, graphics, telephony, camera and application compatibility that already exists.
2. **Rust for new security-sensitive native services, libraries and bounded components**, using AOSP's supported Rust build and Binder/AIDL integrations.
3. **SOMA / Spore** as the mobile embodiment and on-device cognitive runtime, with its FFI, sensor admission, transport and lifecycle boundaries hardened before it gains privileges.
4. **HARDSEC** as the independent device evidence, measurement qualification, policy evaluation and residual-threat assurance plane.
5. **A declarative policy compiler** that converts security intent into target-specific enforcement and verification obligations.
6. **Symthaea-assisted migration** as a research workflow for inventory, modeling, candidate generation, differential testing and audit—not as an autonomous security authority.

A future microkernel/capability-OS lane can be explored independently. It must not block the useful AOSP-derived system.

## 2. What "rewrite Android in Rust" should mean

There are at least four distinct rewrite scopes, and conflating them creates a false programme.

| Scope | Practicality | Recommended posture |
|---|---|---|
| SOMA and Luminous mobile services | High | Rust-first now; remove unnecessary handwritten unsafe boundaries |
| New AOSP native services and libraries | High where supported | Rust-first; use official AOSP build and stable AIDL paths |
| Existing framework, HAL, kernel and driver components | Component-dependent | Incremental migration with ABI, vendor and conformance evidence |
| Entire Android framework, Linux kernel, all drivers and firmware | Not a credible first milestone | Research only; do not make this the product dependency |

Android's Rust integration is already substantial. The official AOSP documentation describes Rust platform modules and a stable Rust AIDL backend. Rust AIDL is built on `libbinder_ndk`; the Rust portion can be bundled and shipped in an APEX. Stable AIDL is the preferred direction for new HAL interfaces where upstream support exists. Sources:

- [AOSP: Android Rust introduction](https://source.android.com/docs/setup/build/rust/building-rust-modules/overview)
- [AOSP: AIDL backends](https://source.android.com/docs/core/architecture/aidl/aidl-backends)
- [AOSP: AIDL for HALs](https://source.android.com/docs/core/architecture/aidl/aidl-hals)

Google's November 2025 Android Rust report says Rust changes had approximately 20% fewer revisions, 25% less code-review time and roughly 4x lower rollback rate for medium and large changes in its measured Android engineering population. It also reports a modeled memory-safety vulnerability-density estimate of 0.2 per million lines for roughly five million Rust LOC, compared with about 1,000 per MLOC for historical Android C/C++ code. These are Google's own measurements and estimates over its specific populations and methodology; they are a benchmark reference, not an automatic prediction for our code. The same report documents a pre-release near-miss in unsafe Rust and emphasizes defense in depth and careful unsafe review.

Source: [Google Security Blog, Rust in Android: move fast and fix things, 2025-11-13](https://blog.google/security/rust-in-android-move-fast-fix-things/).

The Linux kernel is a separate boundary. Rust support has been merged upstream, but current kernel documentation still describes the support as experimental and says in-tree Rust drivers/modules are not yet intended for production use. Google separately reports that Android's 6.12 kernel is its first with Rust support enabled and its first production Rust driver. These facts are compatible: a downstream Android product can ship a Rust driver while the general upstream in-tree driver ecosystem remains limited. We must qualify the precise kernel branch and driver rather than assume all kernels support the same production Rust surface.

Sources:
- [Linux kernel Rust documentation](https://android.googlesource.com/kernel/common/+/android-mainline-tracking/Documentation/rust/index.rst)
- [Google's Android Rust report](https://blog.google/security/rust-in-android-move-fast-fix-things/)

## 3. SOMA is our existing mobile substrate—not a greenfield substitute

The repository already contains:

- `symthaea-spore`: the smaller Rust cognitive runtime.
- `symthaea-soma`: the mobile embodiment crate, with a Rust library and Android Kotlin/JNI integration.
- Android sensor, haptic, metabolism, BLE, screen-vision and Holon bridge paths.
- `symthaea-phone-embodiment`: Android screen acquisition and observation support.
- Pixel 8 Pro build/deployment scripts and a Nix mobile development shell.
- `docs/HOLON_SOMA_ROADMAP.md`: earlier phone/desktop transport and observation research.

The existing architecture is valuable, but it must be treated as prior work with bounded evidence, not as a current security certification. The April 2026 remote-display roadmap itself records limitations: small physical-frame sample counts, an out-of-band placeholder key in the initial RDP path, and the need for later transport qualification. Historical measurements should be repeated against exact current artifacts before being quoted as current results.

### 3.1 Current-source findings requiring remediation

Reviewed at repository baseline `36b2105811200dd60b2d9eed1177f819e7ac9467`.

**Native FFI — `crates/domains/symthaea-soma/src/native_ffi.rs`**

- 61 exported C ABI functions were present; 56 are declared `unsafe extern "C"`.
- Engine instances cross the JNI boundary as raw pointers encoded in `jlong`.
- `soma_engine_cycle` dereferences the engine pointer directly. A null or invalid handle is undefined behavior under this API contract.
- `soma_engine_free` uses `Box::from_raw` and drops the allocation. Its safety comment says the first eight bytes are overwritten by a sentinel, but the implementation does not perform that operation.
- Thread safety is imposed as a caller obligation rather than enforced by a handle registry or API design.
- Pointer/length, C-string, output-buffer and panic-boundary rules need to be audited function by function.

This finding does not establish that an attacker can remotely supply an invalid pointer. It establishes that the native boundary has a broad unsafe contract whose lifecycle and failure behavior need to be made much harder to misuse.

**Holon bridge — `crates/domains/symthaea-soma/src/holon_bridge.rs`**

- `HolonBridge` stores an optional 32-byte `session_key` and exposes a setter.
- The current message queue still serializes outbound messages as ordinary JSON; the stored key is not used by this queue to encrypt or authenticate them.
- The source comment describes key establishment as future work.
- Do not report the bridge as end-to-end encrypted merely because a key field exists.
- The separate Holon RDP path has its own roadmap and historical placeholder-key caveat. It must be audited independently and must not be conflated with HolonBridge.

**Pixel build workflow — `scripts/build-soma-pixel8.sh`**

The script currently resolves Android assets from `crates/symthaea-soma/android`, but the current crate lives under `crates/domains/symthaea-soma/android`. The path should be corrected and validated from a clean checkout.

These are tracked in [SOMA-SEC-001 / #9773](https://github.com/Luminous-Dynamics/symthaea/issues/9773).

## 4. Proposed architecture

```text
            Declarative Mobile Security Policy
                 (versioned, typed, signed)
                              |
                              v
                 Rust Policy Compiler
            / compatibility / obligations \
           v                               v
   AOSP Build Realization          Evidence Plan
   Soong / APEX / AIDL             HARDSEC-027 profile
   kernel/vendor constraints       attestation / sensor
           |                       / measurement needs
           v                               |
   AOSP-derived image                       |
           |                               |
           v                               v
   Verified Boot / GKI              Independent Verifiers
   vendor HAL / firmware            provenance / freshness
   Rust system services             metrology / claim ceiling
           |                               |
           +---------------+---------------+
                           |
                           v
                  SOMA / Spore runtime
             sensor admission / energy policy
             local cognition / user-controlled sync
                           |
                           v
                       HARDSEC
           detection / abstention / residual risk
                           |
                           v
                Auditable decision receipt
```

### 4.1 Declarative policy is desired state plus proof obligations

A policy file must not pretend that every desired control is an OS knob. Each requirement has to state:

- what is desired;
- which device/platform scope it applies to;
- which layer can enforce it;
- which evidence can verify it;
- how fresh and trustworthy that evidence must be;
- what happens when the control is unsupported or unobservable.

The first language can be TOML or JSON for operator readability, but its semantic model should be typed Rust enums and structs. The schema should distinguish at least:

- hardware capability;
- build-time configuration;
- runtime enablement;
- protected process/service scope;
- cryptographic attestation;
- measurement qualification;
- desired versus observed state;
- enforcement point;
- evidence source and claim ceiling.

The evaluator must never promote vendor documentation, CPU feature strings, ADB properties, or an event-name inventory into runtime enforcement or verified attestation.

### 4.1.1 Declarative build envelope: Nix around Soong, not instead of Soong

Use Nix to pin and reproduce the *build environment*:
- host compiler/toolchain packages, Java/JDK, Android SDK/NDK where applicable, repo client and required utilities;
- exact AOSP `repo` manifest URL, revision and resolved project SHAs;
- Rust toolchain and crate dependencies for Luminous-owned components;
- build flags, environment variables, locale and source-date configuration;
- SBOM, provenance metadata, build logs and artifact hashes.

Keep AOSP's own `repo` manifest, Soong/Ninja graph, platform-specific generators, signing, image assembly and vendor build steps authoritative. Nix should orchestrate a pinned environment and wrap the build, not replace Soong or silently reimplement vendor integration.

A release candidate should bind:
`Nix flake lock + repo manifest digest + source tree digests + toolchain digests + build configuration digest + output image digests`.

The resulting build is reproducible only to the extent that the complete toolchain and all inputs are captured and the output is independently rebuilt and compared. A lockfile alone is not a reproducible-build proof.



The compiler should emit an **obligation graph**, not just a pass/fail bit:

`policy -> required configuration -> acquisition -> verifier -> claim -> residual risk`

Every control should resolve to one of:
- verified as required;
- verified contrary to policy;
- supported but not measured;
- unsupported by target;
- not enforceable at the selected layer;
- unobservable under current privilege;
- evidence invalid, stale or inconclusive.

This is the first concrete deliverable in [HARDSEC-029 / #9771](https://github.com/Luminous-Dynamics/symthaea/issues/9771).

### 4.2 Rust-first platform layers

| Layer | First implementation strategy | Boundary to retain |
|---|---|---|
| Policy compiler/evaluator | Rust | None beyond serialized schema |
| SOMA engine and security services | Rust | Minimal JNI/platform adapter while needed |
| Android system-side service | Rust AIDL service | Android system APIs and framework contracts |
| HAL service with stable AIDL | Rust where upstream/device support exists | Vendor ABI and proprietary implementation |
| Kernel subsystem/driver | Only supported Rust surface and narrowly scoped work | Existing Linux/GKI and C/vendor drivers |
| Bootloader / TEE / secure element | Consume and verify vendor/hardware evidence first | Proprietary firmware and roots of trust |
| UI and app integration | Existing Kotlin/Java where Android APIs require it | Framework lifecycle, permissions and compatibility |

Android's Project Treble, partition layout, VINTF and stable interfaces exist to separate generic OS updates from hardware-specific vendor code. Mainline modules can update selected system components independently through stable interfaces. We should preserve these contracts and migrate bounded components around them rather than fork every hardware layer.

Sources:
- [AOSP partition architecture](https://source.android.com/docs/core/architecture/partitions)
- [AOSP Mainline modular system](https://source.android.com/docs/core/ota/modular-system)
- [AIDL HAL guidance](https://source.android.com/docs/core/architecture/aidl/aidl-hals)

### 4.3 What to learn from other OS architectures

- **GrapheneOS:** the hardening reference for the first Pixel-based comparison. It builds on AOSP and focuses on reducing attack surface, exploiting hardware mitigations, improving sandboxing, privacy controls, patching and device-specific hardening. Do not assume a Rust rewrite alone beats its security posture.
- **Fuchsia:** the capability-oriented component and driver model is an architectural reference for service isolation and driver boundaries. Its published driver framework currently says the ABI is still evolving and not yet stable, so do not present it as a ready-made drop-in mobile hardware platform.
- **AOSP:** provides the practical baseline for real device support, updates, app compatibility and vendor integration.

Sources:
- [GrapheneOS features](https://grapheneos.org/features)
- [Fuchsia driver framework](https://fuchsia.dev/fuchsia-src/development/drivers)

## 5. Symthaea-assisted operating-system migration

Symthaea can help make large migration work tractable if it is used as an evidence-producing engineering orchestrator.

It should not be described as a magic compiler that can safely rewrite Android wholesale. The actual implementation loop still needs compiler diagnostics, static analysis, established migration tooling, fuzzers, build systems, hardware tests and human review.

### 5.1 Component model

Create a typed component record with:

- component name, revision and source digest;
- source language, build target and generated-code status;
- privileges, reachable inputs and exposed interfaces;
- upstream owner and update cadence;
- dependencies and ABI/API contracts;
- memory-safety and concurrency risk;
- test suites, fuzz targets and known invariants;
- vendor coupling and device coverage;
- migration candidate score and reasons.

Represent component dependencies and trust boundaries as a graph. Keep observed, inferred and proposed edges distinct.

### 5.2 Migration loop

1. **Inventory:** locate code, APIs, privilege, input sources, dependencies and test coverage.
2. **Model:** write behavioral invariants and security properties before translation.
3. **Prioritize:** migrate bounded high-exposure memory-unsafe parsers, protocol handlers and service components first.
4. **Propose:** use Symthaea/Broca and conventional code tools to suggest Rust designs.
5. **Test before replacing:** preserve old/new fixtures; create differential, property, fuzz and malformed-input tests.
6. **Verify compatibility:** check AIDL/API stability, ABI, parcel behavior, timing, cancellation, concurrency and error handling.
7. **Stage:** use feature flags, rollback-safe packaging and an explicit old-implementation fallback where safe.
8. **Independently qualify:** exact source SHA, toolchain, build artifact digest, test receipt and reviewer decision.

HDC/CfC can support context organization, change prioritization and adaptive monitoring. It cannot turn a generated patch into a proof. The independent verifier must remain outside the component-generation path.

### 5.3 First migration targets

Do not begin with camera, modem, GPU or bootloader drivers. First target:
- SOMA JNI/FFI ownership and lifecycle;
- SOMA message transport and authenticated session lifecycle;
- small, isolated parsers and serializers;
- a narrowly scoped AOSP Rust AIDL service prototype under Cuttlefish/emulator testing;
- a declarative policy evaluator with synthetic profiles.

Only move toward HAL/driver work after exact device-tree, kernel, vendor and integration constraints are understood.

## 6. Benchmark programme

All comparisons should use exact device, firmware, patch level, build flags, runtime, privilege, workload and environmental strata. Keep prevention, attestation, measurement quality, detection, and usability as separate scorecards.

### 6.1 Baseline matrix

| Baseline | Purpose |
|---|---|
| Stock AOSP/Pixel software | Platform compatibility and base security |
| Stock Google Pixel build | Actual vendor configuration and update behavior |
| GrapheneOS on the same supported Pixel | Strong mobile hardening reference |
| Luminous hardened AOSP configuration | Incremental improvements we contribute |
| Luminous declarative build | Policy and reproducibility contribution |
| SOMA with no adaptive/security monitor | Application/runtime overhead baseline |
| SOMA with qualified HARDSEC monitor | Incremental value and overhead |

A fair GrapheneOS comparison must use a currently supported identical device and comparable patch/build conditions. If the builds differ, report the differences rather than normalize them away.

### 6.2 Security and migration metrics

| Dimension | Metric |
|---|---|
| Memory safety | Memory-safety defects per MLOC and by reachable-input component; severity and exposure stratification |
| Unsafe Rust | LOC and block inventory, safety-invariant coverage, independent review coverage |
| Fuzzing | Reachable parser coverage, fuzz CPU-hours, unique crashes, time-to-triage, regression recurrence |
| Behavioral parity | Differential mismatches on frozen valid and invalid corpora |
| Compatibility | CTS/VTS and selected AOSP/Tradefed integration outcomes |
| Isolation | Unauthorized capability/IPC/DMA access attempts; expected-deny and expected-allow tests |
| Boot/update | Image verification, rollback, interrupted update and recovery outcomes |
| Supply chain | Source/toolchain/dependency provenance, SBOM, reproducibility and signing verification |
| Resource cost | CPU, RSS/PSS, power, thermal behavior, startup time and p50/p95 latency |
| Maintenance | Patch latency, review iterations, rollback rate, CVE recurrence and upstream mergeability |

### 6.3 Initial qualification gates

These are proposed research gates, not established industry standards or achieved results.

- 100% of required policy controls map to an enforcement point and a verification obligation, or an explicit unsupported/unobservable state.
- 0 clean verdicts for stale, replayed, invalid, unverifiable or source-mismatched attestation/evidence.
- 0 silent policy downgrades.
- 0 missing measurements converted to zero, false or benign.
- 100% of FFI pointer/slice/string conversions have a documented local safety invariant and negative-path tests.
- 100% of new Rust unsafe blocks in the migrated scope receive explicit safety comments and independent review.
- 100% of selected untrusted-input parsers have deterministic regression corpora and fuzz harnesses.
- No CTS/VTS regression attributable to a migrated component; any exclusion is documented and justified.
- No unapproved capability grant or network/sensor egress under negative controls.
- For the first SOMA migration, target no more than 5% p95 latency regression and 2% measured incremental energy overhead on a matched workload; revise only before held-out evaluation with written justification.
- Build artifacts and receipts bind to exact source, configuration, toolchain and dependency digests.

Do not set a single "Rust percentage" as the success metric. A small, high-risk parser replaced with a well-tested safe implementation may be more valuable than rewriting large amounts of low-risk code.

### 6.4 Statistical and operational discipline

- Split by device model, SoC family, firmware/build lineage and physical execution session.
- Freeze policy and detector thresholds on validation data.
- Report device-hours for false-alert rates and full-day repeated runs for energy claims.
- Keep calibrated power measurements separate from Android battery-counter proxies.
- Report worst device/family and p95/max, not just pooled means.
- Preserve every failed receipt and regression.
- Keep the evaluator deterministic and test it against mutation/adversarial evidence.
- Never allow Symthaea to be the only evaluator of its own generated migration.

## 7. Sequenced implementation plan

### P0 — Close existing SOMA security debt

- Fix Pixel build path drift.
- Replace raw pointer handles with a generation-checked registry or an equally reviewed lifecycle design.
- Define a single-threaded or synchronized ownership model enforced by the API.
- Validate all pointers, lengths, bounds and UTF-8 assumptions.
- Prevent panics from crossing the C ABI; define explicit error/status returns.
- Design authenticated, per-peer Holon sessions with freshness, replay protection, key rotation and fail-closed behavior.
- Add tests for null/stale/duplicate/concurrent handles, malformed and oversized input, output-buffer bounds and unauthenticated inbound/outbound messages.

Issue: [SOMA-SEC-001 / #9773](https://github.com/Luminous-Dynamics/symthaea/issues/9773).

### P1 — Compile security policy into evidence obligations

- Define the policy schema and typed Rust model.
- Evaluate synthetic HARDSEC-027 profiles.
- Emit deterministic obligation graphs and residual-risk reports.
- Add source substitution, stale-evidence, version-skew and unknown-state mutation tests.

Issue: [HARDSEC-029 / #9771](https://github.com/Luminous-Dynamics/symthaea/issues/9771).

### P2 — Qualify the first physical Android profile

- Resolve the formatting gate in HARDSEC-028.
- Pass injected-runner tests and external-profile audit.
- Run the fixed read-only collector against the authorized Pixel 8 Pro.
- Keep the raw file local; audit the exported profile before sharing.
- Keep cryptographic attestation, MTE process state and PMU measurement quality unqualified until separate verifiers/measurements exist.

Issues: [HARDSEC-027 / #9736](https://github.com/Luminous-Dynamics/symthaea/issues/9736), [HARDSEC-028 / #9739](https://github.com/Luminous-Dynamics/symthaea/issues/9739).

### P3 — Rust AOSP service prototype

- Pin a known AOSP manifest and build environment.
- Build a Cuttlefish/emulator target first.
- Add a small Rust AIDL service with a narrow permission and stable interface.
- Test service lifecycle, caller identity, SELinux policy, parcel compatibility, restarts and negative authorization.
- Do not require a new device kernel or proprietary driver.

### P4 — Pixel image integration

- Establish exact Pixel build/firmware/bootloader/vendor constraints.
- Preserve OEM boot chain, vendor partitions and existing drivers.
- Integrate only the qualified system modules and policy.
- Test verified boot, OTA, rollback and recovery on an authorized device with a reversible plan.

### P5 — Expand only where evidence supports it

Evaluate Rust migration candidates in system services, parsers, protocol handlers, selected HALs, kernel drivers and firmware. Each candidate gets its own threat model, compatibility proof, performance envelope and rollback plan.

### P6 — Optional capability-OS research

Explore Fuchsia and formally verified microkernel ideas as an independent research line. It should be compared on driver availability, hardware coverage, update chain, application compatibility, security model and verified properties. It is not a prerequisite for delivering the AOSP-derived mobile platform.

## 8. Immediate operational blockers and boundaries

- HARDSEC-028's inspected exact-head workflow failed at `rustfmt --check`; its test steps were skipped. It is not qualified until a corrected exact-head run passes.
- HARDSEC-027's dedicated profile workflow passed on the profile-only head. This qualifies the synthetic profile/privacy/evidence contract, not a physical device or runtime security state.
- No current physical Pixel capture is claimed by this document.
- The Pixel build-path correction is proposed here; a clean build/deploy has not yet been demonstrated.
- The historical RDP placeholder key is development-only and must not be used for production network transport.
- Never collect, publish or commit IMEI, serial number, Android ID, raw build fingerprint, raw attestation certificates, user files or other stable identifiers in a public fixture.

## 9. Research references

1. [Google Security Blog — Rust in Android: move fast and fix things (2025-11-13)](https://blog.google/security/rust-in-android-move-fast-fix-things/)
2. [AOSP — Android Rust introduction](https://source.android.com/docs/setup/build/rust/building-rust-modules/overview)
3. [AOSP — AIDL backends](https://source.android.com/docs/core/architecture/aidl/aidl-backends)
4. [AOSP — AIDL for HALs](https://source.android.com/docs/core/architecture/aidl/aidl-hals)
5. [AOSP — Mainline modular system](https://source.android.com/docs/core/ota/modular-system)
6. [AOSP — partition architecture](https://source.android.com/docs/core/architecture/partitions)
7. [Linux kernel — Rust documentation](https://android.googlesource.com/kernel/common/+/android-mainline-tracking/Documentation/rust/index.rst)
8. [GrapheneOS — features overview](https://grapheneos.org/features)
9. [Fuchsia — driver framework](https://fuchsia.dev/fuchsia-src/development/drivers)
10. [FSE 2026 — An Empirical Analysis of Rust Integration in AOSP](https://doi.org/10.1145/3803437.3805227)

The 2026 FSE study analyzes 16 quarterly AOSP snapshots from January 2021 through October 2024, maps Rust adoption across 67 components, and studies memory-safety outcomes and unsafe-code governance. It is useful as a migration-methodology reference; its observation window predates the latest AOSP Rust developments described above.
