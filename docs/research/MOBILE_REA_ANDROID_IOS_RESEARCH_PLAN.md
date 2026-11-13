# REA + Symthaea for Android and Apple Mobile Systems

**Status:** research plan and implementation contract  
**Date:** 2026-11-01  
**Issue:** [MOBILE-REA-001 / #9836](https://github.com/Luminous-Dynamics/symthaea/issues/9836)  
**Related:** [ENG-REA-001 / #8022](https://github.com/Luminous-Dynamics/symthaea/issues/8022), [MOBILE-OS-001 / #9772](https://github.com/Luminous-Dynamics/symthaea/issues/9772), [MOBILE-BENCH-001 / #9799](https://github.com/Luminous-Dynamics/symthaea/issues/9799), [HARDSEC-027 / #9736](https://github.com/Luminous-Dynamics/symthaea/issues/9736)

## Decision

**Yes: Symthaea should use REA as an optional reverse-engineering evidence producer for mobile engineering.** The strongest near-term use is Android. The iPhone path is valuable too, but it is currently an application/artifact analysis lane, not a general-purpose iOS device or firmware reverse-engineering lane.

Do not make REA a production dependency of Symthaea's Rust core. Treat it as an external MCP/CLI provider whose evidence is imported, normalized and independently checked. Symthaea's differentiator should be what happens *after* REA returns evidence: cross-layer reasoning, causal diagnosis, experiment selection, and migration planning with explicit claim limits.

This is a staged investigation workflow, not a proposal to replace Android or iOS wholesale.

## 1. Current REA capability matrix

The following describes the canonical upstream documentation inspected on 2026-11-01. Pin an exact REA commit, npm package version, provider executable digest and configuration before each reproducible run; upstream documentation and release contents can move independently.

| Surface | Documented REA capability | What it does not establish | Pilot decision |
|---|---|---|---|
| Android connected device | Caller-supplied ADB can inspect devices, displays/windows, features/services, packages, processes, settings, logs, dumps and acquire files/APKs | Hardware attestation, complete privileged device state, or iOS-equivalent coverage | Use an explicit read-only command allowlist |
| Android application | Static APK inspection via JADX; Android manifest/resources via caller-supplied Apktool; optional JEB; APK signature inspection | Runtime behavior, server-side behavior, effective permissions in every runtime state | Use now, preserve tool and coverage metadata |
| Android firmware | Binwalk region inspection and Unblob extraction on Linux x64 | Android OTA semantics, AVB verification, partition meaning, boot behavior, device compatibility | Generic discovery only; add Android-specific provider |
| Apple application | IPA/ZIP/DMG inventory, application graph, iOS bundle recognition, Mach-O/native analysis, plist and signature operations | The inventory projection alone does not parse all plist/CMS semantics or prove runtime loading/calls | Use now for supplied IPA/artifact analysis |
| Apple crash artifacts | Static Apple IPS crash-report metadata inspection | Live device state or a complete sysdiagnose interpretation | Use when explicitly supplied |
| iPhone live device | No equivalent general REA device-acquisition lane is documented in the inspected capability catalogue | Live iOS process, service, sensor, device state or system-log acquisition | Unsupported until a separate provider and consent model are qualified |
| iPhone firmware | Generic Binwalk/Unblob can inspect files on its supported host; REA does not document a semantic IPSW/IMG4 pipeline | BuildManifest/device matching, IMG4 meaning, kernelcache/DeviceTree/dyld semantics, signing-chain validation | Evaluate a separate pinned Apple firmware provider |

### Android nuance

REA's ADB surface includes both read operations and operations that mutate the target (for example install/uninstall and file push). The pilot must not expose the full tool catalogue as an unrestricted agent action surface. Define an allowlist of read-only operations and make all mutation tools unavailable to the Symthaea mobile-analysis profile.

ADB output is useful observation, but properties, feature lists and service inventories are not hardware-backed attestation. Feed those observations to HARDSEC with their source and trust class intact; do not upgrade them to verified hardware facts.

The generic firmware provider is not a substitute for Android's OTA and Verified Boot formats. Android A/B OTA payloads contain metadata describing operations that transform partition blocks; Android Verified Boot establishes a chain from a hardware-protected root through the bootloader and verified partitions, with rollback protection. Those semantics need their own parser/verification lane.

### Apple nuance

REA's Apple application inventory supports IPA, ZIP and DMG artifacts, and can identify iOS application roots through shallow bundle structure. Its documented workflow separately offers plist/signature inspection and Mach-O analysis. The graph itself explicitly limits its claims: path-derived framework/runtime hints are hypotheses, not proof that a framework was loaded.

The iPhone firmware surface is different. Apple's secure-boot chain checks components such as bootloaders, kernel, kernel extensions and cellular baseband firmware. A public IPSW being inspectable does not imply that an arbitrary replacement iOS image can be installed. Ordinary iPhone deployment remains an app/framework integration problem, not an AOSP-style custom-OS deployment problem.

## 2. Recommended architecture

Use four planes with one-way evidence flow:

```
Android / Apple artifacts and explicitly authorized observations
                         |
                         v
          REA + platform-specific providers
                         |
                         v
        immutable raw evidence + provenance
                         |
                         v
      Rust normalization / claim-ceiling layer
                         |
                         v
 Symthaea: graph, hypotheses, causal diagnosis,
     experiment selection, migration proposals
                         |
                         v
 HARDSEC + independent oracles + human review
                         |
                         v
     qualified findings / tests / change plans
```

### Plane A — evidence producers

- Keep REA optional and caller-configured.
- Prefer saved REA JSON Evidence for deterministic replay; MCP is the interactive path, not the source of truth.
- Keep external tools in isolated, pinned dev shells or containers.
- Use exact executable digests, versions, host OS/architecture and effective options.
- Never run recovered firmware or extracted native programs during static analysis.

### Plane B — Rust evidence normalization

Consume the original REA result without deleting or rewriting it. Preserve:

- source artifact SHA-256 and byte length;
- raw producer-report bytes and digest;
- producer name, version, commit/release identity and executable digest where available;
- host/architecture, configuration and selected range;
- output artifact hashes and sizes;
- derivation edges and coordinate spaces;
- unknown, unavailable, encrypted, partial, unsupported and failed states;
- exact source references for every derived claim.

Reuse the existing REA work on coverage integrity, lineage, semantic normalization, portable receipts, authentication and cross-provider truth. Do not make a second generic provenance format. Add only the mobile-specific fields that cannot be represented by the current contract.

### Plane C — Symthaea reasoning

Construct a typed graph whose edges carry evidence and uncertainty:

- platform build → partition/image → filesystem/container → executable/library;
- application bundle → executable → dylib/framework → imported API/entitlement;
- ADB observation or supplied diagnostic → process/service/package → event/time window;
- source component → compiled artifact → observed behavior → regression test;
- migration proposal → expected compatibility/security benefit → required proof.

Symthaea should use the graph to:
1. localize which component could explain an observed behavior;
2. distinguish build noise, semantic code changes, configuration changes and runtime behavior;
3. rank plausible causes with calibrated uncertainty;
4. choose the next experiment with high expected information gain and low device/energy risk;
5. propose a minimal migration and generate differential/fuzz/property tests;
6. produce an evidence-linked engineering explanation.

A generated explanation is a hypothesis, not a verification result. Keep independent oracles and human review outside the model's decision loop.

### Plane D — HARDSEC qualification

HARDSEC decides what the evidence may support. Keep at least these tiers distinct:

1. **Raw observation:** exact bytes or command response captured.
2. **Structural parse:** parser reports a field/range/relationship.
3. **Semantically checked:** an independent oracle validates the field against known-answer input.
4. **Runtime observation:** a controlled test records a behavior on a named target.
5. **Hardware-backed evidence:** a separately authenticated platform mechanism attests a narrowly defined property.
6. **Causal support:** an intervention or discriminating experiment provides evidence for a causal explanation.

No lower tier automatically promotes itself to a higher tier. Missing and encrypted data remain unknown.

## 3. Android-specific plan

### 3.1 Artifact semantics

Add an optional Android provider lane for:

- factory-image manifests and partition inventories;
- OTA payload header, CrAU manifest and operation graph;
- source/destination partition identity, block ranges and payload-data ranges;
- boot, vendor_boot, init_boot, dtbo, vbmeta and super metadata;
- AVB descriptors, chained vbmeta, rollback-index locations/values and verification-key identity;
- dynamic partition groups, sizes and slot relationships;
- APK/APEX identities and native ELF dependencies.

Candidate upstream references and tools are AOSP update_engine/payload definitions, avbtool, unpack_bootimg and the sparse/super image utilities. Pin and qualify them independently. Prefer wrapping trusted upstream parsers over writing a parallel format parser; a Rust-native implementation is justified only for a specific bounded contract with known-answer tests.

Do not apply OTA operations to a device in the benchmark. Decode the plan, validate block arithmetic and hashes, and reconstruct only inert fixture outputs in a sandbox.

### 3.2 Device observation

Start on an emulator/Cuttlefish for reproducible service-level tests, then use the Pixel 8 Pro only with explicit authorization.

Read-only capture may cover build fingerprint, security patch level, verified-boot state as reported, bootloader state as reported, package inventory, services, process inventory, display/window facts and selected logs. Each field records its actual source and whether it is software-reported, hardware-backed, unavailable or not measured.

A separate HARDSEC path must validate KeyMint/StrongBox attestation, challenge freshness, certificate chain, boot state, rollback resistance and target binding. ADB never supplies that proof merely by returning a plausible property.

### 3.3 Rust migration

Use AOSP's supported Rust build and Binder/AIDL integration. Prioritize parsers and services that accept attacker-controlled input across privilege boundaries. Keep the existing vendor, kernel, radio, graphics and boot chain unless an independently qualified change is justified.

For each candidate component: threat model → input reachability → contract tests → parser differential → fuzzing → resource/latency profile → unsafe Rust review → compatibility tests → independent security review.

## 4. Apple-specific plan

### 4.1 Application lane first

Use REA's current IPA and Apple application workflows to inspect:

- Info.plist declarations and app identity;
- executable candidates, Mach-O load commands and dylib/framework relationships;
- entitlements and code-signing metadata where the selected operation supports them;
- app extensions, helper bundles, embedded frameworks and resource relationships;
- native/Swift/Objective-C/Flutter/React Native/Unity components, with framework-family hints kept as hints;
- Apple IPS crash reports supplied by the user.

The aim is to map behavior and trust boundaries, not copy proprietary source code or infer runtime behavior from bundle paths alone.

### 4.2 Firmware lane as a separate provider

Evaluate the open-source `blacktop/ipsw` tool as an optional, pinned provider for public or otherwise authorized IPSW/OTA files. Its documented areas include IPSW/OTA inspection and extraction, Mach-O, dyld shared cache, kernel, IMG4, iBoot and device-specific firmware analysis.

The first pilot should be read-only and offline:

1. inventory the archive and capture its whole-file digest;
2. parse BuildManifest/device-build identity;
3. identify manifest-selected component paths and device variants;
4. extract only components supported by the pinned provider;
5. hash every output and retain parent-child lineage;
6. distinguish encrypted, unavailable, unsupported and successfully parsed components;
7. compare two builds only when device/build identity and selected component scope match.

Do not infer that an encrypted image is empty, that one device variant applies to another, or that a parsed IMG4 container validates Apple's signing chain. A cryptographic trust claim requires a separate validator and trusted roots.

### 4.3 Physical iPhone boundary

Do not assume an ADB-equivalent interface exists. Start with supplied IPA files, crash reports and sysdiagnose exports. Any future paired-device acquisition must have a reviewed command allowlist, explicit on-device/user consent, clear disclosure of data collected, bounded retention and a no-write default.

### 4.4 Rust on iOS

Rust can be built for ARM64 iOS targets, but the Rust target documentation identifies these as Tier 2 without host tools and requires the corresponding iOS SDK from Xcode. Therefore:

- put portable Symthaea cognition, HDC and bounded local inference behind a Rust library boundary;
- link the Rust library into a Swift/SwiftUI host through a narrow C ABI or generated binding layer;
- let Swift and documented Apple frameworks own lifecycle, entitlements, sensor access, background execution, UI and platform signing;
- test on both Simulator and physical hardware; simulator results are not device qualification.

This is a realistic iOS product path. It is not a path to replace iOS or bypass the secure-boot chain.

## 5. Paradigm shifts

### Shift 1 — from binary diffs to behavior-linked evidence graphs

A changed byte is not necessarily a changed behavior, and an unchanged binary is not proof of unchanged system behavior. Tie artifacts to source components, interfaces, configuration, trace events and differential tests. Each relation has a named evidence source and confidence ceiling.

### Shift 2 — from decompilation volume to uncertainty reduction

Do not optimize for the number of functions decompiled or lines summarized. Optimize for whether the next observation separates competing hypotheses. Use Symthaea's information-gain/decision-value machinery to select the cheapest useful next probe.

### Shift 3 — from “Rust percentage” to privileged attack-surface reduction

Rank components by reachable untrusted inputs × privilege × exploitability × blast radius × updateability. A small network parser in a privileged service can matter more than a large low-risk library. Track the number of reachable unsafe-input paths actually removed, not just Rust LOC.

### Shift 4 — from tool output to portable evidence

Every claim carries input digest, tool identity, output digest, target scope, coverage and limitations. Independent verifiers consume the evidence artifact offline. CI summaries are not the final authority.

### Shift 5 — from a single mobile OS to a cross-platform cognitive core

Share the Rust cognitive core and evidence graph. Keep Android's system-service/AIDL integration and iOS's app/framework/Swift integration as distinct adapters. Share behavior contracts and tests only where semantics truly match.

## 6. Benchmark programme

Use three ablations on held-out artifacts and tasks:

- **A — REA alone:** conventional REA investigation with the same allowed tools.
- **B — Symthaea alone:** Symthaea receives the same permitted raw facts but no REA analysis.
- **C — REA + Symthaea:** Symthaea reasons over REA Evidence plus the identical source/target context.

Report the per-platform results separately; do not mix Android and Apple device results into one average.

| Scorecard | Primary metrics | Proposed gate |
|---|---|---|
| Byte/region truth | known payload recall, false emitted bytes, hash correctness, unexplained-byte coverage, lineage correctness | 100% expected payload recall and zero wrong-content outputs for supported corpus cases |
| Android semantic truth | CrAU operations, partition/slot mapping, AVB chain/rollback metadata | 100% agreement with independent known-answer oracle for fields declared supported |
| Apple semantic truth | BuildManifest identity, device variant, IMG4 structure, Mach-O and dyld relationships | 100% agreement with independent known-answer oracle for fields declared supported |
| Cross-build diagnosis | top-1/top-3 component localization, false-cause rate, build-noise discrimination | at least 20% relative top-3 gain over the stronger individual ablation, without higher false-cause rate |
| Behavior reconstruction | differential-test precision/recall, state-transition coverage, false equivalence | zero false equivalence on the frozen corpus |
| Security migration | reachable unsafe-input paths by privilege, fuzz crashes, compatibility and performance cost | at least 20% fewer reachable unsafe-input paths in the migrated scope, with no privilege increase |
| Evidence assurance | false-pass mutation suite, digest/identity binding, unknown-state preservation | zero false passes; 100% binding and explicit abstention under missing/stale/replayed/mismatched evidence |
| Engineer value | time-to-first-correct hypothesis, useful experiment rate, citation completeness, human correction rate | statistically supported improvement over the best ablation; publish confidence intervals and tail latency |

Targets are research goals, not industry standards or measured outcomes. Set the supported-case denominator before evaluating; unsupported formats cannot be quietly removed after a failure.

### Essential negative controls

- truncate the OTA/firmware input and corrupt archive central-directory metadata;
- duplicate, reorder or omit partition/component records;
- substitute a different model/board/build identity;
- alter AVB key identifiers, descriptors, rollback metadata or signed digest;
- change one payload output byte while retaining the original receipt;
- confuse a decompressed child offset with an original-file offset;
- present unavailable/encrypted components as if they were zero-length or fully inspected;
- replay an old device observation after an OTA/reboot;
- inject a false ADB property or stale log line;
- substitute an IPA component, bundle path, entitlement or signing record;
- permute harmless report ordering and temporary paths to test semantic-digest stability;
- seed build noise that changes bytes but not behavior, and a configuration/runtime change that changes behavior without a simple binary diff.

Every mutation must either be correctly detected or produce an explicit abstention; none may produce a clean security or equivalence pass.

## 7. Industry comparison and the realistic way to win

| Reference | Existing strength | Luminous opportunity |
|---|---|---|
| AOSP / Google Android | Maintained OS, vendor integration, OTA/AVB, Android Rust and AIDL support | Better evidence-linked component migration prioritization and independent claim qualification |
| GrapheneOS | Hardened Android configuration, device-specific security posture and update discipline | Complement it with reproducible analysis and measured comparisons; do not claim to supersede it without same-device results |
| Apple iOS | Hardware-rooted secure boot, signed platform components and tightly integrated app/runtime model | Build a high-quality local-first Rust cognitive app and app-level evidence/assurance tooling, not an alternative iOS OS |
| REA | Broad agent-facing analysis across application artifacts, binaries and selected runtime/provider surfaces | Add calibrated cross-layer reasoning and decision-aware next-experiment selection |
| blacktop/ipsw | Specialized Apple firmware research and extraction | Supply verified evidence adapters, known-answer corpus and migration/behavior hypotheses |
| Ghidra/JADX/Apktool/JEB | Mature decompilation, bytecode and resource analysis | Use them as producers; evaluate the quality of conclusions and downstream tests, not raw output volume |

The defensible competitive claim is not “we rewrote both phone operating systems” or “we found more functions.” It is: **given the same artifacts and budget, Symthaea + REA can identify the right component, produce a better next experiment, and reduce migration risk with fewer unsupported claims.** Only the three-way ablation can validate that claim.

## 8. Delivery sequence

1. **Capability and policy baseline:** pin upstream REA; record exact supported MCP/CLI tools; create a read-only ADB allowlist; record explicitly unsupported iPhone operations.
2. **Android artifacts:** build a small known-answer corpus for OTA metadata, boot/vbmeta and dynamic partitions; compare AOSP-native tools and independent oracle.
3. **Apple applications:** build an IPA known-answer corpus for bundle graph, Info.plist, Mach-O, signatures and entitlements; include nested extensions and unavailable components.
4. **Apple firmware:** evaluate pinned ipsw on a small, licensed/public IPSW corpus; preserve unsupported/encrypted results.
5. **Rust ingestion:** implement a small adapter only after the output schemas are pinned; no REA dependency in the default Symthaea runtime.
6. **Causal diagnosis:** connect the existing ENG-REA-002 corpus to mobile source-derived changes and controlled behavioral traces.
7. **Same-device qualification:** Android Pixel 8 Pro comparison with stock/GrapheneOS/Luminous builds; iOS measurements on a named iPhone build with an app-level test harness. Keep OS, hardware and patch scopes separate.

## 9. References

- REA upstream: https://github.com/morluto/rea
- REA firmware boundary: https://github.com/morluto/rea/blob/main/docs/firmware-analysis.md
- REA Android device acquisition: https://github.com/morluto/rea/blob/main/docs/adb-device-analysis.md
- REA Android APK analysis: https://github.com/morluto/rea/blob/main/docs/android-analysis.md
- REA Apple application analysis: https://github.com/morluto/rea/blob/main/docs/apple-application-analysis.md
- Android A/B OTA updates: https://source.android.com/docs/core/ota/ab
- Android update_engine payload format: https://android.googlesource.com/platform/system/update_engine/
- Android Verified Boot: https://source.android.com/docs/security/features/verifiedboot
- Apple iPhone/iPad boot process: https://support.apple.com/guide/security/boot-process-for-ipad-and-iphone-devices-secb3000f149/web
- Rust iOS targets and Xcode SDK requirement: https://doc.rust-lang.org/rustc/platform-support/apple-ios.html
- Apple firmware research tool: https://github.com/blacktop/ipsw
- AOSP Rust integration study (FSE Companion '26): https://doi.org/10.1145/3803437.3805227

## Claim ceiling

This document records a research decision and proposed benchmark. No Android OTA semantic adapter, IPSW provider integration, mobile REA ingestion crate, physical iPhone acquisition, or cross-platform comparative result is implemented or claimed here.

## 10. Industry application-security baselines

Do not invent a substitute for the OWASP Mobile Application Security programme. Use:
- **MASVS** as the cross-platform control baseline (storage, cryptography, authentication, network, platform, code, resilience and privacy).
- **MASWE** as the mobile weakness taxonomy.
- **MASTG** as the test/technique catalogue used to verify controls.

Pin the exact repository revision, checklist and test identifiers for every run. A MASVS control without a corresponding MASTG test should be explicitly marked as an oracle gap; do not silently invent equivalence. A static heuristic cannot pass a control whose verification requires runtime behavior.

### Tool comparisons

- **REA**: evidence-driven MCP investigations over APKs, bundles, native artifacts and selected runtime/provider operations.
- **MobSF**: an established cross-platform mobile app-security platform with static analysis for APK/IPA and dynamic testing paths. Its dynamic paths have environment constraints: the documented iOS lane uses Corellium or jailbroken iOS; Android may use rooted targets or Frida Gadget mode. Those results cannot be pooled with tests on stock, non-jailbroken devices.
- **JADX / Apktool / Ghidra / JEB**: treat as underlying specialist engines and isolate them as ablations where the question is which evidence producer adds value. Do not count a feature of an engine as a REA feature unless the exact selected REA operation exposes it.

### Required ablations

A. MobSF alone.  
B. REA alone.  
C. REA + Symthaea.  
D. Symthaea with equivalent raw observations but no REA output, where the input conditions can be made fair.

Compare seeded weakness detection, control coverage, false positives, false negatives, evidence citation quality, time-to-correct finding and analyst correction rate. Report Android and iOS separately. Do not claim certification from a scanner report.

### Android platform compatibility

For a system-level AOSP-derived build, use CTS for framework API/behavior compatibility and VTS for HAL/kernel/vendor-facing compatibility, in addition to focused platform tests. CTS/VTS are compatibility suites, not substitutes for the security evidence and threat-model scorecards in this plan. Preserve test plan, device/build identity and exact result artifacts. A failure must be triaged; never remove or skip a compatibility gate merely to improve the headline score.

References:
- OWASP MASVS: https://mas.owasp.org/MASVS/
- OWASP MASTG: https://mas.owasp.org/MASTG/
- OWASP MASWE: https://mas.owasp.org/MASWE/
- MobSF: https://github.com/MobSF/Mobile-Security-Framework-MobSF
- AOSP CTS/VTS: https://source.android.com/docs/core/tests
- Related implementation benchmark: [MOBILE-REA-005 / #9842](https://github.com/Luminous-Dynamics/symthaea/issues/9842)
