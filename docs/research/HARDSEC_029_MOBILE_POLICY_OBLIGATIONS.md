# HARDSEC-029 — Declarative mobile security policy obligations

**Status:** First deterministic policy-to-evidence obligation evaluator and synthetic Pixel 8 Pro policy fixture. This is an offline contract prototype; it does not configure a device or prove its security.

Issue: [HARDSEC-029 / #9771](https://github.com/Luminous-Dynamics/symthaea/issues/9771)  
Parent profile: [HARDSEC-027 / #9736](https://github.com/Luminous-Dynamics/symthaea/issues/9736)  
Acquisition: [HARDSEC-028 / #9739](https://github.com/Luminous-Dynamics/symthaea/issues/9739)

## 1. Research question

Can a mobile security requirement be compiled into an explicit set of enforcement and verification obligations, while preserving the difference between:

- what a policy desires;
- what the target platform can support;
- what a platform layer can enforce;
- what the profile reports;
- what evidence is linked to that exact claim;
- what an independent verifier has actually qualified.

The compiler should not return “secure” because a model string, CPU feature token, ADB property or vendor document looks reassuring. Unknown or unqualified states must remain visible.

## 2. Initial artifacts

- `schemas/hardsec-mobile-policy-v1.schema.json`: strict versioned policy schema.
- `schemas/hardsec-mobile-policy-receipt-v1.schema.json`: receipt schema with an explicit claim ceiling.
- `tests/fixtures/hardsec/mobile-policy-v1/pixel-8-pro-research-policy.json`: six-control synthetic research policy.
- `examples/hardsec_mobile_policy_compile.rs`: deterministic obligation evaluator.
- Opt-in harness feature: `hardsec-mobile-policy-compile`.

Run from the repository root:

```sh
cargo test --locked \
  --manifest-path tools/hardsec-adaptive-evasion-audit-harness/Cargo.toml \
  --features hardsec-mobile-policy-compile \
  --example hardsec_mobile_policy_compile

cargo run --locked \
  --manifest-path tools/hardsec-adaptive-evasion-audit-harness/Cargo.toml \
  --features hardsec-mobile-policy-compile \
  --example hardsec_mobile_policy_compile
```

The command accepts `--policy PATH` and `--profile PATH` for external JSON input. Inputs are read-only.

## 3. First policy controls

The synthetic Pixel policy defines six obligations:

1. Verified Boot state must be `verified`.
2. KeyMint/StrongBox security level must be `trusted_environment` or `strongbox`.
3. Attestation challenge freshness must be `fresh_verified`.
4. Attestation certificate chain must be `trusted`.
5. MTE runtime state must be `enabled_verified`.
6. PMU observer effect must be `within_budget`.

Each policy control specifies its supported claim path, allowed values, acceptable evidence source types, enforcement layer, and verification action. Unknown states must `abstain`; unsupported requirements are marked `incompatible`.

The evaluator intentionally uses a fixed claim-path allowlist. Policy JSON cannot choose arbitrary fields, traverse a sensitive identifier, or redirect a control to an unrelated evidence list.

## 4. Evidence and outcome semantics

For each obligation, the compiler:

1. Resolves a known typed claim path.
2. Loads only the evidence IDs attached to that claim.
3. Resolves those IDs against a unique evidence ledger.
4. Checks source types against the policy's accepted set.
5. Emits an obligation with desired values, observed value, evidence references, enforcement layer, verification action, status and next action.

Statuses:
- `evidence_incomplete`: the claim is unknown or not measured; action is abstain.
- `incompatible`: the target explicitly lacks the required capability.
- `unqualified_evidence`: a desired value is present but no claim-linked evidence comes from an accepted source.
- `policy_violated`: the observed claim contradicts the desired values.
- `independent_verifier_required`: a non-synthetic claim has a desired value and accepted source-type link, but an authenticated independent verifier receipt is absent; the action remains abstain.

The prototype deliberately never marks a control satisfied. Source-type labels are untrusted metadata, even when a label reads `hardware_attestation_verified` or `physical_lab_measurement`. A separate independent verifier must authenticate the source, freshness, scope, quality and exact profile/evidence digests before a later qualification layer can report a passed obligation.

Synthetic fixtures are categorically prevented from satisfying a policy control, even if a test mutates a field to say `enabled_verified`.

## 5. Current expected result

The supplied Pixel 8 Pro fixture is synthetic. It documents model-family MTE capability but leaves boot, attestation, process-level MTE runtime and PMU observer-effect states unmeasured. Even a mutated non-synthetic profile with accepted-looking source labels cannot satisfy a control: this compiler has no independent receipt authentication module.

Therefore the deterministic policy receipt should report:

| Result field | Expected |
|---|---:|
| Total obligations | 6 |
| Claim-satisfied | 0 (the compiler never issues this verdict) |
| Evidence incomplete | 6 |
| Device security proved | false |
| Runtime integrity verified | false |

This is the correct outcome for missing physical-device evidence. It is not a detector failure.

## 6. Adversarial tests

The example tests:
- synthetic-profile abstention;
- vendor documentation cannot qualify process-level MTE runtime enablement;
- claim-linked physical evidence may satisfy a declared obligation only on a non-synthetic capture;
- policy/profile target mismatch rejection;
- duplicate evidence ID rejection;
- unknown claim path rejection.

The next extension should add property-based mutation tests for evidence source substitution, ID replay, policy version skew, stale attestation, unsupported values, duplicate controls and mismatched claim scope. The policy evaluator must stay deterministic and offline.

## 7. Claim ceiling and next research

This first slice is a deterministic policy-obligation compiler prototype. It is not a build-system compiler yet: it does not emit Soong, SELinux, AIDL, kernel, bootloader or vendor configuration. Those mappings should be added only after each enforcement path is independently documented and exercised.

The next step is to add target-specific realization adapters:
- AOSP build-time policy and SELinux constraints;
- AIDL/system-service permissions;
- Verified Boot and update lifecycle obligations;
- process-level memory tagging checks;
- sensor admission and telemetry privacy requirements;
- per-device unsupported/unobservable constraints.

The evaluator should continue to distinguish a syntactically satisfied declaration from independently verified platform enforcement.
