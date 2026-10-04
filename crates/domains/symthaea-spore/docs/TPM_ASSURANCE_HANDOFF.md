# Spore / Nixward TPM Assurance Handoff

## Boundary

Spore is the deployment and hardware-observation layer. Nixward is the configuration/reasoning layer. Neither layer may mint the authoritative freshness capability defined by regenerative-health.

The key distinction is:

`TPM present != TPM configured != Secure Boot enabled != measured boot observed != TPM evidence verified != freshness authority`.

## Existing Spore evidence

The relay `probe_hardware` currently reports:

- `tpm2_available`, based on `/dev/tpmrm0` presence;
- `tpm2_spec_major`, when the TPM sysfs version observation is readable;
- EFI availability;
- Secure Boot state;
- `measured_uki`, when `bootctl` reports a measured UKI;
- firmware Setup Mode;
- architecture and other hardware observations.

`/dev/tpmrm0` presence is a useful deployment preflight signal but is not proof of TPM identity, NV-index identity, attestation-key identity, PCR state, or anti-rollback state.

Linux also exposes TPM sysfs observations such as `tpm_version_major` and PCR-bank snapshots, and the kernel TPM documentation notes that PCR contents are snapshots whose meaning is stronger when accompanied by the firmware event log. These can improve preflight diagnostics but remain observations until cryptographically appraised.

## Existing Spore TPM2 enrollment

The relay's TPM2 post-install path uses `systemd-cryptenroll` and adds `tpm2-device=auto` to the systemd initrd crypttab configuration.

That is useful for LUKS key release, but it is not equivalent to freshness attestation. In particular:

- LUKS TPM enrollment proves that a key can be released under its configured TPM policy; it does not prove that a later recovery record is current.
- the chosen PCR policy must be treated as an explicit product decision and validated against the actual Secure Boot / UKI lifecycle;
- a successful enrollment command must not be converted into a statement that the machine is attested;
- enrollment metadata must never be accepted as a substitute for a fresh TPM Quote or NV certification.

Current systemd guidance describes PCR 7, PCR 11, and PCR 14 as common policy inputs for encrypted volumes and specifically cautions that direct firmware PCRs such as PCR 0 can be brittle across updates. For measured UKIs, PCR 11 covers the kernel/UKI measurement path. This is a LUKS policy concern, not by itself a regenerative-health freshness proof.

The installer must not infer the final installed system's measured-UKI PCR policy from the live installer environment. A future policy-aware enrollment step should inspect the target's finalized boot artifacts and explicitly record the chosen PCR policy before enrollment.

Current systemd guidance treats PCR 7, PCR 11, and in some configurations PCR 14 as common encrypted-volume policy inputs; it also notes that direct firmware measurements such as PCR 0 are more brittle across updates. PCR 11 covers the systemd-stub kernel/UKI measurement path, and signed PCR policies can make software updates less brittle than binding to one fixed PCR value. These are LUKS policy mechanisms, not regenerative-health attestation by themselves. See systemd-cryptenroll(1): https://man7.org/linux/man-pages/man1/systemd-cryptenroll.1.html

## Nixward handoff

Nixward currently receives a `HardwareProfile` with `has_tpm`, `has_secure_boot`, and related deployment facts, and can reason about TPM2 configuration.

Nixward should treat these values as configuration inputs and uncertainty sources. It should not derive:

- `FreshnessAnchorBacking::HardwareProtected`;
- rollback resistance;
- verified TPM identity;
- verified NV Index identity;
- attestation freshness;
- authoritative recovery generation

from those booleans alone.

A useful future normalized handoff is an observation bundle containing:

- TPM device/resource-manager presence;
- TPM specification major version;
- Secure Boot state;
- measured-UKI state;
- EFI availability and firmware Setup Mode;
- TPM event-log availability;
- selected TPM device path;
- explicit deployment policy for the expected attestation key and NV counter.

Even this bundle remains non-authoritative until passed into the concrete TPM verifier.

## Regenerative-health handoff

The eventual concrete adapter should consume Spore/Nixward facts only as preconditions and policy selectors.

The authoritative chain remains:

`Spore observation -> Nixward deployment policy -> concrete TPM verifier -> Quote + NV_Certify -> canonical evidence binding -> FreshnessAnchorVerificationReceipt -> verifier attestation -> authoritative recovery commit`.

Spore/Nixward may therefore improve discoverability, installation correctness, and policy selection without becoming a second trust oracle.

## Immediate engineering value

The current codebase already provides:

- TPM device detection;
- Secure Boot / Setup Mode detection;
- `swtpm` in the Nix development environment;
- QEMU-based VM test infrastructure;
- NixOS `crypttabExtraOpts` generation;
- a Nixward configuration reasoning path.

These are ideal inputs for the next adapter qualification campaign:

1. emulate TPM availability with `swtpm`;
2. verify observation parsing;
3. verify LUKS enrollment behavior;
4. inject controlled TPM/NV state mutations;
5. run the concrete TPM freshness adapter;
6. exercise crash/recovery permutations;
7. promote only cryptographically verified evidence into authoritative freshness.

## Claim ceiling

Neither the installer nor the configuration generator should ever describe TPM presence, LUKS TPM enrollment, Secure Boot selection, or NixOS configuration as proof that the machine is trustworthy.

Those are preparation and policy layers. The trust boundary begins only at cryptographic appraisal of fresh evidence.
