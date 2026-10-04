# TPM Adapter Qualification Matrix

## Purpose

This matrix defines the evidence levels for the Linux TPM freshness adapter. A higher level requires the lower level's exact artifacts; evidence does not transfer across source rewrites or repository extraction.

| Level | Environment | What may be established | What remains unproven |
| --- | --- | --- | --- |
| Q0 | Source inspection | Contract structure, bindings, fail-closed invariants | Any runtime behavior |
| Q1 | swtpm emulator | TPM2 command plumbing, NV counter semantics, Quote/NV_Certify syntax, signature verification | Hardware provenance, physical TPM identity, production persistence characteristics |
| Q2 | swtpm adversarial | Replay rejection, counter persistence across restart, generation advance, partial-range negative cases, policy substitution rejection | Physical hardware trust, firmware/board behavior |
| Q3 | Real TPM2 hardware | Device TCTI behavior, real NV public area, real attestation-key evidence, real PCR/event-log observations, crash/reboot behavior | Independent platform endorsement unless separately verified |
| Q4 | Independent verifier campaign | Independent reconstruction/appraisal of exact evidence and policy | Compromise of the underlying TPM or platform outside evidence scope |

## Q1/Q2 hard invariants

Every emulator campaign must demonstrate:

- TPM 2.0 family is actually reported;
- an `TPM_NT_COUNTER` NV index has exactly 8 bytes;
- counter generation is observed and advanced only by `TPM2_NV_Increment`;
- the same fresh challenge binds PCR Quote and NV certification;
- a Quote replay under a different challenge is rejected;
- the counter survives a controlled TPM restart when the emulator is configured with persistent state;
- the counter advances after restart;
- a post-restart attestation key is used only after recreating its transient handle;
- full NV certification covers offset 0 and size 8;
- the TPM is capable of certifying a weaker partial range, and that weaker proof is explicitly treated as non-authoritative by the domain contract.

## Q3 hardware gates

A real-hardware run may not claim Q3 until the adapter captures, validates, and records:

- exact TPM device identity and TCTI configuration;
- exact NV public area and canonical NV Index Name;
- exact NV attributes, name algorithm, authorization policy, and data size;
- exact attestation-key identity and its deployment authorization chain;
- fresh Quote and NV certification with the same challenge;
- exact PCR selection and the corresponding measured-boot/event-log evidence where applicable;
- synchronous persistence qualification for the selected NV counter;
- controlled reboot/crash observations showing the expected generation semantics;
- all raw signed attestation bytes needed for independent re-verification.

## Claim ceiling

Q1/Q2 never establish hardware-backed authority.

Q3 establishes only that one concrete deployment produced evidence satisfying the adapter's hardware observations and cryptographic checks. It does not establish that the TPM, firmware, motherboard, or physical platform is uncompromised.

Q4 adds independent appraisal of the evidence chain. It still does not create physical authority; it validates the cryptographic/provenance claims within the configured trust model.

## Source identity

All qualification packets must include:

- exact Git commit SHA;
- exact source tree/changed-file inventory;
- exact toolchain versions;
- exact TSS library versions;
- exact TPM policy/profile fingerprint;
- challenge digests only (never raw challenge material in retained public evidence);
- test results and failure diagnostics;
- explicit authority claim level.

## Relationship to Spore and Nixward

Spore supplies deployment observations and TPM enrollment. Nixward supplies configuration/policy reasoning. Neither layer can advance this matrix beyond Q0/Q1-style observation/plumbing claims.

The regenerative-health verifier remains the sole owner of authoritative freshness capability.

## Current status

Current repository work provides Q0 source verification and prepares Q1/Q2 through the swtpm smoke suite. Q3/Q4 remain gated on a concrete Rust TPM adapter and independently verified real-hardware evidence.
