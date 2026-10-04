# TPM NV Counter Adapter Boundary

## Purpose

This document defines the minimum operations a concrete Linux TPM adapter must perform before it can implement the deployment-neutral freshness_anchor_tpm contract.

The adapter is intentionally outside the domain crate's current dependency graph. This keeps the security contract testable without a TPM while preventing a partially implemented FFI binding from being mistaken for hardware-backed authority.

## Required operation sequence

1. **Read and pin the NV public area.**

   Use the TPM NV public-read operation and obtain the complete public area plus the canonical NV Index Name.

   Verify, against the deployment profile:

   - selected NV index identity;
   - NV Index Name;
   - name algorithm;
   - authorization policy;
   - NV attributes;
   - TPM_NT_COUNTER;
   - dataSize == 8;
   - TPMA_NV_ORDERLY is clear for the crash-persistent authoritative profile.

   The adapter must commit the complete marshaled public-area representation and the canonical Index Name into the evidence.

2. **Read the current counter value.**

   Read the selected NV counter and decode the counter value according to the TPM API's canonical representation.

   The resulting value must equal the recovery generation. A disagreement is a hard failure; the adapter must not choose the newer or older value heuristically.

3. **Generate a fresh verifier challenge.**

   Generate TpmQuoteChallenge using OS randomness.

   The same challenge is used for both TPM attestations. The persisted freshness receipt contains only the domain-separated challenge digest.

4. **Produce a PCR Quote.**

   Invoke TPM Quote with the configured attestation key, the required PCR selection, and the exact challenge as qualifying data.

   The adapter must retain the raw TPM2B_ATTEST and signature bytes because later verification must operate on the exact signed attestation statement.

5. **Produce NV certification.**

   Invoke TPM2_NV_Certify against the exact NV counter, using the same attestation key and exact challenge as qualifying data.

   For this contract the selected range is fixed to:

   - attestation form: TPM_ST_ATTEST_NV;
   - offset: 0;
   - size: 8.

   The returned attestation must be parsed as an NV certification structure. The adapter must verify that the embedded NV Index Name is the same canonical Name obtained from the public-area read, and that the certified eight-octet contents correspond to the observed counter value.

6. **Verify both TPM signatures before constructing trusted evidence.**

   Verify both signatures against the attestation-key public area.

   The adapter must also verify:

   - Quote attestation type is the expected Quote type;
   - NV attestation type is the expected full-contents NV type;
   - Quote qualifying data equals the exact challenge;
   - NV qualifying data equals the exact same challenge;
   - NV Index Name equals the pinned public-area Name;
   - NV certificate offset and size equal 0 and 8;
   - certified NV contents decode to the same counter value;
   - PCR content satisfies the deployment's measured-state policy.

7. **Bind the attestation key identity.**

   The attestation key used for Quote and NV certification must be the same deployment-authorized key.

   Its TPM identity relationship must be independently established by the platform's key-attestation / endorsement policy. Merely copying an opaque key identifier into the evidence envelope is insufficient.

8. **Construct the domain evidence envelope.**

   Populate TpmNvCounterEvidence only from values that were independently observed or cryptographically verified.

   The following fields are security bindings, not descriptive metadata:

   - TPM identity;
   - NV Index Name;
   - NV public area;
   - authorization policy;
   - Quote signer identity;
   - NV certification signer identity;
   - Quote challenge;
   - NV certification challenge;
   - Quote attestation;
   - NV certification attestation;
   - NV Index Name within the certification;
   - certified NV contents;
   - PCR binding;
   - counter value;
   - synchronous persistence mode.

9. **Compute the canonical evidence digest.**

   Set receipt.evidence_digest to the exact result of TpmNvCounterEvidence::binding_digest().

   Do not independently reimplement the domain framing in the adapter.

10. **Pass through the fail-closed domain gate.**

    Call the challenge-aware, policy-bound verification path only after all raw TPM evidence has been collected.

    The authoritative entry point is `verify_tpm_nv_counter_with_trust_policy()`. Its trust policy is pre-authorized configuration: it must be constructed independently of the incoming attestation and must contain the expected TPM identity, NV Index Name, NV public-area digest, authorization-policy digest, attestation-key identity, and PCR-policy binding.

    The older structural `verify_tpm_nv_counter()` helper remains useful for adapter/qualification tests but must not be used by itself to mint authoritative freshness. A successful adapter call is not itself sufficient authority; the resulting evidence must still pass the generic freshness-anchor verification and authoritative commit gates.

## API mapping for a Linux implementation

The current tss-esapi 7.7.0 release provides typed NvPublic accessors for the NV index, name algorithm, attributes, authorization policy, and data size. It also exposes the underlying tss-esapi-sys Esys_NV_Certify binding for the NV certification command.

That means the adapter should keep a narrow unsafe/FFI boundary around the missing high-level certification operation rather than weakening the domain contract or approximating certification with nv_read.

The high-level Context API in tss-esapi 7.7.0 provides Context::new(TctiNameConf), nv_read_public, nv_increment, nv_read, and quote. TctiNameConf supports both a real device TCTI and an Swtpm network TCTI.
The corresponding tss-esapi-sys 0.6.0 layer exposes Esys_NV_Certify, which is the appropriate narrow FFI seam for the one command the high-level wrapper does not currently expose.
The adapter should isolate that unsafe call in one module and convert its TPM2B_ATTEST and TPMT_SIGNATURE outputs into deployment-neutral evidence only after independently verifying all structural bindings.

## Explicit non-goals

This adapter contract does not claim:

- continuous runtime integrity;
- trusted wall-clock time;
- firmware correctness;
- physical ownership or authority;
- immunity from compromise between attestations;
- persistence of a TPMA_NV_ORDERLY counter across crash;
- protection against a compromised TPM itself.

Those require independent evidence and remain above this contract's claim ceiling.
