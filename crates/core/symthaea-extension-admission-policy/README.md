# symthaea-extension-admission-policy

Fail-closed composition of technical inspection, signer verification, and local
policy for Symthaea extension admission.

```text
exact manifest bytes + exact payload bytes
        |
        +--> TechnicalInspector
        |       |
        |       +--> parsed manifest + exact digests
        |
        +--> independent re-parse + SHA-256
        |       |
        |       +--> exact equality required
        |
        +--> domain-separated PackageCommitment
                |
                +--> SignerVerifier
                        |
                        +--> principal
                        +--> trust ceiling
                        +--> trust generation
                                |
                                +--> AdmissionPolicy
                                        |
                                        +--> AdmissionRecord
```

The evaluator implements neither cryptography nor a runtime. Those are explicit
injected trust dependencies, keeping Wasmtime/Xenia/Mycelix out of this core
policy contract.

## Package commitment

The verifier authenticates:

```text
SHA256(
  "symthaea.extension.package.v1\0"
  || manifest_sha256
  || payload_sha256
)
```

A valid signature for one payload therefore cannot authorize another payload
under the same manifest.

## Independent consistency check

The technical inspector runs first so runtime-specific size/format/sandbox
policy can reject a candidate before the generic evaluator performs its own
parse/hash pass. The evaluator then independently parses the exact manifest
bytes and hashes both inputs, requiring exact equality with the inspection
receipt. A runtime adapter therefore cannot silently substitute semantic input.

## Currentness

`SignerVerifier` returns a trust generation alongside the authenticated
principal. The issued `AdmissionRecord` binds that generation. Point-of-use
activation later requires both the admission-policy generation and signer-trust
generation to remain current.

## Restart boundary

`AdmissionRecord` is in-process authority and is intentionally not
deserializable. It may be serialized as audit evidence, but after a restart the
host must rerun this evaluator against the exact package/signature/current trust
state. Stored `AdmissionRecordEvidence` cannot be promoted back into authority.

## Policy commitment

`AdmissionPolicy` canonicalizes unordered capability and permission lists and
computes a domain-separated SHA-256 commitment over its generation, issuer,
maximum trust, capability allowlist, and permission ceiling. That digest is
stored in the admission record and propagated into routing evidence.

## Boundaries

This crate does not:

- choose or implement signature algorithms;
- maintain the trust/revocation database;
- execute Wasm;
- grant ambient filesystem/network/WASI access;
- treat signer trust as scientific/model quality;
- deserialize persisted admission evidence into authority;
- skip the later point-of-use `ActiveAdmission` currentness check.

A concrete `symthaea-extension-host::ControlInspection` adapter should remain a
thin bridge once the sibling host/admission stacks share ancestry.
