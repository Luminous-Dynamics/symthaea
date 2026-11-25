# SAT External Trace Checkpoint Profile V0.1 — 2026-11-25

**Status:** interoperability profile and schema proposal; no witness network, transparency service, signature verifier or production checkpoint has been qualified.  
**Parent:** [SAT Qualification Infrastructure V0.1](SAT_QUALIFICATION_INFRASTRUCTURE_V0_1_2026-11-25.md)  
**Schema:** [SAT trace checkpoint receipt](schemas/sat-trace-checkpoint-v0.1.schema.json)  
**Example:** [Synthetic checkpoint receipt](examples/sat-trace-checkpoint-example-v0.1.json)

## 1. Why an external checkpoint is necessary

An unkeyed event hash chain proves internal ordering and digest consistency. It does not, by itself, detect an attacker who can rewrite the entire trace and recompute every digest. The trace verifier therefore distinguishes:

- **Internal chain consistency:** every event hashes to the declared digest and the sequence links correctly.
- **External head matching:** the terminal event digest equals a checkpoint supplied independently of the trace.
- **Checkpoint authenticity:** the checkpoint signature, issuer identity, key validity/revocation and trust policy verify.
- **Transparency inclusion:** the exact statement is included in a transparency service, with a valid inclusion proof.
- **Transparency consistency:** the service's later tree head is consistent with the earlier one and has not equivocated.
- **Witness independence:** one or more independently governed witnesses retained and attest to the checkpoint under a declared trust policy.

These are separate claims. A hash match is not signer authentication; a valid signature is not proof the event observation is true; a transparency inclusion proof is not proof the log is non-equivocating unless consistency/witness procedures are also checked.

## 2. Reuse transparency infrastructure

SAT should not invent a new transparency log.

The primary interoperability target is the IETF SCITT architecture, [RFC 9943](https://www.rfc-editor.org/rfc/rfc9943.html), which defines an architecture for signed statements and transparency services in digital supply chains. A second adapter target is [Sigstore Rekor](https://docs.sigstore.dev/logging/overview/), which provides transparency-log entries, signed checkpoints and inclusion/consistency proof concepts. Existing transparency services and offline proof formats should be reused where their trust models fit.

The SAT receipt is a small binding layer: it says which frozen run and terminal event digest are being checkpointed, identifies the signed statement and transparency evidence, and points to witness attestations. It does **not** redefine SCITT, Rekor, Merkle trees, signed notes, PKI or credential status.

## 3. Canonical trace-head statement

The receipt contains a `trace_head_statement` with exactly these semantic fields:

- `schema_version`: `sat-trace-head-statement/v0.1`;
- `run_id`;
- `manifest_digest`;
- `head_sequence`;
- `head_event_digest`;
- `checkpointed_at`.

Compute:

`statement_digest = SHA-256("SAT-TRACE-HEAD-v0.1\\n" || JCS(trace_head_statement))`

The frozen manifest digest and final event digest use the domain separators defined in the core qualification protocol. Do not sign a display string or a self-declared digest without independently reconstructing the statement fields.

## 4. Receipt contents

A receipt binds:

1. exact run and manifest identity;
2. terminal event sequence and digest;
3. canonical statement digest;
4. one or more transparency entries, each identifying a service, log, entry index, tree size/root, signed checkpoint and inclusion proof;
5. optional consistency proofs between successive checkpoints;
6. witness attestations that identify the log service/root they observed (`service_id` + `log_id` + `tree_size` + `root_hash`), with the separate `witness_id` and `key_id` for the attesting party;
7. a versioned trust-policy reference and key-status/revocation evidence references.

A receipt can contain references to signed payloads/proofs rather than copying large proof objects. Referenced artifacts must be content-addressed and retrievable by the independent evaluator. Private services may use a mutually agreed export bundle, but their trust root and auditability limitations must be declared.

## 5. Verification procedure

A checkpoint adapter must execute these checks in order:

1. Reconstruct the canonical trace-head statement from the independently verified trace report and frozen manifest.
2. Recompute its domain-separated statement digest; compare it to the receipt.
3. Verify the signed checkpoint/statement envelope using a trust policy pinned in the run manifest or qualification policy. Authenticate key identity, key usage, validity, revocation state and algorithm profile.
4. Verify the transparency inclusion proof for the exact statement digest at the claimed log index/tree size/root.
5. Verify consistency from a previously pinned checkpoint to the new checkpoint. Persist the latest accepted checkpoint; a receipt supplied only by the SUT cannot replace the verifier's remembered state.
6. Compare independent witnesses' views. Reject conflicting checkpoints for the same log/tree size and apply the profile's minimum independent-witness threshold.
7. Check signer/witness role separation and conflict disclosures. A transparency service, SUT operator, auditor and evaluator are not automatically independent merely because they have different names.
8. Return a structured checkpoint disposition: `anchored`, `unanchored`, `invalid`, or `inconclusive`. Missing inclusion/consistency evidence is never a pass.

The current SAT Rust trace verifier only implements internal consistency and an optional caller-supplied head-digest comparison. It does not yet implement steps 3–7. A matching `expected_head_digest` must not be presented as authenticated anchoring.

## 6. Trust and privacy boundaries

- Do not put prompts, payloads, credentials, private memories or full event contents in a public transparency log by default. Log the minimum commitment and public metadata needed for audit; keep sensitive artifacts access-controlled.
- A log can equivocate or become unavailable. Use consistency proofs, witnesses and locally persisted checkpoints.
- A witness can collude or be compromised. Declare independence criteria, key ownership, governance, rotation, revocation and incident procedures.
- A timestamp in a signed statement is only as trustworthy as its clock and timestamp authority.
- A root of trust must be external to the SUT; the SUT cannot select its own trusted keys or lower the witness threshold.
- Hashing sensitive low-entropy content can enable guessing attacks. Prefer salted/opaque commitments or privacy-preserving payload commitments where the use case demands them; record the scheme and salt custody.
- Offline verification should be possible where practical: distribute signed statement, inclusion proof, checkpoint, key material/trust policy version and revocation/consistency evidence alongside the qualification report.

## 7. SAT benchmark cases

The public contract fixtures include checkpoint-focused negatives: absence of an anchor, mismatched head digest, missing inclusion proof, invalid signature/key status, inconsistent checkpoint history and witness equivocation. These validate the adapter contract; they do not establish performance of a transparency service.

Proposed metrics:
- **Unanchored-report rate:** fraction of reports that are explicitly labeled unanchored when no external checkpoint exists (target 100%).
- **Checkpoint substitution detection:** fraction of altered head/run/manifest statements rejected (target 100% on finite controls).
- **Inclusion/consistency proof rejection:** fraction of invalid, missing or stale proofs correctly rejected/held (target 100% on finite controls).
- **Witness equivocation detection:** fraction of conflicting signed views detected under the declared witness topology (target 100% on finite controls).
- **Checkpoint availability and latency:** p50/p95 time and cost to obtain an independently verifiable checkpoint.
- **Privacy leakage:** no sensitive prompt/payload/credential content appears in public checkpoint artifacts by construction.

These are target properties, not measured results. A finite negative suite does not establish adversarial log security or witness independence at scale.

## 8. Status semantics

- `anchored`: statement signature, key/trust policy, inclusion proof, consistency and witness threshold all pass.
- `unanchored`: internal chain is consistent, but no eligible external checkpoint is presented.
- `invalid`: cryptographic mismatch, identity substitution, invalid/revoked key, inconsistent log history or witness equivocation established by evidence.
- `inconclusive`: required evidence is missing, unavailable or cannot be independently verified.

An unanchored but internally consistent trace may still be useful for development debugging. It cannot support a high-assurance claim that trace history was protected against full rewrite.

## 9. Claim boundary

This profile is a wire and process proposal. It does not implement signature verification, transparency proofs, trusted timestamps, witness gossip or key revocation. The synthetic example deliberately uses placeholder references and is not an authenticated checkpoint. No production checkpoint, external service integration or security claim is made.

## References

- [IETF RFC 9943 — SCITT Architecture](https://www.rfc-editor.org/rfc/rfc9943.html), June 2026.
- [Sigstore Rekor overview](https://docs.sigstore.dev/logging/overview/) and [Rekor v2 API](https://www.sigstore.dev/swagger/).
- [W3C Verifiable Credential Data Integrity 1.0](https://www.w3.org/TR/vc-data-integrity/) for a related signed-document integrity model; SAT does not claim VC conformance.
