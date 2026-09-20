# SCIP Text-Claim Operational Blinding Witness V1 — V22 Evidence Note

## Boundary

V20 records human-reference process declarations, including source-blind extraction and candidate-output blinding, but deliberately does not treat those declarations as independently witnessed facts. V21 binds a validated V20 human reference into one V19 corpus-case candidate.

V22 adds a separate, content-addressed **role-session process witness**:

```text
exact V20 annotation bundle
+ exact role slot / participant provenance handle
+ exact V21 case-admission receipt reference
+ session manifest evidence root
+ pre-session access-control snapshot evidence root
+ complete-session audit-capture evidence root
+ exact allowed artifact profile
    -> structural process-boundary witness
```

This is evidence about a process boundary. It is not a semantic-fidelity capability and it does not authenticate the evidence producer.

## No invented global claim-schema digest

The repository did not expose a pre-existing canonical public claim-schema artifact with an exact digest at this boundary. V22 therefore does not fabricate one.

Each role-session witness instead binds the exact `public_claim_schema_sha256` actually used in that session. The validator requires that digest to occupy the `public-claim-schema` slot in the role's exact allowed-artifact profile. A later case-level witness composition can require every session for one case to bind the same schema digest.

## Role-specific visibility profiles

Extraction annotators may receive exactly:

```text
surface-text
public-claim-schema
```

and are explicitly separated from source inventory, source claim IDs, candidate extractor output, peer extraction annotations, and alignment evidence.

An extraction adjudicator may additionally receive the exact two extraction-annotation artifacts bound by V20, while source/candidate/alignment material remains forbidden.

Alignment annotators may receive exactly:

```text
frozen-surface-inventory
source-inventory
public-claim-schema
```

but not candidate extractor output or the peer alignment annotation.

An alignment adjudicator may additionally receive the exact two V20 alignment-annotation artifacts.

For every role slot, the participant fingerprint must equal the corresponding opaque V20 actor fingerprint. The witness issuer fingerprint must be distinct from the participant fingerprint.

## Three evidence roots

Each witness requires three distinct non-zero SHA-256 commitments:

1. `session_manifest_sha256`;
2. `access_control_snapshot_sha256`;
3. `audit_capture_sha256`.

The witness also requires declarations that the access-control snapshot was bound before the session, audit capture covers the full session, audit capture is complete, zero forbidden-access events were recorded, and the evidence is retained outside the V20 annotation bundle.

These declarations plus evidence roots are structurally stronger than a single `source_hidden=true` flag, but the validator still does not claim the retained evidence is authentic or truthful.

## Frozen policy identity

V22 semantic policy SHA-256:

`edb837213b56f4423bc096d3bb11c90dd07de03cf5b672398b1a17de2a6b6cc0`

It binds:

```text
V20 annotation policy
12a8431fd8a93dd8ba158758bb4089ef951960240fcab11a5178159e67f19bb9

V21 case-admission policy
452f4201c230ca3278d1ddc744fb70bca5944fa0d77b89e494096cfbf9c3dfb9
```

Role-session receipts use the actual NUL-terminated domain:

`symthaea-scip-text-claim-blinding-witness-v1\0`

and commit policy identity, exact V20 annotation receipt, V21 case-admission receipt reference, role slot, participant fingerprint, exact witness body, and a canonical role access-profile digest.

## Local execution

Before Git object construction, the exact final validator/harness pair was executed using the plain Python interpreter:

```text
python3 -m py_compile scripts/scip_text_claim_blinding_witness.py \
  scripts/test_scip_text_claim_blinding_witness.py
PASS

python3 -B scripts/test_scip_text_claim_blinding_witness.py
PASS_BLINDING_WITNESS_ADVERSARIAL
```

The hostile suite covers:

- participant fingerprint substitution;
- witness issuer = participant;
- annotation-receipt substitution;
- missing V21 admission receipt identity;
- source material injected into an extraction role;
- wrong surface digest;
- allowed-artifact reordering;
- incomplete audit capture;
- partial-session audit capture;
- late/unbound access-control snapshot;
- recorded forbidden access;
- aliased evidence roots;
- session/challenge identity aliasing;
- PII/shadow fields;
- nonexistent adjudicator role;
- duplicate JSON keys.

It also runs positive witnesses for extraction annotator, extraction adjudicator, alignment annotator, and alignment adjudicator roles.

Exact SHA-256 values:

```text
validator
13d899085c08d026f27ed9d84ae2e2f6dc2484622c00efc2d24a5d71d256e7d6

harness
2bcce011444d323e47529a4b2079729c6bbc131a78ea3e410ee58298a7a962c5

policy bytes
1c63dea703d3a1f80a6161d706359d1884e3ee4e2788b1df9610bf3962a5ff2c
```

Local Git blob identities:

```text
validator e39812a8e00545c3d8f31f46f1b48a70e0b7da03
harness   4c928f324c9e187698a61b34ef5dac126e42a352
policy    bf9f23a94a2096c90b05cc3c9d1be72796f9da58
```

## Positive claim and explicit nonclaims

A valid witness may report only:

```text
process_boundary_structurally_attested_under_witness=true
role_artifact_profile_matches_v20=true
evidence_roots_bound=true
case_admission_binding_present=true
```

It simultaneously reports:

```text
case_admission_receipt_recomputed=false
actor_authentication_established=false
audit_truthfulness_established=false
audit_completeness_in_reality_established=false
human_independence_established=false
human_expertise_established=false
human_correctness_established=false
surface_fidelity_established=false
confirmatory_execution_authorized=false
```

V22 therefore upgrades the evidence structure without upgrading semantic or operational authority.

## Next boundary

V23 should compose the complete required set of role-session witnesses for one V21-admitted case. It should require one witness for every human role actually present in the V20 bundle, require a common annotation receipt / case-admission receipt / public-claim-schema digest, reject missing or extra role sessions, and produce a **case-level process-evidence completeness receipt**.

That composition should still stop below evidence authenticity. A separate authenticated evidence boundary (potentially Xenia-backed) can later bind the retained session/access/audit artifacts without making Broca responsible for cryptographic identity.
