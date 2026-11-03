# Signed Dishcare Calibration Manifest — exploratory interoperability draft

Status: concept draft v0.1, 3 November 2026. Not a published standard, regulatory certification, patentability conclusion or FTO opinion. The scope and claims require review under issue #10219 before implementation or commercialization.

## Purpose

Allow a dishwasher controller to verify that a detergent formulation's permitted operating envelope is current and authentic without requiring the supplier to disclose the entire proprietary formula and without requiring a cloud connection for each cycle.

The manifest describes **what the appliance is permitted to do**, not how the formulation is chemically constructed. It is a signed calibration contract between a formulation supplier, appliance maker and test/evidence issuer.

This is a possible interoperability layer, not a claim that detergent profiles or automatic dosing are new. Existing patent and product prior art is summarized in [detergent-control-prior-art.md](detergent-control-prior-art.md).

## Trust model

A valid signature establishes that a holder of a trusted signing key signed the exact manifest bytes. It does **not** prove that the chemistry is safe, the calibration is accurate, the test was independent or the claimed cleaning result is true.

The appliance must verify:
- issuer/key identity against a locally provisioned trust store;
- signature over canonical manifest payload;
- schema version and supported features;
- issue/expiry time, monotonic manifest version and revocation state;
- exact product/formulation ID and formulation batch scope;
- appliance model/region/firmware compatibility;
- component and interaction evidence digests;
- all schedule conditions against local safety constraints.

Unknown key, invalid signature, expired or revoked manifest, version rollback, unsupported schema, missing evidence or out-of-envelope condition => **reject the adaptive plan and use the appliance's conservative approved programme**. Do not silently use stale cached calibration.

## Cryptographic encoding proposal

- UTF-8 JSON using the I-JSON constraints and deterministic canonicalization defined by RFC 8785 (JCS) before hashing/signing.
- SHA-256 for manifest and evidence-bundle digests.
- Ed25519 signatures using an established JOSE/COSE encoding profile rather than a custom cryptographic primitive.
- Key IDs, rotation and revocation handled by signed trust metadata installed by appliance firmware/service updates; no online lookup is required during a wash.
- Arrays with set semantics must be sorted by stable ID before canonicalization; array order must be preserved where it is semantically meaningful (such as timed dose events).
- All quantities use explicit units. Prefer integer fixed-point fields for dosing (e.g. milligrams) rather than binary floating-point JSON values.

Standards references:
- RFC 8785 — JSON Canonicalization Scheme: https://www.rfc-editor.org/rfc/rfc8785.html
- RFC 8037 — EdDSA/Ed25519 in JOSE (read together with its updates): https://www.rfc-editor.org/rfc/rfc8037.html
- RFC 9864 updates RFC 8037: https://www.rfc-editor.org/rfc/rfc9864.html

The exact JOSE or COSE envelope, protected headers, algorithm identifiers, key trust and revocation mechanism require a security review before implementation.

## Draft payload shape

This is a conceptual shape only; no schema is finalized.

```json
{
  "manifest_schema": "dishcare-calibration/0.1",
  "issuer": {
    "organization_id": "supplier-namespace",
    "key_id": "ed25519-key-id"
  },
  "manifest_id": "opaque-stable-id",
  "manifest_version": 1,
  "issued_at": "RFC3339-UTC",
  "expires_at": "RFC3339-UTC",
  "revocation_epoch": 1,
  "formulation": {
    "product_id": "opaque-product-id",
    "formulation_id": "opaque-formulation-id",
    "formulation_version": "supplier-version",
    "batch_scope": "lot-or-batch-scope",
    "confidential_composition": "not-disclosed"
  },
  "device_compatibility": {
    "markets": ["ZA"],
    "machine_models": ["exact-approved-model-id"],
    "firmware_ranges": ["signed-compatibility-range"]
  },
  "calibration_scope": {
    "soil_classes": ["supplier-defined-soil-class"],
    "surface_materials": ["ceramic", "glass"],
    "water_hardness_mg_l_as_caco3": {"minimum": 20, "maximum": 200},
    "phase_temperature_c": {"minimum": 35, "maximum": 65},
    "protocol_id": "protocol-revision"
  },
  "components": [
    {
      "component_id": "opaque-component-id",
      "public_role": "supplier-defined-role-label",
      "permitted_phases": ["main_wash"],
      "release_mass_mg": {"minimum": 100, "maximum": 2000},
      "cycle_mass_limit_mg": 2000,
      "calibration_evidence_sha256": "64-hex-digest"
    }
  ],
  "interaction_constraints": [
    {
      "component_ids": ["opaque-component-a", "opaque-component-b"],
      "minimum_release_separation_s": 120,
      "evidence_sha256": "64-hex-digest"
    }
  ],
  "evidence": [
    {
      "evidence_id": "opaque-evidence-id",
      "evidence_class": "supplier_lab|independent_lab|regulatory",
      "protocol_id": "protocol-revision",
      "archive_sha256": "64-hex-digest"
    }
  ],
  "signature": {
    "protected_algorithm": "EdDSA",
    "key_id": "ed25519-key-id",
    "signature_value": "base64url-signature"
  }
}
```

The example values are placeholders only; none is a recommended formulation, dose, compatibility rule or product approval.

## Confidentiality and interoperability

A manifest should expose the minimum safe information needed by the appliance:
- opaque component IDs rather than full ingredient composition;
- calibrated release mass and stage envelope;
- supported soils/surfaces and environmental constraints;
- evidence provenance and versioning;
- formulation-owner contact and recall/revocation metadata.

The formulation supplier may retain confidential concentrations, ingredient ratios, processing details and manufacturing know-how. The appliance needs permission boundaries and measured delivery ranges, not the full recipe.

Appliance makers must not treat a signed supplier manifest as overriding mechanical/electrical safety, user safety instructions, national regulations, rinse limits or their own independent product validation.

## Security and lifecycle threats

| Threat | Required mitigation |
|---|---|
| Manifest tampering | Canonical payload hash + signature verification |
| Untrusted supplier key | Firmware-provisioned trust store and signed key rotation |
| Expired/recalled formulation | Signed revocation metadata, expiry, conservative fallback |
| Rollback to unsafe old calibration | Persist highest accepted version / revocation epoch |
| Wrong region or machine | Exact model/region/firmware compatibility |
| Dose unit confusion | Explicit units and integer fixed-point values |
| Evidence substitution | Hash each evidence bundle; audit source records separately |
| False claims in a validly signed profile | Independent endpoint tests; signature is not proof of truth |
| Network/cloud outage | Offline local verification with cached trust state; stale state fails closed |
| Proprietary formula leakage | Opaque component identifiers and minimum necessary envelope disclosure |

## Relationship to InstantClean implementation

Current `chemistry::validate_dose_plan` consumes a `FormulationCalibrationProfile` with calibration/evidence digests, but does not verify a supplier signature or external evidence record. It is an in-process bounds checker only.

Do not rename the current object "signed" or "authenticated". The manifest verifier would be a separate layer:
1. parse and schema-validate the manifest;
2. canonicalize the unsigned payload;
3. verify signature against trusted issuer key;
4. verify version, expiry, revocation and device compatibility;
5. verify referenced evidence records where available;
6. map the permitted envelope into the local `FormulationCalibrationProfile`;
7. validate every proposed dose plan locally;
8. log the manifest digest, key ID, calibration version and fallback reason.

## Adoption questions before building

- Will formulation suppliers publish calibrated bounds and interaction evidence, or only marketing claims?
- What independent lab and protocol qualifies a manifest?
- Who operates key trust, rotation, revocation and recall?
- How are product versions and regional formulation changes represented?
- Can the manifest stay useful across appliance brands without exposing trade secrets?
- Which patent claims cover automatic dosing, formulation scheduling, calibration exchange or signed profiles?
- Would a neutral industry consortium, standards body or detergent/appliance partnership host the registry/trust anchors?

This draft is a basis for discussion. It should not be represented as a standard or patentable invention. FTO and interoperability review is tracked in issue #10219.
