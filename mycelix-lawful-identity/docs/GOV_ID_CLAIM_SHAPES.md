# Government-ID Claim-Shape Conventions

The existing [`mycelix-identity/crates/eidas-zkp/`](../../mycelix-identity/crates/eidas-zkp/) crate handles the credential envelope (W3C VC 2.0 + DASTARK + Dilithium5 + Merkle selective disclosure) and accepts arbitrary `serde_json::Value` claims. This document defines the **claim-key conventions** for government-ID-backed credentials so different issuers and verifiers agree on shape.

No new Rust code is needed for these conventions — they are a shared schema convention, nothing more.

---

## Passport (ICAO 9303)

Credential `type` MUST include `"PassportCredential"` alongside `"VerifiableCredential"` and `"EidasCredential"`.

Claim shape:

```json
{
  "documentType": "P",
  "issuingCountry": "USA",
  "documentNumber": "123456789",
  "givenName": "Alice",
  "familyName": "Smith",
  "nationality": "USA",
  "dateOfBirth": "1990-03-15",
  "sex": "F",
  "dateOfIssue": "2022-05-10",
  "dateOfExpiry": "2032-05-10",
  "placeOfBirth": "San Francisco, CA, USA",
  "portraitImage": "data:image/jpeg;base64,...",
  "mrzLine1": "P<USASMITH<<ALICE<<<<<<<<<<<<<<<<<<<<<<<<<<<",
  "mrzLine2": "1234567890USA9003154F3205102<<<<<<<<<<<<<<00"
}
```

**Selective-disclosure recommendation**: almost all fields should be presented via `ProvenClaim` (range / membership / equality) rather than plaintext. Verifier almost never needs the document number; they need "holder is ≥ 18" or "holder's nationality is in set {US, CA, MX}".

---

## Mobile Driver's License (mDL, ISO 18013-5)

Credential `type` MUST include `"MobileDriversLicense"`.

Claim shape (per ISO 18013-5 data elements):

```json
{
  "family_name": "Smith",
  "given_name": "Alice",
  "birth_date": "1990-03-15",
  "issue_date": "2022-05-10",
  "expiry_date": "2027-05-10",
  "issuing_country": "USA",
  "issuing_authority": "California DMV",
  "document_number": "DL123456789",
  "portrait": "data:image/jpeg;base64,...",
  "driving_privileges": [
    {
      "vehicle_category_code": "C",
      "issue_date": "2010-03-15",
      "expiry_date": "2027-05-10"
    }
  ],
  "un_distinguishing_sign": "USA",
  "administrative_number": "AN12345",
  "sex": 2,
  "height": 165,
  "weight": 60,
  "eye_colour": "brown",
  "hair_colour": "brown",
  "birth_place": "San Francisco, CA",
  "resident_address": "1 Main St, San Francisco, CA",
  "portrait_capture_date": "2022-05-10",
  "age_over_18": true,
  "age_over_21": true
}
```

**Selective-disclosure recommendation**: mDL was designed for selective disclosure. Ship convenience ProvenClaims like `age_over_18`, `age_over_21`, `has_class_C_license` without touching other fields.

---

## US SSN-equivalent (`schema.org/PersonalIdentification`)

**Warning**: SSN is extraordinarily sensitive. Recommend against embedding the raw number in any credential. Preferred pattern is an "SSN-derived identity attestation" where a trusted intermediary (SSA-approved KYC provider) attests "this holder is a uniquely-identified US person" without transmitting the SSN.

Credential `type`: `"SsnDerivedAttestation"` (custom; not a passport or mDL).

Claim shape:

```json
{
  "attestationType": "ssn_derived",
  "attestingAuthority": "did:web:ssa-approved-kyc.example",
  "attestationDate": "2026-04-18",
  "holderHash": "b3sum256:7a8b9c...",
  "isUsPerson": true,
  "usTaxResident": true,
  "ssnSuffixLastFour": null,
  "expiresAt": "2027-04-18"
}
```

Never:
- `"ssn": "123-45-6789"` (plaintext SSN) — rejected by CLI at import time.
- `"ssnFull": ...` — rejected.

The `holderHash` is a BLAKE3 hash of the SSN with a per-attestor salt, suitable for deduplication ("same person across attestations") without revealing the SSN.

---

## SA ID Number (SA-specific)

South African green-bar-coded or smart-card ID. Credential `type`: `"SaIdCredential"`.

```json
{
  "id_number": "9003155800087",
  "given_name": "Alice",
  "family_name": "Smith",
  "date_of_birth": "1990-03-15",
  "sex": "F",
  "citizenship": "ZA",
  "place_of_issue": "Johannesburg",
  "date_of_issue": "2014-03-20"
}
```

First 6 digits of `id_number` encode DOB; digits 7-10 encode sex; digit 11 encodes citizenship (0=SA, 1=permanent resident). Verifiers can derive `date_of_birth`, `sex`, and `is_sa_citizen` via `ProvenClaim` without revealing the full number.

---

## EU eIDAS 2.0 (EUDI Wallet)

The existing `eidas-zkp` crate already handles this format natively. Claim keys follow the Person Identification Data (PID) schema per the EU Architecture Reference Framework:

```json
{
  "family_name": "Smith",
  "given_name": "Alice",
  "birth_date": "1990-03-15",
  "birth_place": "Berlin, Germany",
  "nationality": "DE",
  "resident_address": "...",
  "gender": 2,
  "age_over_18": true
}
```

---

## Required `ProvenClaim` types (reused from `eidas-zkp`)

| `proof_type` field | What it proves | Use case |
|---|---|---|
| `"range"` | value ∈ [min, max] | `age >= 21`, `latitude ∈ [42.3, 42.4]` |
| `"equality"` | value == constant | `nationality == "USA"` (when issuer-certified) |
| `"membership"` | value ∈ published set | `issuing_country ∈ {"USA", "CAN", "MEX"}` |

Verifier UIs SHOULD present `description` text ("age ≥ 21", "US / Canadian / Mexican citizen") rather than raw proof bytes.

---

## Forbidden claim keys

The `legal-did` zome validation layer rejects credentials containing:

- Any field named `private_key`, `secret_key`, `seed`, `mnemonic`, `recovery_phrase`
- Raw SSN fields (`ssn`, `social_security_number`, `socialSecurityNumber`) — must go through attestation pattern above
- Biometric raw template data (`fingerprint_raw`, `iris_raw`, `face_template_raw`) — rejected even if base64 — only hashes of biometrics
- Signed statements that could be replayed as general-purpose signatures (protection against credential-based identity hijack)

---

## Issuer DID conventions

See `zomes/issuer-trust-tier/` for the three-tier scheme:

| Tier | Example | Who sets it |
|------|---------|-------------|
| Sovereign | `did:web:state.gov`, `did:web:home.affairs.gov.za`, `did:web:gov.uk` | User (never canonical) |
| RegulatedIntermediary | `did:web:jumio.com`, `did:web:onfido.com` | User |
| Peer | anything else | Default |

Verifiers declare which tier they require. No tier influences Mycelix governance weight.
