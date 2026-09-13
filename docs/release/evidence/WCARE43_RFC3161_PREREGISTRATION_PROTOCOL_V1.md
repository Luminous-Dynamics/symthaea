# WCARE-43 — RFC 3161 preregistration verifier protocol v1

Status: `SOURCE_REVIEW_CANDIDATE`
Authority: `MeasurementOnly`
Verifier protocol: `wcare43-rfc3161-preregistration-verifier-v1`

## Purpose

WCARE-41 separates builder-evidence authentication from temporal preregistration. WCARE-43 implements only the temporal channel using RFC 3161 Time-Stamp Protocol verification through an exact, plan-bound OpenSSL backend.

The governing distinctions are:

`timestamp string != verified timestamp token`

`verified timestamp token != universally trusted time authority`

`builder signature validity != preregistration precedence`

No WCARE-43 artifact grants runtime authority.

## Exact subject

The temporal subject is the exact byte SHA-256 of the WCARE-40 replication plan. A qualifying verification binds:

- exact WCARE-41 authentication-plan bytes and SHA-256;
- exact WCARE-43 verifier bytes and SHA-256;
- exact WCARE-40 plan bytes and SHA-256;
- exact WCARE-40 result bytes and SHA-256;
- exact WCARE-41 temporal proof-package bytes and SHA-256;
- exact RFC 3161 timestamp response bytes and SHA-256;
- exact WCARE-43 backend-policy bytes and SHA-256;
- exact trust-anchor bundle bytes and SHA-256;
- exact untrusted/TSA certificate bundle bytes and SHA-256 when present;
- exact OpenSSL executable SHA-256 and version-output SHA-256;
- exact qualifying WCARE-39 FINAL capsule bytes used to derive execution start times.

Similar content or filenames do not substitute for exact bytes.

## Two-layer verifier provenance

WCARE-43 deliberately separates orchestration identity from cryptographic-backend identity.

The WCARE-41 authentication plan and temporal proof package bind:

- the exact WCARE-43 backend ID;
- the exact SHA-256 of `scripts/wcare43_rfc3161_verify.py`;
- the exact WCARE-43 backend-policy SHA-256.

The WCARE-43 backend policy separately binds:

- exact OpenSSL executable SHA-256;
- exact `openssl version` output SHA-256;
- exact trust-anchor bytes;
- exact untrusted/TSA certificate bytes;
- exact expected TSA signer-certificate identity;
- required SHA-256 message-imprint algorithm;
- required `timestampsign` certificate purpose;
- explicit revocation mode.

Changing either the WCARE-43 verifier or OpenSSL backend creates a different evidence subject.

## Backend execution profile

v1 uses the system OpenSSL `ts` implementation rather than implementing CMS/RFC 3161 cryptography in Symthaea.

The verifier MUST:

1. locate the OpenSSL executable without a shell;
2. hash the exact executable bytes;
3. capture and hash exact `openssl version` output;
4. require those identities to match the preregistered WCARE-43 backend policy;
5. invoke all OpenSSL commands by argv array with `shell=False` semantics;
6. bind the exact trust material bytes before verification.

An unavailable backend is `INDETERMINATE`. An executable/policy identity mismatch is `INVALID`.

## RFC 3161 verification sequence

The timestamp response is never trusted because it parses successfully.

The verifier first extracts these **candidate** fields from `openssl ts -reply -in RESPONSE -text`:

- `genTime`;
- TSA policy label;
- message-imprint hash algorithm.

These parsed fields are not yet evidence.

The token MUST declare `Hash Algorithm: sha256`. Any other algorithm is `INVALID` under v1, even if it has the same output length.

The verifier then runs the exact equivalent of:

`openssl ts -verify -digest PLAN_SHA256 -in RESPONSE -CAfile TRUST -untrusted UNTRUSTED -purpose timestampsign -attime GEN_TIME_EPOCH`

with `-untrusted` omitted only when the bound policy explicitly permits an empty untrusted bundle.

Only a successful cryptographic verification under the exact trust material may promote candidate `genTime` to a verified commitment time.

This sequence binds the timestamp token to the exact SHA-256 plan digest and verifies the token signer under the supplied trust chain at the token's signed time. A wrong plan digest, modified response, malformed response, invalid signature or unusable certificate chain must not establish time.

## Proof-package binding

The WCARE-41 temporal proof package must bind:

- exact WCARE-40 plan SHA-256;
- exact WCARE-43 backend ID;
- exact WCARE-43 verifier SHA-256;
- exact WCARE-43 backend-policy SHA-256;
- exact RFC 3161 response SHA-256;
- exact TSA service-identity commitment;
- the commitment time copied from the timestamp token;
- `proof_format = rfc3161-timestamp-response-der`;
- `proof_version = rfc3161-v1`.

The package's timestamp is not trusted independently. It must equal the token's signed `genTime` after parsing, and only becomes evidentiary after token verification succeeds.

## TSA/service identity

Trust-chain success alone does not establish that the signer is the specific TSA identity allowed by the WCARE-43 policy.

After token verification, the verifier extracts the timestamp token and embedded certificates using exact OpenSSL commands. It identifies timestamp-signing certificate candidates by `Time Stamp signing : Yes` and requires exactly one candidate.

The DER SHA-256 of that certificate MUST equal:

- the policy's expected TSA signer-certificate DER SHA-256; and
- the WCARE-41 temporal proof package's `service_identity_commitment_sha256`.

Otherwise temporal preregistration is not established.

This is a bounded certificate-identity claim under the supplied trust material. It does not establish universal or institutional trust in the TSA.

## Certificate validity and revocation boundary

v1 requires certificate-path verification at the signed candidate `genTime` via OpenSSL `-attime` and explicit `-purpose timestampsign`.

The initial profile supports only:

`revocation_mode = OfflineStaticNoRevocation`

Therefore v1 makes **no claim that live certificate revocation status was checked**. The result must keep:

- `revocation_status_checked = false`;
- `global_tsa_trust_established = false`.

A future CRL/OCSP profile must be separately specified and qualified rather than silently strengthening this profile.

## Replica-start binding

WCARE-43 may establish preregistration only for a WCARE-40 result with disposition `REPLICATION_SUPPORTED`.

For every `subject_eligible_replica_id` in that result, the verifier requires the exact WCARE-39 FINAL capsule whose SHA-256 equals `final_capsule_sha256_by_replica[replica_id]`.

Each required capsule must be a WCARE-39 FINAL capsule with:

- `classification = QUALIFIED_EXECUTION`;
- `environment_integrity = QUALIFIED`;
- at least one non-null command `started_utc`.

The replica execution start is the earliest non-null `commands[].started_utc` in that exact FINAL capsule.

Let `T_commit` be the verified RFC 3161 commitment time and `T_i` each qualifying replica start. WCARE-43 requires:

`T_commit < T_i` for every qualifying replica.

Equality is not sufficient.

## Synthetic fixtures

A WCARE-43 backend policy includes `synthetic_fixture_policy`.

When it is `true`, a fixture may prove that:

- RFC 3161/CMS verification succeeds;
- the exact plan imprint is verified;
- certificate-time validation succeeds;
- TSA signer identity matches the synthetic policy;
- strict temporal precedence holds.

But the final production disposition MUST remain `NOT_ESTABLISHED`, with:

- `external_temporal_authority_established = false`;
- `preregistration_temporal_precedence_established = false`.

Synthetic local-CA/TSA fixtures therefore test mechanics without manufacturing production preregistration evidence.

For a non-synthetic preregistered policy, `ESTABLISHED` may set `external_temporal_authority_established = true` only for the exact TSA certificate/trust policy under evaluation. `global_tsa_trust_established` still remains false.

## Dispositions

WCARE-43 yields exactly one primary disposition:

- `ESTABLISHED` — a non-synthetic, policy-authorized exact plan has a verified RFC 3161 commitment strictly before every qualifying replica start in a supported WCARE-40 result;
- `NOT_ESTABLISHED` — cryptographic token verification succeeded, but synthetic-policy, TSA identity/policy, strict temporal-precedence or WCARE-40 support requirements were not met;
- `INDETERMINATE` — the exact required verifier/backend cannot execute or required environmental verifier capability is unavailable;
- `INVALID` — malformed/binding-invalid evidence, wrong digests, wrong backend/policy identity, non-SHA256 token profile, malformed token, failed signature/imprint/trust verification, missing required capsules or other structural evidence failure.

## Generic timestamps are insufficient

None of these can establish preregistration:

- WCARE-40 `plan_created_utc`;
- Git author or committer timestamps;
- filesystem timestamps;
- JSON timestamp strings;
- an unverified RFC 3161 `genTime`;
- a self-authored receipt.

## Claim boundary

WCARE-43 may establish only bounded temporal precedence under the exact verifier, exact policy, exact TSA certificate identity and exact trust material supplied.

It does not establish builder authentication, reviewer independence, subject correctness, universal TSA trust, live revocation status, network/sandbox isolation, consciousness, phenomenal experience, suffering, moral patienthood, binding consent, veto/self-preservation authority or solved alignment.
