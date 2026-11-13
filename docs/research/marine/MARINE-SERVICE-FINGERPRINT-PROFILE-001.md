# MARINE-SERVICE-FINGERPRINT-PROFILE-001
## Canonical transport-service identity v1

**Profile ID:** `marine-service-payload-v1+symthaea-canonical-json-sha256-v1`  
**Status:** implemented in validator source; Rust tests committed but not executed  
**Schema:** marine propulsion case v0.2.0  
**Parent:** #11406  
**Implementation:** #11412  
**Workstream:** #11443

## 1. Purpose

A service fingerprint commits to the declared transport service and route/loading requirements used to compare vessel architectures. It prevents records from being called service-matched merely because they repeat an arbitrary 64-character string.

It does not prove that the declared service is factually true, that a route/weather dataset is authentic, or that the vessel can physically deliver the service. It is a deterministic commitment to a normalized payload.

## 2. Canonical payload projection

The payload contains exactly these fields:

- `fingerprint_profile`: the profile ID above.
- `cargo_class`, `functional_unit`, `service_contract_id`, `arrival_policy`.
- Optional `declared_other_unit`.
- Service quantities when declared: `annual_delivered_work`, `annual_delivered_cargo_mass`, `annual_delivered_teu`, `required_arrival_window_hours`, `service_availability_target`. Quantity projection keeps only `value`, `unit`, `unknown_reason` or `not_applicable_reason`; provenance and uncertainty evidence remain in the full case and are not part of commercial service identity.
- `loading_condition` from the vessel record.
- `route_id` and `route_revision`.
- Ordered `segments`: `segment_id`, `speed_policy`, required `distance`, optional `target_speed`, optional `arrival_time_window`, and optional `port_or_waypoint_id`. Quantities use the same value/unit/reason projection.

Array order is significant: route-segment order represents the service sequence. Object key order is not significant. Absent optional fields remain absent; they are not silently turned into zero or a placeholder value.

## 3. What is intentionally outside this identity

Metocean realization, forecast revision, wind/wave/current traces, fuel pathway, candidate propulsion architecture, WAPS device state, powertrain configuration, machinery revision and trial measurements are not part of this commercial service fingerprint.

Those belong to separate environment/counterfactual, configuration, evidence and trial identities. Pair comparison must still check the required environmental and measurement boundaries independently. Excluding weather from this service identity does not make unlike weather conditions comparable.

## 4. Quantity-state semantics

For any projected quantity:

- A numeric zero is a measured/declared numeric value and hashes as zero.
- Unknown data use `value: null` plus non-empty `unknown_reason`.
- Explicit non-applicability uses `value: null` plus non-empty `not_applicable_reason`.
- An omitted optional field remains absent from the payload.
- The schema forbids a quantity from carrying both reason fields in the same union alternative.

These states yield different payloads and fingerprints. Unknown remains incomplete for evidence qualification; a valid not-applicable declaration is not automatically incomplete, but required service denominator quantities cannot be dismissed as not applicable and still be scored.

## 5. Encoding and digest

1. Construct the projection above as a JSON value.
2. Serialize using Symthaea's existing `symthaea-canonical-json-sha256-v1` algorithm:
   - object keys sorted lexicographically;
   - arrays preserved in order;
   - strings and keys escaped using serde_json-compatible JSON string serialization;
   - null and booleans use JSON literals;
   - numbers use the serde_json `Number::to_string()` representation.
3. SHA-256 the resulting UTF-8 bytes.
4. Encode the 32-byte digest as 64 lowercase hexadecimal characters.

This profile is **not RFC 8785/JCS**. Do not label it JCS or assume another language's generic JSON serializer is byte-identical. Any non-Rust implementation must pass the published byte-for-byte vectors. Numeric range/representation policy must be revisited before accepting values that cannot be represented exactly by the consumer language (in particular integers beyond IEEE-754 exact range).

The implementation deliberately reuses the same canonical algorithm already present in `crates/domains/symthaea-muse/src/evidence_digest.rs`. The isolated marine validator carries the same implementation to avoid a dependency on the larger domain crate.

## 6. Known-answer vector

See [service-fingerprint-v1-vector-0001.json](./fixtures/service-fingerprint-v1-vector-0001.json).

The synthetic container-service vector has digest:

`e3ec23f5308213b6d7ea3743e6cf483623436bec5384e29d3044d525d736fca9`

The case fixture is synthetic and must never be treated as a real ship, route, service or performance record.

## 7. Acceptance and adversarial tests

- 100% agreement with published known-answer vectors across Rust and at least one independent implementation.
- Object-key reordering does not change the digest.
- Route-segment reordering changes the digest.
- Mutating cargo class, functional unit, contract, delivered work, availability, arrival window, segment distance/speed, port sequence or loading condition changes the digest.
- Numeric zero, unknown, not-applicable and omitted quantity states remain distinct.
- Any unsupported profile version or digest mismatch is INVALID.
- Legacy v0.1.0 cases with no profile remain INCOMPLETE; do not infer a profile or rehash historical evidence without migration.
- No service digest grants physical, operational, safety, procurement or actuation authority.

## 8. Current implementation boundary

Schema v0.2.0 requires the profile field. The Rust validator recomputes the payload hash during case validation and tests the synthetic known-answer vector and mutation behavior. Full pair-comparison adversarial coverage, cross-language execution and Rust test execution remain pending.
