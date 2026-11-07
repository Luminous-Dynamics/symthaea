# REA numeric intent profile v0 — design contract and independent fixtures

**Issue:** [#10989 — schema-bound numeric intent beyond JCS binary64](https://github.com/Luminous-Dynamics/symthaea/issues/10989)  
**Composition:** [#10920 JCS precision warnings](https://github.com/Luminous-Dynamics/symthaea/issues/10920), [#10927 producer warnings](https://github.com/Luminous-Dynamics/symthaea/pull/10927), [#10951 raw integer lexemes](https://github.com/Luminous-Dynamics/symthaea/issues/10951), [#10952 raw lexeme auditor](https://github.com/Luminous-Dynamics/symthaea/pull/10952), [#10958 precision-aware replay](https://github.com/Luminous-Dynamics/symthaea/pull/10958), and [#6283 measurement validity / units](https://github.com/Luminous-Dynamics/symthaea/issues/6283).

**Status:** design and fixture matrix only. This document does not implement numeric coercion, arithmetic, Evidence v1 changes, or claim-authority expansion.

## 1. The paradigm shift: four things that must not be collapsed

A JSON number participates in at least four distinct contracts:

1. **Canonical representation:** how a parsed JSON value is represented for interoperable hashing and signing. REA Evidence v1 currently uses RFC 8785 JCS and ECMAScript-compatible binary64 semantics.
2. **Source lexeme:** the exact token present in the original JSON bytes, such as 9007199254740993, 9.007199254740993e15, 9007199254740993.0, or the string "9007199254740993".
3. **Domain numeric intent:** whether a particular field means binary64, an exact bounded integer, an exact integer string, a fixed decimal string, or an opaque numeric token.
4. **Measurement meaning:** the quantity kind, unit, coordinate/frame, calibration, scale and allowed comparison/conversion policy.

These commitments answer different questions. A JCS digest does not preserve every source integer lexeme. A raw-token audit does not tell us whether the field is a counter, a monetary amount, a physical measurement or an opaque identifier. A numeric type does not establish unit correctness or calibration. No model/provider explanation may infer the missing contract.

**Core rule:** numeric intent is declared by a versioned profile at an exact JSON Pointer and bound to the exact profile/schema digest. It is never inferred from magnitude, token spelling, provider confidence, provider name, or apparent semantic usefulness.

## 2. Compatibility and non-negotiable boundaries

- Do not change REA Evidence v1 semantic IDs or its established ECMAScript binary64/JCS behavior.
- Do not reject an upstream-compatible Evidence record as a side effect of precision diagnostics.
- Do not coerce exact integer or decimal strings through binary64.
- Do not infer numeric intent from an unsafe-integer warning. The warning establishes a lexical risk, not a domain type.
- A parsed serde_json::Value does not prove that raw-token coverage occurred. A Value-only input has lexical coverage **unavailable** unless tied to the exact raw source digest and an audit receipt.
- Every raw-token warning binds to the exact source SHA-256, RFC 6901 JSON Pointer, original integer lexeme and numeric-intent profile digest when a profile is involved.
- Preserve unit, scale, signedness, range and profile version through comparison. 1000 m is not 1000 ft; a quantity with a unit is not a dimensionless scalar.
- Do not silently round, normalize, clamp, reinterpret or convert a value. Any transformation must be an explicit, versioned operation with its own result and claim ceiling.
- Reuse the repository's existing quantity/unit/frame contracts under #6283. This design does not introduce another unit ontology.

RFC 8785 defines canonical JSON serialization using ECMAScript-compatible IEEE-754 binary64 number behavior; it does not declare a domain-specific exact integer or decimal type. RFC 7493 says interoperable JSON receivers cannot be expected to treat integers outside ±(2^53−1) as exact and recommends strings for exact interchange outside that range. These constraints are compatible: retain JCS as the existing canonicalization contract, and make domain intent a separate typed validation contract. [RFC 8785](https://www.rfc-editor.org/rfc/rfc8785.html) · [RFC 7493 §2.2](https://www.rfc-editor.org/rfc/rfc7493.html)

## 3. Numeric intent algebra

V0 defines five intent classes. A rule applies to one exact RFC 6901 JSON Pointer. Wildcards, array templates, ancestor inheritance and automatic path inference are not part of v0.

| Intent | JSON representation | Authorized semantics | Forbidden inference |
|---|---|---|---|
| binary64 | JSON number | ECMAScript/IEEE-754 binary64 value, under declared finite-value, negative-zero, exponent, significant-digit, unit and comparison policy | Exact preservation of source decimal or integer spelling |
| safe_integer | JSON integer token | Exact signed/unsigned integer inside the declared range, which must be contained in ±(2^53−1) | Accepting an out-of-range token because JCS rounded it to a binary64 value |
| exact_integer_string | JSON string | Exact decimal integer under an explicit grammar, signedness, range/width and optional unit contract | Converting it to a JSON number or binary64 before validation |
| fixed_decimal_string | JSON string | Exact base-10 coefficient/scale under a declared precision, scale, trailing-zero, rounding and unit/currency contract | Treating numerically similar strings with different scale as interchangeable unless the profile explicitly permits it |
| opaque_numeric | JSON number or string | Retain the exact source value/token for transport and audit only | Arithmetic, ordering, equality-as-quantity, unit conversion, or domain conclusion |

### 3.1 binary64

A binary64 rule declares that binary floating-point interpretation is intended. It must identify policies for negative zero, exponent-form input, underflow, overflow, maximum significant digits and comparison. If a tolerance is used, the comparison mode and tolerance representation must be declared in the profile; tolerance values themselves should be decimal strings to avoid introducing an unreviewed binary64 conversion.

JCS equality is meaningful only between values whose numeric intent is compatible. The profile must not silently compare a binary64 field with an exact decimal string or an exact integer field.

### 3.2 safe_integer

V0 requires an integer-form JSON token (-?(0|[1-9][0-9]*)) and an explicit signedness and inclusive min/max range. The declared range must be a subset of [-9007199254740991, +9007199254740991]. Fractional and exponent spellings such as 42.0 and 4.2e1 are rejected by this lexical contract even though they may represent an integral mathematical value after parsing. This is intentional: the field declares an exact integer token, not generic binary64 numeric equivalence.

### 3.3 exact_integer_string

The field must be a JSON string containing canonical base-10 integer syntax. The profile declares signedness, optional bit width, minimum/maximum decimal strings and unit reference. Leading plus, leading zeroes, whitespace, exponent notation and -0 are invalid unless a future profile version explicitly adds those forms. The string is never parsed through f64.

### 3.4 fixed_decimal_string

The field must be a JSON string under an explicit decimal grammar. The profile declares maximum precision, exact scale, trailing-zero policy, rounding mode for any separately authorized calculation, range and unit/currency contract. Validation preserves the original string and scale. V0 validation does not round a source value to make it fit; a value requiring quantization is invalid or underdetermined under the selected policy.

### 3.5 opaque_numeric

The source value is retained, but the profile explicitly grants no arithmetic or numeric-equivalence authority. This is useful for protocol values whose interpretation belongs to a later, unavailable schema. An opaque token must not enter aggregation, threshold checks, ranking or arithmetic merely because it parses as a JSON number.

## 4. Profile binding and validation result

The machine-readable profile schema is docs/research/rea-numeric-intent-profile-v0.schema.json.

A profile binds:
- profile ID and version;
- exact source schema ID and SHA-256;
- a closed set of exact JSON Pointer rules;
- the numeric intent and type-specific grammar/range/policy;
- unit/quantity references to existing contracts when applicable;
- a JCS SHA-256 profile digest.

profile_sha256 is computed over RFC 8785 canonical bytes of the profile object with the profile_sha256 field omitted. This is an integrity commitment, not a signature, trust anchor or proof that the profile is authoritative. The caller must separately establish why this profile is the correct one for the subject/operation.

A future validator result should distinguish at least:
- valid: present value conforms to the selected rule and all required source/profile bindings are established;
- invalid: a present value violates a declared grammar, type, range, scale, unit or policy;
- unverifiable: required raw-token coverage, source binding, profile authority or unit contract is missing;
- not_applicable: the requested path/predicate does not apply to the supplied subject/scope;
- opaque: value was preserved but no numeric arithmetic/equivalence authority is granted.

Missing and explicit null are distinct. A missing required path is unverifiable or invalid under the profile's explicit missing policy; it is never silently replaced by zero. Explicit null is handled by a declared null policy and never interpreted as a numeric zero.

## 5. Comparison and conversion rules

| Left | Right | Permitted comparison in v0 |
|---|---|---|
| Same profile + binary64 | Same profile + binary64 | JCS/binary64 equality or the profile-declared tolerance; units and frame must match |
| Same profile + safe_integer | Same profile + safe_integer | Exact integer comparison after lexical/range validation |
| Same profile + exact_integer_string | Same profile + same grammar/range | Exact decimal-string integer comparison; no floating conversion |
| Same profile + fixed_decimal_string | Same profile + same scale/rounding | Exact coefficient/scale comparison; no implicit stripping of trailing zeroes |
| opaque_numeric | Any numeric type | No numeric comparison authority |
| Different intent classes | Each other | No implicit equivalence, promotion or coercion |
| Same numeric intent, different units | Each other | Incompatible unless an existing versioned conversion contract explicitly authorizes conversion and binds the result |

A numeric profile does not provide a new unit registry. It references an existing unit/quantity contract with an ID, version and digest. Conversion requires that contract, source and target units, conversion method, rounding behavior and conversion receipt all be bound.

## 6. Adversarial fixture matrix

The independent, machine-readable fixture matrix is docs/research/rea-numeric-intent-fixtures-v0.json. Expected outcomes are authored as protocol expectations, not generated by the producer validator. Keep it separate from the raw-lexeme auditor's own parser tests.

Required fault families:

1. Safe integer boundaries ±(2^53−1), first unsafe integers ±2^53, u64::MAX, i64::MIN and arbitrary-length integer tokens.
2. Integer token versus identical digits in a decimal string.
3. Integer token versus exponent token (1000 versus 1e3) under binary64 and exact-integer profiles.
4. 0.1, 0.10000000000000001, decimal midpoints/ties and high-scale fixed decimals.
5. Negative zero versus positive zero under each declared policy.
6. Exponent overflow, underflow, smallest subnormal and exponent boundaries.
7. Decimal trailing zeroes and declared scale mismatch.
8. Exact integer strings with leading zeroes, plus sign, negative zero, range overflow and wrong signedness.
9. Unknown/mismatched profile ID, version or digest.
10. Duplicate profile paths and escaped RFC 6901 pointer keys.
11. Missing, null, array/object substitution and unexpected fields.
12. Unit mismatch or conversion without a declared conversion policy.
13. Value-only Evidence with no raw source digest: lexical coverage remains unavailable.
14. Raw-token audit report whose source SHA-256 does not match the exact Evidence bytes.
15. A syntactically valid but unauthorized schema profile: valid shape is not profile authority.

## 7. Benchmark and release targets

These are proposed finite-suite gates, not achieved results.

| Dimension | Target |
|---|---:|
| Profile schema and semantic validation | 100% expected result on frozen profile fixtures |
| Exact integer-string preservation | 100%; zero accepted binary64 coercions |
| Fixed-decimal grammar/scale | 100% exact lexical/scale compliance |
| Binary64 compatibility | 100% parity with pinned upstream JCS behavior |
| Raw unsafe integer detection | 100% recall on frozen lexeme corpus |
| False numeric-intent inference | 0% |
| Profile/source/schema digest binding | 0 wrong-profile positive results |
| Unit/frame mismatch | 100% detection on declared mismatch controls |
| Missing/null distinction | 100% expected-result parity |
| Value-only lexical coverage | 100% explicitly unavailable unless raw source/audit binding exists |

Report per intent/path/fault family. Keep schema accuracy, exactness, source coverage, unit compatibility and resource cost separate. Do not create one composite numeric score.

For implementation performance, measure profile validation separately from raw lexeme scanning, Evidence identity verification, JCS canonicalization and claim replay. Report p50/p95/p99, allocations, peak RSS, bytes parsed and exact toolchain/build/workload metadata. Correctly abstaining on unavailable lexical coverage is preferable to a fast false exactness claim.

## 8. Industry and standard comparison

- **RFC 8785 / JCS:** interoperable binary64 canonical representation; not a domain exact-decimal contract. The JCS document explicitly discusses binary64 semantics and recommends strings for use cases requiring more precision.
- **RFC 7493 / I-JSON:** describes the safe integer range and recommends strings when exact interchange requires values outside the binary64-safe range.
- **RFC 8949 / CBOR:** provides bignum tags and decimal-fraction tag 4 (m × 10^e) and bigfloat tag 5 (m × 2^e). Those representations show that exact numeric encodings are possible, but domain intent, unit semantics and conversion policy still require an application-level contract.
- **JSON Schema / OpenAPI:** useful for constraining JSON types and ranges, but number alone does not specify whether a field is an exact decimal, measurement, counter, binary64 quantity or opaque numeric token. The profile must make those meanings explicit and versioned.
- **IEEE 754:** defines floating-point representations and arithmetic behavior. It does not assign domain units, measurement frames, currency scale or business meaning.

References:
- [RFC 8785 — JSON Canonicalization Scheme](https://www.rfc-editor.org/rfc/rfc8785.html)
- [RFC 7493 §2.2 — I-JSON numbers](https://www.rfc-editor.org/rfc/rfc7493.html)
- [RFC 8949 §3.4.3–3.4.4 — CBOR bignums, decimal fractions and bigfloats](https://www.rfc-editor.org/rfc/rfc8949.html)

## 9. Implementation sequence

1. Freeze this profile schema and independent fixture expectations.
2. Independently validate schema/profile semantics, exact JSON Pointer handling, duplicate declarations and profile digest.
3. Validate values using raw-token source coverage from #10952; Value-only inputs must return lexical coverage unavailable.
4. Add type-specific parsers/comparators without changing the REA Evidence v1 identity function.
5. Add unit/frame composition through #6283's existing contracts.
6. Only then consider a separate, versioned claim predicate/rule that consumes a validated numeric-intent result. Do not retrofit these rules into generic v0 claim evaluation.

## 10. Claim ceiling

A passing numeric-intent validator establishes only that a raw field conforms to a declared profile's representation, grammar, range, scale and referenced unit contract. It does not establish that the source value is true, that a measurement was calibrated correctly, that the profile is authoritative, that the producer is authentic, or that a downstream domain conclusion follows.
