# REA-RUST-044 — schema-bound numeric intent beyond JCS

**Status:** design contract and known-answer fixture matrix only. No production numeric validator, REA Evidence v1 change, claim-authority expansion or performance result is included.

## Executive thesis

The precision work has now separated three previously conflated questions:

1. **Raw token:** which exact bytes and numeric lexeme appeared in the source?
2. **Canonical representation:** what value does the existing ECMAScript binary64 + RFC 8785 JCS contract commit to?
3. **Domain intent:** what is this field supposed to mean—an IEEE-754 measurement, an exact counter, an exact decimal amount, or an opaque number?

The first is owned by the raw numeric-lexeme audit (#10952) and its fuzz lane (#10986). The second remains the REA Evidence v1 compatibility contract and the precision-aware claim/replay work (#10927/#10958). Neither determines the third.

**Paradigm shift:** numbers are not typed by their magnitude or spelling. Their meaning is declared by a versioned schema/profile at an exact data path.

## Ownership and non-duplication

This contract composes:

- RFC 8785/JCS and the existing REA Evidence v1 identity contract;
- RFC 7493 safe-integer interoperability guidance;
- #10920 precision warning semantics;
- #10951/#10952 raw integer-token preservation;
- #10958 independent replay of precision-aware claims;
- #5662 ENG-UNITS-001 and #4861 SE-SEM-001 for canonical unit, quantity-kind and frame semantics;
- #6283 for measurement validity and uncertainty.

It does **not** create a new quantity ontology, unit registry, measurement/calibration model, Evidence identity, generic claim rule, or cryptographic receipt.

The unit and quantity references in a profile are opaque, versioned references to the existing canonical owners. Until those owners publish a stable identifier for a requested kind/unit, the profile must say "unsupported", not introduce a local spelling as an authoritative unit code.

## 1. Numeric-intent algebra v0

The profile declares one interpretation per exact RFC 6901 JSON Pointer. Wildcards and heuristic path matching are out of scope in v0.

| Intent | Input meaning | Permitted comparison | Explicit non-claim |
|---|---|---|---|
| binary64 | IEEE-754 binary64 with RFC 8785/ECMAScript serialization | JCS canonical equality under the pinned profile | Not exact decimal/integer preservation |
| safe_integer | JSON integer-form token in the declared range, constrained to ±(2^53−1) | Exact integer comparison inside the safe range | No acceptance of exponent/decimal lexical forms as integer tokens |
| exact_integer_string | String carrying a signed/unsigned decimal integer under a declared grammar/range | Exact decimal-integer string comparison | No conversion to binary64 |
| fixed_decimal_string | String carrying exact base-10 digits with declared scale/range | Exact lexical/scale comparison under the profile | No hidden rounding or unit conversion |
| opaque_numeric | Number-like source value whose domain meaning is intentionally unknown | No arithmetic, ordering or equivalence inference | Not a numeric measurement or comparable scalar |

The five intents are not an ordering of confidence or truth. A value cannot be promoted from opaque_numeric to binary64, safe_integer, or an exact string type by inspection of the token.

### 1.1 Binary64

The profile explicitly chooses RFC 8785 semantics. Examples:

- 42 and 42.0 represent the same canonical binary64 value.
- 0.1 and 0.10000000000000001 can canonicalize to the same binary64 value.
- -0 canonicalizes as 0.
- Exponent notation is a numeric representation, not proof of an exact integer.

The precision warning system remains active. A warning is diagnostic context, not a reason to mutate Evidence identity or reinterpret the number as an exact integer.

### 1.2 Safe integer

This intent is deliberately narrower than "mathematically integral after parsing." V0 requires an integer-form source token (no decimal point and no exponent) and an explicitly declared inclusive range contained within the ECMAScript safe-integer interval.

The first unsafe positive integer is 9007199254740992; the inclusive safe upper boundary is 9007199254740991. Exact bounds are represented as decimal strings in the profile itself so the profile does not depend on the numeric representation it is defining.

### 1.3 Exact integer string

The source JSON value must be a string. The profile declares signedness, maximum digit count and inclusive decimal-string bounds. V0 rejects plus signs, leading zeros (except the single digit 0), and negative zero. No arithmetic conversion to binary64 is permitted.

### 1.4 Fixed decimal string

The source JSON value must be a string. V0 declares signedness, exact fractional scale, maximum integer digits and strict lexical policies. Excess scale and exponent notation are rejected; no implicit rounding occurs. Trailing zeros are semantically retained when the declared scale requires them.


### 1.4 Fixed-decimal range and grammar

A fixed-decimal profile binds both inclusive minimum and maximum values as decimal strings, the maximum integer-part width, an exact fractional scale, and the grammar `base10_fixed_point_exact_scale`. The validator must require exactly the declared number of fractional digits, reject exponent notation, reject leading-zero and plus-sign forms unless a future profile version explicitly changes that grammar, and reject values outside the declared decimal bounds. Bounds are compared as exact decimal strings/integers at the declared scale; they are never parsed through binary64. A schema profile must not rely on an unconstrained JSON number for its own decimal bounds.

### 1.5 Opaque numeric

The source token is retained as data, but no arithmetic, ordering, or equivalence relation is authorized. Opaque is not a fallback that permits callers to compare values anyway; it is a deliberate no-inference state.

## 2. Exact profile binding

The profile schema at docs/research/rea-numeric-intent-profile-v0.schema.json proposes:

- a profile ID and version;
- the digest of the domain value schema to which it applies;
- a list of exact JSON Pointers and numeric intent contracts;
- optional quantity-kind and unit references to #4861/#5662;
- a JCS SHA-256 commitment over the complete profile body, excluding the digest field itself.

A production validator must reject duplicate path declarations, unknown/uncovered paths, unknown profile versions, mismatched value-schema digests and invalid path/contract combinations.

No wildcard matching is permitted in v0. Two paths that happen to carry the same token do not share intent unless the profile declares each exact path.

The profile digest is not a signature, source-authenticity proof, owner authorization, or evidence that the declared semantics are appropriate. Those claims require separate governance/authority evidence.

## 3. Unit and quantity-kind rule

A numeric value does not carry its physical meaning by itself:

1000 mm != 1000 m

and:

1000 N·m (torque) != 1000 J (energy)

even though torque and energy share SI dimensions.

The profile may bind a path to both a canonical quantity_kind_ref and unit_code_ref owned by the existing engineering semantics. V0 permits no implicit unit conversion. A conversion requires a separately declared conversion rule, exact source and target unit identity, and a qualified measurement/quantity contract.

A canonical unit code does not establish calibration, measurement validity, physical truth, or applicability.

## 4. Missing, null and unavailable source bytes

These are distinct:

- missing: the declared path does not occur;
- null: the path occurs with JSON null;
- type_mismatch: the path exists but its JSON type violates the intent;
- lexical_coverage_unavailable: a caller supplied only a constructed serde_json::Value and no exact raw source bytes;
- raw_audit_rejected: raw-token admission failed due to duplicate keys, invalid syntax or resource limits.

A programmatically constructed VerifiedEvidence must never be labelled as raw-token audited unless a compatible audit result is bound to the exact raw SHA-256. Partial/unavailable lexical coverage must remain visible.

The raw audit result is tied to source_raw_sha256; it may only be attached to an Evidence record when its digest exactly equals that record's raw_file_sha256. Semantic Evidence ID and raw file SHA-256 remain distinct.

## 5. Adversarial known-answer matrix

The machine-readable matrix at tools/rea-numeric-lexeme-audit/tests/fixtures/rea-numeric-intent-v0.json is a design fixture, not an execution result. It preserves raw numeric lexemes as strings so the fixture file's own JSON parser does not erase the test condition.

It includes:

- JCS binary64 equivalence, exponent and negative-zero behavior;
- the inclusive safe-integer boundary and first unsafe value;
- exact large-integer strings and invalid grammar;
- fixed-decimal scale/exponent constraints;
- opaque no-arithmetic behavior;
- JSON number vs string type confusion;
- unit and quantity-kind mismatch;
- undeclared paths, duplicate profile pointers and profile-digest mismatch;
- constructed-value lexical coverage unavailable;
- explicit null vs numeric value.

Expected labels are independently authored. They are not generated from a validator or inferred from the replay result. The matrix is not a benchmark score and no implementation pass is claimed.

## 6. Qualification targets

The follow-on implementation must report each denominator separately.

| Dimension | Proposed target |
|---|---:|
| Profile schema and path binding | 100% exact outcomes on the frozen finite corpus |
| Binary64 parity | 100% parity with pinned upstream RFC 8785/JCS vectors |
| Safe-integer boundaries | 100% boundary precision and zero out-of-range accepts |
| Exact integer strings | 100% lexeme preservation; zero binary64 coercions |
| Fixed decimal strings | 100% exact scale/grammar compliance; zero hidden rounding |
| Opaque semantics | Zero arithmetic/order/equivalence upgrades |
| Profile/schema digest binding | Zero stale or mismatched positive results |
| Quantity-kind/unit binding | 100% mismatch detection; zero implicit conversion |
| Missing/null/type distinction | 100% exact classification |
| Lexical coverage | 100% explicit complete/partial/unavailable outcomes |
| Resource safety | 100% enforcement of declared byte/depth/path/value limits |
| Cross-implementation replay | 100% agreement on supported profile cases |

There is no composite scalar. A false acceptance of an exact integer or unit mismatch cannot be offset by speed or binary64 compatibility.

## 7. Baselines and industry positioning

This is complementary to, not a replacement for:

- RFC 8785 JCS: interoperable canonical representation of JSON under ECMAScript binary64 number serialization;
- RFC 7493 I-JSON: safe interoperability guidance for integers beyond ±(2^53−1), with strings recommended when exact interchange is required;
- IEEE 754: floating-point representation and arithmetic, not domain units or exact-decimal intent;
- JSON Schema/OpenAPI: structural/type/range validation, not a universal exact numeric/unit semantics contract;
- decimal arithmetic implementations: arithmetic engines, not source-intent or profile-authority systems;
- SLSA v1.2, SCITT and Sigstore: provenance/statement/transparency evidence, not numeric-domain semantics or the truth of the statement.

The competitive opportunity is composition: preserve raw token, canonical value, declared numeric intent and quantity/unit context as four separate commitments, then independently replay the profile evaluation.

References:
- RFC 7493: https://www.rfc-editor.org/rfc/rfc7493.html
- RFC 8785: https://www.rfc-editor.org/rfc/rfc8785.html
- SLSA v1.2: https://slsa.dev/spec/v1.2/

## 8. Staged implementation plan

1. Freeze this profile schema, design note and independent fixture matrix.
2. Implement a standalone validator outside production Evidence v1; reuse #10952's exact raw-token audit and #10958's JCS/replay primitives without duplicating them.
3. Add an independent validator/replay path and mutation tests for profile digest, path, intent, unit, scale, grammar and raw-source SHA changes.
4. Only after exact-head validation, define a companion lexical-coverage attachment for claim results. Do not silently add fields to or mutate Evidence v1.
5. Integrate into producer/replay claims through an additive API that marks constructed-value-only inputs as lexical coverage unavailable.
6. Benchmark admission, raw lexical audit, profile validation, producer evaluation and replay separately.

### Exit criterion

An independent consumer can tell, for a particular schema path and exact source bytes, whether the value was interpreted as binary64, a safe integer, an exact integer string, a fixed decimal string, or an opaque token; which profile and unit/quantity references authorized that interpretation; and exactly what is still unknown.

## 9. Claim ceiling

A passing profile validator establishes only conformance to a declared path-specific numeric grammar, representation, range, scale and unit reference. It does not establish that the source value is true, that a measurement is calibrated, that a producer is authentic, that a unit mapping is appropriate, or that any downstream engineering or security claim follows.
