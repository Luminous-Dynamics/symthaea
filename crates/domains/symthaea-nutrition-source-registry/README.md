# Symthaea Nutrition Source Registry

This crate is a small, deterministic gate between discovered food-composition records and downstream nutrition reasoning. It records declared source provenance and rights, checks source-review freshness, matches a fact to its declared source, and distinguishes assessability from legal permission and data quality.

It is not a food database, a nutrient calculator, a license attorney, a publisher authenticator, a laboratory verifier, a food-safety certificate, a crop planner, or clinical nutrition advice. Numeric nutrient aggregation belongs to Symtropy. Community and traditional-knowledge authorization belongs to Mycelix and the relevant rights holders.

## Core behavior

- Rights are separate explicit decisions for assessment use, caching, transformation, redistribution, commercial use and public fixtures. Unknown, denied and unresolved written-permission requirements fail closed.
- The source manifest records the source's declared rights; a separate SourceUseRequestV1 states what the caller intends to do. A request cannot grant itself permission. Permission-reference requirements are evaluated only for rights exercised by the current request.
- Every affirmative right decision—whether allowed by license or granted by written permission—must resolve to a typed evidence record scoped to the exact source and right. The record must match the grant basis, use a valid HTTP(S) evidence URI, have a valid review/expiry window, and remain active. References are separate for assessment, cache/retention, transformation, commercial use, public fixtures and redistribution; evidence for one right cannot satisfy another right's requirement. The underlying license clause or document still needs human review for authenticity and legal sufficiency.
- A public fixture necessarily retains source-derived data, even if a caller omits the separate retention flag; cache permission and verified artifact bytes remain required.
- Source manifests record retrieval and review dates. Future-dated retrieval, expired review windows, revoked, withdrawn or review-required sources block use.
- Retaining source-derived records requires cache permission and a canonical lower-case SHA-256 plus the exact source-artifact bytes. A digest-shaped string by itself is not proof: verification recomputes SHA-256 over the supplied bytes.
- If a declared digest is supplied without its bytes, its integrity is Unverified and numeric use is blocked. Without a declared digest and with no retained artifact, the result explicitly says NoDigestClaimed, not Verified.
- Fact records bind source ID/record/citation, food identity, taxon/cultivar references when relevant, identity confidence, preparation state, nutrient definition, unit, integer scale and basis mass. A missing nutrient value is NotAssessable rather than zero.
- Every declared preparation/conversion keeps a step-by-step state chain with method and evidence references. A transformed value without lineage, or a chain that does not end in the declared food state, is blocked.
- Approximate/geographic fallback rows are visibly QualifiedFallback. Unresolved food matches and unknown geographic matches block.
- Synthetic rows return SyntheticFixtureOnly and cannot be misrepresented as empirical source data.

No SAFOODS values or other real food-composition rows are included in this crate's fixtures; all fixtures use synthetic example bytes and identifiers. The actual source licensing and data quality still require human review. This validator checks declarations and supplied bytes, not whether those declarations are truthful or legally sufficient.


The evidence-record lookup goes beyond a non-empty-string check: unknown or duplicate IDs, wrong source/right/basis, invalid URI/date metadata, and non-current evidence block the requested use. It does not fetch or authenticate the referenced document; human review of the underlying license or permission remains necessary.


- Every evaluation returns a self-describing decision receipt containing a separate receipt-schema version and an explicit `rfc8785_jcs_v1` canonicalization-profile ID, evaluation date, source/record IDs, input schema versions, declared and actually supplied artifact digests, per-component SHA-256 digests for the source manifest, food fact and caller use request, an overall input-envelope SHA-256, and a `decision_sha256` over the complete reported receipt with that digest field omitted. The profile ID is part of the input-envelope preimage, so a future hashing-profile change is not silent. Receipt schema version 2 also records evaluator package name/version, a separate evaluator-contract version and an optional build ID supplied through `LUMINOUS_BUILD_ID`; a missing build ID remains explicit rather than being invented. The decision digest binds the emitted status and issue fields to the receipt, but is unkeyed: it is a consistency/tamper-evidence checksum, not a signature or proof of who issued the receipt.
- The input envelope includes the complete manifest, food fact, use request, evaluation date and supplied artifact-byte digest; its bytes are produced by the pinned RFC 8785 JCS canonicalizer. The fact's `value_scaled`, `value_scale` and `basis_grams` are decimal strings on the JSON wire (missing `value_scaled` is `null`) so `u64` precision is not lost by JSON clients that use IEEE-754 numbers. The digest binds the exact input values for replay and change detection, not truth, authorship or legal validity.
- The `decision_sha256` is recomputed over the receipt after omitting that single field; a verifier must follow the documented omission rule. It detects inconsistent receipt fields when compared with a trusted copy of the digest, but an attacker able to rewrite the receipt can recompute it. It is not a digital signature, publisher authentication, proof of legal permission or proof of nutrient truth.
- The receipt schema was bumped to version 2 for the canonicalization/profile, outcome-digest and evaluator-identity contract. It uses the pinned `serde_json_canonicalizer = 0.3.2` implementation and identifies the profile as `rfc8785_jcs_v1`; hashes are computed over RFC 8785 JCS bytes rather than a custom key-sorting encoding. Wide nutrient quantities remain decimal strings because JCS numeric values are IEEE-754 based. The profile transition changes digest values and must be treated as a receipt-hash version migration. Two RFC-derived number/string and UTF-16 property-ordering vectors include fixed canonical-byte SHA-256 values. A separate fixture test pins component hashes for manifest, fact and use request plus the overall input envelope, detecting unintended typed-wire or receipt-preimage changes. These local tests do not replace an independent replay checker; exact-head CI and independent golden-vector verification must pass before claiming conformance. Any raw JSON verifier must reject duplicate object property names before typed parsing. JCS does not authenticate a source, authorize reuse or prove legal sufficiency.

## Independent receipt verification

The package includes a separate verifier binary that recomputes component digests, the input-envelope digest and the decision-receipt digest without calling the primary assessment function. It rejects duplicate property names at every level of the incoming JSON before converting to typed structs.

The JSON bundle must contain the exact evaluation inputs and the serialized assessment:

- `manifest`: exact NutritionSourceManifestV1 input;
- `fact`: exact NutritionFoodFactV1 input;
- `evaluation_date`: exact date string supplied to the evaluator;
- `use_request`: exact SourceUseRequestV1 input;
- `assessment`: the full serialized NutritionRegistryAssessmentV1 result.

Run:

```sh
cargo run --locked -p symthaea-nutrition-source-registry --bin nutrition-receipt-verify -- bundle.json [artifact-bytes.bin]
```

When the assessment records supplied artifact bytes, pass a local file containing those exact bytes as the optional final argument. The verifier reads and hashes that file; it does not copy artifact contents into the JSON bundle. If no supplied artifact digest is recorded, omit the argument.

A successful result means only that the receipt's input/component/outcome digests are internally consistent with the supplied bundle and bytes. The checker shares the declared Rust wire types and pinned JCS library, but does not call the production assessor or independently recalculate its eligibility rules. It does not prove publisher authenticity, source permissions, nutrition truth or field validity. Exact-head formatter/tests/Clippy and an independent cross-language replay remain required.

## Validation

Run focused checks from the repository root:

- cargo fmt --check -p symthaea-nutrition-source-registry
- cargo test -p symthaea-nutrition-source-registry
- cargo clippy -p symthaea-nutrition-source-registry --all-targets -- -D warnings

The source registry contract is tracked in [Symthaea issue #7481](https://github.com/Luminous-Dynamics/symthaea/issues/7481), with broader nutrition reasoning in [PR #7476](https://github.com/Luminous-Dynamics/symthaea/pull/7476). Exact-head checks must pass before the implementation is treated as validated.
