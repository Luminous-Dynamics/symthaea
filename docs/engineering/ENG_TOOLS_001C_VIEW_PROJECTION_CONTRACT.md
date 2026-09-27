# ENG-TOOLS-001C-A — Engineering view projection contract v1

Issue: #6134  
Parent program: #6131  
First consumer: #6132 / #6130  
Frozen source parent: `fbd7a754ea8389ca93f7680d93ed8b48553e6376`

## Purpose

Freeze the first lightweight, authority-free wire contract for engineering data rendered by `symthaea-ui`.

The projection layer exists because the current UI intentionally consumes generic `serde_json::Value` while the canonical Rust wire/domain types live in heavyweight crates. Engineering views require stronger guarantees than ad-hoc JSON indexing without importing those heavyweight/authority-bearing types into WASM.

The intended dependency direction is:

```text
canonical engineering owners
        ↓ explicit read-only projection
portable view DTOs
        ↓
symthaea-service bounded read-only endpoint
        ↓
symthaea-ui Engineer pane
```

The view layer never becomes a second source of engineering truth.

## Non-equivalences

```text
view DTO != canonical engineering object
projection serialization != evidence serialization
browser projection != authority capability
projection field present != source fact current
friendly label != exact subject identity
rendered prose != machine witness
planning proposal != accepted design
unknown schema != safe default
```

## Schema envelope

Every claim-bearing projection MUST carry an explicit schema envelope equivalent to:

```text
ProjectionSchemaV1 {
  namespace,
  major,
  minor,
  kind,
}
```

V1 namespace is conceptually `luminous.engineering.view`.

Unsupported major versions MUST fail visibly. Minor-version compatibility is allowed only when an explicit compatibility rule exists; it must never depend on `serde(default)` producing apparently healthy engineering state.

## Exact subject references

A projection subject reference MUST preserve, where applicable:

- canonical owner namespace;
- subject kind;
- exact subject ID;
- exact generation/revision;
- optional friendly label marked non-semantic.

Changing only a friendly label does not change semantic projection identity. Changing a claim-relevant generation does.

## Source class

Engineering values MUST retain source class. V1 supports at least:

```text
Designed
ObservedPhysical
DerivedPhysical
Inferred
Synthetic
ExternalReference
PlanningAssumption
Historical
UnknownUnsupported
```

Numeric equality never collapses source classes.

## Currentness projection

The view may display canonical currentness but MUST NOT derive it locally. V1 display states may include:

```text
Current
Historical
Stale
Expired
Blocked
Unknown
NotApplicable
```

The authoritative mapping remains with ETK/currentness owners.

## Claim ceiling

Every engineering projection MUST expose the strongest class of claim it is permitted to communicate. V1 display vocabulary may include:

```text
PlanningOnly
SyntheticReference
CandidateEvidence
AdmittedEvidenceReference
HistoricalReceipt
CurrentQualifiedFactReference
ExternalReviewRequired
NoPhysicalAuthority
```

These labels are descriptive projections, not browser-mintable authority states.

## Quantities

A claim-bearing numeric quantity MUST include canonical quantity/unit identity separately from display formatting.

Rules:

- missing unit identity is invalid where units are required;
- NaN and infinity are invalid unless a future schema explicitly defines them;
- missing quantity does not become zero;
- unavailable, unsupported, unknown, and not-applicable remain distinct when the source owner distinguishes them;
- display unit conversion must not change the underlying semantic quantity identity.

## Findings

A projection finding binds:

- categorical finding code;
- exact subject/source refs where applicable;
- machine-readable cause refs;
- optional display severity independent of engineering authority;
- optional rendered explanation explicitly excluded from semantic identity.

Unknown future finding codes MUST render visibly as unsupported/unknown rather than disappear or map to success.

## First concrete projection — C0 feasibility

`C0FeasibilityProjectionV1` is the first concrete consumer. It projects qualified outputs from #6130 for #6132 and may include:

- exact feasibility-profile generation;
- evaluator identity/version;
- typed parameter descriptors/ranges;
- canonical unit IDs + display units;
- bounded interval results;
- feasible/infeasible cells or compact region representation;
- categorical findings per point/region;
- planning/material/source refs;
- unresolved/unsupported refs;
- claim ceiling;
- explicit proposal-only export identity when requested.

It MUST NOT contain the canonical beam equations or authoritative feasibility policy. Those belong to the evaluator/owner beneath the view layer.

## Semantic identity

Semantic projection identity MUST depend on claim-relevant machine fields such as exact source generations, source classes, currentness references, quantity identities, categorical findings, evaluator identity, and claim ceiling.

Semantic identity MUST NOT depend on purely presentational fields such as:

- localization;
- rendered prose;
- theme;
- chart dimensions;
- panel layout;
- friendly labels explicitly declared non-semantic.

## Security and authority boundary

View DTOs MUST NOT contain:

- physical actuation tokens;
- evidence-admission capabilities;
- private signing material;
- unchecked authority constructors;
- generic `approved`, `safe`, `ready`, or `current` booleans that can be set by browser state to mint authority.

Authority-looking unknown input fields cannot increase the claim ceiling.

The browser may create a separately typed planning/draft proposal where an owner API explicitly permits it. Such a proposal is not an accepted design and cannot deserialize as one.

## Payload bounds

Read-only engineering endpoints and clients MUST apply explicit bounded payload policies. V1 hostile fixtures use a reference item bound of 10,000 solely to exercise bounded rejection semantics; this number is not a universal production limit.

Untrusted payload size, collection counts, strings, and nested structures require limits before physical/session/twin data are exposed to the browser.

## Canonical serialization

The reference corpus uses UTF-8 JSON for known-answer fixtures. A future Rust DTO implementation MUST provide deterministic round-trip behavior for the machine projection under its declared canonicalization profile.

Do not claim generic JSON object ordering alone is a universal content-addressing theorem. If semantic content addressing is required, freeze a separate canonical serialization profile and vectors.

## Hostile/reference corpus

The companion file `eng-tools-001c-view-projection-corpus-v1.json` freezes 16 cases covering:

1. valid C0 projection;
2. unknown major schema;
3. missing required parameter/value;
4. unknown finding code;
5. equal physical/synthetic numeric values;
6. same friendly label with different currentness;
7. friendly-label-only change;
8. claim-relevant generation change;
9. prose/localization-only change;
10. authority-looking unknown fields;
11. oversized payload;
12. non-finite quantity;
13. missing unit identity;
14. synthetic value relabeled visually as physical;
15. planning proposal attempting design-acceptance promotion;
16. deterministic machine round-trip.

Canonical SHA-256 of the exact companion corpus bytes:

`07a5c95df0862f9bdf50324bb2864c266e28f2deec120a3c28f3c7b20392da6a`

## Qualification direction

The independent qualifier should:

- bind exact source parent/head and the two source blobs;
- verify the corpus SHA-256;
- derive all 16 expected dispositions independently;
- reject unknown/missing authority-critical fields fail-closed;
- prove presentation-only mutations do not alter semantic identity under the frozen test profile;
- prove source generation/source class changes do alter or preserve identity as preregistered;
- verify no corpus case can mint evidence, acceptance, safety, currentness, or physical authority.

No browser/UI source is required for this tranche.

## Claim ceiling

A future PASS may establish only the projection/schema/reference semantics frozen here. It establishes no underlying engineering truth, evidence admission/currentness, model credibility, requirement satisfaction, design acceptance, fabrication readiness, safety approval, or physical execution authority.
