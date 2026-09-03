# Symthaea Content Fabric HDC Shadow Planner

`Symthaea Content Fabric` is a deliberately narrow, authority-free bridge from the Mycelix Content Fabric CF-06C external-planner protocol into Symthaea's HDC primitives.

It does **not** talk to Holochain, Iroh, the content-addressed store, Finance, Marketplace, leases, payment systems, or executors.

## Trust boundary

```text
Mycelix CF-06A hard-qualified pool
        |
        v
Mycelix CF-06B deterministic baseline
        |
        v
CF-06C ExternalPlannerRequestV1
        |
        | JSON interoperability encoding
        v
symthaea-content-fabric
        |
        | validates + recomputes request commitment
        | deterministic HDC shadow rank
        | local diversity-aware advisory subset
        v
ExternalPlannerRecommendationV1
        |
        v
Mycelix CF-06C acceptance
        |
        | exact request/input replay checks
        | CF-06A validate_selection()
        v
RecommendationOnly
```

Mycelix remains the authority that decides whether a recommendation is acceptable. This crate cannot make an ineligible provider eligible and cannot grant execution authority.

## Independent request validation

The bridge mirrors the narrow CF-06C JSON DTO and independently recomputes `ExternalPlannerRequestIdV1` using the same domain-separated, field-framed BLAKE3 commitment as Mycelix.

JSON bytes are not treated as canonical identity bytes.

Holochain action and agent identifiers remain opaque byte arrays. Symthaea validates their 39-byte wire shape but does not import HDK/Holochain types or interpret their contents.

## HDC shadow model

Each soft metric has two deterministic Symthaea HDC anchors:

- ideal anchor;
- worst anchor.

A candidate's normalized Mycelix penalty interpolates between those anchors. The storage intent's cost/latency/energy/locality weights then bundle the four metric representations into a candidate placement vector. The ideal placement vector bundles the four ideal anchors using the same preference weights.

Candidates are ordered by full-vector cosine similarity to that ideal.

The bridge uses Symthaea's deterministic `ContinuousHV` generation and bundling, but deliberately computes a local full-vector cosine rather than calling the runtime-global `ContinuousHV::similarity()` path. Content placement therefore cannot change merely because another Symthaea subsystem changes the global cognitive stride.

If all soft preference weights are zero, the HDC engine preserves the Mycelix deterministic baseline exactly.

## Local selection is advisory

After HDC ranking, the shadow planner begins with the entire qualified candidate universe and removes worst-ranked candidates only when the request's replica count and supplied failure-domain requirements remain locally satisfiable.

This is defense in depth only. The accepted failure-domain values came from Mycelix's hard-policy surface, and Mycelix must still re-run CF-06A `validate_selection()` when the recommendation returns.

## Shadow trace

`HdcShadowPlanV1` retains a local diagnostic trace containing:

- baseline rank;
- HDC rank;
- similarity to the HDC ideal;
- baseline weighted penalty;
- baseline selected subset;
- HDC selected subset;
- whether ranking changed;
- whether selection changed.

The trace is not part of the CF-06C recommendation wire response. It exists so future evaluation can measure whether the HDC planner actually improves placement behavior relative to the deterministic baseline.

## Non-goals

This version does not claim that HDC ranking is superior to the deterministic baseline. It is a reproducible shadow planner suitable for comparative evaluation.

It does not use Phi/consciousness values as policy, trust, or execution authority. Any later adaptive or consciousness-informed planner must preserve the same external protocol and pass the same Mycelix acceptance boundary.
