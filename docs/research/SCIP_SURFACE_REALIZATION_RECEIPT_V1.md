# SCIP Surface Realization Receipt v1

Status: research architecture + implementation candidate. This document defines the claim boundary for the v15 child of the preserved cognitive-interchange v14 stack.

## Governing theorem

```text
backend returned bytes
    != surface accepted
    != semantic fidelity established
    != grounded truth
    != epistemic authority
```

The v14 transactional bridge already separates backend success from accepted persistent language state: blank or oversized surfaces are rejected without committing successful-generation accounting.

V15 adds a distinct opaque receipt only after that acceptance boundary:

```text
validated SCIP request
    -> transactional backend execution
    -> accepted bounded nonblank surface
    -> ScipSurfaceRealizationReceiptV1
```

The only positive claim in that receipt is:

```text
SurfaceAcceptedOnly
```

There is intentionally no `faithful`, `grounded`, `verified`, `true`, or `authorized` flag.

## Bound identity

The v1 receipt content-addresses:

- receipt profile;
- upstream adapter profile;
- exact request digest;
- recomputed exact UTF-8 surface digest;
- source SCIP message identity;
- source grounded semantic hash;
- exact source-confidence binary32 bits;
- canonicalized source evidence-set digest and count;
- canonicalized source provenance digest;
- backend's human-readable label;
- realization mode;
- accepted surface byte count.

The receipt identity excludes backend latency. Identical content can therefore produce the same receipt ID across multiple executions.

That is deliberate:

```text
same content identity
    != same execution event
```

V1 contains no trusted timestamp, nonce, monotonic sequence, signature, remote attestation, or anti-replay proof.

## Opaque capability boundary

`ScipSurfaceRealizationReceiptV1` has private fields and no deserialization constructor. It is minted only by the wrapper that first calls the existing transactional execution path.

The receipt can check that a supplied `ScipLlmOutput` still matches its bound content. That is an integrity consistency check, not authentication of an arbitrary externally supplied output.

## Explicit non-claims

A v1 receipt does not establish:

- semantic equivalence between source graph and generated prose;
- preservation of negation, modality, quantity, attribution, temporal scope, or causal language;
- absence of unsupported details;
- correctness or truth of the grounded source itself;
- authentication of the backend label, provider, deployment, or model weights;
- uniqueness, freshness, ordering, or replay prevention;
- user authorization, action authority, or policy compliance beyond the inherited language-surface acceptance boundary.

## Next boundary: independent semantic verification

The next tranche should consume:

```text
ScipSurfaceRealizationReceiptV1
+ exact accepted surface
+ exact grounded source semantics
+ independent verification profile
```

and produce a *different* typed verification capability.

That verifier should test claim-preserving dimensions such as:

- polarity and negation scope;
- epistemic strength and modality;
- attribution/source preservation;
- quantifier and cardinality preservation;
- temporal scope;
- entity/reference fidelity;
- causal-vs-correlational language;
- unsupported-detail introduction;
- required-detail omission.

The v1 receipt itself must never be widened into semantic verification by adding a boolean.
