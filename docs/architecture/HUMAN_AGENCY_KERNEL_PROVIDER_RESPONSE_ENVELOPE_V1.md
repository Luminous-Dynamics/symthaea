# Human Agency Kernel — Provider Response Envelope + Link Projection v1

Status: architecture + audit/evidence tooling candidate

## Purpose

HAK-019a closes a source-content gap left intentionally open by HAK-018a/b.

HAK-018a can verify an internally consistent retained traversal and HAK-018b can conserve selector coverage across every retained page, but neither establishes that a retained continuation token was actually projected from retained provider response metadata.

The central boundary is:

```text
ContinuationEdgeConsistency
!=
ContinuationSourceVerification
```

HAK-019a introduces two adjacent artifacts:

```text
ProviderResponseEnvelopeV1
        ↓
LinkPaginationProjectionReceiptV1
```

The first retains the exact response body bytes plus selected HTTP response-header field values. The second deterministically replays `rel=next` from the retained `Link` header representation under an exact policy and parser identity.

This tranche does not authenticate the provider, retain the raw HTTP wire representation, interpret the JSON body, prove collection exhaustion, prove temporal snapshot coherence, or grant runtime authority.

## 1. Evidence-channel separation

A provider response contains distinct evidence channels.

For the GitHub workflow-jobs list profile used by this tranche:

```text
JSON body
-> total_count + jobs

Link response header
-> pagination relations
```

These channels must not borrow assurance from one another.

```text
HeaderProjectionVerified
!=
BodyProjectionVerified

BodyDigestMatches
!=
HeaderSetMatches

ContinuationVerified
!=
EntityProjectionVerified
```

HAK-019a handles only the header/continuation side. HAK-019b should handle retained-body/entity projection separately.

## 2. ProviderResponseEnvelopeV1

The envelope retains:

- provider and typed resource kind;
- exact request reference;
- exact HTTPS request URL;
- request method (`GET` in v1);
- exact response reference;
- response status;
- selected response-header field values as ordered `{name, value}` entries;
- digest of the retained header representation;
- exact body bytes encoded in canonical base64;
- exact body length and raw SHA-256;
- collector-observed request/response bounds;
- exact HAK canonicalization-profile identity;
- envelope digest.

The envelope explicitly records:

```text
body_bytes_retained = true
header_representation = RetainedFieldValuesV1
raw_http_wire_representation = NotRetained
provider_authentication = NotEstablished
```

Therefore:

```text
RetainedHeaderFieldValues
!=
RawHTTPWireBytes

RetainedBodyBytes
!=
ProviderAuthenticatedBody
```

### 2.1 Why body bytes are retained in HAK-019a

Body bytes belong to the same response envelope and are useful for the later body-projection join. HAK-019a does not parse or interpret them.

```text
BodyBytesRetained
!=
BodySemanticsInterpreted
```

## 3. Exact Link projection policy

Policy:

`hak019a-github-rest-link-pagination-v1`

Current content digest:

`sha256:8ab07e7c06f15aa75b71da7c7f9713d897763691f356e3b473781e2c4f0fce2e`

The profile fixes:

- provider family `github-rest`;
- request method `GET`;
- accepted response status `200`;
- header name `Link` (case-insensitive lookup);
- relation `next`;
- parser profile `hak.rfc8288-link-pagination.strict@1`;
- exact retained-field-value semantics;
- duplicate parameter / relation / next rejection;
- relative target resolution against the exact request URL;
- continuation target scope `SameSchemeAuthorityAndPathNoFragment`;
- missing-Link semantics `NoNextRelationObserved`;
- no provider-authentication claim.

```text
PolicyName
!=
PolicyContent
```

The receipt binds the exact policy digest and an exact Git artifact reference.

## 4. Strict Link parser

HAK-019a uses a deliberately strict RFC-8288-compatible profile rather than a permissive generic HTTP parser.

It supports:

- multiple retained `Link` field values;
- multiple comma-separated link-values;
- URI references inside `<...>`;
- token and quoted-string parameter values;
- registered and extension relation types;
- multiple relation types in a quoted `rel` value;
- relative next targets resolved against the exact request URL.

It rejects tested ambiguity including:

- duplicate link parameters;
- duplicate relation types;
- more than one `rel=next` target;
- malformed angle/quoted syntax;
- CR/LF inside retained header values;
- continuation targets outside the exact resource path/origin;
- fragment-bearing continuation targets.

### 4.1 Wire whitespace semantics

HTTP optional whitespace is explicitly:

```text
SP / HTAB
```

HAK-019a does not use a runtime's generic Unicode whitespace rules for Link grammar boundaries.

Relation-type lists use SP separation only.

Therefore:

```text
RuntimeWhitespaceConvenience
!=
WireGrammarWhitespace
```

A tab or Unicode whitespace embedded inside a `rel` relation list is not silently interpreted as an ordinary relation separator.

## 5. Continuation target scope

For this GitHub list-endpoint profile, pagination is permitted to alter the query but not the API resource identity.

A resolved continuation target must preserve:

```text
scheme
host
port
path
```

and must contain no fragment.

The query may change.

```text
ContinuationTarget
!=
ArbitraryProviderURL
```

Cross-origin, different-path, and fragment-bearing next targets are rejected even if they are otherwise valid HTTPS URLs.

This is a profile rule for this adapter, not a universal RFC 8288 rule.

## 6. Projection receipt

`LinkPaginationProjectionReceiptV1` binds:

- exact envelope digest;
- exact response reference;
- exact retained-header digest;
- exact projection-policy identity/digest/ref;
- exact parser identity/version/ref/content digest;
- parsed retained links;
- the derived `next_relation`;
- the number of retained Link field values;
- explicit assurance boundaries;
- receipt digest.

Validation deterministically replays the parser from the exact supplied envelope, policy and executing parser snapshot.

```text
ReceiptSelfConsistency
!=
ProjectionReplayEquivalence
```

A coherently redigested forged next target must fail replay.

## 7. Missing next is not exhaustion

When no `rel=next` relation is present, HAK-019a records:

```text
next_relation.state = Absent
exhaustion_verification = NotEstablished
```

It does not infer collection exhaustion.

```text
NoNextRelationObserved
!=
CollectionExhaustionVerified
```

A later composition layer may establish exhaustion only by binding the projection to an exact pagination contract and whatever additional evidence that contract requires.

## 8. Assurance boundaries

The machine artifacts fix:

```text
provider_authentication = NotEstablished
http_wire_verification = NotEstablished
exhaustion_verification = NotEstablished
```

The envelope additionally fixes:

```text
raw_http_wire_representation = NotRetained
```

Thus:

```text
VerifiedAgainstRetainedEnvelope
!=
ProviderAuthenticated
!=
SemanticTruth
!=
Authority
```

## 9. Composition with HAK-018

The intended future join is:

```text
HAK-019a Link projection receipt
+
HAK-018a retained traversal
-> continuation-source composition evidence
```

where each HAK-018a page continuation is required to equal the exact HAK-019a projection derived from that page's retained response envelope.

That composition is not established merely because both artifacts carry similar response refs.

Body/entity projection remains separate:

```text
HAK-019b body/entity projection
+
HAK-018a retained entity IDs
-> entity-projection composition evidence
```

Only after both joins should higher layers consider strengthening HAK-018's current `NotEstablished` source-assurance fields.

## 10. Qualification boundary

The HAK-019a E5 target covers only:

- response-envelope integrity and exact retained representation semantics;
- strict Link parsing under the exact committed policy/parser;
- deterministic next-relation replay;
- tested continuation target-scope enforcement;
- tested ambiguity rejection;
- schema parity;
- tested anti-oracle boundaries.

It does not establish provider authentication, raw HTTP wire replay, collection exhaustion, body/entity projection, temporal snapshot stability, scientific truth, legal/governance legitimacy, human worth, or runtime authority.

## 11. Core theorems

```text
RawResponseDigestPresent
!=
RawResponseBytesReplayed

RetainedHeaderValues
!=
RawHTTPWireRepresentation

ContinuationEdgeConsistency
!=
ContinuationSourceVerification

NoNextRelationObserved
!=
CollectionExhaustionVerified

HeaderProjectionVerified
!=
BodyProjectionVerified

ContinuationTarget
!=
ArbitraryProviderURL

VerifiedAgainstRetainedEnvelope
!=
ProviderAuthenticated
```
