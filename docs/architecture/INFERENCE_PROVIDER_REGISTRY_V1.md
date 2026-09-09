# Symthaea Inference Fabric — IF-10 Provider Registry v1

Status: provider-truth child of IF-9.

## Purpose

IF-10 prevents provider/model configuration from being treated as timeless truth.
Executable remote candidates are materialized only from locally versioned,
source-attributed claims whose validity windows are still fresh.

The registry is provider-neutral. This tranche does not hard-code Groq,
OpenRouter, Gemini, Cloudflare, or any other provider preset.

## Claim roles and source authority

Claims are typed by semantic role, and source kinds are not interchangeable:

- endpoint, lifecycle, and capability claims require a first-party model catalog;
- data-policy claims require a first-party policy document;
- executable request-cost bounds require first-party account state.

A public pricing document is therefore useful evidence for a future estimator but
cannot by itself mint `max_charge_microusd = 0` for an account/request. Likewise, a
model catalog cannot substitute for a privacy/retention policy source.

`SignedDeploymentManifest` and `ThirdPartyCurator` are representable source classes
for future verification/observability, but IF-10 deliberately rejects both as
executable authority. Merely labeling a claim "signed" is not signature
verification. A later tranche must verify a manifest and produce an opaque verified
capability before manifest claims can participate in executable qualification.

## Freshness

Every claim records:

- source kind;
- bounded source locator;
- BLAKE3 digest of the exact observed snapshot;
- locally observed tick;
- locally chosen expiration tick.

Qualification fails closed when any critical claim is expired or not yet valid.
The resulting qualified candidate carries the minimum expiration across all claims.

Expiration does not mutate or refresh provider truth. A newer locally installed
profile is required.

## Rollback resistance

Profiles are keyed by `provider + deployment + model` and carry a local
`profile_epoch`. Replacing an existing key requires a strictly larger epoch.
Equal or older epochs are rejected, preventing accidental rollback to a previous
claim bundle.

## Deployment identity

Provider identity is not deployment/account identity. Two accounts or endpoints for
the same provider/model can have different cost entitlement, credential state,
freshness, and endpoint configuration.

IF-10 therefore retains a separate `deployment_id` and account-scope id in the
qualified result. The legacy `InferenceCandidate` remains provider/model-oriented
for stack compatibility.

## Lifecycle

Production models qualify by default. Preview models require an explicit local
qualification policy. Deprecated and unavailable models fail closed.

This is intentionally a hard qualification decision, not a soft routing score.

## Profile digest

Each qualified profile has a domain-separated BLAKE3 digest over explicit stable
field encodings, including semantic values and provenance evidence. Capability
purpose lists are canonicalized as semantic sets.

Changing a source snapshot or claim provenance changes the profile digest even when
the resulting model semantics are otherwise identical.

## Logging/privacy

Claim source locators can contain account-scoped identifiers. They are retained for
audit but omitted from `Debug`. Snapshot digests and source kinds remain visible.

No bearer secret, prompt, response text, or credential material is stored in the
provider registry.

## Current public-provider motivation

Current provider offerings demonstrate why the claim classes must remain separate:
free allocation, public token pricing, model lifecycle, account quota, and data-use
policy can all change independently and can differ by plan/model/account.

Those current public facts motivate IF-10 but are deliberately not compiled into
this generic registry tranche.

## Explicit non-claims

- no vendor presets;
- no live network refresh/discovery;
- no automatic trust of public pricing as account entitlement;
- no signed-manifest verification or manifest authority yet;
- no provider-profile digest bound into IF-2/IF-4 execution permits yet;
- no router integration with deployment identity yet;
- no production runtime export or `LLMBackend` factory change.

The next integration theorem should bind the exact qualified profile/deployment
digest into route/execution state so a stale or substituted profile cannot survive
between registry qualification and network execution.
