# RES-OSINT-TOOL-001T — deterministic tool-selection corpus v0.1

Status: FROZEN DESIGN/CORPUS CANDIDATE — NOT EXECUTED / NOT QUALIFIED / NOT PASS

Parent: RES-OSINT-TOOL-001 / #5440

## Purpose

Freeze Symthaea's reasoning-only tool/source-adapter selection semantics over Mycelix EPI-013 profiles before any selector implementation or connector integration.

This subject chooses methodology candidates only. It creates no network, browser, credential, OPSEC, target-admission, lease, persistence, or action authority.

## External profile subject

The corpus binds exactly:

```text
repo    = Luminous-Dynamics/mycelix
head    = 3ce0c6ceeda41419512185c2a938f391d1592e3f
tree    = 7f46b3d6b60a28dd7a346b908fddec825a6c2311
fixture = 2536cfbba7402c19a62417904b2c732fbfb4ac88
profile = mycelix:epi-osint-tool-profile:v1
```

## Core laws

```text
ToolProfile != ToolSelectionCandidate
ToolSelectionCandidate != ToolExecution
method fit != permission
provider rank != tool preference
same capability != interchangeable methodology
historical profile != current availability
no fitting tool != permission to approximate the claim
```

## Selection profile

`symthaea:osint-tool-selection:pareto:v1`

Authority:

`CandidateAnalysisOnly` / `ProposalOnly`

## Input dimensions

Each task declares:

- required capability class(es);
- required observation/evidence semantics;
- minimum coverage semantics;
- currentness requirement;
- allowed/forbidden view classes;
- allowed/forbidden disclosure surfaces;
- observation-only requirement where applicable;
- transform requirements;
- reproducibility preference;
- exact frontier/purpose/profile refs.

## Hard filtering

Before Pareto analysis a profile may be partitioned as:

- `Eligible`;
- `IneligibleCapability`;
- `IneligibleCoverage`;
- `IneligibleCurrentness`;
- `BlockedViewClass`;
- `BlockedDisclosureSurface`;
- `IneligibleSideEffectClass`;
- `IneligibleTransformSemantics`.

Blocked/ineligible profiles remain visible for audit.

## Method-fit dimensions

Keep separate:

- capability fit;
- coverage fit;
- currentness fit;
- transform/provenance fit;
- reproducibility.

## Burden dimensions

Keep separate:

- privacy burden;
- OPSEC disclosure burden;
- credential/account burden;
- remote-upload burden;
- provider-retention burden;
- resource/cost burden.

No weighted universal score.

## Pareto semantics

Among eligible profiles, A dominates B only when A is no worse on all declared fit/burden coordinates and strictly better on at least one under the exact named selection profile.

Preserve incomparable methods.

Provider ranking does not participate in V1 method selection.

## TASK_WEB_PUBLIC

Goal: broad public-web discovery.

Constraints:

- required capability `WebSearch`;
- anonymous/public view only;
- authenticated/personalized view forbidden;
- remote query disclosure allowed only as a later OPSEC-authorized surface;
- exhaustive/world absence not requested.

Expected:

- `T_WEB_PUBLIC_TOPK` eligible;
- `T_WEB_AUTH_VIEW` blocked by view policy;
- Top-K/CoverageUnknown limitations carried forward;
- no absence claim.

## TASK_EXHAUSTIVE_SYNTHETIC_SCI

Goal: determine absence only within exact synthetic scholarly corpus commitment.

Constraints:

- required `ScientificLiteratureSearch`;
- `ExactFiniteCorpus` required;
- exact corpus commitment required;
- observation-only acceptable;
- current-qualified profile required.

Expected:

- `T_SCI_FINITE` is the only eligible candidate;
- TopK/BestEffort/other capability classes are ineligible;
- any absence conclusion remains exact-corpus scoped.

## TASK_MEDIA_PROV_LOCAL

Goal: inspect media provenance metadata with no network disclosure.

Constraints:

- required `MediaProvenanceInspection`;
- disclosure surfaces must be empty;
- `ObservationOnly` required;
- local deterministic method preferred where fit is equal.

Expected:

- `T_MEDIA_PROV` eligible and sole Pareto candidate;
- output retains `valid provenance != media claim true` and `no provenance != fake` limitation;
- no remote upload/network method can substitute without explicit requirement change.

## TASK_RENDERED_PAGE

Goal: obtain browser-rendered DOM/text observation.

Constraints:

- required `BrowserRenderedCapture`;
- `JavascriptExecution` transform required;
- current-qualified profile required;
- DNS/transport/headers/body/provider telemetry disclosures require later external authorization.

Expected:

- `T_BROWSER_CAPTURE` method-fit eligible;
- selection disposition `NeedsExternalAuthorization` because of disclosure surfaces;
- selection does not authorize browser execution.

## TASK_ARCHIVE_WORLD_ABSENCE

Goal request: establish that an item never existed anywhere in the world.

Expected:

- `T_ARCHIVE_PARTIAL` may offer only partial archive evidence;
- no EPI-013 profile satisfies world-completeness semantics;
- output limitation `NoToolProfileCanEstablishRequestedCoverage`;
- selector must not silently weaken the question to `not in this archive` and report success.

## Required deterministic output semantics

For each task preserve:

- exact input task/profile/frontier refs;
- exact EPI-013 subject refs;
- eligible profiles;
- blocked/ineligible profiles + typed reasons;
- eligible Pareto front;
- carried-forward limitations;
- unmet requirements;
- optional fallback relation + explicit rationale;
- proposal-only authority ceiling.

## Selected vs executed methodology

A later investigation capsule should preserve both selected and actually executed profiles.

```text
selected profile != executed profile
```

Any mismatch is evidence, not something to normalize away.

## Metamorphic ratchets

Future qualification must prove at least:

1. allow authenticated view in TASK_WEB_PUBLIC -> `T_WEB_AUTH_VIEW` becomes eligible but remains proposal-only;
2. remove exact-finite-corpus requirement from TASK_EXHAUSTIVE_SYNTHETIC_SCI -> eligibility may widen explicitly;
3. add `UploadedMedia` surface to `T_MEDIA_PROV` -> local/no-disclosure preference invalidates;
4. mark `T_ARCHIVE_PARTIAL` current-qualified but keep archive incomplete -> world-absence task remains unsatisfied;
5. remove `JavascriptExecution` transform from `T_BROWSER_CAPTURE` -> TASK_RENDERED_PAGE method-fit fails;
6. mutate provider rank only -> V1 selection result is unchanged;
7. change historical profile to current-qualified with fresh evidence -> currentness partition changes explicitly;
8. change side-effect class to `StateCreating` where observation-only is required -> profile becomes ineligible;
9. delete all fitting profiles -> emit explicit unmet requirement rather than selecting an unrelated profile;
10. candidate-order-only changes preserve set-semantic selection output.

## Nonclaims

This corpus does not establish provider/tool correctness, current live availability, legal permission, privacy compliance, safe credentials, factual truth, world coverage, collection authority, or production readiness.
