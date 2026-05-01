# Mycelix Personal Frontend Plan

## Purpose

This document defines the first implementation-grade plan for the
`mycelix-personal` frontend.

It translates the existing Personal backend into a real standalone Leptos app
that can become the sovereign user shell for:

- identity profile and key posture
- credential wallet
- private health-vault visibility
- consent and data-sharing preferences
- selective disclosure handoffs into the rest of Mycelix

This plan assumes the architecture in `FRONTEND_ARCHITECTURE.md`, the build
order in `FRONTEND_ROADMAP.md`, and the phase sequencing in
`FRONTEND_EXECUTION_BACKLOG.md`.

## Current Reality

`mycelix-personal` already has meaningful domain structure:

- `identity-vault`
  - private profile storage
  - master key registration
  - selective profile disclosure
- `health-vault`
  - private health records
  - biometrics
  - consent grants
- `credential-wallet`
  - stored credentials
  - proof generation
  - trust credentials and presentations
- `personal-bridge`
  - dispatch
  - audited query flow
  - event broadcast
  - selective disclosure gateway
- `data_preferences`
  - present in the workspace and should become the preferences-control surface

What is missing is not backend capability. What is missing is the user-facing
shell that makes these capabilities coherent.

## Product Role

Personal should be:

- a standalone-first Leptos app
- the canonical user-owned vault shell
- Sensorium-summary-integrated
- mobile-compatible
- handoff-oriented rather than workflow-duplicative

Personal should not try to replace Identity or Health.

Its role is:

- hold the user's private posture
- summarize sensitive personal state
- expose consent and disclosure controls
- route users into deeper domain workflows when needed

## User Promise

The simplest product statement is:

"This is where I understand what is mine, what is private, what I can prove,
and what I am allowing to flow."

If the first release does not make that obvious, it is too abstract or too
fragmented.

## Target Users

Primary users:

- a sovereign end user managing credentials and private data
- a high-trust Mycelix participant who needs to understand trust posture and
  disclosure state

Secondary users:

- users entering Identity or Health who need a single control surface for
  handoff and permissions
- mobile users needing vault entry and quick posture checks

## Existing Backend Surface to Build On

The MVP should align directly to the current zome surface.

### Identity Vault

Existing functions support:

- `set_profile`
- `get_my_profile`
- `register_key`
- `get_my_keys`
- `disclose_profile`

Frontend implication:

- profile view and edit
- key inventory and status
- disclosure preview for identity fields

### Health Vault

Existing functions support:

- `create_health_record`
- `record_biometric`
- `grant_consent`
- `get_my_records`
- `get_my_biometrics`
- `get_my_consents`

Frontend implication:

- health-vault summary
- biometric summary
- consent grant list and state
- handoff into full Health workflows for deeper record work

### Credential Wallet

Existing functions support:

- `store_credential`
- `create_proof`
- `get_my_credentials`
- `get_credentials_by_type`
- `present_credential`
- trust credential issuance and presentation workflows

Frontend implication:

- wallet inventory
- credential detail
- presentation or proof actions
- trust posture summary

### Personal Bridge

Existing functions support:

- `dispatch_call`
- `query_personal`
- `resolve_query`
- `broadcast_event`

Frontend implication:

- event feed or disclosure log
- cross-domain handoff visibility
- audited activity surface for "what left my vault and why"

## App Strategy

The first Personal frontend should use the canonical cluster pattern:

- `HolochainProviderAuto`
- shared provider stack from `mycelix-leptos-core`
- shared shell primitives where they fit
- Personal-specific vault and sensitivity UI layered on top

It should look distinct, but not invent a separate app architecture.

## Information Architecture

The Personal app should launch into a compact vault overview, then break into
five top-level sections.

### 1. Vault

Purpose:

- overall posture
- lock state
- recent sensitive events
- quick launch into the rest of the app

Core content:

- vault state
- profile completeness
- credential count
- active consent count
- recent disclosure or bridge events

### 2. Identity

Purpose:

- private identity posture
- profile management
- key posture

Core content:

- display profile
- edit profile
- key inventory
- active signing/encryption/issuance key indicators
- handoff to Identity app for DID, MFA, and recovery flows

### 3. Wallet

Purpose:

- all credentials and proofs
- trust posture visibility

Core content:

- credential list by type
- credential detail
- expiry and revocation status
- proof/presentation actions
- trust tier summary

### 4. Health Vault

Purpose:

- private health state overview without replacing Health

Core content:

- record count by type
- recent biometrics
- consent grants
- handoff to Health for records and privacy workflows

### 5. Preferences

Purpose:

- data-sharing and disclosure control
- domain-level trust posture

Core content:

- current sharing posture
- domain-level preferences
- disclosure policy summaries
- bridge activity and recent cross-domain accesses

## Proposed Routes

The first route set should stay narrow.

- `/`
  - vault overview
- `/identity`
  - profile and key posture
- `/wallet`
  - credential wallet
- `/wallet/:kind`
  - filtered credential view
- `/health`
  - health-vault summary
- `/preferences`
  - data-sharing and disclosure controls
- `/activity`
  - disclosure log and bridge events
- `/unlock`
  - vault entry or lock-state surface

Optional later routes:

- `/wallet/trust`
- `/wallet/present`
- `/health/consents`
- `/identity/disclosure`

## MVP Definition

The first release should be intentionally narrow.

### Must Ship

- standalone Personal Leptos app scaffold
- canonical provider stack
- vault overview page
- profile read and edit
- key inventory display
- credential wallet list and detail
- health-vault summary
- consent grant list
- preferences placeholder wired to real zome-backed state where available
- portal launch integration

### Should Ship

- credential presentation preview
- trust-tier summary
- recent activity feed sourced from bridge events and query logs
- handoff links into Identity and Health

### Should Not Ship Yet

- every possible credential issuance flow
- deep medical record editing that belongs in Health
- full DID lifecycle management that belongs in Identity
- a universal settings maze

## UX Priorities

This app needs to feel private, deliberate, and high-signal.

The UX should emphasize:

- lock state
- sensitivity
- disclosure consequences
- trust posture
- clear ownership boundaries

The UI should avoid:

- generic dashboard clutter
- trying to show too many domain details at once
- unclear duplication with Identity or Health

## Shared Components Needed

Building Personal should produce reusable frontend primitives for other
domains.

### Vault Primitives

- lock-state banner
- sensitive-state panel
- secure summary cards
- disclosure preview modal

### Credential Primitives

- credential list item
- credential type badge
- expiry and revocation badge
- trust-tier badge

### Consent and Preference Primitives

- grant list item
- sensitivity legend
- domain-sharing toggle rows
- access-event timeline item

## Data Mapping

The frontend should map directly to current Personal zomes.

| UI Surface | Primary Calls | Notes |
|---|---|---|
| Vault overview | `get_my_profile`, `get_my_credentials`, `get_my_records`, `get_my_consents` | Aggregate, do not overcompute |
| Identity page | `get_my_profile`, `set_profile`, `get_my_keys`, `register_key` | Keep editing narrow |
| Wallet page | `get_my_credentials`, `get_credentials_by_type`, `present_credential` | Add trust summary from trust entries |
| Health summary | `get_my_records`, `get_my_biometrics`, `get_my_consents` | Summary-first, not full charting |
| Preferences page | `data_preferences` calls plus bridge state | If preferences zome is incomplete, ship partial controls |
| Activity page | `query_personal` / event retrieval patterns | May begin as recent local activity only |

## Sensorium Integration

Personal should become a first-class Sensorium domain module.

### Domain Metadata

Recommended domain metadata shape:

- id: `personal`
- name: `Personal`
- bio name: a sovereignty-adjacent metaphor, likely "Vault" or "Sovereignty"
- minimum tier: `Observer`
- dependencies:
  - `identity`
  - `health` optional
- sensitivity profile:
  - profile: `Private`
  - keys: `Sensitive`
  - credentials: `Sensitive`
  - consent grants: `Protected`

### Sensorium Summary Content

The Sensorium should show:

- vault state
- credential count
- active consent count
- trust-tier snapshot if available
- most recent disclosure or access event

### Launch Routes

Recommended launch points:

- `/personal`
- `/personal/wallet`
- `/personal/preferences`
- `/personal/identity`

## Handoff Rules

Personal must integrate with other domains cleanly.

### To Identity

Use Personal for:

- profile posture
- local key posture
- credential overview

Use Identity for:

- DID management
- MFA
- recovery
- identity trust detail

### To Health

Use Personal for:

- private health summary
- consent overview
- access posture

Use Health for:

- full record workflows
- privacy budget detail
- domain-specific health operations

## Delivery Plan

### Milestone 1: App Skeleton

Deliver:

- new Personal Leptos app workspace
- canonical provider stack
- routes and shell
- mock-safe page scaffolds bound to real zome client interfaces

Done when:

- app boots
- routes exist
- Holochain provider connects using Personal role

### Milestone 2: Vault and Identity

Deliver:

- vault overview
- profile retrieval and edit
- key inventory

Done when:

- a user can create or update profile state
- a user can see registered keys and their purposes

### Milestone 3: Wallet

Deliver:

- credential inventory
- detail views
- type filters
- trust summary

Done when:

- the wallet is usable as a real personal credential surface

### Milestone 4: Health Summary and Consents

Deliver:

- health-vault summary
- biometric summary
- consent grant list
- Health handoff links

Done when:

- a user can understand what private health material exists and what is shared

### Milestone 5: Preferences and Activity

Deliver:

- preferences page
- disclosure log or access timeline
- bridge activity visibility

Done when:

- a user can understand how data leaves Personal and with what posture

### Milestone 6: Sensorium Integration

Deliver:

- Personal domain module in Sensorium
- summary card and launch targets
- sovereignty dashboard integration for Personal entry types

Done when:

- the Sensorium can summarize Personal without duplicating the app

## Risks

### Biggest Product Risk

Trying to make Personal into a universal dashboard for everything the user owns.

That would blur boundaries and slow delivery.

### Biggest UX Risk

Showing sensitive-state concepts without clear actions:

- if something is locked, show how to unlock
- if something is shared, show with whom
- if something is missing, show where to go next

### Biggest Technical Risk

The frontend may expose zome capabilities unevenly because `data_preferences`
and bridge retrieval surfaces may not yet be as complete as wallet and vault
CRUD.

That is acceptable for MVP if the plan is explicit:

- ship strong vault, identity, wallet, and consent summaries first
- add richer preference controls as the zome surface stabilizes

## Recommended Implementation Order

If implementation starts immediately, the order should be:

1. scaffold `mycelix-personal/apps/leptos`
2. create zome client layer for identity-vault, health-vault, credential-wallet
3. build vault overview aggregation
4. build identity and wallet pages
5. build health summary and consents
6. wire portal domain module and launch routes
7. deepen preferences and activity visibility

## Acceptance Criteria

The first Personal frontend is successful when:

- a user can open one app and understand their private Mycelix posture
- the app makes credentials, consents, and profile state visible without
  exposing unnecessary raw complexity
- the handoff to Identity and Health is obvious and not duplicative
- the portal can summarize Personal in one card with meaningful state
- the implementation strengthens shared Leptos primitives rather than inventing
  new frontend infrastructure
