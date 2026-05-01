# Mycelix Sensorium Summary Contract

## Purpose

This document defines the next contract layer needed by `mycelix-sensorium`, the
current repo and crate path for the public-facing shell now named `Mycelix
Sensorium`.

The existing Sensorium contract already covers:

- domain identity
- domain colors
- civic tier gating
- navigation metadata
- cluster dependencies
- entry-type inventory

What it does not yet cover is the thing Sensorium actually needs in order to be a
real ecosystem shell:

- a typed summary payload for each domain
- stable launch targets
- consistent attention and urgency semantics
- fallback behavior when live conductor data is absent

This document proposes that contract.

It is derived from:

- the new `mycelix-personal/apps/leptos` shell
- the current `sensorium-domain-trait`
- the current Sensorium domain overview pages

## Problem

Right now Sensorium can know:

- what a domain is called
- what color it should glow
- what tier is allowed to see it
- what sidebar items it exposes

But it cannot yet know, in a normalized way:

- what state to summarize
- what action to emphasize
- whether the domain needs the user's attention
- what route should open first
- how to represent locked, empty, mock, degraded, or live states

That leads to Sensorium being evocative but under-specified as a product shell.

## Product Rule

Sensorium should not clone each domain app.

Sensorium should provide:

1. orientation
2. health and attention state
3. recent cross-domain movement
4. a small number of recommended launch actions
5. fallback and locked-state guidance

Domain apps should still own deep workflows.

## Contract Layers

The full Sensorium contract should be split into four layers.

### 1. Domain Identity Layer

This already mostly exists in `sensorium-domain-trait`.

It includes:

- `id`
- `name`
- `bio_name`
- `description`
- `color_family`
- `min_tier`
- `nav_items`
- `dependencies`
- `entry_types`

### 2. Launch Layer

This is missing today.

Sensorium needs a stable way to know:

- the primary route to open for a domain
- secondary routes worth surfacing
- whether launch is external, internal, or unavailable
- what to do if the domain is present but not yet live

### 3. Summary Layer

This is the core missing piece.

Sensorium needs a typed payload that can be rendered without domain-specific
special cases.

### 4. Attention Layer

Sensorium needs a consistent vocabulary for:

- urgency
- unread or pending items
- approvals needed
- degraded state
- locked state

Without that, the shell cannot be meaningfully action-oriented.

## Proposed Core Types

These are conceptual shapes for the next iteration of
`mycelix-sensorium/crates/sensorium-domain-trait`.

```rust
pub enum DomainAvailability {
    Live,
    Mock,
    Empty,
    Locked,
    Degraded,
    Unavailable,
}

pub enum LaunchKind {
    InternalRoute,
    ExternalApp,
    Disabled,
}

pub struct DomainLaunchTarget {
    pub id: &'static str,
    pub label: &'static str,
    pub path: &'static str,
    pub kind: LaunchKind,
    pub requires_unlock: bool,
    pub recommended: bool,
}

pub enum AttentionLevel {
    Quiet,
    Notice,
    ActionNeeded,
    Urgent,
}

pub struct DomainAttentionItem {
    pub id: String,
    pub label: String,
    pub detail: String,
    pub level: AttentionLevel,
    pub path: Option<String>,
}

pub struct DomainMetric {
    pub id: &'static str,
    pub label: String,
    pub value: String,
    pub hint: Option<String>,
    pub tone: Option<&'static str>,
}

pub struct DomainSummaryCard {
    pub domain_id: &'static str,
    pub title: String,
    pub availability: DomainAvailability,
    pub status_line: String,
    pub metrics: Vec<DomainMetric>,
    pub attention: Vec<DomainAttentionItem>,
    pub primary_launch: DomainLaunchTarget,
    pub secondary_launches: Vec<DomainLaunchTarget>,
    pub updated_at: Option<i64>,
}
```

The key point is not the exact Rust syntax. The key point is that the Sensorium
shell should consume a uniform summary object instead of reaching into each
domain with ad hoc assumptions.

## Minimum Summary Requirements

Every strategic domain should eventually provide one summary card with:

1. availability
2. one status line
3. three to five meaningful metrics
4. zero to three attention items
5. one primary launch target
6. up to three secondary launch targets
7. a freshness timestamp when live data exists

If a domain cannot provide that, it is not ready for first-class Sensorium
presence.

## Standard Availability Semantics

These values must mean the same thing across all domains.

### `Live`

- connected to conductor or equivalent live backend
- summary reflects current state
- launch should work

### `Mock`

- app shell is usable
- values are illustrative
- user should not mistake it for live state

### `Empty`

- domain is live
- no records or actions exist yet
- launch should help the user begin

### `Locked`

- user is permitted to access the domain
- sensitive content is intentionally withheld pending unlock or consent

### `Degraded`

- partial data available
- one or more required dependencies are failing
- launch may still work, but with caution

### `Unavailable`

- domain not compiled, not installed, or not reachable

## Standard Attention Semantics

Sensorium should use the same attention language everywhere.

### `Quiet`

No action is needed. Domain is healthy.

### `Notice`

Informational state worth surfacing, but not actionable urgency.

Examples:

- new knowledge claims
- new recognition score
- recent health biometrics recorded

### `ActionNeeded`

Something should be handled soon.

Examples:

- consent expiring
- treasury vote pending
- unpaid invoice
- unresolved fact-check assignment

### `Urgent`

Immediate, high-salience action.

Examples:

- emergency coordination item
- security or vault issue
- critical domain degradation

## Launch Contract

Every strategic domain should expose:

1. one primary launch target
2. up to three secondary launch targets
3. explicit unlock requirement per launch target
4. explicit disabled behavior if the domain cannot be entered

### Primary Launch Rule

The primary launch target should be the route a user most likely wants from
Sensorium.

Examples:

- Personal -> `/`
- Health -> `/records` or `/consent`, depending on actual product priority
- Finance -> `/treasury` or `/tend`
- Knowledge -> `/browse`
- Pulse -> `/`

### Secondary Launch Rule

Secondary targets should represent common intent pivots, not the full route
table.

Sensorium should not mirror the entire domain nav.

## Personal-First Summary Contract

Personal should be the first domain to implement this contract because it is
the user-owned anchor for the ecosystem.

### Personal Summary Fields

Sensorium should be able to ask Personal for:

1. vault state
   - unlocked, locked, unavailable
2. profile posture
   - display name present or missing
   - key posture available
3. credential posture
   - total credentials
   - expiring or revoked credentials
4. consent posture
   - active consent count
   - consent changes needing review
5. health posture
   - recent biometric or record activity count
6. disclosure posture
   - recent bridge activity
   - number of recent outbound disclosures

### Personal Primary Launch

- `/`

### Personal Secondary Launches

- `/identity`
- `/preferences`
- `/activity`

### Personal Attention Examples

- vault locked
- credential expiring soon
- new disclosure event
- consent changed or revoked
- no recovery or key posture configured

## Example Personal Summary Shape

```rust
DomainSummaryCard {
    domain_id: "personal",
    title: "Sovereign Vault".into(),
    availability: DomainAvailability::Locked,
    status_line: "Vault locked. Identity and disclosure posture available after unlock.".into(),
    metrics: vec![
        DomainMetric { id: "credentials", label: "Credentials".into(), value: "8".into(), hint: None, tone: None },
        DomainMetric { id: "consents", label: "Active Consents".into(), value: "3".into(), hint: Some("1 changed recently".into()), tone: Some("notice") },
        DomainMetric { id: "activity", label: "Recent Disclosures".into(), value: "2".into(), hint: Some("last 7 days".into()), tone: Some("notice") },
    ],
    attention: vec![
        DomainAttentionItem {
            id: "vault-locked".into(),
            label: "Unlock required".into(),
            detail: "Sensitive vault content is hidden until unlock.".into(),
            level: AttentionLevel::ActionNeeded,
            path: Some("/unlock".into()),
        }
    ],
    primary_launch: DomainLaunchTarget {
        id: "vault",
        label: "Open Vault",
        path: "/",
        kind: LaunchKind::InternalRoute,
        requires_unlock: true,
        recommended: true,
    },
    secondary_launches: vec![
        DomainLaunchTarget {
            id: "preferences",
            label: "Review Preferences",
            path: "/preferences",
            kind: LaunchKind::InternalRoute,
            requires_unlock: true,
            recommended: false,
        }
    ],
    updated_at: Some(1_713_553_600_000_000),
}
```

## Domain-Specific Summary Guidance

The shell should be uniform, but the actual metrics should be domain-relevant.

### Health

Good metrics:

- record count
- active consents
- privacy budget remaining
- FL dividend total

Good attention items:

- expiring consent
- vault locked
- privacy budget low

Good primary launch:

- consent or records, depending on product emphasis

### Finance

Good metrics:

- balance
- staked amount
- pending rewards
- recent payment flow

Good attention items:

- claimable rewards
- pending treasury action
- failed payment or degraded ledger state

### Knowledge

Good metrics:

- claims under review
- verified claims
- prediction market volume
- fact-check queue size

Good attention items:

- review requested
- disputed claim linked to the user
- market nearing resolution

## Fallback Rules

Sensorium needs deterministic fallback behavior.

### If Summary Fetch Fails

- show `Degraded` or `Unavailable`
- preserve launch metadata if still known
- show one shell-level explanation instead of domain-specific noise

### If Domain Is Installed But Empty

- show `Empty`
- primary launch should be a begin or onboarding path

### If Domain Requires Personal Unlock

- show `Locked`
- route through Personal or domain-local unlock flow according to product role

## Implementation Order

This contract should be introduced in order.

1. document the summary and launch contract
2. implement Personal summary provider first
3. render one generic summary card in Sensorium from that payload
4. add Health and Finance
5. normalize Knowledge and Commons
6. then expand to the rest of the ecosystem

Do not implement summary payloads for every domain at once.

## Suggested Next Code Step

The next code step should be to extend `sensorium-domain-trait` with a summary and
launch layer, but only behind a minimal first pass.

That first pass should include:

1. `DomainAvailability`
2. `DomainLaunchTarget`
3. `DomainMetric`
4. `DomainSummaryCard`
5. one Personal implementation returning mock or live summary data

That is enough to prove the contract without overcommitting to a final shape.

## Definition of Done

Sensorium summary integration is “good enough to scale” when:

1. Personal can provide a typed summary payload
2. Sensorium can render that payload generically
3. locked, mock, empty, degraded, and live states are distinct
4. launch actions are consistent across domains
5. adding a new domain summary does not require a new page-specific ad hoc
   rendering strategy
