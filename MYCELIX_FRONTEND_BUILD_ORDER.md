# Mycelix Frontend Build Order

## Purpose

This document is the execution-grade frontend priority plan for the Mycelix
ecosystem as of 2026-04-21.

It is stricter than the earlier roadmap documents. Its job is to answer:

1. which frontend work should happen now
2. which apps should be deepened rather than replaced
3. which frontends are missing and worth building
4. what route-level slices should ship first
5. what should be deferred until the ecosystem contract is cleaner

This document complements:

- `FRONTEND_ARCHITECTURE.md`
- `FRONTEND_ROADMAP.md`
- `FRONTEND_MATRIX.md`
- `FRONTEND_EXECUTION_BACKLOG.md`
- `PERSONAL_FRONTEND_PLAN.md`
- `SENSORIUM_SUMMARY_CONTRACT.md`

Note on naming: `mycelix-sensorium` is now the current repo and crate path, and
the public-facing name for the canonical ecosystem shell is `Mycelix
Sensorium`.

## Executive Summary

The best way to make Mycelix better is not to start many new frontends at
once. The best move is to establish one canonical user-owned shell, one
canonical ecosystem shell, and one or two deeply credible domain apps that use
shared contracts instead of improvising local infrastructure.

That means:

1. finish `mycelix-personal/apps/leptos` as the reference vault shell
2. define and lock the frontend contract it proves
3. use that contract to sharpen `mycelix-sensorium` as Mycelix Sensorium
4. converge `mycelix-pulse/apps/leptos` toward the standard without flattening
   its product identity
5. deepen existing thin domain apps before opening many new greenfield apps

The largest risk right now is not missing screens. It is ecosystem drift:

- different provider stacks
- different route semantics
- different loading and write patterns
- duplicated identity, consent, and disclosure logic
- Sensorium surfaces that summarize domains inconsistently

## Strategic Rule Set

These rules should govern frontend work across Mycelix:

1. do not build a new major app if a higher-leverage coordination surface is
   still missing
2. prefer one real workflow over six placeholder routes
3. Personal owns vault posture, disclosure posture, and consent posture
4. Sensorium owns ecosystem orientation, launch, and cross-domain summary
5. domain apps own deep operational workflows
6. shared frontend improvements count as product work, not background cleanup
7. every strategic app should pass a route-level view pass before major build
8. raw Holochain records should not be the long-term frontend contract

## Ecosystem Classification

### Tier A: Core Canonical Surfaces

These are the frontends that should define the ecosystem experience.

- `mycelix-personal/apps/leptos`
  - role: sovereign vault shell
  - status: newly established, still incomplete
- `mycelix-sensorium/crates/sensorium-shell`
  - role: ecosystem orientation and launch shell, publicly named Mycelix Sensorium
  - status: strong concept, needs stricter product contract
- `mycelix-pulse/apps/leptos`
  - role: flagship communications product
  - status: real app, but architecturally more custom than the newer cluster
    pattern

### Tier B: Strong Domain References

These are existing apps that already show credible shape and should inform
future work.

- `mycelix-finance/apps/leptos`
- `mycelix-health/apps/leptos`
- `mycelix-knowledge/apps/leptos`
- `mycelix-climate/apps/leptos`
- `mycelix-energy/apps/leptos`
- `mycelix-supplychain/apps/leptos`
- `mycelix-identity/apps/leptos`
- `mycelix-praxis/apps/leptos`
- `mycelix-craft/apps/leptos`
- `mycelix-governance/apps/leptos`

These should be treated as deepen-and-converge surfaces, not rebuild targets.

### Tier C: Existing But Thin or Ambiguous Surfaces

- `mycelix-commons/apps/leptos`
  - real route inventory, but likely shallow domain depth relative to the
    breadth of the subject
- `mycelix-civic/apps/leptos`
  - scaffold-level state; not yet a real public workflow surface
- `mycelix-desci/apps/leptos`
  - sizable route inventory, but strategic role needs clarification relative to
    Knowledge
- `mycelix-music/apps/leptos`
  - interesting surface, but not currently a core ecosystem hinge
- `mycelix-space`
  - frontend direction remains mixed

### Tier D: Missing or Under-Productized Surfaces

- `mycelix-personal`
  - now partially resolved via new Leptos app
- `mycelix-attribution`
  - important concept, not yet a first-class product surface
- `mycelix-manufacturing`
  - backend-rich, frontend-thin
- `mycelix-position`
  - unclear standalone surface
- `mycelix-lawful-identity`
  - likely specialized admin or verification UI, still missing

### Tier E: Defer or Freeze

- `mycelix-pulse/ui/frontend`
  - legacy React/Vite line; preserve for reference only
- `mycelix-marketplace/frontend`
  - separate product line, not near-term core priority
- `mycelix-commerce/ui`
  - unclear current ecosystem role

## Dependency Graph

The frontend order should follow dependency reality, not repo order.

### Foundation Dependencies

`mycelix-leptos-core`
-> shared provider stack
-> route shell patterns
-> loading/error/toast conventions
-> Sensorium summary contract

### User-Shell Dependencies

`mycelix-personal`
-> typed identity, health, wallet, and preference views
-> unlock/session model
-> disclosure and consent controls
-> activity and audit semantics

### Ecosystem-Shell Dependencies

`mycelix-sensorium`
-> depends on stable Personal summary semantics
-> depends on stable launch-route conventions
-> depends on domain summary card shape

### Domain Dependencies

Domain apps should depend on:

- the shared substrate for bootstrapping
- Personal for vault and disclosure posture
- Sensorium for ecosystem orientation

They should not each reinvent:

- vault unlock UX
- consent posture
- disclosure history
- shared cross-domain launch semantics

## Recommended Build Order

## Phase 0: Contract Stabilization

This is the work that must be done before multiplying frontends further.

### Deliverables

1. Personal route contract
2. Sensorium summary and launch contract
3. app bootstrap checklist for new Leptos frontends
4. typed read/write adapter conventions for cluster apps
5. shared mutation pattern
   - optimistic local update only when safe
   - authoritative refresh after successful write
   - toast and rollback behavior defined

### Why First

Without this phase, every new app will continue encoding local assumptions.

## Phase 1: Finish Personal

Personal is the current highest-leverage frontend in the ecosystem.

### Goal

Turn `mycelix-personal/apps/leptos` from a promising shell into the canonical
user-owned vault product.

### Route-Level Build Order

1. `/unlock`
   - local lock state
   - session timeout
   - first-entry and re-entry behavior
   - secure handoff into `/`
2. `/`
   - vault overview
   - profile completeness
   - credential count
   - consent posture
   - recent activity
3. `/identity`
   - stable profile editing
   - key posture
   - handoff into Identity for DID, MFA, recovery
4. `/preferences`
   - domain-level sharing policy
   - reasoned allow/block posture
   - recent preference changes
5. `/health`
   - health summary
   - biometric entry
   - consent create, revoke, expire, inspect
   - handoff to Health for deeper record work
6. `/wallet`
   - credential inventory
   - proof and presentation actions
   - expiry and revocation state
7. `/activity`
   - query/event audit trail
   - who asked, why, what flowed, whether it succeeded

### Must-Have Work Before Calling Personal “Complete Enough”

1. unlock flow is real, not placeholder
2. health write flows are broader than consent create
3. wallet has at least one real proof or presentation action
4. activity trail explains disclosures in human terms
5. handoffs to Identity and Health are explicit and stable

### Personal Quality Bar

If a user cannot answer these questions after using Personal, it is not done:

- what is in my vault?
- what can I prove?
- what have I shared?
- who can access what?
- where do I go for deeper workflow detail?

## Phase 2: Sensorium Contract and Productization

Mycelix Sensorium should be the ecosystem shell, not an abstract art piece plus domain
orbits.

The current orbital model is valuable, but the product contract needs to be
sharper so the shell is actionable as well as evocative.

### Goal

Make `mycelix-sensorium/crates/sensorium-shell` the canonical cross-domain launch and
summary surface.

### Route and View Priority

Sensorium does not need many classic routes, but it does need stable view modes:

1. unlocked home state
   - identity summary
   - Personal vault posture
   - domain orbit summaries
   - recent cross-domain activity
2. domain zoom state
   - concise domain summary
   - 2-4 launch actions
   - current health of the domain surface
3. launch state
   - standard app deep-link behavior
   - fallback when a domain is unavailable or mock-only
4. notifications and attention state
   - open items
   - alerts
   - pending disclosures or approvals

### What Sensorium Should Not Do

Sensorium should not become:

- a second Personal
- a full domain dashboard clone
- a place where deep workflows are reimplemented

### Required Contracts

Every strategic domain should eventually expose:

1. a summary payload
2. a primary launch target
3. a small set of secondary launch targets
4. a stable icon, label, state, and urgency vocabulary
5. a clear fallback when live data is unavailable

### First Domains To Normalize In Sensorium

1. Personal
2. Pulse
3. Health
4. Finance
5. Knowledge
6. Commons

## Phase 3: Pulse Convergence

Pulse is already real. It should be improved through convergence, not
reimagining.

### Goal

Preserve Pulse's communications identity while reducing infrastructure drift.

### Priority Areas

1. provider stack normalization where shared primitives already exist
2. launch and summary integration with Sensorium
3. handoff into Personal for identity, credentials, and consent posture
4. explicit classification of legacy React frontend as frozen

### Keep As Product-Specific

These are likely justified local specializations:

- command palette richness
- offline behaviors
- mail-specific navigation model
- inbox and compose interaction patterns

Pulse should converge at the contract layer, not lose its product character.

## Phase 4: Deepen Thin Existing Apps

After Personal and Sensorium are stronger, the next best work is not necessarily
building brand-new domains. It is deepening the thin surfaces that already
exist.

### 4A. Commons

Priority: very high

Why:

- already has route inventory
- high systemic value
- likely still thin across too many subdomains

Recommended slice order:

1. `/resources`
2. `/care`
3. `/housing`
4. `/water`
5. `/transport`

Build one operational flow per slice before widening further.

### 4B. Civic

Priority: very high

Why:

- major narrative differentiator
- likely still scaffold-level

Recommended first slices:

1. justice intake or case queue
2. emergency coordination board
3. media verification flow

Do not attempt all of Civic at once.

### 4C. Attribution

Priority: high

Why:

- core ecosystem differentiator
- currently under-productized

Recommended surface:

Start summary-first, then decide whether a full standalone app is necessary.

## Phase 5: Clarify Before Building

These domains need product-definition work before major frontend investment.

### DeSci

Question:

Is DeSci a distinct app family, a specialized Knowledge extension, or both?

Action:

perform a dedicated view pass and relationship pass against Knowledge before
more large-scale UI work

### Space

Question:

Is Space Svelte-first, Leptos-first, or Sensorium-summary-first?

Action:

decide canonical frontend direction before broad UI expansion

### Music

Question:

Is Music a flagship experiential app, a demo-heavy app, or a supporting
surface?

Action:

classify before prioritizing

### Manufacturing, Position, Lawful Identity

Question:

Are these standalone-first, admin-first, or embedded-first?

Action:

do product-role definition before frontend implementation

## Now / Next / Later / Defer

### Now

- finish Personal verticals
- lock Personal route and data contract
- normalize Sensorium summary and launch contract
- converge Pulse at the provider and handoff layer

### Next

- deepen Commons
- build Civic first real workflows
- productize Attribution summaries

### Later

- Manufacturing
- Position
- Lawful Identity

### Defer Until Clarified

- DeSci expansion
- Space expansion
- Music expansion
- marketplace and commerce lines

## View Pass Standard

Every strategic frontend should pass a formal view pass before major build or
major refactor.

The pass should include:

1. product role
   - what this app is for
   - what it explicitly does not own
2. route map
   - top-level routes
   - route priority order
   - which routes must be real in MVP
3. state boundaries
   - what state is local
   - what state comes from the conductor
   - what state is derived or cached
4. contract dependencies
   - which typed views and mutations are required
   - which shared primitives are expected
5. handoff boundaries
   - which deeper domain app this app launches into
   - where duplication is forbidden
6. security posture
   - unlock requirements
   - sensitive data exposure rules
   - audit or disclosure expectations

## Concrete 30-Day Focus

If the team wants the highest-value next month of frontend work, it should be:

1. Personal
   - finish unlock
   - finish health and wallet writes
   - strengthen activity trail
2. Sensorium
   - implement Personal-first summary contract
   - standardize launch semantics
3. Pulse
   - reduce drift where safe
   - add explicit Personal handoffs
4. Commons
   - choose one real operational slice and make it credible

This is a better order than starting several missing apps because it increases
coherence across the whole ecosystem.

## Definition of Better

Mycelix frontend work is getting better when:

1. users can tell which app owns which problem
2. shared flows feel consistent without feeling visually identical
3. Personal becomes the default place for identity, consent, and disclosure
   posture
4. Sensorium becomes the default place for ecosystem orientation and launch
5. domain apps become deeper rather than merely broader
6. new frontends can be scaffolded from a clear contract instead of copied from
   whichever old app happened to be nearby

## Immediate Recommended Actions

1. continue implementation in `mycelix-personal/apps/leptos` until unlock,
   wallet action, and richer health writes are real
2. create a small Sensorium summary contract note derived from Personal
3. classify Pulse convergence tasks into keep, merge, or freeze
4. choose exactly one Commons operational slice for real build-out
5. schedule clarification passes for DeSci and Space before any major new UI
   investment there
