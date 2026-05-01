# Mycelix Frontend Architecture

## Purpose

This document defines the target frontend architecture for the Mycelix
ecosystem.

It exists to answer five recurring questions:

1. What is shared across Mycelix frontends?
2. What belongs inside a standalone domain app?
3. What belongs inside the unified Sensorium shell?
4. When is a frontend deviation justified?
5. Which surfaces should be converged, frozen, or retired?

This is an architectural baseline, not a final visual-design decree.

## Frontend Categories

Mycelix has three frontend categories.

### 1. Shared Substrate

The shared substrate provides generic runtime and UI systems for Leptos-based
Mycelix applications.

Primary crates:

- `mycelix-leptos-client`
- `mycelix-leptos-core`
- `mycelix-leptos-i18n`

Responsibilities:

- browser/WASM Holochain transport
- typed zome calls
- connection/runtime handling
- provider stack primitives
- app shell primitives
- toasts, badges, loading states
- graph and telemetry widgets
- shared theme and token foundations
- shared localization support

Non-responsibilities:

- domain workflows
- domain-specific page state
- product-specific shells where the interaction model is unique

### 2. Standalone Domain Apps

Standalone domain apps are the primary deep-work surfaces for Mycelix users.

Examples:

- Pulse
- Health
- Knowledge
- Finance
- Governance
- Identity
- Praxis
- Commons
- Hearth
- Craft
- Climate
- Energy
- Supply Chain

Responsibilities:

- domain workflows
- domain-specific state and orchestration
- domain-specific navigation where needed
- domain-specific visual language layered on top of shared foundations

These are the places where real work should happen.

### 3. Sensorium Shell

`mycelix-sensorium` is the unified ecosystem shell, publicly named `Mycelix
Sensorium`.

Responsibilities:

- cross-domain orientation
- sovereignty overview
- activity synthesis
- discovery
- domain launch and deep-linking

Non-responsibilities:

- replacing every standalone app
- duplicating full domain workflows
- becoming a second-rate clone of every product surface

The Sensorium is the ecosystem shell, not the universal replacement UI.

## Canonical Cluster Frontend Standard

A frontend should be considered a canonical Mycelix cluster app when it follows
this shape by default:

- `HolochainProviderAuto` or a thin wrapper around it
- shared provider stack where relevant:
  - theme
  - consciousness
  - homeostasis
  - toasts
  - thermodynamic state
  - i18n when needed
- shared shell primitives where appropriate:
  - `AppShell`
  - `ToastContainer`
  - `ClusterLauncher`
- domain-specific pages and state layered on top
- domain-specific CSS and token mapping layered over shared base tokens

Current reference implementations are closest to this pattern:

- Climate
- Energy
- Finance
- Knowledge
- Supply Chain

These apps should be treated as the default architectural shape for new
standalone cluster frontends unless there is a strong product reason to do
otherwise.

## Sensorium Integration Standard

Every domain should integrate with the Sensorium in three layers.

### 1. Metadata Layer

This layer describes the domain to the Sensorium.

Required concepts:

- domain id
- display name
- tier gate
- dependency list
- zome inventory
- entry-type inventory
- color family
- launch routes

This mostly aligns with the existing `DomainModule` contract.

### 2. Summary Layer

This layer gives the Sensorium enough information to orient the user without
replacing the standalone app.

Typical summary content:

- important counts
- recent activity
- alerts or risk indicators
- upcoming actions
- shortcuts into real workflows

Summary views should be lightweight and action-oriented.

### 3. Launch Layer

This layer deep-links from the Sensorium into the standalone domain app.

Examples:

- `/mail/inbox`
- `/mail/trust`
- `/health/records`
- `/finance/...`
- `/knowledge/...`

The Sensorium should summarize and route. Standalone apps should execute deep work.

## Standalone vs Sensorium Responsibilities

The core rule is:

- The Sensorium owns cross-domain context.
- Standalone apps own deep domain work.

### The Sensorium Should Own

- identity and access orientation
- domain visibility and availability
- sovereignty overview
- data sensitivity overview
- dependency and topology visualization
- cross-domain activity streams
- domain discovery
- launch affordances
- lightweight summary cards

### Standalone Apps Should Own

- dense operational workflows
- high-frequency task execution
- keyboard-heavy productivity interactions
- domain-specific editing, authoring, or triage flows
- detailed state transitions and specialized tooling

## Pulse Position

Pulse should be treated as:

- the canonical communications domain app
- standalone-first
- Sensorium-summary-integrated
- desktop-wrapper-supported
- mobile-wrapper-compatible

### Sensorium Should Surface From Pulse

- unread count
- priority or trust-weighted summary
- recent high-signal messages
- upcoming calendar or meet events
- communication health indicators
- trust anomalies
- offline queue state
- launch links to Inbox, Compose, Trust, and Calendar/Meet

### Pulse Should Retain

- inbox triage
- compose, reply, forward
- thread reading
- trust review at message level
- chat, meet, and calendar workflows
- dense split-pane workspace behavior
- product-specific communication UX

Pulse is not just another dashboard app. It is a heavy productivity workspace.

## Health Position

Health should remain a standalone-first domain app with a strong domain-authored
shell.

The Sensorium may summarize:

- consent posture
- privacy status
- recent access events
- high-level record state

The standalone app should retain:

- records workflows
- consent actions
- privacy actions
- sensitive patient interactions

## Sensorium Position

The Sensorium should be treated as:

- the ecosystem shell
- the sovereignty and orientation layer
- the cross-domain synthesis surface

The Sensorium should not attempt to become:

- a full inbox
- a full health records system
- a full governance workbench
- a full finance console

## Justified Deviation Rules

Not every app needs to look or behave identically.

A deviation from the canonical cluster pattern is justified when:

- the domain requires a fundamentally different work surface
- the shared shell does not yet support the domain’s interaction model
- the app is intentionally a shell-level experience rather than a domain app

Examples of justified deviation:

- Sensorium
- Pulse
- Health
- Hearth

Even justified deviations should still prefer the shared substrate for:

- Holochain transport/runtime
- connection status handling
- toast mechanisms
- badges and generic feedback primitives
- theme plumbing where practical
- reusable graph or telemetry primitives

Rule of thumb:

- deviate in workflow and presentation
- avoid unnecessary deviation in generic infrastructure

## Keep / Freeze / Retire Policy

### Keep

- canonical standalone Leptos domain apps
- the Sensorium
- desktop wrappers around canonical apps
- mobile wrappers when aligned with canonical frontend surfaces

### Freeze

- legacy or parallel frontends retained for reference, migration, or asset reuse

### Retire

- duplicate product surfaces that compete with the canonical frontend without
  adding architectural value

## Migration Priorities

### Priority 1: Pulse

Pulse is the highest-value convergence target.

Target outcomes:

- wrap shared `HolochainProviderAuto`
- unify or extract toast behavior
- normalize theme base where safe
- keep Pulse-specific workspace UX
- freeze the old React frontend as non-canonical

### Priority 2: Sensorium

Target outcomes:

- clarify role as shell, not super-app clone
- add a stronger summary and launch model
- consume more shared substrate where useful

### Priority 3: Praxis

Target outcomes:

- continue reducing custom runtime duplication
- preserve Praxis-specific learning UX

### Priority 4: Outliers

Apps that need focused review:

- Music
- DeSci
- Commons
- Governance

## Design-System Direction

The target is:

- shared systems
- shared primitives
- domain-owned skins and interaction language

The target is not:

- one rigid visual shell for all apps

This preserves coherence without flattening the product family.

## Architectural Decisions

The following decisions are considered the target baseline:

1. `mycelix-leptos-client` is the canonical browser Holochain transport layer.
2. `mycelix-leptos-core` is the canonical shared frontend substrate for Leptos
   apps in Mycelix.
3. Standalone cluster apps are the primary deep-work environments.
4. `mycelix-sensorium` is the ecosystem shell, publicly named `Mycelix Sensorium`,
   not the replacement for standalone
   apps.
5. Pulse is the canonical communications app and the top convergence target.
6. Parallel legacy frontends should be frozen unless there is a specific
   migration reason to keep them active.
7. Deviations in shell and workflow are acceptable; unnecessary duplication of
   generic frontend infrastructure is not.

## Near-Term Deliverables

The architecture becomes operational when paired with the following documents:

- `FRONTEND_MATRIX.md`
  - app-by-app classification and next action
- `PULSE_MIGRATION_PLAN.md`
  - convergence plan for Pulse runtime and shared systems
- `PORTAL_INTEGRATION_STANDARD.md`
  - metadata, summary, and launch contract for domains

## Status

This document is the current target architecture baseline for Mycelix
frontends. It should be refined as additional architecture passes clarify:

- Sensorium summary contracts
- Pulse convergence boundaries
- treatment of outlier apps
- shared substrate ownership and maintenance
