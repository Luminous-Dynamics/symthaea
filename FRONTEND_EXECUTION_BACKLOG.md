# Mycelix Frontend Execution Backlog

## Purpose

This document turns the frontend roadmap into an execution backlog.

It is intended to answer:

1. what should be built first
2. what each phase must deliver
3. what dependencies need to be resolved before work begins
4. how to tell when a phase is actually complete

This backlog complements:

- `FRONTEND_ARCHITECTURE.md`
- `FRONTEND_ROADMAP.md`
- `FRONTEND_MATRIX.md`

## Planning Rules

The following rules should govern frontend execution across Mycelix:

- do not start a new major frontend if a higher-priority dependency surface is
  still missing
- prefer converging active apps before multiplying custom infrastructure
- require Sensorium summary and launch contracts for every strategic domain
- ship thin but real workflows before broad but shallow page inventories
- treat shared-substrate improvements as product work, not background cleanup

## Phase Structure

Each phase should produce four kinds of output:

- product scope
- shared-substrate changes
- Sensorium integration
- proof of completion

## Phase 0: Foundation Convergence

### Objective

Reduce architectural drift before the next wave of domain frontend builds.

### Scope

- converge Pulse toward the canonical cluster frontend pattern
- lock the Sensorium summary and launch contract
- identify which shared providers belong in every cluster app by default
- define a repeatable app bootstrap shape for new Leptos frontends

### Work Items

1. Pulse runtime convergence
   - review local provider stack
   - collapse custom runtime scaffolding where shared primitives now exist
   - preserve Pulse-specific UX where it is part of the product identity
2. Shared-shell standardization
   - define the default `AppShell` composition for cluster apps
   - define theme/token override boundaries
   - define how toasts and loading states should be surfaced consistently
3. Sensorium contract stabilization
   - standardize metadata required by each domain module
   - standardize summary card data shape
   - standardize launch-route registration and deep-link expectations
4. New-app template baseline
   - write down the expected crate/app structure for future cluster frontends
   - identify the minimum files a new frontend should own locally vs inherit

### Dependencies

- current behavior in Pulse must be preserved during convergence
- `mycelix-leptos-core` and `mycelix-leptos-client` ownership boundaries must be
  explicit
- portal-domain trait expectations for Sensorium must be stable enough to document

### Exit Criteria

- Pulse is classified as `Canonical` or has a short remaining delta list
- Sensorium contracts are explicit enough that new domains can implement them
  without bespoke interpretation
- a new frontend can be scaffolded from a consistent baseline rather than by
  copying a random existing app

### Recommended Deliverables

- Pulse convergence checklist
- Sensorium summary contract note
- frontend bootstrap template note

## Phase 1: Personal

### Objective

Create the sovereign user shell that unifies personal vault, identity-linked
credentials, and data preference management.

### Why This Comes First

Personal has the highest cross-domain leverage among missing frontends.

Without it, Mycelix has domain apps but no strong user-owned coordination
surface for:

- credentials
- permissions
- personal records
- data-sharing posture

### MVP Scope

- onboarding and vault entry
- credential wallet
- identity-linked profile and trust handoff
- health-vault summary handoff
- data-sharing and consent preferences
- Sensorium launch integration

### Shared-Substrate Work

- reusable vault-shell primitives
- secure state and lock-state UI patterns
- credential card/list/detail primitives
- sensitivity-state indicators that other domains can reuse

### Sensorium Work

- Personal summary module
- recent credentials and trust-state summary
- launch routes into wallet, preferences, and vault sections

### Dependencies

- Identity frontend and backend routes should be stable enough for handoff
- Health summary handoff points should be defined
- Sensorium launch conventions must already exist from Phase 0

### Exit Criteria

- a user can enter Personal and understand where their identity, credentials,
  and data preferences live
- at least one real handoff to Identity and one to Health works cleanly
- Sensorium summary cards provide meaningful orientation without duplicating the
  full app

### Risks

- overbuilding into a universal dashboard instead of a focused personal vault
- duplicating flows that should remain inside Identity or Health

## Phase 2: Civic

### Objective

Turn the Civic scaffold into a real operational frontend for justice,
emergency, and media-trust workflows.

### Why Now

Civic has high narrative value and strategic differentiation, but its current
frontend state is still too skeletal.

### MVP Scope

- justice case or docket queue
- emergency coordination panel
- media verification or signal validation flow
- trust and legitimacy indicators

### Shared-Substrate Work

- queue/triage interaction patterns
- urgency and risk-state UI components
- provenance and verification status presentation

### Sensorium Work

- Civic summary module
- alerts, open items, and active coordination summaries
- launch routes into justice and emergency views

### Dependencies

- domain workflows must be prioritized; Civic is broad enough to sprawl quickly
- verification and legitimacy signals need a stable domain-language mapping

### Exit Criteria

- Civic is no longer a shell with placeholders
- at least two concrete workflows are executable end-to-end
- Sensorium summary surfaces actionable state, not just descriptive copy

### Risks

- trying to build justice, emergency, and media as three separate products at
  once
- investing in visuals before the operational flows are credible

## Phase 3: Commons Deepening

### Objective

Convert Commons from a thin shell into a product with real subdomain utility.

### Why Before New Adjacent Apps

Commons already exists. Deepening it is higher leverage than opening another
greenfield frontend with the same coordination concepts.

### MVP Scope

- choose 2-3 subdomains with clear backend maturity
- build at least one operational workflow per selected subdomain
- make resource and stewardship state visible in a non-placeholder way

### Candidate Slices

- watershed
- housing
- care
- resource coordination

### Shared-Substrate Work

- reusable resource-state cards and maps
- stewardship-status and contribution-state components
- domain-switching patterns if Commons retains multiple subdomain views

### Sensorium Work

- Commons summary module
- subdomain-specific highlights
- shortcuts into live workflows

### Dependencies

- a small number of subdomains must be chosen explicitly before implementation
- avoid turning Commons into an unbounded meta-app

### Exit Criteria

- Commons contains real operational views rather than mainly shell navigation
- at least one subdomain has enough depth to demo as a serious product surface
- Sensorium summary reflects live state from real workflows

## Phase 4: Attribution Productization

### Objective

Move Attribution from observability-adjacent visibility into a stable
user-facing product surface.

### Why Here

Attribution is a distinctive ecosystem capability, but today it appears too
buried inside dashboard-style surfaces.

### MVP Scope

- attribution summary cards in the Sensorium
- dependency and usage receipt visibility
- reciprocity or contribution explanation surface
- links into any deeper attribution records

### Shared-Substrate Work

- lineage and provenance display patterns
- contribution/receipt timeline components
- reciprocity indicators

### Sensorium Work

- likely the primary first release surface
- determine whether Attribution remains summary-first or merits standalone app

### Dependencies

- attribution concepts must be expressed in user-facing language, not only
  system language
- source data must be stable enough to avoid misleading summaries

### Exit Criteria

- a user can understand what they contributed, what they received, and what is
  owed or credited
- Attribution no longer depends on internal observability framing to be usable

## Phase 5: Manufacturing

### Objective

Create the first clear user-facing operational frontend for manufacturing.

### Why Here

Manufacturing appears to have meaningful backend structure, but no equivalent
operator-facing UI.

### MVP Scope

- job or work-order overview
- material/resource state
- progress and exception visibility
- handoff into supply-chain coordination when relevant

### Shared-Substrate Work

- workflow-stage components
- operator dashboards
- exception-state treatments

### Sensorium Work

- lightweight manufacturing operations summary
- launch routes into active work-order views

### Dependencies

- backend maturity must be validated before UI work starts
- relationship to Supply Chain must be explicit to avoid duplicated surfaces

### Exit Criteria

- manufacturing work can be monitored and acted on through a real frontend
- the app has a clear role distinct from Supply Chain

## Phase 6: Position

### Objective

Decide whether Position should be embedded into other apps, exposed as a
specialized standalone surface, or both.

### Why Not Earlier

Position may be a capability layer rather than a broad standalone product. Its
role should be clarified before major UI investment.

### Discovery Questions

- who is the primary operator?
- what decisions happen in Position directly?
- which domains consume Position data or tooling?
- is the first release embedded-first rather than app-first?

### Exit Criteria

- Position is classified as embedded-first, standalone-first, or hybrid
- if standalone is justified, a narrow MVP is defined

## Phase 7: Lawful Identity

### Objective

Build a specialized frontend only if the domain genuinely needs operator-facing
verification and review workflows beyond current Identity surfaces.

### Why Later

This likely serves narrower operator roles than Personal, Civic, or Commons.

### MVP Candidates

- verification queue
- credential review
- audit or policy status
- lawful attestation workflow

### Exit Criteria

- either a focused admin/operator app is justified and scoped
- or the required workflows are folded into Identity and Personal

## Phase 8: Space Clarification

### Objective

Settle the frontend direction for Space before additional implementation.

### Why Last

Space already has multiple UI directions. More implementation before strategic
clarity would likely increase fragmentation.

### Decision Options

- keep Svelte as the canonical UI
- migrate toward Leptos cluster conventions
- use Sensorium summaries and leave Space as a specialized adjunct tool

### Exit Criteria

- one direction is chosen
- the alternative directions are explicitly frozen or deprecated

## Cross-Cutting Backlog

These items should be worked in parallel with the domain phases where useful.

### Shared Frontend Platform

- strengthen `mycelix-leptos-core` as the standard shell and primitive library
- keep `mycelix-leptos-client` as the default browser-native Holochain path
- document the minimum provider stack for a canonical cluster frontend
- define CSS token boundaries for local brand variation vs shared foundations

### Sensorium Contracts

- domain metadata schema
- summary-card data schema
- launch/deep-link registration rules
- domain health and readiness indicators

### UX System

- loading and empty-state conventions
- error-state conventions
- urgency/sensitivity/provenance visual language
- keyboard and dense-workflow interaction expectations

### Product Governance

- classify every active frontend as canonical, converging, justified deviation,
  thin, scaffold, clarify, or legacy
- refuse new custom app scaffolds that bypass the shared substrate without an
  explicit reason
- review roadmap position quarterly

## Immediate Next Actions

If work starts now, the most defensible sequence is:

1. finish Phase 0 convergence notes for Pulse and Sensorium contracts
2. create a dedicated `PERSONAL_FRONTEND_PLAN.md`
3. define Civic MVP workflow selection before any broad visual redesign work
4. choose the 2-3 Commons subdomains for deepening

## Definition of Done for the Roadmap

The roadmap is succeeding when:

- the ecosystem has fewer duplicate frontend lines
- new frontends start from a clear canonical baseline
- the Sensorium becomes a meaningful ecosystem shell instead of a loose index
- missing strategic domains gain real user-facing surfaces
- active apps are judged by workflow depth, not by page count
