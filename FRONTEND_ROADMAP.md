# Mycelix Frontend Roadmap

> Detailed execution priority, route-level sequencing, and now/next/later
> classification now live in `MYCELIX_FRONTEND_BUILD_ORDER.md`.

## Purpose

This document proposes a build order for the Mycelix frontend ecosystem.

The goal is not to build every frontend as fast as possible. The goal is to
build the right frontends in the right order so that:

- the ecosystem gains usable end-to-end product paths
- the shared frontend substrate keeps improving
- Sensorium integration becomes meaningful
- thin or duplicate surfaces do not consume attention ahead of missing
  high-leverage products

This roadmap assumes the architecture baseline in `FRONTEND_ARCHITECTURE.md`.

Note on naming: this document now references `mycelix-sensorium` as the repo and
crate path, but the intended public-facing product name for the canonical
ecosystem shell is now `Mycelix Sensorium`.

## Current State Summary

### Strong Existing Frontend Wave

The ecosystem already has a substantial first wave of standalone frontends:

- Pulse
- Praxis
- Craft
- Climate
- Energy
- Finance
- Governance
- Health
- Hearth
- Identity
- Knowledge
- Music
- Commons
- Supply Chain
- DeSci
- Civic scaffold
- Sensorium shell (`mycelix-sensorium`)

### Sensorium Summary State

The Sensorium is no longer purely decorative summary chrome.

- Personal summary is now live-backed from the conductor
- Governance summary can now be treated as ready for live Sensorium integration
- Commons summary should move to partial-live integration immediately, but its
  standalone frontend still has a structural blocker: `apps/leptos` references
  `crates/commons-leptos-types`, and that crate is not present in the current
  repository tree

This matters for build order. Domains with stable WASM-safe view types can be
integrated into Sensorium faster and more honestly than domains whose frontend
type layer is still implied rather than present.

### Non-Leptos or Parallel Frontend Surfaces

These exist, but are not the main direction for the Mycelix Leptos cluster
family:

- `mycelix-sensorium` mobile wrapper for Mycelix Sensorium
- `mycelix-marketplace` SvelteKit frontend
- `mycelix-space` SvelteKit dashboard plus partial Leptos directory
- `mycelix-attribution` dashboard surfaced via observatory
- `mycelix-commerce/ui`
- Pulse React/Vite legacy frontend

These should be handled selectively, not treated as the center of gravity.

### Strategically Important Gaps

The biggest frontend gaps in the core Mycelix ecosystem appear to be:

- `mycelix-personal` has no standalone user-facing app despite being the vault
  and wallet layer for identity, health, and credentials
- `mycelix-civic` has a scaffold, but still needs the actual justice,
  emergency, and media workflows
- `mycelix-commons` has a shell, but domain-specific views are still described
  as thin
- `mycelix-commons` frontend type infrastructure is incomplete in-tree because
  `commons-leptos-types` is referenced by the app but missing from the repo
- `mycelix-attribution` appears to rely on observatory/dashboard integration
  rather than a first-class cluster app
- `mycelix-manufacturing` has substantial backend structure but no clear
  frontend surface
- `mycelix-position` has no visible primary frontend surface
- `mycelix-lawful-identity` appears CLI and backend heavy with no visible user
  app

## Priority Model

Frontend priority should be driven by six factors:

1. **Cross-ecosystem leverage**
   - does this frontend unlock or improve multiple other domains?
2. **User path completeness**
   - does this complete a meaningful end-to-end product story?
3. **Backend maturity**
   - is there enough real system behind it to justify UI work?
4. **Shared-substrate value**
   - will building it improve `mycelix-leptos-core` or Sensorium integration?
5. **Demo and narrative value**
   - does it help people understand Mycelix as a coherent system?
6. **Risk of fragmentation**
   - if not built, will users be forced into ad hoc or duplicate interfaces?

## Build Order

## Phase 0: Converge the Foundation

This phase is not about adding a new domain app. It is about reducing frontend
drift before the next wave of domain UIs is built.

### Goals

- converge Pulse toward the canonical cluster frontend standard
- stabilize the shared Leptos substrate
- clarify Sensorium summary vs standalone responsibilities

### Deliverables

- Pulse migration plan execution, especially runtime/provider convergence
- Sensorium summary and launch contract definition
- Sensorium live-summary adapters for domains with stable conductor-facing types
- app-by-app frontend matrix maintained as a living reference

### Why First

If the foundation remains fragmented, every new app will repeat the same
architectural choices and inconsistencies.

## Phase 1: Personal

### Priority

Very high.

### Why

`mycelix-personal` is the missing sovereign user shell for:

- identity vault
- health vault
- credential wallet
- data preferences

This is one of the most leverage-heavy missing frontends in the ecosystem. It
connects directly to Identity, Health, Craft, and broader sovereignty claims.

Without it, users have domain apps but no clear unified personal vault surface.

### Product Role

Standalone-first app.

### Minimum Viable Scope

- vault unlock and onboarding
- credential wallet
- health vault summary handoff into Health
- identity vault summary handoff into Identity
- data-sharing preferences
- Sensorium launch integration

### Reason To Build Now

It turns Mycelix from “many domains” into “my domains, my vault, my data.”

## Phase 2: Civic

### Priority

Very high.

### Why

Civic is repeatedly called out as strategically important but frontend-thin.
The scaffold exists, but the real product workflows still need to be built.

The domain covers:

- justice
- emergency coordination
- media verification or commons

That makes it one of the strongest “why Mycelix matters” frontends.

### Product Role

Standalone-first app with strong Sensorium summary integration.

### Minimum Viable Scope

- justice docket or case queue
- emergency coordination panel
- media verification surface
- trust and legitimacy indicators
- strong semantic token adoption through shared core

### Reason To Build Now

It has high public and narrative value and fills a major ecosystem gap.

## Phase 3: Commons Deepening

### Priority

High.

### Why

Commons already has a shell, but the important domain views are described as
thin. That means the next step is not “create a Commons frontend” but “make the
Commons frontend real.”

Likely high-value slices:

- watershed
- housing
- care
- resource coordination

### Product Role

Strengthen an existing standalone app.

### Minimum Viable Scope

- 2-3 real domain views, not just shell pages
- at least one operational flow per subdomain
- Sensorium summaries for commons state
- restore or create the missing `commons-leptos-types` contract so the standalone
  Commons app and Sensorium can share real view models instead of diverging mocks

### Reason To Build Now

Commons is one of the clearest examples of the federated multi-domain promise.

## Phase 4: Attribution

### Priority

High, but narrower.

### Why

Attribution has strong backend and ecosystem importance, but appears to depend
on observatory integration rather than having a first-class standalone
experience.

That is acceptable for internal dashboards, but weak if attribution and
reciprocity are meant to be user-facing ecosystem concepts.

### Product Role

Likely summary-first, then standalone if warranted.

### Minimum Viable Scope

- Sensorium-facing attribution summary
- dependency and usage receipt visibility
- reciprocity flows surfaced somewhere stable

### Reason To Build Now

Attribution is one of the unique ecosystem-level differentiators. It should not
remain hidden behind an internal observability surface forever.

## Phase 5: Manufacturing

### Priority

Medium-high.

### Why

`mycelix-manufacturing` appears backend-rich and operationally significant, but
has no visible user-facing frontend surface.

This is a good candidate once the more identity- and civics-adjacent frontends
are established.

### Product Role

Standalone-first operational app.

### Minimum Viable Scope

- workorders
- planning
- bill of materials
- machine or operations visibility

### Reason To Build After Earlier Phases

High value, but less foundational to ecosystem identity than Personal, Civic,
and Commons.

## Phase 6: Position

### Priority

Medium.

### Why

`mycelix-position` appears technically rich but without a visible primary
frontend. Its value depends heavily on what adjacent domains need from it.

It may function best as:

- a platform capability surfaced through other apps first
- and only later as its own standalone app

### Product Role

Sensorium summary or embedded capability first, standalone later if justified.

### Minimum Viable Scope

- anchor and ranging status
- position estimate visualization
- handoff into dependent apps if any

### Reason To Delay Slightly

Position feels more like a systems capability than a first user entry point.

## Phase 7: Lawful Identity

### Priority

Medium.

### Why

`mycelix-lawful-identity` looks backend-heavy and compliance-oriented, but there
is no obvious user-facing app surface yet.

This likely becomes important after:

- Personal
- Identity
- Civic

are already strong enough to use it well.

### Product Role

Probably a specialized administrative or verification frontend rather than a
mass-user-first app.

### Minimum Viable Scope

- issuer classification and lookup UI
- verification and trust-tier UI
- integration with Identity and Personal

## Phase 8: Space Clarification

### Priority

Medium, but mostly architectural.

### Why

`mycelix-space` appears to have:

- a SvelteKit dashboard
- a partial Leptos app directory
- a “built” status in documentation

Before building more, the ecosystem needs one answer:

- is Space part of the canonical Mycelix Leptos cluster family?
- or is it intentionally on a separate frontend line?

### Required Outcome

Decide whether Space should:

- stay Svelte-first
- migrate to Leptos
- or expose itself to the Sensorium while staying separate

This is a clarification phase more than a build phase.

## What Not To Prioritize Early

These should not jump ahead of the phases above unless there is a very specific
user or revenue driver.

### Marketplace

`mycelix-marketplace` is explicitly archived in `REPO_REGISTRY.md`. It may have
value, but it should not consume early roadmap attention ahead of the core
ecosystem shells.

### Legacy Parallel Frontends

- Pulse React/Vite
- duplicate or partial alternate frontends

These should be frozen unless needed for migration or asset reuse.

### Purely Internal Dashboards

Observability or internal-only admin surfaces are useful, but they should not
displace missing user-facing foundational apps.

## Recommended Roadmap Sequence

If a single ordered build sequence is needed, the recommended order is:

1. Foundation convergence
   - Pulse convergence
   - Sensorium integration standard
   - shared substrate hardening
2. Personal
3. Civic
4. Commons deepening
5. Attribution productization
6. Manufacturing
7. Position
8. Lawful Identity
9. Space clarification and decision

## Capacity Guidance

If only one major frontend effort can run at a time:

- do Foundation convergence
- then Personal
- then Civic

If two tracks can run in parallel:

- Track A: Foundation convergence -> Pulse -> shared substrate
- Track B: Personal -> Civic

If three tracks can run in parallel:

- Track A: Foundation convergence
- Track B: Personal
- Track C: Commons deepening or Attribution, depending on product pressure

## Success Criteria

The roadmap is working if, after these phases:

- Mycelix has a clear sovereign user shell through Personal
- Pulse is converging rather than drifting
- the Sensorium meaningfully summarizes and launches domains
- Civic becomes a real product surface, not just a scaffold
- Commons is no longer shell-heavy and domain-light
- backend-rich domains without UIs begin getting first real entry points

## Review Notes

This roadmap should be revisited after:

- Personal MVP
- Civic MVP
- the Pulse migration work

At that point, the next-order priority may shift depending on:

- adoption pressure
- demo goals
- infrastructure readiness
- cross-domain dependencies discovered during implementation
