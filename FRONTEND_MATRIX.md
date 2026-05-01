# Mycelix Frontend Matrix

## Purpose

This matrix provides an operational inventory of major Mycelix frontend
surfaces.

It is intended to support:

- build-order decisions
- convergence decisions
- Sensorium integration planning
- identification of missing or thin frontends

This document complements:

- `FRONTEND_ARCHITECTURE.md`
- `FRONTEND_ROADMAP.md`

## Status Legend

- `Canonical` — active, strategically valid frontend surface aligned with the
  current architecture direction
- `Near-Canonical` — active and valid, but should converge further toward the
  standard cluster pattern
- `Justified Deviation` — active and intentionally different because of product
  needs
- `Scaffold` — frontend exists, but is not yet a real product surface
- `Thin` — frontend exists, but major domain views or workflows are still
  shallow
- `Summary-Only` — domain is surfaced primarily through another shell or
  dashboard, not a first-class app
- `Missing` — no clear primary user-facing frontend surface
- `Legacy` — parallel or superseded frontend line that should not receive
  first-class product investment
- `Clarify` — surface exists, but the strategic direction is not yet clear

## Matrix

| Domain / Surface | Frontend Surface | Category | Status | Current Read | Recommended Next Action |
|---|---|---|---|---|---|
| Sensorium | `mycelix-sensorium` | Sensorium shell | Canonical | Real ecosystem shell with domain module system, publicly named Mycelix Sensorium | Keep; strengthen summary and launch contracts |
| Pulse | `mycelix-pulse/apps/leptos` | Standalone domain app | Near-Canonical | Canonical comms app, but still more custom than newer cluster apps | Converge runtime/theme/toasts where safe |
| Pulse legacy web | `mycelix-pulse/ui/frontend` | Parallel legacy app | Legacy | Old React/Vite product surface | Freeze; do not treat as primary frontend |
| Pulse desktop | `mycelix-pulse/desktop` | Wrapper | Canonical | Canonical desktop shell around Pulse | Keep |
| Pulse mobile | `mycelix-pulse/mobile` | Wrapper / adjunct | Clarify | Exists, but should remain aligned to canonical Pulse app | Keep only if it follows Pulse canonical frontend |
| Praxis | `mycelix-praxis/apps/leptos` | Standalone domain app | Near-Canonical | Strong app, partly converged to shared substrate | Continue convergence incrementally |
| Craft | `mycelix-craft/apps/leptos` | Standalone domain app | Near-Canonical | Real app using shared substrate, but not yet in the cleanest standard shape | Normalize shell/runtime where useful |
| Identity | `mycelix-identity/apps/leptos` | Standalone domain app | Near-Canonical | Real app, important ecosystem entry point | Keep; lightly standardize |
| Health | `mycelix-health/apps/leptos` | Standalone domain app | Justified Deviation | Strong domain-authored shell and product logic | Keep; preserve uniqueness while using shared substrate |
| Hearth | `mycelix-hearth/apps/leptos` | Standalone domain app | Justified Deviation | Warm, ambient, product-specific shell | Keep; align APIs, not shell identity |
| Governance | `mycelix-governance/apps/leptos` | Standalone domain app | Near-Canonical | Active app, but not yet the cleanest shared-shell user | Normalize gradually |
| Finance | `mycelix-finance/apps/leptos` | Standalone domain app | Canonical | Strong shared-core adoption | Keep as reference |
| Climate | `mycelix-climate/apps/leptos` | Standalone domain app | Canonical | Strong shared-core adoption | Keep as reference |
| Energy | `mycelix-energy/apps/leptos` | Standalone domain app | Canonical | Strong shared-core adoption | Keep as reference |
| Knowledge | `mycelix-knowledge/apps/leptos` | Standalone domain app | Canonical | Strong shared-core adoption | Keep as reference |
| Supply Chain | `mycelix-supplychain/apps/leptos` | Standalone domain app | Canonical | Strong shared-core adoption | Keep as reference |
| Commons | `mycelix-commons/apps/leptos` | Standalone domain app | Thin | Shell exists, but docs suggest domain pages remain thin | Deepen core views before building new adjacent apps |
| Civic | `mycelix-civic/apps/leptos` | Standalone domain app | Scaffold | Greenfield scaffold, not yet real justice/emergency/media product | Build real workflows next |
| Music | `mycelix-music/apps/leptos` | Standalone domain app | Clarify | Real app, but appears architecturally special and mock-leaning in current shell | Decide real-vs-demo path, then converge selectively |
| DeSci | `mycelix-desci/apps/leptos` | Standalone domain app | Clarify | Frontend exists, but not yet clearly aligned with the shared cluster pattern | Review separately before major investment |
| Personal | no clear standalone app | Standalone domain app | Missing | Backend/domain importance is high, but no obvious user-facing app | Build high-priority frontend |
| Attribution | observatory/dashboard integration | Domain summary surface | Summary-Only | Important ecosystem capability, but not a first-class app | Productize summary surface, then evaluate standalone app |
| Manufacturing | no clear primary frontend | Standalone domain app | Missing | Backend-rich, UI-poor | Plan MVP frontend after Personal/Civic/Commons |
| Position | no clear primary frontend | Capability or domain app | Missing | Rich systems layer, no obvious user app | Decide whether embedded-first or standalone-first |
| Lawful Identity | no clear primary frontend | Specialized domain / admin app | Missing | Backend/CLI oriented | Build later as specialized verification/admin surface |
| Space | Svelte dashboard + partial Leptos area | Standalone domain app | Clarify | Surface exists, but canonical frontend direction is unclear | Decide whether Svelte-first, Leptos migration, or Sensorium-summary-only |
| Marketplace | `mycelix-marketplace/frontend` | Separate product line | Clarify / Legacy-priority | Built SvelteKit line, but archived in registry and not core to current roadmap | Do not prioritize ahead of core ecosystem gaps |
| Commerce | `mycelix-commerce/ui` | Separate product line | Clarify | UI artifacts exist, but strategic role in current ecosystem is unclear | Defer until core priorities are complete |
| Personal vault concepts | embedded across domains today | Cross-domain concern | Missing as unified surface | Important functionality spread across apps | Consolidate into Personal frontend |
| Sovereign admin / xenia admin | `mycelix-sovereign` adjacencies | Specialized admin surface | Clarify | Documentation references frontend gaps around identity wiring | Review once core user-facing surfaces stabilize |

## Highest-Priority Gaps

These are the most important missing or underbuilt frontends from an ecosystem
point of view.

### 1. Personal

Why it matters:

- likely becomes the sovereign user shell
- connects identity, health, credentials, and data preferences
- turns the ecosystem into “my domains, my data, my vault”

Classification:

- `Missing`

Recommended action:

- build next after foundation convergence

### 2. Civic

Why it matters:

- high public-facing narrative value
- justice, emergency coordination, and media verification are core ecosystem
  differentiators

Classification:

- `Scaffold`

Recommended action:

- turn scaffold into real product surface

### 3. Commons Deepening

Why it matters:

- the shell exists, but domain views remain thin
- this is a multiplier on already-started frontend work

Classification:

- `Thin`

Recommended action:

- deepen instead of starting a brand-new adjacent app first

### 4. Attribution Productization

Why it matters:

- ecosystem differentiator
- currently not represented as a clear first-class user-facing surface

Classification:

- `Summary-Only`

Recommended action:

- create a stable summary/product surface, then reassess standalone need

## Convergence Targets

These are not missing apps. They are active apps that should be converged
architecturally.

### Pulse

- highest-value convergence target
- active and important
- still carries extra local infrastructure

### Praxis

- already moving toward shared substrate usage
- should continue incrementally

### Governance

- active and valuable
- should continue moving toward clearer standardization

### Identity

- important ecosystem hinge
- should stay close to the canonical cluster pattern

## Defer / Clarify Surfaces

These should not absorb near-term roadmap attention until their strategic role
is clearer.

### Space

- UI exists in multiple forms
- canonical direction unclear

### Marketplace

- built, but not core to the current ecosystem build order

### Commerce

- UI artifacts exist
- strategic role in current frontend family is unclear

### Music

- real app, but product and architecture direction needs a more focused pass

### DeSci

- frontend exists, but not yet cleanly classified in the current architecture

## Suggested Working Order

A practical order derived from this matrix is:

1. Foundation convergence
2. Personal
3. Civic
4. Commons deepening
5. Attribution productization
6. Manufacturing
7. Position
8. Lawful Identity
9. Space clarification

## How To Use This Matrix

When deciding whether to start frontend work in a domain, check:

1. Is the surface missing, thin, or only summary-level?
2. Does it unlock multiple domains or complete a major user path?
3. Is there already a good enough frontend that only needs deepening?
4. Would building it now fragment the frontend substrate further?

If a domain is:

- `Missing` and high-leverage -> move it up the roadmap
- `Thin` but strategically important -> deepen it before starting distant new
  domains
- `Clarify` -> decide architecture direction before building more UI
- `Legacy` -> freeze, do not expand

## Status

This matrix is a planning tool and should be updated as:

- new frontends are created
- scaffolds turn into product surfaces
- summary-only domains become standalone apps
- legacy or duplicate surfaces are retired
