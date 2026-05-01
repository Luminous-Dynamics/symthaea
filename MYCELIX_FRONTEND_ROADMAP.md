# Mycelix Frontend Roadmap

## Purpose

This document defines the recommended build order for the Mycelix frontend
ecosystem after the initial Sensorium convergence pass.

It is intended to answer three questions:

- which frontends are canonical
- which frontends should be normalized next
- which frontends should be deferred until the shared substrate is stronger

## Current Position

The frontend estate is large, but it is not equally mature.

Today there are three important truths:

1. `mycelix-sensorium` is the canonical ecosystem shell.
2. `mycelix-portal` is effectively the legacy package lineage of the same shell
   and should be treated as a migration and compatibility concern, not a
   parallel product.
3. `mycelix-leptos-core` is now the right place for shared posture primitives:
   summary cards, freshness badges, availability states, and activity feeds.

The recent convergence pass has already moved these shells onto a shared
language:

- Sensorium
- Personal
- Commons
- Governance

That is the beginning of a coherent frontend platform, not the end of it.

## Canonical Shells

These are the frontends that define the user-facing shape of Mycelix and should
be prioritized over more speculative surfaces.

| Shell | Role | Status | Recommendation |
| --- | --- | --- | --- |
| Sensorium | ecosystem shell and cross-domain posture | active | canonical outer shell |
| Personal | identity, vault, preferences, credentials | active | canonical sovereignty shell |
| Health | health vault and consent-sensitive posture | active | canonical safety shell |
| Finance | balances, obligations, exchange, yield | active | canonical metabolic shell |
| Knowledge | claims, verification, graph posture | active | canonical epistemic shell |
| Commons | stewardship and mutual aid coordination | active | canonical communal shell |
| Governance | councils, proposals, voting, thresholds | active | canonical civic shell |
| Pulse | communication and triage | active but separate workspace lineage | canonical communications shell |

Everything else should be judged by whether it strengthens one of these shells
or whether it is still an exploratory domain.

## Shared Substrate Status

The current substrate adoption status is:

| Frontend | Shared summary/status/activity substrate |
| --- | --- |
| Sensorium | yes |
| Personal | yes |
| Commons | yes |
| Governance | yes |
| Health | no |
| Finance | no |
| Knowledge | no |
| Energy | no |
| Pulse | separate lineage |

This matters because fragmented shells create three costs:

- posture semantics drift
- stale or mock state is rendered inconsistently
- every domain reinvents its own shell language

The next phase should reduce those costs before building more unique chrome.

## Recommended Build Order

### Phase 0. Hold The Line

Target:

- stop creating parallel shell concepts
- treat Sensorium as the canonical name
- keep `mycelix-portal` on a compatibility path only

Concrete rule:

- no net-new feature should land in `mycelix-portal` unless it is also part of
  the Sensorium cutover or is needed to keep the old package buildable

### Phase 1. Complete Canonical Posture Coverage

Target:

- make the seven core shells speak the same posture language

Build next:

1. Health
2. Finance
3. Knowledge

Reason:

- these complete the most important sovereign posture loop after Personal
- they are already first-class domains in Sensorium summaries
- they have enough existing UI to benefit immediately from substrate reuse

Definition of done for each shell:

- uses `AvailabilityState` for empty, locked, degraded, mock, and unavailable
  states
- uses `FreshnessBadge` for its top-level posture card or summary band
- uses `ActivityFeed` for recent domain events where applicable
- removes one-off status footer language where a shared primitive is better
- passes `cargo check` inside the repo-root Nix shell

### Phase 2. Make Sensorium A Truthful Router

Target:

- Sensorium reflects live state from the canonical shells instead of carrying
  authored atmosphere as if it were truth

Build:

1. live `health` summary adapter
2. live `finance` summary adapter
3. live `knowledge` summary adapter
4. normalized cross-domain attention queue
5. computed shell vitality and recent activity

Reason:

- once the domain shells and Sensorium share the same posture vocabulary, the
  shell can become a reliable routing layer rather than a decorative dashboard

### Phase 3. Bring Pulse Into The Same View Of Reality

Target:

- communications become a first-class posture signal instead of a separate
  unread island

Build:

1. define a Pulse summary contract with unread count, urgency, and queue health
2. surface Pulse state inside Sensorium as a real card
3. align Pulse shell states with the same availability and freshness semantics

Reason:

- communications drive action across every other shell
- unread triage is one of the highest-leverage cross-domain signals

### Phase 4. Expand To Adjacent Operational Domains

Target:

- normalize the next ring of likely near-term domains without overcommitting

Suggested order:

1. Energy
2. Hearth
3. Praxis
4. Civic
5. Identity

These should only move forward after Phase 1 and Phase 2 are materially done.

### Phase 5. Frontier Domains And Experiments

These domains should be treated as exploratory unless a concrete operator need
pulls them forward:

- Climate
- Craft
- Desci
- Music
- Supplychain
- Space
- Marketplace
- Manufacturing

The right default here is metadata, launch surfaces, and lightweight discovery
before full shell investment.

## Product Rules

Frontend work across Mycelix should follow these rules:

1. Truth before atmosphere.
2. Shared posture before bespoke status widgets.
3. Sensorium summarizes and routes; domain shells hold deep workflows.
4. No duplicate ecosystem shell products.
5. New domains should start from the shared Leptos substrate, not invent a new
   shell grammar.

## Immediate Next Work

The highest-value next implementation sequence is:

1. Apply the shared substrate to `mycelix-health/apps/leptos`.
2. Apply the shared substrate to `mycelix-finance/apps/leptos`.
3. Apply the shared substrate to `mycelix-knowledge/apps/leptos`.
4. Wire the matching live summary adapters into Sensorium.
5. Define the Pulse summary contract and route it into Sensorium.

That sequence keeps the work compounding:

- each domain shell becomes more coherent on its own
- Sensorium gains more truthful input
- the platform becomes easier to extend for later domains

## Operational Guidance

All verification should continue to run inside the repo-root Nix environment.

Recommended habit:

- use `nix develop . -c ...` from `/srv/luminous-dynamics`
- prefer background `nohup` runs only for long-lived checks or builds
- keep `cargo check` green for each frontend before widening the scope
- use `bacon` for continuous feedback while normalizing the next shells

## Decision

If the goal is to make Mycelix feel like one living system instead of a set of
loosely related apps, the correct next move is not to build more frontends in
parallel.

The correct next move is to finish convergence across the canonical shells,
starting with Health, Finance, and Knowledge, and then make Sensorium consume
that reality cleanly.
