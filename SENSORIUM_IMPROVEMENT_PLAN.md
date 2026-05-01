# Mycelix Sensorium Improvement Plan

## Purpose

This document turns the current state of `mycelix-sensorium` into an actionable
product and implementation roadmap.

It is not a speculative rebrand memo. It is a build plan for making Sensorium a
truthful, high-signal ecosystem shell that can orient a user across Mycelix
without trying to replace the standalone apps.

## Working Definition

Sensorium should be the living outer shell of Mycelix:

- the place where a user first sees current posture
- the place where cross-domain state becomes legible
- the place where the next right action is obvious
- the place where deep work launches into the correct domain app

Sensorium should not be a second dashboard copy of every domain.

## Current State

The current shell already has real strengths:

- distinctive product identity and visual language
- a strong “ecosystem shell” framing rather than a generic dashboard
- typed summary contracts in `sensorium-domain-trait`
- live-backed summaries for `personal`, `governance`, and `commons`
- launch affordances into standalone domain apps
- multiple view phenotypes that can eventually support different cognitive modes

The current shell also has clear gaps:

- much of the top-level activity and orbital vitality is still authored or static
- only a minority of major domains provide live summary cards
- some top-level shell indicators are expressive but not yet grounded in real state
- phenotype switching exists before the shell has fully trustworthy summary data
- there is no clear distinction between “truth-bearing posture” and “atmospheric visualization”

The practical result is that Sensorium is already compelling, but not yet fully
reliable as the canonical ecosystem shell.

## Product Rule

Sensorium should obey four rules:

1. Truth before theater.
2. Summarize before simulating.
3. Launch deep work instead of cloning it.
4. Expose cross-domain posture that domain apps cannot show on their own.

Whenever a design choice conflicts with these rules, the shell should favor
truthful orientation over spectacle.

## What Sensorium Should Become

The best version of Sensorium is a three-layer experience:

### 1. Posture Layer

This is the first thing the user should trust.

It should answer:

- Is my conductor live?
- Is my vault unlocked?
- Which domains are real, degraded, empty, or mock?
- What needs attention now?
- What changed recently?

### 2. Coordination Layer

This is where cross-domain meaning emerges.

It should answer:

- What state in one domain affects another?
- What approvals, obligations, or risks are propagating across domains?
- Where are there bottlenecks, unread items, or pending actions?

### 3. Phenotype Layer

This is where the shell becomes experiential and personalized.

It should answer:

- Which view helps this user think best right now?
- Which representation helps them navigate the current task?

The current shell already has phenotype work. The next step is to make the
posture and coordination layers more real than the phenotype layer is stylized.

## Current Truthfulness Gaps

These are the highest-value gaps to close.

### 1. Top-Level Activity Is Not Yet Live

The root `events` stream in
`mycelix-sensorium/crates/sensorium-shell/src/app.rs` is still static and
contains illustrative content.

That means the shell’s most visible synthesis layer can imply ecosystem motion
that is not actually coming from the user’s current state.

Recommendation:

- replace the static stream with a normalized live “recent activity” feed
- aggregate from summary adapters first, not from bespoke per-domain UI logic
- fall back explicitly to `Mock` wording when no live synthesis exists

Priority: `P0`

### 2. Orbital Activity Values Are Authored, Not Computed

Domain node glow and activity percentages are still hardcoded in the orbital
model.

That makes the orb beautiful, but not trustworthy as a posture surface.

Recommendation:

- derive orbital vitality from live summary state
- compute from availability, attention, recent activity, and freshness
- make “quiet but healthy” visually distinct from “inactive because unbuilt”

Priority: `P0`

### 3. Shell Badges Need Real Semantics

The shell currently shows items like `0 leaks`, `Live`, and vitality signals,
but not all of them are yet backed by domain-level evidence.

Recommendation:

- define one typed shell posture model
- drive sovereignty, conductor, freshness, and attention badges from that model
- never show a definitive badge unless its data source is explicit

Priority: `P0`

### 4. Summary Coverage Is Incomplete

Live summaries currently exist for:

- `personal`
- `governance`
- `commons`

That is good progress, but Sensorium cannot function as a canonical shell until
the next strategic domains are also live-backed.

Priority: `P0`

## Domain Summary Maturity

This is the current recommended maturity framing for Sensorium integration.

| Domain | Shell Role | Current Maturity | Next Requirement | Priority |
| --- | --- | --- | --- | --- |
| Personal | identity, vault, consent, posture anchor | Live-backed | add freshness and stronger risk semantics | P0 |
| Health | sensitive posture and consent-adjacent signals | Partial / fallback | live summary adapter | P0 |
| Finance | metabolic state and obligations | Partial / fallback | live summary adapter | P0 |
| Knowledge | claims, learning, epistemic posture | Partial / fallback | live summary adapter | P1 |
| Commons | coordination and shared stewardship | Live-backed | deepen metrics and action routing | P1 |
| Governance | proposals, councils, thresholds | Live-backed | expose timelocks, votes, urgent actions | P1 |
| Pulse / Mail | communications and signal triage | Fallback / separate app | summary adapter with unread, priority, queue state | P1 |
| Hearth | kinship and household coordination | Fallback | define summary contract and launch paths | P2 |
| Praxis | growth and education | Fallback | define learner posture summary | P2 |
| Lucid | inference and cognitive state | Fallback | define safe shell-level summary scope | P2 |
| Admin / System | operational runtime health | Minimal | convert into explicit system posture card | P2 |
| Space and other frontier domains | discovery only | Early | metadata first, summaries later | P3 |

## Recommended Build Order

The build order should follow ecosystem leverage, not just what is easiest to
render.

### Phase 1. Make Sensorium Trustworthy

Target outcome:

- the shell becomes safe to trust as the first screen

Build:

- live shell posture model
- live activity synthesis
- live orbital vitality computation
- live summary adapters for `health` and `finance`
- explicit `Mock`, `Empty`, `Locked`, and `Degraded` rendering everywhere

Reason:

Personal, Health, and Finance define the user’s immediate sense of sovereignty,
safety, and material reality. If those are live, the shell stops feeling
illustrative and starts feeling real.

### Phase 2. Make Sensorium Coordinating

Target outcome:

- the shell shows cross-domain obligations and civic posture

Build:

- deepen `commons` and `governance`
- add `knowledge` summary adapter
- add normalized cross-domain attention queue
- add freshness timestamps and stale-data handling

Reason:

This is the phase where Sensorium becomes uniquely valuable rather than merely
convenient.

### Phase 3. Make Sensorium Operationally Useful

Target outcome:

- communications, approvals, and pending work become visible from one shell

Build:

- Pulse summary adapter
- admin/runtime summary card
- global launch index and quick actions
- recent activity filtering by urgency and domain

Reason:

This is where Sensorium becomes a daily coordination surface rather than just an
entry point.

### Phase 4. Make Sensorium Adaptive

Target outcome:

- phenotype modes become task-aware rather than purely aesthetic

Build:

- “best view for current posture” recommendation
- role-based default phenotype
- task-mode switching based on urgency or workload

Reason:

Phenotypes become much more powerful once the underlying posture model is live.

## View Strategy

The current four phenotype modes are directionally good, but they should be
reframed as task views rather than identity theater.

### Orb

Best for:

- ecosystem orientation
- visibility of domain relationships
- “what areas of my life/network are active?”

Improve by:

- deriving node energy from live summary signals
- rendering dependency and degradation links
- showing stale vs fresh data visibly

### Stream

Best for:

- recent changes
- cross-domain causality
- triage and catch-up

Improve by:

- replacing static authored events with live normalized activity
- grouping by urgency, freshness, and affected domain

### Garden

Best for:

- long-horizon cultivation
- growth posture
- areas needing tending rather than urgent response

Improve by:

- mapping plot growth to longitudinal signals rather than arbitrary activity
- using readiness, progress, or completion arcs per domain

### Pulse

Best for:

- quick status checks
- low-friction mobile or lightweight usage
- “am I okay?” posture scans

Improve by:

- showing only live critical indicators
- becoming the most truthful minimal shell, not just the most minimal

## Architecture Improvements

These are the next architectural moves that will improve the shell without
turning it into domain sprawl.

### 1. Add a Shell-Level Posture Aggregator

Create one internal model that consumes all available domain summaries and
produces:

- freshness
- urgency rollups
- cross-domain counts
- top actions
- shell badges
- orbital vitality scores

This keeps `app.rs` from encoding product semantics directly.

### 2. Separate Summary Truth From View Decoration

Keep the summary contract narrow and truthful.

Then let each phenotype transform the same underlying state differently:

- Orb transforms into topology
- Stream transforms into chronology
- Garden transforms into cultivation state
- Pulse transforms into minimal vital signs

This avoids reimplementing business logic per view.

### 3. Add Freshness Everywhere

Every live summary should expose:

- `updated_at`
- stale threshold behavior
- degraded fallback behavior

Sensorium should never silently render old posture as if it were current.

### 4. Normalize Attention Semantics

Every summary should use the same semantic ladder:

- `Quiet`
- `Notice`
- `ActionNeeded`
- `Urgent`

The shell should then sort and present attention consistently across domains.

### 5. Distinguish Three Kinds of “Unavailable”

Right now missing state can blur together.

Sensorium should distinguish:

- not installed
- installed but disconnected
- installed and live but empty

Those are very different product states.

## Concrete Near-Term Implementation Plan

These are the next implementation steps I would actually schedule.

### Now

- build `health` live summary adapter
- build `finance` live summary adapter
- replace static shell event stream with normalized live recent-activity feed
- compute orbital activity from summary state instead of hardcoded percentages
- add a shell posture aggregator module

### Next

- add `knowledge` live summary adapter
- deepen `commons` and `governance` metrics and attention items
- add summary freshness rendering and stale-state styling
- add system/admin runtime summary card

### Later

- add Pulse summary integration
- make phenotype defaults adaptive
- add role-specific startup views
- add quick actions and launch palettes

## UX Direction

The best UX move is not “more dashboard.”

The best UX move is:

- fewer decorative signals that are not yet true
- stronger contrast between live and fallback state
- more explicit next actions
- faster launch into real workflows
- one obvious answer to “what should I do now?”

Sensorium should feel more like a living instrument panel and less like an
ambient concept demo.

## Success Criteria

Sensorium is materially improved when all of the following are true:

- the first screen is mostly live-backed for strategic domains
- the user can tell what is real, mock, stale, locked, or degraded
- the shell’s main visual intensity reflects actual posture
- the top activity stream is generated from current state
- the next action is clearer than the UI concept
- standalone apps still own deep work

## Recommendation

The highest-leverage next sequence is:

1. make shell-level posture and activity synthesis live
2. add live summary adapters for `health` and `finance`
3. compute Orb, Stream, Garden, and Pulse views from the same truthful summary base
4. then expand to `knowledge`, Pulse, and runtime/system posture

That sequence improves both product integrity and implementation quality without
throwing away the shell’s current visual ambition.
