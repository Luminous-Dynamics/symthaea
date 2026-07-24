# Muse Journey Wiring Plan

**Status:** DONE (2026-07-24, same session this plan was written in — the
"future session" framing below is stale, kept only for the historical
scoping rationale in §1). All three steps landed:
`00ec090522` (real `ArtifactIdentity`), `d591f92953` (`journey.rs` wired
into `MuseState`), `24c64e52ca` (`JourneyPolicy` selector UI). Verified via
real `trunk build` + headless-Chromium runs against a live `muse_studio`
backend (not just `cargo check`), twice — once for the reducer wiring,
once for the policy selector — both showing real composed audio/title/
duration in the DOM and zero console errors. The open question in §3
(what the three policies should actually do differently) remains
genuinely open and undecided; everything else here is closed.

**Scope:** Replace `symthaea-muse-ui`'s ad-hoc `next_piece()` prefetch/
staleness logic with the already-written, already-tested `journey.rs`
reducer (`crates/domains/symthaea-muse-ui/src/journey.rs`, 316 lines, 3
passing tests, added earlier this session but never wired to a live
component).

---

## 1. Why this isn't a quick wire-up

`journey.rs`'s `JourneyState`/`JourneyCommand`/`JourneyEffect` reducer
already exists, compiles, and is unit-tested (`stale_prefetch_cannot_enter_
queue_after_policy_change`, `advance_is_coherent_and_requests_replenishment`,
`event_replay_is_deterministic`). But `MuseState::next_piece()`
(`crates/domains/symthaea-muse-ui/src/state.rs:290-327`) already has a
**working** mechanism for the same problem: a `queue: Vec<Candidate>`
prefetch buffer and a `request_generation` counter that marks in-flight
composes stale when superseded by a newer call. It isn't broken — it's
just less principled than `journey.rs`'s `prefetch_epoch`+
`composition_request_id` matching.

Wiring `journey.rs` in for real means four separate pieces of work, not
one:

1. **Replace `MuseState`'s prefetch/staleness fields with `JourneyState`**
   — `queue`/`request_generation` go away, `journey: RwSignal<JourneyState>`
   (or similar) comes in.
2. **Rewrite `next_piece()`/`prefetch()`** to dispatch `JourneyCommand`s
   (`RequestNext`, `Advance`, `ReturnToPrevious`) and interpret the
   returned `JourneyEffect`s — `ComposeNext` drives the existing
   `spawn_local` + `api::compose_listen_piece` call, `CurrentChanged`
   drives `show_piece()`.
3. **Add a policy selector UI** — `JourneyPolicy::{Resonance, Discovery,
   Contrast}` has no surface anywhere today; `next_piece()` currently
   just calls `palette::random_style()` with no policy concept at all.
   This is a product/UX decision (what do the three policies actually
   *do* to piece selection?), not just plumbing — see §3.
4. **Source `JourneyArtifact.identity: ArtifactIdentity`** — checked the
   actual wire type: `symthaea-muse-protocol::Candidate` (the compose
   response DTO consumed by `symthaea-muse-ui/src/api.rs`) has **no**
   `ArtifactIdentity`/`score_content`/`composition`/`rendition` fields at
   all today. There is nothing to source this from without first
   extending `/api/compose`'s response — either add `ArtifactIdentity` to
   `Candidate` server-side (computed the same way `piece_provenance`
   already computes `score_sha256`/`recipe_sha256`/audio hash — that
   logic already exists in `muse_studio.rs`, just not attached to the
   compose response), or synthesize a placeholder client-side (NOT
   recommended: `ArtifactIdentity` is specifically a real content-hash
   identity elsewhere in the system; a synthetic client-side stand-in
   would be a quiet honesty regression).

## 2. Recommended sequence (all four steps DONE — see commit ids in Status)

1. ✅ **Backend first, small and self-contained**: add `ArtifactIdentity` to
   `symthaea-muse-protocol::Candidate` and populate it in `compose()`'s
   response construction (`muse_studio.rs`) using the same
   `serialized_sha256`/`sha256_hex` helpers `piece_provenance` already
   uses. Landed exactly as scoped (`00ec090522`).
2. ✅ **`relation_from_previous` decision**: went with the client-side
   heuristic as recommended (`state.rs::relation_from_previous`, style-
   match comparison) — presentation text, not asserted fact.
3. ✅ **`MuseState` rewrite**: `JourneyState` + a `candidate_cache` swapped
   in for `queue`/`request_generation`/`prefetch()` (`d591f92953`).
   Live-verified via real `trunk build` + headless Chromium against a live
   backend, not just unit tests — the Playwright-style regression bar this
   step called for.
4. ✅ **Policy selector UI**: three buttons wired to
   `JourneyCommand::ChangePolicy` via `MuseState::set_journey_policy()`
   (`24c64e52ca`). All three policies compose identically, disclosed in
   both the code and the CSS comment — §3's question stayed open on
   purpose, not shipped as a guess.

## 3. Open design question (needs a decision before/during step 4)

What should `Resonance` / `Discovery` / `Contrast` actually change about
which piece comes next? None of this is decided yet. Candidate framings,
not a recommendation:
- **Resonance**: bias toward styles/seeds similar to `current`.
- **Discovery**: bias toward `recent_compositions`-avoiding novelty (the
  reducer already threads `recent_compositions` through `ComposeNext` for
  exactly this).
- **Contrast**: deliberately pick a style far from `current`.

All three need either a server-side scoring hook (heavier) or a
client-side style-distance heuristic before the next `compose()` call
picks its `style`/`seed` (lighter, recommended first). Until this is
decided, the policy selector can ship as UI-only with all three policies
behaving identically — that's honest as long as it's disclosed, not silent.

## 4. Non-goals for this pass

- No change to `journey.rs` itself — it's already correct and tested;
  this plan is entirely about wiring it up, not redesigning it.
- No multi-policy scoring backend in the first cut (see §3).
- No genealogy/analyst UI surface — unrelated to this plan (see
  `MUSE_PROVENANCE_FEDERATION_REALITY_AND_BOUNDARIES_2026-07-24.md` for
  that track's own status).

## 5. Verification bar

Done, with one honest gap disclosed: real `trunk build` + headless-
Chromium runs against a live `muse_studio` backend confirmed cold-start
compose/prefetch (real audio/title/duration in the DOM, status line
correctly cleared) and the policy selector's render + default selection,
zero console/page errors both times. `journey.rs`'s own 3 tests plus a 4th
still pass unchanged, run natively.

**Not exercised: `JourneyCommand::ReturnToPrevious`.** The reducer supports
it, but no UI anywhere in this app — before or after this plan — ever
called it; Listen Mode's only "back" affordance is `restart()` (replay the
current piece from 0:00), which predates this plan and is unrelated. So
"return-to-previous" was never a real regression risk here, but it's also
genuinely untested end-to-end, not merely "verified and I forgot to say
so." If a "previous piece" UI action is ever added, it exercises brand new
ground, not a already-covered path.
