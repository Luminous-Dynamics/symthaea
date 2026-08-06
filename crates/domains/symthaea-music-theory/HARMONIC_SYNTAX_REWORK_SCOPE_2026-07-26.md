# Harmonic Syntax Rework — Scope Document (2026-07-26)

> ## ✅ UNBLOCKED 2026-07-30 (`ecae4f2da0`) — `HarmonicPalette` landed
>
> The blocker described below is **cleared**. `ProgressionSpec::GrammarWithPalette(..)`
> supplies a style's own harmonic vocabulary to the same universal T-PD-D-T grammar, so
> converting a style no longer makes it a clone of Classical.
>
> | style | distinct / 32 seeds | character |
> |---|---|---|
> | Classical | 26 | wanders to vi, 1-in-4 deceptive close (**unchanged**) |
> | BaroqueSuite | 16 | ii-heavy sequential chains, decisive close |
> | March | 4 | plain I-IV-V-I, never vi, always authentic |
>
> Pairwise identical-seed counts: **≤1/32** (were 32/32 for any pair).
>
> **A shipped defect surfaced immediately.** The new guard
> `grammar_generated_styles_must_not_be_harmonically_interchangeable` failed on its first
> run: **Classical and BaroqueSuite were 32/32 identical** — introduced by this doc's own
> 2026-07-26 pilot, unnoticed for four days. Fixed with a real baroque palette.
>
> **What the remaining conversions now need.** The doc's original "zero new
> harmonic-generation code, a field change per style" framing is *true again*, with one
> addition: each converted style needs a palette expressing its character, or it trips the
> guard. Writing a palette is a small, musical, reviewable act — five fields — not new
> machinery. `HarmonicPalette::classical()` / `baroque()` / `march()` are worked examples.
>
> Still holds from the blocked-attempt note below: **Impressionism should join
> Lullaby/SacredChoral/ModalFolk in the held list** (its `[1, 4, 1, 2]` avoids V by design),
> and `Waltz` needs its duplicate hardcoded arm removed from `Style::progression()` before
> conversion, since that legacy method would otherwise keep returning the old pachelbel loop.
>
> ---
>
> ## ⚠️ ATTEMPTED 2026-07-30 — BLOCKED, and the block is a prerequisite this doc missed
>
> *(Kept for the record — this is the investigation that produced the fix above.)*
>
> The next batch (March, Nocturne → `ProgressionSpec::Grammar`) was implemented, measured,
> and **reverted**. 541/541 tests stayed green throughout; the blocker is musical and
> architectural, not a test failure.
>
> **Measured, 32 seeds per style:**
>
> | | distinct progressions | vs Classical at matched seed |
> |---|---|---|
> | March / Nocturne *before* | **1** | distinct |
> | March / Nocturne *after* | **26** | **bit-identical, 96/96** |
>
> **The conversion trades one defect for another.** Each style gains real within-style
> variety (1 → 26) and simultaneously becomes harmonically **indistinguishable from every
> other `Grammar` style**, because `Progression::generate(bars, seed)` is entirely
> style-agnostic. That is precisely the cross-style bleed the diversity census flagged for
> Folk/Cinematic/Playful (40-51% nearest-neighbour bleed) and fixed by giving each its own
> degree set — reintroduced here at 100%.
>
> **This doc's pilot design could not have caught it.** With exactly one converted style
> (BaroqueSuite), there is no second Grammar style to be identical *to*. The defect only
> becomes observable at N≥2. The pilot passing was therefore not evidence that the approach
> transfers — a structural limit of one-style pilots worth remembering generally.
>
> **A per-style seed salt was tried and also reverted.** Salting `Progression::generate` with
> the spec's own name fixes the identity problem, but it silently changed **Classical**'s
> output — the reference style the BaroqueSuite pilot was itself A/B'd against — and broke
> `composer::tests::grammar_progression_styles_are_unaffected_by_the_grammar_aware_pipeline`.
> Chasing that surfaced the real reason a salt is not a local change: **there are four
> independent progression sources**, not one.
>
> | # | source | used by |
> |---|---|---|
> | 1 | `CompositionSpec::progression()` | the composer (`composer.rs:505`) — the live path |
> | 2 | `Style::progression()` | `live.rs:114`; hardcodes Classical/Waltz/Folk/Cinematic/Playful, else delegates to (1) |
> | 3 | `form.rs:239,324,332` | B-sections/episodes, `Progression::generate(bars, seed ^ 0x5EC7_104B)` — already salted, independently |
> | 4 | `sonata.rs`, `opera.rs` | dedicated engines, own `seed_variant` |
>
> **Revised prerequisite, replacing this doc's "zero new harmonic-generation code" claim:**
> `Progression::generate` needs genuine per-style character (its own degree palette and/or
> cadence bias per style, the way Folk/Cinematic/Playful already have) **before** any further
> style is converted. Until then, converting a style makes it more varied and less itself.
>
> Also worth flagging: **Impressionism should probably be held with Lullaby/SacredChoral/
> ModalFolk.** Its current `[1, 4, 1, 2]` avoids V entirely — a deliberate non-functional
> choice matching impressionist practice (planing, modal colour, avoided cadences). Forcing
> it onto a T-PD-D-T walk with forced authentic/deceptive cadences would flatten exactly the
> character that makes it a distinct style. This doc's list of three held styles should be
> four.

**Status: scoping only, not implemented.** Deliberately not attempted in the
same pass as this session's other fixes — the muse improvement-roadmap
synthesis flagged this item with the *lowest* engineering-risk score (1/10)
of anything on its list ("the most invasive possible change, touching the
shared progression-generation path for nearly every style") and explicitly
recommended a design pass and staged rollout, not a blind rewrite. This
document is that design pass's first output: what's actually true today,
what a *safe, small* first slice looks like, and what's deliberately
excluded.

## The finding, precisely stated

Of `Style`'s 29 presets, **only `Classical` uses real functional-harmony
generation** (`ProgressionSpec::Grammar` → `harmony::Progression::generate`,
a genuine T→PD→D→T random walk with a forced cadence — see `harmony.rs:329`).
The other 28 use `ProgressionSpec::Archetype`/`ArchetypePool`: a fixed chord
vector, cycled to length. No T-PD-D-T logic, no per-seed harmonic variety —
the SAME chord sequence every time (or one of a small fixed pool, per
seed, for the handful of styles already given an `ArchetypePool`).

**The sharper finding**: this isn't "1 real style vs. 28 stylistically
different ones needing bespoke harmony." `GrammarFamily::PeriodSentence`
(`grammar.rs:408-414`) assigns **`HarmonicSyntax::Functional` to all 12** of
its member styles uniformly — Classical, Waltz, Folk, Cinematic, Playful,
Nocturne, March, Lullaby, ModalFolk, Impressionism, SacredChoral,
BaroqueSuite (`style.rs:311-323`). Eleven of those twelve declare the exact
same harmonic syntax Classical does, yet don't actually generate it —
they're stuck on a fixed loop. That's not "these styles need different
harmony," it's "these styles claim the SAME harmony Classical has, and
don't get it."

## Why this is NOT one uniform 28-style rewrite

The other 16 non-PeriodSentence styles need genuinely different treatment,
not "more Grammar":

- **Blues, JazzBallad**: have their own dedicated grammar engines
  (`call_response.rs`, `jazz_chorus.rs`, both landed this session) with
  their own progression handling already fixed for per-chorus variety
  (`ProgressionSpec::ArchetypePool`, Blues this session, JazzChorus's own
  harmonic-variety fast-follow tracked separately in `jazz_chorus.rs`'s own
  module doc). `Progression::generate`'s classical T-PD-D-T walk is the
  WRONG model for either — jazz needs ii-V-I circle-of-fifths chains and
  secondary dominants; blues needs its own 12-bar cyclical form (already
  built).
- **AfroCuban (GrooveCycle), Minimalism (ProcessAdditive), HindustaniInspired
  (RagaModalArc)**: dedicated engines built on ostinato/vamp/modal-arc logic
  that doesn't route through `spec.progression()` as a harmonic-function
  generator at all — vamp-based or modal, not cadential.
- **Ambient (AmbientTextural), Opera (DramaticAdaptive), ProgFolk/Sonata/
  RenaissancePolyphony (Developmental)**: real, structurally distinct
  engines/bypasses (fugue, sonata, renaissance, prog-suite) with their own
  tonal logic already, or drone/stasis textures where "functional harmony"
  isn't the right lens at all (`HarmonicSyntax::StaticProcessField`/
  `SpectralStasis`).
- **Tango, Celtic, Flamenco, BossaNova, IrishTraditional (StrophicSong)**:
  currently run through the SAME generic ternary/rondo pipeline as
  PeriodSentence (the other, separately-tracked "3-family structural gap" —
  JazzChorus just closed one of three; StrophicSong/AmbientTextural remain
  open). These need a real verse/refrain STRUCTURAL engine before their
  harmony is even the next bottleneck — reharmonizing a style that has no
  chorus-cycling identity of its own yet is solving the wrong problem
  first.

So "give every non-Classical style real harmony" is at minimum 4-5
DIFFERENT engineering problems (functional-walk extension, jazz ii-V-I
generation, modal/verse-refrain harmony, and the two categories above that
arguably don't need this at all), not one. Collapsing them into a single
initiative is exactly how a low-risk pilot turns into the roadmap's
1/10-risk mega-change.

## Recommended first slice: the 11 falsely-Functional PeriodSentence styles

**Claim**: switching Waltz/Folk/Cinematic/Playful/Nocturne/March/Lullaby/
ModalFolk/Impressionism/SacredChoral/BaroqueSuite from `ProgressionSpec::
Archetype(...)` to `ProgressionSpec::Grammar` is a **materially smaller and
safer** change than "harmonic rework" as a category, because:

1. **Zero new harmonic-generation code.** `Progression::generate` already
   exists, is already tested, and is already live for Classical. This slice
   is a `CompositionSpec` FIELD CHANGE per style, not new logic.
2. **No mismatch with declared intent.** These 11 styles already claim
   `HarmonicSyntax::Functional` — this closes the gap between what they
   claim and what they do, it doesn't change what they claim.
3. **The cadence half is already grammar-aware.** Phase 4 of this session's
   work (`Period::parallel_in_for_grammar`) already makes closing cadences
   respect `HarmonicSyntax` per-family; this slice is specifically about the
   MID-PHRASE chord choices Phase 4 didn't touch (that phase deliberately
   scoped to cadences only, leaving "the largest remaining piece of the
   original diversity critique" for later — this is that later).

**This is still a real, non-trivial change**, not a trivial toggle:

- Every one of these 11 styles almost certainly has a test asserting its
  exact `ProgressionSpec::Archetype(vec![...])` value (see this session's
  own precedent fixing `blues_really_is_the_twelve_bar_i_iv_v_turnaround`
  and `jazz_ballad_is_aeolian_...` after their own progression-field
  changes) — each needs its assertion updated to check the new variant
  instead, the same pattern already used twice this session.
- `Progression::generate`'s randomized T-PD-D-T walk may not suit every
  one of these 11 styles' character equally well. Lullaby in particular
  wants harmonic predictability/simplicity a randomized walk doesn't
  obviously provide — this needs REAL LISTENING verification (per this
  crate's own standing rule: taste-dependent judgments aren't something to
  guess at from code alone), not just "it compiles and tests pass."
- Downstream tests that assert specific composed pitches/behavior tied to
  the OLD fixed progression (there are precedents for this kind of
  incidental coupling in this crate) will need auditing per style, not just
  the direct progression-equality assertions.

**Recommended execution shape** (do not attempt in one pass):
1. Pilot ONE style first — **Cinematic** or **March** are good first
   candidates (no existing modal/liturgical/lullaby-specific harmonic
   character that a randomized walk risks flattening; check `spec.rs`'s
   `Attitude`/`MelodicDna` fields per style before picking, some of these
   11 may have narrower stylistic constraints than they look).
2. Full listening verification of that one style (multiple seeds) before
   touching any other style in the list — the whole point of doing ONE
   first is to learn whether `Progression::generate` needs its own
   per-style tuning (e.g. a style-specific bias on the T→{PD,vi} branch,
   or a different final-cadence deceptive-resolution rate) before
   assuming it transfers unchanged to the other 10.
3. Only after the pilot's listening result is judged genuinely better (or
   at least not worse) should the remaining 10 be attempted, and even then
   one small batch at a time with the existing full-suite green-bar
   discipline this session used throughout (498→508 tests, never regress).
4. Explicitly hold Lullaby, SacredChoral, and ModalFolk for LAST (or for a
   separate follow-up decision) — these three have the most reason to want
   something other than Classical's exact randomized walk (a lullaby's
   harmonic predictability, a sacred/choral style's own modal-adjacent
   conventions per its Phrygian/plagal identity noted elsewhere in this
   crate, and ModalFolk's own mode-specific cadence machinery already
   built) — forcing them onto the unmodified Classical generator without
   checking first risks flattening exactly the character that makes them
   distinct styles in the first place.

## Explicitly out of scope for any near-term follow-up

- StrophicSong's 5 styles and AmbientTextural: structural-identity gap
  (their own dedicated grammar engine) is the correct next investment for
  THOSE families, not harmonic-syntax work — reharmonizing a style with no
  chorus-cycling identity solves the wrong problem first (same reasoning
  `jazz_chorus.rs`'s own module doc gives for treating structural identity
  and harmonic identity as different problems).
- A bespoke jazz ii-V-I / secondary-dominant generator for JazzChorus
  (JazzBallad currently reuses ONE fixed 8-bar turnaround across every
  chorus, same disclosed limitation `jazz_chorus.rs` already states) — a
  real, separate design effort, not a byproduct of this slice.
- Any change to Blues, the GrooveCycle/ProcessAdditive/RagaModalArc
  dedicated engines, or the fugue/sonata/renaissance/prog-suite bypass
  forms — none of these route mid-phrase harmony through
  `Progression::generate` in a way this slice would touch, and forcing
  them to would be a regression, not an improvement.

## Bottom line

The "1/10 engineering-risk, largest remaining piece of the diversity
critique" framing describes the full 28-style ambition. The ACTUAL
low-risk, well-justified first step is much smaller: 11 styles that already
claim functional harmony and don't have it, fixed by reusing existing,
tested code, one style at a time, with real listening verification gating
each step — not a mechanical field-flip across all 11 at once. That pilot
is deliberately not started here; it's the next session's or next
explicitly-requested unit of work.
