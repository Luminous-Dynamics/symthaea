# Muse — Diversity Truth & Migration Plan (2026-07-18)

**Trigger**: an external listening critique ("Muse has many style presets but fewer
genuinely independent compositional grammars — the same composer wearing different
clothes") plus Atlas-export evidence (exact duplicates, lens-invariant neighbors,
keepers missing from the taste map). This plan is the result of a 5-track code
review (grammar architecture, Atlas/fingerprint, diversity gating, renderer,
Leptos migration) that verified every claim against the actual code, file:line.

Companion to `IMPROVEMENT_PLAN.md` (the historical journal — deliberately not
extended further; this is the active plan). Branch context: written on
`review/symthaea-bridges-security`; see Phase 0 for the branch-divergence problem.

---

## Part I — Verdict: the critique, claim by claim

| # | Claim | Verdict | Evidence |
|---|-------|---------|----------|
| 1 | Styles are presets over one shared machinery | **CONFIRMED for 23/29 styles** (nuance below) | `composer.rs:277-662` single pipeline; `spec.rs:156-158,190-192` and `premise.rs:5-12` *admit it in their own comments* |
| 2 | Weak melodic identity/development sharing | **CONFIRMED with nuance** — `DevelopmentDna` (6 programs) and `PhraseRhetoric` (4) exist, but 11/29 styles use no-op development, 22/29 default rhetoric; 12/29 use the identical default hook pools | `spec.rs:194-224`, `hook.rs:135,254-265` |
| 3 | Shared harmonic motion | **CONFIRMED** — 28/29 styles use `ProgressionSpec::Archetype` (fixed degree loop); only Classical uses the functional grammar generator; cadence steering *overwrites* every style's phrase-end harmony into the same Half→Authentic rhetoric | `phrase.rs:148-171`, `harmony.rs:329-360` |
| 4 | Repetitive phrase/form proportions | **CONFIRMED** — every phrase is exactly `intent.bars` bars, period = 2 phrases, section = 1 period, 3 form templates, fixed 4-value intensity table; period-vs-sentence choice is arousal-driven, not style-driven | `phrase.rs:50-112`, `form.rs:131-344`, `composer.rs:424` |
| 5 | No hard diversity gate | **CONFIRMED** — a real batch diversity *selector* exists (Identity Explorer max-min seed picking + premise layer, on by default) but no *gate*: no floor, no rejection, no dedup, no cross-batch/keeper/history comparison; novelty is display-only by explicit policy | `explorer.rs:186-301`, `muse_studio.rs:1292-1582` |
| 6 | Atlas fingerprint too coarse | **PARTIAL** — it's 40-dim in 5 layers, not literally 5 numbers; but no tempo, no meter, no pitch-class/interval content, no instrument identity, whole-piece aggregates below the form layer, and layers are not scale-normalized against each other | `fingerprint.rs:37-198,205-211` |
| 7 | Lenses don't change neighbors | **CONFIRMED** — NN computed once on raw unweighted fingerprints *by documented design*; the `weighted()` primitive for per-lens distance exists and is unused for NN | `muse_studio.rs:3336-3358`, `fingerprint.rs:226-241` |
| 8 | Exact duplicates reach the UI | **CONFIRMED, mechanism found** — the *legacy* Studio's `composeToday()` fires on every page load with a date-deterministic (seed, style); the whole pipeline is deterministic, so every reload that day re-inserts identical candidates under fresh IDs. 26-entry style cycle also collides day D with day D+26 (38/64-seed window overlap). The Leptos UI uses random seeds and doesn't have this. No dedup anywhere catches it | `studio/muse-studio.js:452-477`, `explorer.rs:39,224` |
| 9 | Keepers mostly absent from Atlas | **CONFIRMED** — 25 of 28 keeper records are silently dropped by the recipe filter (older schema generations lack a reconstructible recipe); survivors are *recomposed* rather than indexed from stored artifacts. One dropped keeper has real MIDI on disk today | `muse_studio.rs:3247-3276`, `data/taste/keepers.jsonl` |
| 10 | Renderer flattens differences | **CONFIRMED — but via different mechanisms than guessed** (below). Velocity/timing-quantization sub-claims are FALSE (continuous velocities, ~1ms tick resolution) | see Part II |

**Overstated parts, for honesty:** six styles (Fugue, Passacaglia, RenaissancePolyphony,
Sonata, ProgSuite, Opera) route to genuinely separate structural engines
(`composer.rs:326-416`); three of them bypass phrase construction, harmonic pacing,
accompaniment, and cadence machinery entirely. The engine also already has more
dormant diversity machinery than the critique assumes — the problem is wiring and
defaults, not a missing concept.

## Part II — The renderer finding (biggest surprise of the review)

The live server (checked via the running process environment) renders through
**FluidSynth + one FluidR3 GM soundfont** — the "common soundfont" premise is
literally true for every style. And the symbolic→MIDI boundary *discards* most of
the designed differentiation:

1. **No CC7/CC10**: `MidiEventType` (`midi_export.rs:31-37`) has no ControlChange
   variant, so the carefully computed per-voice volumes (melody 1.0 / harmony 0.5 /
   bass 0.85…) and stereo stage (`pan_for_role`) are computed then thrown away.
   Every piece renders at uniform GM balance, near-mono.
2. **No percussion at all** on the live path: style-gated drums (`DrumPolicy` per
   style — Playful backbeat, Folk pulse, Cinematic bar-kick) exist only in the
   native fallback renderer; `export_performance_midi` writes no channel-9 track.
3. **GM program collisions**: `Oud → 24` (same as AcousticGuitar), `Ney → 75`,
   `PianoPP → 0` — Flamenco's signature timbre renders as Folk's guitar.
4. **One room, one master**: `RenderColor` derives from sliders only (never style);
   the real mastering chain runs only on the native path; fluid path gets peak
   normalization only.
5. **Style-blind performance model**: rubato depths, jitter SD, articulation curve,
   and the MAESTRO (piano!) expressive model apply identical constants to a Lullaby
   and a March. Only swing (2 styles) and drums (inaudible) are style-conditioned.

So a large concrete fraction of "the same band in the same room" is rendering, and
much of it is *days* of work to fix, not weeks. This reorders the critique's
priorities: fix the renderer truth first, because until then, listening judgments
about compositional sameness are confounded.

---

## Part III — The plan

Ordering rationale: (0) unblock the tree, (1) renderer fixes are the cheapest audible
wins AND a prerequisite for any honest listening census, (2) dedup/gating stops
known-bad outputs reaching ears, (3) Atlas instrumentation gives the measurement to
steer by, (4) grammar families are the strategic investment, (5) the blind census is
the release gate, (6) Leptos migration proceeds in parallel.

### Phase 0 — Branch reconciliation + hygiene (BLOCKER for UI work) [~1 day]

`main` has ~20 muse-ui commits this branch lacks (audio-reactive visualizer, Listen
style picker + real intent diversity, seek bar, spec save/load, Motifs/Harmony/
Orchestration Research views, discovery-first landing, Tauri shell, **its own Atlas
implementation** `9a60e4baa5`/`1decc40021`) while this branch independently built
Atlas (`151d1fe062`, `c20a54b147`) + the unified EvidenceView (`1c76631d86`).

- [ ] Decide the canonical line and merge. Recommendation: merge main's muse-ui line
      in, keep this branch's Atlas *backend + compare endpoint* and EvidenceView
      (this branch's Evidence architecture supersedes main's narrower Motifs
      endpoint per its own commit message), and reconcile the two Atlas frontends
      by feature-diffing rather than picking blind.
- [ ] Resolve the uncommitted `symthaea-muse-protocol/Cargo.toml` edit (deletes the
      `[dev-dependencies]` header, silently promoting `serde_json` to a runtime dep
      the wasm graph doesn't need — looks like abandoned prep; revert or commit
      deliberately with a reason). Check it isn't another live session's WIP first.
- [ ] Commit or discard the `UI_mocks/` churn.

### Phase 1 — Renderer truth [days; highest leverage/effort ratio in the plan]

- [x] **CC7 + CC10 at tick 0 per track** — DONE 2026-07-18 (`86918ad042`, on main):
      `ControlChange` variant + per-voice volume/pan CCs; drum channel at CC7 80.
- [x] **Channel-9 drum track** — DONE same commit: new
      `theory_realize::performance_drum_hits` (same swing∘rubato timeline),
      `DrumType::gm_note()` (36/38/42); `DrumPolicy::None` styles stay kit-free.
      Tests: mix/pan CCs asserted per track; Playful gets kick+snare+hats,
      Classical gets none; 702/702 lib suite green.
- [x] **Fix GM program collisions** — DONE same commit: Oud 24→25 (steel guitar,
      nearest DISTINCT plucked timbre; Flamenco's lead no longer renders as its
      own accompaniment's nylon guitar). Ney 75 verified collision-free; PianoPP→0
      documented as deliberate GM practice, not a collision.
- [ ] **Style-conditioned room + mastering**: blend style family into `RenderColor`
      (Cinematic large/wet, Blues dry/close, SacredChoral cathedral) and a per-family
      `MasteringConfig` table instead of `::default()`.
- [ ] **`PerformanceDialect` — per-style performance model** (naming adopted from
      the reviewer's follow-up): scale rubato depth, jitter SD, articulation curve,
      and the MAESTRO expressive model's blend by style family — e.g.
      ClassicalRubato / DanceLocked / JazzLaidBack / FolkLift / ProcessExact /
      DroneElastic / FlamencoCompas. Today one classical-piano-derived performer
      interprets every tradition (flamenco, Afro-Cuban, Hindustani, minimalism all
      breathe/pause/cadence the same way); the MAESTRO model should be ONE dialect,
      not the universal default. Start as a constants lookup keyed off the spec —
      the constants are already named/centralized in `theory_realize.rs` /
      `performance.rs` — not a new ML model.
- [ ] **Run the two-part ablation** the critique asks for, now cheap:
      (a) extend `bin/listening_test.rs` from 3 → 6-8 contrasting styles, neutral
      renderer, style-ID protocol (composition-carries-identity arm);
      (b) scripted `POST /api/compose` loop: one (intent, spec, seed) × N render
      dressings — native vs fluid, state extremes, swing, ensembles (rendering-
      carries-identity arm). Provenance fields already label arms honestly.
      **Record the split before starting Phase 4** — it tells us how much grammar
      work the ears actually need.

### Phase 2 — Dedup + the diversity gate [days]

**P0 within this phase — the Listen path bypasses the whole diversity system**
(elevated 2026-07-18 by the external reviewer's follow-up; independently confirmed
by this review's "silent explorer bypass" finding). The Leptos Listen path sends
`valence 0.15 / arousal 0.45 / energy 0.5 / bars 4 / n_candidates 1` for every
piece: fixed emotional+temporal premise, AND `n_candidates == 1` disables the
Identity Explorer *and* the premise layer server-side. The main listening surface
receives almost none of the diversity machinery. Fix at BOTH layers: (a)
server-side — premise-vary single-candidate composes too (benefits legacy + Leptos
+ any future client) — **DONE 2026-07-18 (`19a541d12c`, on main):
`ComposeRequest.vary_premise` opt-in, default off so authored composes keep their
exact spec**; (b) UI — still open: set `vary_premise: true` on the Listen radio
paths (both UIs) and port the legacy `journeyIntent()` halton trajectories
(already the fix the legacy page made; a backend test even asserts the fixed
constants were removed from the legacy JS).

**Architecture upgrade (adopted from the reviewer's follow-up): select from
finished symbolic scores, not pre-composition proxies.** The Identity Explorer
scores seed *intentions* (hook, form index, ensemble index…) — seeds can look
distant before composition and converge after realization, and parts of its
featurization read the base spec the premise layer then rewrites. Since symbolic
composition is cheap (rendering is the cost center — ~70s/piece measured), the
enforced-output design is feasible: compose 32-64 *symbolic* candidates → drop
exact score-hash duplicates → rich finished-score fingerprints (Phase 3's v2) →
distance vs batch + recent Listen history + keepers → select the N most distinct
that fit the intent → render only those. Diversity becomes an enforced output
property, not a prediction.

Policy note: the standing rule "novelty is an observable, NEVER a fitness function"
stays intact — a *rejection floor* is not an optimization target. Keep the max-min
selector; add a floor and cross-history checks. Do not hill-climb novelty.

- [ ] **Exact dedup on the score hash** (`score_sha256` already computed in
      `piece_provenance`): at candidate-store insert, at the keeper endpoint
      (double-keep currently appends two entries), and at Atlas assembly (collapse
      into one point with a `multiplicity` count — itself diagnostic signal).
      Recipe hash logged alongside for provenance; score hash is the dedup key
      because different recipes provably produce identical music
      (`explorer.rs` test :463-480).
- [ ] **Pre-render novelty floor** at the identified slot (`muse_studio.rs` after
      :1385, before the render loop — symbolic, zero render cost): if the batch's
      min pairwise identity distance is below a configurable floor, re-pick with a
      widened seed window; if still floored (e.g. archetype styles whose harmonic
      channel is structurally near-zero), *vary a high-level decision* — form_pool
      draw, DevelopmentDna, mode/meter from the pools — not just the seed. This is
      the critique's "regeneration must alter a high-level decision" made concrete.
- [ ] **Cross-history novelty**: featurize recent keepers + session history into the
      same Identity space (keeper recipes carry intent + resolved_spec + seed) and
      floor-check candidates against them, not just batch peers. Fixes the blind
      spot where a batch shows "healthy novelty" while cloning yesterday's keeper.
- [ ] **Fix `composeToday()` duplicates** (legacy JS is still served and maintained):
      per-load nonce or server-side dedup of identical (spec, seed) recomposes; and
      widen the 26-day style cycle (styles × primes, or hash the date).
- [ ] **Fix the silent explorer bypass**: `n_candidates == 1` or stride ≠ 1 gets no
      premise variation and no novelty telemetry — exactly when diversity is
      weakest (this is the Listen radio's everyday path). At minimum, premise-vary
      single composes and compute novelty vs recent history.

### Phase 3 — Atlas truth [days]

- [ ] **Multiscale fingerprint v2** — all from data already computed symbolically
      (Score, Form, motif/cadence/sonority/orchestration analysis in the
      listen-bundle pipeline; no audio needed):
      tempo + meter dims (currently absent!); 12-bin pitch-class + 12-bin melodic
      interval histograms; motif layer (count, occurrence density, transformation
      mix) or hashed melody trigrams; harmony layer upgrade (chord-quality
      histogram from `sonority_regions`, cadence-type counts, real mode one-hot
      instead of one 0/0.5/1 scalar); instrument identity from the resolved
      ensemble; per-section rhythm/contour/register stats (multiscale, not just
      whole-piece); **per-layer normalization** so a layer's dimensionality stops
      being an implicit weight. Target ~60-100 dims. Version the fingerprint
      (`fp_version` field) so stored vectors don't silently mix schemes.
- [ ] **Per-lens distance + NN**: run the NN loop on `weighted()` vectors after lens
      resolution (O(n²) over ≤~200 points is trivial); pass the lens into
      `atlas_compare` so "why nearby" matches the current view. Return BOTH
      `nearest_global` (raw) and `nearest_for_lens` — users can then distinguish
      "overall closest" from "closest rhythmically". Also fix the `motif_form`
      lens's honesty: it boosts form + the 3-value contour layer and contains no
      actual motif representation — either rename or back it with the v2 motif layer.
- [ ] **Fingerprint the bypass forms** (reviewer's follow-up, verify then fix):
      Fugue/Sonata/Opera/Renaissance/ProgSuite/ground forms return no generic
      `Form`, so their form-layer dims degrade to zeros — the Atlas is least
      informed about exactly the pieces with the most distinctive architecture.
      Emit a form descriptor from each bypass engine (they all know their own
      sectional plan) or derive sections from the score.
- [ ] **Transposition invariance**: pitch content should compare
      transposition-normalized (and tonic represented circularly, not linearly —
      B and C are currently maximally far apart); report absolute key as metadata,
      not distance.
- [ ] **Persist fingerprints + score hash at keep time** in the keeper jsonl —
      indexing stops depending on recompose determinism across engine versions
      (the recipe records engine versions precisely because drift is possible).
- [ ] **Keeper backfill**: (i) fingerprint the one legacy keeper with real MIDI on
      disk via a small MIDI→notes importer; (ii) one-time migration of the other 24
      using the legacy Listen intent constants as *documented assumptions*, writing
      reconstructed recipes with `reproduction_gaps`, rendered in the Atlas with an
      "approximate provenance" flag rather than dropped. 2-3 usable keeper points →
      up to 28. Reproducibility status is displayed, not used as an existence filter
      — exactly as the critique prescribes.

### Phase 4 — Grammar families [the strategic investment; staged]

**Target architecture (adopted 2026-07-18 from the external architecture review —
an upgrade of this phase, not a rewrite of the engine):**

```
Intent + History + Taste ──▶ HDC Composer Memory ──▶ CfC Temporal Executive
        ──▶ CompositionPlan (grammar · arc · obligations · identity)
        ──▶ Grammar-Specific Composer ──▶ Theory/Style Invariants
        ──▶ Development/Revision Loop ──▶ Finished-Score Diversity Gate
        ──▶ PerformanceDialect ──▶ Renderer
```

Key commitments, mapped to what already exists:
- **Grammar-family routing** formalizes the existing bypass-engine dispatch
  (`composer.rs:326-416` early returns) into a first-class concept: ~10 families
  (period/sentence, groove-cycle, strophic song, call-and-response,
  process/additive, ground-and-variation, contrapuntal, developmental, raga arc,
  through-composed dramatic). A style selects and specializes a family. Each
  family must also DECLARE which intent axes it honors — the census proved bypass
  grammars are legitimately deaf to valence/energy/bars, and
  `Style::supported_intent_axes()` is already the hook.
- **`PhraseGrammar` / `HarmonicGrammar` interfaces** (extract before building new
  families): phrase-length distributions, repetition/continuation/cadence
  behavior, elision, silence per grammar; harmonic TRANSITION MODELS per family
  (functional, blues chorus, modal drone, turnaround, Phrygian descent, planing,
  cyclic loops, pedal, raga pitch hierarchy). **Cadences become proposals from
  the grammar, never universal corrections** — the direct fix for
  `Period::parallel_in` overwriting every style's phrase-end harmony.
- **`CompositionPlan` planning layer**: grammar, formal arc, phrase/section
  lengths, tonal regions, motif roles, tension/texture trajectories, climax
  strategy, return/transformation obligations, performance dialect — built
  BEFORE notes, fulfilled by generation, deviations recorded. Substrate already
  in-tree: `ObligationLedger` (sonata-only today) is the obligations mechanism;
  `plan_damage()` is the diagnose-then-select precedent. This is what stops the
  pass stack from independently editing pieces toward one rhetoric.
- **Grammar-aware passes**: every universal pass (cadence steering, climax,
  damage, rhetoric, rubato) declares which families it supports + preconditions.
  A classical cadence pass must not rewrite blues or raga harmony; nocturne
  rubato must not touch a groove-locked dance.
- **HDC/CfC as the EXECUTIVE, not the note generator** (deliberately LAST, and
  evidence-gated): grammar proposes valid actions → HDC represents piece/
  history/taste/obligations → CfC evolves composer state → policy selects the
  next symbolic action → invariants validate → diversity gate rejects collapse.
  Gate: pre-registered ablations proving musical benefit (same discipline as the
  keystone A/B — note the cognitive loop passed its FIRST learning gates only on
  2026-07-18 after earlier claimed benefits self-retracted as trainer artifacts,
  and Muse's cognition-study apparatus is dormant infrastructure with no study
  run; do not wire an executive on vibes).

Upgrade sequence (supersedes the flat tier list below where they overlap):
1. Renderer truth + Listen routing (Phases 1-2 — in flight, first commits landed
   2026-07-18 on `muse/renderer-truth`).
2. Extract `PhraseGrammar` / `HarmonicGrammar` / `PerformanceDialect` interfaces.
3. Build three unmistakably different families first: groove-cycle,
   process/additive, raga arc.
4. HDC/CfC planner/decision-selector with ablation gates.


The engine's own comments concede the critique; the counter-assets are the six real
bypass engines and a pile of built-but-unwired mechanisms. Sequence by
leverage-per-risk:

- [ ] **Tier 0 — data-only rewiring [days]**:
      cross-pollinate `form_pool` (let JazzBallad/Cinematic seed-pick Sonata or
      Variations; Celtic pick Passacaglia — zero new engine code, the Fugue preset
      even documents the possibility); populate `meter_pool`/`mode_pool` beyond the
      3 styles that use them; ship non-`None` `Attitude` on suitable presets (fully
      wired, used by zero styles); reassign the 3 styles sharing Sequential
      development now that Wandering/Intensifying exist; new cheap
      `DevelopmentDna`/`PhraseRhetoric` variants (each = enum arm + mutation pass).
- [ ] **Tier 1 — per-style harmonic syntax + phrase asymmetry [1-2 wks each]**:
      make `Progression::generate`'s transition table a spec field (functional /
      rock ♭VII / jazz turnaround / Phrygian-descent grammars) so styles get
      harmonic *syntax*, not progression pools — the critique's §3 verbatim;
      irregular phrase lengths (5-bar, elision) in `Phrase::build`/`Form` — the
      hard part is the `si * 2 * bars` uniform-span arithmetic sprinkled through
      the pass stack.
- [ ] **Tier 2 — new bypass engines [2-4 wks each, template proven]**: each existing
      engine is 320-1,400 LOC + tests. Highest-payoff candidates, mapped to the
      critique's grammar families: verse/refrain **song form**; **call-and-response**
      (the Flamenco ship explicitly deferred it); **raga-style alap-jor-jhala**
      (Hindustani currently fakes it inside the period pipeline via full_drone);
      groove/cyclic engine (clave/tala cycles as the *form*, not the accompaniment).
- [ ] **Tier 3 — break the phrase-grammar monopoly [1-2 mo]**: a `PhraseGrammar`
      abstraction (period / sentence / fortspinnung / chorus / ostinato-cell)
      selected by spec — today "sentence vs period" is the only choice and it's
      arousal-driven; hybridize the *second engine* (the trained `MelodyPredictor` /
      implication-realization path, currently disconnected from the theory path) as
      a phrase-continuation candidate source, filtered by the engine's invariants.
- [ ] **Melodic identity contract** (critique §2): extend `MelodicDna` from "which
      hook pools" to a per-piece contract — interval vocabulary, phrase-length
      distribution, permitted transformations, repetition threshold, register
      trajectory — enforced as invariants the way the spec already enforces
      wrong-note safety. The 12/29 styles on identical default hook pools are the
      first fix.
- Also wire dormant assets where cheap: `counterpoint::fit_against` into the period
  pipeline's counter-melody; `ObligationLedger` for style-specific long-range plans.

### Phase 5 — Blind listening census [the release gate]

After Phase 1 (renderer honest) and at least Tier 0-1 of Phase 4:
- [ ] Run the extended Test A (style/grammar-family ID, 6-8 styles) + Test B
      (identity-grammar recognition) from the existing battery design, with the
      existing confidence scoring + confusion table.
- [ ] **Release gate, adopted from the critique**: listeners reliably distinguish
      grammar families; pieces within one style remain recognizably related without
      being interchangeable. No new styles until the gate passes — "another twenty
      presets could make the problem more obvious."

### Phase 6 — Leptos migration completion [parallel track, after Phase 0]

Gap list (legacy → Leptos, post-reconciliation some close automatically via main):
- [ ] Listen: journeys (Resonance/Discovery/Contrast halton intents) — NB the
      current Leptos port *reintroduced* the fixed `valence 0.15/arousal 0.45/
      energy 0.5` intent that a backend test celebrates the legacy page removing;
      remaining viz modes; previous-history; next-preview + relation label; seek
      scrubber; section legend; current-moment card; voice blend; emotion arc.
- [ ] Create: tier system; identity-grammar selector; base seed; More-like-this /
      distant-cousin; novelty breakdown display; neuromod dials + presets;
      per-voice instrument pickers + preview; spec editor (save/load); per-candidate
      piano roll; CLAP similarity display.
- [ ] Research: Performance / Identity / Provenance sub-views
      (`/api/piece/{id}/performance-bundle` and `/provenance` currently have zero
      Leptos consumers); seekable canvas timelines; explanation inspector;
      analysis-availability panel.
- [ ] Architecture during the port: backend origin override (window-var pattern, not
      the hardcoded `127.0.0.1:8400`); make `MuseState::show_piece` public (Create
      re-implements it); factor the now-triplicated rAF+`SendWrapper`+`on_cleanup`
      scaffold; add generation guards to EvidenceView/ScoreView fetches (same race
      class `request_generation` already fixes); prefer `create_resource` over
      hand-rolled loaded-flag effects; extend `symthaea-muse-protocol`'s
      `ComposeRequest` (grammar/spec/seed_stride/explore/neuromods) instead of
      hand-rolling UI types.
- [ ] Retire the legacy page only when the gap list is empty — it is still the more
      capable UI today and backend tests assert its content.

---

## What we are explicitly NOT doing

- Adding styles before the Phase 5 gate passes (per the critique and the standing
  style-rule).
- Making novelty/Φ a fitness function — floors and observables only.
- Neural end-to-end audio (unchanged standing decision).
- Trusting the "native fallback renderer" assumption ever again without checking
  the live process env — the review found the opposite was true.

## Measurement of success

1. Phase 1 ablation quantifies composition-vs-render sameness (recorded number).
2. Atlas v2: style clustering measurably improves (within-style NN fraction);
   duplicates appear as multiplicity, not points; ≥25/28 keepers indexed.
3. Zero exact-duplicate candidates reach any UI (score-hash gate).
4. Phase 5 census: grammar families distinguishable blind, above chance with
   calibrated confidence.
