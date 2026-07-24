# Muse: Comprehensive Review & Plan (2026-07-21)

## Update 4: renderer-correctness fixes, then the real Hybrid map

Two rounds of follow-up after Update 3 shipped, both prompted by close user inspection
against the design spec rather than a fresh review pass.

**Round A — renderer correctness + header CSS (commit `bf83755bf2`).** The user compared a
screenshot against `MUSE_LISTEN_MODE_VISUALIZATION_DESIGN_SPEC.md` and flagged specific,
verifiable bugs: the "scattered dots" texture used `(i * 2654435761) >> 8`, which for `i` in
`0..40` only ever spans a narrow low sliver of `[0, u32::MAX)`, clustering every dot into one
arc instead of scattering them — fixed with a proper SplitMix64-style hash finalizer. The
Radial geometry used fixed pixel radii (`r0 = 46.0 + …`, ring at `r0 + 165.0`) that clipped
off a short 320px-tall canvas — fixed by making every radius proportional to `min(w, h)`.
There was no HiDPI handling (`devicePixelRatio`) — added, capped at 2x. The visualizer never
read real Web Audio samples even while something was actually playing — wired
`audio_reactivity::waveform()` into `bass`/`mid`/`progress`, added a real playhead on the
outer ring, made the 72 spokes angularly stationary (previously rotated with wall-clock time,
which the design spec explicitly calls out as wrong: "structure stays spatially stable").
Added `prefers-reduced-motion` support and a hidden-tab draw-skip. Separately, the header's
`.brand-lockup`/`.header-now-playing`/nav `<span>+<small>` markup had **zero matching CSS**
(confirmed via grep — the classes existed in `app.rs` but the corresponding rules had been
deleted or never written), which is why nav labels rendered stacked with no line break
("ListenImmerse"). Rewrote the header CSS for the markup that's actually there.
Screenshot-verified before/after: dots genuinely scattered, no clipping, playhead visible
with a trailing arc, nav labels correctly two-line.

**Round B — the actual composition-evidence Hybrid map (this update).** Everything in Round A
made the Radial view a correct *audio-reactive* map; it still had no connection to the
piece's actual composed structure — exactly the gap the design spec's "Hybrid" model and the
user's own critique called the real missing piece ("Restore the evidence-backed Hybrid map
first… reads as 'the living architecture of the piece,' not just an audio-reactive orb").
`GET /api/piece/{id}/listen-bundle` already existed and was already trusted by Research
Mode's `evidence_view.rs` — real sections (start/end seconds + intensity), phrases (with
`closes_with_cadential_marker`), cadences (exact arrival time), and scored motif occurrences,
each with an honest `EvidenceBasis` (Observed/Reconstructed/Inferred). Wired the same fetch
into `ListenPage`, extracted into a lightweight `CompositionMap` (drawing-only scalars, so
cloning it every animation frame stays cheap — the full bundle also carries every note event
and the sonority/orchestration timelines). New `draw_composition_evidence()` renders, in a
band just outside the audio-reactive spoke ring: alternating-tone section arcs (thickness/
opacity from the section's own `intensity`, so a returning section like ternary form's "A
(return)" visually echoes its first appearance), phrase-boundary ticks (brighter at a genuine
cadential close), cadence markers on the ring itself, and outlined motif-occurrence rings
between the spoke ring and the section arcs — honestly empty when the bundle's own
conservative similarity threshold found no qualifying returns, never fabricated.

Verified against two different real candidates (ids 20 and 22, 4 and 6 sections, 9 and 22
motif occurrences respectively) at 2x scale-factor screenshots — evidence counts in the
rendered image were cross-checked against the raw bundle JSON and matched exactly; no
clipping against the reduced `outer_ring` cap (0.42, down from 0.47) needed to make room for
the new `evidence_ring` band (0.48 cap). Section-arc alternation (`palette.a`/`palette.b`
by index) is present in the code but visually subtle for styles whose two palette tones are
close in hue — a real but minor legibility tradeoff, not a bug, and left as-is rather than
over-designing a legend onto an otherwise deliberately minimal-chrome view. Bars/Waves modes
were separately re-checked for the same class of bugs (fixed-pixel clipping, missing HiDPI/
reduced-motion/reactivity) and found to already inherit all of it for free from the shared
`start_visualizer`/`draw_frame` preamble — their own geometry was already proportional to
`w`/`h` with no clipping, so no changes were needed there; per the design spec and the user's
own critique, Bars/Waves are meant to stay ambience-only modes rather than carry composition
evidence, so this was a verify-not-fix pass, not a gap.

**Not done, deliberately deferred**: `VisualizationKind`/`VizMode` unification (the two
still-separate enum systems — evidence-backed `visualization.rs`, currently unwired, vs. the
simpler `VizMode` this update extended); a typed `ListenVisualFrame` to decouple rendering
from ad hoc state queries; the composed/performed/overlay toggle (design spec §4.2/§19); a
dedicated context-rail region (§5.3); and the "Why This Piece?" content mismatch (spec says
it should explain *selection*, i.e. why this piece over the alternatives Muse considered;
`Candidate.why` currently contains composition-mechanics prose instead) — all real gaps,
all bigger scope than a follow-up round.

**Round C — corrections from an independent code review of Round B.** A closer read of
Round B's own diff caught five real issues, all verified against the actual code before
fixing (not taken on faith):

1. **Structure was drifting with audio energy.** `outer_ring`/`evidence_ring` were derived
   from `r0`, and `r0` reacts to bass (`base * 0.15 + bass * base * 0.06`) — meaning every
   section arc, phrase tick, cadence marker, motif ring, and the playhead's own path would
   subtly expand and contract with the music, directly contradicting this view's own stated
   principle ("musical structure remains spatially stable; playback and audio make it come
   alive"). Fixed by making `outer_ring`/`evidence_ring` flat constants (`base * 0.44` /
   `base * 0.48`) with no `bass` term at all — a "doesn't drift" guarantee by construction,
   not something that needs re-verifying by eyeballing screenshots at different volumes.
   `r0` itself (glow + spoke origin) keeps its audio reactivity, sized so even a max-bass
   spoke tip (`0.21 + 0.02 + 0.19 = 0.42 * base`) stays inside the now-fixed ring.
2. **Bass/spoke energy came from the wrong analyser tap.** `bass`/`mid` were derived from
   `audio_reactivity::waveform()` (time-domain amplitude — "how loud right now"), not
   frequency data. `audio_reactivity::spectrum()` (real 64-bin frequency data via
   `get_byte_frequency_data`) already existed in the crate but was dead code. Switched
   `bass` to the mean of its lowest 6 bins and `mid` (spoke/bar energy) to sample across all
   64 bins — `waveform()` stays reserved for Waves mode's raw line and the player bar's own
   timeline wave, where preserving polarity/zero-crossings is actually wanted.
3. **`CompositionMap` was being deep-cloned every animation frame.** `composition.get_untracked()`
   cloned every section/phrase/cadence/motif `Vec` at 60fps. Wrapped in `Rc` — a `RwSignal`
   over a non-`Send`/`Sync` type needs Leptos's `LocalStorage` variant
   (`RwSignal::new_local`/`RwSignal<T, LocalStorage>`) rather than the default, which is
   sound here since wasm32 is single-threaded.
4. **Stale-response race on both evidence fetches.** Neither the `sections` (`/api/motifs/{id}`)
   nor `composition` (`/api/piece/{id}/listen-bundle`) fetch checked whether the user had
   already moved to a different piece by the time the response arrived — a slower response
   for piece A landing after a faster one for piece B would silently overwrite B's evidence
   with A's. This is a data-integrity bug, not just a glitch: the rendered evidence would be
   real, just for the wrong piece. Fixed by checking `muse.current` still matches the
   fetch's own id before committing either result.
5. **Section coloring alternated by array index only**, which doesn't track section
   *identity* — a Rondo's two "A" returns could land on different colors than each other by
   coincidence of index parity. Investigated before fixing: `muse_studio.rs`'s
   `section_regions()`/`expected_section_labels()` DOES emit real semantic role slugs (e.g.
   Rondo's refrain/episode/return, Passacaglia's cycle/climax) for the handful of forms it
   recognizes, falling back to the literal role `"region"` (an honest "no identity signal"
   marker) for every other form. Colors by first-seen-role-position when `role != "region"`
   (correctly unifies e.g. both Rondo "return" sections, and correctly makes Passacaglia's
   lone "climax" stand out instead of blending into the surrounding cycles under plain
   index-parity), and falls back to index-parity when the role carries no real identity —
   avoiding the regression a naive "always group by role" would have caused for the common
   unlabeled case, where every section shares the literal string `"region"`.

Re-verified via 2x-scale-factor screenshot after the rebuild: ring/evidence still render
cleanly, no clipping, idle-state rendering unchanged. The reviewer's own suggested next
architectural move — a real `ListenVisualFrame` struct unifying structure/playback/audio/
capabilities/preferences — remains correctly identified as the right direction but bigger
scope than this pass; also still deferred: cadence *types* (only arrival markers are
exposed), motif-duration arcs (currently start-point markers only), voice lanes, and a
multi-DPR/multi-palette/reduced-motion screenshot regression suite.

## Update 3: recovered more stranded work by checking pushed topic branches

A user screenshot comparison ("this doesn't look anywhere near as good") plus an explicit
"check what else may be in the uncommitted work" prompted a wider sweep than the worktrees
checked earlier: `git branch -a` and `git reflog show --all` turned up **several small,
real, already-pushed-to-origin topic branches** nobody had ever checked, none ancestors of
our HEAD: `origin/feat/muse-listen-section-progress` (a real live current-section indicator
+ progress bar), `origin/feat/muse-listen-diversity-and-form-viz` (style picker + real
per-piece intent diversity + a Form ring viz), `origin/feat/muse-listen-polish-audio-reactive-wave`,
`origin/fix/muse-listen-crossorigin-analyser`, `origin/fix/muse-listen-analyser-gesture-and-lava-lamp`.
All converge into `muse/wave2` — the branch the *currently active* `session-muse-renderer`
worktree sits on (verified stalled: last commit 3+ days old, no live process in that
directory, before touching anything). Verified `origin/main` separately: fully caught up to
our HEAD for muse (0 commits we're missing), so this was never a "pull from main" gap —
the work is genuinely stuck only on these isolated branches, unmerged anywhere including
mainline.

Ported the highest-value, most tractable pieces as targeted extractions (not a branch
merge — same lesson as `muse-evidence-port` earlier: a real `git merge` of `muse/wave2`
would pull in ~180 commits of unrelated changes):

- **Real current-section indicator**: found the server-side commit
  (`53346399e2`, `GET /api/motifs/{id}`) this depends on, verified all its prerequisites
  (`compose_with_spec_and_form`, `section_bar_map`, `SectionRole`, the `form` field on
  `Candidate`) already exist in our HEAD — `form` is already stored and used for Atlas's
  structural fingerprint, just never exposed via its own endpoint. Added the route +
  handler (adapted, not copy-pasted verbatim) and the client fetch. Placed the badge
  **top-left** as requested, with genuinely more info than the original branch's plain
  text line (role, key, and a bar range) — positioned over the canvas, not fighting the
  transport controls that live elsewhere in the layout.
- **Real per-piece intent diversity**: Listen pieces were composing with a hardcoded
  valence=0.15/arousal=0.45/energy=0.5 for literally every single piece (only tonic and
  seed varied) — found while reading the topic branch's own commit message, which measured
  the same problem independently. Randomized within Create Mode's own slider bounds.
  Deliberately did NOT adopt the topic branch's full `JourneyIntent` system — it predates
  today's reducer migration and would need its own reconciliation; this captures the real
  diversity benefit without that overhead.
- **Deliberately not ported this pass**: the Form ring visualization and style/diversity
  picker UI — real, but built against a state model (pre-reducer, pre-shared-`VizMode`)
  that's since been superseded twice over (once by today's reducer migration, once by
  `muse/wave2`'s own now-different state.rs). A genuine UI feature to revisit, not a
  quick port.

Verified: `cargo check` clean for both crates (native muse_studio + wasm32 UI), no new
warnings beyond pre-existing ones.

## Update: same-day execution log

All of Phase 0 and Phase 1 below were executed the same day, plus corrections to two
Phase 2/3 items that turned out to rest on an incomplete read of the actual code. Landed:

1. **FluidSynth is now actually active.** The long-running `muse_studio` process (pid
   3717787, up since Jul 19 without the right environment) was killed and relaunched via
   `nix develop .#muse` — no rebuild needed, since fluid_render's backend selection is a
   runtime env-var check, not a compile-time feature. Startup log now reads
   `Renderer: FluidSynth (...) + soundfont (...)` instead of silently falling back.
2. **The `symthaea-muse-ui` build is green again** (commit `c38bcbe6e8`). Root-caused via
   git archaeology, not guesswork: `mod icons;` was simply never added to `main.rs` despite
   `icons.rs` existing since a 2026-07-17 visual-polish commit, and the
   `list_specs`/`spec_preset`/`load_named_spec`/`save_spec` API functions plus
   `ComposeRequest`'s `vary_premise`/`spec` fields existed only on the stalled
   `muse-evidence-port` worktree's branch (`muse/evidence-and-title-provenance`, last
   commit 3 days old, no active process — confirmed genuinely stalled, not just quiet,
   before touching it). A full branch merge was attempted first and aborted after pulling
   in ~20 unrelated conflicts across quantum-chemistry/vision-manifold/mycelix-pulse (a
   full topic-branch merge drags in everything both sides touched since diverging, not
   just the muse-relevant part) — ported the specific pieces by hand instead: the protocol
   crate's file is a pure additive superset of ours (verified via diff, safe wholesale
   replace), `api.rs` needed a manual merge (our HEAD had its own independent
   `compose_listen_piece` the other branch lacked), and `harmony_view.rs`/`motifs_view.rs`/
   `orchestration_view.rs`/`journey.rs`/`player_bar.rs` turned out to be **byte-identical
   to files already sitting uncompiled in our own tree** — brought in by an earlier
   patchset commit but never `mod`-declared. Dropped the harmony/motifs fetch functions
   initially ported since this branch's server has no matching routes (confirmed by grep)
   — calling them would just 404. `player_bar.rs`/`journey.rs`'s richer transport-state
   integration was deliberately NOT pulled in — it depends on a different, more complete
   `MuseState` transport model (`is_playing`/`volume`/`current_time` vs. our
   `playing`/`playback_seconds`) that would need real reconciliation, not a blind copy;
   scoped as follow-up, not done today. Verified: `cargo check -p symthaea-muse-ui --target
   wasm32-unknown-unknown` went from 7 hard errors to a clean build (1 pre-existing,
   unrelated warning).
3. **Also resolved by inspection, not code**: `harmony_view.rs`/`motifs_view.rs`/
   `orchestration_view.rs` turn out to be **deliberately superseded**, not blocked —
   `muse-evidence-port`'s own doc comment says so explicitly ("the earlier per-topic
   views... still exist... but ResearchPage no longer renders them — superseded by the
   unified Evidence panel"). Left unwired on purpose, matching the newer design's intent.
4. **`foundry_review_page.rs`/`add_music_page.rs` non-issue confirmed**: neither is
   `mod`-declared, and `app.rs`'s actual `<Routes>` never references `/foundry` or
   `/add-music` — so the "Foundry"/"+ Add Music" nav items in the screenshot that kicked
   off this whole investigation are purely an artifact of the **stale build** (the running
   process predates even the commit that added those two files, confirmed by timestamp).
   Once rebuilt from current source, those nav items won't appear at all. No backend work
   started for them — that would be new product scope needing its own decision.
5. **Two corrections to the original plan**, both verified by reading the actual code
   rather than trusting the first-pass characterization:
   - **CLAP/FAD → critic.rs was never a valid fix.** `FadScore::compute_with_clap` computes
     a Fréchet distance between two *distributions* of already-rendered 48kHz audio — a
     corpus-level evaluation metric. `music_auto_improve`'s tight round loop scores
     unrendered symbolic `Composition`s and needs a fast per-round scalar; forcing FAD in
     would mean rendering audio every round (expensive) and misusing a distributional
     metric as a single-sample score. Also found CLAP isn't actually a gap elsewhere:
     `steering.rs` already uses it correctly and live for prompt-to-audio candidate ranking
     (wired into `muse_studio.rs`'s compose path). No change made — `critic.rs`'s own doc
     comment already discloses its heuristic honestly, which is the right call given the
     loop's real-time constraints.
   - **Neither neural checkpoint fix is a quick wire-up.** `spectral_vocoder.rs` and
     `mel_mlp.rs` each define their *own* distinct `MelDecoder` type (a naming collision
     that misled the first review pass) — `spectral_vocoder::MelDecoder` is a fixed
     genesis-seeded random-projection decoder with no save/load format at all (there is no
     checkpoint to load, ever). `mel_mlp::MelDecoder` is the real trainable one with real
     checkpoints (verified: all 4 files in `/opt/datasets/maestro/checkpoints/` parse with
     correct self-describing headers and real non-NaN weights) — but its only consumer is
     `predict_mel.rs`, an offline evaluation tool; `SpectralVocoder::new` (confirmed dead
     code, zero callers anywhere) never touches it. `neural_melody.rs`'s
     `load_trained_projections()` is genuinely load-only — there is no save/train path
     anywhere in the crate that ever produces a compatible file, so "train it for real"
     needs a training objective designed from scratch, not a rerun of something that
     already exists. Giving `mel_mlp`'s real trained weights an actual audio output would
     mean bridging it into `spectral_vocoder`'s oscillator-bank renderer and exactly
     replicating `train_mel_mlp.rs`'s state-vector encoding at inference time — a classic
     train/serve-skew risk that can't be validated in this environment (no way to listen to
     the output here). Documented as a concrete next step rather than attempted blind.

**Not yet done, deliberately**: the `player_bar`/`journey`-picker/live-`AnalyserNode`-visualizer
feature set from `muse-evidence-port` (real, tested, valuable) needs the `MuseState`
transport-model reconciliation noted above before it can land safely. A `trunk build
--release` is running to produce a fresh, correct `dist/` to replace the stale Jul 19 one
— once done, restart `muse_studio` (or however the Leptos UI is served) against it.

## Update 2: the transport-model reconciliation happened same-day too

Turned out to be the right call to defer, then revisit: the user's own screenshot comparison
(rich uncommitted build vs. the plain rebuilt-from-source page) made it worth doing properly.
Root cause of the mismatch, confirmed empirically rather than assumed: `muse_studio.rs`'s own
router only ever served the *legacy* vanilla-JS `studio/index.html` via `include_str!` — the
process actually answering `:8400` was built from an **uncommitted local change** that added
real static-file serving for the Leptos `dist/`, which never made it into any commit. Restored
properly: `tower_http::services::{ServeDir, ServeFile}` now serves the Leptos build at `/`
with SPA fallback to `index.html`, and the legacy page moved to `/legacy` rather than deleted
(needs the `fs` feature added to this crate's own `tower-http` dependency — the workspace-level
Cargo.toml already had it for a different crate, this one didn't).

With that fixed, merged `state.rs`'s two independently-evolved transport models: kept our
`next_piece`/`show_piece`/`keep`/2-arg-style `compose_listen_piece` core, adopted
`muse-evidence-port`'s richer `is_playing`/`volume`/`current_time`/`duration` + lazy real
Web-Audio `AnalyserNode` tap (`toggle_play`/`restart`/`audio_analyser`), deliberately leaving
out its `journey`/`desired_style`/style-pinning layer (real, but bigger scope than asked for
today). Wired `<PlayerBar>` into `app.rs` (mounted outside `<Routes>`, plus the `crossorigin`
audio attribute + `play`/`pause`/`timeupdate`/`loadedmetadata` handlers the analyser tap and
seek bar need). Added a `VizMode` shared between the Listen hero canvas and a new mini
corner-preview in the player bar (`state.rs`, was a page-local enum before) plus a new `Still`
mode (frozen single frame) and 4 new icons (Radial/Bars/Waves/Still) matching the existing
2px-stroke icon set. Time display now two-tone (elapsed/total colored from the current piece's
own palette) with a slow pulse, larger, moved next to a mini radial + mode-toggle icons on the
left of the timeline.

Also added real renderer choice, both ends: `ComposeRequest.renderer: Option<String>`
(`"native"` forces the in-crate synth even when FluidSynth's available; anything else keeps
the existing auto-preference), a matching Auto/FluidSynth/Native selector in the player bar,
and — found while wiring it — **`vary_premise` (added earlier today) was a silent no-op**: the
server's own local `ComposeRequest` struct (a hand-maintained wire-compatible duplicate of the
protocol crate's, not the same Rust type) never had the field at all, so setting it on the
client did nothing. Fixed for real, and used the same pass to implement Phase 2's first item
from the Diversity Truth plan: exact-duplicate detection via `exact_fingerprint` (already
existed in `symthaea-music-theory`, unused), a `mark_duplicate` helper with its own unit tests,
wired into the compose loop's per-candidate insertion — `Candidate.duplicate_of` is now real,
not just a wire-format placeholder.

**Deliberately not done**: a live per-section "current section" indicator — current source has
no per-section timing reachable client-side, and faking one would violate this codebase's own
"never fabricated" discipline. Needs a new `/api/motifs/{id}`-style backend endpoint first;
scoped as real follow-up, not attempted today.

**Not yet verified**: this wave of changes (Cargo.toml `fs` feature, ServeDir/ServeFile router
change, state.rs merge, player_bar.rs additions, icons.rs additions, muse_studio.rs's
`renderer`/`vary_premise` fields) was implemented under heavy shared-CPU contention — the
monorepo's `cargo-gate.sh` serializes all `cargo` invocations across every concurrent session,
and the queue was long. `cargo check -p symthaea-muse-ui --target wasm32-unknown-unknown`
caught one real issue (missing `web-sys` features for `AudioContext`/`AnalyserNode`/etc.,
fixed) and is expected clean on the next run; the native `muse_studio` binary has not yet been
rebuilt+restarted with this wave's server-side changes (dedup, `vary_premise`, `renderer`) —
the live process is still the earlier restart (FluidSynth-only fix).


Six-fork parallel review of `symthaea-muse` (96K LOC, ~140 modules), `symthaea-muse-ui`,
and `symthaea-muse-protocol`, followed by a deeper investigation into the highest-leverage
findings. Every claim below was verified against the code or a live process, not inferred
from doc comments or MEMORY alone — several prior characterizations (including ones from
this session's own first-pass synthesis) were corrected in the deeper pass.

## The three findings that actually change what to do next

### 1. The live Muse Studio server is running the worse of two renderers, right now
`fluid_render.rs` wraps FluidSynth + a real soundfont. Its own doc comment records that an
A/B listening test **already settled this decisively**: "no longer fights the composition
... if I had heard these renders first I would not have said the instruments sound harsh."
It's correctly wired into `muse_studio.rs`'s render paths, gated on `SYMTHAEA_SOUNDFONT`
(required) and `fluidsynth` being resolvable (`available()`, `fluid_render.rs:89-102`).

The `muse_studio` process actually serving `localhost:8400` (pid 3717787, running since
2026-07-19 20:36, i.e. the process behind the screenshot in this session) was started with
neither set — `/proc/<pid>/environ` shows no `SYMTHAEA_SOUNDFONT` and no `fluidsynth` on
`PATH`. So every piece rendered by this instance uses the "zero-dependency fallback"
in-crate synth that the crate's own A/B test found sounds harsher. This is a live, easy fix:
restart `muse_studio` inside `nix develop`/`nix-shell` with a soundfont path set (the doc
comment says the Muse Studio launcher already knows how to do this) and re-listen.

### 2. A day and a half of real, tested UI work is sitting on an unmerged branch — the "broken build" is a merge gap, not missing implementation
The first review pass found `symthaea-muse-ui` fails `cargo check` (wasm32 target): 7 errors
in `create_page.rs` (unresolved `icons` module, four missing `api::` functions, one missing
`ComposeRequest` field), plus 11 of 21 UI source files never `mod`-declared in `main.rs`,
including `harmony_view.rs`, `motifs_view.rs`, `journey.rs`, `player_bar.rs`.

Git archaeology traced this precisely: commits `591baa644d` (harmony endpoint), `f79d03cc78`
(Motifs view + protocol type), `8ef19379c7` (spec save/load), `275ddcddfa` (`vary_premise`
protocol field), and the icons/player-bar visual-polish pass are **all real, and all still
reachable** — but `git merge-base --is-ancestor` confirms none of them are ancestors of this
branch's or `origin/main`'s HEAD. They live on `muse/evidence-and-title-provenance`, the
branch backing the **currently-active** `muse-evidence-port` worktree (tip `c18716a942`).
Diff: 164 commits ahead / 191 behind our HEAD (genuinely diverged, not just behind).

Confirmed directly: that worktree's `main.rs` declares every module ours is missing, and its
`api.rs`/protocol crate already contain every symbol `create_page.rs` needs
(`list_specs`/`spec_preset`/`load_named_spec`/`save_spec`, `vary_premise`). **Merging that
branch is very likely a near-complete fix for the UI build**, not a rebuild.

Two files are genuine exceptions, present only on our HEAD and absent from *both* other
worktrees checked (`muse-evidence-port`, `session-muse-renderer`): `foundry_review_page.rs`
and `add_music_page.rs`, both introduced by a single bulk-patchset commit (`82330c0c0a`,
"apply muse native campaign, evidence, and preflight patches", 2026-07-20). These want
backend endpoints (`fetch_foundry_qualification`, `import_music`) that don't exist on any
branch checked — these are the ones that are still genuinely speculative.

**Action, not yet taken pending coordination**: do not silently merge another session's
active branch. Check whether `muse-evidence-port` is close to landing this itself; if
stalled, a deliberate merge (in a fresh worktree, reviewed for conflicts) recovers real work
rather than requiring anyone to rebuild it.

### 3. The neural layer's diagnosis needs a correction, and the "should this use Symthaea" question has different answers for its two halves
The first review pass characterized both `neural_melody.rs` and `mel_mlp.rs`/`spectral_vocoder.rs`
as "bespoke nets disconnected from Symthaea's real substrate." Verified more precisely:

- **`neural_melody.rs` already imports and runs on the real substrate**
  (`symthaea_core::hdc::hdc_ltc_unified`, `ContinuousHV`, `GenesisSeed` —
  `neural_melody.rs:11-13`). It's on the live `streaming.rs` render path. The actual gap is
  narrower than first reported: `load_trained_projections()` exists but nothing ever calls
  it, so it runs with genesis-random (untrained) projection weights on top of otherwise-real
  substrate dynamics. This is a training-and-wiring gap, not an architecture gap.
- **`mel_mlp.rs`/`spectral_vocoder.rs` is the one that's genuinely a standalone bespoke MLP**
  (hand-rolled Adam optimizer, no `symthaea_core`/`symthaea_fep` import at all) — and it's
  the one confirmed unwired from any render path (`grep` finds no caller of `SpectralVocoder`
  outside its own file).
- **Training data is not the blocker for either.** `/opt/datasets/maestro/` exists locally:
  121GB of raw MAESTRO v3.0.0, 8.6GB of already-extracted `training_pairs/`, and — most
  importantly — `/opt/datasets/maestro/checkpoints/` already contains multiple trained
  checkpoint files (`baseline_full.bin`, `ctx2_full.bin`, `ctx4_full.bin`, etc., dated
  2026-04-13/14) that look like exactly the format `MelDecoder::save`/`load` produces. But
  `SpectralVocoder::new` (the production constructor) never calls `MelDecoder::load(...)` —
  it always re-initializes randomly. **This may be a one-line fix**, contingent on verifying
  the checkpoint format still matches the current `MelDecoder` shape.
- **A real perceptual critic already exists in this crate and isn't used where it matters
  most.** `clap_embed.rs` wraps a real pretrained CLAP ONNX audio-tower model
  (`laion/clap-htsat-unfused`) for genuine Fréchet Audio Distance, and it's wired into
  `creative_bench.rs`, `steering.rs`, and `muse_studio.rs`. But `critic.rs`'s
  `music_auto_improve` loop — the actual generate→score→mutate→regenerate optimizer — still
  scores against the shallow structural proxy (pitch-variety ratio, IOI variance) that
  `critic.rs`'s own doc comment admits "does not listen to audio." The real perceptual signal
  is one crate over; it just isn't plugged into the loop that needs it most.

## Everything else confirmed by the first review pass (holds up, no correction needed)

- **DSP/audio core is genuinely strong**: real Woodworth binaural model, real envelope-follower
  sidechain, byte-correct SMF MIDI export, doc-comment evidence of real bugs found and fixed
  (shared-delay-line stereo bleed, phase-warp artifacts, a self-corrected vocoder overclaim).
- **The symbolic-composition gate is genuinely causal**: `musical_policy::select_by_musical_policy`
  returns `None` (→ HTTP 422 in `muse_studio.rs`) when nothing qualifies — no silent fallback,
  unlike this monorepo's documented `gate_civic` disease elsewhere.
- **Grammar/structure is real** (Huron/Narmour melodic rules, genuine motif retention with
  thematic development), but note: the "grammar-society" architecture MEMORY describes lives in
  the sibling `symthaea-music-theory` crate, not in `symthaea-muse`'s `melodic_grammar.rs` —
  a naming trap for future sessions. That sibling-crate work landed as one imported-patchset
  commit, not iteratively verified in this repo's own history — worth a closer look if it
  becomes load-bearing.
- **The research/evidence apparatus (54 files, ~27.7K LOC)** is methodologically sound —
  real seeded Latin-square blinding, a correctly-gated multi-signer unblinding protocol — but
  has **never processed real listener data**, only `.example.json` templates. Scale is
  disproportionate to current evidentiary stage; legitimate infrastructure-before-need, but
  worth freezing further expansion until a real pilot runs through what already exists.
- **Prior UI fixes hold**: CORS exact-match, router-based nav, `next_piece()` race guard,
  single compose-call-site, bounded `evict_oldest_by_id` eviction — all re-verified directly
  against source, no regressions found.
- Two well-built but disconnected DSP capabilities: `spectral_vocoder.rs` and `synesthesia.rs`
  (the latter unrelated to the neural vocoder — an HDC-based color/sound mapping) — both
  real, tested, called from nowhere but their own module.

## Recommended plan, in order

**Phase 0 — same-day, no coordination needed**
1. Restart `muse_studio` with `SYMTHAEA_SOUNDFONT`/`fluidsynth` available (via `nix develop`)
   and re-listen — free quality win, already built and tested.
2. Check whether an existing `/opt/datasets/maestro/checkpoints/*.bin` is load-compatible
   with current `MelDecoder`; if yes, wire `SpectralVocoder::new` to load it instead of
   random-initializing. If incompatible, this narrows to "needs one training run," not
   "needs new infrastructure" — the data and training code already exist.
3. Wire `clap_embed`'s real CLAP/FAD score into `critic.rs`'s `music_auto_improve` loop
   (or run both scores and log the delta first, to see how much they actually disagree).

**Phase 1 — coordination needed**
4. Check in on `muse-evidence-port`'s progress on `muse/evidence-and-title-provenance`
   before touching it. If it's stalled or abandoned, merge it deliberately (fresh worktree,
   review conflicts) to recover `harmony_view`/`motifs_view`/`journey`/`player_bar`/`icons`
   rather than rebuilding them.
5. Once merged, re-run `cargo check -p symthaea-muse-ui --target wasm32-unknown-unknown` —
   expect it to be very close to green, modulo `foundry_review_page.rs`/`add_music_page.rs`.

**Phase 2 — decide, don't just build**
6. `foundry_review_page.rs`/`add_music_page.rs`: decide whether the Foundry/Add-Music
   features are still wanted; if so, scope the missing backend for real rather than leaving
   the UI half-built and silently broken.
7. `neural_melody.rs`: decide whether to invest in training its projection layer for real
   (data's already there) given it's the one neural component genuinely on the real substrate.
8. `spectral_vocoder.rs`/`mel_mlp.rs`: decide train-for-real vs. retire — it's the one neural
   path NOT on the real substrate, and currently wired nowhere.

**Phase 3 — lower urgency, real but not blocking**
9. Freeze further growth of the confirmatory/replication/pilot apparatus until a real pilot
   has actually run through the existing 54-file pipeline once.
10. Reconcile `symthaea-music-theory`'s grammar-society patchset against this crate's own
    grammar work if/when the two need to interoperate more directly.

## Not independently re-verified this pass (flag, not clean)
- `synth.rs` (1892 lines) and `instruments.rs`'s full body — only partially sampled in the
  first review pass.
- `reproducibility_attestation.rs`, `study_runner.rs` (1,671 lines, largest file in the
  evidence apparatus) — not read in comparable depth to `confirmatory_unblinding.rs`/
  `blinded_study.rs`.
- Native `cargo check -p symthaea-muse --bin muse_studio --features studio` did not finish
  inside the reviewing fork's time budget (shared 9-session CPU) — likely clean (no UI-crate
  dependency) but not empirically confirmed.
- Canvas-resize-guard regression check in `score_view.rs`/`visualization.rs` — `visualization.rs`
  is one of the currently-unwired files, so moot until Phase 1 lands it.
