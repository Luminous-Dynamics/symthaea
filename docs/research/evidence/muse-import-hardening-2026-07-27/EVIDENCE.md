# Symbolic import: authorization, persistence, concurrency, audition — E2E evidence

**Verdict: PASS (operationally closed).** Verified through live isolated HTTP
execution against the real `muse_studio` binary with real FluidSynth
rendering. UI radio interaction and disclosure presentation were
compile-verified and source-verified but **not** exercised through browser
automation (no headless-browser tool was available in this session) — that
is a disclosed limitation, not evidence of a defect.

Covers the two fix commits on `review/symthaea-bridges-security`:
- `e38759449e` — `AuthorizationBasis`, spawn_blocking, reimport-timestamp
  fix, FluidSynth audition.
- `62663f2611` — `LegacyUnspecified`, temp-file/staging nonce fix + race-loser
  recovery, `instrumentation_note` disclosure.

## Environment

- Binary: `symthaea-muse` `muse_studio` bin, built with `--features
  theory,studio`, debug profile, from the exact commit `62663f2611`.
- Isolated instance: bound to `127.0.0.1:18400` (temporary 1-line local edit
  to the hardcoded `:8400` bind, reverted via `git checkout --` immediately
  after the run — confirmed zero diff afterward) with its own scratch
  working directory (`data/music/imports` created fresh under
  `/tmp/.../scratchpad/muse_e2e/`), so it never touched the real dev
  instance already running on `:8400` (verified via `ss -ltnp` before
  starting: PID `3150994`, untouched throughout, confirmed still alive
  afterward).
- FluidSynth: resolved via `nix build --no-link --print-out-paths
  nixpkgs#fluidsynth nixpkgs#soundfont-fluid` (not `nix develop`, which has
  been unreliable under this host's concurrent-session load this session)
  → `SYMTHAEA_FLUIDSYNTH=/nix/store/kbpv79q9aqbz4xm9df81w6hwxk8fkypz-fluidsynth-2.5.5/bin/fluidsynth`,
  `SYMTHAEA_SOUNDFONT=/nix/store/jhscm2bcp5g92dabkp3sz56mynvq03f8-Fluid-3/share/soundfonts/FluidR3_GM2-2.sf2`.
  Confirmed via the server's own startup log:
  `Renderer: FluidSynth (.../fluidsynth) + soundfont (.../FluidR3_GM2-2.sf2)`.
- Test uploads: 4 real MIDI files from the earlier Baroque harmonic-syntax
  pilot (`baroque_new_functional_walk_seed{1,7}.mid`,
  `baroque_old_fixed_i_iv_v_i_seed{1,7}.mid`), used only as convenient
  real symbolic content already on disk — unrelated to their harmonic
  content, just needed real parseable MIDI.

## Test matrix (live `curl` against `POST /api/music/import`)

| Case | Expected | Result |
|---|---|---|
| `authorization_basis=own_work` | 200; `declared_authorship: true` | ✅ `work_id b54d5e84...`, `authorization_basis: "own_work"`, `declared_authorship: true` |
| `authorization_basis=authorized_import` (different file) | 200; `declared_authorship: false` | ✅ `work_id f6a075bc...`, `authorization_basis: "authorized_import"`, `declared_authorship: false`, `audio_renderer: "fluidsynth"`, `instrumentation_note` present with the reconstructed-palette disclosure text |
| neither field | 400 | ✅ `"authorship or import authorization must be confirmed"` |
| invalid value (`authorization_basis=bogus_value`) | 400 | ✅ same message |
| legacy `authorized=true` (old-client compat) | 200; mapped to `authorized_import`, never ownership | ✅ `authorization_basis: "authorized_import"`, `declared_authorship: false` |
| reimport: identical file+title+contributor+basis submitted twice | identical `work_id` and identical `imported_at_unix_ms` | ✅ both fields byte-identical across the two responses (`imported_at_unix_ms: 1785176670523` both times); zero staging (`.{work_id}.tmp-*`) residue after |
| two **truly concurrent** imports of two **different** works (backgrounded shell jobs, `wait`) | distinct `work_id`s, no collision | ✅ `work_id 3f68f0cd...` / `3d7cf015...`, both HTTP 200, both `audio_renderer: "fluidsynth"`, zero staging residue |

**Not tested** (correctly flagged by the reviewer as a distinct case from
the above): two concurrent requests racing to persist the exact **same**
`work_id` (i.e. a genuine double-submit of the identical file+title+
contributor+basis). The nonce fix and race-loser-reads-the-winner's-record
logic in `62663f2611` address this path in code, but it was not exercised
live this session. Tracked as backlog item #2 below.

## Audition WAV checks

All 4 fetched via `GET /api/music/import/{work_id}/audio`, all HTTP 200,
all valid `RIFF....WAVE` headers (`xxd` on the first 12 bytes), all
26,505,004 bytes (same duration/format across these particular source
files — expected, not a bug: FluidSynth WAV size here is driven by
duration/sample-rate/channel-count, not note content).

Distinct-content confirmation (no cross-contamination from the temp-file
nonce fix):

```
own_work import:         e3161fc24a0d6c867ba4da9ca3e4cfb8ebf5d71ca771288349673ae92e9cdd25
authorized_import:       3e736a8b8841cc14ed081ecf1d9cc1ccec92357ad8b2affc4c38576c97965ddd
concurrent import A:     4e5aa08dd5492459ef1325038685fa81c4e1722f76b55f9a61d12646a662b0af
concurrent import B:     44a1ce93d4bb264d813a6a878170fc08d21a082e97c7f6095be30409fc99c10b
```

All four checksums distinct — no request's audition audio leaked into
another's response, including under genuine concurrency.

## Staging-residue checks

`find data/music/imports -maxdepth 1 -name ".*"` returned empty after every
single import, every reimport, and the concurrent pair — no leftover
`.{work_id}.tmp-*` directories at any point during the run.

## Known limitation (disclosed, not a defect)

No headless-browser tool was available in this session. Boundaries not
crossed by this evidence:
1. Actual radio-button click → `FormData` construction in the browser.
2. Rendering of `instrumentation_note`/renderer disclosure in the live DOM.

Both are supported by (a) a clean `cargo check --target wasm32-unknown-unknown`
of `symthaea-muse-ui` against this exact code, and (b) direct inspection
confirming `api.rs`'s `FormData` field names (`file`, `title`, `contributor`,
`authorization_basis`) match exactly what the `curl` requests above sent —
but this is inference from source, not an observed browser run.

## Backlog (explicitly deferred, not silently dropped)

1. Multipart integration-test table (the matrix above, as real
   `#[tokio::test]`s against the axum router, not ad hoc `curl`).
2. Same-`work_id` concurrency race test (two requests for the identical
   content racing to persist — distinct from the different-works
   concurrency case this evidence run covers).
3. Content identity vs. import-record identity design (should a
   title/contributor typo mint a new `work_id` for identical source
   bytes?).
4. License and authorization-authority UX (`content_license` selector in
   the Studio UI; clarifying that import-authorization ≠ license-setting
   authority).
5. Browser-level import smoke test (closes the one disclosed gap above).
