# Symthaea Spore artifacts for Mycelix Pulse

These files form the on-device consciousness kernel — the
"Symthaea Edge" path in `ROADMAP.md`. Pulse's Athena (calendar
NER, recipient autocomplete, intent classification) will run
through this WASM module instead of shipping Big Tech model
weights.

| File | Tracked in git? | Source |
|------|-----------------|--------|
| `symthaea_spore.js` | yes (~58 KB) | wasm-bindgen bindings shim |
| `symthaea_spore_bg.wasm` | **no** (gitignored `*.wasm`) | 5 MB kernel, regenerate locally |
| `README.md` | yes | this file |

## Regenerating the WASM

```bash
cd /srv/luminous-dynamics/symthaea/crates/symthaea-spore
wasm-pack build --target web --out-dir www/pkg

# Copy into Pulse:
cp www/pkg/symthaea_spore_bg.wasm \
   www/pkg/symthaea_spore.js \
   /srv/luminous-dynamics/mycelix-workspace/mycelix-pulse/apps/leptos/public/spore/

# Rebuild Pulse (flush the post-build flatten if needed):
cd /srv/luminous-dynamics/mycelix-workspace/mycelix-pulse/apps/leptos
trunk build --release
cp -fr dist/public/. dist/          # flatten, in case Trunk's hook races
```

## Wiring

- `index.html` `<script id="spore-loader">` probes
  `/spore/symthaea_spore_bg.wasm` synchronously via XHR HEAD and
  sets `window.__SPORE_AVAILABLE`.
- `app.rs` `AppInner` calls
  `mycelix_leptos_core::provide_spore_bridge()` which reads the
  flag, logs `[Spore] Consciousness kernel detected`, and (in
  full integration, currently stubbed) instantiates the
  `SporeEngine` for periodic Phi computation.

## What's still TODO

`spore_bridge.rs` has six TODO items in its `spawn_local` block
for the real integration:

1. Import the WASM module via wasm-bindgen
2. Call `init()` to initialize the Spore engine
3. Set up a periodic tick (e.g., every 250ms) to run cognitive cycles
4. Read Phi from the engine and update `set_phi`
5. Map Phi to consciousness profile dimensions
6. Update `ConsciousnessState` signals

Plus the Pulse-specific calendar NER surface (`extract_intent`)
isn't yet exposed on `SporeEngine` — see the "Symthaea Edge"
architecture sketch in `ROADMAP.md` for the planned shape.
