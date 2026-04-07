# Leptos NixOS Installer Plan

## Discovery: We Already Have It

`symthaea-web` (crates/symthaea-web/) is a Leptos 0.8 CSR app that:
- Has an Inoculate page (`src/pages/inoculate.rs`, 12.6KB)
- Talks to SporeEngine via Web Worker
- Already calls `generate_flake()`, `generate_disko_config()`, `generate_hardware_nix()`
- Builds to 943KB WASM + 1.1MB Spore WASM = ~500KB gzipped total
- Uses Trunk for building
- Has reactive state, components, routing

## What Needs to Change

### Instead of building from scratch, extend symthaea-web:

1. **New page: `src/pages/installer.rs`** — The clean NixOS installer
   - Hardware detection (WebGL GPU, browser APIs)
   - "Paste your apps" textarea → parse in Rust (not JS)
   - Conversation with Symthaea (reactive signals, not DOM manipulation)
   - Config preview with reasoning
   - Connect to relay for deployment
   - All type-safe, all Rust

2. **New component: `src/components/app_matcher.rs`**
   - Import `symthaea_nix::app_database::AppDatabase` (the WASM-safe parts)
   - OR: embed the app matching logic directly (it's pure Rust, no OS deps)
   - Show migration report reactively

3. **New component: `src/components/config_preview.rs`**
   - Show generated Nix code with syntax highlighting
   - Expandable reasoning per decision
   - "Swap" buttons for alternatives

4. **Modified: `src/worker.rs`**
   - Add `sovereign_chat` action to worker protocol
   - Bridge conversation to SporeEngine or separate handler

5. **Route: `/install`**
   - Maps to the installer page
   - Default route for install.nixforhumanity.org

## Architecture

```
Browser
├── symthaea-web (Leptos 0.8 CSR, 943KB WASM)
│   ├── /install route → InstallerPage component
│   │   ├── HardwareDetect (WebGL, browser APIs)
│   │   ├── AppPaste (textarea → AppDatabase::parse + match)
│   │   ├── Conversation (reactive chat with Symthaea)
│   │   ├── ConfigPreview (generated Nix + reasoning)
│   │   └── DeployPanel (connect relay → install)
│   └── Other routes (chat, topology, dreams — consciousness demo)
│
└── Web Worker (spore-worker.js)
    └── SporeEngine (1.1MB WASM)
        ├── Consciousness cycles
        ├── Flake generation
        └── Conversation (via sovereign_chat action)
```

## What Goes Where

### In Leptos WASM (client-side, no server):
- Hardware detection (WebGL, navigator APIs)
- App list parsing (AppDatabase::parse_app_list — pure Rust)
- App matching (AppDatabase::match_list — pure Rust)
- Migration report rendering (reactive Leptos components)
- Config preview display
- UI state management

### In Spore Worker (off-thread):
- SovereignConversation (chat with Symthaea)
- SovereignConfigGenerator (generate NixOS config)
- Consciousness cycles (for ceremony)
- Broca text generation (for personalized welcome)

### On Backend (SSH relay, only for deployment):
- SSH connection to target
- Hardware probe via lspci/lsblk (deep, OS-level)
- Install script execution
- nixos-install

## Build

```bash
# Build Spore WASM (consciousness + config gen)
./crates/symthaea-spore/build-wasm.sh

# Build Leptos frontend
cd crates/symthaea-web
trunk build --release

# Copy Spore WASM into Leptos dist
cp ../symthaea-spore/www/pkg/* dist/assets/pkg/

# Deploy
# dist/ has everything — serve statically
```

## What This Fixes

| Problem | JS Approach | Leptos Approach |
|---------|-------------|-----------------|
| Can't see rendering issues | Invisible to me | Compiler catches type errors |
| DOM manipulation bugs | String templates | Reactive signals |
| Event wiring | Manual addEventListener | Declarative `on:click` |
| State management | Global vars, race conditions | Leptos signals, predictable |
| Testing | Can't test UI | Rust tests for components |
| Hardware detection | Separate JS function | Leptos component with web-sys |
| App matching | JS reimplementation of Rust | Direct Rust call in WASM |

## Migration from Current JS Portal

The JS portal (`www/portal.html`, `www/installer.html`) becomes legacy. The Leptos app replaces it entirely. We keep:
- `spore-worker.js` (unchanged, bridges to SporeEngine WASM)
- `ceremony.js` (port to Leptos component)
- `constellation.js` (port to Leptos canvas component)

We remove:
- All tab-*.js files
- portal-shell.html, installer-shell.html
- build-portal.sh, build-installer.sh
- css/portal.css, css/installer.css
- app.js

## Timeline

| Phase | Work | Sessions |
|-------|------|----------|
| 1 | Add `/install` route to symthaea-web with hardware detection | 1 |
| 2 | Port AppDatabase to Leptos component (paste → match → report) | 1 |
| 3 | Wire conversation to worker (sovereign_chat action) | 1 |
| 4 | Config preview + deploy panel components | 1 |
| 5 | Deploy to install.nixforhumanity.org, retire JS portal | 1 |
