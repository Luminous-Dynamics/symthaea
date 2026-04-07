# Symthaea WASM Config Generation — Architecture Plan

## The Insight

Symthaea already runs in the browser as a 1.1MB WASM kernel (symthaea-spore). She can think, reason, generate text, and analyze consciousness — all client-side.

The `SovereignConfigGenerator` and `SovereignConversation` use only:
- `symthaea-core` HDC vectors (already in WASM)
- `NixCodebook` (pure computation, no OS calls)
- `NixActiveInference` (pure computation)
- `NixCausalGraph` (170 patterns, pure data)
- `UserInputEncoder` (text → HDC, pure)
- No `std::process::Command`, no `std::fs` in the config/conversation modules

**This means the config generator CAN run in WASM.** Symthaea can generate your NixOS configuration entirely in the browser, with no server needed.

## What This Enables

### Two-Mode Architecture

**Mode 1: Browser-Only (installing on THIS machine)**
```
User visits install.nixforhumanity.org
  ↓
WASM detects hardware via browser APIs:
  - WebGL → GPU vendor + model (e.g., "NVIDIA GeForce RTX 2070")
  - navigator.hardwareConcurrency → CPU cores
  - navigator.deviceMemory → RAM
  - navigator.storage.estimate() → disk space
  - WebGPU → GPU features (if available)
  - navigator.connection → network type
  ↓
User chats with Symthaea (conversation runs in WASM)
  ↓
Symthaea generates NixOS config (reasoning runs in WASM)
  ↓
User downloads:
  - Custom ISO with their config baked in, OR
  - configuration.nix + flake.nix files to use with any ISO
```

No relay. No SSH. No server. Everything happens in the browser.

**Mode 2: Remote Install (installing on ANOTHER machine)**
```
User visits install.nixforhumanity.org
  ↓
Connects to target via SSH relay (existing flow)
  ↓
Relay probes target hardware (lspci, lsblk, etc.)
  ↓
Hardware data sent to browser → WASM generates config
  ↓
Config sent back to target via relay → nixos-install
```

The config generation STILL runs in WASM — only the hardware probe and install execution need the relay.

### The User Experience

**Visiting install.nixforhumanity.org on any device:**

1. Page loads. WASM initializes (~1s).
2. Symthaea immediately detects the browser's hardware via WebGL/WebGPU.
3. She greets: "I see you're on an NVIDIA RTX 2070 with 16GB RAM. Want to install NixOS on this machine?"
4. User says "Yes, I do music production and Rust development"
5. Symthaea generates config in WASM (~0.5s):
   - NVIDIA drivers with PRIME offload (detected hybrid GPU)
   - PipeWire + JACK (music production)
   - Rust toolchain (development)
   - GNOME (best NVIDIA Wayland support)
6. Shows config preview with reasoning
7. User clicks "Download" → gets either:
   - A `sovereign-config.tar.gz` with flake.nix + configuration.nix + disko.nix
   - Or a link to build a custom ISO with their config

**No relay needed for step 1-6.** The relay only enters the picture if they want automated remote installation.

## Browser Hardware Detection

| Data | Browser API | Accuracy | NixOS Config Impact |
|------|------------|----------|-------------------|
| GPU vendor + model | WebGL `UNMASKED_RENDERER_STRING` | High | `hardware.nvidia`, `services.xserver.videoDrivers` |
| GPU hybrid | WebGL vendor ≠ renderer vendor | Medium | `hardware.nvidia.prime` |
| CPU cores | `navigator.hardwareConcurrency` | Exact | Thread count for builds |
| RAM (approx) | `navigator.deviceMemory` | Low (capped at 8GB) | Swap size, zram config |
| Disk space | `navigator.storage.estimate()` | Approximate | Layout recommendations |
| OS platform | `navigator.platform` | Exact | Alongside detection |
| Screen | `screen.width/height`, `devicePixelRatio` | Exact | HiDPI config |
| Battery | `navigator.getBattery()` | Exact | Power management config |
| Network | `navigator.connection` | Varies | WiFi vs ethernet |
| Audio | `navigator.mediaDevices` | Exact | Audio device config |

**What we CAN'T detect from browser (needs SSH relay for remote):**
- Disk partitions and layout (lsblk)
- EFI/Secure Boot status
- TPM presence
- Existing OS installations
- Installed applications (for migration)
- Exact disk model/serial

## Implementation Plan

### Phase 1: Add config gen to WASM kernel

1. Add `sovereign_config` and `sovereign_conversation` modules to symthaea-spore
2. Gate behind `wasm` feature (no filesystem ops needed)
3. Add WASM bindings:
   ```rust
   #[wasm_bindgen]
   pub fn sovereign_chat(&mut self, message: &str) -> JsValue {
       // Returns: { message, is_question, config_preview, decisions, ready_to_deploy }
   }

   #[wasm_bindgen]
   pub fn set_hardware_profile(&mut self, profile: JsValue) {
       // Receives browser-detected hardware
   }

   #[wasm_bindgen]
   pub fn generate_config(&mut self) -> String {
       // Returns complete sovereign-config.nix
   }

   #[wasm_bindgen]
   pub fn download_config_bundle(&self) -> Vec<u8> {
       // Returns tar.gz of flake.nix + configuration.nix + disko.nix
   }
   ```

4. Browser JS detects hardware, passes to WASM, gets config back

### Phase 2: Browser hardware detection

```javascript
async function detectHardware() {
    const hw = {};

    // GPU via WebGL
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
    if (gl) {
        const ext = gl.getExtension('WEBGL_debug_renderer_info');
        if (ext) {
            hw.gpu_renderer = gl.getParameter(ext.UNMASKED_RENDERER_STRING_WEBGL);
            hw.gpu_vendor = gl.getParameter(ext.UNMASKED_VENDOR_STRING_WEBGL);
            // Parse: "NVIDIA GeForce RTX 2070" → vendor=nvidia, model=RTX 2070
        }
    }

    // GPU via WebGPU (more detailed)
    if (navigator.gpu) {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) {
            const info = await adapter.requestAdapterInfo();
            hw.webgpu_vendor = info.vendor;
            hw.webgpu_device = info.device;
            hw.webgpu_architecture = info.architecture;
        }
    }

    hw.cpu_cores = navigator.hardwareConcurrency;
    hw.memory_gb = navigator.deviceMemory;  // approximate, capped
    hw.platform = navigator.platform;

    // Detect if this is the machine to install on
    hw.is_linux = /Linux/.test(navigator.platform);
    hw.is_windows = /Win/.test(navigator.platform);
    hw.is_mac = /Mac/.test(navigator.platform);

    return hw;
}
```

### Phase 3: Config download

Instead of requiring a relay, offer to download the generated config:

```javascript
function downloadConfig(config) {
    const files = {
        'configuration.nix': config.configuration_nix,
        'sovereign-config.nix': config.sovereign_config_nix,
        'hardware-notes.md': config.hardware_notes,
    };

    // Create a zip/tar.gz in browser
    // User downloads, then:
    // 1. Boot any NixOS ISO
    // 2. Copy files to /mnt/etc/nixos/
    // 3. nixos-install

    // Or: generate a nixos-install one-liner they can paste
    const oneLiner = `curl -sL https://install.nixforhumanity.org/config/${configId} | tar xz -C /mnt/etc/nixos/ && nixos-install`;
}
```

### Phase 4: One-click ISO generation (future)

The ultimate: user's choices → custom ISO built on our server → download → flash → boot → done.

This requires a build server running `nix build` with the user's config. Could be:
- A GitHub Action triggered by the portal
- A build server at build.nixforhumanity.org
- Hydra/Cachix integration

## What Changes About the Landing Page

```
BEFORE:
  "Boot ISO → Connect SSH → Talk → Install"
  (4 steps, requires relay)

AFTER:
  "Visit page → Talk with Symthaea → Download your config"
  (3 steps, NO relay needed)

  OR for automated install:
  "Visit page → Talk with Symthaea → Connect to target → Install"
  (4 steps, relay for the install part only)
```

The page becomes:
1. **Hero**: "NixOS, configured for you"
2. **Symthaea greets you** (WASM detects your GPU, shows it)
3. **Chat** about what you need
4. **Download** your config OR **Connect** for automated install
5. **Guides** for what to do with the config files

## File Changes

### symthaea-spore changes:
- Add `sovereign_config` + `sovereign_conversation` as optional modules (feature-gated)
- Add WASM bindings for chat + config gen
- Browser hardware detection in JS

### New WASM exports:
- `sovereign_chat(message) → response`
- `set_hardware_from_browser(profile)`
- `generate_sovereign_config() → nix code`
- `get_config_decisions() → reasoning`

### Dependencies:
- `sovereign_config` needs `NixCodebook` → needs `symthaea-core` HDC (already in WASM)
- `NixCausalGraph` → pure data, WASM-safe
- `NixActiveInference` → pure computation, WASM-safe
- `tree-sitter-nix` → CAN compile to WASM (tree-sitter supports it)
- Only the `save/load` methods in causal_graph use `std::fs` → gate behind `#[cfg(not(target_arch = "wasm32"))]`

## Size Impact

Current WASM: 1.1MB
Adding config gen: estimated +200-400KB (codebook vectors + causal patterns)
Total: ~1.3-1.5MB — still very reasonable for a web app
