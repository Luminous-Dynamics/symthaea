# App Intelligence & WASM Config Generation — Comprehensive Plan

## Current State: Three Fragmented Databases

| Database | Entries | Coverage | Location |
|----------|---------|----------|----------|
| `app_migration.rs` (Rust) | 151 apps | Windows/macOS → nixpkgs | symthaea-spore |
| `package_aliases.rs` (Rust) | 554 aliases | Natural language → nixpkgs | symthaea-nix |
| `APP_DB` (JavaScript) | 58 apps | Browser UI display | tab-inoculate.js |

**Problem:** JS UI shows only 38% of known apps. Three independent databases that don't sync.

## Vision: Unified App Intelligence in WASM

One canonical database that:
- Compiles to WASM → runs in the browser (no server needed)
- Powers the conversation ("I see you use Photoshop → GIMP is the NixOS alternative")
- Generates the package list in sovereign-config.nix
- Handles multiple input formats (user paste, SSH scan, browser detection)

## Detection Strategy (Honest, Permission-Based)

### Tier 1: Automatic (browser APIs, no permission needed)
- **OS detection**: `navigator.platform` → Linux/Windows/macOS
- **GPU**: WebGL `UNMASKED_RENDERER_STRING` → NVIDIA/AMD/Intel + model
- **CPU cores**: `navigator.hardwareConcurrency`
- **RAM**: `navigator.deviceMemory` (approximate)
- **Screen**: resolution + DPI
- **Fonts**: Canvas measurement → OS indicators (Consolas=Windows dev, SF Pro=macOS)

### Tier 2: User-Initiated (paste your app list)
The killer feature — user runs ONE command and pastes the result:

**Windows:**
```powershell
winget list --disable-interactivity | clip
```

**macOS:**
```bash
(ls /Applications && brew list 2>/dev/null) | pbcopy
```

**Linux (Ubuntu/Debian):**
```bash
dpkg --list | awk '/^ii/{print $2}' | xclip -selection clipboard
```

**Linux (Arch):**
```bash
pacman -Qe | xclip -selection clipboard
```

**Linux (Fedora):**
```bash
dnf list installed | tail -n+2 | awk '{print $1}' | xclip -selection clipboard
```

**Flatpak:**
```bash
flatpak list --app --columns=application
```

The portal provides a "Paste your app list" textarea. Symthaea parses it in WASM and maps everything.

### Tier 3: SSH Scan (for remote install)
Mount the target's existing OS partition read-only and scan:
- Windows: Program Files + Start Menu
- macOS: /Applications + Homebrew
- Linux: /usr/bin + Flatpak + Snap

This is the existing `scan_apps` relay action — it stays for remote installs.

## Unified Database Architecture

```rust
// Single source of truth: compiles to WASM + native
pub struct AppDatabase {
    // Core mappings: 1000+ apps across all platforms
    apps: Vec<AppEntry>,
    // Package manager format parsers
    parsers: HashMap<PackageManager, Box<dyn AppListParser>>,
    // Category bundles
    bundles: Vec<AppBundle>,
}

pub struct AppEntry {
    // Identity
    canonical_name: String,          // "adobe-photoshop"
    display_name: String,            // "Adobe Photoshop"

    // Detection: how to recognize this app
    windows_names: Vec<String>,      // ["Adobe Photoshop", "Photoshop"]
    macos_names: Vec<String>,        // ["Adobe Photoshop 2024"]
    linux_packages: Vec<String>,     // dpkg/pacman/dnf package names
    flatpak_ids: Vec<String>,        // ["com.adobe.Photoshop"]
    homebrew_names: Vec<String>,     // ["photoshop"] (if exists)
    winget_ids: Vec<String>,         // ["Adobe.Photoshop"]
    snap_names: Vec<String>,         // ["photoshop"]
    protocol_schemes: Vec<String>,   // ["photoshop://"]

    // NixOS equivalent
    nix_packages: Vec<String>,       // ["gimp", "krita"]
    match_kind: MatchKind,           // Alternative
    match_confidence: f32,           // 0.7
    migration_notes: String,         // "GIMP handles most Photoshop workflows..."

    // Classification
    category: AppCategory,
    tags: Vec<String>,               // ["creative", "image-editing", "raster"]
    is_proprietary: bool,
    has_linux_native: bool,
    wine_compatibility: Option<WineRating>,  // Gold/Silver/Bronze from WineHQ/ProtonDB
}

pub enum MatchKind {
    Exact,           // Same app in nixpkgs (firefox → firefox)
    Alternative,     // Open-source replacement (Photoshop → GIMP)
    Compatibility,   // Runs via Wine/Proton (League of Legends → lutris)
    WebApp,          // Use the web version (Google Docs)
    NativeLinux,     // Has a Linux version not in nixpkgs (use Flatpak)
    NoEquivalent,    // Nothing comparable (iMessage)
}

pub enum AppCategory {
    Browser, Editor, IDE, Terminal,
    Creative, Audio, Video, Photo,
    Office, Email, Notes,
    Communication, Social,
    Gaming, GameStore,
    Development, DevOps, Database,
    Security, VPN, Backup,
    Media, Streaming,
    System, Utility, FileManager,
}

pub struct AppBundle {
    name: String,              // "Music Production"
    description: String,       // "Everything for making music on NixOS"
    packages: Vec<String>,     // ["ardour", "lmms", "audacity", ...]
    nix_config: String,        // PipeWire + JACK + realtime config
    detected_when: Vec<String>,// ["ardour", "lmms", "ableton", "fl-studio"]
}
```

## Bundle Intelligence

When Symthaea detects a cluster of related apps, she suggests a whole bundle:

| Detected Apps | Bundle | Includes |
|--------------|--------|----------|
| Ardour, Audacity, LMMS | Music Production | PipeWire+JACK, realtime kernel config, MIDI, ardour, audacity, lmms, hydrogen |
| Photoshop, Illustrator, Lightroom | Creative Suite | gimp, inkscape, darktable, krita, scribus |
| VS Code, Docker, Git | Development | vscode, docker, git, direnv, nix-direnv, devenv |
| Steam, Discord, OBS | Gaming | steam, discord, obs-studio, gamemode, mangohud, lutris |
| Outlook, Word, Excel | Office | thunderbird, libreoffice, onlyoffice-desktopeditors |

## Migration Confidence (ProtonDB-style)

| Level | Confidence | Icon | Meaning |
|-------|-----------|------|---------|
| Native | 100% | ✓ | Exact same app in nixpkgs |
| Excellent | 85-99% | ✓ | Very close alternative, minimal workflow change |
| Good | 70-84% | ~ | Good alternative, some features differ |
| Fair | 50-69% | ~ | Workable alternative, significant differences |
| Poor | 20-49% | ⚠ | Partial alternative, major workflow changes |
| None | 0-19% | ✗ | No equivalent, consider web app or Wine |

## nixpkgs Search

**search.nixos.org Elasticsearch API:**
```
GET https://search.nixos.org/backend/latest-42-nixos-unstable/packages/_search
Content-Type: application/json

{
  "query": { "match": { "package_attr_name": "firefox" } },
  "size": 5
}
```

**For WASM (offline):** Embed top 2000 most-installed packages as a compact lookup table (~200KB JSON). This covers 95%+ of what users have installed.

## Implementation Phases

### Phase 1: Unify the database (1 session)
1. Create `crates/symthaea-nix/src/app_database.rs` — single source of truth
2. 500+ entries (merge app_migration.rs 151 + package_aliases.rs 554 + new additions)
3. Auto-generate APP_DB.js from Rust at build time
4. Add parsers for: winget, brew, dpkg, pacman, dnf, flatpak output formats

### Phase 2: WASM app matching (1 session)
1. Add `app_database` to symthaea-spore's WASM build
2. WASM binding: `match_app_list(text: &str) → MigrationReport`
3. Browser UI: "Paste your app list" textarea
4. Parse + match + show compatibility in real-time (in WASM, no server)

### Phase 3: Bundle recommendations (1 session)
1. Define 10-15 bundles (Music, Creative, Dev, Gaming, Office, Server, etc.)
2. Auto-detect bundles from matched apps
3. Each bundle generates NixOS config snippets
4. Symthaea suggests: "I see you're a musician — want the Music Production bundle?"

### Phase 4: Browser hardware → config (1 session)
1. WebGL GPU detection → NVIDIA PRIME / AMD / Intel config
2. Core count + RAM → swap/zram sizing
3. Platform detection → alongside layout suggestion
4. Combine with app matching → full config in WASM

## What This Changes

**Before:** "Boot ISO → SSH → scan → dropdown forms → install"
**After:** "Visit page → paste your apps → talk with Symthaea → download your config"

The user never needs to boot anything to get a personalized NixOS configuration. They visit the page, Symthaea detects their GPU via WebGL, they paste their app list, she generates the config. They can THEN decide how to apply it (download files, boot ISO, use relay).

This makes the installer accessible to people who are just *considering* NixOS — they can see exactly what their system would look like before committing.
