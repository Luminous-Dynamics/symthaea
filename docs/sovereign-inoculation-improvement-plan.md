# Sovereign Inoculation — Improvement Plan

## Current State Assessment (2026-03-30)

### What Works (Verified E2E)
- Single disk btrfs install (QEMU, automated)
- Win11 alongside dual-boot (QEMU, verified both boot)
- WebSocket SSH relay (connect, discover_disks, install)
- Portal deployed at https://install.nixforhumanity.org

### What Exists but is Untested
- LUKS encryption layout (code written, never run)
- Secure Boot postinstall (code written, never run — needs sbctl on ISO)
- TPM2 enrollment (code written, needs swtpm + LUKS test)
- RAID1 btrfs layout (code written, needs 2-disk QEMU test)
- RAID1 mdadm layout (code written, needs 2-disk QEMU test)
- Data preservation (code written, never run)
- App scanner (code written, partially tested — mounted Windows and saw Program Files)
- Server safety detection (code written, never run on a real server)
- GPU detection (code written, never run — QEMU has no real GPU)
- WiFi scanning (code written, never run — QEMU uses virtio-net)
- DE/locale/GPU config generation (Rust function written, never wired into install scripts)

### Critical Gap: `generate_system_config()` is NOT used
The `generate_system_config()` function exists in Rust but the install scripts still hardcode the NixOS configuration.nix. The DE picker, GPU driver, timezone, and keyboard selections in the portal UI are collected but **never inserted into the actual config**. This is the #1 bug.

### Code Quality Concerns
1. **ssh_relay.rs is 2454 lines** — one massive file with install scripts as embedded strings. Should be split into modules.
2. **Install scripts are shell heredocs in Rust** — fragile, hard to test, variable escaping issues (the awk $1 bug).
3. **No unit tests** for any relay action or install script.
4. **Portal JS is one IIFE** (1459 lines) — should be split into modules with the build-portal.sh bundling.
5. **Hardcoded NixOS stateVersion** ("24.11") — should detect from the ISO.
6. **Passphrase sent in plaintext** over WebSocket — acceptable for localhost, dangerous for remote.

---

## Improvement Plan

### Phase 1: Make What We Have Actually Work (1-2 sessions)

**Goal**: Every feature we claim to have should be tested and verified.

#### 1.1 Wire `generate_system_config()` into install scripts
- [ ] The `single` layout's configuration.nix must use the DE, GPU, timezone, keyboard values
- [ ] The `alongside` layout too
- [ ] The `single-luks` layout too
- [ ] Test: install with GNOME selected → boots into GNOME
- [ ] Test: install with Hyprland → boots into Hyprland

#### 1.2 Test LUKS encrypted layout
- [ ] QEMU test: single-luks on fresh disk
- [ ] Verify: boots, asks for passphrase, decrypts, reaches login
- [ ] Verify: NixOS config has correct `boot.initrd.luks.devices`

#### 1.3 Test RAID1 layouts
- [ ] QEMU 2-disk VM script (extend test-vm-dual-nvme.sh)
- [ ] Test raid1-btrfs: both disks used, btrfs shows raid1 profile
- [ ] Test raid1-mdadm: /proc/mdstat shows active array
- [ ] Verify: remove one disk, system still boots (degraded)

#### 1.4 Test data preservation
- [ ] Spin up a VM with docker + postgres running
- [ ] Run preserve_data action
- [ ] Verify: docker images saved, pg_dump created, /etc backed up

#### 1.5 Fix the alongside layout awk bug
- [ ] The awk `$1` escaping fails in SSH heredoc with `set -u`
- [ ] Fix: use `set -eo pipefail` instead of `set -euo pipefail` in all install scripts
- [ ] Or: rewrite awk commands to avoid `$` in heredocs

#### 1.6 Test the full portal flow locally
- [ ] Start ssh-relay, connect from portal in browser
- [ ] Verify: hardware probe shows results in UI
- [ ] Verify: app scan shows compatibility table
- [ ] Verify: disk selector shows drives
- [ ] Verify: layout selector works
- [ ] Verify: config options (DE, GPU, etc.) appear
- [ ] Verify: deploy button sends correct install command

### Phase 2: Production Quality (2-3 sessions)

#### 2.1 Split ssh_relay.rs into modules
```
src/bin/ssh_relay.rs          → main(), WebSocket handler, routing
src/relay/mod.rs              → pub mod
src/relay/install_scripts.rs  → generate_install_script(), all layouts
src/relay/hardware_probe.rs   → probe_hardware script
src/relay/app_scanner.rs      → scan_apps script
src/relay/safety.rs           → server detection script
src/relay/data_preserve.rs    → preserve_data script
src/relay/config_gen.rs       → generate_system_config(), NixOS config templates
src/relay/postinstall.rs      → git_init, secure_boot, tpm2 postinstall
```

#### 2.2 NixOS config as templates, not heredocs
- Move configuration.nix templates to `templates/` directory
- Use string interpolation with clear markers: `{{hostname}}`, `{{timezone}}`
- Separate the config generation from the install script
- Makes it testable: generate config → validate with `nix eval`

#### 2.3 Automated test suite
```
tests/
  test_single_disk.sh         — QEMU: fresh disk → single layout → boot
  test_single_luks.sh         — QEMU: LUKS layout → passphrase unlock → boot
  test_alongside_windows.sh   — QEMU: Win11 disk → alongside → both boot
  test_raid1_btrfs.sh         — QEMU: 2 disks → raid1 → boot → degrade
  test_hardware_probe.sh      — QEMU: verify probe JSON is valid
  test_app_scanner.sh         — QEMU: Win11 disk → scan → verify apps found
  test_safety_detection.sh    — QEMU: running services → verify blocked
  test_config_generation.sh   — Unit: config generation → nix eval validates
```

#### 2.4 Security hardening
- [ ] WebSocket relay: add token-based auth (generate token on connect, validate on subsequent messages)
- [ ] Passphrase: never log, never echo, clear from memory after use
- [ ] Rate limiting: already has 1 session per IP, add brute-force protection
- [ ] Document the threat model: "relay runs locally, same-machine-only by default"
- [ ] Option for mTLS when relay is remote

#### 2.5 Error handling & recovery
- [ ] If nixos-install fails: show clear error, suggest fixes
- [ ] If partitioning fails: don't leave half-formatted disk
- [ ] If WebSocket drops mid-install: install continues (already works), relay can reconnect
- [ ] Timeout handling: don't block forever waiting for nixos-install

### Phase 3: Differentiating Features (3-5 sessions)

#### 3.1 Flake generator
- [ ] Generate complete flake.nix from user choices
- [ ] Include: nixpkgs, home-manager, disko, lanzaboote (if Secure Boot), nixos-hardware (if detected)
- [ ] Generate disko-config.nix (declarative partitioning)
- [ ] Generate home.nix (basic home-manager config)
- [ ] Validate with `nix flake check` before install

#### 3.2 Recovery partition
- [ ] 2GB ext4 partition with minimal NixOS + installer
- [ ] GRUB/systemd-boot entry: "NixOS Recovery"
- [ ] Contains: terminal, network, btrfs-progs, nixos-rebuild, the SI portal
- [ ] "Break glass" emergency SSH

#### 3.3 Post-install onboarding
- [ ] Day 1 checklist served on localhost:5491 (Luminous Nix port)
- [ ] Items: update system, set up backups, import keys, configure git, install tools
- [ ] Progressive NixOS tutorial: edit config → rebuild → see result
- [ ] Context-aware tips (optional, non-intrusive)

#### 3.4 Desktop preview gallery
- [ ] Pre-rendered screenshots for each DE (GNOME, KDE, Hyprland, Sway, XFCE)
- [ ] Show in portal during DE selection
- [ ] ~50KB WebP each, ~250KB total

#### 3.5 Dotfile migration
- [ ] Scan .bashrc, .gitconfig, .vimrc, .ssh/config on mounted old OS
- [ ] Generate home-manager equivalents
- [ ] Show diff: "Your .gitconfig → programs.git in home-manager"
- [ ] User reviews and approves before including

#### 3.6 Hardware compatibility database
- [ ] Opt-in hardware probe submission after successful install
- [ ] API: POST /api/v1/probe (anonymized PCI/USB IDs, kernel, driver status)
- [ ] API: GET /api/v1/check?pci=10de:2503 (pre-install compatibility query)
- [ ] Simple database (SQLite or Supabase)

### Phase 4: Innovation (5+ sessions)

#### 4.1 Local LLM config assistant
- Bundle gemma3:1b on the ISO (~1.5GB)
- "Describe your use case" → generate NixOS config
- Contextual help during each step
- Hardware-aware recommendations

#### 4.2 Docker Compose → NixOS conversion
- Parse docker-compose.yml from old system
- Generate `virtualisation.oci-containers` NixOS config
- Handle volumes, networks, env vars

#### 4.3 Air-gapped install
- Bundle configurable nix cache on ISO
- Auto-detect offline mode, use local substituter
- Profile-based: "developer" (~5GB), "server" (~2GB), "desktop" (~8GB)

#### 4.4 ARM64 / Raspberry Pi
- Platform detection (device tree)
- U-Boot vs UEFI bootloader selection
- Pi-specific kernel and GPU config
- Lighter DE recommendations for Pi 4

---

## Priority Matrix

| Feature | Impact | Effort | Priority |
|---------|--------|--------|----------|
| Wire config generation into installs | Critical | Small | **NOW** |
| Test LUKS layout | Critical | Small | **NOW** |
| Fix awk escaping in alongside | Critical | Tiny | **NOW** |
| Test full portal flow | Critical | Medium | **NOW** |
| Test RAID layouts | High | Medium | Phase 1 |
| Split ssh_relay.rs | High | Medium | Phase 2 |
| Automated QEMU test suite | High | Medium | Phase 2 |
| Security hardening (relay auth) | High | Small | Phase 2 |
| Flake generator | High | Large | Phase 3 |
| Recovery partition | Medium | Medium | Phase 3 |
| Post-install onboarding | Medium | Medium | Phase 3 |
| Desktop preview gallery | Medium | Small | Phase 3 |
| Dotfile migration | Medium | Medium | Phase 3 |
| LLM config assistant | High | Large | Phase 4 |
| ARM64 support | Medium | Medium | Phase 4 |
| Air-gapped install | Medium | Medium | Phase 4 |

---

## What To Do Next Session

1. **Wire `generate_system_config()` into all install layouts** — this is the #1 bug. Users pick GNOME but get CLI-only.
2. **Test LUKS in QEMU** — fresh disk, encrypted install, verify boot.
3. **Fix awk $1 escaping** — use `set -eo pipefail` in all scripts.
4. **Run full portal flow test** — browser → connect → probe → scan → select → deploy.
5. **Update launch posts** with the custom domain and new features.

## Security Priorities (from research)

1. **Bind relay to 127.0.0.1 ONLY** — verify no `0.0.0.0` bindings
2. **Add session token auth** — random 32-byte token displayed on boot, required for WebSocket
3. **Zeroize passphrases** — use `zeroize` crate, never log, never echo, pipe to cryptsetup stdin
4. **Input validation** — hostname `^[a-z][a-z0-9-]{0,62}$`, username POSIX rules, disk paths must start with `/dev/`
5. **Pre-release**: `cargo audit`, `cargo deny`, fuzz the partition planner with proptest

## Architecture Decisions (from research)

| Decision | Choice | Why |
|----------|--------|-----|
| Templating | **Tera** | Jinja2-like, handles Nix syntax, Rust-native |
| Recovery partition | **2GB ext4, minimal NixOS + installer** | Self-contained, boots independently |
| Testing | **QEMU + proptest + headless browser** | E2E + property-based partition safety |
| Security model | **Localhost + session token + zeroize** | Simple, correct |
| Flake structure | **hosts/modules/home/ with specialArgs** | Community standard |
| Progress | **Bytes-based with phase weighting** | Smooth, no stalling at 50% |
| Onboarding | **Mint-style checklist + contextual tips** | Non-intrusive, genuinely useful |
