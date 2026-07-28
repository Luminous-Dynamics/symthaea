# NixForHumanity

**Install NixOS from your browser. No terminal. No USB required.**

[Install Now](https://install.nixforhumanity.org) · [Download ISO](https://github.com/Luminous-Dynamics/nixforhumanity/releases/tag/v0.1.0) · [Source](https://github.com/Luminous-Dynamics/symthaea)

---

## What is this?

NixForHumanity lets you install and manage NixOS entirely from a web browser on your phone, tablet, or laptop. Boot the target machine, open the website, and your system is configured and installed in under 5 minutes.

No command line. No manual partitioning. No 50-page wiki.

## How it works

```
Your Phone                          Target Machine
┌─────────────┐    WiFi/LAN        ┌──────────────┐
│ Browser      │◄──────────────────►│ USB Boot     │
│              │    WebSocket       │ 1.4MB Relay  │
│ Pick desktop │    (encrypted)     │              │
│ Pick apps    │                    │ Partitions   │
│ Click Install│                    │ Formats      │
│              │                    │ Installs     │
└─────────────┘                    └──────────────┘
```

1. Flash the ISO to a USB drive
2. Boot the target machine from USB
3. Open [install.nixforhumanity.org](https://install.nixforhumanity.org) on any device
4. The installer finds your target automatically — or scan the QR code
5. Choose Express (3 clicks) or Custom (full control)
6. NixOS installs. Reboot. Done.

## Express Mode

For people who just want NixOS working:

1. Pick a preset: **Developer**, **Server**, **Home/Office**, **Gaming**, or **Sovereign Workstation**
2. Set your username and password
3. Connect → Install → Reboot

Under 5 minutes, start to finish.

## Custom Mode

For people who want control:

- **7 desktop environments**: GNOME, KDE Plasma, Cosmic, Hyprland, Sway, XFCE, None
- **Security**: LUKS2 encryption, Secure Boot, TPM2 auto-unlock, FIDO2/YubiKey
- **Filesystems**: btrfs (snapshots, compression) or ZFS (RAID-Z, native encryption)
- **Shell**: Bash, Zsh (with Oh My Zsh), Fish
- **Kernel**: Default, Zen (gaming), LTS, Hardened
- **485 packages** searchable by name or meaning ("web browser" → Firefox, Chromium, Brave)
- **Home Manager** integration for dotfiles and per-user config
- **Swap, Bluetooth, Printing, Laptop mode** — all toggleable
- **Symthaea consciousness engine** and **Mycelix sovereign network** — optional add-ons
- **Config profiles** — save and reuse across machines

## No USB? No problem.

Three ways to install without a USB drive:

**From Linux** (any distro):
```bash
curl -sL https://raw.githubusercontent.com/Luminous-Dynamics/nixforhumanity/main/boot.sh | sudo bash
```

**From Windows** (via WSL2):
```
nsfw.exe sovereign start
```

**Via network** (PXE boot):
```bash
./nix/pxe-serve.sh
```

## After Install

The same web UI manages your running NixOS system:

- **Generations**: View system snapshots, one-click rollback
- **Services**: Start, stop, restart systemd services
- **Storage**: Analyze nix store, clean up old generations
- **Config**: Edit `configuration.nix` in the browser, rebuild with one click
- **System**: Full machine inventory, disk cloning, image backup/restore

## Self-Healing

When a package name changes in nixpkgs, NixForHumanity fixes it automatically:

1. **Before install**: Validates every package against the target's nixpkgs
2. **During install**: If `nixos-install` fails, parses the error, finds alternatives, retries (up to 3 times)
3. **Semantic search**: Finds packages by meaning, not just name — "password manager" → KeePassXC

## Security

- **TLS encrypted** WebSocket connection (self-signed cert, generated per boot)
- **Random auth token** generated each ISO boot (not hardcoded)
- **Constant-time** token comparison (prevents timing attacks)
- **Input validation** on all fields (no shell injection)
- **Session-isolated** log files with restricted permissions
- **Rate limiting** on failed authentication (5 attempts → blocked)
- **No SSH** — the relay executes commands directly, no network services exposed except port 8094

## Technical Details

| Component | Technology | Size |
|-----------|-----------|------|
| Browser UI | Rust + Leptos 0.8 CSR (WASM) | 1.9 MB |
| Relay | Rust + Tokio + tokio-tungstenite | 1.4 MB |
| Config gen | Rust (WASM-compatible) | Part of UI |
| ISO | NixOS 25.05 minimal + relay | 1.4 GB |
| Package DB | 111 curated + 485 aliases + 100 HDC vectors | Embedded |

**163 tests** covering config generation, input validation, package healing, semantic search, and Nix syntax validation via `nix-instantiate`.

**E2E verified** in QEMU: Boot ISO → connect → probe → partition → format → install → reboot → SSH accessible. 10/10 green.

## Comparison

| Feature | Ubuntu | Fedora | Calamares | NixForHumanity |
|---------|--------|--------|-----------|----------------|
| Browser-based | - | - | - | ✓ |
| Phone as controller | - | - | - | ✓ |
| No USB needed | - | - | - | ✓ |
| QR code connect | - | - | - | ✓ |
| Declarative config | - | - | - | ✓ |
| Rollback after install | - | - | - | ✓ |
| Self-healing packages | - | - | - | ✓ |
| Post-install management | - | - | - | ✓ |
| Disk cloning | - | - | - | ✓ |
| 10 languages | ✓ | ✓ | ✓ | ✓ |
| Express mode (3 clicks) | - | - | - | ✓ |

## Languages

English, Deutsch, Français, Español, Português, 日本語, 中文, 한국어, Русский, العربية

## Built With

- [Leptos](https://leptos.dev) — Rust web framework
- [Tokio](https://tokio.rs) — Async runtime
- [NixOS](https://nixos.org) — The operating system
- [Symthaea](https://github.com/Luminous-Dynamics/symthaea) — Consciousness-first infrastructure

## License

AGPL-3.0-or-later

---

*Built by [Luminous Dynamics](https://github.com/Luminous-Dynamics) — consciousness-first technology serving all beings.*
