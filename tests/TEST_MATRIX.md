# Sovereign Inoculation — Test Matrix

## Test Coverage Status

### Relay Actions

| Action | Source Code | Binary Built | E2E Tested | QEMU Verified |
|--------|:----------:|:------------:|:----------:|:-------------:|
| connect | YES | OLD BINARY | YES | YES |
| disconnect | YES | OLD BINARY | YES | YES |
| discover_disks | YES | OLD BINARY | YES | YES |
| exec | YES | OLD BINARY | YES | YES |
| install (single) | YES | OLD BINARY | YES | **YES** (3 times) |
| install (single-luks) | YES | NO | YES (direct SSH) | **YES** |
| install (alongside) | YES | NO | YES (direct SSH) | **YES** (Win11) |
| install (dual) | YES | OLD BINARY | NO | NO |
| install (raid1-btrfs) | YES | NO | NO | NO |
| install (raid1-mdadm) | YES | NO | NO | NO |
| install (sata) | YES | OLD BINARY | NO | NO |
| install (vps) | YES | OLD BINARY | NO | NO |
| probe_hardware | YES | **NO** | NO | NO |
| deep_scan | YES | **NO** | NO | NO |
| scan_apps | YES | **NO** | NO | NO |
| preserve_data | YES | **NO** | NO | NO |

### Config Generation

| Feature | Source Code | Wired In | Tested |
|---------|:----------:|:--------:|:------:|
| generate_system_config() | YES | YES (system_config_patch) | NO |
| DE selection (GNOME/KDE/Hyprland/Sway/XFCE) | YES | YES | NO (binary not rebuilt) |
| GPU driver (NVIDIA/AMD/Intel) | YES | YES | NO |
| Timezone/locale/keyboard | YES | YES | NO |
| PipeWire audio | YES | YES | NO |
| Flakes enabled | YES | YES | NO |
| Secure Boot postinstall | YES | YES | NO |
| TPM2 enrollment | YES | YES | NO |
| Git-init config | YES | YES | NO |
| Disk snapshot | YES | YES | NO |

### Portal UI

| Component | Built | Works on Desktop | Works on Mobile |
|-----------|:-----:|:----------------:|:---------------:|
| Landing page / hero | YES | Untested live | Untested |
| Quick-start guide | YES | Untested live | Untested |
| Tab navigation | YES | YES | Untested |
| SSH connection panel | YES | YES | Untested |
| Relay URL input | YES | Untested | Untested |
| Hardware probe display | YES | Untested (needs new binary) | Untested |
| App compatibility table | YES | Untested (needs new binary) | Untested |
| Deep scan display | YES | Untested (needs new binary) | Untested |
| Disk selector | YES | YES | Untested |
| Layout selector | YES | Untested | Untested |
| Config panel (DE/GPU/timezone) | YES | Untested | Untested |
| Secure Boot toggle | YES | Untested | Untested |
| TPM2 toggle | YES | Untested | Untested |
| Constellation visualization | YES | Untested | Untested |
| Personalized welcome | YES | Untested | Untested |
| System Card | YES | Untested | Untested |
| Safety detection UI | YES | Untested (needs new binary) | Untested |
| Data preservation UI | YES | Untested (needs new binary) | Untested |

### Persona Tests (Offline Validation)

| Persona | Config | Welcome | Safety | Status |
|---------|:------:|:-------:|:------:|:------:|
| Maya (student) | PASS | PASS | PASS | **13/13** |
| Kai (Rust dev) | PASS | PASS | PASS | |
| Jordan (gamer) | PASS | PASS | PASS | |
| River (musician) | PASS | PASS | PASS | |
| Sam (sysadmin) | PASS | PASS | PASS | |
| Alex (privacy) | PASS | PASS | PASS | |
| Pat (teacher) | PASS | PASS | PASS | |
| Robin (DevOps) | PASS | PASS | PASS | |
| Sage (Pi) | PASS | PASS | PASS | |
| Dakota (ARM64) | PASS | PASS | PASS | |
| Casey (Chromebook) | PASS | PASS | PASS | |
| Morgan (accessibility) | PASS | PASS | PASS | |
| Avery (data scientist) | PASS | PASS | PASS | |

### QEMU Install Verification

| Layout | VM Config | Install | Boot | Status |
|--------|-----------|:-------:|:----:|:------:|
| Single disk btrfs | 1×256GB NVMe | **YES** | Verified* | **PASS** |
| Win11 alongside | 1×200GB + Win11 | **YES** | Both OSes | **PASS** |
| LUKS encrypted | 1×64GB | **YES** | Not rebooted** | **PASS** |
| Dual NVMe | 2×256GB NVMe | Not tested | - | PENDING |
| RAID1 btrfs | 2 disks | Not tested | - | PENDING |
| RAID1 mdadm | 2 disks | Not tested | - | PENDING |
| SATA | 1 disk | Not tested | - | PENDING |
| VPS | 1×80GB | Not tested | - | PENDING |

*Boot verified = rebooted into installed system and confirmed login
**LUKS install completed successfully but VM not rebooted (can't enter passphrase headless)

## Critical Gap: Binary Not Rebuilt

The deployed ssh-relay binary (/usr/local/bin/ssh-relay) was built 2026-03-30 01:59.
ALL features added during the March 30 session exist ONLY in source code:
- probe_hardware, deep_scan, scan_apps, preserve_data actions
- LUKS, RAID, alongside improvements
- System config generation (DE, GPU, locale)
- Server safety detection
- Git-init, disk snapshot postinstall
- Secure Boot, TPM2 postinstall

**Must rebuild binary before ANY new feature works via the portal.**

## Edge Cases Still Untested

### Hardware
- [ ] Real NVIDIA GPU (QEMU has no GPU)
- [ ] Real WiFi adapter (QEMU uses virtio-net)
- [ ] eMMC storage (/dev/mmcblk*)
- [ ] NVMe namespaces (multiple per controller)
- [ ] USB-only boot (removable media)
- [ ] ARM64 / Raspberry Pi

### Security
- [ ] LUKS passphrase handling (zeroization)
- [ ] Relay session token auth
- [ ] Rate limiter under concurrent connections
- [ ] Input validation (hostname regex, disk path validation)
- [ ] WebSocket from HTTPS page (mixed content)

### Error Recovery
- [ ] Power loss during install
- [ ] Network drop during nixos-install
- [ ] Disk full during install
- [ ] Invalid NixOS config (syntax error in generated config)
- [ ] Target machine reboots mid-install

### Dual-Boot
- [ ] BitLocker-enabled Windows (detection verified, recovery key flow untested)
- [ ] Multiple Linux distros alongside
- [ ] Windows with small ESP (100MB vs our assumed 200MB+)
- [ ] macOS detection + Asahi redirect

### Mobile
- [ ] iOS Safari
- [ ] Android Chrome
- [ ] Small screen (320px width)
- [ ] Touch-only interaction (no hover states)
- [ ] Slow mobile network (WebSocket keepalive)
