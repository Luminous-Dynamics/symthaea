# Persona-Based E2E Tests

Each persona represents a real person with specific hardware, existing software,
expectations, and fears. The test verifies that Sovereign Inoculation handles
their complete journey — from first connection to personalized welcome.

## Personas

| # | Name | Profile | Hardware | Key Test |
|---|------|---------|----------|----------|
| 1 | Maya | CS student, first Linux | Laptop, Intel iGPU, Win11 | Alongside Windows, GNOME, dev tools |
| 2 | Kai | Rust/Go developer | ThinkPad, dual monitors | LUKS + Hyprland + flakes + dotfile migration |
| 3 | Jordan | Gamer, Windows dual-boot | Desktop, NVIDIA RTX 4070, BitLocker | Alongside, Steam, NVIDIA drivers |
| 4 | River | Music producer | AMD GPU, audio interface | PipeWire low-latency, DAW detection |
| 5 | Sam | Sysadmin, server deploy | Headless rack, 2 disks | RAID1, safety BLOCKED, Docker migration |
| 6 | Alex | Privacy advocate | Old ThinkPad, German kbd | LUKS + Secure Boot + Sway + minimal |
| 7 | Pat | Non-technical teacher | Budget desktop | Simple GNOME, printing, WiFi |
| 8 | Robin | DevOps, fleet deployment | Hetzner VM (80GB) | VPS headless, SSH, automated |
| 9 | Sage | Raspberry Pi hobbyist | Pi 5, 64GB SD/USB | XFCE lightweight, home automation |
| 10 | Dakota | ARM64 cloud architect | AWS Graviton3 | VPS headless, aarch64, Docker |
| 11 | Casey | Chromebook student | 32GB eMMC, Intel | XFCE minimal, tiny disk |
| 12 | Morgan | Visually impaired | Laptop, Win11 alongside | GNOME accessibility, screen magnifier |
| 13 | Avery | Data scientist, Ubuntu→NixOS | NVIDIA laptop, conda | LUKS + GNOME + NVIDIA + Python migration |

## Running

```bash
python3 tests/personas/persona_tests.py
```

## What's Tested

For each persona:
- **Config generation**: Does the NixOS config contain the right options?
- **Config exclusions**: Does it avoid options that shouldn't be there?
- **Welcome message**: Does the personalized welcome mention their tools?
- **Safety detection**: Is the correct safety level triggered?
- **App coverage**: Do their apps have NixOS equivalents?

## Adding Personas

Add a new `Persona()` to the `PERSONAS` list. Define their hardware, apps,
choices, and expected outcomes. The test framework validates automatically.
