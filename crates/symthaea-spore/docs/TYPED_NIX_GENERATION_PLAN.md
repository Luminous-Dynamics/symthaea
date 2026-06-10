# Typed Nix Generation: Migration Plan

## Problem

The SSH relay generates NixOS configurations by interpolating strings into bash
heredocs. This creates a heredoc injection attack surface — any untrusted input
containing the delimiter string (`NIXCONF`, `FLAKEEOF`, `SCRIPTEOF`) can break
out of the heredoc and execute arbitrary shell commands.

We've mitigated this with `sanitize_heredoc()` (strips delimiter lines), but the
root cause remains: **shell scripts should not be the transport for Nix configs**.

## Target Architecture

```
Browser (config UI)
    │
    ▼  JSON over WebSocket
Relay (ssh_relay.rs)
    │
    ▼  SovereignConfigGenerator::generate() → String (Nix source)
    │  rnix::Root::parse() → validate syntax
    │
    ▼  scp/sftp the .nix file directly to /mnt/etc/nixos/
    │  (no heredoc, no shell interpolation)
    │
    ▼  Run: nixos-install --no-root-passwd
Target Machine
```

### Key Changes

1. **Replace `config_write_commands()`** (heredoc-based) with direct file transfer:
   - Generate Nix config as a Rust `String` via `SovereignConfigGenerator`
   - Validate with `rnix::Root::parse()`
   - Write to a temp file on the relay
   - `scp` or `sftp` the file to `/mnt/etc/nixos/configuration.nix` on the target
   - Delete the temp file

2. **Replace `generate_install_script()`** with a phased command sequence:
   - Phase 1: Partition + format (still shell — disk ops require it)
   - Phase 2: Mount filesystems
   - Phase 3: Transfer config files (scp, no heredoc)
   - Phase 4: `nixos-install --no-root-passwd`
   - Phase 5: Post-install (password, Secure Boot, TPM2)

3. **Replace `system_config_patch()`** with composable Nix modules:
   - Instead of sed-patching `configuration.nix`, generate a separate
     `system-config.nix` in Rust and transfer it alongside
   - The main config imports it: `imports = [ ./system-config.nix ];`
   - Already partially done — `system_config_patch()` writes SYSPATCH heredoc

## Migration Steps

### Phase 1: Extract security functions to lib (prerequisite)
- Move `sanitize_heredoc`, `validate_disk_path`, `sanitize_input`,
  `validate_hostname_relay` from `ssh_relay.rs` into `symthaea-spore/src/security.rs`
- Re-export from lib for fuzzing and testing
- Update ssh_relay.rs to use `use symthaea_spore::security::*`

### Phase 2: SCP-based config transfer
- Add `async fn transfer_file(content: &str, remote_path: &str)` to relay
- Uses SSH session to write file directly (no heredoc)
- Replace `config_write_commands()` calls with `transfer_file()`

### Phase 3: Structured install phases
- Split `generate_install_script()` into discrete command groups
- Each group is a small shell snippet without heredocs
- Config files transferred via Phase 2's `transfer_file()`

### Phase 4: Full SovereignConfigGenerator integration
- Browser sends `HardwareProfile` + `UserChoices` as JSON
- Relay calls `SovereignConfigGenerator::generate()`
- Output validated by rnix
- Transferred via scp
- Fallback configs eliminated (generator covers all cases)

## Effort Estimate

| Phase | Effort | Risk |
|-------|--------|------|
| Phase 1 (extract to lib) | 1 hr | Low — pure refactor |
| Phase 2 (scp transfer) | 2-3 hrs | Medium — SSH session management |
| Phase 3 (structured phases) | 4-6 hrs | Medium — test each layout |
| Phase 4 (full generator) | 2-3 hrs | Low — generator already exists |

**Total**: ~10-13 hours across 4 phases.

## Security Impact

- **Eliminates heredoc injection entirely** (the whole vulnerability class)
- Config files are validated Nix before transfer
- Shell scripts only used for disk operations (which require root anyway)
- sed-based patching eliminated in favor of composable Nix modules
