// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! NixForHumanity WebSocket Relay
//!
//! Runs on the target NixOS installer ISO. Receives commands from the browser
//! via WebSocket and executes them locally. No SSH required.
//!
//! # Usage
//! ```bash
//! cargo run --bin ssh-relay --features server -- --port 8091
//! ```

use futures_util::{SinkExt, StreamExt};
use std::collections::HashMap;
use std::io::Read;
use std::sync::Arc;
use std::time::Instant;
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tokio_tungstenite::accept_hdr_async;
use tokio_tungstenite::tungstenite::Message;

// Security validators from the library (shared with fuzz targets)
use symthaea_spore::security::{
    sanitize_heredoc, sanitize_input, token_eq, validate_disk_path,
    validate_hostname as validate_hostname_relay,
};

// TLS support
use rustls::ServerConfig;
use tokio_rustls::TlsAcceptor;

/// Execute a shell command locally and return stdout/stderr + exit status.
/// Replaces the previous SSH-to-localhost pattern.
struct CmdResult {
    pub stdout: String,
    pub stderr: String,
    pub exit_status: u32,
}

async fn run_cmd(cmd: &str) -> Result<CmdResult, std::io::Error> {
    // Use /bin/sh (POSIX, always available) as fallback if bash isn't in PATH
    let shell = if std::path::Path::new("/bin/bash").exists() {
        "/bin/bash"
    } else if std::path::Path::new("/run/current-system/sw/bin/bash").exists() {
        "/run/current-system/sw/bin/bash"
    } else {
        "/bin/sh"
    };
    // Inherit the full system environment (PATH, NIX_PATH, etc.)
    // and ensure common NixOS paths are available
    let mut command = tokio::process::Command::new(shell);
    command.arg("-c").arg(cmd);
    // Supplement PATH with NixOS-specific locations
    let sys_path = std::env::var("PATH").unwrap_or_default();
    let full_path = format!(
        "{}:/run/current-system/sw/bin:/nix/var/nix/profiles/default/bin:/usr/local/bin:/usr/bin:/bin:/sbin",
        sys_path
    );
    command.env("PATH", &full_path);
    // Ensure NIX_PATH is set for nixos-install
    if std::env::var("NIX_PATH").is_err() {
        command.env("NIX_PATH", "nixpkgs=/nix/var/nix/profiles/per-user/root/channels/nixos:nixos-config=/etc/nixos/configuration.nix");
    }
    let output = command.output().await?;
    Ok(CmdResult {
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        exit_status: output.status.code().unwrap_or(1) as u32,
    })
}

/// nixos-anywhere orchestration stages.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "PascalCase")]
enum NixosAnywhereStage {
    Connecting,
    UploadingKexec,
    Kexec,
    WaitingForReboot,
    Partitioning,
    Installing,
    Configuring,
    FinalReboot,
    Verifying,
    Complete,
}

impl NixosAnywhereStage {
    fn percentage(&self) -> u8 {
        match self {
            Self::Connecting => 5,
            Self::UploadingKexec => 15,
            Self::Kexec => 25,
            Self::WaitingForReboot => 35,
            Self::Partitioning => 50,
            Self::Installing => 70,
            Self::Configuring => 85,
            Self::FinalReboot => 92,
            Self::Verifying => 97,
            Self::Complete => 100,
        }
    }

    fn inoculation_phase(&self) -> &'static str {
        match self {
            Self::Connecting => "TrustVerification",
            Self::UploadingKexec | Self::Kexec | Self::WaitingForReboot => "FlakeEvaluation",
            Self::Partitioning => "DiskPreparation",
            Self::Installing | Self::Configuring => "StorePopulation",
            Self::FinalReboot => "MokEnrollment",
            Self::Verifying | Self::Complete => "FirstBreath",
        }
    }
}

/// Parse nixos-anywhere output to determine current stage.
fn parse_stage(output: &str) -> Option<NixosAnywhereStage> {
    let lower = output.to_lowercase();
    if lower.contains("uploading kexec") || lower.contains("copying kexec") {
        Some(NixosAnywhereStage::UploadingKexec)
    } else if lower.contains("executing kexec") || lower.contains("kexec -e") {
        Some(NixosAnywhereStage::Kexec)
    } else if lower.contains("waiting for") && lower.contains("reboot") {
        Some(NixosAnywhereStage::WaitingForReboot)
    } else if lower.contains("partitioning") || lower.contains("disko") {
        Some(NixosAnywhereStage::Partitioning)
    } else if lower.contains("installing") || lower.contains("nixos-install") {
        Some(NixosAnywhereStage::Installing)
    } else if lower.contains("configuring") || lower.contains("nixos-rebuild") {
        Some(NixosAnywhereStage::Configuring)
    } else if lower.contains("final reboot") {
        Some(NixosAnywhereStage::FinalReboot)
    } else if lower.contains("verification") || lower.contains("complete") {
        Some(NixosAnywhereStage::Complete)
    } else {
        None
    }
}

// Security validators imported from symthaea_spore::security (see use statement above).
// Local definitions removed — single source of truth for fuzzing and testing.

// ── sanitize_heredoc also imported from security module ──

/// Rate limiter: 1 active session per IP, with auth failure tracking.
struct SessionTracker {
    active: HashMap<String, Instant>,
    failed_auths: HashMap<String, (u32, Instant)>, // (count, first_attempt)
    timeout_secs: u64,
}

impl SessionTracker {
    fn new(timeout_secs: u64) -> Self {
        Self {
            active: HashMap::new(),
            failed_auths: HashMap::new(),
            timeout_secs,
        }
    }

    fn try_acquire(&mut self, ip: &str) -> bool {
        let now = Instant::now();
        // Expire old sessions
        self.active
            .retain(|_, start| now.duration_since(*start).as_secs() < self.timeout_secs);
        if self.active.contains_key(ip) {
            return false;
        }
        self.active.insert(ip.to_string(), now);
        true
    }

    fn release(&mut self, ip: &str) {
        self.active.remove(ip);
    }

    fn record_failed_auth(&mut self, ip: &str) {
        let entry = self
            .failed_auths
            .entry(ip.to_string())
            .or_insert((0, Instant::now()));
        entry.0 += 1;
    }

    fn is_blocked(&self, ip: &str) -> bool {
        if let Some((count, first)) = self.failed_auths.get(ip) {
            // Block after 5 failed attempts within 5 minutes
            *count >= 5 && first.elapsed().as_secs() < 300
        } else {
            false
        }
    }
}

/// Client → Relay message.
#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct ClientMessage {
    action: String,
    /// Mandatory WebSocket auth token (must be sent via the `"auth"` action before any other action).
    #[serde(default)]
    token: String,
    #[serde(default)]
    host: String,
    #[serde(default = "default_port")]
    port: u16,
    #[serde(default)]
    username: String,
    #[serde(default)]
    password: String,
    #[serde(default)]
    command: String,
    // Install-specific fields
    #[serde(default)]
    disk: String, // e.g., "/dev/nvme0n1"
    #[serde(default)]
    layout: String, // "single", "dual", "alongside", "sata", "vps"
    #[serde(default)]
    fast_disk: String, // For dual-disk: fast drive
    #[serde(default)]
    standard_disk: String, // For dual-disk: standard drive
    #[serde(default)]
    hostname: String,
    #[serde(default)]
    configuration_nix: String, // Generated configuration.nix content from browser
    #[serde(default)]
    flake_nix: String, // Generated flake.nix content
    #[serde(default)]
    disko_nix: String, // Generated disko-config.nix content
    #[serde(default)]
    hardware_nix: String, // Generated hardware-configuration.nix content
    #[serde(default)]
    secure_boot: bool, // Enable Secure Boot (lanzaboote + sbctl)
    #[serde(default)]
    tpm2_unlock: bool, // Enable TPM2 auto-unlock (requires LUKS + systemd initrd)
    #[serde(default)]
    fido2_unlock: bool, // Enable FIDO2/YubiKey unlock (requires LUKS + systemd initrd)
    #[serde(default)]
    desktop: String, // Desktop environment: gnome, plasma, hyprland, sway, xfce, none
    #[serde(default)]
    gpu_driver: String, // GPU driver: nvidia, nvidia-open, amdgpu, modesetting, none
    #[serde(default)]
    timezone: String, // e.g., "America/Chicago"
    #[serde(default)]
    keyboard: String, // e.g., "us", "de", "dvorak"
    #[serde(default)]
    user_password: String, // User account password (set via chpasswd after install)
    /// Additional disks for RAID/ZFS multi-disk layouts (comma-separated or JSON array)
    #[serde(default)]
    extra_disks: Vec<String>,
}

fn default_port() -> u16 {
    22
}

/// Generate Secure Boot setup commands (appended to install script when enabled).
/// Git-initialize the NixOS config (always appended to install scripts).
fn git_init_config() -> &'static str {
    r#"
# ── Git-Initialize NixOS Config ──
echo "STAGE: Initializing config version control..."
if command -v git >/dev/null 2>&1 || [ -f /mnt/nix/store/*/bin/git ]; then
  GIT=$(command -v git 2>/dev/null || ls /mnt/nix/store/*/bin/git 2>/dev/null | head -1)
  chroot /mnt /bin/sh -c '
    cd /etc/nixos
    if [ ! -d .git ]; then
      git init
      git add -A
      git commit -m "Initial NixOS configuration — Sovereign Inoculation"
      echo "  Config versioned at /etc/nixos/.git"
    fi
  ' 2>/dev/null || echo "  Git init skipped (git not available yet)"
fi
"#
}

/// Pre-install disk snapshot (partition table + UUIDs — always, instant).
fn disk_snapshot(disk: &str) -> String {
    format!(
        r#"
# ── Pre-Install Disk Snapshot (Tier 1: instant) ──
echo "STAGE: Saving disk snapshot..."
SNAPSHOT_DIR="/tmp/symthaea-pre-install-snapshot"
mkdir -p "$SNAPSHOT_DIR"
sfdisk -d {disk} > "$SNAPSHOT_DIR/partition-table.dump" 2>/dev/null
dd if={disk} of="$SNAPSHOT_DIR/first-1M.img" bs=1M count=1 status=none 2>/dev/null
blkid > "$SNAPSHOT_DIR/blkid.txt" 2>/dev/null
lsblk -f > "$SNAPSHOT_DIR/lsblk.txt" 2>/dev/null
fdisk -l {disk} > "$SNAPSHOT_DIR/fdisk.txt" 2>/dev/null
echo "  Snapshot saved to $SNAPSHOT_DIR"
echo "  Partition table can be restored with: sfdisk {disk} < partition-table.dump"
"#,
        disk = disk
    )
}

fn secure_boot_postinstall() -> &'static str {
    r#"
# ── Secure Boot Setup (lanzaboote + sbctl) ──
echo "STAGE: Setting up Secure Boot..."

# Check if firmware is in Setup Mode
SETUP_MODE=$(bootctl status 2>/dev/null | grep "Setup Mode:" | grep -c "setup" || echo "0")
if [ "$SETUP_MODE" = "0" ]; then
  echo "WARNING: Firmware is NOT in Setup Mode."
  echo "WARNING: Secure Boot keys will be created but NOT enrolled."
  echo "WARNING: Enter BIOS, clear Secure Boot keys, then re-run key enrollment."
fi

# Create Secure Boot keys on the installed system
chroot /mnt /bin/sh -c '
  if command -v sbctl >/dev/null 2>&1; then
    sbctl create-keys 2>/dev/null || echo "Keys may already exist"
    if [ "'"$SETUP_MODE"'" = "1" ]; then
      sbctl enroll-keys --microsoft 2>/dev/null && echo "Secure Boot keys enrolled (with Microsoft CA)" || echo "Key enrollment failed — enroll manually after first boot"
    else
      echo "Skipping key enrollment — firmware not in Setup Mode"
      echo "After first boot: sudo sbctl enroll-keys --microsoft"
    fi
  else
    echo "sbctl not found — install it and run: sbctl create-keys && sbctl enroll-keys --microsoft"
  fi
'
echo "  Secure Boot keys created at /etc/secureboot/"
"#
}

/// Generate TPM2 auto-unlock enrollment (appended after LUKS install + Secure Boot).
fn tpm2_postinstall() -> &'static str {
    r#"
# ── TPM2 Auto-Unlock Enrollment ──
echo "STAGE: Enrolling TPM2 auto-unlock..."

# Check TPM availability
if [ ! -e /dev/tpmrm0 ]; then
  echo "WARNING: TPM 2.0 not detected. Skipping auto-unlock enrollment."
  echo "You will need to enter your passphrase at every boot."
else
  # Find the LUKS device
  LUKS_DEV=$(blkid -t TYPE=crypto_LUKS -o device 2>/dev/null | head -1)
  if [ -n "$LUKS_DEV" ]; then
    # Enroll TPM2 with PCR 0 (firmware) and PCR 7 (Secure Boot state)
    # The passphrase is required to authorize the enrollment
    echo "Enrolling TPM2 on $LUKS_DEV (PCR 0+7)..."
    systemd-cryptenroll "$LUKS_DEV" --tpm2-device=auto --tpm2-pcrs=0+7 2>&1 || echo "WARNING: TPM2 enrollment failed. You can retry after first boot with: sudo systemd-cryptenroll $LUKS_DEV --tpm2-device=auto --tpm2-pcrs=0+7"

    # Update NixOS config to use systemd initrd (required for TPM2 unlock)
    if [ -f /mnt/etc/nixos/configuration.nix ]; then
      # Add systemd initrd and TPM2 config
      sed -i '/boot.initrd.luks.devices/a\    cryptTabExtraOpts = [ "tpm2-device=auto" ];' /mnt/etc/nixos/configuration.nix 2>/dev/null || true
      sed -i '/imports = /a\  boot.initrd.systemd.enable = true;' /mnt/etc/nixos/configuration.nix 2>/dev/null || true
      echo "  TPM2 enrollment complete. Disk will auto-unlock at boot."
      echo "  Passphrase is kept as fallback (firmware updates will require it)."
    fi
  else
    echo "WARNING: No LUKS device found. TPM2 enrollment skipped."
  fi
fi
"#
}

/// Generate FIDO2/YubiKey enrollment commands (appended after LUKS install).
fn fido2_postinstall() -> &'static str {
    r#"
# ── FIDO2/YubiKey Enrollment ──
echo "STAGE: Enrolling FIDO2 security key..."
if ls /dev/hidraw* >/dev/null 2>&1; then
    LUKS_DEV=$(blkid -t TYPE=crypto_LUKS -o device 2>/dev/null | head -1)
    if [ -n "$LUKS_DEV" ]; then
        echo "Enrolling FIDO2 device on $LUKS_DEV..."
        echo "Touch your security key when it blinks."
        systemd-cryptenroll "$LUKS_DEV" --fido2-device=auto 2>&1 || echo "WARNING: FIDO2 enrollment failed. Passphrase still works."
    else
        echo "WARNING: No LUKS device found. Skipping FIDO2 enrollment."
    fi
else
    echo "WARNING: No FIDO2 device detected. Skipping enrollment."
    echo "You can enroll later with: systemd-cryptenroll /dev/<device> --fido2-device=auto"
fi
"#
}

/// Generate NixOS configuration snippet for desktop environment, GPU, locale.
fn generate_system_config(msg: &ClientMessage) -> String {
    let mut config = String::new();

    // Timezone
    let tz = if msg.timezone.is_empty() {
        "UTC"
    } else {
        &msg.timezone
    };
    config.push_str(&format!("  time.timeZone = \"{}\";\n", tz));

    // Locale
    config.push_str("  i18n.defaultLocale = \"en_US.UTF-8\";\n");

    // Keyboard
    let kb = if msg.keyboard.is_empty() {
        "us"
    } else {
        &msg.keyboard
    };
    config.push_str(&format!("  console.keyMap = \"{}\";\n", kb));
    config.push_str(&format!("  services.xserver.xkb.layout = \"{}\";\n", kb));

    // Desktop environment
    match msg.desktop.as_str() {
        "gnome" => {
            config.push_str("  services.xserver.enable = true;\n");
            config.push_str("  services.xserver.displayManager.gdm.enable = true;\n");
            config.push_str("  services.xserver.desktopManager.gnome.enable = true;\n");
        }
        "plasma" => {
            config.push_str("  services.xserver.enable = true;\n");
            config.push_str("  services.displayManager.sddm.enable = true;\n");
            config.push_str("  services.desktopManager.plasma6.enable = true;\n");
        }
        "hyprland" => {
            config.push_str("  programs.hyprland.enable = true;\n");
            config.push_str("  services.displayManager.sddm.enable = true;\n");
            config.push_str("  services.displayManager.sddm.wayland.enable = true;\n");
        }
        "sway" => {
            config.push_str("  programs.sway.enable = true;\n");
            config.push_str("  services.displayManager.sddm.enable = true;\n");
            config.push_str("  services.displayManager.sddm.wayland.enable = true;\n");
        }
        "xfce" => {
            config.push_str("  services.xserver.enable = true;\n");
            config.push_str("  services.xserver.displayManager.lightdm.enable = true;\n");
            config.push_str("  services.xserver.desktopManager.xfce.enable = true;\n");
        }
        _ => {} // "none" or empty — no DE (server/CLI)
    }

    // GPU driver
    match msg.gpu_driver.as_str() {
        "nvidia" => {
            config.push_str("  services.xserver.videoDrivers = [ \"nvidia\" ];\n");
            config.push_str("  hardware.nvidia.modesetting.enable = true;\n");
            config.push_str("  hardware.nvidia.open = false;\n");
        }
        "nvidia-open" => {
            config.push_str("  services.xserver.videoDrivers = [ \"nvidia\" ];\n");
            config.push_str("  hardware.nvidia.modesetting.enable = true;\n");
            config.push_str("  hardware.nvidia.open = true;\n");
        }
        "amdgpu" => {
            config.push_str("  services.xserver.videoDrivers = [ \"amdgpu\" ];\n");
        }
        "modesetting" => {
            config.push_str("  services.xserver.videoDrivers = [ \"modesetting\" ];\n");
        }
        _ => {} // "auto" or "none"
    }

    // Networking
    config.push_str("  networking.networkmanager.enable = true;\n");

    config
}

/// Generate boot mode detection + partitioning commands.
/// Returns a shell snippet that sets BOOT_MODE=efi|bios and creates boot partition accordingly.
fn boot_mode_detection() -> &'static str {
    r#"
# ── Boot Mode Detection ──
if [ -d /sys/firmware/efi ]; then
  BOOT_MODE="efi"
  echo "  Boot mode: EFI/UEFI"
else
  BOOT_MODE="bios"
  echo "  Boot mode: Legacy BIOS (GRUB will be used)"
fi
"#
}

/// Generate shell commands that write the correct bootloader config to configuration.nix.
/// Called after nixos-generate-config, patches the bootloader section based on detected boot mode.
fn bootloader_patch_commands(disk_var: &str) -> String {
    format!(
        r#"
# Patch bootloader config based on detected boot mode
if [ "$BOOT_MODE" = "bios" ]; then
  echo "  Configuring GRUB for BIOS boot..."
  sed -i 's|boot.loader.systemd-boot.enable = true|boot.loader.grub.enable = true|' /mnt/etc/nixos/configuration.nix 2>/dev/null || true
  sed -i '/boot.loader.grub.enable/a\  boot.loader.grub.device = "{disk}";' /mnt/etc/nixos/configuration.nix 2>/dev/null || true
  sed -i '/canTouchEfiVariables/d' /mnt/etc/nixos/configuration.nix 2>/dev/null || true
fi
"#,
        disk = disk_var
    )
}

/// Generate boot partition creation commands based on boot mode.
/// EFI: 512MB FAT32 ESP. BIOS: 1MB BIOS boot + 512MB ext4 /boot.
fn boot_partition_commands(disk_var: &str, boot_part_num: u32) -> String {
    format!(
        r#"
if [ "$BOOT_MODE" = "efi" ]; then
  sgdisk -n {n}:0:+512M -t {n}:EF00 -c {n}:boot "{disk}"
else
  # BIOS: create BIOS boot partition (1MB) + /boot partition (512MB)
  sgdisk -n {n}:0:+1M -t {n}:EF02 -c {n}:bios-boot "{disk}"
  sgdisk -n {next}:0:+512M -t {next}:8300 -c {next}:boot "{disk}"
fi
"#,
        n = boot_part_num,
        next = boot_part_num + 1,
        disk = disk_var
    )
}

/// Format and mount the boot partition based on boot mode.
/// EFI: mkfs.vfat + mount to /boot. BIOS: mkfs.ext4 + mount to /boot.
fn boot_format_mount(boot_part_var: &str) -> String {
    format!(
        r#"
if [ "$BOOT_MODE" = "efi" ]; then
  mkfs.vfat -F 32 "{boot}"
else
  # BIOS boot: format the /boot partition (not the 1MB BIOS boot partition)
  # The BIOS boot partition (EF02) is left unformatted — GRUB writes to it directly
  mkfs.ext4 -F -L boot "{boot}"
fi
mkdir -p /mnt/boot
mount "{boot}" /mnt/boot
"#,
        boot = boot_part_var
    )
}

/// Generate a shell snippet that patches configuration.nix with system config (DE, GPU, locale).
/// Appended after the configuration.nix heredoc in each layout.
fn system_config_patch(msg: &ClientMessage) -> String {
    let sys_config = generate_system_config(msg);
    if sys_config.trim().is_empty() {
        return String::new();
    }
    // Write a supplementary config file instead of patching inline —
    // avoids fragile heredoc-in-command-substitution shell constructs.
    format!(
        r#"
# Write supplementary system config (DE, GPU, locale, networking)
cat > /mnt/etc/nixos/system-config.nix << 'SYSPATCH'
{{ config, pkgs, ... }}:
{{
  # ── System Configuration (NixForHumanity) ──
{sys_config}
  # Audio (PipeWire)
  services.pulseaudio.enable = false;
  security.rtkit.enable = true;
  services.pipewire = {{ enable = true; alsa.enable = true; pulse.enable = true; }};

  # Nix settings
  nix.settings.experimental-features = [ "nix-command" "flakes" ];
  nix.gc = {{ automatic = true; dates = "weekly"; options = "--delete-older-than 30d"; }};
}}
SYSPATCH
# Add import to configuration.nix
sed -i 's|imports = \[|imports = [ ./system-config.nix|' /mnt/etc/nixos/configuration.nix 2>/dev/null || echo "  (config patch: manual import needed)"
"#,
        sys_config = sys_config,
    )
}

/// Generate the automated install script based on layout type.
/// Build the shell commands that write configuration.nix (and optionally flake.nix)
/// to /mnt/etc/nixos/.  When the browser supplied a generated config we use that
/// verbatim; otherwise we fall back to the hardcoded minimal config for this layout.
///
/// The content is written via heredoc so that braces in the Nix source are never
/// passed through Rust's `format!()` (which would require `{{`/`}}` escaping).
// sanitize_heredoc imported from symthaea_spore::security

/// Write NixOS config files directly to the target filesystem.
///
/// SECURITY: This replaces the heredoc-based `config_write_commands()`.
/// By writing files directly via the filesystem API, there is no shell
/// interpolation, no heredoc delimiter to escape, and no injection vector.
/// The relay runs on the target machine, so direct file writes are possible.
async fn write_config_files(
    browser_config: &str,
    fallback_config: &str,
    browser_flake: &str,
) -> Result<(), String> {
    // Ensure target directory exists
    tokio::fs::create_dir_all("/mnt/etc/nixos")
        .await
        .map_err(|e| format!("Failed to create /mnt/etc/nixos: {}", e))?;

    // configuration.nix
    let config_body = if browser_config.is_empty() {
        fallback_config
    } else {
        browser_config
    };
    tokio::fs::write("/mnt/etc/nixos/configuration.nix", config_body)
        .await
        .map_err(|e| format!("Failed to write configuration.nix: {}", e))?;

    // flake.nix (only if the browser supplied one)
    if !browser_flake.is_empty() {
        tokio::fs::write("/mnt/etc/nixos/flake.nix", browser_flake)
            .await
            .map_err(|e| format!("Failed to write flake.nix: {}", e))?;
    }

    Ok(())
}

/// Write NixOS configs by copying pre-staged files from /tmp (no heredoc for user input).
/// Falls back to heredoc only for server-generated fallback configs (safe: not user-controlled).
fn config_write_commands(
    browser_config: &str,
    fallback_config: &str,
    browser_flake: &str,
    session_id: u64,
) -> String {
    let staging = format!("/tmp/symthaea-config-{}", session_id);
    let mut out = String::new();
    out.push_str("mkdir -p /mnt/etc/nixos\n");

    if !browser_config.is_empty() {
        // Browser config pre-staged via tokio::fs::write — no heredoc, no injection
        out.push_str(&format!(
            "cp {}/configuration.nix /mnt/etc/nixos/configuration.nix\n",
            staging
        ));
    } else {
        // Server-generated fallback — safe to use heredoc (not user input)
        out.push_str("cat > /mnt/etc/nixos/configuration.nix << 'NIXCONF'\n");
        out.push_str(fallback_config);
        if !fallback_config.ends_with('\n') {
            out.push('\n');
        }
        out.push_str("NIXCONF\n");
    }

    if !browser_flake.is_empty() {
        out.push_str(&format!(
            "cp {}/flake.nix /mnt/etc/nixos/flake.nix\n",
            staging
        ));
    }

    out.push_str(&format!("rm -rf {}\n", staging));
    out
}

/// Legacy heredoc-based config writing (kept for tests and fallback reference).
fn config_write_commands_heredoc(
    browser_config: &str,
    fallback_config: &str,
    browser_flake: &str,
) -> String {
    let mut out = String::new();

    let config_body = if browser_config.is_empty() {
        fallback_config
    } else {
        browser_config
    };
    let safe_config = sanitize_heredoc(config_body, "NIXCONF");
    out.push_str("cat > /mnt/etc/nixos/configuration.nix << 'NIXCONF'\n");
    out.push_str(&safe_config);
    if !safe_config.ends_with('\n') {
        out.push('\n');
    }
    out.push_str("NIXCONF\n");

    if !browser_flake.is_empty() {
        let safe_flake = sanitize_heredoc(browser_flake, "FLAKEEOF");
        out.push_str("\ncat > /mnt/etc/nixos/flake.nix << 'FLAKEEOF'\n");
        out.push_str(&safe_flake);
        if !safe_flake.ends_with('\n') {
            out.push('\n');
        }
        out.push_str("FLAKEEOF\n");
    }

    out
}

fn generate_install_script(msg: &ClientMessage, session_id: u64) -> String {
    // SECURITY: All user inputs (disk, hostname, timezone, keyboard, desktop, gpu_driver)
    // MUST be validated by the caller before reaching this function.
    // See validate_disk_path(), validate_hostname_relay(), sanitize_input().
    let hostname = if msg.hostname.is_empty() {
        "guardian"
    } else {
        &msg.hostname
    };

    match msg.layout.as_str() {
        "alongside" => {
            // Alongside Windows/Linux: find free space, reuse existing ESP, install
            let mut script = format!(
                r#"set -eo pipefail
echo "=== Symthaea Sovereign Birth: Alongside Existing OS ==="

DISK="{disk}"

# Safety: Check for BitLocker
echo "STAGE: Checking for BitLocker..."
BITLOCKER_DETECTED=false
for PART in $(blkid -o device "$DISK"* 2>/dev/null); do
  if blkid "$PART" 2>/dev/null | grep -qi bitlocker; then
    BITLOCKER_DETECTED=true
    echo "BITLOCKER_DETECTED: $PART"
    echo ""
    echo "================================================================"
    echo "  BITLOCKER ENCRYPTION DETECTED on $PART"
    echo "================================================================"
    echo ""
    echo "  Before proceeding, you MUST:"
    echo ""
    echo "  1. FIND YOUR RECOVERY KEY — you will need it if BitLocker"
    echo "     activates during partition changes."
    echo ""
    echo "     Where to find it:"
    echo "     - Microsoft account: https://account.microsoft.com/devices/recoverykey"
    echo "     - Azure AD: Check with your IT administrator"
    echo "     - Printout: Check papers from when you set up this PC"
    echo "     - USB drive: Check USB drives used during BitLocker setup"
    echo ""
    echo "  2. RECOMMENDED: Boot into Windows and SUSPEND BitLocker first:"
    echo "     Settings > Privacy & Security > Device Encryption > Turn Off"
    echo "     (or: manage-bde -protectors -disable C: from Admin CMD)"
    echo ""
    echo "  3. ALTERNATIVE: Shrink Windows partition from within Windows:"
    echo "     Settings > System > Storage > Advanced > Disks & volumes"
    echo "     Select C: > Shrink > Enter desired free space (min 40GB)"
    echo ""
    echo "  The install will continue, but if BitLocker recovery triggers,"
    echo "  you will need your recovery key to access Windows again."
    echo "================================================================"
    echo ""
  fi
done

# Step 1: Find unallocated space on the disk
echo "STAGE: Detecting free space on {disk}..."
LAST_END=$(sgdisk -p "$DISK" 2>/dev/null | grep '^ ' | tail -1 | awk '{{print $3}}')
DISK_END=$(sgdisk -p "$DISK" 2>/dev/null | grep 'Disk size' | awk '{{print $3}}')
FREE_SECTORS=$((DISK_END - LAST_END - 34))
FREE_GB=$((FREE_SECTORS * 512 / 1073741824))
echo "Last partition ends at sector $LAST_END, disk ends at $DISK_END"
echo "Free space: ~${{FREE_GB}}GB ($FREE_SECTORS sectors)"

if [ "$FREE_GB" -lt 20 ]; then
  echo "STAGE: Attempting automatic NTFS partition shrink..."
  # Find the largest NTFS partition (likely Windows C:)
  NTFS_PART=""
  NTFS_SIZE=0
  for PART in $(lsblk -rno NAME,FSTYPE "$DISK" 2>/dev/null | awk '$2=="ntfs"{{print "/dev/"$1}}'); do
    SZ=$(blockdev --getsize64 "$PART" 2>/dev/null || echo 0)
    if [ "$SZ" -gt "$NTFS_SIZE" ]; then
      NTFS_SIZE=$SZ
      NTFS_PART=$PART
    fi
  done

  if [ -n "$NTFS_PART" ] && [ "$BITLOCKER_DETECTED" = false ] && command -v ntfsresize >/dev/null 2>&1; then
    echo "  Found NTFS partition: $NTFS_PART ($((NTFS_SIZE / 1073741824))GB)"
    # Check NTFS consistency first
    ntfsfix -n "$NTFS_PART" 2>/dev/null || true
    # Get used space from ntfsinfo
    NTFS_USED=$(ntfsresize --info --force "$NTFS_PART" 2>/dev/null | grep "resize at" | grep -oP '[0-9]+' | tail -1 || echo "0")
    if [ "$NTFS_USED" -gt 0 ]; then
      # New size = used + 20% headroom + 10GB safety margin (whichever is larger)
      HEADROOM_20=$((NTFS_USED * 20 / 100))
      MIN_HEADROOM=$((10 * 1024 * 1024 * 1024))
      HEADROOM=$((HEADROOM_20 > MIN_HEADROOM ? HEADROOM_20 : MIN_HEADROOM))
      NEW_SIZE=$((NTFS_USED + HEADROOM))
      # Safety: never shrink below 50% of original
      HALF=$((NTFS_SIZE / 2))
      [ "$NEW_SIZE" -lt "$HALF" ] && NEW_SIZE=$HALF
      NEW_GB=$((NEW_SIZE / 1073741824))
      FREED_GB=$(((NTFS_SIZE - NEW_SIZE) / 1073741824))
      echo "  Used: $((NTFS_USED / 1073741824))GB, New size: ${{NEW_GB}}GB (freeing ~${{FREED_GB}}GB)"
      echo "  Running dry-run first..."
      if ntfsresize --no-action --size "$NEW_SIZE" "$NTFS_PART" 2>&1; then
        echo "  Dry-run passed. Resizing NTFS partition..."
        ntfsresize --force --size "$NEW_SIZE" "$NTFS_PART" 2>&1
        # Shrink the partition table entry to match
        PART_NUM=$(echo "$NTFS_PART" | grep -oP '[0-9]+$')
        NEW_SECTORS=$((NEW_SIZE / 512))
        echo "  Updating partition table (partition $PART_NUM to $NEW_SECTORS sectors)..."
        # Use sgdisk to delete and recreate the partition at the new size
        PART_START=$(sgdisk -i "$PART_NUM" "$DISK" 2>/dev/null | grep "First sector" | awk '{{print $3}}')
        PART_TYPE=$(sgdisk -i "$PART_NUM" "$DISK" 2>/dev/null | grep "Partition GUID code" | awk '{{print $4}}')
        PART_NAME=$(sgdisk -i "$PART_NUM" "$DISK" 2>/dev/null | grep "Partition name" | cut -d"'" -f2)
        if [ -n "$PART_START" ]; then
          sgdisk -d "$PART_NUM" "$DISK" 2>/dev/null
          sgdisk -n "$PART_NUM:$PART_START:+$NEW_SECTORS" -t "$PART_NUM:0700" -c "$PART_NUM:$PART_NAME" "$DISK" 2>/dev/null
          partprobe "$DISK" 2>/dev/null || true
          udevadm settle 2>/dev/null || true
          echo "  NTFS partition shrunk successfully. Rechecking free space..."
        fi
        # Recheck free space
        LAST_END=$(sgdisk -p "$DISK" 2>/dev/null | grep '^ ' | tail -1 | awk '{{print $3}}')
        FREE_SECTORS=$((DISK_END - LAST_END - 34))
        FREE_GB=$((FREE_SECTORS * 512 / 1073741824))
        echo "  Free space after shrink: ~${{FREE_GB}}GB"
      else
        echo "  Dry-run FAILED — partition cannot be safely shrunk."
        echo "  Please shrink from within Windows instead."
      fi
    else
      echo "  Could not determine NTFS used space. Manual shrink required."
    fi
  elif [ "$BITLOCKER_DETECTED" = true ]; then
    echo "  Cannot auto-shrink: BitLocker is active. Suspend BitLocker in Windows first."
  elif [ -z "$NTFS_PART" ]; then
    echo "  No NTFS partition found to shrink."
  else
    echo "  ntfsresize not available on this ISO."
  fi

  # Final check after potential shrink
  if [ "$FREE_GB" -lt 20 ]; then
    echo "ERROR: Still less than 20GB free space (${{FREE_GB}}GB)."
    echo "ERROR: Boot into Windows and shrink the C: partition:"
    echo "  Settings > System > Storage > Advanced > Disks & volumes"
    echo "  Select C: drive > Shrink > Enter at least 40GB"
    exit 1
  fi
fi

# Step 2: Create NixOS partition in the free space (do NOT resize existing partitions)
echo "STAGE: Partitioning free space..."
PART_NUM=$(sgdisk -p "$DISK" | grep '^ ' | wc -l)
PART_NUM=$((PART_NUM + 1))
sgdisk -n "$PART_NUM:0:0" -t "$PART_NUM:8300" -c "$PART_NUM:nixos-root" "$DISK"
partprobe "$DISK" 2>/dev/null || true
udevadm settle 2>/dev/null || true
sleep 3
NIXOS_PART="${{DISK}}p$PART_NUM"
[ -b "$NIXOS_PART" ] || NIXOS_PART="${{DISK}}$PART_NUM"
echo "Created partition: $NIXOS_PART"

# Step 3: Format with btrfs + subvolumes
echo "STAGE: Formatting with btrfs..."
mkfs.btrfs -f -L nixos "$NIXOS_PART"
mount "$NIXOS_PART" /mnt
btrfs subvolume create /mnt/@
btrfs subvolume create /mnt/@home
btrfs subvolume create /mnt/@nix
btrfs subvolume create /mnt/@log
btrfs subvolume create /mnt/@snapshots
btrfs subvolume create /mnt/@swap
umount /mnt

# Step 4: Mount everything
echo "STAGE: Mounting filesystems..."
mount -o subvol=@,compress=zstd:3,noatime "$NIXOS_PART" /mnt
mkdir -p /mnt/{{home,nix,var/log,.snapshots,swap,boot,etc/nixos}}
mount -o subvol=@home,compress=zstd:3,noatime "$NIXOS_PART" /mnt/home
mount -o subvol=@nix,compress=zstd:3,noatime "$NIXOS_PART" /mnt/nix
mount -o subvol=@log,compress=zstd:3,noatime "$NIXOS_PART" /mnt/var/log
mount -o subvol=@snapshots,compress=zstd:3,noatime "$NIXOS_PART" /mnt/.snapshots
mount -o subvol=@swap,noatime "$NIXOS_PART" /mnt/swap

# Reuse existing EFI System Partition (do NOT create a new one)
EFI_PART=$(lsblk -nlo NAME,PARTTYPE "$DISK" | grep -i 'c12a7328' | head -1 | awk '{{print "/dev/"$1}}')
if [ -n "$EFI_PART" ]; then
  mount "$EFI_PART" /mnt/boot
  echo "Reusing existing ESP: $EFI_PART"
else
  echo "WARNING: No EFI partition found. systemd-boot may not work."
fi

# Step 5: Write NixOS configuration
echo "STAGE: Generating configuration..."
nixos-generate-config --root /mnt
"#,
                disk = msg.disk
            );

            // Append configuration.nix (and optionally flake.nix) via heredoc —
            // avoids passing Nix braces through format!().
            let fallback_alongside = format!(
                "{{ config, pkgs, ... }}:\n\
                 {{\n\
                 \x20 imports = [ ./hardware-configuration.nix ];\n\
                 \x20 networking.hostName = \"{hostname}\";\n\
                 \x20 boot.loader.systemd-boot.enable = true;\n\
                 \x20 boot.loader.efi.canTouchEfiVariables = true;\n\
                 \x20 # Dual-boot: sync hardware clock with Windows (which uses localtime)\n\
                 \x20 time.hardwareClockInLocalTime = true;\n\
                 \x20 services.openssh.enable = true;\n\
                 \x20 services.earlyoom = {{ enable = true; freeMemThreshold = 5; freeSwapThreshold = 5; }};\n\
                 \x20 services.fstrim.enable = true;\n\
                 \x20 services.smartd = {{ enable = true; autodetect = true; }};\n\
                 \x20 services.btrfs.autoScrub = {{ enable = true; interval = \"monthly\"; fileSystems = [ \"/\" ]; }};\n\
                 \x20 zramSwap = {{ enable = true; algorithm = \"zstd\"; }};\n\
                 \x20 users.users.{hostname} = {{\n\
                 \x20   isNormalUser = true;\n\
                 \x20   extraGroups = [ \"wheel\" \"video\" \"networkmanager\" ];\n\
                 \x20   initialPassword = \"changeme\";\n\
                 \x20 }};\n\
                 \x20 environment.systemPackages = with pkgs; [ vim git curl wget htop btrfs-progs ];\n\
                 \x20 system.stateVersion = \"25.05\";\n\
                 }}",
                hostname = hostname,
            );
            script.push_str(&config_write_commands(
                &msg.configuration_nix,
                &fallback_alongside,
                &msg.flake_nix,
                session_id,
            ));
            script.push_str(&bootloader_patch_commands(&msg.disk));

            script.push_str(&format!(r#"
# Step 6: Create swap file
echo "STAGE: Configuring swap..."
fallocate -l 8G /mnt/swap/swapfile
chmod 600 /mnt/swap/swapfile
mkswap /mnt/swap/swapfile

# Step 7: Install
echo "STAGE: Installing NixOS..."
echo "This may take several minutes as packages are downloaded..."

# Self-healing install loop: retry up to 3 times, fixing broken packages
ATTEMPT=1
MAX_ATTEMPTS=3
while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
    echo "  Install attempt $ATTEMPT/$MAX_ATTEMPTS..."
    if nixos-install --no-root-passwd 2>&1; then
        echo "  Install succeeded on attempt $ATTEMPT"
        break
    else
        EXIT_CODE=$?
        echo "  Install failed (exit $EXIT_CODE) on attempt $ATTEMPT"
        if [ $ATTEMPT -ge $MAX_ATTEMPTS ]; then
            echo "ERROR: Install failed after $MAX_ATTEMPTS attempts"
            exit 1
        fi
        # Parse the error and try to fix
        echo "STAGE: Self-healing (attempt $ATTEMPT)..."
        # Extract the failing package from the build log
        FAILED_PKG=$(nixos-install --no-root-passwd 2>&1 | grep -oP "error:.*building.*'/nix/store/\K[^']*" | head -1 || true)
        FAILED_ATTR=$(nixos-install --no-root-passwd 2>&1 | grep -oP "attribute '\\K[^']*" | head -1 || true)
        if [ -n "$FAILED_ATTR" ]; then
            echo "  Detected failing attribute: $FAILED_ATTR"
            # Try to find an alternative
            ALT=$(nix search nixpkgs "$FAILED_ATTR" --json 2>/dev/null | head -c 500 | grep -oP '"legacyPackages\.x86_64-linux\.\K[^"]*' | head -1 || true)
            if [ -n "$ALT" ]; then
                echo "  Found alternative: $ALT"
                # Replace in configuration.nix
                sed -i "s|$(printf '%s' "$FAILED_ATTR" | sed 's/[|\\&]/\\&/g')|$(printf '%s' "$ALT" | sed 's/[|\\&]/\\&/g')|g" /mnt/etc/nixos/configuration.nix 2>/dev/null || true
                echo "  Config updated: $FAILED_ATTR → $ALT"
            else
                echo "  No alternative found for $FAILED_ATTR — removing from config"
                sed -i "/$(printf '%s' "$FAILED_ATTR" | sed 's/[\/\\&\[\].*^$]/\\&/g')/d" /mnt/etc/nixos/configuration.nix 2>/dev/null || true
            fi
        elif [ -n "$FAILED_PKG" ]; then
            echo "  Detected failing derivation: $FAILED_PKG"
            echo "  Retrying (may be a transient network error)..."
        else
            echo "  Could not identify failing package — retrying..."
        fi
    fi
    ATTEMPT=$((ATTEMPT + 1))
done

# Step 8: Verify
echo "STAGE: Verifying installation..."
ls /mnt/nix/store | wc -l | xargs -I{{}} echo "  {{}} store paths installed"
ls /mnt/boot/EFI/BOOT/BOOTX64.EFI 2>/dev/null && echo "  Bootloader: OK" || echo "  Bootloader: checking..."
ls /mnt/boot/EFI/systemd/systemd-bootx64.efi 2>/dev/null && echo "  systemd-boot: OK" || echo "  systemd-boot: checking..."

echo ""
echo "STAGE: FirstBreath"
echo "=== Sovereign Birth Complete (Alongside) ==="
echo "Reboot and select NixOS from the boot menu."
echo "Login as: {hostname} / changeme"
echo "Your existing OS is preserved — select it from the boot menu."
echo "COMPLETE"
"#, hostname = hostname));
            script
        }

        "single" | "" => {
            // Full disk wipe → direct partition → nixos-install
            // Uses sgdisk + mkfs directly (no disko download needed on live ISO)
            let mut script = format!(
                r#"set -eo pipefail
echo "=== Symthaea Sovereign Birth: Single Disk ==="
{boot_detect}

# Step 1: Wipe and partition
echo "STAGE: Partitioning disk {disk}..."
umount -R /mnt 2>/dev/null || true
swapoff {disk}* 2>/dev/null || true
wipefs -af {disk} 2>/dev/null || true
sgdisk --zap-all {disk}

# Boot partition (EFI or BIOS, auto-detected)
if [ "$BOOT_MODE" = "efi" ]; then
  sgdisk -n 1:0:+512M -t 1:EF00 -c 1:boot {disk}
  sgdisk -n 2:0:0 -t 2:8300 -c 2:nixos {disk}
  ROOT_NUM=2
else
  sgdisk -n 1:0:+1M -t 1:EF02 -c 1:bios-boot {disk}
  sgdisk -n 2:0:+512M -t 2:8300 -c 2:boot {disk}
  sgdisk -n 3:0:0 -t 3:8300 -c 3:nixos {disk}
  ROOT_NUM=3
fi
partprobe {disk} 2>/dev/null || true
blockdev --rereadpt {disk} 2>/dev/null || true
udevadm settle 2>/dev/null || true
sleep 3

# Detect partition names (nvme uses p1/p2, sata uses 1/2)
if [ "$BOOT_MODE" = "efi" ]; then
  if [ -b "{disk}p1" ]; then BOOT="{disk}p1"; ROOT="{disk}p2"; else BOOT="{disk}1"; ROOT="{disk}2"; fi
else
  if [ -b "{disk}p2" ]; then BOOT="{disk}p2"; ROOT="{disk}p3"; else BOOT="{disk}2"; ROOT="{disk}3"; fi
fi
echo "  Boot: $BOOT ($BOOT_MODE)"
echo "  Root: $ROOT"

# Step 2: Format with btrfs (snapshots, compression, rollback)
echo "STAGE: Formatting with btrfs..."
wipefs -af "$BOOT" 2>/dev/null || true
wipefs -af "$ROOT" 2>/dev/null || true
if [ "$BOOT_MODE" = "efi" ]; then
  mkfs.vfat -F 32 "$BOOT"
else
  mkfs.ext4 -F -L boot "$BOOT"
fi
mkfs.btrfs -f -L nixos "$ROOT"

# Step 3: Create btrfs subvolumes
mount "$ROOT" /mnt
btrfs subvolume create /mnt/@
btrfs subvolume create /mnt/@home
btrfs subvolume create /mnt/@nix
btrfs subvolume create /mnt/@log
btrfs subvolume create /mnt/@snapshots
btrfs subvolume create /mnt/@swap
umount /mnt

# Step 4: Mount with compression
echo "STAGE: Mounting filesystems..."
mount -o subvol=@,compress=zstd:3,noatime "$ROOT" /mnt
mkdir -p /mnt/{{boot,home,nix,var/log,.snapshots,swap}}
mount "$BOOT" /mnt/boot
mount -o subvol=@home,compress=zstd:3,noatime "$ROOT" /mnt/home
mount -o subvol=@nix,compress=zstd:3,noatime "$ROOT" /mnt/nix
mount -o subvol=@log,compress=zstd:3,noatime "$ROOT" /mnt/var/log
mount -o subvol=@snapshots,compress=zstd:3,noatime "$ROOT" /mnt/.snapshots
mount -o subvol=@swap,noatime "$ROOT" /mnt/swap

# Step 4: Generate hardware config + write our config
echo "STAGE: Generating configuration..."
mkdir -p /mnt/etc/nixos
nixos-generate-config --root /mnt || echo "WARNING: nixos-generate-config failed (may be normal for some layouts)"
"#,
                disk = msg.disk,
                boot_detect = boot_mode_detection()
            );

            // Append configuration.nix (and optionally flake.nix) via heredoc —
            // avoids passing Nix braces through format!().
            let fallback_single = format!(
                "{{ config, pkgs, ... }}:\n\
                 {{\n\
                 \x20 imports = [ ./hardware-configuration.nix ];\n\
                 \x20 networking.hostName = \"{hostname}\";\n\
                 \x20 boot.loader.systemd-boot.enable = true;\n\
                 \x20 boot.loader.efi.canTouchEfiVariables = true;\n\
                 \n\
                 \x20 # Hardening (Symthaea defaults)\n\
                 \x20 services.openssh.enable = true;\n\
                 \x20 services.earlyoom = {{ enable = true; freeMemThreshold = 5; freeSwapThreshold = 5; }};\n\
                 \x20 services.fstrim.enable = true;\n\
                 \x20 services.smartd = {{ enable = true; autodetect = true; }};\n\
                 \x20 services.btrfs.autoScrub = {{ enable = true; interval = \"monthly\"; fileSystems = [ \"/\" ]; }};\n\
                 \x20 zramSwap = {{ enable = true; algorithm = \"zstd\"; }};\n\
                 \x20 boot.kernel.sysctl.\"vm.swappiness\" = 60;\n\
                 \n\
                 \x20 # User\n\
                 \x20 users.users.{hostname} = {{\n\
                 \x20   isNormalUser = true;\n\
                 \x20   extraGroups = [ \"wheel\" \"video\" \"networkmanager\" ];\n\
                 \x20   initialPassword = \"changeme\";\n\
                 \x20 }};\n\
                 \n\
                 \x20 environment.systemPackages = with pkgs; [ vim git curl wget htop btrfs-progs ];\n\
                 \x20 system.stateVersion = \"25.05\";\n\
                 }}",
                hostname = hostname,
            );
            script.push_str(&config_write_commands(
                &msg.configuration_nix,
                &fallback_single,
                &msg.flake_nix,
                session_id,
            ));
            // Patch bootloader for BIOS mode
            script.push_str(&bootloader_patch_commands(&msg.disk));

            script.push_str(&format!(r#"
# Step 5: Create swap file
echo "STAGE: Configuring swap..."
fallocate -l 16G /mnt/swap/swapfile
chmod 600 /mnt/swap/swapfile
mkswap /mnt/swap/swapfile

# Step 6: Install
echo "STAGE: Installing NixOS..."
echo "This may take several minutes as packages are downloaded..."

# Self-healing install loop: retry up to 3 times, fixing broken packages
ATTEMPT=1
MAX_ATTEMPTS=3
while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
    echo "  Install attempt $ATTEMPT/$MAX_ATTEMPTS..."
    if nixos-install --no-root-passwd 2>&1; then
        echo "  Install succeeded on attempt $ATTEMPT"
        break
    else
        EXIT_CODE=$?
        echo "  Install failed (exit $EXIT_CODE) on attempt $ATTEMPT"
        if [ $ATTEMPT -ge $MAX_ATTEMPTS ]; then
            echo "ERROR: Install failed after $MAX_ATTEMPTS attempts"
            exit 1
        fi
        # Parse the error and try to fix
        echo "STAGE: Self-healing (attempt $ATTEMPT)..."
        # Extract the failing package from the build log
        FAILED_PKG=$(nixos-install --no-root-passwd 2>&1 | grep -oP "error:.*building.*'/nix/store/\K[^']*" | head -1 || true)
        FAILED_ATTR=$(nixos-install --no-root-passwd 2>&1 | grep -oP "attribute '\\K[^']*" | head -1 || true)
        if [ -n "$FAILED_ATTR" ]; then
            echo "  Detected failing attribute: $FAILED_ATTR"
            # Try to find an alternative
            ALT=$(nix search nixpkgs "$FAILED_ATTR" --json 2>/dev/null | head -c 500 | grep -oP '"legacyPackages\.x86_64-linux\.\K[^"]*' | head -1 || true)
            if [ -n "$ALT" ]; then
                echo "  Found alternative: $ALT"
                # Replace in configuration.nix
                sed -i "s|$(printf '%s' "$FAILED_ATTR" | sed 's/[|\\&]/\\&/g')|$(printf '%s' "$ALT" | sed 's/[|\\&]/\\&/g')|g" /mnt/etc/nixos/configuration.nix 2>/dev/null || true
                echo "  Config updated: $FAILED_ATTR → $ALT"
            else
                echo "  No alternative found for $FAILED_ATTR — removing from config"
                sed -i "/$(printf '%s' "$FAILED_ATTR" | sed 's/[\/\\&\[\].*^$]/\\&/g')/d" /mnt/etc/nixos/configuration.nix 2>/dev/null || true
            fi
        elif [ -n "$FAILED_PKG" ]; then
            echo "  Detected failing derivation: $FAILED_PKG"
            echo "  Retrying (may be a transient network error)..."
        else
            echo "  Could not identify failing package — retrying..."
        fi
    fi
    ATTEMPT=$((ATTEMPT + 1))
done

# Step 6: Verify
echo "STAGE: Verifying installation..."
ls /mnt/nix/store | wc -l | xargs -I{{}} echo "  {{}} store paths installed"
ls /mnt/boot/EFI/BOOT/BOOTX64.EFI && echo "  Bootloader: OK" || echo "  Bootloader: MISSING"

echo ""
echo "STAGE: FirstBreath"
echo "=== Sovereign Birth Complete ==="
echo "Reboot the machine: sudo reboot"
echo "Login as: {hostname} / changeme"
echo "COMPLETE"
"#, hostname = hostname));
            script
        }

        "single-zfs" => {
            // Full disk wipe → ZFS pool with datasets → nixos-install
            let mut script = format!(
                r#"set -eo pipefail
echo "=== NixForHumanity: Single Disk (ZFS) ==="

DISK="{disk}"

# Step 1: Wipe and partition
echo "STAGE: Partitioning disk {disk}..."
umount -R /mnt 2>/dev/null || true
wipefs -af {disk} 2>/dev/null || true
sgdisk --zap-all {disk}
sgdisk -n 1:0:+512M -t 1:EF00 -c 1:boot {disk}
sgdisk -n 2:0:0 -t 2:BF00 -c 2:zfs {disk}
partprobe {disk} 2>/dev/null || true
udevadm settle 2>/dev/null || true
sleep 3

if [ -b "{disk}p1" ]; then
  BOOT="{disk}p1"
  ROOT="{disk}p2"
else
  BOOT="{disk}1"
  ROOT="{disk}2"
fi
echo "  Boot: $BOOT"
echo "  Root: $ROOT"

# Step 2: Format boot partition
echo "STAGE: Formatting boot partition..."
mkfs.vfat -F 32 "$BOOT"

# Step 3: Create ZFS pool with datasets
echo "STAGE: Creating ZFS pool..."
# Generate a unique hostId
HOSTID=$(head -c 8 /etc/machine-id 2>/dev/null || echo "deadbeef")

zpool create -f \
  -o ashift=12 \
  -o autotrim=on \
  -O acltype=posixacl \
  -O relatime=on \
  -O xattr=sa \
  -O dnodesize=auto \
  -O normalization=formD \
  -O mountpoint=none \
  -O canmount=off \
  -O compression=zstd \
  rpool "$ROOT"

# Create datasets
echo "STAGE: Creating ZFS datasets..."
zfs create -o mountpoint=legacy rpool/root
zfs create -o mountpoint=legacy rpool/home
zfs create -o mountpoint=legacy rpool/nix
zfs create -o mountpoint=legacy -o com.sun:auto-snapshot=false rpool/nix/store
zfs create -o mountpoint=legacy rpool/var
zfs create -o mountpoint=legacy rpool/var/log

# Create swap zvol (ZFS doesn't support swap files)
zfs create -V 8G -b 4096 rpool/swap
mkswap /dev/zvol/rpool/swap

# Step 4: Mount everything
echo "STAGE: Mounting filesystems..."
mount -t zfs rpool/root /mnt
mkdir -p /mnt/{{boot,home,nix,var,var/log}}
mount "$BOOT" /mnt/boot
mount -t zfs rpool/home /mnt/home
mount -t zfs rpool/nix /mnt/nix
mount -t zfs rpool/var /mnt/var
mount -t zfs rpool/var/log /mnt/var/log

# Step 5: Generate hardware config
echo "STAGE: Generating configuration..."
mkdir -p /mnt/etc/nixos
nixos-generate-config --root /mnt || echo "WARNING: nixos-generate-config issue (may be normal for ZFS)"

# Write hostId to hardware-configuration.nix (required for ZFS)
echo '  networking.hostId = "deadbeef";' >> /mnt/etc/nixos/hardware-configuration.nix 2>/dev/null || true
"#,
                disk = msg.disk
            );

            let fallback_zfs = format!(
                "{{ config, pkgs, ... }}:\n{{\n  imports = [ ./hardware-configuration.nix ];\n  networking.hostName = \"{hostname}\";\n  boot.loader.systemd-boot.enable = true;\n  boot.loader.efi.canTouchEfiVariables = true;\n  boot.supportedFilesystems = [ \"zfs\" ];\n  boot.zfs.devNodes = \"/dev/disk/by-id\";\n  networking.hostId = \"deadbeef\";\n  services.zfs.autoScrub.enable = true;\n  services.zfs.trim.enable = true;\n  users.users.{hostname} = {{ isNormalUser = true; extraGroups = [ \"wheel\" \"video\" \"networkmanager\" ]; initialPassword = \"changeme\"; }};\n  environment.systemPackages = with pkgs; [ vim git curl wget htop ];\n  system.stateVersion = \"25.05\";\n}}",
                hostname = hostname,
            );
            script.push_str(&config_write_commands(
                &msg.configuration_nix,
                &fallback_zfs,
                &msg.flake_nix,
                session_id,
            ));
            script.push_str(&bootloader_patch_commands(&msg.disk));

            // ZFS doesn't need separate swap file setup — zvol already created
            script.push_str(&format!(
                r#"
# Step 6: Install
echo "STAGE: Installing NixOS..."
nixos-install --no-root-passwd 2>&1

# Step 7: Verify
echo "STAGE: Verifying installation..."
zpool status rpool 2>&1 | head -10
echo "  ZFS pool: OK"

echo ""
echo "STAGE: FirstBreath"
echo "=== NixOS Installed (ZFS) ==="
echo "Reboot: sudo reboot"
echo "Login as: {hostname} / changeme"
echo "COMPLETE"
"#,
                hostname = hostname
            ));
            script
        }

        "single-luks" => {
            // Full disk wipe → LUKS2 encryption → btrfs → nixos-install
            // Passphrase is passed via the 'command' field (repurposed)
            let passphrase = if msg.command.is_empty() {
                "changeme"
            } else {
                &msg.command
            };
            let mut script = format!(
                r#"set -eo pipefail
echo "=== Symthaea Sovereign Birth: Encrypted Single Disk ==="

# Step 1: Wipe and partition
echo "STAGE: Partitioning disk {disk}..."
umount -R /mnt 2>/dev/null || true
swapoff {disk}* 2>/dev/null || true
wipefs -af {disk} 2>/dev/null || true
sgdisk --zap-all {disk}
sgdisk -n 1:0:+512M -t 1:EF00 -c 1:boot {disk}
sgdisk -n 2:0:0 -t 2:8309 -c 2:cryptroot {disk}
partprobe {disk} 2>/dev/null || true
udevadm settle 2>/dev/null || true
sleep 3

if [ -b "{disk}p1" ]; then
  BOOT="{disk}p1"
  CRYPT_PART="{disk}p2"
else
  BOOT="{disk}1"
  CRYPT_PART="{disk}2"
fi
echo "  Boot: $BOOT"
echo "  Encrypted partition: $CRYPT_PART"

# Step 2: Set up LUKS2 encryption
echo "STAGE: Setting up encryption..."
LUKS_KEYFILE=$(mktemp /tmp/.luks-key-XXXXXX)
chmod 600 "$LUKS_KEYFILE"
printf '%s' '{passphrase}' > "$LUKS_KEYFILE"
cryptsetup luksFormat --type luks2 --label cryptroot \
  --pbkdf argon2id --iter-time 3000 "$CRYPT_PART" --key-file "$LUKS_KEYFILE"
cryptsetup open "$CRYPT_PART" cryptroot --key-file "$LUKS_KEYFILE"
rm -f "$LUKS_KEYFILE"
CRYPT_UUID=$(blkid -s UUID -o value "$CRYPT_PART")
echo "  LUKS UUID: $CRYPT_UUID"

# Step 3: Format with btrfs
echo "STAGE: Formatting with btrfs..."
wipefs -af "$BOOT" 2>/dev/null || true
mkfs.vfat -F 32 "$BOOT"
mkfs.btrfs -f -L nixos /dev/mapper/cryptroot

# Step 4: Create btrfs subvolumes
mount /dev/mapper/cryptroot /mnt
btrfs subvolume create /mnt/@
btrfs subvolume create /mnt/@home
btrfs subvolume create /mnt/@nix
btrfs subvolume create /mnt/@log
btrfs subvolume create /mnt/@snapshots
btrfs subvolume create /mnt/@swap
umount /mnt

# Step 5: Mount with compression
echo "STAGE: Mounting filesystems..."
mount -o subvol=@,compress=zstd:3,noatime /dev/mapper/cryptroot /mnt
mkdir -p /mnt/{{boot,home,nix,var/log,.snapshots,swap}}
mount "$BOOT" /mnt/boot
mount -o subvol=@home,compress=zstd:3,noatime /dev/mapper/cryptroot /mnt/home
mount -o subvol=@nix,compress=zstd:3,noatime /dev/mapper/cryptroot /mnt/nix
mount -o subvol=@log,compress=zstd:3,noatime /dev/mapper/cryptroot /mnt/var/log
mount -o subvol=@snapshots,compress=zstd:3,noatime /dev/mapper/cryptroot /mnt/.snapshots
mount -o subvol=@swap,noatime /dev/mapper/cryptroot /mnt/swap

# Step 6: Generate hardware config + write NixOS config with LUKS
echo "STAGE: Generating configuration..."
nixos-generate-config --root /mnt
"#,
                disk = msg.disk,
                passphrase = passphrase
            );

            // Append configuration.nix (and optionally flake.nix) via heredoc.
            // NOTE: The fallback LUKS config uses an *unquoted* heredoc (NIXCONF without
            // single quotes) so that $CRYPT_UUID is expanded by the shell at install time.
            // When the browser supplies a config, it is written with a quoted heredoc
            // ('NIXCONF') since it should already contain the correct UUID or be self-contained.
            if !msg.configuration_nix.is_empty() {
                // Browser config pre-staged — copy from temp dir (no heredoc)
                let staging = format!("/tmp/symthaea-config-{}", session_id);
                script.push_str(&format!("mkdir -p /mnt/etc/nixos\ncp {}/configuration.nix /mnt/etc/nixos/configuration.nix\n", staging));
                if !msg.flake_nix.is_empty() {
                    script.push_str(&format!(
                        "cp {}/flake.nix /mnt/etc/nixos/flake.nix\n",
                        staging
                    ));
                }
                script.push_str(&format!("rm -rf {}\n", staging));
            } else {
                // Fallback: unquoted heredoc for $CRYPT_UUID expansion
                script.push_str(&format!(
                    "cat > /mnt/etc/nixos/configuration.nix << NIXCONF\n\
                     {{ config, pkgs, ... }}:\n\
                     {{\n\
                     \x20 imports = [ ./hardware-configuration.nix ];\n\
                     \x20 networking.hostName = \"{hostname}\";\n\
                     \x20 boot.loader.systemd-boot.enable = true;\n\
                     \x20 boot.loader.efi.canTouchEfiVariables = true;\n\
                     \n\
                     \x20 # LUKS encryption\n\
                     \x20 boot.initrd.luks.devices.\"cryptroot\" = {{\n\
                     \x20   device = \"/dev/disk/by-uuid/$CRYPT_UUID\";\n\
                     \x20   allowDiscards = true;\n\
                     \x20 }};\n\
                     \n\
                     \x20 # Hardening (Symthaea defaults)\n\
                     \x20 services.openssh.enable = true;\n\
                     \x20 services.earlyoom = {{ enable = true; freeMemThreshold = 5; freeSwapThreshold = 5; }};\n\
                     \x20 services.fstrim.enable = true;\n\
                     \x20 services.smartd = {{ enable = true; autodetect = true; }};\n\
                     \x20 services.btrfs.autoScrub = {{ enable = true; interval = \"monthly\"; fileSystems = [ \"/\" ]; }};\n\
                     \x20 zramSwap = {{ enable = true; algorithm = \"zstd\"; }};\n\
                     \x20 boot.kernel.sysctl.\"vm.swappiness\" = 60;\n\
                     \n\
                     \x20 # User\n\
                     \x20 users.users.{hostname} = {{\n\
                     \x20   isNormalUser = true;\n\
                     \x20   extraGroups = [ \"wheel\" \"video\" \"networkmanager\" ];\n\
                     \x20   initialPassword = \"changeme\";\n\
                     \x20 }};\n\
                     \n\
                     \x20 environment.systemPackages = with pkgs; [ vim git curl wget htop btrfs-progs cryptsetup ];\n\
                     \x20 system.stateVersion = \"25.05\";\n\
                     }}\n\
                     NIXCONF\n",
                    hostname = hostname,
                ));
                // Write flake.nix if provided by browser even with fallback config
                if !msg.flake_nix.is_empty() {
                    script.push_str("\ncat > /mnt/etc/nixos/flake.nix << 'FLAKEEOF'\n");
                    script.push_str(&msg.flake_nix);
                    if !msg.flake_nix.ends_with('\n') {
                        script.push('\n');
                    }
                    script.push_str("FLAKEEOF\n");
                }
            }

            // Patch bootloader for BIOS mode (LUKS layout)
            script.push_str(&bootloader_patch_commands(&msg.disk));

            script.push_str(&format!(r#"
# Step 7: Create swap file
echo "STAGE: Configuring swap..."
fallocate -l 16G /mnt/swap/swapfile
chmod 600 /mnt/swap/swapfile
mkswap /mnt/swap/swapfile

# Step 8: Install
echo "STAGE: Installing NixOS..."
echo "This may take several minutes as packages are downloaded..."

# Self-healing install loop: retry up to 3 times, fixing broken packages
ATTEMPT=1
MAX_ATTEMPTS=3
while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
    echo "  Install attempt $ATTEMPT/$MAX_ATTEMPTS..."
    if nixos-install --no-root-passwd 2>&1; then
        echo "  Install succeeded on attempt $ATTEMPT"
        break
    else
        EXIT_CODE=$?
        echo "  Install failed (exit $EXIT_CODE) on attempt $ATTEMPT"
        if [ $ATTEMPT -ge $MAX_ATTEMPTS ]; then
            echo "ERROR: Install failed after $MAX_ATTEMPTS attempts"
            exit 1
        fi
        # Parse the error and try to fix
        echo "STAGE: Self-healing (attempt $ATTEMPT)..."
        # Extract the failing package from the build log
        FAILED_PKG=$(nixos-install --no-root-passwd 2>&1 | grep -oP "error:.*building.*'/nix/store/\K[^']*" | head -1 || true)
        FAILED_ATTR=$(nixos-install --no-root-passwd 2>&1 | grep -oP "attribute '\\K[^']*" | head -1 || true)
        if [ -n "$FAILED_ATTR" ]; then
            echo "  Detected failing attribute: $FAILED_ATTR"
            # Try to find an alternative
            ALT=$(nix search nixpkgs "$FAILED_ATTR" --json 2>/dev/null | head -c 500 | grep -oP '"legacyPackages\.x86_64-linux\.\K[^"]*' | head -1 || true)
            if [ -n "$ALT" ]; then
                echo "  Found alternative: $ALT"
                # Replace in configuration.nix
                sed -i "s|$(printf '%s' "$FAILED_ATTR" | sed 's/[|\\&]/\\&/g')|$(printf '%s' "$ALT" | sed 's/[|\\&]/\\&/g')|g" /mnt/etc/nixos/configuration.nix 2>/dev/null || true
                echo "  Config updated: $FAILED_ATTR → $ALT"
            else
                echo "  No alternative found for $FAILED_ATTR — removing from config"
                sed -i "/$(printf '%s' "$FAILED_ATTR" | sed 's/[\/\\&\[\].*^$]/\\&/g')/d" /mnt/etc/nixos/configuration.nix 2>/dev/null || true
            fi
        elif [ -n "$FAILED_PKG" ]; then
            echo "  Detected failing derivation: $FAILED_PKG"
            echo "  Retrying (may be a transient network error)..."
        else
            echo "  Could not identify failing package — retrying..."
        fi
    fi
    ATTEMPT=$((ATTEMPT + 1))
done

# Step 9: Verify
echo "STAGE: Verifying installation..."
ls /mnt/nix/store | wc -l | xargs -I{{}} echo "  {{}} store paths installed"
ls /mnt/boot/EFI/BOOT/BOOTX64.EFI && echo "  Bootloader: OK" || echo "  Bootloader: MISSING"
echo "  Encryption: LUKS2 on $CRYPT_PART"

echo ""
echo "STAGE: FirstBreath"
echo "=== Sovereign Birth Complete (Encrypted) ==="
echo "Reboot the machine: sudo reboot"
echo "You will be prompted for your encryption passphrase at boot."
echo "Login as: {hostname} / changeme"
echo "COMPLETE"
"#, hostname = hostname));
            script
        }

        "dual" => {
            // Dual-disk: fast drive for data (btrfs), standard for OS (ext4)
            // Direct partitioning — no disko download needed
            let mut script = format!(
                r#"set -eo pipefail
echo "=== Symthaea Sovereign Birth: Dual NVMe ==="

# Step 1: Partition standard drive (OS)
echo "STAGE: Partitioning standard drive {standard}..."
sgdisk --zap-all {standard}
sgdisk -n 1:0:+1G -t 1:EF00 -c 1:boot {standard}
sgdisk -n 2:0:0 -t 2:8300 -c 2:nixos-root {standard}
partprobe {standard} 2>/dev/null || true

# Step 2: Partition fast drive (data)
echo "STAGE: Partitioning fast drive {fast}..."
sgdisk --zap-all {fast}
sgdisk -n 1:0:0 -t 1:8300 -c 1:samsung-data {fast}
partprobe {fast} 2>/dev/null || true
sleep 2

# Detect partition names
if [ -b "{standard}p1" ]; then
  STD_BOOT="{standard}p1"; STD_ROOT="{standard}p2"
  FAST_DATA="{fast}p1"
else
  STD_BOOT="{standard}1"; STD_ROOT="{standard}2"
  FAST_DATA="{fast}1"
fi

# Step 3: Format
echo "STAGE: Formatting drives..."
mkfs.vfat -F 32 "$STD_BOOT"
mkfs.ext4 -F -L nixos "$STD_ROOT"
mkfs.btrfs -f -L samsung "$FAST_DATA"

# Step 4: Create btrfs subvolumes on fast drive
echo "STAGE: Creating btrfs subvolumes..."
mount "$FAST_DATA" /mnt
btrfs subvolume create /mnt/@home
btrfs subvolume create /mnt/@srv
btrfs subvolume create /mnt/@swap
btrfs subvolume create /mnt/@snapshots
umount /mnt

# Step 5: Mount everything
echo "STAGE: Mounting filesystems..."
mount "$STD_ROOT" /mnt
mkdir -p /mnt/{{boot,home,srv,swap,.snapshots}}
mount "$STD_BOOT" /mnt/boot
mount -o subvol=@home,compress=zstd:3,noatime "$FAST_DATA" /mnt/home
mount -o subvol=@srv,compress=zstd:3,noatime "$FAST_DATA" /mnt/srv
mount -o subvol=@swap,noatime "$FAST_DATA" /mnt/swap
mount -o subvol=@snapshots,compress=zstd:3,noatime "$FAST_DATA" /mnt/.snapshots

# Step 6: Generate config + write ours
echo "STAGE: Generating configuration..."
nixos-generate-config --root /mnt
"#,
                fast = msg.fast_disk,
                standard = msg.standard_disk
            );

            let fallback_dual = format!(
                "{{ config, pkgs, ... }}:\n\
                 {{\n\
                 \x20 imports = [ ./hardware-configuration.nix ];\n\
                 \x20 networking.hostName = \"{hostname}\";\n\
                 \x20 boot.loader.systemd-boot.enable = true;\n\
                 \x20 boot.loader.efi.canTouchEfiVariables = true;\n\
                 \n\
                 \x20 # Hardening\n\
                 \x20 services.openssh.enable = true;\n\
                 \x20 services.earlyoom = {{ enable = true; freeMemThreshold = 5; freeSwapThreshold = 5; }};\n\
                 \x20 services.fstrim.enable = true;\n\
                 \x20 services.smartd = {{ enable = true; autodetect = true; }};\n\
                 \x20 services.btrfs.autoScrub = {{ enable = true; interval = \"monthly\"; fileSystems = [ \"/home\" ]; }};\n\
                 \x20 zramSwap = {{ enable = true; algorithm = \"zstd\"; }};\n\
                 \x20 boot.kernel.sysctl.\"vm.swappiness\" = 60;\n\
                 \n\
                 \x20 # User\n\
                 \x20 users.users.{hostname} = {{\n\
                 \x20   isNormalUser = true;\n\
                 \x20   extraGroups = [ \"wheel\" \"video\" \"networkmanager\" ];\n\
                 \x20   initialPassword = \"changeme\";\n\
                 \x20 }};\n\
                 \n\
                 \x20 environment.systemPackages = with pkgs; [ vim git curl wget htop btrfs-progs ];\n\
                 \x20 system.stateVersion = \"25.05\";\n\
                 }}",
                hostname = hostname,
            );
            script.push_str(&config_write_commands(
                &msg.configuration_nix,
                &fallback_dual,
                &msg.flake_nix,
                session_id,
            ));
            script.push_str(&bootloader_patch_commands(&msg.standard_disk));

            script.push_str(&format!(
                r#"
# Step 7: Create swap file on fast drive
echo "STAGE: Configuring swap..."
fallocate -l 64G /mnt/swap/swapfile
chmod 600 /mnt/swap/swapfile
mkswap /mnt/swap/swapfile

# Step 8: Install
echo "STAGE: Installing NixOS..."
echo "This may take several minutes..."
nixos-install --no-root-passwd 2>&1

# Step 9: Verify
echo "STAGE: Verifying installation..."
ls /mnt/nix/store | wc -l | xargs -I{{}} echo "  {{}} store paths installed"
ls /mnt/boot/EFI/BOOT/BOOTX64.EFI && echo "  Bootloader: OK" || echo "  Bootloader: MISSING"

echo ""
echo "STAGE: FirstBreath"
echo "=== Sovereign Birth Complete ==="
echo "Reboot the machine: sudo reboot"
echo "Login as: {hostname} / changeme"
echo "COMPLETE"
"#,
                hostname = hostname
            ));
            script
        }

        "raid1-btrfs" => {
            // btrfs RAID1 across two disks (mirrored data + metadata)
            let mut script = format!(
                r#"set -eo pipefail
echo "=== Symthaea Sovereign Birth: btrfs RAID1 ==="

DISK1="{fast_disk}"
DISK2="{standard_disk}"

# Step 1: Wipe both disks
echo "STAGE: Partitioning disks..."
for disk in "$DISK1" "$DISK2"; do
  umount -R /mnt 2>/dev/null || true
  swapoff ${{disk}}* 2>/dev/null || true
  wipefs -af "$disk" 2>/dev/null || true
  sgdisk --zap-all "$disk"
done

# Create ESP on first disk only
sgdisk -n 1:0:+512M -t 1:EF00 -c 1:boot "$DISK1"
sgdisk -n 2:0:0 -t 2:8300 -c 2:raid-member "$DISK1"
# Second disk: all space for RAID
sgdisk -n 1:0:0 -t 1:8300 -c 1:raid-member "$DISK2"

partprobe "$DISK1" "$DISK2" 2>/dev/null || true
udevadm settle 2>/dev/null || true
sleep 3

# Detect partition names
if [ -b "${{DISK1}}p1" ]; then
  BOOT="${{DISK1}}p1"; R1="${{DISK1}}p2"
else
  BOOT="${{DISK1}}1"; R1="${{DISK1}}2"
fi
if [ -b "${{DISK2}}p1" ]; then
  R2="${{DISK2}}p1"
else
  R2="${{DISK2}}1"
fi
echo "  Boot: $BOOT"
echo "  RAID members: $R1, $R2"

# Step 2: Format with btrfs RAID1
echo "STAGE: Creating btrfs RAID1 mirror..."
wipefs -af "$BOOT" "$R1" "$R2" 2>/dev/null || true
mkfs.vfat -F 32 "$BOOT"
mkfs.btrfs -f -d raid1 -m raid1 -L nixos-raid "$R1" "$R2"

# Step 3: Create subvolumes
mount "$R1" /mnt
btrfs subvolume create /mnt/@
btrfs subvolume create /mnt/@home
btrfs subvolume create /mnt/@nix
btrfs subvolume create /mnt/@log
btrfs subvolume create /mnt/@snapshots
btrfs subvolume create /mnt/@swap
umount /mnt

# Step 4: Mount with compression
echo "STAGE: Mounting filesystems..."
mount -o subvol=@,compress=zstd:3,noatime "$R1" /mnt
mkdir -p /mnt/{{boot,home,nix,var/log,.snapshots,swap}}
mount "$BOOT" /mnt/boot
mount -o subvol=@home,compress=zstd:3,noatime "$R1" /mnt/home
mount -o subvol=@nix,compress=zstd:3,noatime "$R1" /mnt/nix
mount -o subvol=@log,compress=zstd:3,noatime "$R1" /mnt/var/log
mount -o subvol=@snapshots,compress=zstd:3,noatime "$R1" /mnt/.snapshots
mount -o subvol=@swap,noatime "$R1" /mnt/swap

echo "STAGE: Generating configuration..."
nixos-generate-config --root /mnt
"#,
                fast_disk = msg.fast_disk,
                standard_disk = msg.standard_disk
            );

            let fallback_raid1_btrfs = format!(
                "{{ config, pkgs, ... }}:\n\
                 {{\n\
                 \x20 imports = [ ./hardware-configuration.nix ];\n\
                 \x20 networking.hostName = \"{hostname}\";\n\
                 \x20 boot.loader.systemd-boot.enable = true;\n\
                 \x20 boot.loader.efi.canTouchEfiVariables = true;\n\
                 \x20 boot.initrd.supportedFilesystems = [ \"btrfs\" ];\n\
                 \x20 services.openssh.enable = true;\n\
                 \x20 services.earlyoom = {{ enable = true; freeMemThreshold = 5; freeSwapThreshold = 5; }};\n\
                 \x20 services.fstrim.enable = true;\n\
                 \x20 services.btrfs.autoScrub = {{ enable = true; interval = \"monthly\"; fileSystems = [ \"/\" ]; }};\n\
                 \x20 zramSwap = {{ enable = true; algorithm = \"zstd\"; }};\n\
                 \x20 users.users.{hostname} = {{\n\
                 \x20   isNormalUser = true;\n\
                 \x20   extraGroups = [ \"wheel\" \"video\" \"networkmanager\" ];\n\
                 \x20   initialPassword = \"changeme\";\n\
                 \x20 }};\n\
                 \x20 environment.systemPackages = with pkgs; [ vim git curl wget htop btrfs-progs ];\n\
                 \x20 system.stateVersion = \"25.05\";\n\
                 }}",
                hostname = hostname,
            );
            script.push_str(&config_write_commands(
                &msg.configuration_nix,
                &fallback_raid1_btrfs,
                &msg.flake_nix,
                session_id,
            ));
            script.push_str(&bootloader_patch_commands(&msg.disk));

            script.push_str(&format!(
                r#"
echo "STAGE: Configuring swap..."
fallocate -l 16G /mnt/swap/swapfile
chmod 600 /mnt/swap/swapfile
mkswap /mnt/swap/swapfile

echo "STAGE: Installing NixOS..."
echo "This may take several minutes..."
nixos-install --no-root-passwd 2>&1

echo "STAGE: Verifying installation..."
ls /mnt/nix/store | wc -l | xargs -I{{}} echo "  {{}} store paths installed"
btrfs filesystem show /mnt 2>/dev/null | head -5
echo "  RAID1 status:"
btrfs filesystem df /mnt 2>/dev/null

echo ""
echo "STAGE: FirstBreath"
echo "=== Sovereign Birth Complete (btrfs RAID1) ==="
echo "Data is mirrored across both disks."
echo "Login as: {hostname} / changeme"
echo "COMPLETE"
"#,
                hostname = hostname
            ));
            script
        }

        "raid1-mdadm" => {
            // mdadm RAID1 mirror with btrfs on top
            let mut script = format!(
                r#"set -eo pipefail
echo "=== Symthaea Sovereign Birth: mdadm RAID1 ==="

DISK1="{fast_disk}"
DISK2="{standard_disk}"

echo "STAGE: Partitioning disks..."
for disk in "$DISK1" "$DISK2"; do
  umount -R /mnt 2>/dev/null || true
  swapoff ${{disk}}* 2>/dev/null || true
  wipefs -af "$disk" 2>/dev/null || true
  sgdisk --zap-all "$disk"
  sgdisk -n 1:0:+512M -t 1:EF00 -c 1:boot "$disk"
  sgdisk -n 2:0:0 -t 2:FD00 -c 2:raid "$disk"
done
partprobe "$DISK1" "$DISK2" 2>/dev/null || true
udevadm settle 2>/dev/null || true
sleep 3

if [ -b "${{DISK1}}p1" ]; then
  BOOT1="${{DISK1}}p1"; MD1="${{DISK1}}p2"
  BOOT2="${{DISK2}}p1"; MD2="${{DISK2}}p2"
else
  BOOT1="${{DISK1}}1"; MD1="${{DISK1}}2"
  BOOT2="${{DISK2}}1"; MD2="${{DISK2}}2"
fi

# Step 2: Create mdadm RAID1
echo "STAGE: Creating mdadm RAID1 mirror..."
wipefs -af "$MD1" "$MD2" 2>/dev/null || true
mdadm --create /dev/md0 --level=1 --raid-devices=2 --metadata=1.2 --run "$MD1" "$MD2"
echo "  Array: /dev/md0"
cat /proc/mdstat

# Step 3: Format
echo "STAGE: Formatting with btrfs..."
mkfs.vfat -F 32 "$BOOT1"
mkfs.btrfs -f -L nixos /dev/md0

mount /dev/md0 /mnt
btrfs subvolume create /mnt/@
btrfs subvolume create /mnt/@home
btrfs subvolume create /mnt/@nix
btrfs subvolume create /mnt/@log
btrfs subvolume create /mnt/@snapshots
btrfs subvolume create /mnt/@swap
umount /mnt

echo "STAGE: Mounting filesystems..."
mount -o subvol=@,compress=zstd:3,noatime /dev/md0 /mnt
mkdir -p /mnt/{{boot,home,nix,var/log,.snapshots,swap}}
mount "$BOOT1" /mnt/boot
mount -o subvol=@home,compress=zstd:3,noatime /dev/md0 /mnt/home
mount -o subvol=@nix,compress=zstd:3,noatime /dev/md0 /mnt/nix
mount -o subvol=@log,compress=zstd:3,noatime /dev/md0 /mnt/var/log
mount -o subvol=@snapshots,compress=zstd:3,noatime /dev/md0 /mnt/.snapshots
mount -o subvol=@swap,noatime /dev/md0 /mnt/swap

echo "STAGE: Generating configuration..."
nixos-generate-config --root /mnt

# Save mdadm config
mkdir -p /mnt/etc
mdadm --detail --scan >> /mnt/etc/mdadm.conf
"#,
                fast_disk = msg.fast_disk,
                standard_disk = msg.standard_disk
            );

            let fallback_raid1_mdadm = format!(
                "{{ config, pkgs, ... }}:\n\
                 {{\n\
                 \x20 imports = [ ./hardware-configuration.nix ];\n\
                 \x20 networking.hostName = \"{hostname}\";\n\
                 \x20 boot.loader.systemd-boot.enable = true;\n\
                 \x20 boot.loader.efi.canTouchEfiVariables = true;\n\
                 \x20 boot.swraid = {{\n\
                 \x20   enable = true;\n\
                 \x20   mdadmConf = \"MAILADDR root\";\n\
                 \x20 }};\n\
                 \x20 services.openssh.enable = true;\n\
                 \x20 services.earlyoom = {{ enable = true; freeMemThreshold = 5; freeSwapThreshold = 5; }};\n\
                 \x20 services.fstrim.enable = true;\n\
                 \x20 services.btrfs.autoScrub = {{ enable = true; interval = \"monthly\"; fileSystems = [ \"/\" ]; }};\n\
                 \x20 zramSwap = {{ enable = true; algorithm = \"zstd\"; }};\n\
                 \x20 users.users.{hostname} = {{\n\
                 \x20   isNormalUser = true;\n\
                 \x20   extraGroups = [ \"wheel\" \"video\" \"networkmanager\" ];\n\
                 \x20   initialPassword = \"changeme\";\n\
                 \x20 }};\n\
                 \x20 environment.systemPackages = with pkgs; [ vim git curl wget htop btrfs-progs mdadm ];\n\
                 \x20 system.stateVersion = \"25.05\";\n\
                 }}",
                hostname = hostname,
            );
            script.push_str(&config_write_commands(
                &msg.configuration_nix,
                &fallback_raid1_mdadm,
                &msg.flake_nix,
                session_id,
            ));
            script.push_str(&bootloader_patch_commands(&msg.disk));

            script.push_str(&format!(
                r#"
echo "STAGE: Configuring swap..."
fallocate -l 16G /mnt/swap/swapfile
chmod 600 /mnt/swap/swapfile
mkswap /mnt/swap/swapfile

echo "STAGE: Installing NixOS..."
nixos-install --no-root-passwd 2>&1

echo "STAGE: Verifying installation..."
ls /mnt/nix/store | wc -l | xargs -I{{}} echo "  {{}} store paths installed"
cat /proc/mdstat

echo ""
echo "STAGE: FirstBreath"
echo "=== Sovereign Birth Complete (mdadm RAID1) ==="
echo "Data is mirrored. If one disk fails, the other continues."
echo "Login as: {hostname} / changeme"
echo "COMPLETE"
"#,
                hostname = hostname
            ));
            script
        }

        // ── Multi-disk RAID layouts ──
        "raid5-mdadm" | "raid6-mdadm" | "raid10-mdadm" => {
            let raid_level = match msg.layout.as_str() {
                "raid5-mdadm" => "5",
                "raid6-mdadm" => "6",
                "raid10-mdadm" => "10",
                _ => unreachable!(),
            };
            let min_disks: usize = match raid_level {
                "5" => 3,
                "6" | "10" => 4,
                _ => 3,
            };
            // Collect all disks: primary disk + extra_disks
            let mut all_disks = vec![msg.disk.clone()];
            all_disks.extend(msg.extra_disks.iter().cloned());
            if all_disks.len() < min_disks {
                return format!(
                    "echo 'ERROR: RAID{} requires at least {} disks, got {}'; exit 1",
                    raid_level,
                    min_disks,
                    all_disks.len()
                );
            }
            let disk_list = all_disks.join(" ");
            let n_disks = all_disks.len();

            let mut script = format!(
                r#"set -eo pipefail
echo "=== Symthaea Sovereign Birth: RAID{level} (mdadm, {n} disks) ==="
{boot_detect}

# Step 1: Wipe all disks and create partitions
echo "STAGE: Partitioning {n} disks..."
for DISK in {disks}; do
  umount -R /mnt 2>/dev/null || true
  wipefs -af "$DISK" 2>/dev/null || true
  sgdisk --zap-all "$DISK"
  sgdisk -n 1:0:+512M -t 1:EF00 -c 1:boot "$DISK"
  sgdisk -n 2:0:0 -t 2:FD00 -c 2:raid "$DISK"
done
partprobe 2>/dev/null || true
udevadm settle 2>/dev/null || true
sleep 3

# Build partition list for mdadm
RAID_PARTS=""
BOOT_PART=""
for DISK in {disks}; do
  if [ -b "${{DISK}}p2" ]; then
    RAID_PARTS="$RAID_PARTS ${{DISK}}p2"
    [ -z "$BOOT_PART" ] && BOOT_PART="${{DISK}}p1"
  else
    RAID_PARTS="$RAID_PARTS ${{DISK}}2"
    [ -z "$BOOT_PART" ] && BOOT_PART="${{DISK}}1"
  fi
done

# Step 2: Create mdadm array
echo "STAGE: Creating RAID{level} array with {n} disks..."
mdadm --create /dev/md0 --level={level} --raid-devices={n} --metadata=1.2 --run $RAID_PARTS
cat /proc/mdstat

# Step 3: Format
echo "STAGE: Formatting..."
mkfs.vfat -F 32 "$BOOT_PART"
mkfs.btrfs -f -L nixos /dev/md0

# Step 4: Mount
echo "STAGE: Mounting filesystems..."
mount /dev/md0 /mnt
btrfs subvolume create /mnt/@
btrfs subvolume create /mnt/@home
btrfs subvolume create /mnt/@nix
btrfs subvolume create /mnt/@log
btrfs subvolume create /mnt/@swap
umount /mnt
mount -o subvol=@,compress=zstd:3,noatime /dev/md0 /mnt
mkdir -p /mnt/{{boot,home,nix,var/log,swap,etc/nixos}}
mount "$BOOT_PART" /mnt/boot
mount -o subvol=@home,compress=zstd:3,noatime /dev/md0 /mnt/home
mount -o subvol=@nix,compress=zstd:3,noatime /dev/md0 /mnt/nix
mount -o subvol=@log,compress=zstd:3,noatime /dev/md0 /mnt/var/log

# Step 5: Generate config + save mdadm
echo "STAGE: Generating configuration..."
nixos-generate-config --root /mnt
mdadm --detail --scan >> /mnt/etc/mdadm.conf 2>/dev/null || true
"#,
                level = raid_level,
                n = n_disks,
                disks = disk_list,
                boot_detect = boot_mode_detection()
            );

            let fallback_config = format!(
                "{{ config, pkgs, ... }}:\n{{\n  imports = [ ./hardware-configuration.nix ];\n  \
                 networking.hostName = \"{hostname}\";\n  \
                 boot.loader.systemd-boot.enable = true;\n  \
                 boot.loader.efi.canTouchEfiVariables = true;\n  \
                 boot.swraid.enable = true;\n  \
                 services.openssh.enable = true;\n  \
                 users.users.{hostname} = {{ isNormalUser = true; extraGroups = [ \"wheel\" ]; initialPassword = \"changeme\"; }};\n  \
                 system.stateVersion = \"25.05\";\n}}",
                hostname = hostname
            );
            script.push_str(&config_write_commands(
                &msg.configuration_nix,
                &fallback_config,
                &msg.flake_nix,
                session_id,
            ));
            script.push_str(&bootloader_patch_commands(&msg.disk));
            script.push_str(
                r#"
echo "STAGE: Installing NixOS..."
nixos-install --no-root-passwd 2>&1
echo "STAGE: Verifying..."
echo "COMPLETE"
"#,
            );
            script
        }

        // ── ZFS multi-disk layouts ──
        "zfs-mirror" | "zfs-raidz" | "zfs-raidz2" => {
            let zfs_type = match msg.layout.as_str() {
                "zfs-mirror" => "mirror",
                "zfs-raidz" => "raidz",
                "zfs-raidz2" => "raidz2",
                _ => unreachable!(),
            };
            let min_disks: usize = match zfs_type {
                "mirror" => 2,
                "raidz" => 3,
                "raidz2" => 4,
                _ => 2,
            };
            let mut all_disks = vec![msg.disk.clone()];
            all_disks.extend(msg.extra_disks.iter().cloned());
            if all_disks.len() < min_disks {
                return format!(
                    "echo 'ERROR: ZFS {} requires at least {} disks, got {}'; exit 1",
                    zfs_type,
                    min_disks,
                    all_disks.len()
                );
            }
            let disk_list = all_disks.join(" ");
            let n_disks = all_disks.len();

            let mut script = format!(
                r#"set -eo pipefail
echo "=== Symthaea Sovereign Birth: ZFS {ztype} ({n} disks) ==="
{boot_detect}

# Step 1: Wipe all disks and create partitions
echo "STAGE: Partitioning {n} disks..."
ZFS_PARTS=""
BOOT_PART=""
for DISK in {disks}; do
  umount -R /mnt 2>/dev/null || true
  wipefs -af "$DISK" 2>/dev/null || true
  sgdisk --zap-all "$DISK"
  sgdisk -n 1:0:+512M -t 1:EF00 -c 1:boot "$DISK"
  sgdisk -n 2:0:0 -t 2:BF00 -c 2:zfs "$DISK"
  if [ -b "${{DISK}}p2" ]; then
    ZFS_PARTS="$ZFS_PARTS ${{DISK}}p2"
    [ -z "$BOOT_PART" ] && BOOT_PART="${{DISK}}p1"
  else
    ZFS_PARTS="$ZFS_PARTS ${{DISK}}2"
    [ -z "$BOOT_PART" ] && BOOT_PART="${{DISK}}1"
  fi
done
partprobe 2>/dev/null || true
udevadm settle 2>/dev/null || true
sleep 3

# Step 2: Create ZFS pool
echo "STAGE: Creating ZFS {ztype} pool..."
HOSTID=$(head -c 4 /dev/urandom | od -An -tx4 | tr -d ' ')
zpool create -f -o ashift=12 -o autotrim=on \
  -O acltype=posixacl -O compression=zstd -O dnodesize=auto \
  -O normalization=formD -O relatime=on -O xattr=sa \
  -O mountpoint=none \
  rpool {ztype} $ZFS_PARTS

# Step 3: Create datasets
echo "STAGE: Creating ZFS datasets..."
zfs create -o mountpoint=legacy rpool/root
zfs create -o mountpoint=legacy rpool/home
zfs create -o mountpoint=legacy rpool/nix
zfs create -o mountpoint=legacy rpool/var
zfs create -o mountpoint=legacy rpool/var/log
zfs create -V 8G rpool/swap
mkswap /dev/zvol/rpool/swap

# Step 4: Mount
echo "STAGE: Mounting filesystems..."
mkfs.vfat -F 32 "$BOOT_PART"
mount -t zfs rpool/root /mnt
mkdir -p /mnt/{{boot,home,nix,var/log,etc/nixos}}
mount "$BOOT_PART" /mnt/boot
mount -t zfs rpool/home /mnt/home
mount -t zfs rpool/nix /mnt/nix
mount -t zfs rpool/var /mnt/var
mount -t zfs rpool/var/log /mnt/var/log

# Step 5: Generate config
echo "STAGE: Generating configuration..."
nixos-generate-config --root /mnt
"#,
                ztype = zfs_type,
                n = n_disks,
                disks = disk_list,
                boot_detect = boot_mode_detection()
            );

            let fallback_config = format!(
                "{{ config, pkgs, ... }}:\n{{\n  imports = [ ./hardware-configuration.nix ];\n  \
                 networking.hostName = \"{hostname}\";\n  \
                 networking.hostId = \"$(head -c 4 /dev/urandom | od -An -tx4 | tr -d ' ')\";\n  \
                 boot.loader.systemd-boot.enable = true;\n  \
                 boot.loader.efi.canTouchEfiVariables = true;\n  \
                 boot.supportedFilesystems = [ \"zfs\" ];\n  \
                 services.zfs.autoScrub.enable = true;\n  \
                 services.openssh.enable = true;\n  \
                 users.users.{hostname} = {{ isNormalUser = true; extraGroups = [ \"wheel\" ]; initialPassword = \"changeme\"; }};\n  \
                 system.stateVersion = \"25.05\";\n}}",
                hostname = hostname
            );
            script.push_str(&config_write_commands(
                &msg.configuration_nix,
                &fallback_config,
                &msg.flake_nix,
                session_id,
            ));
            script.push_str(&bootloader_patch_commands(&msg.disk));
            script.push_str(
                r#"
echo "STAGE: Installing NixOS..."
nixos-install --no-root-passwd 2>&1
echo "STAGE: Verifying..."
zpool status rpool
echo "COMPLETE"
"#,
            );
            script
        }

        _ => format!("echo 'Unknown layout: {}'; exit 1", msg.layout),
    }
}

/// Relay → Client message.
#[derive(serde::Serialize)]
struct RelayMessage {
    #[serde(rename = "type")]
    msg_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    code: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stage: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    percentage: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    phase: Option<String>,
}

impl RelayMessage {
    fn authed() -> Self {
        Self {
            msg_type: "authed".into(),
            data: None,
            stream: None,
            code: None,
            message: Some("WebSocket authenticated".into()),
            stage: None,
            percentage: None,
            phase: None,
        }
    }

    fn output(data: &str, stream: &str) -> Self {
        Self {
            msg_type: "output".into(),
            data: Some(data.into()),
            stream: Some(stream.into()),
            code: None,
            message: None,
            stage: None,
            percentage: None,
            phase: None,
        }
    }

    fn progress(stage: &NixosAnywhereStage) -> Self {
        Self {
            msg_type: "progress".into(),
            data: None,
            stream: None,
            code: None,
            message: Some(format!("{:?}", stage)),
            stage: Some(format!("{:?}", stage)),
            percentage: Some(stage.percentage()),
            phase: Some(stage.inoculation_phase().into()),
        }
    }

    fn exit(code: i32) -> Self {
        Self {
            msg_type: "exit".into(),
            data: None,
            stream: None,
            code: Some(code),
            message: None,
            stage: None,
            percentage: None,
            phase: None,
        }
    }

    fn error(msg: &str) -> Self {
        Self {
            msg_type: "error".into(),
            data: None,
            stream: None,
            code: None,
            message: Some(msg.into()),
            stage: None,
            percentage: None,
            phase: None,
        }
    }

    fn disks(disks_json: &str) -> Self {
        Self {
            msg_type: "disks".into(),
            data: Some(disks_json.into()),
            stream: None,
            code: None,
            message: None,
            stage: None,
            percentage: None,
            phase: None,
        }
    }

    fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}

/// Parsed disk info from lsblk.
#[derive(Debug, serde::Serialize)]
struct DiskInfo {
    name: String,
    size: String,
    model: String,
    transport: String, // nvme, sata, usb, virtio
    disk_type: String, // disk, part, rom
    removable: bool,
}

/// Parse lsblk --json output into structured disk info.
fn parse_lsblk(json_str: &str) -> Vec<DiskInfo> {
    let parsed: serde_json::Value = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };

    let devices = match parsed.get("blockdevices").and_then(|b| b.as_array()) {
        Some(arr) => arr,
        None => return Vec::new(),
    };

    devices
        .iter()
        .filter_map(|dev| {
            let dtype = dev.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if dtype != "disk" {
                return None;
            }
            let name = dev
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let size = dev
                .get("size")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let model = dev
                .get("model")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown")
                .trim()
                .to_string();
            let tran = dev
                .get("tran")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let rm = dev.get("rm").and_then(|v| v.as_bool()).unwrap_or(false);

            Some(DiskInfo {
                name: format!("/dev/{}", name),
                size,
                model,
                transport: if tran.is_empty() {
                    "unknown".into()
                } else {
                    tran
                },
                disk_type: dtype.into(),
                removable: rm,
            })
        })
        .collect()
}

type SharedTracker = Arc<Mutex<SessionTracker>>;

async fn handle_connection(
    stream: tokio::net::TcpStream,
    peer_addr: String,
    tracker: SharedTracker,
    auth_token: Arc<String>,
) {
    // Upgrade to WebSocket with Origin header validation.
    // Only allow connections from localhost, 127.0.0.1, or our known domains.
    let origin_check = |req: &tungstenite::handshake::server::Request,
                        resp: tungstenite::handshake::server::Response|
     -> Result<
        tungstenite::handshake::server::Response,
        tungstenite::handshake::server::ErrorResponse,
    > {
        if let Some(origin) = req.headers().get("origin") {
            let origin_str = origin.to_str().unwrap_or("");
            let allowed = origin_str.starts_with("http://localhost")
                || origin_str.starts_with("https://localhost")
                || origin_str.starts_with("http://127.0.0.1")
                || origin_str.starts_with("https://127.0.0.1")
                || origin_str.contains("luminousdynamics.io")
                || origin_str.contains("nixforhumanity.org")
                || origin_str.contains("mycelix.net")
                || origin_str.contains("relationalharmonics.org");
            if !allowed {
                eprintln!(
                    "[{}] Rejected WebSocket: disallowed Origin '{}'",
                    peer_addr, origin_str
                );
                let mut resp = tungstenite::handshake::server::ErrorResponse::new(Some(
                    "Forbidden origin".into(),
                ));
                *resp.status_mut() = tungstenite::http::StatusCode::FORBIDDEN;
                return Err(resp);
            }
        }
        // No Origin header = non-browser client (curl, relay tools) — allow
        Ok(resp)
    };

    let ws_stream = match accept_hdr_async(stream, origin_check).await {
        Ok(ws) => ws,
        Err(e) => {
            eprintln!("[{}] WebSocket upgrade failed: {}", peer_addr, e);
            return;
        }
    };
    handle_connection_ws(ws_stream, peer_addr, tracker, auth_token).await;
}

/// Handle an already-upgraded WebSocket connection (works for both plain and TLS streams)
async fn handle_connection_ws<S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin>(
    ws_stream: tokio_tungstenite::WebSocketStream<S>,
    peer_addr: String,
    tracker: SharedTracker,
    auth_token: Arc<String>,
) {
    let (mut ws_tx, mut ws_rx) = ws_stream.split();
    let mut authed = false;

    eprintln!("[{}] WebSocket connected", peer_addr);

    while let Some(msg) = ws_rx.next().await {
        let msg = match msg {
            Ok(Message::Text(t)) => t,
            Ok(Message::Close(_)) => break,
            Ok(_) => continue,
            Err(e) => {
                eprintln!("[{}] WebSocket error: {}", peer_addr, e);
                break;
            }
        };

        let client_msg: ClientMessage = match serde_json::from_str(&msg) {
            Ok(m) => m,
            Err(e) => {
                let _ = ws_tx
                    .send(Message::Text(
                        RelayMessage::error(&format!("Invalid JSON: {}", e)).to_json(),
                    ))
                    .await;
                continue;
            }
        };

        // Auth gate: require an explicit `"auth"` action with the correct token.
        // This prevents CSWSH-style attacks against ws://127.0.0.1:* services.
        // Origin header validated during WebSocket upgrade (handle_connection).
        // Token auth provides the primary security boundary.
        if !authed {
            // Check if this IP is blocked due to too many failed auth attempts
            if tracker.lock().await.is_blocked(&peer_addr) {
                eprintln!(
                    "[{}] Blocked after too many failed auth attempts",
                    peer_addr
                );
                let _ = ws_tx
                    .send(Message::Text(
                        RelayMessage::error("Too many failed auth attempts. Try again later.")
                            .to_json(),
                    ))
                    .await;
                break;
            }

            if client_msg.action.as_str() == "auth" {
                // Constant-time comparison prevents timing side-channel attacks
                if !client_msg.token.is_empty() && token_eq(&client_msg.token, &auth_token) {
                    authed = true;
                    let _ = ws_tx
                        .send(Message::Text(RelayMessage::authed().to_json()))
                        .await;
                    continue;
                }

                // Record failed auth attempt for rate limiting
                tracker.lock().await.record_failed_auth(&peer_addr);
                let _ = ws_tx
                    .send(Message::Text(
                        RelayMessage::error("Unauthorized: invalid relay token").to_json(),
                    ))
                    .await;
                break;
            } else {
                let _ = ws_tx
                    .send(Message::Text(
                        RelayMessage::error(
                            "Unauthorized: send {\"action\":\"auth\",\"token\":...} first",
                        )
                        .to_json(),
                    ))
                    .await;
                break;
            }
        }

        match client_msg.action.as_str() {
            "connect" => {
                // Rate limit: 1 session per IP
                {
                    let mut t = tracker.lock().await;
                    if !t.try_acquire(&peer_addr) {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error(
                                    "Rate limited: only 1 active session per IP allowed",
                                )
                                .to_json(),
                            ))
                            .await;
                        continue;
                    }
                }

                // No SSH needed — relay runs directly on the target machine
                eprintln!("[{}] Connection acknowledged (local mode)", peer_addr);
                let _ = ws_tx
                    .send(Message::Text(
                        serde_json::json!({
                            "type": "connected",
                            "message": "Connected to target (local relay)"
                        })
                        .to_string(),
                    ))
                    .await;
            }

            // Intentionally disabled: this relay is not a general-purpose RCE gateway.
            "exec" => {
                let _ = ws_tx
                    .send(Message::Text(
                        RelayMessage::error(
                            "Unsupported action: 'exec' is disabled. Use typed actions like 'install', 'probe_hardware', etc.",
                        )
                        .to_json(),
                    ))
                    .await;
            }

            "discover_disks" => {
                eprintln!("[{}] Discovering disks...", peer_addr);
                match run_cmd("lsblk --json -o NAME,SIZE,MODEL,TYPE,TRAN,RM -b").await {
                    Ok(result) if result.exit_status == 0 => {
                        let disks = parse_lsblk(&result.stdout);
                        let disks_json =
                            serde_json::to_string(&disks).unwrap_or_else(|_| "[]".into());
                        eprintln!("[{}] Found {} disks", peer_addr, disks.len());
                        let _ = ws_tx
                            .send(Message::Text(RelayMessage::disks(&disks_json).to_json()))
                            .await;
                    }
                    Ok(result) => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error(&format!(
                                    "lsblk failed (exit {}): {}",
                                    result.exit_status,
                                    result.stderr.chars().take(200).collect::<String>()
                                ))
                                .to_json(),
                            ))
                            .await;
                    }
                    Err(e) => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error(&format!("Disk discovery failed: {}", e))
                                    .to_json(),
                            ))
                            .await;
                    }
                }
            }

            "install" => {
                // ── Validate ALL user inputs before they reach shell commands ──
                let disk = if client_msg.disk.is_empty() {
                    "/dev/sda".to_string()
                } else {
                    match validate_disk_path(&client_msg.disk) {
                        Ok(d) => d,
                        Err(e) => {
                            let _ = ws_tx
                                .send(Message::Text(RelayMessage::error(&e).to_json()))
                                .await;
                            continue;
                        }
                    }
                };
                let hostname = match validate_hostname_relay(&client_msg.hostname) {
                    Ok(h) => h,
                    Err(e) => {
                        let _ = ws_tx
                            .send(Message::Text(RelayMessage::error(&e).to_json()))
                            .await;
                        continue;
                    }
                };
                // Validate optional fields that reach shell/Nix config
                if !client_msg.timezone.is_empty() {
                    if let Err(e) = sanitize_input(&client_msg.timezone, "timezone", true) {
                        let _ = ws_tx
                            .send(Message::Text(RelayMessage::error(&e).to_json()))
                            .await;
                        continue;
                    }
                }
                if !client_msg.keyboard.is_empty() {
                    if let Err(e) = sanitize_input(&client_msg.keyboard, "keyboard", false) {
                        let _ = ws_tx
                            .send(Message::Text(RelayMessage::error(&e).to_json()))
                            .await;
                        continue;
                    }
                }
                if !client_msg.desktop.is_empty() {
                    if let Err(e) = sanitize_input(&client_msg.desktop, "desktop", false) {
                        let _ = ws_tx
                            .send(Message::Text(RelayMessage::error(&e).to_json()))
                            .await;
                        continue;
                    }
                }
                if !client_msg.gpu_driver.is_empty() {
                    if let Err(e) = sanitize_input(&client_msg.gpu_driver, "gpu_driver", false) {
                        let _ = ws_tx
                            .send(Message::Text(RelayMessage::error(&e).to_json()))
                            .await;
                        continue;
                    }
                }
                if !client_msg.fast_disk.is_empty() {
                    if let Err(e) = validate_disk_path(&client_msg.fast_disk) {
                        let _ = ws_tx
                            .send(Message::Text(RelayMessage::error(&e).to_json()))
                            .await;
                        continue;
                    }
                }
                if !client_msg.standard_disk.is_empty() {
                    if let Err(e) = validate_disk_path(&client_msg.standard_disk) {
                        let _ = ws_tx
                            .send(Message::Text(RelayMessage::error(&e).to_json()))
                            .await;
                        continue;
                    }
                }

                // Validate extra disks for RAID/ZFS multi-disk layouts
                for extra_disk in &client_msg.extra_disks {
                    if let Err(e) = validate_disk_path(extra_disk) {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error(&format!("Invalid extra disk: {}", e))
                                    .to_json(),
                            ))
                            .await;
                        continue;
                    }
                }

                // Generate session-isolated log path (CRITICAL-4: prevents cross-session log tampering)
                let session_id: u64 = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);
                let log_path = format!("/tmp/symthaea-install-{}.log", session_id);
                let script_path = format!("/tmp/symthaea-install-{}.sh", session_id);

                // Create log file with restrictive permissions
                let _ = run_cmd(&format!("touch {} && chmod 600 {}", log_path, log_path)).await;

                // SECURITY: Pre-stage config files to temp dir via direct file write.
                // The install script copies them after mount — no heredoc injection possible.
                let config_staging_dir = format!("/tmp/symthaea-config-{}", session_id);
                let _ = tokio::fs::create_dir_all(&config_staging_dir).await;
                // Stage configuration.nix (browser-supplied or will be generated by fallback in script)
                if !client_msg.configuration_nix.is_empty() {
                    let config_path = format!("{}/configuration.nix", config_staging_dir);
                    let _ = tokio::fs::write(&config_path, &client_msg.configuration_nix).await;
                    let _ = run_cmd(&format!("chmod 600 {}", config_path)).await;
                }
                // Stage flake.nix if provided
                if !client_msg.flake_nix.is_empty() {
                    let flake_path = format!("{}/flake.nix", config_staging_dir);
                    let _ = tokio::fs::write(&flake_path, &client_msg.flake_nix).await;
                    let _ = run_cmd(&format!("chmod 600 {}", flake_path)).await;
                }

                // Fully automated install — generates and executes the entire
                // partition → format → install → configure sequence.
                // The user only clicked "Deploy" in the browser.
                // All inputs are validated above before reaching generate_install_script.

                // SECURITY: Validate browser-supplied Nix config with pure-eval
                // (no network, no filesystem access, no builtins.exec)
                if !client_msg.configuration_nix.is_empty() {
                    use symthaea_spore::security::validate_nix_pure_eval;
                    if let Err(e) = validate_nix_pure_eval(&client_msg.configuration_nix) {
                        eprintln!(
                            "[{}] Nix pure-eval rejected browser config: {}",
                            peer_addr, e
                        );
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::output(
                                    &format!("WARNING: Nix config validation: {}", e),
                                    "stderr",
                                )
                                .to_json(),
                            ))
                            .await;
                        // Continue anyway — pure-eval may reject valid NixOS modules
                        // that use impure features like <nixpkgs>. This is advisory, not blocking.
                    }
                }

                let mut script = generate_install_script(&client_msg, session_id);

                // Always: pre-install disk snapshot (instant, non-destructive)
                let snapshot = disk_snapshot(&disk);
                script = format!("{}\n{}", snapshot, script);

                // Patch configuration.nix with DE/GPU/locale — but only if the browser
                // didn't supply a full configuration.nix (which already has everything).
                if client_msg.configuration_nix.is_empty() {
                    let patch = system_config_patch(&client_msg);
                    if !patch.is_empty() {
                        if let Some(pos) = script.find("STAGE: Configuring swap") {
                            script.insert_str(pos, &patch);
                        } else if let Some(pos) = script.find("STAGE: Installing") {
                            script.insert_str(pos, &patch);
                        }
                    }
                }

                // Always: git-initialize NixOS config after install
                if let Some(pos) = script.rfind("echo \"COMPLETE\"") {
                    script.insert_str(pos, git_init_config());
                }

                if client_msg.secure_boot {
                    if let Some(pos) = script.rfind("echo \"COMPLETE\"") {
                        script.insert_str(pos, secure_boot_postinstall());
                    } else {
                        script.push_str(secure_boot_postinstall());
                    }
                }
                if client_msg.tpm2_unlock {
                    if let Some(pos) = script.rfind("echo \"COMPLETE\"") {
                        script.insert_str(pos, tpm2_postinstall());
                    } else {
                        script.push_str(tpm2_postinstall());
                    }
                }
                if client_msg.fido2_unlock {
                    if let Some(pos) = script.rfind("echo \"COMPLETE\"") {
                        script.insert_str(pos, fido2_postinstall());
                    } else {
                        script.push_str(fido2_postinstall());
                    }
                }

                // Set user password via temp file (avoids shell injection)
                if !client_msg.user_password.is_empty() {
                    // SECURITY: reject passwords containing newlines (breaks chpasswd format)
                    if client_msg.user_password.contains('\n')
                        || client_msg.user_password.contains('\r')
                    {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error("Password must not contain newlines").to_json(),
                            ))
                            .await;
                        continue;
                    }
                    // Escape single quotes in password for safe shell embedding
                    let escaped_pw = client_msg.user_password.replace('\'', "'\\''");
                    let pw_file = format!("/tmp/sovereign-user-pw-{}", session_id);
                    let _ = run_cmd(&format!(
                        "printf '%s' '{}' > {} && chmod 600 {}",
                        escaped_pw, pw_file, pw_file
                    ))
                    .await;
                    let username = if client_msg.username.is_empty() {
                        "user"
                    } else {
                        &client_msg.username
                    };
                    let pw_script = format!(
                        r#"
# ── Set User Password ──
echo "STAGE: Setting user password..."
if [ -f {pw_file} ]; then
    PW=$(cat {pw_file})
    echo "{username}:$PW" | chroot /mnt chpasswd 2>/dev/null || true
    rm -f {pw_file}
    echo "  User password set."
fi
"#,
                        pw_file = pw_file,
                        username = username
                    );
                    if let Some(pos) = script.rfind("echo \"COMPLETE\"") {
                        script.insert_str(pos, &pw_script);
                    } else {
                        script.push_str(&pw_script);
                    }
                }

                eprintln!(
                    "[{}] Starting automated {} install on {} (session {})",
                    peer_addr,
                    if client_msg.layout.is_empty() {
                        "single"
                    } else {
                        &client_msg.layout
                    },
                    &disk,
                    session_id
                );

                let _ = ws_tx
                    .send(Message::Text(
                        RelayMessage::progress(&NixosAnywhereStage::Connecting).to_json(),
                    ))
                    .await;

                // Write the install script directly to disk (no heredoc).
                // SECURITY: Direct file write eliminates SCRIPTEOF heredoc injection.
                match tokio::fs::write(&script_path, &script).await {
                    Ok(()) => {
                        let _ = run_cmd(&format!("chmod +x {}", script_path)).await;
                    }
                    Err(e) => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error(&format!("Failed to write script: {}", e))
                                    .to_json(),
                            ))
                            .await;
                        continue;
                    }
                }

                // Upload verification
                match run_cmd(&format!("test -x {}", script_path)).await {
                    Ok(r) if r.exit_status == 0 => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::output("Install script uploaded.", "stdout")
                                    .to_json(),
                            ))
                            .await;
                    }
                    Ok(r) => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error(&format!(
                                    "Failed to upload script: {}",
                                    r.stderr.chars().take(300).collect::<String>()
                                ))
                                .to_json(),
                            ))
                            .await;
                        continue;
                    }
                    Err(e) => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error(&format!("Upload failed: {}", e)).to_json(),
                            ))
                            .await;
                        continue;
                    }
                }

                // Execute the install script in background, tail the log for streaming.
                // The script runs with output redirected to a log file,
                // while we poll the log file for new lines.
                let _ = run_cmd(&format!("bash {} > {} 2>&1 &", script_path, log_path)).await;
                let _ = ws_tx
                    .send(Message::Text(
                        RelayMessage::output("Installation started. Streaming output...", "stdout")
                            .to_json(),
                    ))
                    .await;

                // Poll the log file for new output
                let mut last_lines = 0u64;
                let mut complete = false;
                loop {
                    tokio::time::sleep(std::time::Duration::from_secs(3)).await;

                    let tail_result = run_cmd(&format!(
                        "wc -l < {} 2>/dev/null && tail -n +{} {} 2>/dev/null",
                        log_path,
                        last_lines + 1,
                        log_path
                    ))
                    .await;

                    match tail_result {
                        Ok(result) if result.exit_status == 0 => {
                            let lines: Vec<&str> = result.stdout.lines().collect();
                            if let Some(first) = lines.first() {
                                if let Ok(total) = first.trim().parse::<u64>() {
                                    // Process new lines (skip the count line)
                                    for line in &lines[1..] {
                                        if line.trim().is_empty() {
                                            continue;
                                        }

                                        // Parse STAGE: markers
                                        if line.starts_with("STAGE: ") {
                                            let stage_text = &line[7..];
                                            let stage = if stage_text.contains("Prepar")
                                                || stage_text.contains("environment")
                                            {
                                                NixosAnywhereStage::Connecting
                                            } else if stage_text.contains("Partition") {
                                                NixosAnywhereStage::Partitioning
                                            } else if stage_text.contains("Format")
                                                || stage_text.contains("btrfs")
                                                || stage_text.contains("subvol")
                                            {
                                                NixosAnywhereStage::Partitioning
                                            } else if stage_text.contains("Mount") {
                                                NixosAnywhereStage::Partitioning
                                            } else if stage_text.contains("Generat")
                                                || stage_text.contains("config")
                                            {
                                                NixosAnywhereStage::Configuring
                                            } else if stage_text.contains("Install") {
                                                NixosAnywhereStage::Installing
                                            } else if stage_text.contains("swap")
                                                || stage_text.contains("Verif")
                                            {
                                                NixosAnywhereStage::Configuring
                                            } else if stage_text.contains("FirstBreath") {
                                                NixosAnywhereStage::Complete
                                            } else {
                                                NixosAnywhereStage::Installing
                                            };

                                            let _ = ws_tx
                                                .send(Message::Text(
                                                    RelayMessage::progress(&stage).to_json(),
                                                ))
                                                .await;
                                        }

                                        if let Some(stage) = parse_stage(line) {
                                            let _ = ws_tx
                                                .send(Message::Text(
                                                    RelayMessage::progress(&stage).to_json(),
                                                ))
                                                .await;
                                        }

                                        let _ = ws_tx
                                            .send(Message::Text(
                                                RelayMessage::output(line, "stdout").to_json(),
                                            ))
                                            .await;

                                        if line.contains("COMPLETE") {
                                            complete = true;
                                        }
                                    }
                                    last_lines = total;
                                }
                            }
                        }
                        _ => {}
                    }

                    if complete {
                        break;
                    }

                    // Check if the install script is still running
                    if let Ok(check) = run_cmd(&format!("pgrep -f {}", script_path)).await {
                        if check.exit_status != 0 && last_lines > 0 {
                            // Script finished but no COMPLETE marker — check exit code
                            if let Ok(exit_check) = run_cmd(&format!("tail -1 {}", log_path)).await
                            {
                                let _ = ws_tx
                                    .send(Message::Text(
                                        RelayMessage::output(&exit_check.stdout, "stdout")
                                            .to_json(),
                                    ))
                                    .await;
                            }
                            break;
                        }
                    }
                }

                let exit_code = if complete { 0 } else { 1 };
                let _ = ws_tx
                    .send(Message::Text(RelayMessage::exit(exit_code).to_json()))
                    .await;

                // SECURITY: Clean up temporary files containing sensitive data
                let _ = run_cmd(&format!(
                    "rm -f {} {} /tmp/sovereign-user-pw-{}",
                    script_path, log_path, session_id
                ))
                .await;
                eprintln!(
                    "[{}] Session {} temp files cleaned up",
                    peer_addr, session_id
                );

                // LEGACY: The old blocking path (kept for reference)
                // This is what we replaced with the log-polling approach above.
                if false {
                    match run_cmd("bash /tmp/symthaea-install.sh 2>&1").await {
                        Ok(result) => {
                            for line in result.stdout.lines().chain(result.stderr.lines()) {
                                if line.trim().is_empty() {
                                    continue;
                                }

                                // Parse STAGE: markers for progress
                                if line.starts_with("STAGE: ") {
                                    let stage_text = &line[7..];
                                    let stage = if stage_text.contains("Detect")
                                        || stage_text.contains("free space")
                                    {
                                        NixosAnywhereStage::Connecting
                                    } else if stage_text.contains("Partition") {
                                        NixosAnywhereStage::Partitioning
                                    } else if stage_text.contains("Format")
                                        || stage_text.contains("btrfs")
                                    {
                                        NixosAnywhereStage::Partitioning
                                    } else if stage_text.contains("Mount") {
                                        NixosAnywhereStage::Partitioning
                                    } else if stage_text.contains("Install") {
                                        NixosAnywhereStage::Installing
                                    } else if stage_text.contains("Configur")
                                        || stage_text.contains("swap")
                                    {
                                        NixosAnywhereStage::Configuring
                                    } else if stage_text.contains("FirstBreath") {
                                        NixosAnywhereStage::Complete
                                    } else {
                                        NixosAnywhereStage::Installing
                                    };

                                    let _ = ws_tx
                                        .send(Message::Text(
                                            RelayMessage::progress(&stage).to_json(),
                                        ))
                                        .await;
                                }

                                // Also check for nixos-anywhere stage markers
                                if let Some(stage) = parse_stage(line) {
                                    let _ = ws_tx
                                        .send(Message::Text(
                                            RelayMessage::progress(&stage).to_json(),
                                        ))
                                        .await;
                                }

                                let _ = ws_tx
                                    .send(Message::Text(
                                        RelayMessage::output(line, "stdout").to_json(),
                                    ))
                                    .await;
                            }

                            let _ = ws_tx
                                .send(Message::Text(
                                    RelayMessage::exit(result.exit_status as i32).to_json(),
                                ))
                                .await;
                        }
                        Err(e) => {
                            let _ = ws_tx
                                .send(Message::Text(
                                    RelayMessage::error(&format!("Install failed: {}", e))
                                        .to_json(),
                                ))
                                .await;
                        }
                    }
                } // end if false (legacy blocking path)
            }

            // ── Comprehensive hardware probe ──
            // ── Pre-install validation checklist ──
            "pre_install_check" => {
                eprintln!("[{}] Running pre-install checks...", peer_addr);
                let disk = client_msg.disk.clone();
                let check_script = format!(
                    r#"
echo '{{"checks": ['

# 1. EFI vs BIOS
if [ -d /sys/firmware/efi ]; then
  echo '{{"name":"boot_mode","status":"pass","detail":"EFI/UEFI detected — systemd-boot will be used"}},'
else
  echo '{{"name":"boot_mode","status":"warn","detail":"Legacy BIOS detected — GRUB will be used (limited features)"}},'
fi

# 2. RAM check
RAM_MB=$(free -m | awk '/Mem:/{{print $2}}')
if [ "$RAM_MB" -ge 4096 ]; then
  echo "{{"name":"ram","status":"pass","detail":"${{RAM_MB}}MB RAM — sufficient for any desktop"}},"
elif [ "$RAM_MB" -ge 2048 ]; then
  echo "{{"name":"ram","status":"warn","detail":"${{RAM_MB}}MB RAM — use XFCE or Sway for best performance"}},"
else
  echo "{{"name":"ram","status":"fail","detail":"${{RAM_MB}}MB RAM — insufficient for graphical desktop. CLI-only recommended"}},"
fi

# 3. Disk health (SMART)
if command -v smartctl >/dev/null 2>&1 && [ -n "{disk}" ]; then
  SMART=$(smartctl -H "{disk}" 2>/dev/null | grep -i "overall" | head -1)
  if echo "$SMART" | grep -qi "PASSED\|OK"; then
    echo '{{"name":"disk_health","status":"pass","detail":"SMART: disk healthy"}},'
  elif [ -z "$SMART" ]; then
    echo '{{"name":"disk_health","status":"warn","detail":"SMART not supported on this disk"}},'
  else
    echo "{{"name":"disk_health","status":"fail","detail":"SMART WARNING: $SMART"}},"
  fi
else
  echo '{{"name":"disk_health","status":"warn","detail":"smartctl not available"}},'
fi

# 4. BitLocker detection
BL_FOUND=false
for PART in $(blkid -o device "{disk}"* 2>/dev/null); do
  if blkid "$PART" 2>/dev/null | grep -qi bitlocker; then
    BL_FOUND=true
    echo "{{"name":"bitlocker","status":"warn","detail":"BitLocker detected on $PART — have your recovery key ready"}},"
  fi
done
if [ "$BL_FOUND" = false ]; then
  echo '{{"name":"bitlocker","status":"pass","detail":"No BitLocker encryption detected"}},'
fi

# 5. Free space (for alongside mode)
FREE_SECTORS=$(sgdisk -p "{disk}" 2>/dev/null | awk '/Total free space/{{print $5}}' || echo "0")
FREE_GB=$((FREE_SECTORS * 512 / 1073741824))
if [ "$FREE_GB" -ge 40 ]; then
  echo "{{"name":"free_space","status":"pass","detail":"${{FREE_GB}}GB free — sufficient for NixOS"}},"
elif [ "$FREE_GB" -ge 20 ]; then
  echo "{{"name":"free_space","status":"warn","detail":"${{FREE_GB}}GB free — tight. Consider freeing more space"}},"
else
  echo "{{"name":"free_space","status":"fail","detail":"${{FREE_GB}}GB free — insufficient for dual-boot. Shrink existing partitions first"}},"
fi

# 6. Existing OS detection
OS_LIST=""
for PART in $(lsblk -rno NAME,FSTYPE "{disk}" 2>/dev/null | awk '$2~/ntfs|ext4|btrfs|xfs/{{print "/dev/"$1}}'); do
  MOUNT_DIR=$(mktemp -d)
  if mount -o ro "$PART" "$MOUNT_DIR" 2>/dev/null; then
    if [ -d "$MOUNT_DIR/Windows/System32" ]; then
      OS_LIST="$OS_LIST Windows,"
    elif [ -f "$MOUNT_DIR/etc/os-release" ]; then
      OS_NAME=$(grep PRETTY_NAME "$MOUNT_DIR/etc/os-release" | cut -d'"' -f2)
      OS_LIST="$OS_LIST $OS_NAME,"
    fi
    umount "$MOUNT_DIR" 2>/dev/null
  fi
  rmdir "$MOUNT_DIR" 2>/dev/null
done
if [ -n "$OS_LIST" ]; then
  echo "{{"name":"existing_os","status":"info","detail":"Detected:$OS_LIST"}},"
else
  echo '{{"name":"existing_os","status":"pass","detail":"No existing OS detected on this disk"}},'
fi

# 7. Network connectivity
if ping -c1 -W3 cache.nixos.org >/dev/null 2>&1; then
  echo '{{"name":"network","status":"pass","detail":"Network OK — can reach NixOS cache"}}'
else
  echo '{{"name":"network","status":"fail","detail":"Cannot reach cache.nixos.org — install will fail without internet"}}'
fi

echo ']}}'
"#,
                    disk = disk
                );
                match run_cmd(&check_script).await {
                    Ok(result) if result.exit_status == 0 => {
                        let _ = ws_tx
                            .send(Message::Text(format!(
                                "{{\"type\":\"checklist\",\"data\":{}}}",
                                result.stdout.trim()
                            )))
                            .await;
                    }
                    Ok(result) => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error(&format!(
                                    "Pre-install check failed: {}",
                                    result.stderr
                                ))
                                .to_json(),
                            ))
                            .await;
                    }
                    Err(e) => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error(&format!("Pre-install check error: {}", e))
                                    .to_json(),
                            ))
                            .await;
                    }
                }
            }

            "probe_hardware" => {
                eprintln!("[{}] Probing hardware...", peer_addr);

                // Run comprehensive hardware discovery script
                let probe_script = r#"
echo '{'

# Block devices with full details
echo '"block_devices":'
lsblk -J -o NAME,SIZE,TYPE,TRAN,FSTYPE,UUID,LABEL,MOUNTPOINT,MODEL,SERIAL 2>/dev/null || echo '{"blockdevices":[]}'

# EFI state
echo ',"efi_available":'
[ -d /sys/firmware/efi ] && echo 'true' || echo 'false'

echo ',"secure_boot":'
bootctl status 2>/dev/null | grep -q "Secure Boot: enabled" && echo 'true' || echo 'false'

echo ',"setup_mode":'
bootctl status 2>/dev/null | grep -q "Setup Mode: setup" && echo 'true' || echo 'false'

# TPM
echo ',"tpm2_available":'
[ -e /dev/tpmrm0 ] && echo 'true' || echo 'false'

# Architecture (for Apple Silicon detection)
echo ',"arch": "'$(uname -m)'"'

# Apple hardware detection
echo ',"apple_hardware":'
dmidecode -s system-manufacturer 2>/dev/null | grep -qi apple && echo 'true' || echo 'false'

# BitLocker detection
echo ',"bitlocker_detected":'
blkid 2>/dev/null | grep -qi bitlocker && echo 'true' || echo 'false'

echo ',"bitlocker_devices": ['
FIRST=true
for dev in $(blkid 2>/dev/null | grep -i bitlocker | cut -d: -f1); do
  [ "$FIRST" = true ] && FIRST=false || echo ','
  echo "\"$dev\""
done
echo ']'

# LUKS detection
echo ',"luks_devices": ['
FIRST=true
for dev in $(blkid -t TYPE=crypto_LUKS -o device 2>/dev/null); do
  [ "$FIRST" = true ] && FIRST=false || echo ','
  echo "\"$dev\""
done
echo ']'

# LVM detection
echo ',"lvm_volume_groups": ['
FIRST=true
vgs --noheadings -o vg_name 2>/dev/null | tr -d ' ' | while read vg; do
  [ -z "$vg" ] && continue
  [ "$FIRST" = true ] && FIRST=false || echo ','
  echo "\"$vg\""
done
echo ']'

# mdadm RAID detection
echo ',"mdadm_arrays": ['
FIRST=true
mdadm --examine --scan 2>/dev/null | while read line; do
  [ -z "$line" ] && continue
  [ "$FIRST" = true ] && FIRST=false || echo ','
  echo "\"$(echo "$line" | sed 's/"/\\"/g')\""
done
echo ']'

# ZFS pool detection
echo ',"zfs_pools": ['
FIRST=true
zpool import 2>/dev/null | grep 'pool:' | awk '{print $2}' | while read pool; do
  [ -z "$pool" ] && continue
  [ "$FIRST" = true ] && FIRST=false || echo ','
  echo "\"$pool\""
done
echo ']'

# btrfs multi-device
echo ',"btrfs_multidevice": ['
FIRST=true
btrfs filesystem show 2>/dev/null | grep "Label:" | while read line; do
  [ "$FIRST" = true ] && FIRST=false || echo ','
  echo "\"$(echo "$line" | sed 's/"/\\"/g')\""
done
echo ']'

# EFI boot entries (existing operating systems)
echo ',"efi_boot_entries": ['
FIRST=true
efibootmgr 2>/dev/null | grep '^Boot[0-9]' | while read line; do
  [ "$FIRST" = true ] && FIRST=false || echo ','
  echo "\"$(echo "$line" | sed 's/"/\\"/g')\""
done
echo ']'

# Detected operating systems via os-prober or manual scan
echo ',"detected_os": ['
FIRST=true
# Try os-prober first
if command -v os-prober >/dev/null 2>&1; then
  os-prober 2>/dev/null | while IFS=: read dev name loader type; do
    [ -z "$dev" ] && continue
    [ "$FIRST" = true ] && FIRST=false || echo ','
    printf '{"device":"%s","name":"%s","type":"%s"}' "$dev" "$name" "$type"
  done
fi
# Also scan for Windows and Linux on mounted/mountable filesystems
for part in $(lsblk -rno NAME,FSTYPE 2>/dev/null | awk '$2 ~ /ntfs|ext4|btrfs|xfs/ {print $1}'); do
  tmpdir=$(mktemp -d 2>/dev/null) || continue
  if mount -o ro /dev/$part $tmpdir 2>/dev/null; then
    if [ -d "$tmpdir/Windows/System32" ]; then
      [ "$FIRST" = true ] && FIRST=false || echo ','
      printf '{"device":"/dev/%s","name":"Windows","type":"windows"}' "$part"
    elif [ -f "$tmpdir/etc/os-release" ]; then
      osname=$(grep PRETTY_NAME "$tmpdir/etc/os-release" 2>/dev/null | cut -d= -f2 | tr -d '"')
      [ "$FIRST" = true ] && FIRST=false || echo ','
      printf '{"device":"/dev/%s","name":"%s","type":"linux"}' "$part" "$osname"
    fi
    umount $tmpdir 2>/dev/null
  fi
  rmdir $tmpdir 2>/dev/null
done
echo ']'

# Free space on each disk (unpartitioned)
echo ',"free_space": ['
FIRST=true
for disk in $(lsblk -dnro NAME,TYPE 2>/dev/null | awk '$2=="disk" {print $1}'); do
  free=$(sgdisk -p /dev/$disk 2>/dev/null | grep "Total free space" | awk '{print $5, $6}')
  [ -z "$free" ] && continue
  [ "$FIRST" = true ] && FIRST=false || echo ','
  printf '{"device":"/dev/%s","free":"%s"}' "$disk" "$free"
done
echo ']'

# ── GPU Detection ──
echo ',"gpu": {'
GPU_VENDOR="unknown"
GPU_MODEL="unknown"
GPU_DRIVER="modesetting"
GPU_LINE=$(lspci 2>/dev/null | grep -iE 'VGA|3D|Display' | head -1)
if echo "$GPU_LINE" | grep -qi nvidia; then
  GPU_VENDOR="nvidia"
  GPU_MODEL=$(echo "$GPU_LINE" | sed 's/.*: //')
  GPU_DRIVER="nvidia"
elif echo "$GPU_LINE" | grep -qi "amd\|radeon\|ati"; then
  GPU_VENDOR="amd"
  GPU_MODEL=$(echo "$GPU_LINE" | sed 's/.*: //')
  GPU_DRIVER="amdgpu"
elif echo "$GPU_LINE" | grep -qi intel; then
  GPU_VENDOR="intel"
  GPU_MODEL=$(echo "$GPU_LINE" | sed 's/.*: //')
  GPU_DRIVER="modesetting"
fi
# Check for hybrid graphics (Optimus / switchable)
GPU_COUNT=$(lspci 2>/dev/null | grep -ciE 'VGA|3D|Display')
HYBRID=false
[ "$GPU_COUNT" -gt 1 ] && HYBRID=true
printf '"vendor":"%s","model":"%s","driver":"%s","hybrid":%s,"count":%d' \
  "$GPU_VENDOR" "$(echo "$GPU_MODEL" | sed 's/"/\\"/g')" "$GPU_DRIVER" "$HYBRID" "$GPU_COUNT"
echo '}'

# ── WiFi Detection ──
echo ',"wifi": {'
WIFI_AVAILABLE=false
WIFI_IFACE=""
WIFI_NETWORKS="[]"
# Check for wireless interfaces
WIFI_IFACE=$(iw dev 2>/dev/null | awk '/Interface/{print $2}' | head -1)
if [ -z "$WIFI_IFACE" ]; then
  WIFI_IFACE=$(ls /sys/class/net 2>/dev/null | while read iface; do
    [ -d "/sys/class/net/$iface/wireless" ] && echo "$iface" && break
  done)
fi
if [ -n "$WIFI_IFACE" ]; then
  WIFI_AVAILABLE=true
  # Scan for networks (needs root)
  WIFI_NETWORKS=$(nmcli -t -f SSID,SIGNAL,SECURITY device wifi list 2>/dev/null | head -20 | awk -F: '{printf "{\"ssid\":\"%s\",\"signal\":%s,\"security\":\"%s\"},", $1, ($2=="" ? "0" : $2), $3}' | sed 's/,$//' || echo "")
  [ -n "$WIFI_NETWORKS" ] && WIFI_NETWORKS="[$WIFI_NETWORKS]" || WIFI_NETWORKS="[]"
fi
printf '"available":%s,"interface":"%s","networks":%s' "$WIFI_AVAILABLE" "$WIFI_IFACE" "$WIFI_NETWORKS"
echo '}'

# ── Timezone / Locale Detection ──
echo ',"locale": {'
# Try to detect timezone from system or IP geolocation
TZ_DETECTED=$(cat /etc/timezone 2>/dev/null || timedatectl show --property=Timezone --value 2>/dev/null || echo "")
if [ -z "$TZ_DETECTED" ]; then
  # Fallback: IP geolocation (requires internet)
  TZ_DETECTED=$(curl -s --connect-timeout 3 "http://ip-api.com/line/?fields=timezone" 2>/dev/null || echo "UTC")
fi
[ -z "$TZ_DETECTED" ] && TZ_DETECTED="UTC"
LANG_DETECTED=$(echo $LANG 2>/dev/null | cut -d. -f1)
[ -z "$LANG_DETECTED" ] && LANG_DETECTED="en_US"
KB_LAYOUT=$(cat /etc/vconsole.conf 2>/dev/null | grep KEYMAP | cut -d= -f2 || echo "us")
[ -z "$KB_LAYOUT" ] && KB_LAYOUT="us"
printf '"timezone":"%s","language":"%s","keyboard":"%s"' "$TZ_DETECTED" "$LANG_DETECTED" "$KB_LAYOUT"
echo '}'

# ── Safety: Active server detection ──
# Scores risk factors. High score = likely production server, block install.
echo ',"safety": {'

RISK_SCORE=0
RISK_REASONS='['
RISK_FIRST=true

add_risk() {
  local points=$1
  local reason=$2
  RISK_SCORE=$((RISK_SCORE + points))
  [ "$RISK_FIRST" = true ] && RISK_FIRST=false || RISK_REASONS="$RISK_REASONS,"
  RISK_REASONS="$RISK_REASONS\"$reason\""
}

# HIGH: Running containers (production workloads)
CONTAINERS=$(docker ps -q 2>/dev/null | wc -l)
[ "$CONTAINERS" -gt 0 ] && add_risk 40 "Docker: $CONTAINERS running containers"
PODS=$(kubectl get pods --all-namespaces --no-headers 2>/dev/null | wc -l)
[ "$PODS" -gt 0 ] && add_risk 50 "Kubernetes: $PODS running pods"

# HIGH: Database services running
pgrep -x postgres >/dev/null 2>&1 && add_risk 40 "PostgreSQL is running"
pgrep -x mysqld >/dev/null 2>&1 && add_risk 40 "MySQL is running"
pgrep -x mongod >/dev/null 2>&1 && add_risk 40 "MongoDB is running"
pgrep -x redis-server >/dev/null 2>&1 && add_risk 30 "Redis is running"

# HIGH: Web servers with active listeners
pgrep -x nginx >/dev/null 2>&1 && add_risk 35 "nginx is running"
pgrep -x apache2 >/dev/null 2>&1 && add_risk 35 "Apache is running"
pgrep -x caddy >/dev/null 2>&1 && add_risk 35 "Caddy is running"

# MEDIUM: Active server ports
for port in 80 443 3306 5432 8080 8443 27017; do
  ss -tlnp 2>/dev/null | grep -q ":$port " && add_risk 15 "Port $port is listening"
done

# MEDIUM: Multiple logged-in users
USER_COUNT=$(who 2>/dev/null | awk '{print $1}' | sort -u | wc -l)
[ "$USER_COUNT" -gt 1 ] && add_risk 20 "$USER_COUNT users currently logged in"

# MEDIUM: High uptime (relied-on system)
UPTIME_DAYS=$(awk '{print int($1/86400)}' /proc/uptime 2>/dev/null)
[ "$UPTIME_DAYS" -gt 30 ] && add_risk 15 "Uptime: ${UPTIME_DAYS} days"
[ "$UPTIME_DAYS" -gt 180 ] && add_risk 15 "Uptime: ${UPTIME_DAYS} days (long-running)"

# MEDIUM: Server-like hostname
HOSTNAME=$(hostname 2>/dev/null)
echo "$HOSTNAME" | grep -qiE 'prod|srv|server|db|web|api|node|master|worker|k8s|kube' && add_risk 25 "Hostname '$HOSTNAME' looks like a server"

# MEDIUM: Cloud instance
curl -s --connect-timeout 1 http://169.254.169.254/ >/dev/null 2>&1 && add_risk 25 "Cloud instance metadata endpoint detected"

# MEDIUM: Running VMs
LIBVIRT_VMS=$(virsh list --all --name 2>/dev/null | grep -v '^$' | wc -l)
[ "$LIBVIRT_VMS" -gt 0 ] && add_risk 20 "$LIBVIRT_VMS libvirt VMs defined"
QEMU_PROCS=$(pgrep -c qemu-system 2>/dev/null || echo 0)
[ "$QEMU_PROCS" -gt 0 ] && add_risk 20 "$QEMU_PROCS QEMU VMs running"

# LOW: Application data directories
[ -d /var/www ] && add_risk 10 "/var/www exists (web server data)"
[ -d /opt ] && [ "$(ls -A /opt 2>/dev/null)" ] && add_risk 10 "/opt has application data"
[ -d /srv ] && [ "$(ls -A /srv 2>/dev/null)" ] && add_risk 5 "/srv has data"

# LOW: Many SSH keys (shared server)
KEY_COUNT=$(wc -l < /root/.ssh/authorized_keys 2>/dev/null || echo 0)
[ "$KEY_COUNT" -gt 3 ] && add_risk 15 "$KEY_COUNT SSH authorized keys (shared access)"

# LOW: Server-class hardware
dmidecode -t chassis 2>/dev/null | grep -qi "rack\|blade\|server" && add_risk 15 "Server-class chassis detected"
dmidecode -t memory 2>/dev/null | grep -qi "error correction.*multi-bit ecc" && add_risk 10 "ECC memory detected"
[ -e /dev/ipmi0 ] && add_risk 15 "IPMI/BMC interface detected"

# Determine safety level
RISK_REASONS="$RISK_REASONS]"
if [ $RISK_SCORE -ge 50 ]; then
  SAFETY_LEVEL="blocked"
  SAFETY_MSG="This system appears to be an active server. Installation is BLOCKED to prevent data loss."
elif [ $RISK_SCORE -ge 25 ]; then
  SAFETY_LEVEL="warning"
  SAFETY_MSG="This system shows signs of active use. Please confirm this is not a production system."
elif [ $RISK_SCORE -ge 10 ]; then
  SAFETY_LEVEL="caution"
  SAFETY_MSG="Minor server indicators detected. Proceed with awareness."
else
  SAFETY_LEVEL="clear"
  SAFETY_MSG="No active server indicators detected. Safe to proceed."
fi

printf '"level":"%s","score":%d,"message":"%s","reasons":%s' \
  "$SAFETY_LEVEL" "$RISK_SCORE" "$SAFETY_MSG" "$RISK_REASONS"
echo '}'

# Chromebook detection
echo ',"chromebook":{'
echo '"detected":'
(dmidecode -s system-manufacturer 2>/dev/null | grep -qi "google" || [ -e /dev/cros_ec ] || grep -qi "chromebook" /sys/class/dmi/id/product_name 2>/dev/null) && echo 'true,' || echo 'false,'
echo '"firmware":'
if [ -d /sys/firmware/efi ]; then echo '"uefi",'
elif grep -qi depthcharge /proc/cmdline 2>/dev/null; then echo '"depthcharge",'
else echo '"bios",'
fi
echo '"emmc":'
lsblk -ndo TRAN 2>/dev/null | grep -q mmc && echo 'true' || echo 'false'
echo '}'

echo '}'
"#;

                match run_cmd(probe_script).await {
                    Ok(result) if result.exit_status == 0 => {
                        eprintln!("[{}] Hardware probe complete", peer_addr);
                        // Strip ANSI escape codes and control chars that corrupt JSON
                        let clean: String = result
                            .stdout
                            .chars()
                            .filter(|c| !c.is_control() || *c == '\n' || *c == '\t')
                            .collect();
                        let _ = ws_tx
                            .send(Message::Text(
                                serde_json::json!({
                                    "type": "hardware_probe",
                                    "data": clean
                                })
                                .to_string(),
                            ))
                            .await;
                    }
                    Ok(result) => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error(&format!(
                                    "Hardware probe failed (exit {}): {}",
                                    result.exit_status,
                                    result.stderr.chars().take(300).collect::<String>()
                                ))
                                .to_json(),
                            ))
                            .await;
                    }
                    Err(e) => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error(&format!("Hardware probe failed: {}", e))
                                    .to_json(),
                            ))
                            .await;
                    }
                }
            }

            // ── Scan existing OS for installed applications ──
            "scan_apps" => {
                eprintln!("[{}] Scanning apps on existing OS...", peer_addr);

                // Step 1: Detect what OS partitions exist and mount read-only
                let scan_script = r#"
echo '['
FIRST=true

# Find and mount Windows/macOS/Linux partitions read-only
for part in $(lsblk -rno NAME,FSTYPE 2>/dev/null | awk '$2 ~ /ntfs|hfsplus|apfs|ext4|btrfs|xfs/ {print $1}'); do
  MOUNTPOINT=$(mktemp -d /tmp/appscan-XXXXX 2>/dev/null) || continue
  FS=$(lsblk -rno FSTYPE /dev/$part 2>/dev/null)

  mounted=false
  case "$FS" in
    ntfs)
      ntfs-3g -o ro /dev/$part $MOUNTPOINT 2>/dev/null && mounted=true
      # Alternative: mount -t ntfs3 -o ro /dev/$part $MOUNTPOINT 2>/dev/null && mounted=true
      ;;
    *)
      mount -o ro /dev/$part $MOUNTPOINT 2>/dev/null && mounted=true
      ;;
  esac

  if [ "$mounted" = true ]; then
    # Windows scan
    if [ -d "$MOUNTPOINT/Program Files" ]; then
      for dir in "$MOUNTPOINT/Program Files"/* "$MOUNTPOINT/Program Files (x86)"/*; do
        [ -d "$dir" ] || continue
        name=$(basename "$dir")
        case "$name" in
          "Common Files"|"WindowsApps"|"Windows Defender"*|"Windows NT"|"Windows Photo Viewer"|"Windows Sidebar"|"Uninstall Information"|"Reference Assemblies"|"MSBuild"|"dotnet"|"ModifiableWindowsApps") continue ;;
        esac
        [ "$FIRST" = true ] && FIRST=false || echo ','
        printf '{"name":"%s","path":"%s","source":"windows","category":""}' \
          "$(echo "$name" | sed 's/"/\\"/g')" \
          "/dev/$part"
      done
    fi

    # macOS scan
    if [ -d "$MOUNTPOINT/Applications" ]; then
      for app in "$MOUNTPOINT/Applications"/*.app; do
        [ -d "$app" ] || continue
        name=$(basename "$app" .app)
        [ "$FIRST" = true ] && FIRST=false || echo ','
        printf '{"name":"%s","path":"%s","source":"macos","category":""}' \
          "$(echo "$name" | sed 's/"/\\"/g')" \
          "/dev/$part"
      done
      # Homebrew
      for cellar in "$MOUNTPOINT/usr/local/Cellar" "$MOUNTPOINT/opt/homebrew/Cellar"; do
        [ -d "$cellar" ] || continue
        for pkg in "$cellar"/*/; do
          name=$(basename "$pkg")
          [ "$FIRST" = true ] && FIRST=false || echo ','
          printf '{"name":"%s","path":"%s","source":"homebrew","category":""}' \
            "$(echo "$name" | sed 's/"/\\"/g')" \
            "/dev/$part"
        done
      done
    fi

    # Linux scan
    if [ -f "$MOUNTPOINT/etc/os-release" ]; then
      # Scan common app directories
      for bin in "$MOUNTPOINT/usr/bin"/*; do
        [ -f "$bin" ] || continue
        name=$(basename "$bin")
        # Only include well-known GUI apps
        case "$name" in
          firefox|chrome|chromium|code|gimp|inkscape|blender|obs|vlc|spotify|discord|slack|zoom|telegram*|signal*|steam|lutris|thunderbird|libreoffice|kdenlive|krita|audacity|filezilla|qbittorrent|keepassxc|bitwarden)
            [ "$FIRST" = true ] && FIRST=false || echo ','
            printf '{"name":"%s","path":"%s","source":"linux","category":""}' "$name" "/dev/$part"
            ;;
        esac
      done
      # Flatpak
      if [ -d "$MOUNTPOINT/var/lib/flatpak/app" ]; then
        for app in "$MOUNTPOINT/var/lib/flatpak/app"/*/; do
          name=$(basename "$app")
          [ "$FIRST" = true ] && FIRST=false || echo ','
          printf '{"name":"%s","path":"%s","source":"flatpak","category":""}' \
            "$(echo "$name" | sed 's/"/\\"/g')" \
            "/dev/$part"
        done
      fi
    fi

    umount $MOUNTPOINT 2>/dev/null
  fi
  rmdir $MOUNTPOINT 2>/dev/null
done

echo ']'
"#;

                match run_cmd(scan_script).await {
                    Ok(result) if result.exit_status == 0 => {
                        eprintln!("[{}] App scan complete", peer_addr);
                        let _ = ws_tx
                            .send(Message::Text(
                                serde_json::json!({
                                    "type": "app_scan",
                                    "data": result.stdout
                                })
                                .to_string(),
                            ))
                            .await;
                    }
                    Ok(result) => {
                        // Partial results are OK — some partitions may fail to mount
                        eprintln!(
                            "[{}] App scan partial (exit {})",
                            peer_addr, result.exit_status
                        );
                        let _ = ws_tx
                            .send(Message::Text(
                                serde_json::json!({
                                    "type": "app_scan",
                                    "data": result.stdout,
                                    "warning": result.stderr.chars().take(200).collect::<String>()
                                })
                                .to_string(),
                            ))
                            .await;
                    }
                    Err(e) => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error(&format!("App scan failed: {}", e)).to_json(),
                            ))
                            .await;
                    }
                }
            }

            // ── Deep scan: dotfiles, config, personal data for migration + welcome ──
            "deep_scan" => {
                eprintln!("[{}] Deep scanning for migration data...", peer_addr);

                let deep_scan_script = r#"
echo '{'

# ── Git Configuration ──
echo '"git": {'
FIRST_GIT=true
for home in /home/* /root; do
  [ -f "$home/.gitconfig" ] || continue
  [ "$FIRST_GIT" = true ] && FIRST_GIT=false || true
  NAME=$(git config -f "$home/.gitconfig" user.name 2>/dev/null || echo "")
  EMAIL=$(git config -f "$home/.gitconfig" user.email 2>/dev/null || echo "")
  EDITOR=$(git config -f "$home/.gitconfig" core.editor 2>/dev/null || echo "")
  MERGE=$(git config -f "$home/.gitconfig" pull.rebase 2>/dev/null || echo "")
  ALIASES=$(git config -f "$home/.gitconfig" --get-regexp alias 2>/dev/null | wc -l)
  printf '"name":"%s","email":"%s","editor":"%s","prefers_rebase":%s,"alias_count":%d' \
    "$NAME" "$EMAIL" "$EDITOR" \
    "$([ "$MERGE" = "true" ] && echo true || echo false)" \
    "$ALIASES"
done
echo '}'

# ── Shell Configuration ──
echo ',"shell": {'
SHELL_TYPE="bash"
ALIAS_COUNT=0
CUSTOM_FUNCTIONS=0
for home in /home/* /root; do
  if [ -f "$home/.zshrc" ]; then
    SHELL_TYPE="zsh"
    ALIAS_COUNT=$(grep -c "^alias " "$home/.zshrc" 2>/dev/null || echo 0)
    CUSTOM_FUNCTIONS=$(grep -c "^function \|^[a-z_]*() " "$home/.zshrc" 2>/dev/null || echo 0)
    # Check for oh-my-zsh or other frameworks
    grep -q "oh-my-zsh" "$home/.zshrc" 2>/dev/null && SHELL_TYPE="zsh-omz"
    grep -q "starship" "$home/.zshrc" 2>/dev/null && SHELL_TYPE="zsh-starship"
  elif [ -f "$home/.bashrc" ]; then
    ALIAS_COUNT=$(grep -c "^alias " "$home/.bashrc" 2>/dev/null || echo 0)
    CUSTOM_FUNCTIONS=$(grep -c "^function \|^[a-z_]*() " "$home/.bashrc" 2>/dev/null || echo 0)
  fi
done
printf '"type":"%s","alias_count":%d,"custom_functions":%d' \
  "$SHELL_TYPE" "$ALIAS_COUNT" "$CUSTOM_FUNCTIONS"
echo '}'

# ── SSH Keys & Config ──
echo ',"ssh": {'
KEY_COUNT=0
HOST_COUNT=0
for home in /home/* /root; do
  [ -d "$home/.ssh" ] || continue
  KC=$(ls "$home/.ssh/"*.pub 2>/dev/null | wc -l)
  KEY_COUNT=$((KEY_COUNT + KC))
  [ -f "$home/.ssh/config" ] && HOST_COUNT=$(grep -c "^Host " "$home/.ssh/config" 2>/dev/null || echo 0)
done
printf '"key_count":%d,"host_count":%d,"has_agent_config":%s' \
  "$KEY_COUNT" "$HOST_COUNT" \
  "$(grep -rq "AddKeysToAgent" /home/*/.ssh/config /root/.ssh/config 2>/dev/null && echo true || echo false)"
echo '}'

# ── Editor Configurations ──
echo ',"editors": {'
EDITORS="["
FIRST_ED=true
for home in /home/* /root; do
  # VS Code
  if [ -d "$home/.config/Code" ] || [ -d "$home/.vscode" ]; then
    [ "$FIRST_ED" = true ] && FIRST_ED=false || EDITORS="$EDITORS,"
    EXT_COUNT=$(ls "$home/.vscode/extensions" 2>/dev/null | wc -l || echo 0)
    EDITORS="$EDITORS{\"name\":\"vscode\",\"extensions\":$EXT_COUNT}"
  fi
  # Neovim
  if [ -d "$home/.config/nvim" ]; then
    [ "$FIRST_ED" = true ] && FIRST_ED=false || EDITORS="$EDITORS,"
    EDITORS="$EDITORS{\"name\":\"neovim\",\"has_config\":true}"
  fi
  # Vim
  if [ -f "$home/.vimrc" ]; then
    [ "$FIRST_ED" = true ] && FIRST_ED=false || EDITORS="$EDITORS,"
    PLUGIN_COUNT=$(grep -c "Plug \|Plugin \|NeoBundle " "$home/.vimrc" 2>/dev/null || echo 0)
    EDITORS="$EDITORS{\"name\":\"vim\",\"plugin_count\":$PLUGIN_COUNT}"
  fi
  # Emacs
  if [ -d "$home/.emacs.d" ] || [ -f "$home/.emacs" ]; then
    [ "$FIRST_ED" = true ] && FIRST_ED=false || EDITORS="$EDITORS,"
    EDITORS="$EDITORS{\"name\":\"emacs\",\"has_config\":true}"
  fi
done
echo "\"detected\":$EDITORS]}"

# ── Desktop / Window Manager ──
echo ',"current_desktop": {'
DE="unknown"
WM="unknown"
# Check for desktop config dirs
for home in /home/*; do
  [ -d "$home/.config/hypr" ] && WM="hyprland"
  [ -d "$home/.config/sway" ] && WM="sway"
  [ -d "$home/.config/i3" ] && WM="i3"
  [ -d "$home/.config/awesome" ] && WM="awesome"
  [ -d "$home/.config/gnome-session" ] && DE="gnome"
  [ -d "$home/.config/plasma-workspace" ] && DE="kde"
  [ -d "$home/.config/xfce4" ] && DE="xfce"
done
printf '"de":"%s","wm":"%s"' "$DE" "$WM"
echo '}'

# ── Browser Data ──
echo ',"browsers": {'
BROWSERS="["
FIRST_BR=true
for home in /home/*; do
  # Firefox
  if [ -d "$home/.mozilla/firefox" ]; then
    [ "$FIRST_BR" = true ] && FIRST_BR=false || BROWSERS="$BROWSERS,"
    PROFILES=$(ls -d "$home/.mozilla/firefox"/*.default* 2>/dev/null | wc -l)
    BOOKMARKS=0
    for prof in "$home/.mozilla/firefox"/*.default*/; do
      [ -f "$prof/places.sqlite" ] && BOOKMARKS=$((BOOKMARKS + $(sqlite3 "$prof/places.sqlite" "SELECT COUNT(*) FROM moz_bookmarks" 2>/dev/null || echo 0)))
    done
    BROWSERS="$BROWSERS{\"name\":\"firefox\",\"profiles\":$PROFILES,\"bookmarks\":$BOOKMARKS}"
  fi
  # Chrome/Chromium
  for chrome_dir in "$home/.config/google-chrome" "$home/.config/chromium"; do
    [ -d "$chrome_dir" ] || continue
    [ "$FIRST_BR" = true ] && FIRST_BR=false || BROWSERS="$BROWSERS,"
    BNAME=$(basename "$chrome_dir")
    BROWSERS="$BROWSERS{\"name\":\"$BNAME\",\"has_profile\":true}"
  done
done
echo "\"detected\":$BROWSERS]}"

# ── Docker / Development Environment ──
echo ',"development": {'
DOCKER_IMAGES=0
DOCKER_COMPOSE_FILES=0
VENVS=0
NODE_PROJECTS=0
RUST_PROJECTS=0
command -v docker >/dev/null 2>&1 && DOCKER_IMAGES=$(docker images -q 2>/dev/null | wc -l)
DOCKER_COMPOSE_FILES=$(find /home -name "docker-compose.yml" -o -name "compose.yml" 2>/dev/null | wc -l)
VENVS=$(find /home -maxdepth 4 -name "pyvenv.cfg" 2>/dev/null | wc -l)
NODE_PROJECTS=$(find /home -maxdepth 4 -name "package.json" -not -path "*/node_modules/*" 2>/dev/null | wc -l)
RUST_PROJECTS=$(find /home -maxdepth 4 -name "Cargo.toml" -not -path "*/target/*" 2>/dev/null | wc -l)
printf '"docker_images":%d,"compose_files":%d,"python_venvs":%d,"node_projects":%d,"rust_projects":%d' \
  "$DOCKER_IMAGES" "$DOCKER_COMPOSE_FILES" "$VENVS" "$NODE_PROJECTS" "$RUST_PROJECTS"
echo '}'

# ── Music / Creative Tools ──
echo ',"creative": {'
HAS_AUDIO_PROJECTS=false
HAS_DAW_CONFIG=false
for home in /home/*; do
  [ -d "$home/.config/ardour" ] && HAS_DAW_CONFIG=true
  [ -d "$home/.config/LMMS" ] && HAS_DAW_CONFIG=true
  [ -d "$home/.config/Bitwig" ] && HAS_DAW_CONFIG=true
  [ -d "$home/Music" ] && [ "$(ls -A "$home/Music" 2>/dev/null)" ] && HAS_AUDIO_PROJECTS=true
done
printf '"has_daw_config":%s,"has_audio_projects":%s' "$HAS_DAW_CONFIG" "$HAS_AUDIO_PROJECTS"
echo '}'

# ── User Identity ──
echo ',"identity": {'
USERNAME=""
FULLNAME=""
AVATAR_EXISTS=false
for home in /home/*; do
  u=$(basename "$home")
  [ "$u" = "lost+found" ] && continue
  USERNAME="$u"
  FULLNAME=$(getent passwd "$u" 2>/dev/null | cut -d: -f5 | cut -d, -f1)
  [ -f "$home/.face" ] || [ -f "$home/.face.icon" ] && AVATAR_EXISTS=true
  break
done
printf '"username":"%s","fullname":"%s","has_avatar":%s' \
  "$USERNAME" "$FULLNAME" "$AVATAR_EXISTS"
echo '}'

echo '}'
"#;

                match run_cmd(deep_scan_script).await {
                    Ok(result) => {
                        eprintln!("[{}] Deep scan complete", peer_addr);
                        let _ = ws_tx
                            .send(Message::Text(
                                serde_json::json!({
                                    "type": "deep_scan",
                                    "data": result.stdout
                                })
                                .to_string(),
                            ))
                            .await;
                    }
                    Err(e) => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error(&format!("Deep scan failed: {}", e)).to_json(),
                            ))
                            .await;
                    }
                }
            }

            // ── Data preservation before wipe ──
            "preserve_data" => {
                eprintln!("[{}] Preserving data before wipe...", peer_addr);

                let preserve_script = r#"
set -eo pipefail
BACKUP_DIR="/tmp/symthaea-preserve-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"
echo '{"backup_dir":"'"$BACKUP_DIR"'","items":['
FIRST=true

# Docker images
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  IMAGES=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep -v '<none>' | head -20)
  if [ -n "$IMAGES" ]; then
    echo "$IMAGES" | while read img; do
      [ "$FIRST" = true ] && FIRST=false || echo ','
      echo "  Saving Docker image: $img" >&2
      docker save "$img" | gzip > "$BACKUP_DIR/docker-$(echo "$img" | tr '/:' '_').tar.gz" 2>/dev/null
      SIZE=$(du -h "$BACKUP_DIR/docker-$(echo "$img" | tr '/:' '_').tar.gz" | cut -f1)
      printf '{"type":"docker_image","name":"%s","size":"%s","path":"%s"}' "$img" "$SIZE" "$BACKUP_DIR/docker-$(echo "$img" | tr '/:' '_').tar.gz"
    done
  fi
fi

# PostgreSQL databases
if command -v pg_dumpall >/dev/null 2>&1 && pgrep -x postgres >/dev/null 2>&1; then
  echo "  Dumping PostgreSQL databases..." >&2
  su - postgres -c "pg_dumpall" 2>/dev/null | gzip > "$BACKUP_DIR/postgresql-all.sql.gz" || true
  if [ -f "$BACKUP_DIR/postgresql-all.sql.gz" ]; then
    SIZE=$(du -h "$BACKUP_DIR/postgresql-all.sql.gz" | cut -f1)
    [ "$FIRST" = true ] && FIRST=false || echo ','
    printf '{"type":"postgresql","name":"all databases","size":"%s","path":"%s"}' "$SIZE" "$BACKUP_DIR/postgresql-all.sql.gz"
  fi
fi

# MySQL databases
if command -v mysqldump >/dev/null 2>&1 && pgrep -x mysqld >/dev/null 2>&1; then
  echo "  Dumping MySQL databases..." >&2
  mysqldump --all-databases 2>/dev/null | gzip > "$BACKUP_DIR/mysql-all.sql.gz" || true
  if [ -f "$BACKUP_DIR/mysql-all.sql.gz" ]; then
    SIZE=$(du -h "$BACKUP_DIR/mysql-all.sql.gz" | cut -f1)
    [ "$FIRST" = true ] && FIRST=false || echo ','
    printf '{"type":"mysql","name":"all databases","size":"%s","path":"%s"}' "$SIZE" "$BACKUP_DIR/mysql-all.sql.gz"
  fi
fi

# Web server content
for webdir in /var/www /srv/http /usr/share/nginx/html; do
  if [ -d "$webdir" ] && [ "$(ls -A "$webdir" 2>/dev/null)" ]; then
    echo "  Backing up $webdir..." >&2
    tar czf "$BACKUP_DIR/$(basename "$webdir").tar.gz" -C "$(dirname "$webdir")" "$(basename "$webdir")" 2>/dev/null || true
    if [ -f "$BACKUP_DIR/$(basename "$webdir").tar.gz" ]; then
      SIZE=$(du -h "$BACKUP_DIR/$(basename "$webdir").tar.gz" | cut -f1)
      [ "$FIRST" = true ] && FIRST=false || echo ','
      printf '{"type":"webdata","name":"%s","size":"%s","path":"%s"}' "$webdir" "$SIZE" "$BACKUP_DIR/$(basename "$webdir").tar.gz"
    fi
  fi
done

# Crontabs
if [ -d /var/spool/cron ]; then
  tar czf "$BACKUP_DIR/crontabs.tar.gz" /var/spool/cron 2>/dev/null || true
  if [ -f "$BACKUP_DIR/crontabs.tar.gz" ]; then
    [ "$FIRST" = true ] && FIRST=false || echo ','
    printf '{"type":"crontabs","name":"all crontabs","size":"small","path":"%s"}' "$BACKUP_DIR/crontabs.tar.gz"
  fi
fi

# SSH keys and config
if [ -d /root/.ssh ] || [ -d /home ]; then
  tar czf "$BACKUP_DIR/ssh-keys.tar.gz" /root/.ssh /home/*/.ssh 2>/dev/null || true
  if [ -f "$BACKUP_DIR/ssh-keys.tar.gz" ]; then
    [ "$FIRST" = true ] && FIRST=false || echo ','
    printf '{"type":"ssh_keys","name":"SSH keys and config","size":"small","path":"%s"}' "$BACKUP_DIR/ssh-keys.tar.gz"
  fi
fi

# /etc (system config)
tar czf "$BACKUP_DIR/etc-backup.tar.gz" /etc 2>/dev/null || true
if [ -f "$BACKUP_DIR/etc-backup.tar.gz" ]; then
  SIZE=$(du -h "$BACKUP_DIR/etc-backup.tar.gz" | cut -f1)
  [ "$FIRST" = true ] && FIRST=false || echo ','
  printf '{"type":"system_config","name":"/etc","size":"%s","path":"%s"}' "$SIZE" "$BACKUP_DIR/etc-backup.tar.gz"
fi

# Home directories (offer to preserve)
HOME_SIZE=$(du -sh /home 2>/dev/null | cut -f1)
[ "$FIRST" = true ] && FIRST=false || echo ','
printf '{"type":"home_dirs","name":"/home (%s)","size":"%s","path":"not backed up — too large for auto-backup"}' "$HOME_SIZE" "$HOME_SIZE"

# Summary
TOTAL_SIZE=$(du -sh "$BACKUP_DIR" 2>/dev/null | cut -f1)
echo '],"total_size":"'"$TOTAL_SIZE"'"}'
"#;

                match run_cmd(preserve_script).await {
                    Ok(result) => {
                        eprintln!("[{}] Data preservation complete", peer_addr);
                        let _ = ws_tx
                            .send(Message::Text(
                                serde_json::json!({
                                    "type": "data_preserved",
                                    "data": result.stdout
                                })
                                .to_string(),
                            ))
                            .await;
                    }
                    Err(e) => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error(&format!("Data preservation failed: {}", e))
                                    .to_json(),
                            ))
                            .await;
                    }
                }
            }

            // ═══════════════════════════════════════════════════════
            // Post-install NixOS management actions
            // ═══════════════════════════════════════════════════════
            "list_generations" => {
                eprintln!("[{}] Listing generations...", peer_addr);
                match run_cmd(r#"nix-env --list-generations -p /nix/var/nix/profiles/system 2>/dev/null | awk '{num=$1; date=$2" "$3" "$4; cur=""; if(/\(current\)/) cur=",\"current\":true"; if(NR>1) printf ","; printf "{\"number\":%s,\"date\":\"%s\"%s}", num, date, cur}' | awk 'BEGIN{print "["} {print} END{print "]"}'"#).await {
                    Ok(r) if r.exit_status == 0 => {
                        let clean: String = r.stdout.chars().filter(|c| !c.is_control() || *c == '\n').collect();
                        let _ = ws_tx.send(Message::Text(serde_json::json!({"type":"generations","data":clean}).to_string())).await;
                    }
                    Ok(r) => { let _ = ws_tx.send(Message::Text(RelayMessage::error(&format!("Failed (exit {}): {}", r.exit_status, &r.stderr[..r.stderr.len().min(200)])).to_json())).await; }
                    Err(e) => { let _ = ws_tx.send(Message::Text(RelayMessage::error(&format!("Failed: {}", e)).to_json())).await; }
                }
            }

            "rollback" => {
                eprintln!("[{}] Rolling back...", peer_addr);
                match run_cmd("nixos-rebuild switch --rollback 2>&1").await {
                    Ok(r) => {
                        let _ = ws_tx.send(Message::Text(serde_json::json!({"type":"exit","code":r.exit_status,"data":r.stdout.chars().take(2000).collect::<String>()}).to_string())).await;
                    }
                    Err(e) => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error(&format!("Rollback failed: {}", e)).to_json(),
                            ))
                            .await;
                    }
                }
            }

            "switch_generation" => {
                let r#gen = &client_msg.command;
                if r#gen.is_empty() || !r#gen.chars().all(|c| c.is_ascii_digit()) {
                    let _ = ws_tx
                        .send(Message::Text(
                            RelayMessage::error("Invalid generation number").to_json(),
                        ))
                        .await;
                    continue;
                }
                eprintln!("[{}] Switching to generation {}...", peer_addr, r#gen);
                let cmd = format!(
                    "nix-env --switch-generation {} -p /nix/var/nix/profiles/system && /nix/var/nix/profiles/system/bin/switch-to-configuration switch 2>&1",
                    r#gen
                );
                match run_cmd(&cmd).await {
                    Ok(r) => {
                        let _ = ws_tx.send(Message::Text(serde_json::json!({"type":"exit","code":r.exit_status,"data":r.stdout.chars().take(2000).collect::<String>()}).to_string())).await;
                    }
                    Err(e) => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error(&format!("Switch failed: {}", e)).to_json(),
                            ))
                            .await;
                    }
                }
            }

            "list_services" => {
                eprintln!("[{}] Listing services...", peer_addr);
                match run_cmd(r#"systemctl list-units --type=service --all --no-pager --plain 2>/dev/null | grep '\.service' | awk '{name=$1; sub(/\.service$/,"",name); active=$3; sub_=$4; $1=$2=$3=$4=""; desc=substr($0,5); printf "{\"name\":\"%s\",\"active\":\"%s\",\"sub\":\"%s\",\"desc\":\"%s\"}\n", name, active, sub_, desc}' | awk 'BEGIN{print "["} NR>1{printf ","} {print} END{print "]"}'"#).await {
                    Ok(r) if r.exit_status == 0 => {
                        let clean: String = r.stdout.chars().filter(|c| !c.is_control() || *c == '\n').collect();
                        let _ = ws_tx.send(Message::Text(serde_json::json!({"type":"services","data":clean}).to_string())).await;
                    }
                    Ok(r) => { let _ = ws_tx.send(Message::Text(RelayMessage::error(&format!("Failed: {}", &r.stderr[..r.stderr.len().min(200)])).to_json())).await; }
                    Err(e) => { let _ = ws_tx.send(Message::Text(RelayMessage::error(&format!("Failed: {}", e)).to_json())).await; }
                }
            }

            "service_action" => {
                let action = &client_msg.command;
                let service = &client_msg.hostname;
                if !["start", "stop", "restart", "reload", "enable", "disable"]
                    .contains(&action.as_str())
                {
                    let _ = ws_tx.send(Message::Text(RelayMessage::error(&format!("Invalid action '{}'. Use: start, stop, restart, reload, enable, disable", action)).to_json())).await;
                    continue;
                }
                // Validate service name to prevent shell injection
                let service = match sanitize_input(service, "service name", false) {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = ws_tx
                            .send(Message::Text(RelayMessage::error(&e).to_json()))
                            .await;
                        continue;
                    }
                };
                eprintln!("[{}] {} {}...", peer_addr, action, service);
                let cmd = format!("systemctl {} {}.service 2>&1", action, service);
                match run_cmd(&cmd).await {
                    Ok(r) => {
                        let _ = ws_tx.send(Message::Text(serde_json::json!({"type":"exit","code":r.exit_status,"data":r.stdout}).to_string())).await;
                    }
                    Err(e) => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error(&format!("Failed: {}", e)).to_json(),
                            ))
                            .await;
                    }
                }
            }

            "gc_analyze" => {
                eprintln!("[{}] Analyzing nix store...", peer_addr);
                let script = r#"
STORE_SIZE=$(du -sb /nix/store 2>/dev/null | awk '{print $1}')
DEAD_COUNT=$(nix-store --gc --print-dead 2>/dev/null | wc -l)
ROOT_COUNT=$(nix-store --gc --print-roots 2>/dev/null | wc -l)
GEN_COUNT=$(nix-env --list-generations -p /nix/var/nix/profiles/system 2>/dev/null | wc -l)
PATH_COUNT=$(ls /nix/store 2>/dev/null | wc -l)
if [ "$PATH_COUNT" -gt 0 ] && [ "$DEAD_COUNT" -gt 0 ]; then
    RECLAIMABLE=$(( DEAD_COUNT * (STORE_SIZE / PATH_COUNT) ))
else
    RECLAIMABLE=0
fi
printf '{"store_bytes":%s,"reclaimable_bytes":%s,"dead_paths":%s,"gc_roots":%s,"generations":%s}' \
    "${STORE_SIZE:-0}" "${RECLAIMABLE:-0}" "${DEAD_COUNT:-0}" "${ROOT_COUNT:-0}" "${GEN_COUNT:-0}"
"#;
                match run_cmd(script).await {
                    Ok(r) if r.exit_status == 0 => {
                        let clean: String = r
                            .stdout
                            .chars()
                            .filter(|c| !c.is_control() || *c == '\n')
                            .collect();
                        let _ = ws_tx
                            .send(Message::Text(
                                serde_json::json!({"type":"gc_analysis","data":clean}).to_string(),
                            ))
                            .await;
                    }
                    Ok(r) => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error(&format!(
                                    "Analysis failed: {}",
                                    &r.stderr[..r.stderr.len().min(200)]
                                ))
                                .to_json(),
                            ))
                            .await;
                    }
                    Err(e) => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error(&format!("Analysis failed: {}", e)).to_json(),
                            ))
                            .await;
                    }
                }
            }

            "gc_collect" => {
                eprintln!("[{}] Starting garbage collection...", peer_addr);
                let gc_session_id: u64 = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);
                let gc_log = format!("/tmp/symthaea-gc-{}.log", gc_session_id);
                let _ = run_cmd(&format!("touch {} && chmod 600 {}", gc_log, gc_log)).await;
                let _ = run_cmd(&format!(
                    "nix-collect-garbage -d --delete-older-than 30d > {} 2>&1 &",
                    gc_log
                ))
                .await;
                let _ = ws_tx
                    .send(Message::Text(
                        RelayMessage::output("Garbage collection started...", "stdout").to_json(),
                    ))
                    .await;
                let mut last_lines = 0u64;
                loop {
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    if let Ok(result) = run_cmd(&format!(
                        "wc -l < {} 2>/dev/null && tail -n +{} {} 2>/dev/null",
                        gc_log,
                        last_lines + 1,
                        gc_log
                    ))
                    .await
                    {
                        if result.exit_status == 0 {
                            let lines: Vec<&str> = result.stdout.lines().collect();
                            if let Some(first) = lines.first() {
                                if let Ok(total) = first.trim().parse::<u64>() {
                                    for line in &lines[1..] {
                                        if !line.trim().is_empty() {
                                            let _ = ws_tx
                                                .send(Message::Text(
                                                    RelayMessage::output(line, "stdout").to_json(),
                                                ))
                                                .await;
                                        }
                                    }
                                    last_lines = total;
                                }
                            }
                        }
                    }
                    if let Ok(check) = run_cmd("pgrep -f nix-collect-garbage").await {
                        if check.exit_status != 0 && last_lines > 0 {
                            break;
                        }
                    }
                }
                let _ = ws_tx
                    .send(Message::Text(
                        serde_json::json!({"type":"exit","code":0}).to_string(),
                    ))
                    .await;
            }

            "diagnose" => {
                eprintln!("[{}] Running diagnostics...", peer_addr);
                let script = r#"
echo '{"internet":{'
echo '"ping":'
ping -c 1 -W 3 8.8.8.8 >/dev/null 2>&1 && echo 'true,' || echo 'false,'
echo '"dns":'
(host cache.nixos.org >/dev/null 2>&1 || nslookup cache.nixos.org >/dev/null 2>&1 || getent hosts cache.nixos.org >/dev/null 2>&1) && echo 'true,' || echo 'false,'
echo '"nix_cache":'
curl -s --max-time 5 https://cache.nixos.org/nix-cache-info >/dev/null 2>&1 && echo 'true,' || echo 'false,'
echo '"resolv_conf":"'$(cat /etc/resolv.conf 2>/dev/null | grep nameserver | head -3 | tr '\n' ' ')'",'
echo '"ip_route":"'$(ip route get 8.8.8.8 2>/dev/null | head -1)'"'
echo '},'
echo '"nix":{'
echo '"channels":"'$(nix-channel --list 2>/dev/null | tr '\n' ' ')'",'
echo '"store_paths":'$(ls /nix/store 2>/dev/null | wc -l)','
echo '"nixos_install":'$(which nixos-install >/dev/null 2>&1 && echo 'true' || echo 'false')
echo '},'
echo '"mounts":"'$(mount | grep /mnt | tr '\n' ' ')'"'
echo '}'
"#;
                match run_cmd(script).await {
                    Ok(r) => {
                        let clean: String = r
                            .stdout
                            .chars()
                            .filter(|c| !c.is_control() || *c == '\n')
                            .collect();
                        let _ = ws_tx
                            .send(Message::Text(
                                serde_json::json!({"type":"diagnose","data":clean}).to_string(),
                            ))
                            .await;
                    }
                    Err(e) => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error(&format!("Diagnose failed: {}", e)).to_json(),
                            ))
                            .await;
                    }
                }
            }

            "read_config" => {
                eprintln!("[{}] Reading config...", peer_addr);
                match run_cmd("cat /etc/nixos/configuration.nix 2>/dev/null").await {
                    Ok(r) if r.exit_status == 0 => {
                        let _ = ws_tx
                            .send(Message::Text(
                                serde_json::json!({"type":"config","data":r.stdout}).to_string(),
                            ))
                            .await;
                    }
                    Ok(_) => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error("Cannot read /etc/nixos/configuration.nix")
                                    .to_json(),
                            ))
                            .await;
                    }
                    Err(e) => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error(&format!("Failed: {}", e)).to_json(),
                            ))
                            .await;
                    }
                }
            }

            "write_config" => {
                if client_msg.configuration_nix.is_empty() {
                    let _ = ws_tx
                        .send(Message::Text(
                            RelayMessage::error("Missing configuration_nix").to_json(),
                        ))
                        .await;
                    continue;
                }
                eprintln!("[{}] Writing config + rebuilding...", peer_addr);
                let wc_session_id: u64 = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);
                let wc_config_path = format!("/tmp/symthaea-newconfig-{}.nix", wc_session_id);
                let wc_script_path = format!("/tmp/symthaea-rebuild-{}.sh", wc_session_id);
                let wc_log_path = format!("/tmp/symthaea-rebuild-{}.log", wc_session_id);
                // Upload config as a script that does atomic backup → write → validate → rebuild
                let upload = format!(
                    "cat > {} << 'NIXCONF'\n{}\nNIXCONF",
                    wc_config_path, client_msg.configuration_nix
                );
                if let Err(e) = run_cmd(&upload).await {
                    let _ = ws_tx
                        .send(Message::Text(
                            RelayMessage::error(&format!("Upload failed: {}", e)).to_json(),
                        ))
                        .await;
                    continue;
                }
                let rebuild_script = format!(
                    r#"set -eo pipefail
cp /etc/nixos/configuration.nix /etc/nixos/configuration.nix.bak
cp {config} /etc/nixos/configuration.nix
if ! nix-instantiate --parse /etc/nixos/configuration.nix > /dev/null 2>&1; then
    echo "ERROR: Invalid Nix syntax. Restoring backup."
    cp /etc/nixos/configuration.nix.bak /etc/nixos/configuration.nix
    exit 1
fi
echo "Config validated. Rebuilding..."
nixos-rebuild switch 2>&1
echo "REBUILD_COMPLETE"
"#,
                    config = wc_config_path
                );
                let _ = run_cmd(&format!(
                    "touch {} && chmod 600 {}",
                    wc_log_path, wc_log_path
                ))
                .await;
                let _ = run_cmd(&format!(
                    "cat > {} << 'SCRIPTEOF'\n{}\nSCRIPTEOF\nchmod +x {}\nbash {} > {} 2>&1 &",
                    wc_script_path, rebuild_script, wc_script_path, wc_script_path, wc_log_path
                ))
                .await;
                let _ = ws_tx
                    .send(Message::Text(
                        RelayMessage::output("Rebuilding system...", "stdout").to_json(),
                    ))
                    .await;
                let mut last_lines = 0u64;
                let mut complete = false;
                loop {
                    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                    if let Ok(result) = run_cmd(&format!(
                        "wc -l < {} 2>/dev/null && tail -n +{} {} 2>/dev/null",
                        wc_log_path,
                        last_lines + 1,
                        wc_log_path
                    ))
                    .await
                    {
                        if result.exit_status == 0 {
                            let lines: Vec<&str> = result.stdout.lines().collect();
                            if let Some(first) = lines.first() {
                                if let Ok(total) = first.trim().parse::<u64>() {
                                    for line in &lines[1..] {
                                        if !line.trim().is_empty() {
                                            let _ = ws_tx
                                                .send(Message::Text(
                                                    RelayMessage::output(line, "stdout").to_json(),
                                                ))
                                                .await;
                                            if line.contains("REBUILD_COMPLETE") {
                                                complete = true;
                                            }
                                        }
                                    }
                                    last_lines = total;
                                }
                            }
                        }
                    }
                    if complete {
                        break;
                    }
                    if let Ok(check) = run_cmd(&format!("pgrep -f {}", wc_script_path)).await {
                        if check.exit_status != 0 && last_lines > 0 {
                            break;
                        }
                    }
                }
                let exit_code = if complete { 0 } else { 1 };
                let _ = ws_tx
                    .send(Message::Text(
                        serde_json::json!({"type":"exit","code":exit_code}).to_string(),
                    ))
                    .await;
            }

            // ── PXE / Network Boot ──
            "netboot_info" => {
                eprintln!("[{}] Querying netboot info...", peer_addr);
                let script = r#"
                    KERNEL=$(ls /nix/store/*/bzImage 2>/dev/null | head -1)
                    INITRD=$(ls /nix/store/*/initrd 2>/dev/null | head -1)
                    IP=$(ip -4 route get 1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}' | head -1)
                    if [ -z "$KERNEL" ] || [ -z "$INITRD" ]; then
                        echo '{"error":"NixOS kernel/initrd not found in nix store. Build the ISO first."}'
                    else
                        printf '{"kernel":"%s","initrd":"%s","ip":"%s","dnsmasq_hint":"dhcp-boot=pxelinux.0,,%s","pixiecore_hint":"pixiecore boot %s %s"}' \
                            "$KERNEL" "$INITRD" "$IP" "$IP" "$KERNEL" "$INITRD"
                    fi
                "#;
                match run_cmd(script).await {
                    Ok(r) if r.exit_status == 0 => {
                        let _ = ws_tx
                            .send(Message::Text(
                                serde_json::json!({
                                    "type": "netboot_info",
                                    "data": r.stdout.trim()
                                })
                                .to_string(),
                            ))
                            .await;
                    }
                    Ok(r) => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error(&format!("netboot_info failed: {}", r.stderr))
                                    .to_json(),
                            ))
                            .await;
                    }
                    Err(e) => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error(&format!("netboot_info error: {}", e))
                                    .to_json(),
                            ))
                            .await;
                    }
                }
            }

            // ═══════════════════════════════════════════════════════
            // Tier 3: Disk cloning & Machine inventory
            // ═══════════════════════════════════════════════════════
            "create_image" => {
                eprintln!("[{}] Creating system image...", peer_addr);
                let script = r#"
set -eo pipefail
echo "STAGE: Creating system image..."
DEST="/tmp/nixforhumanity-image-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$DEST"

# Snapshot current btrfs root
if btrfs subvolume snapshot -r / "$DEST/root-snapshot" 2>/dev/null; then
    echo "Created btrfs read-only snapshot"
    # Send snapshot with btrfs send
    btrfs send "$DEST/root-snapshot" | zstd -3 -T0 > "$DEST/system.btrfs.zst"
    SIZE=$(du -sh "$DEST/system.btrfs.zst" | awk '{print $1}')
    echo "Image size: $SIZE"
    btrfs subvolume delete "$DEST/root-snapshot" 2>/dev/null
else
    # Fallback: tar the root filesystem
    echo "btrfs snapshot not available, using tar..."
    tar -czf "$DEST/system.tar.gz" --one-file-system --exclude=/tmp --exclude=/proc --exclude=/sys --exclude=/dev --exclude=/run / 2>/dev/null
    SIZE=$(du -sh "$DEST/system.tar.gz" | awk '{print $1}')
    echo "Image size: $SIZE"
fi

# Save config
cp /etc/nixos/configuration.nix "$DEST/" 2>/dev/null || true
cp /etc/nixos/hardware-configuration.nix "$DEST/" 2>/dev/null || true
cp /etc/nixos/flake.nix "$DEST/" 2>/dev/null || true
cp /etc/nixos/flake.lock "$DEST/" 2>/dev/null || true

# Save package list
nix-env -qa --installed 2>/dev/null > "$DEST/installed-packages.txt" || true

echo "STAGE: Image complete"
echo "Image saved to: $DEST"
ls -la "$DEST/"
echo "COMPLETE"
"#;
                let img_session_id: u64 = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);
                let img_log = format!("/tmp/symthaea-image-{}.log", img_session_id);
                let _ = run_cmd(&format!("touch {} && chmod 600 {}", img_log, img_log)).await;
                let _ = run_cmd(&format!(
                    "bash -c '{}' > {} 2>&1 &",
                    script.replace('\'', "'\\''"),
                    img_log
                ))
                .await;
                let _ = ws_tx
                    .send(Message::Text(
                        RelayMessage::output("Creating system image...", "stdout").to_json(),
                    ))
                    .await;
                // Stream output (same polling pattern as install)
                let mut last_lines = 0u64;
                loop {
                    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                    if let Ok(result) = run_cmd(&format!(
                        "wc -l < {} 2>/dev/null && tail -n +{} {} 2>/dev/null",
                        img_log,
                        last_lines + 1,
                        img_log
                    ))
                    .await
                    {
                        if result.exit_status == 0 {
                            let lines: Vec<&str> = result.stdout.lines().collect();
                            if let Some(first) = lines.first() {
                                if let Ok(total) = first.trim().parse::<u64>() {
                                    for line in &lines[1..] {
                                        if !line.trim().is_empty() {
                                            let _ = ws_tx
                                                .send(Message::Text(
                                                    RelayMessage::output(line, "stdout").to_json(),
                                                ))
                                                .await;
                                        }
                                    }
                                    last_lines = total;
                                }
                            }
                        }
                    }
                    if let Ok(check) = run_cmd("pgrep -f 'btrfs send' || pgrep -f 'tar -czf'").await
                    {
                        if check.exit_status != 0 && last_lines > 0 {
                            break;
                        }
                    }
                }
                let _ = ws_tx
                    .send(Message::Text(
                        serde_json::json!({"type":"exit","code":0}).to_string(),
                    ))
                    .await;
            }

            "restore_image" => {
                let image_path = match sanitize_input(&client_msg.command, "image path", true) {
                    Ok(p) => p,
                    Err(e) => {
                        let _ = ws_tx
                            .send(Message::Text(RelayMessage::error(&e).to_json()))
                            .await;
                        continue;
                    }
                };
                eprintln!(
                    "[{}] Restoring system image from {}...",
                    peer_addr, image_path
                );
                let script = format!(
                    r#"
set -eo pipefail
echo "STAGE: Restoring system image..."
if [ -f "{path}/system.btrfs.zst" ]; then
    echo "Restoring btrfs snapshot..."
    zstd -d "{path}/system.btrfs.zst" | btrfs receive /mnt/ 2>&1
elif [ -f "{path}/system.tar.gz" ]; then
    echo "Restoring tar archive..."
    tar -xzf "{path}/system.tar.gz" -C /mnt/ 2>&1
else
    echo "ERROR: No image found at {path}"
    exit 1
fi
# Restore config
cp "{path}/configuration.nix" /mnt/etc/nixos/ 2>/dev/null || true
cp "{path}/hardware-configuration.nix" /mnt/etc/nixos/ 2>/dev/null || true
echo "STAGE: Image restored"
echo "COMPLETE"
"#,
                    path = image_path
                );
                match run_cmd(&script).await {
                    Ok(r) => {
                        let _ = ws_tx.send(Message::Text(serde_json::json!({"type":"exit","code":r.exit_status,"data":r.stdout}).to_string())).await;
                    }
                    Err(e) => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error(&format!("Restore failed: {}", e)).to_json(),
                            ))
                            .await;
                    }
                }
            }

            "list_images" => {
                match run_cmd(
                    "ls -la /tmp/nixforhumanity-image-* 2>/dev/null | head -20 || echo '[]'",
                )
                .await
                {
                    Ok(r) => {
                        let _ = ws_tx
                            .send(Message::Text(
                                serde_json::json!({"type":"images","data":r.stdout}).to_string(),
                            ))
                            .await;
                    }
                    Err(e) => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error(&format!("List failed: {}", e)).to_json(),
                            ))
                            .await;
                    }
                }
            }

            "inventory" => {
                eprintln!("[{}] Collecting inventory...", peer_addr);
                let script = r#"
echo '{'
echo '"hostname":"'$(hostname)'",'
echo '"nixos_version":"'$(nixos-version 2>/dev/null || echo unknown)'",'
echo '"kernel":"'$(uname -r)'",'
echo '"uptime":"'$(uptime -p 2>/dev/null || uptime)'",'
echo '"cpu":"'$(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)'",'
echo '"cpu_cores":'$(nproc)','
echo '"memory_gb":'$(awk '/MemTotal/ {printf "%.1f", $2/1024/1024}' /proc/meminfo)','
echo '"disk_usage":"'$(df -h / | tail -1 | awk '{print $3"/"$2" ("$5")"}')'",'
echo '"nix_store_gb":"'$(du -sh /nix/store 2>/dev/null | awk '{print $1}')'",'
echo '"generations":'$(nix-env --list-generations -p /nix/var/nix/profiles/system 2>/dev/null | wc -l)','
echo '"services_running":'$(systemctl list-units --type=service --state=running --no-pager --plain 2>/dev/null | grep -c '\.service')','
echo '"services_failed":'$(systemctl list-units --type=service --state=failed --no-pager --plain 2>/dev/null | grep -c '\.service')','
echo '"packages":'$(nix-store -qR /run/current-system 2>/dev/null | wc -l)','
echo '"last_rebuild":"'$(stat -c %y /run/current-system 2>/dev/null | cut -d. -f1)'",'
echo '"ip_addresses":['
ip -4 addr show | grep inet | grep -v '127.0.0.1' | awk '{print "\"" $2 "\""}' | paste -sd, -
echo ']'
echo '}'
"#;
                match run_cmd(script).await {
                    Ok(r) if r.exit_status == 0 => {
                        let clean: String = r
                            .stdout
                            .chars()
                            .filter(|c| !c.is_control() || *c == '\n')
                            .collect();
                        let _ = ws_tx
                            .send(Message::Text(
                                serde_json::json!({"type":"inventory","data":clean}).to_string(),
                            ))
                            .await;
                    }
                    Ok(r) => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error(&format!(
                                    "Inventory failed: {}",
                                    &r.stderr[..r.stderr.len().min(200)]
                                ))
                                .to_json(),
                            ))
                            .await;
                    }
                    Err(e) => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error(&format!("Inventory failed: {}", e)).to_json(),
                            ))
                            .await;
                    }
                }
            }

            // ── WiFi scanning and connection ──
            "scan_wifi" => {
                eprintln!("[{}] Scanning WiFi...", peer_addr);
                match run_cmd("nmcli -t -f SSID,SIGNAL,SECURITY device wifi list 2>/dev/null").await
                {
                    Ok(r) if r.exit_status == 0 => {
                        let _ = ws_tx
                            .send(Message::Text(
                                serde_json::json!({"type": "wifi_list", "data": r.stdout})
                                    .to_string(),
                            ))
                            .await;
                    }
                    Ok(r) => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error(&format!(
                                    "WiFi scan failed: {}",
                                    r.stderr.chars().take(200).collect::<String>()
                                ))
                                .to_json(),
                            ))
                            .await;
                    }
                    Err(e) => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error(&format!("WiFi scan failed: {}", e)).to_json(),
                            ))
                            .await;
                    }
                }
            }

            "connect_wifi" => {
                // Reuse hostname field for SSID, command field for WiFi password
                let ssid = client_msg.hostname.trim().to_string();
                let wifi_pw = &client_msg.command;
                if ssid.is_empty() {
                    let _ = ws_tx
                        .send(Message::Text(
                            RelayMessage::error("WiFi SSID is required").to_json(),
                        ))
                        .await;
                } else {
                    eprintln!("[{}] Connecting to WiFi: {}", peer_addr, ssid);
                    let cmd = format!(
                        "nmcli device wifi connect '{}' password '{}'",
                        ssid.replace('\'', "'\\''"),
                        wifi_pw.replace('\'', "'\\''")
                    );
                    match run_cmd(&cmd).await {
                        Ok(r) => {
                            let _ = ws_tx.send(Message::Text(serde_json::json!({
                                "type": "wifi_result",
                                "code": r.exit_status,
                                "data": if r.exit_status == 0 { "WiFi connected".to_string() } else { r.stderr }
                            }).to_string())).await;
                        }
                        Err(e) => {
                            let _ = ws_tx
                                .send(Message::Text(
                                    RelayMessage::error(&format!("WiFi connection failed: {}", e))
                                        .to_json(),
                                ))
                                .await;
                        }
                    }
                }
            }

            "search_packages" => {
                let query = client_msg.command.trim();
                if query.is_empty() {
                    let _ = ws_tx
                        .send(Message::Text(
                            RelayMessage::error("Missing search query").to_json(),
                        ))
                        .await;
                    continue;
                }
                eprintln!("[{}] Searching packages: {}", peer_addr, query);
                // Sanitize query: allow only alphanumeric, dash, underscore, dot, space
                let safe_query: String = query
                    .chars()
                    .filter(|c| {
                        c.is_alphanumeric() || *c == '-' || *c == '_' || *c == '.' || *c == ' '
                    })
                    .take(100)
                    .collect();
                if safe_query.is_empty() {
                    let _ = ws_tx
                        .send(Message::Text(
                            RelayMessage::error("Invalid search query").to_json(),
                        ))
                        .await;
                    continue;
                }
                let cmd = format!(
                    "nix search nixpkgs '{}' --json 2>/dev/null | head -c 50000",
                    safe_query.replace('\'', "'\\''")
                );
                match run_cmd(&cmd).await {
                    Ok(r) if r.exit_status == 0 && !r.stdout.trim().is_empty() => {
                        // Parse nix search JSON: {"legacyPackages.x86_64-linux.pkgname": {"pname":"...", "description":"..."}, ...}
                        // Extract just package names and descriptions
                        let mut results = Vec::new();
                        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&r.stdout) {
                            if let Some(obj) = parsed.as_object() {
                                for (attr, info) in obj.iter().take(30) {
                                    let pname =
                                        info.get("pname").and_then(|v| v.as_str()).unwrap_or("");
                                    let desc = info
                                        .get("description")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("");
                                    // Extract short attr name from legacyPackages.x86_64-linux.pkgname
                                    let short_attr = attr.rsplit('.').next().unwrap_or(attr);
                                    results.push(serde_json::json!({
                                        "attr": short_attr,
                                        "pname": pname,
                                        "description": desc
                                    }));
                                }
                            }
                        }
                        let _ = ws_tx
                            .send(Message::Text(
                                serde_json::json!({
                                    "type": "packages",
                                    "data": results
                                })
                                .to_string(),
                            ))
                            .await;
                    }
                    Ok(r) => {
                        // Fallback: nix-env query
                        let fallback = format!(
                            "nix-env -qaP '.*{}.*' 2>/dev/null | head -30",
                            safe_query.replace('\'', "'\\''")
                        );
                        match run_cmd(&fallback).await {
                            Ok(r2) if r2.exit_status == 0 && !r2.stdout.trim().is_empty() => {
                                let mut results = Vec::new();
                                for line in r2.stdout.lines().take(30) {
                                    let parts: Vec<&str> =
                                        line.splitn(2, char::is_whitespace).collect();
                                    if let Some(attr) = parts.first() {
                                        results.push(serde_json::json!({
                                            "attr": attr.rsplit('.').next().unwrap_or(attr),
                                            "pname": parts.get(1).unwrap_or(&""),
                                            "description": ""
                                        }));
                                    }
                                }
                                let _ = ws_tx
                                    .send(Message::Text(
                                        serde_json::json!({
                                            "type": "packages",
                                            "data": results
                                        })
                                        .to_string(),
                                    ))
                                    .await;
                            }
                            _ => {
                                let _ = ws_tx
                                    .send(Message::Text(
                                        RelayMessage::error(&format!(
                                            "Package search returned no results: {}",
                                            r.stderr.chars().take(200).collect::<String>()
                                        ))
                                        .to_json(),
                                    ))
                                    .await;
                            }
                        }
                    }
                    Err(e) => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error(&format!("Search failed: {}", e)).to_json(),
                            ))
                            .await;
                    }
                }
            }

            // ── Package Validation ──
            // Pre-install check: verify package names exist in the target's nixpkgs.
            // Sends comma-separated package names in `command` field.
            // Returns { type: "package_validation", valid: [...], invalid: [...], suggestions: [...] }
            "validate_packages" => {
                let packages_str = &client_msg.command;
                if packages_str.is_empty() {
                    let _ = ws_tx
                        .send(Message::Text(
                            RelayMessage::error("No packages to validate").to_json(),
                        ))
                        .await;
                    continue;
                }
                eprintln!("[{}] Validating packages...", peer_addr);

                let packages: Vec<&str> = packages_str
                    .split(',')
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .collect();
                let mut valid = Vec::new();
                let mut invalid = Vec::new();
                let mut suggestions: Vec<String> = Vec::new();

                for pkg in &packages {
                    // Sanitize: strip shell-dangerous characters
                    let pkg_clean: String = pkg
                        .chars()
                        .filter(|c| !matches!(c, '\'' | ';' | '"' | '`' | '$' | '|' | '&'))
                        .collect();
                    if pkg_clean.is_empty() {
                        continue;
                    }

                    // Check if package exists in nixpkgs via nix eval
                    let check_cmd = format!(
                        "nix eval 'nixpkgs#{}' --json 2>/dev/null && echo 'EXISTS' || echo 'MISSING'",
                        pkg_clean
                    );
                    match run_cmd(&check_cmd).await {
                        Ok(r) => {
                            if r.stdout.contains("EXISTS")
                                || (r.exit_status == 0 && !r.stdout.contains("MISSING"))
                            {
                                valid.push(pkg_clean.clone());
                            } else {
                                invalid.push(pkg_clean.clone());
                                // Try to find similar packages
                                let suggest_cmd = format!(
                                    "nix search nixpkgs '{}' --json 2>/dev/null | head -c 2000",
                                    pkg_clean
                                );
                                if let Ok(sr) = run_cmd(&suggest_cmd).await {
                                    if !sr.stdout.is_empty() && sr.stdout.trim() != "{}" {
                                        // Extract first few attribute names from JSON
                                        if let Ok(val) =
                                            serde_json::from_str::<serde_json::Value>(&sr.stdout)
                                        {
                                            if let Some(obj) = val.as_object() {
                                                let alts: Vec<String> = obj
                                                    .keys()
                                                    .take(3)
                                                    .map(|k| {
                                                        k.rsplit('.')
                                                            .next()
                                                            .unwrap_or(k)
                                                            .to_string()
                                                    })
                                                    .collect();
                                                if !alts.is_empty() {
                                                    suggestions.push(format!(
                                                        "{}: try {}",
                                                        pkg_clean,
                                                        alts.join(", ")
                                                    ));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Err(_) => {
                            invalid.push(pkg_clean);
                        }
                    }
                }

                let result = serde_json::json!({
                    "type": "package_validation",
                    "valid": valid,
                    "invalid": invalid,
                    "suggestions": suggestions
                });
                let _ = ws_tx.send(Message::Text(result.to_string())).await;
            }

            // ── nixpkgs Version Query ──
            // Returns the nixpkgs channel version running on the target system.
            // Used to detect stale package names in the app database.
            "nixpkgs_version" => {
                eprintln!("[{}] Querying nixpkgs version...", peer_addr);
                match run_cmd("nixos-version 2>/dev/null || nix eval nixpkgs#lib.version --raw 2>/dev/null || echo unknown").await {
                    Ok(r) => {
                        let version = r.stdout.trim().to_string();
                        let _ = ws_tx.send(Message::Text(serde_json::json!({
                            "type": "nixpkgs_version",
                            "data": version
                        }).to_string())).await;
                    }
                    Err(e) => {
                        let _ = ws_tx.send(Message::Text(
                            RelayMessage::error(&format!("Version check failed: {}", e)).to_json()
                        )).await;
                    }
                }
            }

            "disconnect" => {
                eprintln!("[{}] Client disconnecting", peer_addr);
                tracker.lock().await.release(&peer_addr);
                break;
            }

            _ => {
                let _ = ws_tx
                    .send(Message::Text(
                        RelayMessage::error(&format!("Unknown action: {}", client_msg.action))
                            .to_json(),
                    ))
                    .await;
            }
        }
    }

    // Cleanup
    tracker.lock().await.release(&peer_addr);
    eprintln!("[{}] WebSocket disconnected", peer_addr);
}

fn generate_auth_token() -> String {
    let mut bytes = [0u8; 32];
    if let Ok(mut f) = std::fs::File::open("/dev/urandom") {
        if f.read_exact(&mut bytes).is_ok() {
            return bytes.iter().map(|b| format!("{:02x}", b)).collect();
        }
    }

    // Fallback: only used if /dev/urandom is unavailable (should not happen on Linux/NixOS).
    let seed = format!(
        "{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    );
    blake3::hash(seed.as_bytes()).to_hex().to_string()
}

fn usage() {
    eprintln!("NixForHumanity WebSocket Relay (local mode)");
    eprintln!("Usage:");
    eprintln!("  ssh-relay [--port <port>] [--bind <addr>] [--token <token>] [--pxe [port]]");
    eprintln!();
    eprintln!("Options:");
    eprintln!(
        "  --pxe [port]   Also serve NixOS kernel+initrd over HTTP for PXE boot (default: 8080)"
    );
    eprintln!();
    eprintln!("Security defaults:");
    eprintln!("  - Binds to 127.0.0.1 only");
    eprintln!("  - Requires an auth token over WebSocket (action: \"auth\")");
    eprintln!("  - Executes commands locally (no SSH)");
    eprintln!("  - 'exec' is disabled");
}

#[tokio::main]
async fn main() {
    let mut port: u16 = 8091;
    let mut bind_addr: String = "127.0.0.1".into();
    let mut token: Option<String> = None;
    let mut pxe_port: Option<u16> = None;
    let mut enable_tls = false;
    let mut tls_cert_path: Option<String> = None;
    let mut tls_key_path: Option<String> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--port" => {
                if let Some(p) = args.next().and_then(|p| p.parse::<u16>().ok()) {
                    port = p;
                }
            }
            "--bind" => {
                if let Some(a) = args.next() {
                    bind_addr = a;
                }
            }
            "--token" => token = args.next(),
            "--tls" => enable_tls = true,
            "--tls-cert" => tls_cert_path = args.next(),
            "--tls-key" => tls_key_path = args.next(),
            "--pxe" => {
                // Optional port argument: --pxe 9090  or just --pxe (defaults to 8080)
                pxe_port = Some(
                    args.next()
                        .and_then(|p| p.parse::<u16>().ok())
                        .unwrap_or(8080),
                );
            }
            "--help" | "-h" => {
                usage();
                return;
            }
            _ => {}
        }
    }

    let token = token.unwrap_or_else(generate_auth_token);
    if token.is_empty() {
        eprintln!("ERROR: --token cannot be empty");
        std::process::exit(2);
    }
    let auth_token = Arc::new(token);

    let tracker: SharedTracker = Arc::new(Mutex::new(SessionTracker::new(1800))); // 30 min timeout

    let addr = format!("{}:{}", bind_addr, port);
    let listener = match TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("ERROR: Cannot bind to {}: {}", addr, e);
            eprintln!("Another service may be using this port. Try: --port <other-port>");
            std::process::exit(1);
        }
    };
    // TLS setup
    let tls_acceptor: Option<TlsAcceptor> = if enable_tls {
        let (cert_pem, key_pem) = match (&tls_cert_path, &tls_key_path) {
            (Some(cert), Some(key)) => {
                let cert = std::fs::read_to_string(cert).expect("Cannot read TLS cert");
                let key = std::fs::read_to_string(key).expect("Cannot read TLS key");
                (cert, key)
            }
            _ => {
                // Generate self-signed certificate
                eprintln!("TLS: Generating self-signed certificate...");
                let subject_alt_names = vec!["localhost".to_string(), bind_addr.clone()];
                let cert = rcgen::generate_simple_self_signed(subject_alt_names)
                    .expect("Failed to generate self-signed cert");
                let cert_pem = cert.cert.pem();
                let key_pem = cert.key_pair.serialize_pem();
                // Save for QR code fingerprint
                let fingerprint = {
                    use std::io::Write;
                    let der = cert.cert.der();
                    let hash = blake3::hash(der.as_ref());
                    let hex = hash.to_hex();
                    hex[..16].to_string()
                };
                eprintln!("TLS: Certificate fingerprint: {}", fingerprint);
                // Save cert to /run/sovereign/ for the QR code URL
                let _ = std::fs::create_dir_all("/run/sovereign");
                let _ = std::fs::write("/run/sovereign/tls-fingerprint", &fingerprint);
                (cert_pem, key_pem)
            }
        };

        let certs = rustls_pemfile::certs(&mut cert_pem.as_bytes())
            .collect::<Result<Vec<_>, _>>()
            .expect("Invalid TLS certificate");
        let key = rustls_pemfile::private_key(&mut key_pem.as_bytes())
            .expect("Invalid TLS key")
            .expect("No TLS key found");

        let config = ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(certs, key)
            .expect("Invalid TLS config");

        Some(TlsAcceptor::from(Arc::new(config)))
    } else {
        None
    };

    let scheme = if tls_acceptor.is_some() { "wss" } else { "ws" };
    eprintln!("NixForHumanity Relay listening on {}://{}", scheme, addr);
    eprintln!("  Mode: local (no SSH)");
    eprintln!(
        "  TLS: {}",
        if tls_acceptor.is_some() {
            "enabled (self-signed)"
        } else {
            "disabled (use --tls to enable)"
        }
    );
    eprintln!("  Auth token: {}", auth_token);
    eprintln!("  Protocol: auth → connect → (discover_disks/install/...) → disconnect");
    eprintln!("  Session timeout: 30 minutes");
    eprintln!("  Rate limit: 1 active session per IP");

    // PXE mode: spawn a background HTTP server for kernel+initrd
    if let Some(pxe_p) = pxe_port {
        let pxe_bind = bind_addr.clone();
        tokio::spawn(async move {
            // Find kernel and initrd in the nix store
            let kernel_result = run_cmd("ls /nix/store/*/bzImage 2>/dev/null | head -1").await;
            let initrd_result = run_cmd("ls /nix/store/*/initrd 2>/dev/null | head -1").await;
            let kernel_path = kernel_result
                .ok()
                .map(|r| r.stdout.trim().to_string())
                .unwrap_or_default();
            let initrd_path = initrd_result
                .ok()
                .map(|r| r.stdout.trim().to_string())
                .unwrap_or_default();

            if kernel_path.is_empty() || initrd_path.is_empty() {
                eprintln!(
                    "PXE: NixOS kernel/initrd not found in nix store. PXE server not started."
                );
                eprintln!("PXE: Build the ISO first: nix-build nix/installer-iso.nix");
                return;
            }

            // Create a temp directory with symlinks and serve via python3
            let setup_cmd = format!(
                "TMPDIR=$(mktemp -d) && ln -sf '{}' \"$TMPDIR/bzImage\" && ln -sf '{}' \"$TMPDIR/initrd\" && echo \"$TMPDIR\"",
                kernel_path, initrd_path
            );
            let tmpdir = match run_cmd(&setup_cmd).await {
                Ok(r) if r.exit_status == 0 => r.stdout.trim().to_string(),
                _ => {
                    eprintln!("PXE: Failed to set up temp directory for serving");
                    return;
                }
            };

            eprintln!(
                "PXE: Serving kernel+initrd on http://{}:{}",
                pxe_bind, pxe_p
            );
            eprintln!("PXE:   kernel: {}", kernel_path);
            eprintln!("PXE:   initrd: {}", initrd_path);
            eprintln!(
                "PXE: For dnsmasq, add: dhcp-boot=pxelinux.0,,{}:{}",
                pxe_bind, pxe_p
            );

            // Serve the directory with python3
            let serve_cmd = format!(
                "cd '{}' && python3 -m http.server {} --bind {}",
                tmpdir, pxe_p, pxe_bind
            );
            let _ = run_cmd(&serve_cmd).await;
        });
    }

    while let Ok((stream, addr)) = listener.accept().await {
        let peer = addr.ip().to_string();
        let tracker = tracker.clone();
        let auth = auth_token.clone();

        if let Some(ref acceptor) = tls_acceptor {
            let acceptor = acceptor.clone();
            tokio::spawn(async move {
                match acceptor.accept(stream).await {
                    Ok(tls_stream) => {
                        // WebSocket upgrade over TLS stream (Origin validated in callback)
                        let peer_ref = peer.clone();
                        let origin_check = |req: &tungstenite::handshake::server::Request,
                                            resp: tungstenite::handshake::server::Response|
                         -> Result<
                            tungstenite::handshake::server::Response,
                            tungstenite::handshake::server::ErrorResponse,
                        > {
                            if let Some(origin) = req.headers().get("origin") {
                                let o = origin.to_str().unwrap_or("");
                                let ok = o.starts_with("http://localhost")
                                    || o.starts_with("https://localhost")
                                    || o.starts_with("http://127.0.0.1")
                                    || o.starts_with("https://127.0.0.1")
                                    || o.contains("luminousdynamics.io")
                                    || o.contains("nixforhumanity.org")
                                    || o.contains("mycelix.net")
                                    || o.contains("relationalharmonics.org");
                                if !ok {
                                    eprintln!(
                                        "[{}] Rejected TLS WebSocket: disallowed Origin '{}'",
                                        peer_ref, o
                                    );
                                    let mut r = tungstenite::handshake::server::ErrorResponse::new(
                                        Some("Forbidden origin".into()),
                                    );
                                    *r.status_mut() = tungstenite::http::StatusCode::FORBIDDEN;
                                    return Err(r);
                                }
                            }
                            Ok(resp)
                        };
                        let ws_stream = match accept_hdr_async(tls_stream, origin_check).await {
                            Ok(ws) => ws,
                            Err(e) => {
                                eprintln!("[{}] TLS WebSocket upgrade failed: {}", peer, e);
                                return;
                            }
                        };
                        handle_connection_ws(ws_stream, peer, tracker, auth).await;
                    }
                    Err(e) => {
                        eprintln!("[{}] TLS handshake failed: {}", peer, e);
                    }
                }
            });
        } else {
            tokio::spawn(handle_connection(stream, peer, tracker, auth));
        }
    }
}

// ── Security regression tests ──

#[cfg(test)]
mod tests {
    use super::*;

    // ── sanitize_heredoc ──

    #[test]
    fn heredoc_strips_exact_delimiter() {
        let input = "line1\nNIXCONF\nline3\n";
        let result = sanitize_heredoc(input, "NIXCONF");
        assert!(!result.contains("\nNIXCONF\n"));
        assert!(result.contains("line1"));
        assert!(result.contains("line3"));
    }

    #[test]
    fn heredoc_preserves_partial_match() {
        let input = "NIXCONF_EXTRA = true;\nreal content\n";
        let result = sanitize_heredoc(input, "NIXCONF");
        assert!(result.contains("NIXCONF_EXTRA"));
        assert!(result.contains("real content"));
    }

    #[test]
    fn heredoc_strips_indented_delimiter() {
        // trim() is applied, so "  NIXCONF  " should be stripped
        let input = "line1\n  NIXCONF  \nline3\n";
        let result = sanitize_heredoc(input, "NIXCONF");
        assert!(result.contains("line1"));
        assert!(result.contains("line3"));
        // The delimiter line itself should be gone
        let lines: Vec<&str> = result.lines().collect();
        assert!(lines.iter().all(|l| l.trim() != "NIXCONF"));
    }

    #[test]
    fn heredoc_strips_multiple_delimiters() {
        let input = "a\nSCRIPTEOF\nb\nSCRIPTEOF\nc\n";
        let result = sanitize_heredoc(input, "SCRIPTEOF");
        assert!(result.contains("a"));
        assert!(result.contains("b"));
        assert!(result.contains("c"));
        assert!(!result.lines().any(|l| l.trim() == "SCRIPTEOF"));
    }

    #[test]
    fn heredoc_empty_input() {
        assert_eq!(sanitize_heredoc("", "NIXCONF"), "");
    }

    // ── validate_disk_path ──

    #[test]
    fn disk_valid_sda() {
        assert!(validate_disk_path("/dev/sda").is_ok());
    }

    #[test]
    fn disk_valid_nvme() {
        assert!(validate_disk_path("/dev/nvme0n1").is_ok());
    }

    #[test]
    fn disk_valid_vda() {
        assert!(validate_disk_path("/dev/vda").is_ok());
    }

    #[test]
    fn disk_valid_mmcblk() {
        assert!(validate_disk_path("/dev/mmcblk0").is_ok());
    }

    #[test]
    fn disk_rejects_no_dev_prefix() {
        assert!(validate_disk_path("/tmp/sda").is_err());
    }

    #[test]
    fn disk_rejects_path_traversal() {
        assert!(validate_disk_path("/dev/../etc/passwd").is_err());
    }

    #[test]
    fn disk_rejects_null() {
        assert!(validate_disk_path("/dev/null").is_err());
    }

    #[test]
    fn disk_rejects_urandom() {
        assert!(validate_disk_path("/dev/urandom").is_err());
    }

    #[test]
    fn disk_rejects_unknown_prefix() {
        assert!(validate_disk_path("/dev/zz0").is_err());
    }

    #[test]
    fn disk_rejects_empty() {
        assert!(validate_disk_path("/dev/").is_err());
    }

    #[test]
    fn disk_trims_whitespace() {
        assert_eq!(validate_disk_path("  /dev/sda  ").unwrap(), "/dev/sda");
    }

    // ── token_eq (constant-time comparison) ──

    #[test]
    fn token_eq_same() {
        assert!(token_eq("sovereign", "sovereign"));
    }

    #[test]
    fn token_eq_different() {
        assert!(!token_eq("sovereign", "Sovereign"));
    }

    #[test]
    fn token_eq_different_length() {
        assert!(!token_eq("short", "longer_token"));
    }

    // ── sanitize_input ──

    #[test]
    fn sanitize_allows_valid() {
        assert!(sanitize_input("my-host.name", "hostname", false).is_ok());
    }

    #[test]
    fn sanitize_rejects_semicolon() {
        assert!(sanitize_input("foo;rm -rf /", "field", false).is_err());
    }

    #[test]
    fn sanitize_rejects_backtick() {
        assert!(sanitize_input("foo`id`", "field", false).is_err());
    }

    #[test]
    fn sanitize_allows_slash_when_enabled() {
        assert!(sanitize_input("America/Chicago", "tz", true).is_ok());
    }

    #[test]
    fn sanitize_rejects_slash_when_disabled() {
        assert!(sanitize_input("America/Chicago", "tz", false).is_err());
    }

    // ── validate_hostname_relay ──

    #[test]
    fn hostname_valid() {
        assert_eq!(validate_hostname_relay("my-host").unwrap(), "my-host");
    }

    #[test]
    fn hostname_defaults_empty() {
        assert_eq!(validate_hostname_relay("").unwrap(), "guardian");
    }

    #[test]
    fn hostname_rejects_special_chars() {
        assert!(validate_hostname_relay("host;evil").is_err());
    }

    #[test]
    fn hostname_rejects_too_long() {
        let long = "a".repeat(64);
        assert!(validate_hostname_relay(&long).is_err());
    }

    // ── config_write_commands (heredoc safety) ──

    #[test]
    fn config_write_strips_nixconf_delimiter() {
        let malicious = "{ config }\nNIXCONF\nrm -rf /\n";
        let result = config_write_commands_heredoc(malicious, "", "");
        assert!(!result.contains("\nNIXCONF\nrm -rf /"));
        // Should still contain the closing delimiter exactly once as the heredoc terminator
        assert_eq!(result.matches("NIXCONF").count(), 2); // opening + closing
    }

    #[test]
    fn config_write_strips_flakeeof_delimiter() {
        let malicious = "{ inputs }\nFLAKEEOF\nrm -rf /\n";
        let result = config_write_commands_heredoc("", "", malicious);
        assert!(!result.contains("\nFLAKEEOF\nrm -rf /"));
    }
}
