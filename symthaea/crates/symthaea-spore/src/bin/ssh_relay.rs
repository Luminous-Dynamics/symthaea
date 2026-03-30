// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Symthaea SSH WebSocket Relay
//!
//! Bridges browser WebSocket connections to SSH targets for nixos-anywhere
//! orchestration. The browser portal connects via WebSocket, sends SSH
//! commands, and receives streaming output with nixos-anywhere stage detection.
//!
//! # Usage
//! ```bash
//! cargo run --bin ssh-relay --features server -- --port 8091
//! ```

use async_ssh2_tokio::client::{AuthMethod, Client, ServerCheckMethod};
use futures_util::{SinkExt, StreamExt};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::Message;

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

/// Rate limiter: 1 active session per IP.
struct SessionTracker {
    active: HashMap<String, Instant>,
    timeout_secs: u64,
}

impl SessionTracker {
    fn new(timeout_secs: u64) -> Self {
        Self {
            active: HashMap::new(),
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
}

/// Client → Relay message.
#[derive(serde::Deserialize)]
struct ClientMessage {
    action: String,
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
    disk: String,           // e.g., "/dev/nvme0n1"
    #[serde(default)]
    layout: String,         // "single", "dual", "alongside", "sata", "vps"
    #[serde(default)]
    fast_disk: String,      // For dual-disk: fast drive
    #[serde(default)]
    standard_disk: String,  // For dual-disk: standard drive
    #[serde(default)]
    hostname: String,
    #[serde(default)]
    flake_nix: String,      // Generated flake.nix content
    #[serde(default)]
    disko_nix: String,      // Generated disko-config.nix content
    #[serde(default)]
    hardware_nix: String,   // Generated hardware-configuration.nix content
    #[serde(default)]
    secure_boot: bool,      // Enable Secure Boot (lanzaboote + sbctl)
    #[serde(default)]
    tpm2_unlock: bool,      // Enable TPM2 auto-unlock (requires LUKS + systemd initrd)
    #[serde(default)]
    desktop: String,        // Desktop environment: gnome, plasma, hyprland, sway, xfce, none
    #[serde(default)]
    gpu_driver: String,     // GPU driver: nvidia, nvidia-open, amdgpu, modesetting, none
    #[serde(default)]
    timezone: String,       // e.g., "America/Chicago"
    #[serde(default)]
    keyboard: String,       // e.g., "us", "de", "dvorak"
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
    format!(r#"
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
"#, disk = disk)
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

/// Generate NixOS configuration snippet for desktop environment, GPU, locale.
fn generate_system_config(msg: &ClientMessage) -> String {
    let mut config = String::new();

    // Timezone
    let tz = if msg.timezone.is_empty() { "UTC" } else { &msg.timezone };
    config.push_str(&format!("  time.timeZone = \"{}\";\n", tz));

    // Locale
    config.push_str("  i18n.defaultLocale = \"en_US.UTF-8\";\n");

    // Keyboard
    let kb = if msg.keyboard.is_empty() { "us" } else { &msg.keyboard };
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

/// Generate a shell snippet that patches configuration.nix with system config (DE, GPU, locale).
/// Appended after the configuration.nix heredoc in each layout.
fn system_config_patch(msg: &ClientMessage) -> String {
    let sys_config = generate_system_config(msg);
    if sys_config.trim().is_empty() {
        return String::new();
    }
    // Insert the system config lines before the closing `}` of configuration.nix
    format!(
        r#"
# Patch configuration.nix with user's system choices (DE, GPU, locale, networking)
NIXCONF_FILE="/mnt/etc/nixos/configuration.nix"
PATCH_LINES=$(cat << 'SYSPATCH'
  # ── System Configuration (Sovereign Inoculation) ──
{sys_config}
  # Audio (PipeWire)
  services.pulseaudio.enable = false;
  security.rtkit.enable = true;
  services.pipewire = {{ enable = true; alsa.enable = true; pulse.enable = true; }};

  # Nix settings
  nix.settings.experimental-features = [ "nix-command" "flakes" ];
  nix.gc = {{ automatic = true; dates = "weekly"; options = "--delete-older-than 30d"; }};
SYSPATCH
)
# Insert before the last closing brace
sed -i "/^}}$/i\\$PATCH_LINES" "$NIXCONF_FILE" 2>/dev/null || echo "  (config patch: manual insertion needed)"
"#,
        sys_config = sys_config,
    )
}

/// Generate the automated install script based on layout type.
fn generate_install_script(msg: &ClientMessage) -> String {
    let hostname = if msg.hostname.is_empty() { "guardian" } else { &msg.hostname };

    match msg.layout.as_str() {
        "alongside" => {
            // Alongside Windows/Linux: find free space, reuse existing ESP, install
            format!(r#"set -eo pipefail
echo "=== Symthaea Sovereign Birth: Alongside Existing OS ==="

DISK="{disk}"

# Safety: Check for BitLocker
echo "STAGE: Checking for BitLocker..."
if blkid "$DISK"* 2>/dev/null | grep -qi bitlocker; then
  echo "WARNING: BitLocker encryption detected on this disk."
  echo "WARNING: Modifying the partition table may trigger BitLocker recovery."
  echo "WARNING: Ensure you have your BitLocker recovery key before proceeding."
  echo "WARNING: Consider shrinking the Windows partition from within Windows first."
fi

# Step 1: Find unallocated space on the disk
echo "STAGE: Detecting free space on {disk}..."
LAST_END=$(sgdisk -p "$DISK" 2>/dev/null | grep '^ ' | tail -1 | awk '{{print $3}}')
DISK_END=$(sgdisk -p "$DISK" 2>/dev/null | grep 'Disk size' | awk '{{print $3}}')
FREE_SECTORS=$((DISK_END - LAST_END - 34))
FREE_GB=$((FREE_SECTORS * 512 / 1073741824))
echo "Last partition ends at sector $LAST_END, disk ends at $DISK_END"
echo "Free space: ~${{FREE_GB}}GB ($FREE_SECTORS sectors)"

if [ "$FREE_GB" -lt 20 ]; then
  echo "ERROR: Less than 20GB free space available (${{FREE_GB}}GB)."
  echo "ERROR: Shrink existing partitions from within the original OS first."
  exit 1
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
cat > /mnt/etc/nixos/configuration.nix << 'NIXCONF'
{{ config, pkgs, ... }}:
{{
  imports = [ ./hardware-configuration.nix ];
  networking.hostName = "{hostname}";
  boot.loader.systemd-boot.enable = true;
  boot.loader.efi.canTouchEfiVariables = true;
  # Dual-boot: sync hardware clock with Windows (which uses localtime)
  time.hardwareClockInLocalTime = true;
  services.openssh.enable = true;
  services.earlyoom = {{ enable = true; freeMemThreshold = 5; freeSwapThreshold = 5; }};
  services.fstrim.enable = true;
  services.smartd = {{ enable = true; autodetect = true; }};
  services.btrfs.autoScrub = {{ enable = true; interval = "monthly"; fileSystems = [ "/" ]; }};
  zramSwap = {{ enable = true; algorithm = "zstd"; }};
  users.users.{hostname} = {{
    isNormalUser = true;
    extraGroups = [ "wheel" "video" "networkmanager" ];
    initialPassword = "changeme";
  }};
  environment.systemPackages = with pkgs; [ vim git curl wget htop btrfs-progs ];
  system.stateVersion = "24.11";
}}
NIXCONF

# Step 6: Create swap file
echo "STAGE: Configuring swap..."
fallocate -l 8G /mnt/swap/swapfile
chmod 600 /mnt/swap/swapfile
mkswap /mnt/swap/swapfile

# Step 7: Install
echo "STAGE: Installing NixOS..."
echo "This may take several minutes as packages are downloaded..."
nixos-install --no-root-passwd 2>&1 | tail -5

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
"#, disk = msg.disk, hostname = hostname)
        }

        "single" | "" => {
            // Full disk wipe → direct partition → nixos-install
            // Uses sgdisk + mkfs directly (no disko download needed on live ISO)
            format!(r#"set -eo pipefail
echo "=== Symthaea Sovereign Birth: Single Disk ==="

# Step 1: Wipe and partition
echo "STAGE: Partitioning disk {disk}..."
# Unmount anything on this disk first
umount -R /mnt 2>/dev/null || true
swapoff {disk}* 2>/dev/null || true
# Wipe all signatures (filesystem, partition table, raid)
wipefs -af {disk} 2>/dev/null || true
sgdisk --zap-all {disk}
sgdisk -n 1:0:+512M -t 1:EF00 -c 1:boot {disk}
sgdisk -n 2:0:0 -t 2:8300 -c 2:nixos {disk}
partprobe {disk} 2>/dev/null || true
blockdev --rereadpt {disk} 2>/dev/null || true
udevadm settle 2>/dev/null || true
sleep 3

# Detect partition names (nvme uses p1/p2, sata uses 1/2)
if [ -b "{disk}p1" ]; then
  BOOT="{disk}p1"
  ROOT="{disk}p2"
else
  BOOT="{disk}1"
  ROOT="{disk}2"
fi
echo "  Boot: $BOOT"
echo "  Root: $ROOT"

# Step 2: Format with btrfs (snapshots, compression, rollback)
echo "STAGE: Formatting with btrfs..."
wipefs -af "$BOOT" 2>/dev/null || true
wipefs -af "$ROOT" 2>/dev/null || true
mkfs.vfat -F 32 "$BOOT"
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
nixos-generate-config --root /mnt

cat > /mnt/etc/nixos/configuration.nix << 'NIXCONF'
{{ config, pkgs, ... }}:
{{
  imports = [ ./hardware-configuration.nix ];
  networking.hostName = "{hostname}";
  boot.loader.systemd-boot.enable = true;
  boot.loader.efi.canTouchEfiVariables = true;

  # Hardening (Symthaea defaults)
  services.openssh.enable = true;
  services.earlyoom = {{ enable = true; freeMemThreshold = 5; freeSwapThreshold = 5; }};
  services.fstrim.enable = true;
  services.smartd = {{ enable = true; autodetect = true; }};
  services.btrfs.autoScrub = {{ enable = true; interval = "monthly"; fileSystems = [ "/" ]; }};
  zramSwap = {{ enable = true; algorithm = "zstd"; }};
  boot.kernel.sysctl."vm.swappiness" = 60;

  # User
  users.users.{hostname} = {{
    isNormalUser = true;
    extraGroups = [ "wheel" "video" "networkmanager" ];
    initialPassword = "changeme";
  }};

  environment.systemPackages = with pkgs; [ vim git curl wget htop btrfs-progs ];
  system.stateVersion = "24.11";
}}
NIXCONF

# Step 5: Create swap file
echo "STAGE: Configuring swap..."
fallocate -l 16G /mnt/swap/swapfile
chmod 600 /mnt/swap/swapfile
mkswap /mnt/swap/swapfile

# Step 6: Install
echo "STAGE: Installing NixOS..."
echo "This may take several minutes as packages are downloaded..."
nixos-install --no-root-passwd 2>&1 | tail -5

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
"#,
                disk = msg.disk,
                hostname = hostname,
            )
        }

        "single-luks" => {
            // Full disk wipe → LUKS2 encryption → btrfs → nixos-install
            // Passphrase is passed via the 'command' field (repurposed)
            let passphrase = if msg.command.is_empty() { "changeme" } else { &msg.command };
            format!(r#"set -eo pipefail
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
echo -n "{passphrase}" | cryptsetup luksFormat --type luks2 --label cryptroot \
  --pbkdf argon2id --iter-time 3000 "$CRYPT_PART" -
echo -n "{passphrase}" | cryptsetup open "$CRYPT_PART" cryptroot -
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

cat > /mnt/etc/nixos/configuration.nix << NIXCONF
{{ config, pkgs, ... }}:
{{
  imports = [ ./hardware-configuration.nix ];
  networking.hostName = "{hostname}";
  boot.loader.systemd-boot.enable = true;
  boot.loader.efi.canTouchEfiVariables = true;

  # LUKS encryption
  boot.initrd.luks.devices."cryptroot" = {{
    device = "/dev/disk/by-uuid/$CRYPT_UUID";
    allowDiscards = true;
  }};

  # Hardening (Symthaea defaults)
  services.openssh.enable = true;
  services.earlyoom = {{ enable = true; freeMemThreshold = 5; freeSwapThreshold = 5; }};
  services.fstrim.enable = true;
  services.smartd = {{ enable = true; autodetect = true; }};
  services.btrfs.autoScrub = {{ enable = true; interval = "monthly"; fileSystems = [ "/" ]; }};
  zramSwap = {{ enable = true; algorithm = "zstd"; }};
  boot.kernel.sysctl."vm.swappiness" = 60;

  # User
  users.users.{hostname} = {{
    isNormalUser = true;
    extraGroups = [ "wheel" "video" "networkmanager" ];
    initialPassword = "changeme";
  }};

  environment.systemPackages = with pkgs; [ vim git curl wget htop btrfs-progs cryptsetup ];
  system.stateVersion = "24.11";
}}
NIXCONF

# Step 7: Create swap file
echo "STAGE: Configuring swap..."
fallocate -l 16G /mnt/swap/swapfile
chmod 600 /mnt/swap/swapfile
mkswap /mnt/swap/swapfile

# Step 8: Install
echo "STAGE: Installing NixOS..."
echo "This may take several minutes as packages are downloaded..."
nixos-install --no-root-passwd 2>&1 | tail -5

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
"#,
                disk = msg.disk,
                hostname = hostname,
                passphrase = passphrase,
            )
        }

        "dual" => {
            // Dual-disk: fast drive for data (btrfs), standard for OS (ext4)
            // Direct partitioning — no disko download needed
            format!(r#"set -eo pipefail
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

cat > /mnt/etc/nixos/configuration.nix << 'NIXCONF'
{{ config, pkgs, ... }}:
{{
  imports = [ ./hardware-configuration.nix ];
  networking.hostName = "{hostname}";
  boot.loader.systemd-boot.enable = true;
  boot.loader.efi.canTouchEfiVariables = true;

  # Hardening
  services.openssh.enable = true;
  services.earlyoom = {{ enable = true; freeMemThreshold = 5; freeSwapThreshold = 5; }};
  services.fstrim.enable = true;
  services.smartd = {{ enable = true; autodetect = true; }};
  services.btrfs.autoScrub = {{ enable = true; interval = "monthly"; fileSystems = [ "/home" ]; }};
  zramSwap = {{ enable = true; algorithm = "zstd"; }};
  boot.kernel.sysctl."vm.swappiness" = 60;

  # User
  users.users.{hostname} = {{
    isNormalUser = true;
    extraGroups = [ "wheel" "video" "networkmanager" ];
    initialPassword = "changeme";
  }};

  environment.systemPackages = with pkgs; [ vim git curl wget htop btrfs-progs ];
  system.stateVersion = "24.11";
}}
NIXCONF

# Step 7: Create swap file on fast drive
echo "STAGE: Configuring swap..."
fallocate -l 64G /mnt/swap/swapfile
chmod 600 /mnt/swap/swapfile
mkswap /mnt/swap/swapfile

# Step 8: Install
echo "STAGE: Installing NixOS..."
echo "This may take several minutes..."
nixos-install --no-root-passwd 2>&1 | tail -5

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
                fast = msg.fast_disk,
                standard = msg.standard_disk,
                hostname = hostname,
            )
        }

        "raid1-btrfs" => {
            // btrfs RAID1 across two disks (mirrored data + metadata)
            format!(r#"set -eo pipefail
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

cat > /mnt/etc/nixos/configuration.nix << 'NIXCONF'
{{ config, pkgs, ... }}:
{{
  imports = [ ./hardware-configuration.nix ];
  networking.hostName = "{hostname}";
  boot.loader.systemd-boot.enable = true;
  boot.loader.efi.canTouchEfiVariables = true;
  boot.initrd.supportedFilesystems = [ "btrfs" ];
  services.openssh.enable = true;
  services.earlyoom = {{ enable = true; freeMemThreshold = 5; freeSwapThreshold = 5; }};
  services.fstrim.enable = true;
  services.btrfs.autoScrub = {{ enable = true; interval = "monthly"; fileSystems = [ "/" ]; }};
  zramSwap = {{ enable = true; algorithm = "zstd"; }};
  users.users.{hostname} = {{
    isNormalUser = true;
    extraGroups = [ "wheel" "video" "networkmanager" ];
    initialPassword = "changeme";
  }};
  environment.systemPackages = with pkgs; [ vim git curl wget htop btrfs-progs ];
  system.stateVersion = "24.11";
}}
NIXCONF

echo "STAGE: Configuring swap..."
fallocate -l 16G /mnt/swap/swapfile
chmod 600 /mnt/swap/swapfile
mkswap /mnt/swap/swapfile

echo "STAGE: Installing NixOS..."
echo "This may take several minutes..."
nixos-install --no-root-passwd 2>&1 | tail -5

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
                fast_disk = msg.fast_disk,
                standard_disk = msg.standard_disk,
                hostname = hostname,
            )
        }

        "raid1-mdadm" => {
            // mdadm RAID1 mirror with btrfs on top
            format!(r#"set -eo pipefail
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

cat > /mnt/etc/nixos/configuration.nix << 'NIXCONF'
{{ config, pkgs, ... }}:
{{
  imports = [ ./hardware-configuration.nix ];
  networking.hostName = "{hostname}";
  boot.loader.systemd-boot.enable = true;
  boot.loader.efi.canTouchEfiVariables = true;
  boot.swraid = {{
    enable = true;
    mdadmConf = "MAILADDR root";
  }};
  services.openssh.enable = true;
  services.earlyoom = {{ enable = true; freeMemThreshold = 5; freeSwapThreshold = 5; }};
  services.fstrim.enable = true;
  services.btrfs.autoScrub = {{ enable = true; interval = "monthly"; fileSystems = [ "/" ]; }};
  zramSwap = {{ enable = true; algorithm = "zstd"; }};
  users.users.{hostname} = {{
    isNormalUser = true;
    extraGroups = [ "wheel" "video" "networkmanager" ];
    initialPassword = "changeme";
  }};
  environment.systemPackages = with pkgs; [ vim git curl wget htop btrfs-progs mdadm ];
  system.stateVersion = "24.11";
}}
NIXCONF

echo "STAGE: Configuring swap..."
fallocate -l 16G /mnt/swap/swapfile
chmod 600 /mnt/swap/swapfile
mkswap /mnt/swap/swapfile

echo "STAGE: Installing NixOS..."
nixos-install --no-root-passwd 2>&1 | tail -5

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
                fast_disk = msg.fast_disk,
                standard_disk = msg.standard_disk,
                hostname = hostname,
            )
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
    fn connected() -> Self {
        Self {
            msg_type: "connected".into(),
            data: None,
            stream: None,
            code: None,
            message: Some("SSH connection established".into()),
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
            let name = dev.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let size = dev.get("size").and_then(|v| v.as_str()).unwrap_or("").to_string();
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
) {
    // Upgrade to WebSocket
    let ws_stream = match accept_async(stream).await {
        Ok(ws) => ws,
        Err(e) => {
            eprintln!("[{}] WebSocket upgrade failed: {}", peer_addr, e);
            return;
        }
    };

    let (mut ws_tx, mut ws_rx) = ws_stream.split();
    let mut ssh_client: Option<Client> = None;

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

                eprintln!(
                    "[{}] Connecting to {}@{}:{}",
                    peer_addr, client_msg.username, client_msg.host, client_msg.port
                );

                let auth = AuthMethod::with_password(&client_msg.password);

                match Client::connect(
                    (client_msg.host.as_str(), client_msg.port),
                    &client_msg.username,
                    auth,
                    ServerCheckMethod::NoCheck,
                )
                .await
                {
                    Ok(client) => {
                        eprintln!("[{}] SSH connected", peer_addr);
                        ssh_client = Some(client);
                        let _ = ws_tx
                            .send(Message::Text(RelayMessage::connected().to_json()))
                            .await;
                    }
                    Err(e) => {
                        eprintln!("[{}] SSH connection failed: {}", peer_addr, e);
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error(&format!("SSH connection failed: {}", e))
                                    .to_json(),
                            ))
                            .await;
                        tracker.lock().await.release(&peer_addr);
                    }
                }
            }

            "exec" => {
                let client = match ssh_client.as_ref() {
                    Some(c) => c,
                    None => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error("Not connected. Send 'connect' first.")
                                    .to_json(),
                            ))
                            .await;
                        continue;
                    }
                };

                eprintln!("[{}] Executing: {}", peer_addr, &client_msg.command);

                // Send initial connecting stage
                let _ = ws_tx
                    .send(Message::Text(
                        RelayMessage::progress(&NixosAnywhereStage::Connecting).to_json(),
                    ))
                    .await;

                match client.execute(&client_msg.command).await {
                    Ok(result) => {
                        // Process stdout line by line for stage detection
                        let combined = format!("{}{}", result.stdout, result.stderr);
                        for line in combined.lines() {
                            if !line.trim().is_empty() {
                                // Send output
                                let _ = ws_tx
                                    .send(Message::Text(
                                        RelayMessage::output(line, "stdout").to_json(),
                                    ))
                                    .await;

                                // Check for stage transitions
                                if let Some(stage) = parse_stage(line) {
                                    let _ = ws_tx
                                        .send(Message::Text(
                                            RelayMessage::progress(&stage).to_json(),
                                        ))
                                        .await;
                                }
                            }
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
                                RelayMessage::error(&format!("Command execution failed: {}", e))
                                    .to_json(),
                            ))
                            .await;
                    }
                }
            }

            "discover_disks" => {
                let client = match ssh_client.as_ref() {
                    Some(c) => c,
                    None => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error("Not connected. Send 'connect' first.")
                                    .to_json(),
                            ))
                            .await;
                        continue;
                    }
                };

                eprintln!("[{}] Discovering disks...", peer_addr);
                match client
                    .execute("lsblk --json -o NAME,SIZE,MODEL,TYPE,TRAN,RM -b")
                    .await
                {
                    Ok(result) if result.exit_status == 0 => {
                        let disks = parse_lsblk(&result.stdout);
                        let disks_json =
                            serde_json::to_string(&disks).unwrap_or_else(|_| "[]".into());
                        eprintln!(
                            "[{}] Found {} disks",
                            peer_addr,
                            disks.len()
                        );
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
                // Fully automated install — generates and executes the entire
                // partition → format → install → configure sequence.
                // The user only clicked "Deploy" in the browser.
                let client = match ssh_client.as_ref() {
                    Some(c) => c,
                    None => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error("Not connected. Send 'connect' first.")
                                    .to_json(),
                            ))
                            .await;
                        continue;
                    }
                };

                let mut script = generate_install_script(&client_msg);

                // Always: pre-install disk snapshot (instant, non-destructive)
                let snapshot = disk_snapshot(if client_msg.disk.is_empty() { "/dev/sda" } else { &client_msg.disk });
                script = format!("{}\n{}", snapshot, script);

                // Always: patch configuration.nix with user's DE/GPU/locale choices
                let patch = system_config_patch(&client_msg);
                if !patch.is_empty() {
                    // Insert after "STAGE: Generating configuration..." and before "STAGE: Configuring swap..."
                    if let Some(pos) = script.find("STAGE: Configuring swap") {
                        script.insert_str(pos, &patch);
                    } else if let Some(pos) = script.find("STAGE: Installing") {
                        script.insert_str(pos, &patch);
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
                eprintln!(
                    "[{}] Starting automated {} install on {}",
                    peer_addr,
                    if client_msg.layout.is_empty() { "single" } else { &client_msg.layout },
                    if client_msg.disk.is_empty() { "default" } else { &client_msg.disk }
                );

                let _ = ws_tx
                    .send(Message::Text(
                        RelayMessage::progress(&NixosAnywhereStage::Connecting).to_json(),
                    ))
                    .await;

                // Write the install script to the target and execute it
                let setup_cmd = format!(
                    "cat > /tmp/symthaea-install.sh << 'SCRIPTEOF'\n{}\nSCRIPTEOF\nchmod +x /tmp/symthaea-install.sh",
                    script
                );

                // Upload script
                match client.execute(&setup_cmd).await {
                    Ok(r) if r.exit_status == 0 => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::output("Install script uploaded.", "stdout").to_json(),
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

                // Execute the install script in background, tail the log for streaming
                // This works around async-ssh2-tokio's blocking execute():
                // The script runs with output redirected to a log file,
                // while we poll the log file for new lines.
                let _ = client
                    .execute("bash /tmp/symthaea-install.sh > /tmp/symthaea-install.log 2>&1 &")
                    .await;
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

                    let tail_result = client
                        .execute(&format!(
                            "wc -l < /tmp/symthaea-install.log 2>/dev/null && tail -n +{} /tmp/symthaea-install.log 2>/dev/null",
                            last_lines + 1
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
                                            let stage = if stage_text.contains("Prepar") || stage_text.contains("environment") {
                                                NixosAnywhereStage::Connecting
                                            } else if stage_text.contains("Partition") {
                                                NixosAnywhereStage::Partitioning
                                            } else if stage_text.contains("Format") || stage_text.contains("btrfs") || stage_text.contains("subvol") {
                                                NixosAnywhereStage::Partitioning
                                            } else if stage_text.contains("Mount") {
                                                NixosAnywhereStage::Partitioning
                                            } else if stage_text.contains("Generat") || stage_text.contains("config") {
                                                NixosAnywhereStage::Configuring
                                            } else if stage_text.contains("Install") {
                                                NixosAnywhereStage::Installing
                                            } else if stage_text.contains("swap") || stage_text.contains("Verif") {
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
                    if let Ok(check) = client.execute("pgrep -f symthaea-install.sh").await {
                        if check.exit_status != 0 && last_lines > 0 {
                            // Script finished but no COMPLETE marker — check exit code
                            if let Ok(exit_check) = client.execute("tail -1 /tmp/symthaea-install.log").await {
                                let _ = ws_tx
                                    .send(Message::Text(
                                        RelayMessage::output(&exit_check.stdout, "stdout").to_json(),
                                    ))
                                    .await;
                            }
                            break;
                        }
                    }
                }

                let exit_code = if complete { 0 } else { 1 };
                let _ = ws_tx
                    .send(Message::Text(
                        RelayMessage::exit(exit_code).to_json(),
                    ))
                    .await;

                // LEGACY: The old blocking path (kept for reference)
                // This is what we replaced with the log-polling approach above.
                if false {
                match client.execute("bash /tmp/symthaea-install.sh 2>&1").await {
                    Ok(result) => {
                        for line in result.stdout.lines().chain(result.stderr.lines()) {
                            if line.trim().is_empty() {
                                continue;
                            }

                            // Parse STAGE: markers for progress
                            if line.starts_with("STAGE: ") {
                                let stage_text = &line[7..];
                                let stage = if stage_text.contains("Detect") || stage_text.contains("free space") {
                                    NixosAnywhereStage::Connecting
                                } else if stage_text.contains("Partition") {
                                    NixosAnywhereStage::Partitioning
                                } else if stage_text.contains("Format") || stage_text.contains("btrfs") {
                                    NixosAnywhereStage::Partitioning
                                } else if stage_text.contains("Mount") {
                                    NixosAnywhereStage::Partitioning
                                } else if stage_text.contains("Install") {
                                    NixosAnywhereStage::Installing
                                } else if stage_text.contains("Configur") || stage_text.contains("swap") {
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
                                RelayMessage::error(&format!("Install failed: {}", e)).to_json(),
                            ))
                            .await;
                    }
                }
                } // end if false (legacy blocking path)
            }

            // ── Comprehensive hardware probe ──
            "probe_hardware" => {
                let client = match ssh_client.as_ref() {
                    Some(c) => c,
                    None => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error("Not connected. Send 'connect' first.")
                                    .to_json(),
                            ))
                            .await;
                        continue;
                    }
                };

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

echo '}'
"#;

                match client.execute(probe_script).await {
                    Ok(result) if result.exit_status == 0 => {
                        eprintln!("[{}] Hardware probe complete", peer_addr);
                        let _ = ws_tx
                            .send(Message::Text(
                                serde_json::json!({
                                    "type": "hardware_probe",
                                    "data": result.stdout
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
                let client = match ssh_client.as_ref() {
                    Some(c) => c,
                    None => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error("Not connected. Send 'connect' first.")
                                    .to_json(),
                            ))
                            .await;
                        continue;
                    }
                };

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

                match client.execute(scan_script).await {
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
                                RelayMessage::error(&format!("App scan failed: {}", e))
                                    .to_json(),
                            ))
                            .await;
                    }
                }
            }

            // ── Deep scan: dotfiles, config, personal data for migration + welcome ──
            "deep_scan" => {
                let client = match ssh_client.as_ref() {
                    Some(c) => c,
                    None => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error("Not connected. Send 'connect' first.")
                                    .to_json(),
                            ))
                            .await;
                        continue;
                    }
                };

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

                match client.execute(deep_scan_script).await {
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
                                RelayMessage::error(&format!("Deep scan failed: {}", e))
                                    .to_json(),
                            ))
                            .await;
                    }
                }
            }

            // ── Data preservation before wipe ──
            "preserve_data" => {
                let client = match ssh_client.as_ref() {
                    Some(c) => c,
                    None => {
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::error("Not connected. Send 'connect' first.")
                                    .to_json(),
                            ))
                            .await;
                        continue;
                    }
                };

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

                match client.execute(preserve_script).await {
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

            "disconnect" => {
                eprintln!("[{}] Client disconnecting", peer_addr);
                ssh_client = None;
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
    if ssh_client.is_some() {
        tracker.lock().await.release(&peer_addr);
    }
    eprintln!("[{}] WebSocket disconnected", peer_addr);
}

#[tokio::main]
async fn main() {
    let port: u16 = std::env::args()
        .skip_while(|a| a != "--port")
        .nth(1)
        .and_then(|p| p.parse().ok())
        .unwrap_or(8091);

    let tracker: SharedTracker = Arc::new(Mutex::new(SessionTracker::new(1800))); // 30 min timeout

    let addr = format!("0.0.0.0:{}", port);
    let listener = match TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("ERROR: Cannot bind to {}: {}", addr, e);
            eprintln!("Another service may be using this port. Try: --port <other-port>");
            std::process::exit(1);
        }
    };
    eprintln!("Symthaea SSH Relay listening on ws://{}", addr);
    eprintln!("  Protocol: connect → exec → disconnect");
    eprintln!("  Session timeout: 30 minutes");
    eprintln!("  Rate limit: 1 active session per IP");

    while let Ok((stream, addr)) = listener.accept().await {
        let peer = addr.ip().to_string();
        let tracker = tracker.clone();
        tokio::spawn(handle_connection(stream, peer, tracker));
    }
}
