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
}

fn default_port() -> u16 {
    22
}

/// Generate the automated install script based on layout type.
fn generate_install_script(msg: &ClientMessage) -> String {
    let hostname = if msg.hostname.is_empty() { "guardian" } else { &msg.hostname };

    match msg.layout.as_str() {
        "alongside" => {
            // Alongside Windows: find free space, create partition, install
            format!(r#"set -euo pipefail
echo "=== Symthaea Sovereign Birth: Alongside Windows ==="

# Step 1: Find unallocated space on the disk
echo "STAGE: Detecting free space on {disk}..."
DISK="{disk}"
# Get the end of the last partition
LAST_END=$(sgdisk -p "$DISK" 2>/dev/null | grep '^ ' | tail -1 | awk '{{print $3}}')
DISK_END=$(sgdisk -p "$DISK" 2>/dev/null | grep 'Disk size' | awk '{{print $3}}')
echo "Last partition ends at sector $LAST_END, disk ends at $DISK_END"

# Step 2: Create NixOS partition in the free space
echo "STAGE: Partitioning free space..."
PART_NUM=$(sgdisk -p "$DISK" | grep '^ ' | wc -l)
PART_NUM=$((PART_NUM + 1))
sgdisk -n "$PART_NUM:0:0" -t "$PART_NUM:8300" -c "$PART_NUM:nixos-root" "$DISK"
partprobe "$DISK"
sleep 2
NIXOS_PART="${{DISK}}p$PART_NUM"
[ -b "$NIXOS_PART" ] || NIXOS_PART="${{DISK}}$PART_NUM"
echo "Created partition: $NIXOS_PART"

# Step 3: Format with btrfs + subvolumes
echo "STAGE: Formatting with btrfs..."
mkfs.btrfs -f -L symthaea "$NIXOS_PART"
mount "$NIXOS_PART" /mnt
btrfs subvolume create /mnt/@
btrfs subvolume create /mnt/@home
btrfs subvolume create /mnt/@nix
btrfs subvolume create /mnt/@log
btrfs subvolume create /mnt/@swap
umount /mnt

# Step 4: Mount everything
echo "STAGE: Mounting filesystems..."
mount -o subvol=@,compress=zstd:3,noatime "$NIXOS_PART" /mnt
mkdir -p /mnt/{{home,nix,var/log,swap,boot,etc/nixos}}
mount -o subvol=@home,compress=zstd:3,noatime "$NIXOS_PART" /mnt/home
mount -o subvol=@nix,compress=zstd:3,noatime "$NIXOS_PART" /mnt/nix
mount -o subvol=@log,compress=zstd:3,noatime "$NIXOS_PART" /mnt/var/log
mount -o subvol=@swap,noatime "$NIXOS_PART" /mnt/swap

# Find and mount existing EFI partition
EFI_PART=$(lsblk -nlo NAME,PARTTYPE "$DISK" | grep -i 'c12a7328' | head -1 | awk '{{print "/dev/"$1}}')
[ -n "$EFI_PART" ] && mount "$EFI_PART" /mnt/boot
echo "Mounted EFI at $EFI_PART"

# Step 5: Write NixOS configuration
echo "STAGE: Installing NixOS..."
nixos-generate-config --root /mnt
cat > /mnt/etc/nixos/configuration.nix << 'NIXCONF'
{{ config, pkgs, ... }}:
{{
  imports = [ ./hardware-configuration.nix ];
  networking.hostName = "{hostname}";
  boot.loader.systemd-boot.enable = true;
  boot.loader.efi.canTouchEfiVariables = true;
  time.hardwareClockInLocalTime = true;
  services.earlyoom.enable = true;
  services.fstrim.enable = true;
  services.smartd = {{ enable = true; autodetect = true; }};
  zramSwap = {{ enable = true; algorithm = "zstd"; }};
  users.users.guardian = {{
    isNormalUser = true;
    extraGroups = [ "wheel" "video" "networkmanager" ];
    initialPassword = "changeme";
  }};
  services.openssh.enable = true;
  environment.systemPackages = with pkgs; [ vim git curl wget htop ];
  system.stateVersion = "24.11";
}}
NIXCONF

# Step 6: Install
nixos-install --no-root-passwd
echo "STAGE: Installation complete"

# Step 7: Create swap file
echo "STAGE: Configuring swap..."
truncate -s 0 /mnt/swap/swapfile
chattr +C /mnt/swap/swapfile
btrfs property set /mnt/swap/swapfile compression none
fallocate -l 16G /mnt/swap/swapfile
chmod 600 /mnt/swap/swapfile
mkswap /mnt/swap/swapfile

echo "=== Sovereign Birth Complete ==="
echo "STAGE: FirstBreath"
echo "The machine will draw its first breath on reboot."
echo "Default user: guardian (password: changeme)"
echo "COMPLETE"
"#, disk = msg.disk, hostname = hostname)
        }

        "single" | "" => {
            // Full disk wipe → disko → nixos-install
            format!(r#"set -euo pipefail
echo "=== Symthaea Sovereign Birth: Single Disk ==="

# Step 1: Write configs
echo "STAGE: Writing configuration..."
mkdir -p /tmp/symthaea-install
cat > /tmp/symthaea-install/flake.nix << 'FLAKEEOF'
{flake_content}
FLAKEEOF

cat > /tmp/symthaea-install/disko-config.nix << 'DISKOEOF'
{disko_content}
DISKOEOF

cat > /tmp/symthaea-install/hardware-configuration.nix << 'HWEOF'
{hw_content}
HWEOF

# Step 2: Run disko
echo "STAGE: Partitioning disk {disk}..."
nix run github:nix-community/disko -- --mode destroy /tmp/symthaea-install/disko-config.nix

# Step 3: Copy configs to /mnt
echo "STAGE: Installing NixOS..."
mkdir -p /mnt/etc/nixos
cp /tmp/symthaea-install/*.nix /mnt/etc/nixos/

# Step 4: Install
nixos-install --flake /mnt/etc/nixos#{hostname} --no-root-passwd

echo "STAGE: FirstBreath"
echo "=== Sovereign Birth Complete ==="
echo "COMPLETE"
"#,
                disk = msg.disk,
                hostname = hostname,
                flake_content = msg.flake_nix.replace('\'', "'\"'\"'"),
                disko_content = msg.disko_nix.replace('\'', "'\"'\"'"),
                hw_content = msg.hardware_nix.replace('\'', "'\"'\"'"),
            )
        }

        "dual" => {
            // Dual-disk: fast drive for data, standard for OS
            format!(r#"set -euo pipefail
echo "=== Symthaea Sovereign Birth: Dual NVMe ==="

echo "STAGE: Writing configuration..."
mkdir -p /tmp/symthaea-install
cat > /tmp/symthaea-install/flake.nix << 'FLAKEEOF'
{flake_content}
FLAKEEOF

cat > /tmp/symthaea-install/disko-config.nix << 'DISKOEOF'
{disko_content}
DISKOEOF

cat > /tmp/symthaea-install/hardware-configuration.nix << 'HWEOF'
{hw_content}
HWEOF

echo "STAGE: Partitioning drives..."
echo "  Fast: {fast} (data)"
echo "  Standard: {standard} (OS)"
nix run github:nix-community/disko -- --mode destroy /tmp/symthaea-install/disko-config.nix

echo "STAGE: Installing NixOS..."
mkdir -p /mnt/etc/nixos
cp /tmp/symthaea-install/*.nix /mnt/etc/nixos/
nixos-install --flake /mnt/etc/nixos#{hostname} --no-root-passwd

echo "STAGE: Configuring swap..."
mkdir -p /mnt/swap
truncate -s 0 /mnt/swap/swapfile
chattr +C /mnt/swap/swapfile 2>/dev/null || true
fallocate -l 64G /mnt/swap/swapfile
chmod 600 /mnt/swap/swapfile
mkswap /mnt/swap/swapfile

echo "STAGE: FirstBreath"
echo "=== Sovereign Birth Complete ==="
echo "COMPLETE"
"#,
                fast = msg.fast_disk,
                standard = msg.standard_disk,
                hostname = hostname,
                flake_content = msg.flake_nix.replace('\'', "'\"'\"'"),
                disko_content = msg.disko_nix.replace('\'', "'\"'\"'"),
                hw_content = msg.hardware_nix.replace('\'', "'\"'\"'"),
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

                let script = generate_install_script(&client_msg);
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

                // Execute the install script
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
    let listener = TcpListener::bind(&addr).await.unwrap();
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
