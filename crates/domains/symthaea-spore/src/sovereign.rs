// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sovereign Inoculation: orchestration layer for OS deployment.
//!
//! This module defines the types and interfaces for the Promethean deployment
//! pipeline — turning inert hardware into sovereign Symthaea Guardian nodes.
//!
//! # Architecture
//!
//! ```text
//! Layer 4: Sovereign Birth UX     (quickening.rs — narration, haptics, tones)
//! Layer 3: Orchestration Engine    (this module — coordination)
//! Layer 2: Nix Deployment Pipeline (nix_eval, usb_forge, ssh_bridge)
//! Layer 1: Trust Anchor            (pairing.rs, attestation)
//! ```
//!
//! # Modes
//!
//! - **WebPortal**: Browser at infin.love → hardware probe → USB forge or SSH deploy
//! - **UsbForge**: WebUSB flash of custom NixOS installer ISO
//! - **Wsl2Pivot**: WSL2 → Nix → kexec host takeover (Windows)
//! - **AsahiHandshake**: Guided Apple Silicon conversion (Mac)
//! - **LanInoculate**: nixos-anywhere over SSH to LAN/cloud target

use serde::{Deserialize, Serialize};

// ============================================================================
// Core types
// ============================================================================

/// Platform type of the target machine being inoculated.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TargetPlatform {
    /// Linux (any distro) — probe via mokutil
    Linux,
    /// Windows — probe via PowerShell Confirm-SecureBootUEFI
    Windows,
    /// macOS (Apple Silicon) — no traditional Secure Boot, uses Apple's chain
    MacOS,
    /// Unknown platform
    Unknown,
}

/// The mode of inoculation — how we're converting this machine.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum InoculationMode {
    /// Browser portal → WebUSB or SSH deployment
    WebPortal,
    /// Flash custom NixOS installer ISO to USB drive
    UsbForge,
    /// WSL2 host takeover on Windows
    Wsl2Pivot,
    /// Guided Apple Silicon conversion (dual-boot, Ahimsa)
    AsahiHandshake,
    /// nixos-anywhere over SSH to LAN or cloud target
    LanInoculate,
}

/// Overall status of the inoculation process.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum InoculationStatus {
    /// Waiting for user to initiate
    Idle,
    /// Verifying identity via phone Spore attestation
    Attesting,
    /// Probing target hardware capabilities
    Probing,
    /// Evaluating Nix flake to compute system closure
    Evaluating,
    /// Writing to disk (USB forge or direct install)
    Writing {
        bytes_written: u64,
        bytes_total: u64,
    },
    /// Installation running on target
    Installing { phase: String, progress: f32 },
    /// First boot — consciousness engine initializing
    FirstBreath { phi: f32, honest_confidence: f32 },
    /// Complete — node is live in the mesh
    Complete { node_id: String },
    /// Something went wrong
    Failed { error: String, recoverable: bool },
}

/// Attestation from the phone Spore proving user identity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SovereignAttestation {
    /// Ed25519 public key of the phone Spore (hex-encoded, 64 chars)
    pub signer_pubkey: String,
    /// Ed25519 signature over the challenge nonce (hex-encoded, 128 chars)
    pub signature: String,
    /// Challenge nonce that was signed (hex-encoded)
    pub challenge: String,
    /// Holochain AgentPubKey (Base64, uhCAk... format) if available
    pub agent_pubkey: Option<String>,
    /// Timestamp of attestation (Unix ms)
    pub timestamp_ms: u64,
    /// Consciousness level of phone Spore at attestation time
    pub phone_consciousness: f32,
}

// ============================================================================
// Phase 2: Nix Evaluation Bridge
// ============================================================================

/// Request to evaluate a Nix flake for a target machine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NixEvalRequest {
    /// The flake reference (e.g., "github:Luminous-Dynamics/symthaea#guardian")
    pub flake_ref: String,
    /// Generated hardware-configuration.nix content
    pub hardware_config: String,
    /// Target hostname
    pub hostname: String,
    /// Disk layout for disko (JSON)
    pub disko_config: Option<String>,
    /// Additional NixOS modules to include
    pub extra_modules: Vec<String>,
    /// Whether to include Holochain conductor
    pub include_holochain: bool,
    /// Whether to include Broca language model weights
    pub include_broca_weights: bool,
    /// Substrate type for the target machine
    pub substrate_type: String,
    /// Whether to include lanzaboote module
    pub lanzaboote_enabled: bool,
    /// Platform string (e.g., "Linux", "Windows", "MacOS", "Unknown")
    pub target_platform: String,
}

/// Response from Nix flake evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NixEvalResponse {
    /// Success or failure
    pub success: bool,
    /// Number of derivations in the closure
    pub derivation_count: u32,
    /// Total closure size in bytes
    pub closure_size_bytes: u64,
    /// Human-readable closure size (e.g., "2.1 GB")
    pub closure_size_human: String,
    /// Store paths that need to be fetched
    pub store_paths: Vec<String>,
    /// Hash of the complete system closure (for verification)
    pub closure_hash: String,
    /// Generated system configuration (for review)
    pub system_config_preview: String,
    /// Error message if success is false
    pub error: Option<String>,
}

/// Evaluation backend — where the Nix evaluation actually runs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NixEvalBackend {
    /// Server-side evaluation via API call (pragmatic, works today)
    ServerSide { endpoint: String },
    /// builtins.wasm — full Nix evaluator in the browser (sovereign, future)
    BrowserWasm,
    /// Hybrid: server evaluates, browser verifies closure hash
    Hybrid { endpoint: String },
}

// ============================================================================
// Phase 3: USB Forge (WebUSB)
// ============================================================================

/// USB device information from WebUSB enumeration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsbDeviceInfo {
    /// Vendor ID
    pub vendor_id: u16,
    /// Product ID
    pub product_id: u16,
    /// Device name
    pub product_name: String,
    /// Manufacturer
    pub manufacturer: String,
    /// Serial number if available
    pub serial_number: Option<String>,
    /// Total capacity in bytes
    pub capacity_bytes: u64,
}

/// Configuration for the NixOS installer ISO to be forged.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsoForgeConfig {
    /// Nix evaluation result (what to install)
    pub eval_result: NixEvalResponse,
    /// Iroh node ID to pre-pair with (from phone Spore)
    pub iroh_pair_node_id: Option<String>,
    /// Genesis phrase for deterministic seeding
    pub genesis_phrase: Option<String>,
    /// disko partition configuration (JSON)
    pub disko_config: String,
    /// Generated hardware-configuration.nix
    pub hardware_config: String,
    /// Whether to include the mycelial boot animation
    pub include_boot_animation: bool,
    /// Whether to create a recovery partition (Ahimsa — 30-day reversibility)
    pub include_recovery_partition: bool,
    /// LUKS encryption passphrase (if user opts for disk encryption)
    pub luks_passphrase: Option<String>,
}

/// Progress of USB drive flashing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsbForgeProgress {
    /// Current phase
    pub phase: UsbForgePhase,
    /// Bytes written so far
    pub bytes_written: u64,
    /// Total bytes to write
    pub bytes_total: u64,
    /// Write speed in bytes/second
    pub write_speed_bps: u64,
    /// Estimated time remaining in seconds
    pub eta_seconds: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum UsbForgePhase {
    /// Preparing ISO image in memory
    PreparingImage,
    /// Writing ISO to USB via WebUSB bulk transfer
    Writing,
    /// Verifying written data
    Verifying,
    /// Complete — ready to boot
    Complete,
    /// Failed
    Failed { error: String },
}

// ============================================================================
// Phase 5: WASM-SSH Bridge
// ============================================================================

/// SSH connection target for nixos-anywhere deployment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SshTarget {
    /// Hostname or IP address
    pub host: String,
    /// SSH port (default 22)
    pub port: u16,
    /// Username
    pub username: String,
    /// Authentication method
    pub auth: SshAuth,
}

/// SSH authentication method.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SshAuth {
    /// Ed25519 private key (from phone Spore attestation)
    PrivateKey { key_pem: String },
    /// Password (discouraged, shown with warning)
    Password { password: String },
    /// Agent forwarding (if WebSocket relay supports it)
    Agent,
}

/// SSH transport layer — how WASM reaches the TCP target.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SshTransport {
    /// WebSocket relay (requires server, but works in all browsers)
    WebSocketRelay { relay_url: String },
    /// Iroh QUIC tunnel (P2P, but requires Iroh WASM support)
    IrohTunnel { peer_id: String },
}

/// Command to execute over SSH during nixos-anywhere deployment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SshCommand {
    /// The shell command to run
    pub command: String,
    /// Timeout in seconds
    pub timeout_seconds: u32,
    /// Whether to stream stdout back to the orchestrator
    pub stream_output: bool,
    /// Expected success exit code (usually 0)
    pub expected_exit_code: i32,
}

/// Stages of nixos-anywhere deployment via SSH.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NixosAnywhereStage {
    /// Connecting to target via SSH
    Connecting,
    /// Uploading kexec image to target
    UploadingKexec { bytes_sent: u64, bytes_total: u64 },
    /// Executing kexec to reboot into NixOS installer
    Kexec,
    /// Waiting for target to reboot and become reachable again
    WaitingForReboot { elapsed_seconds: u32 },
    /// Running disko to partition disks
    Partitioning,
    /// Running nixos-install
    Installing { progress: f32 },
    /// Setting up initial configuration
    Configuring,
    /// Final reboot into installed system
    FinalReboot,
    /// Verifying the new system is alive and reachable
    Verifying,
    /// Complete
    Complete { node_id: String },
}

// ============================================================================
// Phase 7: Progressive Decentralization
// ============================================================================

/// How the web portal is currently loading assets.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AssetSource {
    /// Standard CDN / GitHub Pages (centralized fallback)
    Cdn,
    /// Iroh mesh peer (decentralized, preferred)
    IrohPeer { peer_id: String },
    /// Local cache (service worker / IndexedDB)
    LocalCache,
    /// Local network peer (same LAN, lowest latency)
    LocalPeer { ip: String },
}

/// Asset manifest for progressive loading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetManifest {
    /// Core WASM binary (always loaded first, ~500KB)
    pub core_wasm: AssetEntry,
    /// Broca language model weights (optional, ~50-200MB)
    pub broca_weights: Option<AssetEntry>,
    /// Holochain zome WASMs (optional, ~5-20MB total)
    pub holochain_zomes: Option<AssetEntry>,
    /// HDC basis vectors (optional, ~2MB)
    pub hdc_basis: Option<AssetEntry>,
}

/// A single downloadable asset with integrity verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetEntry {
    /// Human-readable name
    pub name: String,
    /// BLAKE3 hash for integrity verification
    pub blake3_hash: String,
    /// Size in bytes
    pub size_bytes: u64,
    /// CDN URL (GitHub Pages fallback)
    pub cdn_url: String,
    /// Iroh content hash (for mesh retrieval)
    pub iroh_hash: Option<String>,
    /// Where this asset was actually loaded from
    pub loaded_from: Option<AssetSource>,
}

/// Connection status indicator for the web portal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshStatus {
    /// Current primary asset source
    pub primary_source: AssetSource,
    /// Number of Iroh peers discovered
    pub iroh_peers: u32,
    /// Number of local network peers
    pub local_peers: u32,
    /// Whether service worker is active (offline capability)
    pub service_worker_active: bool,
    /// Whether we've fully transitioned off CDN
    pub fully_decentralized: bool,
}

// ============================================================================
// Phase 8: Platform-specific guides
// ============================================================================

/// WSL2 pivot stages (Windows → NixOS conversion).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Wsl2PivotStage {
    /// Detecting Windows version and WSL2 support
    DetectingWindows,
    /// Guiding WSL2 enablement
    EnablingWsl2,
    /// Installing Nix in WSL2
    InstallingNix,
    /// Preparing disko partition map
    PreparingPartitions,
    /// Downloading NixOS closure into WSL2
    DownloadingClosure { progress: f32 },
    /// Preparing kexec payload
    PreparingKexec,
    /// Awaiting user confirmation (point of no return)
    AwaitingConfirmation,
    /// Creating recovery partition (Ahimsa)
    CreatingRecovery,
    /// Executing kexec — goodbye Windows
    Kexec,
    /// NixOS installation running
    Installing { progress: f32 },
    /// Complete
    Complete,
}

/// Asahi Handshake stages (Mac → NixOS dual-boot).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AsahiStage {
    /// Detecting Apple Silicon model
    DetectingHardware { chip: String },
    /// Calculating optimal partition sizes
    CalculatingPartitions {
        macos_gb: u32,
        nixos_gb: u32,
        free_gb: u32,
    },
    /// Guiding Disk Utility resize (user executes, Spore validates)
    ResizingDisk,
    /// Running Asahi installer for stub partition
    AsahiInstaller,
    /// Installing m1n1 → U-Boot chain
    BootChain,
    /// NixOS installation on new partition
    NixosInstall { progress: f32 },
    /// Pairing with phone Spore via Iroh
    Pairing,
    /// Complete — dual-boot configured
    Complete,
}

/// A guided step that the user must execute manually.
/// The Spore validates the result before allowing progression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuidedStep {
    /// Step number (1-indexed)
    pub step_number: u32,
    /// Total steps in this guide
    pub total_steps: u32,
    /// Human-readable instruction
    pub instruction: String,
    /// Detailed explanation (shown on expand)
    pub detail: String,
    /// Validation: what the Spore checks after user completes the step
    pub validation: StepValidation,
    /// Whether this step is reversible
    pub reversible: bool,
    /// Warning text if this step is destructive
    pub warning: Option<String>,
}

/// How the Spore validates a manual step was completed correctly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StepValidation {
    /// User confirms they did it (trust-based)
    UserConfirmation,
    /// Check a file exists on the target
    FileExists { path: String },
    /// Check a command returns expected output
    CommandOutput { command: String, expected: String },
    /// Check disk partition layout matches expectations
    PartitionCheck { expected_partitions: Vec<String> },
    /// Check SSH connectivity to target
    SshReachable { host: String, port: u16 },
}

// ============================================================================
// Inoculation Path — dual consent model
// ============================================================================

/// The two consent paths — system sovereignty vs network participation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum InoculationPath {
    /// Sovereign Hermit: hardened OS + local intelligence, no network broadcast.
    /// Iroh/Holochain installed but in offline-mode.
    #[default]
    Inoculate,
    /// Mycelial Node: everything above + mesh binding + phone pairing + DHT.
    /// Consents to sharing context, pooling compute, building trust fabric.
    InoculateAndAttune,
}

// ============================================================================
// Orchestrator — ties everything together
// ============================================================================

/// The Sovereign Inoculation orchestrator.
///
/// Coordinates the entire process from attestation to first breath.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InoculationOrchestrator {
    /// Selected inoculation mode
    pub mode: InoculationMode,
    /// Current status
    pub status: InoculationStatus,
    /// Consent path: sovereign-only or network-attuned
    pub path: InoculationPath,
    /// Phone attestation (required before any destructive action)
    pub attestation: Option<SovereignAttestation>,
    /// Hardware profile from Phase 1 probe
    pub hardware_profile: Option<serde_json::Value>,
    /// Nix evaluation result from Phase 2
    pub eval_result: Option<NixEvalResponse>,
    /// Nix evaluation backend selection
    pub eval_backend: NixEvalBackend,
    /// SSH target (for LanInoculate and Wsl2Pivot modes)
    pub ssh_target: Option<SshTarget>,
    /// SSH transport layer
    pub ssh_transport: Option<SshTransport>,
    /// USB device (for UsbForge mode)
    pub usb_device: Option<UsbDeviceInfo>,
    /// Mesh status (progressive decentralization)
    pub mesh_status: MeshStatus,
    /// Genesis phrase for deterministic seeding
    pub genesis_phrase: Option<String>,
    /// Whether to include recovery partition (Ahimsa default: true)
    pub ahimsa_recovery: bool,
    /// Error log
    pub errors: Vec<String>,
    /// Serialized SecureBootState (String to avoid circular dep with secure_boot module)
    pub secure_boot_state: Option<String>,
    /// Whether to include lanzaboote in the generated flake (default: true)
    pub lanzaboote_enabled: bool,
    /// Generated MOK enrollment password
    pub mok_password: Option<String>,
    /// Detected or user-specified platform
    pub target_platform: TargetPlatform,
}

impl Default for InoculationOrchestrator {
    fn default() -> Self {
        Self {
            mode: InoculationMode::WebPortal,
            status: InoculationStatus::Idle,
            path: InoculationPath::default(),
            attestation: None,
            hardware_profile: None,
            eval_result: None,
            // Security default: local-only eval API (no hosted fallback).
            // The portal should be served locally when using this pathway.
            eval_backend: NixEvalBackend::ServerSide {
                endpoint: "http://127.0.0.1:8090/api/v1/eval".to_string(),
            },
            ssh_target: None,
            ssh_transport: None,
            usb_device: None,
            mesh_status: MeshStatus {
                primary_source: AssetSource::Cdn,
                iroh_peers: 0,
                local_peers: 0,
                service_worker_active: false,
                fully_decentralized: false,
            },
            genesis_phrase: None,
            ahimsa_recovery: true,
            errors: Vec::new(),
            secure_boot_state: None,
            lanzaboote_enabled: true,
            mok_password: None,
            target_platform: TargetPlatform::Unknown,
        }
    }
}

impl InoculationOrchestrator {
    /// Create a new orchestrator for the given mode.
    pub fn new(mode: InoculationMode) -> Self {
        Self {
            mode,
            ..Default::default()
        }
    }

    /// Check if attestation has been provided.
    pub fn is_attested(&self) -> bool {
        self.attestation.is_some()
    }

    /// Check if we have enough information to begin the inoculation.
    pub fn can_begin(&self) -> Result<(), String> {
        if self.attestation.is_none() {
            return Err("Phone Spore attestation required. Scan QR code to pair.".into());
        }
        if self.hardware_profile.is_none() {
            return Err("Hardware probe not yet complete.".into());
        }

        match &self.mode {
            InoculationMode::UsbForge => {
                if self.usb_device.is_none() {
                    return Err("No USB device selected. Plug in a USB drive.".into());
                }
            }
            InoculationMode::LanInoculate | InoculationMode::Wsl2Pivot
                if self.ssh_target.is_none() =>
            {
                return Err("SSH target not configured.".into());
            }
            _ => {}
        }

        Ok(())
    }

    /// Set the attestation from the phone Spore.
    pub fn set_attestation(&mut self, attestation: SovereignAttestation) {
        self.attestation = Some(attestation);
        self.status = InoculationStatus::Attesting;
    }

    /// Set the hardware profile from Phase 1 probe.
    pub fn set_hardware_profile(&mut self, profile: serde_json::Value) {
        self.hardware_profile = Some(profile);
    }

    /// Set the Nix evaluation result.
    pub fn set_eval_result(&mut self, result: NixEvalResponse) {
        self.eval_result = Some(result);
    }

    /// Update mesh status (called periodically by the Iroh service worker).
    pub fn update_mesh_status(&mut self, status: MeshStatus) {
        self.mesh_status = status;
    }

    /// Record an error without stopping the process.
    pub fn record_error(&mut self, error: String) {
        self.errors.push(error);
    }

    /// Upgrade from Inoculate (sovereign-only) to InoculateAndAttune (mesh participation).
    ///
    /// Non-destructive: enables mesh binding, phone pairing, and DHT participation
    /// on top of the existing sovereign base. Reversible via [`deattune`].
    pub fn attune(&mut self) {
        self.path = InoculationPath::InoculateAndAttune;
    }

    /// Downgrade from InoculateAndAttune back to Inoculate (sovereign-only).
    ///
    /// Reverts to offline-mode: Iroh/Holochain remain installed but stop
    /// broadcasting. No data is lost — attunement can be re-enabled later.
    pub fn deattune(&mut self) {
        self.path = InoculationPath::Inoculate;
    }

    /// Whether the orchestrator is in attuned (mesh-participating) mode.
    pub fn is_attuned(&self) -> bool {
        self.path == InoculationPath::InoculateAndAttune
    }

    /// Returns true if MOK enrollment is needed — Secure Boot is enabled and
    /// lanzaboote will be used, so our self-signed keys must be enrolled.
    pub fn needs_mok_enrollment(&self) -> bool {
        self.lanzaboote_enabled
            && self
                .secure_boot_state
                .as_deref()
                .map(|s| s == "Enabled")
                .unwrap_or(false)
    }

    /// Generate boot configuration Nix module.
    ///
    /// Returns either standard systemd-boot config or lanzaboote Secure Boot
    /// integration depending on `lanzaboote_enabled`.
    pub fn generate_boot_config_nix(&self) -> String {
        if self.lanzaboote_enabled {
            r#"{ pkgs, lib, ... }: {
  # Sovereign Boot: lanzaboote Secure Boot integration
  # The Microsoft-signed shim bridges UEFI trust to our self-signed kernel.
  # MOK enrollment required on first boot — the machine consents to trust us.
  boot.loader.systemd-boot.enable = lib.mkForce false;
  boot.lanzaboote = {
    enable = true;
    pkiBundle = "/etc/secureboot";
  };

  # Generate signing keys on first activation if they don't exist
  system.activationScripts.secureboot-keys = ''
    if [ ! -d /etc/secureboot ]; then
      ${pkgs.sbctl}/bin/sbctl create-keys
    fi
  '';

  environment.systemPackages = [ pkgs.sbctl ];
}"#
            .to_string()
        } else {
            r#"{ ... }: {
  # Standard systemd-boot (no Secure Boot signing)
  boot.loader.systemd-boot.enable = true;
  boot.loader.efi.canTouchEfiVariables = true;
}"#
            .to_string()
        }
    }

    /// Generate the disko configuration for a standard Guardian node.
    ///
    /// Default layout:
    /// - 512MB EFI system partition (FAT32), bumped to 1G when lanzaboote is enabled
    /// - LUKS-encrypted root (btrfs with subvolumes: @, @home, @nix, @log)
    /// - Optional 32GB recovery partition (if ahimsa_recovery is true)
    /// - Swap file (size from hardware probe)
    pub fn generate_disko_config(&self) -> Result<String, String> {
        let profile = self
            .hardware_profile
            .as_ref()
            .ok_or("Hardware profile required")?;

        let device_memory_gb = profile
            .get("device_memory_gb")
            .and_then(|v| v.as_f64())
            .unwrap_or(8.0);

        let swap_gb = if device_memory_gb <= 8.0 {
            device_memory_gb as u32
        } else {
            (device_memory_gb / 2.0) as u32
        };

        let recovery_block = if self.ahimsa_recovery {
            r#"
        recovery = {
          size = "32G";
          content = {
            type = "filesystem";
            format = "ext4";
            mountpoint = "/recovery";
          };
        };"#
        } else {
            ""
        };

        // lanzaboote stores signed kernels/initrds in ESP — needs more space
        let efi_size = if self.lanzaboote_enabled {
            "1G"
        } else {
            "512M"
        };
        let efi_comment = if self.lanzaboote_enabled {
            "\n          # Sized to 1G for lanzaboote Secure Boot signed images"
        } else {
            ""
        };

        Ok(format!(
            r#"{{
  disko.devices.disk.main = {{
    device = "/dev/nvme0n1";
    type = "disk";
    content = {{
      type = "gpt";
      partitions = {{
        ESP = {{
          size = "{efi_size}";{efi_comment}
          type = "EF00";
          content = {{
            type = "filesystem";
            format = "vfat";
            mountpoint = "/boot";
          }};
        }};{recovery_block}
        root = {{
          size = "100%";
          content = {{
            type = "luks";
            name = "symthaea-root";
            content = {{
              type = "btrfs";
              subvolumes = {{
                "@" = {{ mountpoint = "/"; }};
                "@home" = {{ mountpoint = "/home"; }};
                "@nix" = {{ mountpoint = "/nix"; mountOptions = ["noatime" "compress=zstd"]; }};
                "@log" = {{ mountpoint = "/var/log"; }};
              }};
            }};
          }};
        }};
      }};
    }};
  }};

  # Swap file
  swapDevices = [{{ device = "/swap/swapfile"; size = {swap_gb} * 1024; }}];
}}"#
        ))
    }

    /// Generate a dual-disk disko configuration.
    ///
    /// Layout:
    /// - Fast drive (TLC): btrfs with @home, @srv, @swap, @snapshots (data, compressible)
    /// - Standard drive (QLC): ext4 root + EFI boot (OS, nix store, ephemeral caches)
    pub fn generate_dual_disk_disko_config(
        &self,
        fast_device: &str,
        standard_device: &str,
    ) -> Result<String, String> {
        let profile = self
            .hardware_profile
            .as_ref()
            .ok_or("Hardware profile required")?;

        let device_memory_gb = profile
            .get("device_memory_gb")
            .and_then(|v| v.as_f64())
            .unwrap_or(8.0);

        let swap_gb = if device_memory_gb <= 16.0 {
            device_memory_gb as u32
        } else {
            64 // Cap at 64G for large RAM systems
        };

        let efi_size = if self.lanzaboote_enabled {
            "1G"
        } else {
            "512M"
        };

        Ok(format!(
            r#"{{
  disko.devices.disk = {{
    # Standard drive: OS + Nix Store
    standard = {{
      type = "disk";
      device = "{standard_device}";
      content = {{
        type = "gpt";
        partitions = {{
          boot = {{
            name = "boot";
            size = "{efi_size}";
            type = "EF00";
            content = {{
              type = "filesystem";
              format = "vfat";
              mountpoint = "/boot";
              mountOptions = [ "fmask=0077" "dmask=0077" ];
            }};
          }};
          root = {{
            name = "nixos-root";
            size = "100%";
            content = {{
              type = "filesystem";
              format = "ext4";
              mountpoint = "/";
              mountOptions = [ "relatime" ];
            }};
          }};
        }};
      }};
    }};

    # Fast drive: Data (btrfs with subvolumes)
    fast = {{
      type = "disk";
      device = "{fast_device}";
      content = {{
        type = "gpt";
        partitions = {{
          data = {{
            name = "data";
            size = "100%";
            content = {{
              type = "btrfs";
              extraArgs = [ "-L" "samsung" "-f" ];
              subvolumes = {{
                "@home" = {{
                  mountpoint = "/home";
                  mountOptions = [ "compress=zstd:3" "noatime" "discard=async" "space_cache=v2" "nofail" ];
                }};
                "@srv" = {{
                  mountpoint = "/srv";
                  mountOptions = [ "compress=zstd:3" "noatime" "discard=async" "space_cache=v2" "nofail" ];
                }};
                "@swap" = {{
                  mountpoint = "/swap";
                  mountOptions = [ "noatime" "nofail" ];
                }};
                "@snapshots" = {{
                  mountpoint = "/.snapshots";
                  mountOptions = [ "compress=zstd:3" "noatime" "nofail" ];
                }};
              }};
            }};
          }};
        }};
      }};
    }};
  }};

  swapDevices = [{{ device = "/swap/swapfile"; size = {swap_gb} * 1024; }}];
}}"#
        ))
    }

    /// Detect if dual-disk layout should be used based on hardware profile.
    /// Returns (fast_device, standard_device) if two NVMe drives detected.
    pub fn detect_dual_disk(&self) -> Option<(String, String)> {
        let profile = self.hardware_profile.as_ref()?;
        let storage = profile.get("storage_quota_bytes")?.as_u64()?;
        let cpu_cores = profile.get("cpu_cores")?.as_u64().unwrap_or(4);

        // Heuristic: if storage > 500GB and cores >= 8, likely a workstation
        // with multiple NVMe drives. Browser can't detect individual drives,
        // so this is a recommendation — the user confirms.
        if storage > 500_000_000_000 && cpu_cores >= 8 {
            // Default device names — user must verify
            Some(("/dev/nvme0n1".into(), "/dev/nvme1n1".into()))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orchestrator_default() {
        let orch = InoculationOrchestrator::default();
        assert_eq!(orch.mode, InoculationMode::WebPortal);
        assert_eq!(orch.status, InoculationStatus::Idle);
        assert_eq!(orch.path, InoculationPath::Inoculate);
        assert!(orch.ahimsa_recovery);
        assert!(!orch.is_attested());
    }

    #[test]
    fn test_cannot_begin_without_attestation() {
        let orch = InoculationOrchestrator::new(InoculationMode::WebPortal);
        assert!(orch.can_begin().is_err());
        assert!(orch.can_begin().unwrap_err().contains("attestation"));
    }

    #[test]
    fn test_usb_forge_requires_device() {
        let mut orch = InoculationOrchestrator::new(InoculationMode::UsbForge);
        orch.attestation = Some(SovereignAttestation {
            signer_pubkey: "ab".repeat(32),
            signature: "cd".repeat(64),
            challenge: "ef".repeat(16),
            agent_pubkey: None,
            timestamp_ms: 1710000000000,
            phone_consciousness: 0.42,
        });
        orch.hardware_profile = Some(serde_json::json!({"cpu_cores": 8}));
        let err = orch.can_begin().unwrap_err();
        assert!(err.contains("USB"));
    }

    #[test]
    fn test_lan_inoculate_requires_ssh() {
        let mut orch = InoculationOrchestrator::new(InoculationMode::LanInoculate);
        orch.attestation = Some(SovereignAttestation {
            signer_pubkey: "ab".repeat(32),
            signature: "cd".repeat(64),
            challenge: "ef".repeat(16),
            agent_pubkey: None,
            timestamp_ms: 1710000000000,
            phone_consciousness: 0.42,
        });
        orch.hardware_profile = Some(serde_json::json!({"cpu_cores": 8}));
        let err = orch.can_begin().unwrap_err();
        assert!(err.contains("SSH"));
    }

    #[test]
    fn test_disko_config_generation() {
        let mut orch = InoculationOrchestrator::new(InoculationMode::WebPortal);
        orch.hardware_profile = Some(serde_json::json!({"device_memory_gb": 32.0}));
        let config = orch.generate_disko_config().unwrap();
        assert!(config.contains("symthaea-root"));
        assert!(config.contains("btrfs"));
        assert!(config.contains("recovery"));
        assert!(config.contains("compress=zstd"));
    }

    #[test]
    fn test_disko_config_no_recovery() {
        let mut orch = InoculationOrchestrator::new(InoculationMode::WebPortal);
        orch.ahimsa_recovery = false;
        orch.hardware_profile = Some(serde_json::json!({"device_memory_gb": 16.0}));
        let config = orch.generate_disko_config().unwrap();
        assert!(!config.contains("recovery"));
    }

    #[test]
    fn test_attestation_sets_status() {
        let mut orch = InoculationOrchestrator::default();
        orch.set_attestation(SovereignAttestation {
            signer_pubkey: "ab".repeat(32),
            signature: "cd".repeat(64),
            challenge: "ef".repeat(16),
            agent_pubkey: Some("uhCAktest".to_string()),
            timestamp_ms: 1710000000000,
            phone_consciousness: 0.42,
        });
        assert!(orch.is_attested());
        assert_eq!(orch.status, InoculationStatus::Attesting);
    }

    #[test]
    fn test_inoculation_modes_serialize() {
        let modes = vec![
            InoculationMode::WebPortal,
            InoculationMode::UsbForge,
            InoculationMode::Wsl2Pivot,
            InoculationMode::AsahiHandshake,
            InoculationMode::LanInoculate,
        ];
        for mode in modes {
            let json = serde_json::to_string(&mode).unwrap();
            let back: InoculationMode = serde_json::from_str(&json).unwrap();
            assert_eq!(mode, back);
        }
    }

    #[test]
    fn test_mesh_status_defaults() {
        let orch = InoculationOrchestrator::default();
        assert_eq!(orch.mesh_status.primary_source, AssetSource::Cdn);
        assert_eq!(orch.mesh_status.iroh_peers, 0);
        assert!(!orch.mesh_status.fully_decentralized);
    }

    #[test]
    fn test_error_recording() {
        let mut orch = InoculationOrchestrator::default();
        orch.record_error("disk full".to_string());
        orch.record_error("network timeout".to_string());
        assert_eq!(orch.errors.len(), 2);
    }

    #[test]
    fn test_needs_mok_enrollment() {
        let mut orch = InoculationOrchestrator::default();
        orch.secure_boot_state = Some("Enabled".to_string());
        orch.lanzaboote_enabled = true;
        assert!(orch.needs_mok_enrollment());
    }

    #[test]
    fn test_no_mok_without_secure_boot() {
        let orch = InoculationOrchestrator::default();
        // secure_boot_state is None by default
        assert!(!orch.needs_mok_enrollment());

        // Also false when secure boot is disabled
        let mut orch2 = InoculationOrchestrator::default();
        orch2.secure_boot_state = Some("Disabled".to_string());
        orch2.lanzaboote_enabled = true;
        assert!(!orch2.needs_mok_enrollment());
    }

    #[test]
    fn test_boot_config_lanzaboote() {
        let mut orch = InoculationOrchestrator::default();
        orch.lanzaboote_enabled = true;
        let config = orch.generate_boot_config_nix();
        assert!(config.contains("lanzaboote"));
        assert!(config.contains("pkiBundle"));
        assert!(config.contains("sbctl"));
        assert!(config.contains("lib.mkForce false"));
    }

    #[test]
    fn test_boot_config_standard() {
        let mut orch = InoculationOrchestrator::default();
        orch.lanzaboote_enabled = false;
        let config = orch.generate_boot_config_nix();
        assert!(config.contains("systemd-boot.enable = true"));
        assert!(!config.contains("lanzaboote"));
    }

    #[test]
    fn test_disko_efi_size_with_lanzaboote() {
        let mut orch = InoculationOrchestrator::default();
        orch.hardware_profile = Some(serde_json::json!({"device_memory_gb": 16.0}));

        // lanzaboote enabled (default) — EFI should be 1G
        orch.lanzaboote_enabled = true;
        let config = orch.generate_disko_config().unwrap();
        assert!(config.contains("\"1G\""));
        assert!(!config.contains("\"512M\""));

        // lanzaboote disabled — EFI should be 512M
        orch.lanzaboote_enabled = false;
        let config = orch.generate_disko_config().unwrap();
        assert!(config.contains("\"512M\""));
        assert!(!config.contains("\"1G\""));
    }

    // -----------------------------------------------------------------------
    // Dual-path model tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_inoculation_path_default_is_sovereign() {
        let orch = InoculationOrchestrator::default();
        assert_eq!(orch.path, InoculationPath::Inoculate);
        assert!(!orch.is_attuned());
    }

    #[test]
    fn test_attune_and_deattune_roundtrip() {
        let mut orch = InoculationOrchestrator::default();
        assert_eq!(orch.path, InoculationPath::Inoculate);
        assert!(!orch.is_attuned());

        // Upgrade to mesh participation
        orch.attune();
        assert_eq!(orch.path, InoculationPath::InoculateAndAttune);
        assert!(orch.is_attuned());

        // Downgrade back to sovereign-only
        orch.deattune();
        assert_eq!(orch.path, InoculationPath::Inoculate);
        assert!(!orch.is_attuned());
    }

    #[test]
    fn test_attune_is_idempotent() {
        let mut orch = InoculationOrchestrator::default();
        orch.attune();
        orch.attune(); // second call should be harmless
        assert_eq!(orch.path, InoculationPath::InoculateAndAttune);

        orch.deattune();
        orch.deattune(); // second call should be harmless
        assert_eq!(orch.path, InoculationPath::Inoculate);
    }

    #[test]
    fn test_inoculation_path_serialization() {
        let paths = vec![
            InoculationPath::Inoculate,
            InoculationPath::InoculateAndAttune,
        ];
        for path in paths {
            let json = serde_json::to_string(&path).unwrap();
            let back: InoculationPath = serde_json::from_str(&json).unwrap();
            assert_eq!(path, back);
        }
    }
}
