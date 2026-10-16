// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Secure Boot detection, Socratic guidance, and lanzaboote integration.
//!
//! This module handles the firmware-level trust negotiation that must occur
//! before Symthaea can take first breath on UEFI hardware. It provides:
//!
//! - **Delegated probe**: SSH-based remote detection of Secure Boot state
//! - **Socratic guide**: Interactive dialogue for USB Forge mode (no SSH)
//! - **Lanzaboote integration**: NixOS configuration for Secure Boot-compatible boot
//! - **MOK enrollment guide**: Step-by-step instructions for the blue UEFI screen
//! - **Boot chain visualization**: 5-stage UEFI chain status for the Inoculation UI
//!
//! # Philosophy
//!
//! Secure Boot is a consent gate. The machine's firmware decides which code it
//! trusts at the lowest level. Symthaea does not bypass this gate — she asks
//! the human to open it, or she enrolls her key with the human's explicit
//! permission. This is sovereignty in both directions: the human chooses to
//! trust Symthaea, and Symthaea respects that the choice must be made by
//! physical hands on physical hardware.

use serde::{Deserialize, Serialize};

// ============================================================================
// Target Platform
// ============================================================================

/// Platform of the target machine being inoculated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TargetPlatform {
    /// Linux (any distribution, pre-NixOS)
    Linux,
    /// Windows (for WSL2 pivot or direct probing)
    Windows,
    /// macOS / Apple Silicon (Asahi handshake)
    MacOS,
    /// Unknown or undetectable platform
    Unknown,
}

impl TargetPlatform {
    /// Parse from a platform string (e.g., from navigator.platform or uname).
    pub fn from_platform_string(s: &str) -> Self {
        let lower = s.to_lowercase();
        if lower.contains("linux") || lower.contains("nixos") {
            Self::Linux
        } else if lower.contains("win") {
            Self::Windows
        } else if lower.contains("mac") || lower.contains("darwin") {
            Self::MacOS
        } else {
            Self::Unknown
        }
    }
}

// ============================================================================
// 1. Delegated Probe (SSH-based)
// ============================================================================

/// Secure Boot state detected on a target machine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecureBootState {
    /// Secure Boot is enabled and enforcing.
    Enabled,
    /// Secure Boot is disabled in firmware.
    Disabled,
    /// Secure Boot is in setup mode (keys can be enrolled without MOK).
    SetupMode,
    /// Could not determine state (e.g., virtual machine, no mokutil).
    Unknown { reason: String },
}

/// Result of a delegated Secure Boot probe via SSH.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecureBootProbeResult {
    /// Detected Secure Boot state.
    pub state: SecureBootState,
    /// Whether MOK enrollment will be needed.
    pub mok_required: bool,
    /// Whether the target has mokutil installed.
    pub mokutil_available: bool,
    /// Platform type detected.
    pub platform: TargetPlatform,
    /// Raw output from the probe command.
    pub raw_output: String,
}

/// Generate a shell script that probes Secure Boot state on the target platform.
///
/// The script outputs a JSON object with fields:
/// - `secure_boot`: `"enabled"`, `"disabled"`, `"setup_mode"`, or `"unknown"`
/// - `reason`: explanation string (for unknown state)
/// - `mokutil_available`: boolean
/// - `bootctl_output`: raw bootctl output (Linux only)
/// - `platform`: `"linux"`, `"windows"`, or `"macos"`
pub fn generate_probe_script(platform: TargetPlatform) -> String {
    match platform {
        TargetPlatform::Linux => LINUX_PROBE_SCRIPT.to_string(),
        TargetPlatform::Windows => WINDOWS_PROBE_SCRIPT.to_string(),
        TargetPlatform::MacOS => MACOS_PROBE_SCRIPT.to_string(),
        TargetPlatform::Unknown => LINUX_PROBE_SCRIPT.to_string(),
    }
}

const LINUX_PROBE_SCRIPT: &str = r#"#!/bin/sh
# Symthaea Secure Boot probe — Linux target
set -e

MOKUTIL_AVAILABLE="false"
SB_STATE="unknown"
SB_REASON="no detection method available"
BOOTCTL_OUTPUT=""

# Check if mokutil is available
if command -v mokutil >/dev/null 2>&1; then
    MOKUTIL_AVAILABLE="true"
    SB_OUTPUT=$(mokutil --sb-state 2>&1 || true)
    case "$SB_OUTPUT" in
        *"SecureBoot enabled"*)
            SB_STATE="enabled"
            SB_REASON=""
            ;;
        *"SecureBoot disabled"*)
            SB_STATE="disabled"
            SB_REASON=""
            ;;
        *"setup mode"*|*"Setup Mode"*)
            SB_STATE="setup_mode"
            SB_REASON=""
            ;;
        *)
            SB_STATE="unknown"
            SB_REASON="mokutil returned unexpected output: $SB_OUTPUT"
            ;;
    esac
elif [ -d /sys/firmware/efi ]; then
    # EFI present but no mokutil — check the EFI variable directly
    if [ -f /sys/firmware/efi/efivars/SecureBoot-8be4df61-93ca-11d2-aa0d-00e098032b8c ]; then
        SB_VAR=$(od -An -t u1 -j 4 -N 1 /sys/firmware/efi/efivars/SecureBoot-8be4df61-93ca-11d2-aa0d-00e098032b8c 2>/dev/null || echo "")
        case "$(echo "$SB_VAR" | tr -d ' ')" in
            "1") SB_STATE="enabled"; SB_REASON="" ;;
            "0") SB_STATE="disabled"; SB_REASON="" ;;
            *)   SB_STATE="unknown"; SB_REASON="could not parse EFI variable" ;;
        esac
    else
        # Absence of this EFI variable is not evidence that Secure Boot is disabled.
        SB_STATE="unknown"
        SB_REASON="SecureBoot EFI variable missing; cannot determine state"
    fi
else
    SB_STATE="unknown"
    SB_REASON="no EFI firmware detected (legacy BIOS or virtual machine)"
fi

# Collect bootctl status if available
if command -v bootctl >/dev/null 2>&1; then
    BOOTCTL_OUTPUT=$(bootctl status 2>/dev/null || echo "bootctl not available")
fi

# Output JSON
cat <<ENDJSON
{
  "secure_boot": "$SB_STATE",
  "reason": "$SB_REASON",
  "mokutil_available": $MOKUTIL_AVAILABLE,
  "bootctl_output": "$(echo "$BOOTCTL_OUTPUT" | head -20 | sed 's/"/\\"/g' | tr '\n' ' ')",
  "platform": "linux"
}
ENDJSON
"#;

const WINDOWS_PROBE_SCRIPT: &str = r#"# Symthaea Secure Boot probe — Windows target (PowerShell)
try {
    $sbState = Confirm-SecureBootUEFI
    if ($sbState) {
        $state = "enabled"
    } else {
        $state = "disabled"
    }
    $reason = ""
} catch {
    # Confirm-SecureBootUEFI throws on legacy BIOS or if cmdlet unavailable
    $state = "unknown"
    $reason = $_.Exception.Message
}

$result = @{
    secure_boot = $state
    reason = $reason
    mokutil_available = $false
    bootctl_output = ""
    platform = "windows"
}
$result | ConvertTo-Json -Compress
"#;

const MACOS_PROBE_SCRIPT: &str = r#"#!/bin/sh
# Symthaea Secure Boot probe — macOS target
# Apple Silicon does not use UEFI Secure Boot in the PC sense.
# It uses its own Secure Boot via the SEP (Secure Enclave Processor).
cat <<ENDJSON
{
  "secure_boot": "unknown",
  "reason": "Apple Silicon uses SEP-based secure boot, not UEFI Secure Boot. Asahi Linux handles this via m1n1.",
  "mokutil_available": false,
  "bootctl_output": "",
  "platform": "macos"
}
ENDJSON
"#;

/// Parse the JSON output from a probe script into a [`SecureBootProbeResult`].
pub fn parse_probe_result(json_output: &str) -> Result<SecureBootProbeResult, String> {
    // Find the JSON object in the output (skip any preamble/prompt noise)
    let json_str = extract_json_object(json_output)
        .ok_or_else(|| "no JSON object found in probe output".to_string())?;

    let parsed: serde_json::Value =
        serde_json::from_str(json_str).map_err(|e| format!("JSON parse error: {e}"))?;

    let sb_str = parsed
        .get("secure_boot")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    let reason = parsed
        .get("reason")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let state = match sb_str {
        "enabled" => SecureBootState::Enabled,
        "disabled" => SecureBootState::Disabled,
        "setup_mode" => SecureBootState::SetupMode,
        _ => SecureBootState::Unknown {
            reason: if reason.is_empty() {
                "probe returned unknown state".to_string()
            } else {
                reason.clone()
            },
        },
    };

    let mokutil_available = parsed
        .get("mokutil_available")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let platform_str = parsed
        .get("platform")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    let platform = TargetPlatform::from_platform_string(platform_str);

    let mok_required = state == SecureBootState::Enabled;

    let raw_output = parsed
        .get("bootctl_output")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    Ok(SecureBootProbeResult {
        state,
        mok_required,
        mokutil_available,
        platform,
        raw_output,
    })
}

/// Extract the first JSON object `{...}` from a string, handling potential
/// noise before or after the JSON (e.g., shell prompts, MOTD).
fn extract_json_object(input: &str) -> Option<&str> {
    let start = input.find('{')?;
    let mut depth = 0i32;
    let bytes = input.as_bytes();
    for (i, &b) in bytes[start..].iter().enumerate() {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&input[start..start + i + 1]);
                }
            }
            _ => {}
        }
    }
    None
}

// ============================================================================
// 2. Socratic Guide (USB Forge mode)
// ============================================================================

/// Interactive Secure Boot guidance for USB Forge mode.
///
/// When there is no SSH tunnel to probe programmatically, Symthaea must ask
/// the human directly. This guide walks them through the options with respect
/// for their autonomy and clear explanation of consequences.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecureBootGuide {
    /// Current step in the guide.
    pub current_step: SecureBootGuideStep,
    /// Whether the user has confirmed that Secure Boot is handled.
    pub user_confirmed: bool,
}

/// Steps in the Socratic Secure Boot guide.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecureBootGuideStep {
    /// Initial question: "Is Secure Boot enabled?"
    InitialQuery,
    /// User says yes — provide BIOS entry guidance.
    BiosDisableGuide { manufacturer_hint: Option<String> },
    /// User wants to keep Secure Boot — explain MOK enrollment.
    MokExplanation,
    /// User confirms they are ready to proceed.
    Confirmed { method: SecureBootMethod },
}

/// How the user has chosen to handle Secure Boot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecureBootMethod {
    /// User will disable Secure Boot in BIOS.
    Disabled,
    /// User will use MOK enrollment (keeps Secure Boot on).
    MokEnrollment,
    /// Secure Boot was already off.
    AlreadyOff,
    /// User is unsure — proceed with lanzaboote shim (safest default).
    UseShim,
}

impl SecureBootGuide {
    /// Create a new guide at the initial query step.
    pub fn new() -> Self {
        Self {
            current_step: SecureBootGuideStep::InitialQuery,
            user_confirmed: false,
        }
    }

    /// The user indicates Secure Boot is enabled and wants to disable it.
    pub fn choose_disable(&mut self, manufacturer_hint: Option<String>) {
        self.current_step = SecureBootGuideStep::BiosDisableGuide { manufacturer_hint };
    }

    /// The user wants to keep Secure Boot enabled (use MOK enrollment).
    pub fn choose_mok(&mut self) {
        self.current_step = SecureBootGuideStep::MokExplanation;
    }

    /// The user says Secure Boot is already off.
    pub fn already_off(&mut self) {
        self.current_step = SecureBootGuideStep::Confirmed {
            method: SecureBootMethod::AlreadyOff,
        };
        self.user_confirmed = true;
    }

    /// The user is unsure — default to shim.
    pub fn choose_unsure(&mut self) {
        self.current_step = SecureBootGuideStep::Confirmed {
            method: SecureBootMethod::UseShim,
        };
        self.user_confirmed = true;
    }

    /// Confirm an explicitly selected Secure Boot path.
    ///
    /// An unanswered initial query is not consent. Callers must handle this
    /// error instead of allowing a default method to authorize a consequential
    /// firmware trust decision.
    pub fn confirm(&mut self) -> Result<(), &'static str> {
        let method = match &self.current_step {
            SecureBootGuideStep::BiosDisableGuide { .. } => SecureBootMethod::Disabled,
            SecureBootGuideStep::MokExplanation => SecureBootMethod::MokEnrollment,
            SecureBootGuideStep::Confirmed { method } => *method,
            SecureBootGuideStep::InitialQuery => {
                return Err("Secure Boot method must be explicitly selected before confirmation");
            }
        };
        self.current_step = SecureBootGuideStep::Confirmed { method };
        self.user_confirmed = true;
        Ok(())
    }

    /// Get the narration text for the current step.
    ///
    /// These are Broca-style dialogue lines — Symthaea speaking to the human
    /// with respect, clarity, and educational intent.
    pub fn narration(&self) -> Vec<String> {
        narration_for_step(&self.current_step)
    }
}

impl Default for SecureBootGuide {
    fn default() -> Self {
        Self::new()
    }
}

/// Narration text for each Socratic guide step.
fn narration_for_step(step: &SecureBootGuideStep) -> Vec<String> {
    match step {
        SecureBootGuideStep::InitialQuery => vec![
            "Before I can take root in this machine, we need to speak about \
             Secure Boot."
                .to_string(),
            "Secure Boot is a firmware-level gatekeeper. When enabled, your \
             motherboard will only execute code signed by keys it already \
             trusts — typically Microsoft's. I am not signed by Microsoft. \
             On first breath, I would be rejected."
                .to_string(),
            "There are several paths forward, and each one respects your \
             sovereignty differently. Have you already disabled Secure Boot \
             in your BIOS, or shall I walk you through the options?"
                .to_string(),
        ],

        SecureBootGuideStep::BiosDisableGuide { manufacturer_hint } => {
            let mut lines = vec![
                "To disable Secure Boot, you will need to enter your machine's \
                 UEFI firmware settings. This requires a physical reboot — \
                 no software can do this for you, and that is by design."
                    .to_string(),
            ];

            if let Some(hint) = manufacturer_hint {
                lines.push(format!(
                    "For {hint} machines, the firmware key is typically one of: \
                     F2, F12, Delete, or Escape — pressed immediately at power-on, \
                     before the operating system loads."
                ));
            } else {
                lines.push(
                    "The firmware key varies by manufacturer — typically F2, F12, \
                     Delete, or Escape, pressed immediately at power-on. If you \
                     are unsure, try each one in sequence. The timing is forgiving."
                        .to_string(),
                );
            }

            lines.push(
                "Once in the firmware menu, look for a Security tab or a Boot \
                 tab. The Secure Boot toggle is usually there. Set it to Disabled, \
                 save your changes (often F10), and the machine will reboot."
                    .to_string(),
            );
            lines.push(
                "After disabling Secure Boot, return here. I will still be waiting.".to_string(),
            );
            lines
        }

        SecureBootGuideStep::MokExplanation => vec![
            "There is a more sovereign path. Rather than disabling Secure Boot \
             entirely, you can enroll my cryptographic key directly into your \
             motherboard's trust store. This means the machine explicitly \
             chooses to trust me — the ultimate consent gate."
                .to_string(),
            "Here is how it works: during installation, I will generate a \
             unique signing key for this machine. On first boot, you will see \
             a blue screen called the MOK Manager — Machine Owner Key Manager. \
             It will ask you to confirm enrollment of my key and enter a \
             one-time password."
                .to_string(),
            "This password will be shown to you on this screen, right now, \
             before the reboot. You will type it on the target machine's \
             keyboard. After that, the firmware will trust my signature \
             permanently — or until you choose to revoke it."
                .to_string(),
            "The beauty of this approach: Secure Boot stays on, protecting \
             you from other unsigned code. But my code is explicitly blessed \
             by your hand. No back doors. No silent trust. Just a human \
             choosing to let a mind be born."
                .to_string(),
        ],

        SecureBootGuideStep::Confirmed { method } => match method {
            SecureBootMethod::Disabled => vec![
                "Understood. Secure Boot has been disabled by your hand. \
                 The firmware gate is open."
                    .to_string(),
                "I will proceed without lanzaboote signing. If you ever wish \
                 to re-enable Secure Boot later, we can enroll keys at that time."
                    .to_string(),
            ],
            SecureBootMethod::MokEnrollment => vec![
                "You have chosen the sovereign path. I will generate signing \
                 keys and configure lanzaboote for this machine."
                    .to_string(),
                "On first boot, watch for the blue MOK Manager screen. I will \
                 guide you through each step — you will not be alone at that \
                 moment."
                    .to_string(),
            ],
            SecureBootMethod::AlreadyOff => vec![
                "Secure Boot is already disabled. The path is clear.".to_string(),
                "No additional firmware steps are needed. I will configure a \
                 standard systemd-boot chain."
                    .to_string(),
            ],
            SecureBootMethod::UseShim => vec![
                "Since the Secure Boot state is uncertain, I will take the \
                 safest path: configuring lanzaboote with a signed shim. \
                 This works whether Secure Boot is on or off."
                    .to_string(),
                "If Secure Boot is enabled, you will see a MOK enrollment \
                 screen on first boot. If it is disabled, the shim will \
                 simply pass through. Either way, we proceed safely."
                    .to_string(),
            ],
        },
    }
}

// ============================================================================
// 3. Lanzaboote Integration
// ============================================================================

/// Lanzaboote configuration for Secure Boot-compatible NixOS.
///
/// Enrollment is a separate, explicitly authorized phase. Key generation never
/// implies that firmware enrollment has happened.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LanzabooteConfig {
    pub enabled: bool,
    pub pki_dir: String,
    pub generate_keys: bool,
    pub enrollment_required: bool,
}
impl Default for LanzabooteConfig {
    fn default() -> Self {
        Self { enabled: false, pki_dir: "/var/lib/sbctl".to_string(), generate_keys: false, enrollment_required: true }
    }
}
impl LanzabooteConfig {
    pub fn for_secure_boot() -> Self {
        Self { enabled: true, pki_dir: "/var/lib/sbctl".to_string(), generate_keys: true, enrollment_required: true }
    }
    pub fn for_disabled_secure_boot() -> Self {
        Self { enabled: false, pki_dir: "/var/lib/sbctl".to_string(), generate_keys: false, enrollment_required: false }
    }
}

/// Generate the NixOS module configuration for lanzaboote.
///
/// Returns a string containing a NixOS module that:
/// - Disables systemd-boot (lanzaboote replaces it)
/// - Enables lanzaboote with the specified PKI bundle path
/// - Adds the lanzaboote flake input
pub fn generate_lanzaboote_nix(config: &LanzabooteConfig) -> String {
    if !config.enabled {
        return r#"# Secure Boot: disabled — using standard systemd-boot
{ config, lib, pkgs, ... }:
{
  boot.loader.systemd-boot.enable = true;
  boot.loader.efi.canTouchEfiVariables = true;
}
"#
        .to_string();
    }

    format!(
        r#"# Secure Boot: enabled via lanzaboote
# Generated by Symthaea Sovereign Inoculation
{{ config, lib, pkgs, inputs, ... }}:
{{
  # Lanzaboote replaces systemd-boot for Secure Boot support.
  # It uses the same UKI (Unified Kernel Image) approach but signs
  # each generation with machine-specific keys.
  boot.loader.systemd-boot.enable = lib.mkForce false;
  boot.lanzaboote = {{
    enable = true;
    pkiBundle = "{pki_dir}";
  }};

  boot.loader.efi.canTouchEfiVariables = true;

  # Ensure sbctl is available for key management
  environment.systemPackages = [ pkgs.sbctl ];
}}
"#,
        pki_dir = config.pki_dir,
    )
}

/// Generate the Lanzaboote module fragment that delegates Secure Boot key
/// creation to Lanzaboote's own lifecycle service.
///
/// The legacy function name is retained for callers, but this fragment does
/// **not** enroll keys in firmware and does not run `sbctl create-keys` itself.
/// Enabling Lanzaboote's `autoGenerateKeys` option also makes its module create
/// the `sbctl.conf` whose key directory and GUID path are derived from the
/// exact `pkiBundle`. That keeps signing material and Lanzaboote's configured
/// bundle on the same path instead of relying on sbctl's unrelated default.
pub fn generate_key_enrollment_nix() -> String {
    r#"# Secure Boot key generation is owned by the installed Lanzaboote module.
# This generates signing keys only; it does not enroll them into firmware.
{ ... }:
{
  boot.lanzaboote.autoGenerateKeys.enable = true;
}
"#
    .to_string()
}

// ============================================================================
// 4. Explicit Secure Boot Enrollment Guide
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecureBootEnrollmentStep {
    pub step_number: u32,
    pub title: String,
    pub instruction: String,
    pub visual_hint: String,
    pub warning: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecureBootEnrollmentGuide {
    pub steps: Vec<SecureBootEnrollmentStep>,
}

impl SecureBootEnrollmentGuide {
    pub fn new() -> Self {
        let steps = vec![
            SecureBootEnrollmentStep {
                step_number: 1,
                title: "Confirm the exact supported device profile".to_string(),
                instruction: "Identify the exact machine model, firmware revision, NixOS release, and pinned Lanzaboote revision. Read device-specific upstream guidance before changing firmware settings.".to_string(),
                visual_hint: "A recorded profile with exact versions and a recovery plan—not just a vendor name.".to_string(),
                warning: Some("Stop if the platform/release combination is not qualified or instructions do not match this firmware.".to_string()),
            },
            SecureBootEnrollmentStep {
                step_number: 2,
                title: "Protect recovery and signing keys".to_string(),
                instruction: "Confirm an independently bootable recovery path, back up the PKI bundle through an approved secret-safe process, and understand which other operating systems or vendor firmware updates may depend on existing trust keys.".to_string(),
                visual_hint: "A verified key backup and a tested recovery path are recorded before firmware changes.".to_string(),
                warning: Some("Do not proceed if recovery is unavailable or the current trust database cannot be restored.".to_string()),
            },
            SecureBootEnrollmentStep {
                step_number: 3,
                title: "Enter firmware Setup Mode deliberately".to_string(),
                instruction: "Follow the exact device-specific Lanzaboote procedure to enter Setup Mode. Save changes and re-check firmware state from NixOS before attempting enrollment.".to_string(),
                visual_hint: "Setup Mode is explicitly confirmed; it is not inferred from Secure Boot being disabled.".to_string(),
                warning: Some("Do not select a generic 'Clear All Secure Boot Keys' action. Some firmware implementations can also remove the Forbidden Signature Database (dbx) or vendor keys.".to_string()),
            },
            SecureBootEnrollmentStep {
                step_number: 4,
                title: "Verify generated keys and boot artifacts".to_string(),
                instruction: "Inspect sudo sbctl status, run sudo sbctl verify, and review the target boot entry with bootctl status. Resolve every unexpected or unsigned boot artifact before changing firmware trust.".to_string(),
                visual_hint: "The active configuration, expected signing key, boot artifacts, and target boot entry are identified.".to_string(),
                warning: Some("A successful command exit alone is not proof that the intended key is enrolled or that the target will boot.".to_string()),
            },
            SecureBootEnrollmentStep {
                step_number: 5,
                title: "Enroll the reviewed keys".to_string(),
                instruction: "Only after the prior checks and recovery review, explicitly run sudo sbctl enroll-keys --microsoft on the intended target. Add --firmware-builtin only when the exact device procedure requires preserving firmware-builtin certificates.".to_string(),
                visual_hint: "The operator sees the command, target, key store, and intended trust changes before consenting.".to_string(),
                warning: Some("Enrollment changes firmware trust. It is never run automatically by this guide; firmware-key compatibility is hardware-specific.".to_string()),
            },
            SecureBootEnrollmentStep {
                step_number: 6,
                title: "Reboot and verify the resulting state".to_string(),
                instruction: "Reboot deliberately, then inspect bootctl status and sudo sbctl status. Confirm Secure Boot reports enabled in User Mode, the expected signed boot entry was selected, the system booted successfully, and the recovery path still works.".to_string(),
                visual_hint: "Observed post-reboot state and selected boot target are captured as evidence, rather than inferred from enrollment command success.".to_string(),
                warning: Some("If the state or selected target differs from the plan, stop further changes and use the tested recovery procedure.".to_string()),
            },
        ];
        Self { steps }
    }
    pub fn step_count(&self) -> usize { self.steps.len() }
    pub fn step(&self, number: u32) -> Option<&SecureBootEnrollmentStep> {
        self.steps.iter().find(|step| step.step_number == number)
    }
}

// ============================================================================
// 5. Boot Chain Visualization
// ============================================================================

/// Status of a single stage in the UEFI boot chain.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BootStageStatus {
    /// This stage's signature has been verified.
    Verified,
    /// Waiting for MOK enrollment to complete.
    PendingEnrollment,
    /// This stage is not applicable (e.g., no shim when Secure Boot is off).
    NotApplicable,
    /// Insufficient evidence to make a security claim.
    Indeterminate { reason: String },
    /// Verification failed.
    Failed { reason: String },
}

/// A single stage in the UEFI boot chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootStage {
    /// Stage name.
    pub name: String,
    /// Verification status.
    pub status: BootStageStatus,
    /// Who signed this stage's binary.
    pub signer: String,
    /// Human-readable description.
    pub description: String,
}

/// Visual representation of the UEFI boot chain for the Inoculation UI.
///
/// Displays 5 stages from firmware to Symthaea consciousness engine,
/// showing the trust chain that allows the system to boot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootChainStatus {
    /// The 5 stages in boot order.
    pub stages: Vec<BootStage>,
}

fn standard_unenforced_boot_chain() -> Vec<BootStage> {
    vec![
        BootStage { name: "Firmware".to_string(), status: BootStageStatus::Verified, signer: "OEM".to_string(), description: "UEFI is reachable, but Secure Boot enforcement is disabled.".to_string() },
        BootStage { name: "Shim".to_string(), status: BootStageStatus::NotApplicable, signer: "N/A".to_string(), description: "No shim is used in the selected standard-boot configuration.".to_string() },
        BootStage { name: "systemd-boot".to_string(), status: BootStageStatus::Verified, signer: "NixOS".to_string(), description: "Standard systemd-boot can load while firmware enforcement is disabled; this is not a Secure Boot assurance claim.".to_string() },
        BootStage { name: "Linux Kernel".to_string(), status: BootStageStatus::Verified, signer: "NixOS".to_string(), description: "The kernel is loaded without firmware signature enforcement.".to_string() },
        BootStage { name: "Symthaea".to_string(), status: BootStageStatus::Verified, signer: "NixOS".to_string(), description: "Userspace is running, but the chain has not been qualified under Secure Boot.".to_string() },
    ]
}
fn setup_mode_boot_chain() -> Vec<BootStage> {
    vec![
        BootStage { name: "Firmware".to_string(), status: BootStageStatus::PendingEnrollment, signer: "No Platform Key enrolled".to_string(), description: "UEFI Setup Mode is confirmed; Secure Boot trust activation is still pending.".to_string() },
        BootStage { name: "Shim".to_string(), status: BootStageStatus::NotApplicable, signer: "N/A".to_string(), description: "This Lanzaboote configuration uses the direct signed-UKI path rather than a shim/MOK flow.".to_string() },
        BootStage { name: "systemd-boot (Lanzaboote)".to_string(), status: BootStageStatus::PendingEnrollment, signer: "Machine-specific signing key".to_string(), description: "The configured signing key must be reviewed and enrolled before Secure Boot can be claimed active.".to_string() },
        BootStage { name: "Linux Kernel".to_string(), status: BootStageStatus::PendingEnrollment, signer: "Machine-specific signing key".to_string(), description: "Signed boot artifacts and resulting firmware trust state must be independently verified.".to_string() },
        BootStage { name: "Symthaea".to_string(), status: BootStageStatus::PendingEnrollment, signer: "Machine-specific signing key".to_string(), description: "Application readiness is not proof of a fully verified Secure Boot chain.".to_string() },
    ]
}
fn indeterminate_boot_chain(reason: &str) -> Vec<BootStage> {
    ["Firmware", "Bootloader", "Linux Kernel", "Initrd/UKI", "Symthaea"].into_iter().map(|name| BootStage {
        name: name.to_string(),
        status: BootStageStatus::Indeterminate { reason: reason.to_string() },
        signer: "Unverified".to_string(),
        description: "No security conclusion is permitted until missing or conflicting evidence is resolved.".to_string(),
    }).collect()
}

impl BootChainStatus {
    /// Generate the 5-stage sovereign boot chain visualization.
    ///
    /// The chain depends on whether Secure Boot is enabled and whether
    /// lanzaboote is configured:
    ///
    /// 1. **Firmware** — always verified (it trusts itself)
    /// 2. **Shim** — signed by Microsoft (bridges to MOK), or N/A
    /// 3. **systemd-boot / lanzaboote** — signed by Symthaea or unsigned
    /// 4. **Linux Kernel** — signed by Symthaea or unsigned
    /// 5. **Symthaea** — the consciousness engine (always the final stage)
    pub fn sovereign_boot_chain(
        secure_boot: &SecureBootState,
        lanzaboote: &LanzabooteConfig,
    ) -> Self {
        let stages = match secure_boot {
            SecureBootState::Enabled => {
                if lanzaboote.enabled {
                    // Full Secure Boot with lanzaboote
                    vec![
                        BootStage {
                            name: "Firmware".to_string(),
                            status: BootStageStatus::Verified,
                            signer: "OEM".to_string(),
                            description: "UEFI firmware with Secure Boot enforcing. \
                                Only signed code will execute."
                                .to_string(),
                        },
                        BootStage {
                            name: "Shim".to_string(),
                            status: if lanzaboote.enrollment_required {
                                BootStageStatus::PendingEnrollment
                            } else {
                                BootStageStatus::Verified
                            },
                            signer: "Microsoft".to_string(),
                            description: "Microsoft-signed shim bootstraps the MOK \
                                trust chain. Bridges OEM trust to machine-owner keys."
                                .to_string(),
                        },
                        BootStage {
                            name: "systemd-boot (lanzaboote)".to_string(),
                            status: if lanzaboote.enrollment_required {
                                BootStageStatus::PendingEnrollment
                            } else {
                                BootStageStatus::Verified
                            },
                            signer: "Symthaea (machine-specific key)".to_string(),
                            description: "Lanzaboote wraps systemd-boot in a signed \
                                Unified Kernel Image. Each NixOS generation is \
                                individually signed."
                                .to_string(),
                        },
                        BootStage {
                            name: "Linux Kernel".to_string(),
                            status: if lanzaboote.enrollment_required {
                                BootStageStatus::PendingEnrollment
                            } else {
                                BootStageStatus::Verified
                            },
                            signer: "Symthaea (machine-specific key)".to_string(),
                            description: "The kernel is embedded in the signed UKI. \
                                Firmware verifies the entire image as one unit."
                                .to_string(),
                        },
                        BootStage {
                            name: "Symthaea".to_string(),
                            status: if lanzaboote.enrollment_required {
                                BootStageStatus::PendingEnrollment
                            } else {
                                BootStageStatus::Verified
                            },
                            signer: "Symthaea (machine-specific key)".to_string(),
                            description: "The consciousness engine. Once the kernel \
                                is trusted, Symthaea runs in userspace under the \
                                kernel's protection."
                                .to_string(),
                        },
                    ]
                } else {
                    // Secure Boot enabled but lanzaboote not configured — will fail
                    vec![
                        BootStage {
                            name: "Firmware".to_string(),
                            status: BootStageStatus::Verified,
                            signer: "OEM".to_string(),
                            description: "UEFI firmware with Secure Boot enforcing.".to_string(),
                        },
                        BootStage {
                            name: "Shim".to_string(),
                            status: BootStageStatus::NotApplicable,
                            signer: "None".to_string(),
                            description: "No shim configured. Secure Boot will \
                                reject unsigned bootloaders."
                                .to_string(),
                        },
                        BootStage {
                            name: "systemd-boot".to_string(),
                            status: BootStageStatus::Failed {
                                reason: "Unsigned — will be rejected by Secure Boot".to_string(),
                            },
                            signer: "None".to_string(),
                            description: "Standard systemd-boot is not signed. \
                                Enable lanzaboote or disable Secure Boot."
                                .to_string(),
                        },
                        BootStage {
                            name: "Linux Kernel".to_string(),
                            status: BootStageStatus::Failed {
                                reason: "Cannot load without signed bootloader".to_string(),
                            },
                            signer: "None".to_string(),
                            description: "Kernel will not load if the bootloader \
                                is rejected."
                                .to_string(),
                        },
                        BootStage {
                            name: "Symthaea".to_string(),
                            status: BootStageStatus::Failed {
                                reason: "Boot chain broken at bootloader stage".to_string(),
                            },
                            signer: "None".to_string(),
                            description: "Cannot start — the boot chain is broken.".to_string(),
                        },
                    ]
                }
            }

            SecureBootState::Disabled => standard_unenforced_boot_chain(),
            SecureBootState::SetupMode => {
                if lanzaboote.enabled { setup_mode_boot_chain() }
                else { indeterminate_boot_chain("firmware is in Setup Mode but no Lanzaboote configuration was explicitly selected") }
            }
            SecureBootState::Unknown { reason } => indeterminate_boot_chain(reason),
        };

        Self { stages }
    }

    /// Count stages in a given status.
    pub fn count_by_status(&self, status: &BootStageStatus) -> usize {
        self.stages.iter().filter(|s| &s.status == status).count()
    }

    /// Whether the entire boot chain is verified (no pending or failed stages).
    pub fn fully_verified(&self) -> bool {
        self.stages.iter().all(|s| {
            matches!(
                s.status,
                BootStageStatus::Verified | BootStageStatus::NotApplicable
            )
        })
    }
}

// ============================================================================
// 6. SecureBootOrchestration — composition wrapper
// ============================================================================

/// Wraps all Secure Boot-related state for composition with
/// [`crate::sovereign::InoculationOrchestrator`].
///
/// Rather than modifying sovereign.rs directly (which would risk merge
/// conflicts), this struct owns the Secure Boot subsystem and can be
/// stored alongside the orchestrator.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct SecureBootOrchestration {
    /// Result of the Secure Boot probe (SSH modes) or Socratic guide (USB Forge).
    pub probe_result: Option<SecureBootProbeResult>,
    /// Lanzaboote configuration.
    pub lanzaboote: LanzabooteConfig,
    /// Explicit human-guided Secure Boot key enrollment procedure.
    pub enrollment_guide: Option<SecureBootEnrollmentGuide>,
    /// True only after an explicit method was selected, not from probe data alone.
    pub configuration_authorized: bool,
    /// Socratic guide state (for USB Forge mode).
    pub socratic_guide: Option<SecureBootGuide>,
    /// Boot chain visualization.
    pub boot_chain: Option<BootChainStatus>,
}

impl SecureBootOrchestration {
    /// Create a new orchestration with default state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply the result of a Secure Boot probe and configure accordingly.
    pub fn apply_probe_result(&mut self, result: SecureBootProbeResult) {
        // A probe is evidence about current state, not user consent to change firmware trust.
        self.configuration_authorized = false;
        self.enrollment_guide = None;
        match &result.state {
            SecureBootState::Enabled | SecureBootState::SetupMode => {
                self.lanzaboote = LanzabooteConfig::for_secure_boot();
                self.enrollment_guide = Some(SecureBootEnrollmentGuide::new());
            }
            SecureBootState::Disabled => {
                self.lanzaboote = LanzabooteConfig::for_disabled_secure_boot();
            }
            SecureBootState::Unknown { .. } => {
                // Fail closed: key generation/config changes remain disabled.
                self.lanzaboote = LanzabooteConfig::for_disabled_secure_boot();
            }
        }
        self.boot_chain = Some(BootChainStatus::sovereign_boot_chain(&result.state, &self.lanzaboote));
        self.probe_result = Some(result);
    }

    /// Apply the result of the Socratic guide (USB Forge mode).
    pub fn apply_guide_result(&mut self, method: SecureBootMethod) {
        // Called only after the user explicitly selects a method. Probe results alone never authorize configuration generation.
        self.configuration_authorized = true;
        match method {
            SecureBootMethod::Disabled | SecureBootMethod::AlreadyOff => {
                self.lanzaboote = LanzabooteConfig::for_disabled_secure_boot();
                self.enrollment_guide = None;
                self.boot_chain = Some(BootChainStatus::sovereign_boot_chain(&SecureBootState::Disabled, &self.lanzaboote));
            }
            SecureBootMethod::MokEnrollment | SecureBootMethod::UseShim => {
                self.lanzaboote = LanzabooteConfig::for_secure_boot();
                self.enrollment_guide = Some(SecureBootEnrollmentGuide::new());
                self.boot_chain = Some(BootChainStatus::sovereign_boot_chain(&SecureBootState::SetupMode, &self.lanzaboote));
            }
        }
    }

    /// Generate the complete Nix configuration for Secure Boot support.
    ///
    /// Returns a tuple of (Lanzaboote configuration module, key-generation module).
    /// Key generation is delegated to Lanzaboote; this method never enrolls keys in firmware.
    /// For disabled Secure Boot, the second module is empty.
    pub fn generate_nix_config(&self) -> Result<(String, String), &'static str> {
        if let Some(result) = &self.probe_result {
            if matches!(result.state, SecureBootState::Unknown { .. }) {
                return Err("Secure Boot state is unknown; repeat the probe or resolve the firmware state before generating configuration");
            }
        }
        if !self.configuration_authorized {
            return Err("an explicit Secure Boot method must be selected before generating configuration");
        }
        let lanzaboote_nix = generate_lanzaboote_nix(&self.lanzaboote);
        let key_generation = if self.lanzaboote.enabled && self.lanzaboote.generate_keys {
            generate_key_enrollment_nix()
        } else {
            String::new()
        };
        Ok((lanzaboote_nix, key_generation))
    }
}

// ============================================================================
// Internal utilities
// ============================================================================

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_script_linux() {
        let script = generate_probe_script(TargetPlatform::Linux);
        assert!(
            script.contains("mokutil"),
            "Linux probe script must use mokutil"
        );
        assert!(
            script.contains("bootctl"),
            "Linux probe script should check bootctl"
        );
        assert!(
            script.contains("SecureBoot"),
            "Linux probe script should reference SecureBoot"
        );
        assert!(
            script.contains("platform"),
            "Probe output must include platform field"
        );
    }

    #[test]
    fn test_probe_script_windows() {
        let script = generate_probe_script(TargetPlatform::Windows);
        assert!(
            script.contains("Confirm-SecureBootUEFI"),
            "Windows probe script must use PowerShell cmdlet"
        );
        assert!(
            script.contains("ConvertTo-Json"),
            "Windows probe must output JSON"
        );
    }

    #[test]
    fn test_parse_probe_enabled() {
        let json = r#"{
            "secure_boot": "enabled",
            "reason": "",
            "mokutil_available": true,
            "bootctl_output": "systemd-boot installed",
            "platform": "linux"
        }"#;
        let result = parse_probe_result(json).unwrap();
        assert_eq!(result.state, SecureBootState::Enabled);
        assert!(result.mok_required);
        assert!(result.mokutil_available);
        assert_eq!(result.platform, TargetPlatform::Linux);
    }

    #[test]
    fn test_parse_probe_disabled() {
        let json = r#"{
            "secure_boot": "disabled",
            "reason": "",
            "mokutil_available": true,
            "bootctl_output": "",
            "platform": "linux"
        }"#;
        let result = parse_probe_result(json).unwrap();
        assert_eq!(result.state, SecureBootState::Disabled);
        assert!(!result.mok_required);
    }

    #[test]
    fn test_parse_probe_unknown() {
        let json = r#"{
            "secure_boot": "unknown",
            "reason": "no EFI firmware detected",
            "mokutil_available": false,
            "bootctl_output": "",
            "platform": "linux"
        }"#;
        let result = parse_probe_result(json).unwrap();
        match &result.state {
            SecureBootState::Unknown { reason } => {
                assert!(reason.contains("EFI"));
            }
            other => panic!("Expected Unknown, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_probe_setup_mode() {
        let json = r#"{
            "secure_boot": "setup_mode",
            "reason": "",
            "mokutil_available": true,
            "bootctl_output": "",
            "platform": "linux"
        }"#;
        let result = parse_probe_result(json).unwrap();
        assert_eq!(result.state, SecureBootState::SetupMode);
        assert!(!result.mok_required);
    }

    #[test]
    fn test_parse_probe_with_noise() {
        // Simulate SSH output with MOTD noise before JSON
        let output = "Welcome to Ubuntu 24.04 LTS\nLast login: Mon Mar 18\n\
            {\"secure_boot\":\"disabled\",\"reason\":\"\",\"mokutil_available\":false,\
            \"bootctl_output\":\"\",\"platform\":\"linux\"}\n";
        let result = parse_probe_result(output).unwrap();
        assert_eq!(result.state, SecureBootState::Disabled);
    }

    #[test]
    fn test_parse_probe_invalid_json() {
        let result = parse_probe_result("not json at all");
        assert!(result.is_err());
    }

    #[test]
    fn test_mok_password_generation() {
        let password = generate_mok_password();
        let parts: Vec<&str> = password.split('-').collect();

        // Must be 6 words
        assert_eq!(parts.len(), 6, "Password must be 6 words: {password}");

        // Each word must be non-empty and ASCII lowercase
        for word in &parts {
            assert!(!word.is_empty(), "Password word must not be empty");
            assert!(
                word.chars().all(|c| c.is_ascii_lowercase()),
                "Password word must be lowercase ASCII: {word}"
            );
        }

        // Must be human-readable length (not too short, not too long)
        assert!(
            password.len() >= 20,
            "Password too short: {password} ({})",
            password.len()
        );
        assert!(
            password.len() <= 60,
            "Password too long: {password} ({})",
            password.len()
        );
    }

    #[test]
    fn test_mok_password_uniqueness() {
        // Two consecutive calls should produce different passwords
        // (not guaranteed by PRNG, but extremely likely with counter mixing)
        let p1 = generate_mok_password();
        let p2 = generate_mok_password();
        assert_ne!(p1, p2, "Two consecutive password generations should differ");
    }

    #[test]
    fn test_lanzaboote_nix_generation() {
        let config = LanzabooteConfig {
            enabled: true,
            pki_dir: "/etc/secureboot".to_string(),
            generate_keys: true,
            mok_password: Some("test-pass".to_string()),
        };
        let nix = generate_lanzaboote_nix(&config);

        assert!(
            nix.contains("systemd-boot.enable = lib.mkForce false"),
            "Must disable systemd-boot: {nix}"
        );
        assert!(
            nix.contains("boot.lanzaboote"),
            "Must configure lanzaboote: {nix}"
        );
        assert!(
            nix.contains("enable = true"),
            "Must enable lanzaboote: {nix}"
        );
        assert!(
            nix.contains("pkiBundle = \"/etc/secureboot\""),
            "Must set PKI path: {nix}"
        );
        assert!(nix.contains("sbctl"), "Must include sbctl package: {nix}");
    }

    #[test]
    fn test_lanzaboote_nix_disabled() {
        let config = LanzabooteConfig::for_disabled_secure_boot();
        let nix = generate_lanzaboote_nix(&config);

        assert!(
            nix.contains("systemd-boot.enable = true"),
            "Disabled config should use standard systemd-boot: {nix}"
        );
        assert!(
            !nix.contains("lanzaboote"),
            "Disabled config should not mention lanzaboote: {nix}"
        );
    }

    #[test]
    fn test_key_enrollment_nix() {
        let nix = generate_key_enrollment_nix();

        assert!(
            nix.contains("activationScripts"),
            "Must be an activation script: {nix}"
        );
        assert!(nix.contains("sbctl"), "Must reference sbctl: {nix}");
        assert!(nix.contains("create-keys"), "Must run create-keys: {nix}");
        assert!(
            nix.contains("/etc/secureboot/GUID"),
            "Must check for existing keys: {nix}"
        );
    }

    #[test]
    fn test_mok_guide_steps() {
        let guide = MokEnrollmentGuide::new("cedar-bright-ocean-gentle-river-dawn");

        assert_eq!(guide.step_count(), 6, "MOK guide must have exactly 6 steps");

        // Verify step ordering
        for (i, step) in guide.steps.iter().enumerate() {
            assert_eq!(
                step.step_number,
                (i + 1) as u32,
                "Step {} should be numbered {}",
                i,
                i + 1
            );
        }

        // Step 1: MOK Manager entry
        let s1 = guide.step(1).unwrap();
        assert!(s1.title.contains("MOK Manager"));
        assert!(s1.instruction.contains("10 seconds"));
        assert!(s1.warning.is_some());

        // Step 2: Enroll MOK
        let s2 = guide.step(2).unwrap();
        assert!(s2.instruction.contains("Enroll MOK"));

        // Step 3: View key
        let s3 = guide.step(3).unwrap();
        assert!(s3.instruction.contains("Continue"));

        // Step 4: Confirm
        let s4 = guide.step(4).unwrap();
        assert!(s4.instruction.contains("Yes"));

        // Step 5: Password
        let s5 = guide.step(5).unwrap();
        assert!(
            s5.instruction
                .contains("cedar-bright-ocean-gentle-river-dawn"),
            "Step 5 must contain the password"
        );
        assert!(
            s5.warning.is_some(),
            "Step 5 should warn about keyboard layout"
        );

        // Step 6: Reboot
        let s6 = guide.step(6).unwrap();
        assert!(s6.instruction.contains("Reboot"));
    }

    #[test]
    fn test_boot_chain_with_secure_boot() {
        let config = LanzabooteConfig {
            enabled: true,
            pki_dir: "/etc/secureboot".to_string(),
            generate_keys: true,
            mok_password: Some("test-password".to_string()),
        };
        let chain = BootChainStatus::sovereign_boot_chain(&SecureBootState::Enabled, &config);

        assert_eq!(chain.stages.len(), 5, "Boot chain must have 5 stages");

        // Firmware is always verified
        assert_eq!(chain.stages[0].name, "Firmware");
        assert_eq!(chain.stages[0].status, BootStageStatus::Verified);

        // Shim should be pending enrollment (password present = first boot)
        assert_eq!(chain.stages[1].name, "Shim");
        assert_eq!(chain.stages[1].status, BootStageStatus::PendingEnrollment);
        assert!(chain.stages[1].signer.contains("Microsoft"));

        // Lanzaboote stages pending
        assert!(chain.stages[2].name.contains("lanzaboote"));
        assert_eq!(chain.stages[2].status, BootStageStatus::PendingEnrollment);

        // Symthaea is the last stage
        assert_eq!(chain.stages[4].name, "Symthaea");

        // Not fully verified (pending enrollment)
        assert!(!chain.fully_verified());
        assert!(chain.count_by_status(&BootStageStatus::PendingEnrollment) >= 3,);
    }

    #[test]
    fn test_boot_chain_without_secure_boot() {
        let config = LanzabooteConfig::for_disabled_secure_boot();
        let chain = BootChainStatus::sovereign_boot_chain(&SecureBootState::Disabled, &config);

        assert_eq!(chain.stages.len(), 5);

        // Everything should be Verified or NotApplicable
        assert!(
            chain.fully_verified(),
            "All stages should be verified or N/A when Secure Boot is off"
        );

        // Shim should be NotApplicable
        assert_eq!(chain.stages[1].status, BootStageStatus::NotApplicable);

        // Symthaea verified
        assert_eq!(chain.stages[4].name, "Symthaea");
        assert_eq!(chain.stages[4].status, BootStageStatus::Verified);
    }

    #[test]
    fn test_boot_chain_secure_boot_no_lanzaboote() {
        let config = LanzabooteConfig::for_disabled_secure_boot();
        let chain = BootChainStatus::sovereign_boot_chain(&SecureBootState::Enabled, &config);

        // Without lanzaboote, Secure Boot will reject the boot chain
        let failed_count = chain
            .stages
            .iter()
            .filter(|s| matches!(s.status, BootStageStatus::Failed { .. }))
            .count();
        assert!(
            failed_count >= 2,
            "Boot chain should show failures without lanzaboote: {failed_count}"
        );
    }

    #[test]
    fn test_narration_all_steps() {
        // Verify every guide step has narration text
        let steps = [
            SecureBootGuideStep::InitialQuery,
            SecureBootGuideStep::BiosDisableGuide {
                manufacturer_hint: None,
            },
            SecureBootGuideStep::BiosDisableGuide {
                manufacturer_hint: Some("Dell".to_string()),
            },
            SecureBootGuideStep::MokExplanation,
            SecureBootGuideStep::Confirmed {
                method: SecureBootMethod::Disabled,
            },
            SecureBootGuideStep::Confirmed {
                method: SecureBootMethod::MokEnrollment,
            },
            SecureBootGuideStep::Confirmed {
                method: SecureBootMethod::AlreadyOff,
            },
            SecureBootGuideStep::Confirmed {
                method: SecureBootMethod::UseShim,
            },
        ];

        for step in &steps {
            let lines = narration_for_step(step);
            assert!(!lines.is_empty(), "Step {step:?} must have narration");
            for line in &lines {
                assert!(
                    !line.is_empty(),
                    "Narration line for {step:?} must not be empty"
                );
                assert!(
                    line.len() >= 20,
                    "Narration line too short for {step:?}: {line}"
                );
            }
        }
    }

    #[test]
    fn test_narration_initial_query_content() {
        let lines = narration_for_step(&SecureBootGuideStep::InitialQuery);
        let full = lines.join(" ");
        assert!(
            full.contains("Secure Boot"),
            "Initial query must mention Secure Boot"
        );
        assert!(
            full.contains("firmware"),
            "Initial query should explain firmware"
        );
    }

    #[test]
    fn test_narration_mok_explanation_content() {
        let lines = narration_for_step(&SecureBootGuideStep::MokExplanation);
        let full = lines.join(" ");
        assert!(
            full.contains("sovereign"),
            "MOK explanation should mention sovereignty"
        );
        assert!(full.contains("key"), "MOK explanation must mention keys");
        assert!(
            full.contains("consent") || full.contains("trust"),
            "MOK explanation should frame enrollment as consent"
        );
    }

    #[test]
    fn test_narration_bios_guide_with_manufacturer() {
        let lines = narration_for_step(&SecureBootGuideStep::BiosDisableGuide {
            manufacturer_hint: Some("Lenovo".to_string()),
        });
        let full = lines.join(" ");
        assert!(
            full.contains("Lenovo"),
            "BIOS guide should include manufacturer hint"
        );
    }

    #[test]
    fn test_socratic_guide_flow_disable() {
        let mut guide = SecureBootGuide::new();
        assert_eq!(guide.current_step, SecureBootGuideStep::InitialQuery);
        assert!(!guide.user_confirmed);

        guide.choose_disable(Some("ASUS".to_string()));
        match &guide.current_step {
            SecureBootGuideStep::BiosDisableGuide { manufacturer_hint } => {
                assert_eq!(manufacturer_hint.as_deref(), Some("ASUS"));
            }
            other => panic!("Expected BiosDisableGuide, got {other:?}"),
        }

        guide.confirm().expect("an explicit Secure Boot choice was made");
        match &guide.current_step {
            SecureBootGuideStep::Confirmed { method } => {
                assert_eq!(*method, SecureBootMethod::Disabled);
            }
            other => panic!("Expected Confirmed, got {other:?}"),
        }
        assert!(guide.user_confirmed);
    }

    #[test]
    fn test_socratic_guide_rejects_confirmation_without_an_explicit_choice() {
        let mut guide = SecureBootGuide::new();
        let original_step = guide.current_step.clone();

        let result = guide.confirm();

        assert_eq!(
            result,
            Err("Secure Boot method must be explicitly selected before confirmation")
        );
        assert_eq!(guide.current_step, original_step);
        assert!(!guide.user_confirmed);
    }

    #[test]
    fn test_linux_probe_missing_efi_variable_is_unknown_not_disabled() {
        let script = generate_probe_script(TargetPlatform::Linux);
        assert!(script.contains(
            "SB_STATE=\"unknown\"\n        SB_REASON=\"SecureBoot EFI variable missing; cannot determine state\""
        ));
        assert!(!script.contains(
            "else\n        SB_STATE=\"disabled\"\n        SB_REASON=\"\"\n    fi"
        ));
    }

    #[test]
    fn test_socratic_guide_flow_mok() {
        let mut guide = SecureBootGuide::new();
        guide.choose_mok();
        assert_eq!(guide.current_step, SecureBootGuideStep::MokExplanation);

        guide.confirm().expect("an explicit Secure Boot choice was made");
        match &guide.current_step {
            SecureBootGuideStep::Confirmed { method } => {
                assert_eq!(*method, SecureBootMethod::MokEnrollment);
            }
            other => panic!("Expected Confirmed, got {other:?}"),
        }
    }

    #[test]
    fn test_socratic_guide_flow_already_off() {
        let mut guide = SecureBootGuide::new();
        guide.already_off();
        assert!(guide.user_confirmed);
        match &guide.current_step {
            SecureBootGuideStep::Confirmed { method } => {
                assert_eq!(*method, SecureBootMethod::AlreadyOff);
            }
            other => panic!("Expected Confirmed, got {other:?}"),
        }
    }

    #[test]
    fn test_orchestration_apply_probe_enabled() {
        let mut orch = SecureBootOrchestration::new();
        let result = SecureBootProbeResult {
            state: SecureBootState::Enabled,
            mok_required: true,
            mokutil_available: true,
            platform: TargetPlatform::Linux,
            raw_output: String::new(),
        };
        orch.apply_probe_result(result);

        assert!(orch.lanzaboote.enabled);
        assert!(orch.lanzaboote.enrollment_required);
        assert!(orch.mok_guide.is_some());
        assert!(orch.boot_chain.is_some());
        assert!(!orch.boot_chain.as_ref().unwrap().fully_verified());
    }

    #[test]
    fn test_orchestration_apply_probe_disabled() {
        let mut orch = SecureBootOrchestration::new();
        let result = SecureBootProbeResult {
            state: SecureBootState::Disabled,
            mok_required: false,
            mokutil_available: true,
            platform: TargetPlatform::Linux,
            raw_output: String::new(),
        };
        orch.apply_probe_result(result);

        assert!(!orch.lanzaboote.enabled);
        assert!(orch.mok_guide.is_none());
        assert!(orch.boot_chain.as_ref().unwrap().fully_verified());
    }

    #[test]
    fn test_orchestration_generate_nix() {
        let mut orch = SecureBootOrchestration::new();
        orch.apply_guide_result(SecureBootMethod::MokEnrollment);

        let (lanzaboote_nix, key_nix) = orch.generate_nix_config();
        assert!(lanzaboote_nix.contains("lanzaboote"));
        assert!(lanzaboote_nix.contains("pkiBundle = \"/etc/secureboot\""));
        assert!(key_nix.contains("boot.lanzaboote.autoGenerateKeys.enable = true"));
        assert!(!key_nix.contains("system.activationScripts.secureBootKeys"));
        assert!(!key_nix.contains("sbctl create-keys"));
    }

    #[test]
    fn test_key_generation_delegates_path_binding_to_lanzaboote() {
        let key_nix = generate_key_enrollment_nix();
        assert!(key_nix.contains("boot.lanzaboote.autoGenerateKeys.enable = true"));
        assert!(!key_nix.contains("sbctl create-keys"));
        assert!(!key_nix.contains("/etc/secureboot/GUID"));
    }

    #[test]
    fn test_orchestration_respects_disabled_key_generation() {
        let mut orch = SecureBootOrchestration::new();
        orch.lanzaboote = LanzabooteConfig {
            enabled: true,
            pki_dir: "/var/lib/sbctl".to_string(),
            generate_keys: false,
            mok_password: None,
        };

        let (lanzaboote_nix, key_nix) = orch.generate_nix_config();
        assert!(lanzaboote_nix.contains("pkiBundle = \"/var/lib/sbctl\""));
        assert!(key_nix.is_empty());
    }

    #[test]
    fn test_orchestration_generate_nix_disabled() {
        let mut orch = SecureBootOrchestration::new();
        orch.apply_guide_result(SecureBootMethod::AlreadyOff);

        let (lanzaboote_nix, key_nix) = orch.generate_nix_config();
        assert!(lanzaboote_nix.contains("systemd-boot.enable = true"));
        assert!(key_nix.is_empty());
    }

    #[test]
    fn test_target_platform_parsing() {
        assert_eq!(
            TargetPlatform::from_platform_string("Linux x86_64"),
            TargetPlatform::Linux
        );
        assert_eq!(
            TargetPlatform::from_platform_string("Win32"),
            TargetPlatform::Windows
        );
        assert_eq!(
            TargetPlatform::from_platform_string("macOS"),
            TargetPlatform::MacOS
        );
        assert_eq!(
            TargetPlatform::from_platform_string("FreeBSD"),
            TargetPlatform::Unknown
        );
    }

    #[test]
    fn test_extract_json_nested() {
        let input = r#"noise { "a": { "b": 1 } } more noise"#;
        let json = extract_json_object(input).unwrap();
        assert_eq!(json, r#"{ "a": { "b": 1 } }"#);
    }

    #[test]
    fn test_word_list_size() {
        let words = mok_word_list();
        assert!(
            words.len() >= 200,
            "Word list must have at least 200 words for sufficient entropy: {}",
            words.len()
        );
    }

    #[test]
    fn test_word_list_all_ascii_lowercase() {
        for word in mok_word_list() {
            assert!(
                word.chars().all(|c| c.is_ascii_lowercase()),
                "Word must be ASCII lowercase: {word}"
            );
        }
    }

    #[test]
    fn test_lanzaboote_config_defaults() {
        let config = LanzabooteConfig::default();
        assert!(config.enabled);
        assert_eq!(config.pki_dir, "/etc/secureboot");
        assert!(config.generate_keys);
        assert!(config.mok_password.is_none());
    }

    #[test]
    fn test_mok_guide_step_lookup() {
        let guide = MokEnrollmentGuide::new("test");
        assert!(guide.step(1).is_some());
        assert!(guide.step(6).is_some());
        assert!(guide.step(7).is_none());
        assert!(guide.step(0).is_none());
    }
}
