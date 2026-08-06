// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! wasm_bindgen exports for browser/JS consumption.
//!
//! Activated by the `wasm` feature flag.
//! Build with: `./build-wasm.sh` or see build-wasm.sh for manual steps.

use wasm_bindgen::prelude::*;

use crate::config::SporeConfig;
use crate::engine::SporeEngine as InnerEngine;

/// WASM-exported Spore consciousness engine.
#[wasm_bindgen]
pub struct SporeEngine {
    inner: InnerEngine,
}

#[wasm_bindgen]
impl SporeEngine {
    /// Create a new SporeEngine from a JSON configuration.
    /// Pass `null` or `undefined` for defaults.
    #[wasm_bindgen(constructor)]
    pub fn new(config: JsValue) -> Result<SporeEngine, JsError> {
        console_error_panic_hook::set_once();
        let config: SporeConfig = if config.is_null() || config.is_undefined() {
            SporeConfig::default()
        } else {
            serde_wasm_bindgen::from_value(config)
                .map_err(|e| JsError::new(&format!("Invalid config: {e}")))?
        };
        Ok(Self {
            inner: InnerEngine::new(config),
        })
    }

    /// Run a consciousness cycle with text input. Returns CycleResult as JS object.
    pub fn cycle(&mut self, input: &str) -> Result<JsValue, JsError> {
        let result = self.inner.cycle(input);
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Run a consciousness cycle with raw hypervector input.
    pub fn cycle_hv(&mut self, hv: &[f32]) -> Result<JsValue, JsError> {
        let result = self.inner.cycle_hv(hv);
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Current consciousness level.
    pub fn consciousness_level(&self) -> f32 {
        self.inner.consciousness_level()
    }

    /// Honest confidence in the consciousness measurement (0.0-0.95).
    pub fn honest_confidence(&self) -> f32 {
        self.inner.honest_confidence()
    }

    /// Current harmony alignment score (0.0-1.0).
    pub fn harmony_alignment(&self) -> f32 {
        self.inner.harmony_alignment()
    }

    /// Neuromodulator state as JSON string.
    pub fn neuromod_state(&self) -> String {
        self.inner.neuromod_state_json()
    }

    /// Eight Harmonies scores as JSON array [RC, PSF, IW, IP, UI, SR, EP, SS].
    pub fn harmony_scores(&self) -> String {
        self.inner.harmony_scores_json()
    }

    /// Name of the currently dominant harmony.
    pub fn dominant_harmony(&self) -> String {
        self.inner.dominant_harmony().to_string()
    }

    /// Substrate feasibility score.
    pub fn substrate_feasibility(&self) -> f32 {
        self.inner.substrate_feasibility()
    }

    /// Human-readable consciousness report with epistemic disclaimers.
    pub fn consciousness_report(&self) -> String {
        self.inner.consciousness_report()
    }

    /// Switch substrate type.
    pub fn set_substrate(&mut self, substrate: &str) {
        self.inner.set_substrate(substrate);
    }

    /// Inject neuromodulator impulse.
    pub fn inject_neuromodulator(&mut self, name: &str, amount: f32) {
        self.inner.inject_neuromodulator(name, amount);
    }

    /// Current cycle count.
    pub fn cycle_count(&self) -> u64 {
        self.inner.cycle_count()
    }

    /// Number of active SporeEngine instances globally.
    pub fn active_instance_count() -> usize {
        InnerEngine::active_instance_count()
    }

    // ======================================================================
    // Hypervector access (for visualization)
    // ======================================================================

    /// Get the current network output hypervector (16,384 f32 values).
    /// Used for live waveform visualization in the browser demo.
    pub fn get_output_hv(&self) -> Vec<f32> {
        self.inner.get_output_hv()
    }

    /// Encode text to an HDC hypervector without running a full cycle.
    /// Returns bipolar encoding as f32 values. Used for thought comparison.
    pub fn encode_text(&mut self, text: &str) -> Vec<f32> {
        self.inner.encode_text(text)
    }

    // ======================================================================
    // Language generation (Broca)
    // ======================================================================

    /// Load trained Broca checkpoint weights from a binary buffer.
    ///
    /// The checkpoint file (broca-spore-v1.bin) contains trained token embeddings,
    /// position embeddings, and gate weights. After loading, the autoregressive
    /// generation path produces coherent language at consciousness levels >= 0.15.
    pub fn load_broca_checkpoint(&mut self, data: &[u8]) -> Result<(), JsError> {
        self.inner
            .load_broca_checkpoint(data)
            .map_err(|e| JsError::new(&e))
    }

    /// Load a full Broca pipeline checkpoint (symthaea-broca format).
    ///
    /// Requires `broca-pipeline` feature. After loading, `generate_text()` uses
    /// the production HdcLtcUnifiedNetwork + epistemic gating + semantic veto
    /// pipeline instead of BrocaLite.
    ///
    /// Pass the distilled checkpoint (broca-spore-distilled.bin, ~130MB) for
    /// browser deployment, not the full v6-joint checkpoint (~976MB).
    #[cfg(feature = "broca-pipeline")]
    pub fn load_broca_pipeline_checkpoint(&mut self, data: &[u8]) -> Result<(), JsError> {
        self.inner
            .load_broca_pipeline_checkpoint(data)
            .map_err(|e| JsError::new(&e))
    }

    /// Check if the full Broca pipeline is loaded and ready.
    ///
    /// Returns false if the `broca-pipeline` feature is not enabled or no
    /// checkpoint has been loaded.
    #[cfg(feature = "broca-pipeline")]
    pub fn broca_pipeline_ready(&self) -> bool {
        self.inner.broca_pipeline_ready()
    }

    /// Generate text from current consciousness state.
    /// Returns GenerationResult as JS object with `text`, `num_tokens`, `eos_terminated`.
    pub fn generate_text(&mut self, max_tokens: usize) -> Result<JsValue, JsError> {
        let result = self.inner.generate_text(max_tokens);
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Generate text aware of user input.
    /// Injects intent signals so generated language relates to user message.
    /// Returns GenerationResult as JS object with `text`, `num_tokens`, `eos_terminated`.
    pub fn generate_text_with_input(
        &mut self,
        input: &str,
        max_tokens: usize,
    ) -> Result<JsValue, JsError> {
        let result = self.inner.generate_text_with_input(input, max_tokens);
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
    }

    // ======================================================================
    // Reasoning engine (7-step cognitive cycle)
    // ======================================================================

    /// Run a 7-step reasoning cycle on the given input.
    /// Returns the full ReasoningCycle trace as a JS object.
    pub fn reasoning_cycle(&mut self, input: &str) -> Result<JsValue, JsError> {
        let result = self.inner.reasoning_cycle(input);
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
    }

    // ======================================================================
    // Immune system (threat detection + defense cascade)
    // ======================================================================

    /// Assess an input for threats. Returns ThreatAssessment as JS object.
    pub fn threat_assessment(&mut self, input: &str) -> Result<JsValue, JsError> {
        let result = self.inner.threat_assessment(input);
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Current safety level as a string ("Green", "Yellow", "Orange", "Red").
    pub fn safety_level(&self) -> String {
        self.inner.safety_level().to_string()
    }

    // ======================================================================
    // Causal discovery
    // ======================================================================

    /// Get the current causal graph as a JS object.
    /// Returns { edges: [...], variables: [...] }.
    pub fn causal_graph(&self) -> Result<JsValue, JsError> {
        let graph = self.inner.causal_graph();
        serde_wasm_bindgen::to_value(graph).map_err(|e| JsError::new(&e.to_string()))
    }

    // ======================================================================
    // Dream engine
    // ======================================================================

    /// Run a dream cycle — simulate counterfactual alternatives.
    pub fn dream_cycle(&mut self) -> Result<JsValue, JsError> {
        let result = self.inner.dream_cycle();
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Run a dream session (multiple dream cycles).
    pub fn dream_session(&mut self, cycles: usize) -> Result<JsValue, JsError> {
        let results = self.inner.dream_session(cycles);
        serde_wasm_bindgen::to_value(&results).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Number of wisdom entries accumulated from dreaming.
    pub fn dream_wisdom_count(&self) -> usize {
        self.inner.dream_wisdom_count()
    }

    /// Dream engine statistics.
    pub fn dream_stats(&self) -> Result<JsValue, JsError> {
        let stats = self.inner.dream_stats();
        serde_wasm_bindgen::to_value(&stats).map_err(|e| JsError::new(&e.to_string()))
    }

    // ======================================================================
    // Active inference (FEP)
    // ======================================================================

    /// Current free energy value.
    pub fn free_energy(&self) -> f32 {
        self.inner.free_energy()
    }

    /// Run an explicit FEP cycle. Returns FepCycleResult.
    pub fn fep_cycle(&mut self) -> Result<JsValue, JsError> {
        let result = self.inner.fep_cycle();
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
    }

    // ======================================================================
    // Topology analysis
    // ======================================================================

    /// Analyze consciousness topology. Returns TopologyAnalysis.
    pub fn topology_analysis(&mut self) -> Result<JsValue, JsError> {
        let analysis = self.inner.topology_analysis();
        serde_wasm_bindgen::to_value(&analysis).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Human-readable topology report.
    pub fn topology_report(&mut self) -> String {
        self.inner.topology_report()
    }

    // ======================================================================
    // Memory
    // ======================================================================

    /// Memory subsystem statistics.
    pub fn memory_stats(&self) -> Result<JsValue, JsError> {
        let stats = self.inner.memory_stats();
        serde_wasm_bindgen::to_value(&stats).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Conversation context statistics: turn count, topic coherence, recent consciousness avg.
    ///
    /// Returns `{ turn_count, topic_coherence, recent_consciousness_avg }`.
    pub fn conversation_stats(&self) -> Result<JsValue, JsError> {
        let stats = self.inner.conversation_stats();
        serde_wasm_bindgen::to_value(&stats).map_err(|e| JsError::new(&e.to_string()))
    }

    // ======================================================================
    // Consciousness validation experiments
    // ======================================================================

    /// Run an anesthesia analogue experiment. Returns AnesthesiaResult as JS object.
    ///
    /// Suppresses neuromodulators, observes consciousness collapse, then restores
    /// and observes recovery. Models clinical anesthesia (propofol/sevoflurane).
    pub fn anesthesia_experiment(
        &mut self,
        warmup_cycles: usize,
        suppression_cycles: usize,
        recovery_cycles: usize,
    ) -> Result<JsValue, JsError> {
        let result =
            self.inner
                .anesthesia_experiment(warmup_cycles, suppression_cycles, recovery_cycles);
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Compute Perturbational Complexity Index (PCI). Returns PciResult as JS object.
    ///
    /// Based on Casali et al. (2013): perturb the network and measure spatiotemporal
    /// complexity of the response via Lempel-Ziv compression.
    pub fn measure_pci(
        &mut self,
        perturbation_magnitude: f32,
        observation_cycles: usize,
    ) -> Result<JsValue, JsError> {
        let result = self
            .inner
            .measure_pci(perturbation_magnitude, observation_cycles);
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Run a split-brain experiment. Returns SplitBrainResult as JS object.
    ///
    /// Partitions the network into hemispheres and measures whether splitting
    /// reduces consciousness (IIT prediction).
    pub fn split_brain_experiment(
        &mut self,
        measurement_cycles: usize,
    ) -> Result<JsValue, JsError> {
        let result = self.inner.split_brain_experiment(measurement_cycles);
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Find the consciousness collapse threshold. Returns CollapseThresholdResult as JS object.
    ///
    /// Systematically degrades the network and finds the point where consciousness
    /// collapses. IIT predicts a phase transition, not gradual decline.
    pub fn collapse_threshold_experiment(
        &mut self,
        steps: usize,
        cycles_per_step: usize,
    ) -> Result<JsValue, JsError> {
        let result = self
            .inner
            .collapse_threshold_experiment(steps, cycles_per_step);
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
    }

    // ======================================================================
    // Social Theory of Mind
    // ======================================================================

    /// Social mind state: partners, empathy level, coherence.
    pub fn social_mind_state(&self) -> Result<JsValue, JsError> {
        let state = self.inner.social_mind_state();
        serde_wasm_bindgen::to_value(&state).map_err(|e| JsError::new(&e.to_string()))
    }

    // ======================================================================
    // Knowledge Engine
    // ======================================================================

    /// Query knowledge facts about a subject. Returns QueryResult as JS object.
    pub fn knowledge_query(&mut self, subject: &str) -> Result<JsValue, JsError> {
        let result = self.inner.knowledge_query(subject);
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Knowledge engine statistics: fact count, contradiction count, etc.
    pub fn knowledge_stats(&self) -> Result<JsValue, JsError> {
        let stats = self.inner.knowledge_stats();
        serde_wasm_bindgen::to_value(&stats).map_err(|e| JsError::new(&e.to_string()))
    }

    // ======================================================================
    // Memory Consolidation
    // ======================================================================

    /// Trigger memory consolidation (dream replay). Returns ConsolidationResult.
    pub fn memory_consolidate(&mut self) -> Result<JsValue, JsError> {
        let result = self.inner.memory_consolidate();
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
    }

    // ======================================================================
    // Global Workspace
    // ======================================================================

    /// Current workspace state: broadcast, contents, history.
    pub fn workspace_state(&self) -> Result<JsValue, JsError> {
        let state = self.inner.workspace_state();
        serde_wasm_bindgen::to_value(&state).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Whether the workspace has reached ignition (conscious experience).
    pub fn workspace_ignition(&self) -> bool {
        self.inner.workspace_ignition()
    }
}

// ======================================================================
// Standalone WASM exports: NixOS flake generation from hardware probe
// ======================================================================

/// Generate a complete NixOS flake.nix from hardware probe results.
///
/// Takes the hardware profile JSON from the browser probe and returns
/// a full flake.nix string suitable for `nixos-install`.
#[wasm_bindgen]
pub fn generate_flake(hardware_json: &str, path: &str, hostname: &str) -> Result<String, JsError> {
    let _ = path; // reserved for future mount-point customization
    let profile: crate::hardware_probe::HardwareProfile =
        serde_json::from_str(hardware_json).map_err(|e| JsError::new(&e.to_string()))?;
    let nix_config = crate::hardware_probe::NixHardwareConfig::from_profile(&profile);

    let flake = format!(
        r#"{{
  description = "Symthaea Guardian Node — {hostname}";

  inputs = {{
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    disko.url = "github:nix-community/disko";
    disko.inputs.nixpkgs.follows = "nixpkgs";
  }};

  outputs = {{ self, nixpkgs, disko, ... }}: {{
    nixosConfigurations.{hostname} = nixpkgs.lib.nixosSystem {{
      system = "x86_64-linux";
      modules = [
        disko.nixosModules.disko
        ./disko-config.nix
        ./hardware-configuration.nix
        ({{ pkgs, ... }}: {{
          networking.hostName = "{hostname}";

          # ── Boot ──────────────────────────────────────────────────
          boot.loader.systemd-boot.enable = true;
          boot.loader.efi.canTouchEfiVariables = true;

          # ── Symthaea Runtime ──────────────────────────────────────
          environment.systemPackages = with pkgs; [
            git
            curl
            htop
            neovim
          ];

          # ── Consciousness Parameters ──────────────────────────────
          # HDC Dimension: {hdc_dim}
          # Recommended Neurons: {neurons}
          # Substrate: {substrate}
          # Filesystem: {filesystem}

          # ── Networking ────────────────────────────────────────────
          networking.firewall.enable = true;
          networking.firewall.allowedTCPPorts = [ 22 ];

          # ── Users ─────────────────────────────────────────────────
          users.users.guardian = {{
            isNormalUser = true;
            extraGroups = [ "wheel" "networkmanager" ];
            openssh.authorizedKeys.keys = [
              # Add your SSH public key here
            ];
          }};

          services.openssh.enable = true;

          system.stateVersion = "24.11";
        }})
      ];
    }};
  }};
}}"#,
        hostname = hostname,
        hdc_dim = nix_config.recommended_hdc_dim,
        neurons = nix_config.recommended_neurons,
        substrate = nix_config.recommended_substrate,
        filesystem = nix_config.filesystem,
    );

    Ok(flake)
}

/// Generate disko disk configuration from hardware probe.
///
/// If the hardware profile suggests a workstation (>500GB storage, 8+ cores),
/// generates a dual-disk layout comment suggesting the optimized profile.
#[wasm_bindgen]
pub fn generate_disko_config(hardware_json: &str) -> Result<String, JsError> {
    let profile: crate::hardware_probe::HardwareProfile =
        serde_json::from_str(hardware_json).map_err(|e| JsError::new(&e.to_string()))?;
    let nix_config = crate::hardware_probe::NixHardwareConfig::from_profile(&profile);

    // Detect if dual-disk might be appropriate
    let dual_disk_hint = if profile.cpu_cores >= 8 && profile.storage_quota_bytes > 500_000_000_000
    {
        r#"# NOTE: Your system appears to be a workstation with multiple drives.
# Consider the dual-NVMe layout for optimal performance:
#   Fast drive (TLC) → btrfs data (@home, @srv, @swap, @snapshots)
#   Standard drive (QLC) → ext4 root (OS, nix store, build caches)
# See: symthaea/nix/disko-dual-nvme.nix

"#
    } else {
        ""
    };

    let disko = format!(
        r#"{dual_disk_hint}{{ ... }}: {{
  disko.devices.disk.main = {{
    type = "disk";
    device = "/dev/nvme0n1";  # ← Verify your target disk
    content = {{
      type = "gpt";
      partitions = {{
        boot = {{
          size = "1G";
          type = "EF00";
          content = {{
            type = "filesystem";
            format = "vfat";
            mountpoint = "/boot";
            mountOptions = [ "fmask=0077" "dmask=0077" ];
          }};
        }};
        root = {{
          size = "100%";
          content = {{
            type = "luks";
            name = "symthaea-root";
            settings.allowDiscards = true;
            content = {{
              type = "btrfs";
              extraArgs = [ "-L" "symthaea" "-f" ];
              subvolumes = {{
                "@" = {{
                  mountpoint = "/";
                  mountOptions = [ "compress=zstd:3" "noatime" ];
                }};
                "@home" = {{
                  mountpoint = "/home";
                  mountOptions = [ "compress=zstd:3" "noatime" ];
                }};
                "@nix" = {{
                  mountpoint = "/nix";
                  mountOptions = [ "compress=zstd:3" "noatime" ];
                }};
                "@log" = {{
                  mountpoint = "/var/log";
                  mountOptions = [ "compress=zstd:3" "noatime" ];
                }};
                "@snapshots" = {{
                  mountpoint = "/.snapshots";
                  mountOptions = [ "compress=zstd:3" "noatime" ];
                }};
                "@swap" = {{
                  mountpoint = "/swap";
                  mountOptions = [ "noatime" ];
                }};
              }};
            }};
          }};
        }};
      }};
    }};
  }};

  swapDevices = [{{ device = "/swap/swapfile"; size = {swap_mb}; }}];
}}"#,
        dual_disk_hint = dual_disk_hint,
        swap_mb = nix_config.swap_size_gb as u32 * 1024,
    );

    Ok(disko)
}

/// Generate hardware-configuration.nix from probe results.
#[wasm_bindgen]
pub fn generate_hardware_nix(hardware_json: &str) -> Result<String, JsError> {
    let profile: crate::hardware_probe::HardwareProfile =
        serde_json::from_str(hardware_json).map_err(|e| JsError::new(&e.to_string()))?;
    let nix_config = crate::hardware_probe::NixHardwareConfig::from_profile(&profile);
    Ok(nix_config.nix_hardware_config.clone())
}

// ═══════════════════════════════════════════════════════
// Sovereign NixOS Configuration (AI-powered, WASM-safe)
// ═══════════════════════════════════════════════════════

use std::cell::RefCell;

thread_local! {
    static SOVEREIGN_CONVERSATION: RefCell<Option<nixward::sovereign_conversation::SovereignConversation>> = RefCell::new(None);
}

/// Initialize a sovereign conversation session.
/// Takes hardware profile and migration data as JSON.
/// Call this once, then use sovereign_chat() for each message.
#[wasm_bindgen]
pub fn sovereign_init(hardware_json: &str, migration_json: &str) -> Result<JsValue, JsError> {
    let hardware: nixward::sovereign_config::HardwareProfile =
        serde_json::from_str(hardware_json).unwrap_or_default();
    let migration: nixward::sovereign_config::MigrationData =
        serde_json::from_str(migration_json).unwrap_or_default();

    let mut conv = nixward::sovereign_conversation::SovereignConversation::new(hardware, migration);
    let greeting = conv.greet();

    SOVEREIGN_CONVERSATION.with(|c| {
        *c.borrow_mut() = Some(conv);
    });

    // Return greeting as JSON string (avoids serde_wasm_bindgen issues with static refs)
    let json = serde_json::json!({
        "message": greeting.message,
        "config_preview": greeting.config_preview,
    });
    serde_wasm_bindgen::to_value(&json).map_err(|e| JsError::new(&e.to_string()))
}

/// Send a message to the sovereign conversation.
/// Returns the AI response as JSON with message and config preview.
#[wasm_bindgen]
pub fn sovereign_chat(message: &str) -> Result<JsValue, JsError> {
    SOVEREIGN_CONVERSATION.with(|c| {
        let mut borrow = c.borrow_mut();
        let conv = borrow.as_mut().ok_or_else(|| {
            JsError::new("Conversation not initialized. Call sovereign_init() first.")
        })?;
        let response = conv.respond(message);
        let json = serde_json::json!({
            "message": response.message,
            "config_preview": response.config_preview,
        });
        serde_wasm_bindgen::to_value(&json).map_err(|e| JsError::new(&e.to_string()))
    })
}

/// Generate a full NixOS configuration from hardware profile and user choices.
/// Standalone function — doesn't require a conversation session.
#[wasm_bindgen]
pub fn generate_sovereign_config(
    hardware_json: &str,
    choices_json: &str,
    migration_json: &str,
) -> Result<JsValue, JsError> {
    let hardware: nixward::sovereign_config::HardwareProfile =
        serde_json::from_str(hardware_json).unwrap_or_default();
    let choices: nixward::sovereign_config::UserChoices =
        serde_json::from_str(choices_json).unwrap_or_default();
    let migration: nixward::sovereign_config::MigrationData =
        serde_json::from_str(migration_json).unwrap_or_default();

    let mut b_gen = nixward::sovereign_config::SovereignConfigGenerator::new();
    let config = b_gen.generate(&hardware, &choices, &migration);

    let json = serde_json::json!({
        "configuration_nix": config.configuration_nix,
        "sovereign_config_nix": config.sovereign_config_nix,
        "flake_nix": config.flake_nix,
        "home_nix": config.home_nix,
        "decisions": config.decisions,
        "welcome_message": config.welcome_message,
        "warnings": config.warnings,
    });
    serde_wasm_bindgen::to_value(&json).map_err(|e| JsError::new(&e.to_string()))
}

/// Match a list of app names against the NixOS package database.
/// Input: newline-separated app names (from winget list, brew list, dpkg -l, etc.)
/// Returns: JSON migration report with matches, confidence, and NixOS equivalents.
#[wasm_bindgen]
pub fn match_app_list(app_list_text: &str) -> Result<JsValue, JsError> {
    let mut db = nixward::app_database::AppDatabase::new();
    let names: Vec<String> = app_list_text
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();
    let report = db.match_list(&names);
    // Manual JSON construction to handle &'static references
    let matched: Vec<serde_json::Value> = report
        .matched
        .iter()
        .map(|m| {
            serde_json::json!({
                "source_name": m.source_name,
                "nix_package": m.entry.name,
                "category": format!("{:?}", m.entry.category),
            })
        })
        .collect();
    let bundles: Vec<&str> = report.suggested_bundles.iter().map(|b| b.name).collect();
    let json = serde_json::json!({
        "total_apps": report.total_apps,
        "matched": matched,
        "unmatched": report.unmatched,
        "readiness_score": report.readiness_score,
        "suggested_bundles": bundles,
        "summary": report.summary,
    });
    serde_wasm_bindgen::to_value(&json).map_err(|e| JsError::new(&e.to_string()))
}
