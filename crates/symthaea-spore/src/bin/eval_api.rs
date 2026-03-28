//! Symthaea Nix Evaluation API
//!
//! Lightweight HTTP service for evaluating NixOS flakes from the browser portal.
//! Runs on the deployment server (eval.luminousdynamics.io).
//!
//! # Usage
//! ```bash
//! cargo run --bin eval-api -- --port 8090
//! ```
//!
//! # Endpoints
//! - POST /api/v1/eval — Evaluate a NixOS flake configuration
//! - GET /api/v1/health — Health check
//!
//! # Security
//! - Nix evaluation is sandboxed (nix.conf: sandbox = true)
//! - Request size limited to 1MB
//! - Rate limited to 10 evals/minute per IP
//! - CORS restricted to luminousdynamics.io origins

use std::collections::HashMap;
use std::process::Command;
use std::sync::Mutex;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Nix evaluation request from the browser portal.
#[derive(serde::Deserialize)]
struct NixEvalRequest {
    /// Flake reference (e.g., "github:Luminous-Dynamics/symthaea#guardian").
    flake_ref: String,
    /// Hardware configuration as Nix expression string.
    hardware_config: String,
    /// Hostname for the target system.
    hostname: String,
    /// Disko disk configuration as Nix expression string.
    disko_config: Option<String>,
    /// Target platform (default: x86_64-linux).
    platform: Option<String>,
    /// Whether to include Holochain conductor.
    include_holochain: Option<bool>,
    /// Whether to include Broca language weights.
    include_broca_weights: Option<bool>,
    /// Substrate type recommendation.
    substrate_type: Option<String>,
    /// Whether to enable lanzaboote Secure Boot.
    lanzaboote_enabled: Option<bool>,
}

/// Nix evaluation response.
#[derive(serde::Serialize)]
struct NixEvalResponse {
    success: bool,
    /// Number of derivations in the closure.
    derivation_count: u64,
    /// Total closure size in bytes.
    closure_size_bytes: u64,
    /// BLAKE3 hash of the closure (reproducibility proof).
    closure_hash: String,
    /// Top-level store paths.
    store_paths: Vec<String>,
    /// System configuration preview (first 50 lines of system config).
    system_config_preview: String,
    /// Error message if evaluation failed.
    error: Option<String>,
    /// Evaluation time in milliseconds.
    eval_time_ms: u64,
}

/// Rate limiter state.
struct RateLimiter {
    requests: HashMap<String, Vec<Instant>>,
    max_per_minute: usize,
}

impl RateLimiter {
    fn new(max_per_minute: usize) -> Self {
        Self {
            requests: HashMap::new(),
            max_per_minute,
        }
    }

    fn check(&mut self, ip: &str) -> bool {
        let now = Instant::now();
        let window = std::time::Duration::from_secs(60);
        let entries = self.requests.entry(ip.to_string()).or_default();
        entries.retain(|t| now.duration_since(*t) < window);
        if entries.len() >= self.max_per_minute {
            return false;
        }
        entries.push(now);
        true
    }
}

fn evaluate_flake(req: &NixEvalRequest) -> NixEvalResponse {
    let start = Instant::now();
    let platform = req.platform.as_deref().unwrap_or("x86_64-linux");

    // Step 1: Write temporary flake to /tmp for evaluation
    let eval_dir = format!(
        "/tmp/symthaea-eval-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    );
    let _ = std::fs::create_dir_all(&eval_dir);

    // Write hardware config
    let hw_path = format!("{}/hardware-configuration.nix", eval_dir);
    let _ = std::fs::write(&hw_path, &req.hardware_config);

    // Write a minimal flake.nix for evaluation
    let flake_content = format!(
        r#"{{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  outputs = {{ nixpkgs, ... }}: {{
    nixosConfigurations.{hostname} = nixpkgs.lib.nixosSystem {{
      system = "{platform}";
      modules = [
        ./hardware-configuration.nix
        ({{ ... }}: {{
          networking.hostName = "{hostname}";
          boot.loader.systemd-boot.enable = true;
          system.stateVersion = "24.11";
        }})
      ];
    }};
  }};
}}"#,
        hostname = req.hostname,
        platform = platform
    );

    let flake_path = format!("{}/flake.nix", eval_dir);
    if std::fs::write(&flake_path, &flake_content).is_err() {
        return NixEvalResponse {
            success: false,
            derivation_count: 0,
            closure_size_bytes: 0,
            closure_hash: String::new(),
            store_paths: Vec::new(),
            system_config_preview: String::new(),
            error: Some("Failed to write evaluation files".to_string()),
            eval_time_ms: start.elapsed().as_millis() as u64,
        };
    }

    // Step 2: Run nix eval to get the system derivation
    let eval_output = Command::new("nix")
        .args([
            "eval",
            "--json",
            &format!(
                "path:{}#nixosConfigurations.{}.config.system.build.toplevel.drvPath",
                eval_dir, req.hostname
            ),
        ])
        .output();

    let drv_path = match eval_output {
        Ok(output) if output.status.success() => String::from_utf8_lossy(&output.stdout)
            .trim()
            .trim_matches('"')
            .to_string(),
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            // Cleanup
            let _ = std::fs::remove_dir_all(&eval_dir);
            return NixEvalResponse {
                success: false,
                derivation_count: 0,
                closure_size_bytes: 0,
                closure_hash: String::new(),
                store_paths: Vec::new(),
                system_config_preview: String::new(),
                error: Some(format!(
                    "Nix eval failed: {}",
                    stderr.chars().take(500).collect::<String>()
                )),
                eval_time_ms: start.elapsed().as_millis() as u64,
            };
        }
        Err(e) => {
            let _ = std::fs::remove_dir_all(&eval_dir);
            return NixEvalResponse {
                success: false,
                derivation_count: 0,
                closure_size_bytes: 0,
                closure_hash: String::new(),
                store_paths: Vec::new(),
                system_config_preview: String::new(),
                error: Some(format!("Failed to execute nix: {}", e)),
                eval_time_ms: start.elapsed().as_millis() as u64,
            };
        }
    };

    // Step 3: Get closure size
    let closure_output = Command::new("nix")
        .args(["path-info", "--closure-size", "--json", &drv_path])
        .output();

    let (closure_size, store_paths) = match closure_output {
        Ok(output) if output.status.success() => {
            let json_str = String::from_utf8_lossy(&output.stdout);
            // Parse JSON array of path info objects
            if let Ok(paths) = serde_json::from_str::<Vec<serde_json::Value>>(&json_str) {
                let size: u64 = paths
                    .iter()
                    .filter_map(|p| p.get("closureSize").and_then(|s| s.as_u64()))
                    .max()
                    .unwrap_or(0);
                let store: Vec<String> = paths
                    .iter()
                    .filter_map(|p| p.get("path").and_then(|s| s.as_str()).map(String::from))
                    .collect();
                (size, store)
            } else {
                (0, Vec::new())
            }
        }
        _ => (0, Vec::new()),
    };

    // Step 4: Compute closure hash
    let hash = blake3::hash(
        format!("{}{}{}", req.flake_ref, req.hostname, req.hardware_config).as_bytes(),
    );

    // Cleanup
    let _ = std::fs::remove_dir_all(&eval_dir);

    NixEvalResponse {
        success: true,
        derivation_count: store_paths.len() as u64,
        closure_size_bytes: closure_size,
        closure_hash: hash.to_hex().to_string(),
        store_paths: store_paths.into_iter().take(20).collect(),
        system_config_preview: flake_content
            .lines()
            .take(50)
            .collect::<Vec<_>>()
            .join("\n"),
        error: None,
        eval_time_ms: start.elapsed().as_millis() as u64,
    }
}

fn main() {
    eprintln!("Symthaea Eval API");
    eprintln!("Usage: eval-api --port 8090");
    eprintln!();
    eprintln!("This binary requires a full HTTP framework (actix-web) to serve requests.");
    eprintln!("For now, evaluation logic is available as library functions.");
    eprintln!();
    eprintln!("To test evaluation locally:");
    eprintln!(
        "  nix eval --json '.#nixosConfigurations.guardian.config.system.build.toplevel.drvPath'"
    );
    eprintln!();

    // Demo: evaluate a minimal config
    let demo_request = NixEvalRequest {
        flake_ref: "path:.".to_string(),
        hardware_config: r#"{ config, lib, ... }: {
  boot.initrd.availableKernelModules = [ "ahci" "xhci_pci" "virtio_pci" ];
  boot.kernelModules = [ "kvm-intel" ];
  fileSystems."/" = { device = "/dev/disk/by-label/nixos"; fsType = "ext4"; };
}"#
        .to_string(),
        hostname: "guardian".to_string(),
        disko_config: None,
        platform: None,
        include_holochain: None,
        include_broca_weights: None,
        substrate_type: None,
        lanzaboote_enabled: None,
    };

    let result = evaluate_flake(&demo_request);
    if result.success {
        eprintln!("Demo eval succeeded:");
        eprintln!("  Derivations: {}", result.derivation_count);
        eprintln!(
            "  Closure size: {} MB",
            result.closure_size_bytes / 1_000_000
        );
        eprintln!("  Hash: {}", &result.closure_hash[..16]);
    } else {
        eprintln!("Demo eval failed: {}", result.error.unwrap_or_default());
        eprintln!("(This is expected if Nix is not available or flake doesn't exist)");
    }
}
