// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Nix Store Garbage Collection Manager
//!
//! Intelligent garbage collection with space analysis before executing.
//! Reports how much space will be freed, what roots are holding paths,
//! and recommends retention policies.
//!
//! All destructive operations produce `NixOSCommand` values routed
//! through the Φ-gated executor.

use super::executor::NixOSCommand;
use std::process::Command;

/// Manages Nix store garbage collection with intelligent retention policies.
pub struct GcManager;

/// Space analysis before garbage collection.
#[derive(Debug, Clone)]
pub struct GcAnalysis {
    /// Total store size in bytes.
    pub total_store_bytes: u64,
    /// Estimated reclaimable space in bytes.
    pub reclaimable_bytes: u64,
    /// Number of store paths that would be removed.
    pub dead_path_count: usize,
    /// Number of live roots keeping paths alive.
    pub live_root_count: usize,
    /// Number of generations across all profiles.
    pub total_generations: usize,
}

impl GcAnalysis {
    /// Format store size as human-readable.
    pub fn total_store_human(&self) -> String {
        Self::human_bytes(self.total_store_bytes)
    }

    /// Format reclaimable space as human-readable.
    pub fn reclaimable_human(&self) -> String {
        Self::human_bytes(self.reclaimable_bytes)
    }

    /// Percentage of store that is reclaimable.
    pub fn reclaimable_percent(&self) -> f64 {
        if self.total_store_bytes == 0 {
            return 0.0;
        }
        (self.reclaimable_bytes as f64 / self.total_store_bytes as f64) * 100.0
    }

    fn human_bytes(bytes: u64) -> String {
        const KIB: u64 = 1024;
        const MIB: u64 = KIB * 1024;
        const GIB: u64 = MIB * 1024;

        if bytes >= GIB {
            format!("{:.1} GiB", bytes as f64 / GIB as f64)
        } else if bytes >= MIB {
            format!("{:.1} MiB", bytes as f64 / MIB as f64)
        } else if bytes >= KIB {
            format!("{:.1} KiB", bytes as f64 / KIB as f64)
        } else {
            format!("{bytes} B")
        }
    }
}

/// Recommendation for garbage collection.
#[derive(Debug, Clone)]
pub struct GcRecommendation {
    /// Whether GC is recommended.
    pub recommended: bool,
    /// Reason for the recommendation.
    pub reason: String,
    /// Suggested command.
    pub command: NixOSCommand,
    /// Estimated space to be freed.
    pub estimated_savings: String,
}

impl GcManager {
    /// Analyze the store to estimate reclaimable space.
    ///
    /// This is a read-only operation — it does NOT delete anything.
    pub fn analyze() -> Result<GcAnalysis, std::io::Error> {
        // Get store size
        let store_output = Command::new("nix")
            .args(["path-info", "--all", "-S", "--json"])
            .output();

        let (total_bytes, path_count) = match store_output {
            Ok(output) if output.status.success() => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                Self::parse_store_info(&stdout)
            }
            _ => {
                // Fallback: use du
                let du = Command::new("du").args(["-sb", "/nix/store"]).output()?;
                let stdout = String::from_utf8_lossy(&du.stdout);
                let bytes = stdout
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(0);
                (bytes, 0)
            }
        };

        // Count dead paths (unreferenced)
        let dead_output = Command::new("nix-store")
            .args(["--gc", "--print-dead"])
            .output();

        let (dead_count, reclaimable) = match dead_output {
            Ok(output) if output.status.success() => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let count = stdout.lines().count();
                // Rough estimate: dead paths are ~average size
                let avg_path_size = if path_count > 0 {
                    total_bytes / path_count as u64
                } else {
                    50 * 1024 * 1024 // 50 MiB default estimate
                };
                (count, count as u64 * avg_path_size)
            }
            _ => (0, 0),
        };

        // Count GC roots
        let roots_output = Command::new("nix-store")
            .args(["--gc", "--print-roots"])
            .output();

        let root_count = match roots_output {
            Ok(output) if output.status.success() => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                stdout.lines().count()
            }
            _ => 0,
        };

        // Count generations
        let gen_count = Command::new("nix-env")
            .args(["--list-generations", "-p", "/nix/var/nix/profiles/system"])
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).lines().count())
            .unwrap_or(0);

        Ok(GcAnalysis {
            total_store_bytes: total_bytes,
            reclaimable_bytes: reclaimable,
            dead_path_count: dead_count,
            live_root_count: root_count,
            total_generations: gen_count,
        })
    }

    /// Generate a command for standard GC (delete unreferenced paths).
    pub fn collect() -> NixOSCommand {
        NixOSCommand::CollectGarbage {
            older_than_days: None,
            delete_all: false,
        }
    }

    /// Generate a command for GC with age-based retention.
    pub fn collect_older_than(days: u32) -> NixOSCommand {
        NixOSCommand::CollectGarbage {
            older_than_days: Some(days),
            delete_all: false,
        }
    }

    /// Generate a command for aggressive GC (delete all old generations + GC).
    pub fn collect_aggressive() -> NixOSCommand {
        NixOSCommand::CollectGarbage {
            older_than_days: None,
            delete_all: true,
        }
    }

    /// Generate a recommendation based on store analysis.
    pub fn recommend(analysis: &GcAnalysis) -> GcRecommendation {
        let reclaimable_gib = analysis.reclaimable_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

        if reclaimable_gib > 10.0 {
            GcRecommendation {
                recommended: true,
                reason: format!(
                    "Store has {} reclaimable ({:.0}% of total). {} dead paths found.",
                    analysis.reclaimable_human(),
                    analysis.reclaimable_percent(),
                    analysis.dead_path_count,
                ),
                command: Self::collect_older_than(30),
                estimated_savings: analysis.reclaimable_human(),
            }
        } else if reclaimable_gib > 2.0 {
            GcRecommendation {
                recommended: true,
                reason: format!(
                    "Store has {} reclaimable. Consider cleaning up.",
                    analysis.reclaimable_human(),
                ),
                command: Self::collect_older_than(14),
                estimated_savings: analysis.reclaimable_human(),
            }
        } else if analysis.total_generations > 20 {
            GcRecommendation {
                recommended: true,
                reason: format!(
                    "{} generations found. Consider trimming old generations.",
                    analysis.total_generations,
                ),
                command: Self::collect_older_than(30),
                estimated_savings: "varies".to_string(),
            }
        } else {
            GcRecommendation {
                recommended: false,
                reason: format!(
                    "Store is healthy. {} total, {} reclaimable.",
                    analysis.total_store_human(),
                    analysis.reclaimable_human(),
                ),
                command: Self::collect(),
                estimated_savings: analysis.reclaimable_human(),
            }
        }
    }

    /// Parse `nix path-info --all -S --json` to get total size.
    fn parse_store_info(json: &str) -> (u64, usize) {
        // The JSON is an array of objects with "closureSize" fields,
        // or a single object mapping paths to info.
        // We'll try to sum closureSize / narSize.
        let value: Result<serde_json::Value, _> = serde_json::from_str(json);
        match value {
            Ok(serde_json::Value::Array(arr)) => {
                let total: u64 = arr.iter().filter_map(|v| v["narSize"].as_u64()).sum();
                (total, arr.len())
            }
            Ok(serde_json::Value::Object(map)) => {
                let total: u64 = map.values().filter_map(|v| v["narSize"].as_u64()).sum();
                (total, map.len())
            }
            _ => (0, 0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::executor::SafetyLevel;

    #[test]
    fn test_human_bytes() {
        assert_eq!(GcAnalysis::human_bytes(500), "500 B");
        assert_eq!(GcAnalysis::human_bytes(2048), "2.0 KiB");
        assert_eq!(GcAnalysis::human_bytes(5 * 1024 * 1024), "5.0 MiB");
        assert_eq!(GcAnalysis::human_bytes(3 * 1024 * 1024 * 1024), "3.0 GiB");
    }

    #[test]
    fn test_reclaimable_percent() {
        let analysis = GcAnalysis {
            total_store_bytes: 1000,
            reclaimable_bytes: 250,
            dead_path_count: 10,
            live_root_count: 5,
            total_generations: 3,
        };
        assert!((analysis.reclaimable_percent() - 25.0).abs() < 0.01);
    }

    #[test]
    fn test_reclaimable_percent_zero() {
        let analysis = GcAnalysis {
            total_store_bytes: 0,
            reclaimable_bytes: 0,
            dead_path_count: 0,
            live_root_count: 0,
            total_generations: 0,
        };
        assert!((analysis.reclaimable_percent() - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_collect_command() {
        let cmd = GcManager::collect();
        let (bin, _args) = cmd.to_command();
        assert_eq!(bin, "nix-collect-garbage");
        assert_eq!(cmd.safety_level(), SafetyLevel::Destructive);
    }

    #[test]
    fn test_collect_older_than_command() {
        let cmd = GcManager::collect_older_than(30);
        let (bin, args) = cmd.to_command();
        assert_eq!(bin, "nix-collect-garbage");
        assert!(args.contains(&"30d".to_string()));
    }

    #[test]
    fn test_collect_aggressive_command() {
        let cmd = GcManager::collect_aggressive();
        let (bin, args) = cmd.to_command();
        assert_eq!(bin, "nix-collect-garbage");
        assert!(args.contains(&"--delete-old".to_string()));
    }

    #[test]
    fn test_recommend_large_store() {
        let analysis = GcAnalysis {
            total_store_bytes: 50 * 1024 * 1024 * 1024, // 50 GiB
            reclaimable_bytes: 20 * 1024 * 1024 * 1024, // 20 GiB
            dead_path_count: 5000,
            live_root_count: 50,
            total_generations: 10,
        };
        let rec = GcManager::recommend(&analysis);
        assert!(rec.recommended);
        assert!(rec.reason.contains("reclaimable"));
    }

    #[test]
    fn test_recommend_healthy_store() {
        let analysis = GcAnalysis {
            total_store_bytes: 10 * 1024 * 1024 * 1024, // 10 GiB
            reclaimable_bytes: 500 * 1024 * 1024,       // 500 MiB
            dead_path_count: 50,
            live_root_count: 20,
            total_generations: 5,
        };
        let rec = GcManager::recommend(&analysis);
        assert!(!rec.recommended);
        assert!(rec.reason.contains("healthy"));
    }

    #[test]
    fn test_recommend_many_generations() {
        let analysis = GcAnalysis {
            total_store_bytes: 10 * 1024 * 1024 * 1024,
            reclaimable_bytes: 500 * 1024 * 1024,
            dead_path_count: 50,
            live_root_count: 20,
            total_generations: 30,
        };
        let rec = GcManager::recommend(&analysis);
        assert!(rec.recommended);
        assert!(rec.reason.contains("generations"));
    }

    #[test]
    fn test_parse_store_info_array() {
        let json = r#"[
            {"path": "/nix/store/abc", "narSize": 1000},
            {"path": "/nix/store/def", "narSize": 2000}
        ]"#;
        let (total, count) = GcManager::parse_store_info(json);
        assert_eq!(total, 3000);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_parse_store_info_object() {
        let json = r#"{
            "/nix/store/abc": {"narSize": 1000},
            "/nix/store/def": {"narSize": 2000}
        }"#;
        let (total, count) = GcManager::parse_store_info(json);
        assert_eq!(total, 3000);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_parse_store_info_invalid_json() {
        let (total, count) = GcManager::parse_store_info("not json");
        assert_eq!(total, 0);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_parse_store_info_empty_array() {
        let (total, count) = GcManager::parse_store_info("[]");
        assert_eq!(total, 0);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_parse_store_info_missing_nar_size() {
        let json = r#"[{"path": "/nix/store/abc"}, {"path": "/nix/store/def", "narSize": 500}]"#;
        let (total, count) = GcManager::parse_store_info(json);
        assert_eq!(total, 500);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_human_bytes_boundaries() {
        assert_eq!(GcAnalysis::human_bytes(1024), "1.0 KiB");
        assert_eq!(GcAnalysis::human_bytes(1024 * 1024 - 1), "1024.0 KiB");
        assert_eq!(GcAnalysis::human_bytes(1024 * 1024), "1.0 MiB");
        assert_eq!(GcAnalysis::human_bytes(0), "0 B");
        assert_eq!(GcAnalysis::human_bytes(1), "1 B");
    }

    #[test]
    fn test_gc_analysis_accessors() {
        let analysis = GcAnalysis {
            total_store_bytes: 10 * 1024 * 1024 * 1024,
            reclaimable_bytes: 5 * 1024 * 1024 * 1024,
            dead_path_count: 100,
            live_root_count: 50,
            total_generations: 10,
        };
        assert_eq!(analysis.total_store_human(), "10.0 GiB");
        assert_eq!(analysis.reclaimable_human(), "5.0 GiB");
        assert!((analysis.reclaimable_percent() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_recommend_medium_store() {
        let analysis = GcAnalysis {
            total_store_bytes: 30 * 1024 * 1024 * 1024,
            reclaimable_bytes: 5 * 1024 * 1024 * 1024,
            dead_path_count: 200,
            live_root_count: 30,
            total_generations: 8,
        };
        let rec = GcManager::recommend(&analysis);
        assert!(rec.recommended);
        assert!(rec.reason.contains("reclaimable"));
    }
}
