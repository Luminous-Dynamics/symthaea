// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! I3: Home-Manager Bridge
//!
//! Integration with home-manager for user configuration.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Home-manager configuration entry
#[derive(Debug, Clone)]
pub struct HomeConfig {
    /// User name
    pub user: String,
    /// Configuration file path
    pub config_path: PathBuf,
    /// Home directory
    pub home_dir: PathBuf,
    /// Current generation
    pub generation: Option<u32>,
    /// Last switch time
    pub last_switch: Option<String>,
}

/// Home-manager program entry
#[derive(Debug, Clone)]
pub struct HomeProgram {
    /// Program name
    pub name: String,
    /// Whether it's enabled
    pub enabled: bool,
    /// Configuration options
    pub options: HashMap<String, String>,
}

/// Home-manager service entry
#[derive(Debug, Clone)]
pub struct HomeService {
    /// Service name
    pub name: String,
    /// Whether it's enabled
    pub enabled: bool,
    /// Unit file path
    pub unit_path: Option<PathBuf>,
}

/// Home-manager generation
#[derive(Debug, Clone)]
pub struct HomeGeneration {
    /// Generation number
    pub id: u32,
    /// Creation timestamp
    pub timestamp: String,
    /// Description/notes
    pub description: Option<String>,
    /// Profile path
    pub path: PathBuf,
}

/// Result of home-manager operation
#[derive(Debug, Clone)]
pub struct HomeResult {
    /// Success
    pub success: bool,
    /// Output message
    pub output: String,
    /// Error message if failed
    pub error: Option<String>,
    /// New generation if switched
    pub new_generation: Option<u32>,
}

/// Home-manager bridge
pub struct HomeManagerBridge {
    /// User configurations
    configs: HashMap<String, HomeConfig>,
    /// Detected programs (reserved for program management UI)
    #[allow(dead_code)]
    programs: Vec<HomeProgram>,
    /// Detected services (reserved for service management UI)
    #[allow(dead_code)]
    services: Vec<HomeService>,
    /// Generation history
    generations: Vec<HomeGeneration>,
    /// Current user
    current_user: String,
}

impl HomeManagerBridge {
    /// Create new home-manager bridge
    pub fn new() -> Self {
        let current_user = std::env::var("USER").unwrap_or_else(|_| "root".to_string());

        Self {
            configs: HashMap::new(),
            programs: Vec::new(),
            services: Vec::new(),
            generations: Vec::new(),
            current_user,
        }
    }

    /// Create for specific user
    pub fn for_user(user: impl Into<String>) -> Self {
        Self {
            current_user: user.into(),
            ..Self::new()
        }
    }

    /// Detect home-manager installation
    pub fn detect(&mut self) -> bool {
        // Check for home-manager command
        let output = Command::new("which").arg("home-manager").output();

        if let Ok(out) = output {
            if out.status.success() {
                self.refresh_config();
                return true;
            }
        }

        // Check for nix profile home-manager
        let nix_output = Command::new("nix").args(["profile", "list"]).output();

        if let Ok(out) = nix_output {
            let stdout = String::from_utf8_lossy(&out.stdout);
            if stdout.contains("home-manager") {
                self.refresh_config();
                return true;
            }
        }

        false
    }

    /// Refresh configuration state
    pub fn refresh_config(&mut self) {
        // Get home directory
        let home = std::env::var("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/root"));

        // Look for home-manager config files
        let config_locations = [
            home.join(".config/home-manager/home.nix"),
            home.join(".config/nixpkgs/home.nix"),
            PathBuf::from("/etc/nixos/home.nix"),
        ];

        for path in &config_locations {
            if path.exists() {
                let config = HomeConfig {
                    user: self.current_user.clone(),
                    config_path: path.clone(),
                    home_dir: home.clone(),
                    generation: self.get_current_generation(),
                    last_switch: None,
                };
                self.configs.insert(self.current_user.clone(), config);
                break;
            }
        }

        // Refresh generations
        self.refresh_generations();
    }

    /// Get current generation number
    fn get_current_generation(&self) -> Option<u32> {
        let output = Command::new("home-manager")
            .arg("generations")
            .output()
            .ok()?;

        let stdout = String::from_utf8_lossy(&output.stdout);

        // Parse first line for current generation
        stdout.lines().next().and_then(|line| {
            // Format: "2024-01-15 10:30 : id 123 -> /nix/store/..."
            line.split_whitespace()
                .find(|s| s.parse::<u32>().is_ok())
                .and_then(|s| s.parse().ok())
        })
    }

    /// Refresh generation history
    fn refresh_generations(&mut self) {
        self.generations.clear();

        let output = Command::new("home-manager").arg("generations").output();

        if let Ok(out) = output {
            let stdout = String::from_utf8_lossy(&out.stdout);

            for line in stdout.lines() {
                if let Some(r#gen) = self.parse_generation_line(line) {
                    self.generations.push(r#gen);
                }
            }
        }
    }

    fn parse_generation_line(&self, line: &str) -> Option<HomeGeneration> {
        // Format: "2024-01-15 10:30 : id 123 -> /nix/store/..."
        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.len() < 6 {
            return None;
        }

        let timestamp = format!("{} {}", parts[0], parts[1]);

        let id: u32 = parts
            .iter()
            .find(|s| s.parse::<u32>().is_ok())
            .and_then(|s| s.parse().ok())?;

        let path = parts
            .iter()
            .find(|s| s.starts_with("/nix/store"))
            .map(|s| PathBuf::from(*s))
            .unwrap_or_default();

        Some(HomeGeneration {
            id,
            timestamp,
            description: None,
            path,
        })
    }

    /// Build home-manager configuration
    pub fn build(&self) -> HomeResult {
        let output = Command::new("home-manager").arg("build").output();

        match output {
            Ok(out) => {
                if out.status.success() {
                    HomeResult {
                        success: true,
                        output: String::from_utf8_lossy(&out.stdout).to_string(),
                        error: None,
                        new_generation: None,
                    }
                } else {
                    HomeResult {
                        success: false,
                        output: String::from_utf8_lossy(&out.stdout).to_string(),
                        error: Some(String::from_utf8_lossy(&out.stderr).to_string()),
                        new_generation: None,
                    }
                }
            }
            Err(e) => HomeResult {
                success: false,
                output: String::new(),
                error: Some(e.to_string()),
                new_generation: None,
            },
        }
    }

    /// Switch to new home-manager configuration
    pub fn switch(&mut self) -> HomeResult {
        let output = Command::new("home-manager").arg("switch").output();

        match output {
            Ok(out) => {
                if out.status.success() {
                    self.refresh_config();

                    HomeResult {
                        success: true,
                        output: String::from_utf8_lossy(&out.stdout).to_string(),
                        error: None,
                        new_generation: self.get_current_generation(),
                    }
                } else {
                    HomeResult {
                        success: false,
                        output: String::from_utf8_lossy(&out.stdout).to_string(),
                        error: Some(String::from_utf8_lossy(&out.stderr).to_string()),
                        new_generation: None,
                    }
                }
            }
            Err(e) => HomeResult {
                success: false,
                output: String::new(),
                error: Some(e.to_string()),
                new_generation: None,
            },
        }
    }

    /// Rollback to previous generation
    pub fn rollback(&mut self) -> HomeResult {
        // Find previous generation
        if self.generations.len() < 2 {
            return HomeResult {
                success: false,
                output: String::new(),
                error: Some("No previous generation available".to_string()),
                new_generation: None,
            };
        }

        let prev_id = self.generations[1].id;

        let output = Command::new("home-manager")
            .args(["switch", "--generation", &prev_id.to_string()])
            .output();

        match output {
            Ok(out) => {
                if out.status.success() {
                    self.refresh_config();

                    HomeResult {
                        success: true,
                        output: format!("Rolled back to generation {prev_id}"),
                        error: None,
                        new_generation: Some(prev_id),
                    }
                } else {
                    HomeResult {
                        success: false,
                        output: String::from_utf8_lossy(&out.stdout).to_string(),
                        error: Some(String::from_utf8_lossy(&out.stderr).to_string()),
                        new_generation: None,
                    }
                }
            }
            Err(e) => HomeResult {
                success: false,
                output: String::new(),
                error: Some(e.to_string()),
                new_generation: None,
            },
        }
    }

    /// Switch to specific generation
    pub fn switch_generation(&mut self, gen_id: u32) -> HomeResult {
        let output = Command::new("home-manager")
            .args(["switch", "--generation", &gen_id.to_string()])
            .output();

        match output {
            Ok(out) => {
                if out.status.success() {
                    self.refresh_config();

                    HomeResult {
                        success: true,
                        output: format!("Switched to generation {gen_id}"),
                        error: None,
                        new_generation: Some(gen_id),
                    }
                } else {
                    HomeResult {
                        success: false,
                        output: String::from_utf8_lossy(&out.stdout).to_string(),
                        error: Some(String::from_utf8_lossy(&out.stderr).to_string()),
                        new_generation: None,
                    }
                }
            }
            Err(e) => HomeResult {
                success: false,
                output: String::new(),
                error: Some(e.to_string()),
                new_generation: None,
            },
        }
    }

    /// Delete old generations
    pub fn expire_generations(&mut self, keep: usize) -> HomeResult {
        let output = Command::new("home-manager")
            .args(["expire-generations", &format!("-{keep}d")])
            .output();

        match output {
            Ok(out) => {
                if out.status.success() {
                    self.refresh_generations();

                    HomeResult {
                        success: true,
                        output: String::from_utf8_lossy(&out.stdout).to_string(),
                        error: None,
                        new_generation: None,
                    }
                } else {
                    HomeResult {
                        success: false,
                        output: String::new(),
                        error: Some(String::from_utf8_lossy(&out.stderr).to_string()),
                        new_generation: None,
                    }
                }
            }
            Err(e) => HomeResult {
                success: false,
                output: String::new(),
                error: Some(e.to_string()),
                new_generation: None,
            },
        }
    }

    /// Get current user's config
    pub fn current_config(&self) -> Option<&HomeConfig> {
        self.configs.get(&self.current_user)
    }

    /// Get all generations
    pub fn generations(&self) -> &[HomeGeneration] {
        &self.generations
    }

    /// Get config file path for editing
    pub fn config_path(&self) -> Option<&Path> {
        self.current_config().map(|c| c.config_path.as_path())
    }

    /// Check if home-manager is available
    pub fn is_available(&self) -> bool {
        !self.configs.is_empty()
    }

    /// List installed packages via home-manager
    pub fn list_packages(&self) -> Vec<String> {
        let output = Command::new("home-manager").arg("packages").output();

        match output {
            Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout)
                .lines()
                .map(|s| s.to_string())
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Format status for display
    pub fn format_status(&self) -> String {
        let mut output = String::new();

        if let Some(config) = self.current_config() {
            output.push_str(&format!("User: {}\n", config.user));
            output.push_str(&format!("Config: {}\n", config.config_path.display()));

            if let Some(r#gen) = config.generation {
                output.push_str(&format!("Generation: {gen}\n"));
            }

            output.push_str(&format!("\nGenerations ({}):\n", self.generations.len()));
            for (i, r#gen) in self.generations.iter().take(5).enumerate() {
                let marker = if i == 0 { "→" } else { " " };
                output.push_str(&format!(
                    "  {} {} - {}\n",
                    marker, r#gen.id, r#gen.timestamp
                ));
            }
        } else {
            output.push_str("Home-manager not configured for current user.\n");
        }

        output
    }
}

impl Default for HomeManagerBridge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_creation() {
        let bridge = HomeManagerBridge::new();
        assert!(!bridge.current_user.is_empty());
    }

    #[test]
    fn test_for_user() {
        let bridge = HomeManagerBridge::for_user("testuser");
        assert_eq!(bridge.current_user, "testuser");
    }

    #[test]
    fn test_home_result() {
        let result = HomeResult {
            success: true,
            output: "success".to_string(),
            error: None,
            new_generation: Some(42),
        };

        assert!(result.success);
        assert_eq!(result.new_generation, Some(42));
    }
}
