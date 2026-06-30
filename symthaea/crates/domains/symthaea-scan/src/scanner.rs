// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Platform-specific system scanning — detects installed apps, hardware, and system info.
//!
//! Runs native commands to discover what's installed, then matches against AppDatabase.

use std::collections::HashMap;
use std::process::Command;

use serde::{Deserialize, Serialize};
use symthaea_app_db::{AppDatabase, MatchQuality};

/// Complete scan result — everything we know about this machine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanResult {
    pub os: OsInfo,
    pub hardware: HardwareInfo,
    pub installed_apps: Vec<DetectedApp>,
    pub migration: MigrationSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OsInfo {
    pub name: String,
    pub version: String,
    pub kernel: String,
    pub arch: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub cpu_model: String,
    pub cpu_cores: u32,
    pub memory_gb: f64,
    pub gpu: String,
    pub disk_total_gb: f64,
    pub disk_free_gb: f64,
}

/// An app detected on the system, matched against AppDatabase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedApp {
    /// Name as detected on the system (e.g., "Google Chrome" from winget)
    pub detected_name: String,
    /// Source it was detected from (winget, brew, dpkg, etc.)
    pub source: String,
    /// Matched canonical name from AppDatabase (if matched)
    pub canonical_name: Option<String>,
    /// NixOS package recommendation
    pub nix_package: Option<String>,
    /// NixOS display name
    pub nix_display: Option<String>,
    /// Match quality
    pub quality: Option<String>,
    /// Confidence 0-100
    pub confidence: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationSummary {
    pub total_detected: usize,
    pub matched: usize,
    pub native_available: usize,
    pub alternatives: usize,
    pub no_equivalent: usize,
}

/// Run a full system scan.
pub fn scan() -> ScanResult {
    let os = detect_os();
    let hardware = detect_hardware();
    let raw_apps = detect_installed_apps(&os.name);
    let db = AppDatabase::new();
    let installed_apps = match_apps(&db, &raw_apps);

    let mut native = 0;
    let mut alts = 0;
    let mut none = 0;
    let matched = installed_apps
        .iter()
        .filter(|a| a.canonical_name.is_some())
        .count();
    for app in &installed_apps {
        match app.quality.as_deref() {
            Some("Native") | Some("Official Linux") => native += 1,
            Some("Strong Alternative")
            | Some("Partial Alternative")
            | Some("Wine/Proton")
            | Some("Web App") => alts += 1,
            Some("No Equivalent") => none += 1,
            _ => {}
        }
    }

    ScanResult {
        os,
        hardware,
        migration: MigrationSummary {
            total_detected: installed_apps.len(),
            matched,
            native_available: native,
            alternatives: alts,
            no_equivalent: none,
        },
        installed_apps,
    }
}

// ═══════════════════════════════════════════════════════
// OS Detection
// ═══════════════════════════════════════════════════════

fn detect_os() -> OsInfo {
    let arch = std::env::consts::ARCH.to_string();

    #[cfg(target_os = "linux")]
    {
        let name = run_cmd("lsb_release", &["-is"])
            .or_else(|| read_os_release("NAME"))
            .unwrap_or_else(|| "Linux".into());
        let version = run_cmd("lsb_release", &["-rs"])
            .or_else(|| read_os_release("VERSION_ID"))
            .unwrap_or_default();
        let kernel = run_cmd("uname", &["-r"]).unwrap_or_default();
        return OsInfo {
            name,
            version,
            kernel,
            arch,
        };
    }

    #[cfg(target_os = "macos")]
    {
        let version = run_cmd("sw_vers", &["-productVersion"]).unwrap_or_default();
        let kernel = run_cmd("uname", &["-r"]).unwrap_or_default();
        return OsInfo {
            name: "macOS".into(),
            version,
            kernel,
            arch,
        };
    }

    #[cfg(target_os = "windows")]
    {
        let version = run_cmd("cmd", &["/c", "ver"]).unwrap_or_default();
        let kernel = "NT".into();
        return OsInfo {
            name: "Windows".into(),
            version,
            kernel,
            arch,
        };
    }

    #[allow(unreachable_code)]
    OsInfo {
        name: std::env::consts::OS.into(),
        version: String::new(),
        kernel: String::new(),
        arch,
    }
}

#[cfg(target_os = "linux")]
fn read_os_release(key: &str) -> Option<String> {
    let content = std::fs::read_to_string("/etc/os-release").ok()?;
    for line in content.lines() {
        if let Some(val) = line.strip_prefix(&format!("{key}=")) {
            return Some(val.trim_matches('"').to_string());
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
fn read_os_release(_key: &str) -> Option<String> {
    None
}

// ═══════════════════════════════════════════════════════
// Hardware Detection
// ═══════════════════════════════════════════════════════

fn detect_hardware() -> HardwareInfo {
    let cpu_cores = num_cpus();

    #[cfg(target_os = "linux")]
    {
        let cpu_model = run_cmd(
            "sh",
            &["-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"],
        )
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "Unknown".into());

        let memory_gb = std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("MemTotal:"))
                    .and_then(|l| l.split_whitespace().nth(1))
                    .and_then(|v| v.parse::<f64>().ok())
            })
            .map(|kb| (kb / 1_048_576.0 * 10.0).round() / 10.0)
            .unwrap_or(0.0);

        let gpu = run_cmd(
            "sh",
            &[
                "-c",
                "lspci 2>/dev/null | grep -iE 'VGA|3D|Display' | head -1 | sed 's/.*: //'",
            ],
        )
        .unwrap_or_else(|| "Unknown".into());

        let (disk_total, disk_free) = disk_space("/");

        return HardwareInfo {
            cpu_model,
            cpu_cores,
            memory_gb,
            gpu,
            disk_total_gb: disk_total,
            disk_free_gb: disk_free,
        };
    }

    #[cfg(target_os = "macos")]
    {
        let cpu_model = run_cmd("sysctl", &["-n", "machdep.cpu.brand_string"])
            .unwrap_or_else(|| "Unknown".into());
        let memory_gb = run_cmd("sysctl", &["-n", "hw.memsize"])
            .and_then(|s| s.trim().parse::<f64>().ok())
            .map(|b| (b / 1_073_741_824.0 * 10.0).round() / 10.0)
            .unwrap_or(0.0);
        let gpu = run_cmd("sh", &["-c", "system_profiler SPDisplaysDataType 2>/dev/null | grep 'Chipset Model' | head -1 | sed 's/.*: //'"])
            .unwrap_or_else(|| "Unknown".into());
        let (disk_total, disk_free) = disk_space("/");
        return HardwareInfo {
            cpu_model,
            cpu_cores,
            memory_gb,
            gpu,
            disk_total_gb: disk_total,
            disk_free_gb: disk_free,
        };
    }

    #[cfg(target_os = "windows")]
    {
        let cpu_model = run_cmd("wmic", &["cpu", "get", "name", "/value"])
            .map(|s| {
                s.lines()
                    .find(|l| l.starts_with("Name="))
                    .map(|l| l[5..].to_string())
                    .unwrap_or_default()
            })
            .unwrap_or_else(|| "Unknown".into());
        let memory_gb = run_cmd(
            "wmic",
            &["computersystem", "get", "totalphysicalmemory", "/value"],
        )
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("TotalPhysicalMemory="))
                .and_then(|l| l[20..].trim().parse::<f64>().ok())
        })
        .map(|b| (b / 1_073_741_824.0 * 10.0).round() / 10.0)
        .unwrap_or(0.0);
        let gpu = run_cmd(
            "wmic",
            &["path", "win32_videocontroller", "get", "name", "/value"],
        )
        .map(|s| {
            s.lines()
                .find(|l| l.starts_with("Name="))
                .map(|l| l[5..].to_string())
                .unwrap_or_default()
        })
        .unwrap_or_else(|| "Unknown".into());
        let (disk_total, disk_free) = (0.0, 0.0); // TODO: wmic logicaldisk
        return HardwareInfo {
            cpu_model,
            cpu_cores,
            memory_gb,
            gpu,
            disk_total_gb: disk_total,
            disk_free_gb: disk_free,
        };
    }

    #[allow(unreachable_code)]
    HardwareInfo {
        cpu_model: "Unknown".into(),
        cpu_cores,
        memory_gb: 0.0,
        gpu: "Unknown".into(),
        disk_total_gb: 0.0,
        disk_free_gb: 0.0,
    }
}

fn num_cpus() -> u32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1)
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn disk_space(path: &str) -> (f64, f64) {
    run_cmd("df", &["-BG", path])
        .and_then(|s| {
            let line = s.lines().nth(1)?;
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                let total = parts[1].trim_end_matches('G').parse::<f64>().ok()?;
                let free = parts[3].trim_end_matches('G').parse::<f64>().ok()?;
                Some((total, free))
            } else {
                None
            }
        })
        .unwrap_or((0.0, 0.0))
}

#[cfg(target_os = "windows")]
fn disk_space(_path: &str) -> (f64, f64) {
    (0.0, 0.0)
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn disk_space(_path: &str) -> (f64, f64) {
    (0.0, 0.0)
}

// ═══════════════════════════════════════════════════════
// App Detection — platform-specific package manager queries
// ═══════════════════════════════════════════════════════

#[derive(Debug)]
struct RawApp {
    name: String,
    source: String,
}

fn detect_installed_apps(os_name: &str) -> Vec<RawApp> {
    let mut apps = Vec::new();

    match os_name {
        n if n.contains("indows") || n == "Windows" => {
            // winget
            if let Some(output) = run_cmd("winget", &["list", "--accept-source-agreements"]) {
                apps.extend(parse_winget_list(&output));
            }
            // Scan Program Files
            for dir in &["C:\\Program Files", "C:\\Program Files (x86)"] {
                if let Ok(entries) = std::fs::read_dir(dir) {
                    for entry in entries.flatten() {
                        if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                            apps.push(RawApp {
                                name: entry.file_name().to_string_lossy().into(),
                                source: "program_files".into(),
                            });
                        }
                    }
                }
            }
        }
        "macOS" => {
            // brew
            if let Some(output) = run_cmd("brew", &["list", "--formula"]) {
                for line in output.lines() {
                    let name = line.trim();
                    if !name.is_empty() {
                        apps.push(RawApp {
                            name: name.into(),
                            source: "brew".into(),
                        });
                    }
                }
            }
            if let Some(output) = run_cmd("brew", &["list", "--cask"]) {
                for line in output.lines() {
                    let name = line.trim();
                    if !name.is_empty() {
                        apps.push(RawApp {
                            name: name.into(),
                            source: "brew_cask".into(),
                        });
                    }
                }
            }
            // /Applications
            if let Ok(entries) = std::fs::read_dir("/Applications") {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().into_owned();
                    if name.ends_with(".app") {
                        apps.push(RawApp {
                            name: name.trim_end_matches(".app").into(),
                            source: "applications".into(),
                        });
                    }
                }
            }
        }
        _ => {
            // Linux — try all package managers
            // dpkg (Debian/Ubuntu)
            if let Some(output) = run_cmd("dpkg", &["--get-selections"]) {
                for line in output.lines() {
                    if let Some(name) = line.split_whitespace().next() {
                        if line.contains("install") {
                            apps.push(RawApp {
                                name: name.into(),
                                source: "dpkg".into(),
                            });
                        }
                    }
                }
            }
            // pacman (Arch)
            if let Some(output) = run_cmd("pacman", &["-Qq"]) {
                for line in output.lines() {
                    let name = line.trim();
                    if !name.is_empty() {
                        apps.push(RawApp {
                            name: name.into(),
                            source: "pacman".into(),
                        });
                    }
                }
            }
            // rpm (Fedora/RHEL)
            if let Some(output) = run_cmd("rpm", &["-qa", "--qf", "%{NAME}\n"]) {
                for line in output.lines() {
                    let name = line.trim();
                    if !name.is_empty() {
                        apps.push(RawApp {
                            name: name.into(),
                            source: "rpm".into(),
                        });
                    }
                }
            }
            // flatpak
            if let Some(output) = run_cmd("flatpak", &["list", "--app", "--columns=application"]) {
                for line in output.lines() {
                    let name = line.trim();
                    if !name.is_empty() && name.contains('.') {
                        apps.push(RawApp {
                            name: name.into(),
                            source: "flatpak".into(),
                        });
                    }
                }
            }
            // snap
            if let Some(output) = run_cmd("snap", &["list"]) {
                for line in output.lines().skip(1) {
                    if let Some(name) = line.split_whitespace().next() {
                        apps.push(RawApp {
                            name: name.into(),
                            source: "snap".into(),
                        });
                    }
                }
            }
            // NixOS: system packages (from configuration.nix)
            if let Some(output) = run_cmd(
                "sh",
                &[
                    "-c",
                    "nixos-option environment.systemPackages 2>/dev/null | head -50",
                ],
            ) {
                for pkg in output.split_whitespace() {
                    if pkg.contains("nixpkgs") || pkg.contains("pkgs.") {
                        let name = pkg.rsplit('.').next().unwrap_or(pkg).trim();
                        if !name.is_empty() {
                            apps.push(RawApp {
                                name: name.into(),
                                source: "nixos".into(),
                            });
                        }
                    }
                }
            }
            // Nix profile (user packages) — format: "Name:  <name>" lines
            if let Some(output) = run_cmd("nix", &["profile", "list"]) {
                for line in output.lines() {
                    if let Some(name) = line.strip_prefix("Name:") {
                        // Strip ANSI escape codes and whitespace
                        let name = name.replace("\x1b[1m", "").replace("\x1b[0m", "");
                        let name = name.trim();
                        if !name.is_empty() {
                            apps.push(RawApp {
                                name: name.into(),
                                source: "nix_profile".into(),
                            });
                        }
                    }
                }
            }
            // nix-env (legacy)
            if let Some(output) = run_cmd("nix-env", &["-q"]) {
                for line in output.lines() {
                    let name = line
                        .rsplit('-')
                        .next()
                        .map(|_| {
                            // Strip version suffix: "firefox-121.0" -> "firefox"
                            let parts: Vec<&str> = line.rsplitn(2, '-').collect();
                            if parts.len() == 2 { parts[1] } else { line }
                        })
                        .unwrap_or(line);
                    if !name.is_empty() {
                        apps.push(RawApp {
                            name: name.into(),
                            source: "nix_env".into(),
                        });
                    }
                }
            }
        }
    }

    // Deduplicate by name (case-insensitive)
    let mut seen = std::collections::HashSet::new();
    apps.retain(|a| seen.insert(a.name.to_lowercase()));
    apps
}

fn parse_winget_list(output: &str) -> Vec<RawApp> {
    let mut apps = Vec::new();
    let mut in_table = false;
    for line in output.lines() {
        if line.contains("----") {
            in_table = true;
            continue;
        }
        if !in_table {
            continue;
        }
        let name = line
            .split_whitespace()
            .take(3)
            .collect::<Vec<_>>()
            .join(" ");
        if !name.is_empty() {
            apps.push(RawApp {
                name,
                source: "winget".into(),
            });
        }
    }
    apps
}

// ═══════════════════════════════════════════════════════
// Match detected apps against AppDatabase
// ═══════════════════════════════════════════════════════

fn match_apps(db: &AppDatabase, raw: &[RawApp]) -> Vec<DetectedApp> {
    let mut lookup: HashMap<String, (&'static str, &'static str, MatchQuality, &'static str)> =
        HashMap::new();

    for entry in db.entries() {
        let all_names: Vec<&str> = entry
            .windows_names
            .iter()
            .chain(entry.macos_names.iter())
            .chain(entry.linux_names.iter())
            .chain(entry.flatpak_ids.iter())
            .chain(entry.snap_names.iter())
            .chain(entry.winget_ids.iter())
            .chain(entry.brew_names.iter())
            .copied()
            .collect();

        for alias in all_names {
            lookup.insert(
                alias.to_lowercase(),
                (
                    entry.name,
                    entry.primary.nix_pkg,
                    entry.primary.quality,
                    entry.primary.display_name,
                ),
            );
        }
        // Also index by canonical name
        lookup.insert(
            entry.name.to_lowercase(),
            (
                entry.name,
                entry.primary.nix_pkg,
                entry.primary.quality,
                entry.primary.display_name,
            ),
        );
    }

    raw.iter()
        .map(|app| {
            let key = app.name.to_lowercase();
            // Try exact match first, then substring
            let matched = lookup.get(&key).copied().or_else(|| {
                lookup
                    .iter()
                    .find(|(k, _)| key.contains(k.as_str()) || k.contains(&key))
                    .map(|(_, v)| *v)
            });

            DetectedApp {
                detected_name: app.name.clone(),
                source: app.source.clone(),
                canonical_name: matched.map(|m| m.0.to_string()),
                nix_package: matched.map(|m| m.1.to_string()),
                nix_display: matched.map(|m| m.3.to_string()),
                quality: matched.map(|m| m.2.label().to_string()),
                confidence: matched.map(|m| (m.2.confidence() * 100.0) as u32),
            }
        })
        .collect()
}

// ═══════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════

fn run_cmd(cmd: &str, args: &[&str]) -> Option<String> {
    Command::new(cmd)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_produces_results() {
        let result = scan();
        assert!(!result.os.name.is_empty());
        assert!(result.hardware.cpu_cores > 0);
        assert!(result.hardware.memory_gb > 0.0);
    }

    #[test]
    fn scan_detects_common_tools() {
        let result = scan();
        // On any dev machine, we should detect *something*
        assert!(
            result.installed_apps.len() > 0,
            "Should detect at least one app"
        );
    }

    #[test]
    fn match_quality_labels() {
        assert_eq!(MatchQuality::Native.label(), "Native");
        assert_eq!(
            MatchQuality::StrongAlternative.label(),
            "Strong Alternative"
        );
    }

    #[test]
    fn winget_parser() {
        let output = "Name                   Id                Version\n\
                       -----------------------------------------\n\
                       Google Chrome          Google.Chrome     120.0\n\
                       Firefox                Mozilla.Firefox   121.0\n";
        let apps = parse_winget_list(output);
        assert_eq!(apps.len(), 2);
        assert!(apps[0].name.contains("Google"));
    }
}
