// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Platform-specific system scanning — detects installed apps, hardware, and system info.
//!
//! Runs native commands to discover what's installed, then matches against AppDatabase.

use std::collections::{BTreeMap, BTreeSet};
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
    /// Source(s) it was detected from (winget, program_files, brew, dpkg, etc.)
    pub source: String,
    /// Matched canonical name from AppDatabase (if matched unambiguously)
    pub canonical_name: Option<String>,
    /// NixOS package recommendation
    pub nix_package: Option<String>,
    /// NixOS display name
    pub nix_display: Option<String>,
    /// Match quality
    pub quality: Option<String>,
    /// Legacy replacement-quality percentage. This is not migration probability.
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
        return OsInfo {
            name: "Windows".into(),
            version,
            kernel: "NT".into(),
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

#[derive(Debug, Deserialize)]
struct WindowsHardwareProbe {
    cpu_model: Option<String>,
    memory_bytes: Option<u64>,
    gpu: Option<String>,
    disk_total_bytes: Option<u64>,
    disk_free_bytes: Option<u64>,
}

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
            .map(bytes_to_gb)
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
        // WMIC is no longer available on current Windows releases. Use the
        // supported CIM provider through PowerShell and parse structured JSON.
        let script = r#"$cpu=Get-CimInstance Win32_Processor|Select-Object -First 1;$sys=Get-CimInstance Win32_ComputerSystem;$gpu=Get-CimInstance Win32_VideoController|Select-Object -First 1;$disk=Get-CimInstance Win32_LogicalDisk -Filter \"DeviceID='C:'\";[PSCustomObject]@{cpu_model=[string]$cpu.Name;memory_bytes=[uint64]$sys.TotalPhysicalMemory;gpu=[string]$gpu.Name;disk_total_bytes=[uint64]$disk.Size;disk_free_bytes=[uint64]$disk.FreeSpace}|ConvertTo-Json -Compress"#;
        let probe = run_cmd(
            "powershell.exe",
            &["-NoProfile", "-NonInteractive", "-Command", script],
        )
        .or_else(|| run_cmd("pwsh", &["-NoProfile", "-NonInteractive", "-Command", script]))
        .and_then(|json| parse_windows_hardware_probe(&json));

        let probe = probe.unwrap_or(WindowsHardwareProbe {
            cpu_model: None,
            memory_bytes: None,
            gpu: None,
            disk_total_bytes: None,
            disk_free_bytes: None,
        });
        return HardwareInfo {
            cpu_model: nonempty_or_unknown(probe.cpu_model),
            cpu_cores,
            memory_gb: probe.memory_bytes.map(bytes_to_gb_u64).unwrap_or(0.0),
            gpu: nonempty_or_unknown(probe.gpu),
            disk_total_gb: probe.disk_total_bytes.map(bytes_to_gb_u64).unwrap_or(0.0),
            disk_free_gb: probe.disk_free_bytes.map(bytes_to_gb_u64).unwrap_or(0.0),
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

fn parse_windows_hardware_probe(json: &str) -> Option<WindowsHardwareProbe> {
    serde_json::from_str(json).ok()
}

fn nonempty_or_unknown(value: Option<String>) -> String {
    value
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "Unknown".into())
}

fn bytes_to_gb(bytes: f64) -> f64 {
    (bytes / 1_073_741_824.0 * 10.0).round() / 10.0
}

fn bytes_to_gb_u64(bytes: u64) -> f64 {
    bytes_to_gb(bytes as f64)
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

// ═══════════════════════════════════════════════════════
// App Detection — platform-specific package manager queries
// ═══════════════════════════════════════════════════════

#[derive(Debug, Clone)]
struct RawApp {
    name: String,
    sources: BTreeSet<String>,
    identifiers: BTreeSet<String>,
}

impl RawApp {
    fn new(name: impl Into<String>, source: impl Into<String>) -> Self {
        let mut sources = BTreeSet::new();
        sources.insert(source.into());
        Self {
            name: name.into(),
            sources,
            identifiers: BTreeSet::new(),
        }
    }

    fn with_identifier(mut self, identifier: impl Into<String>) -> Self {
        let identifier = identifier.into();
        if !identifier.trim().is_empty() {
            self.identifiers.insert(identifier);
        }
        self
    }

    fn merge(&mut self, other: RawApp) {
        self.sources.extend(other.sources);
        self.identifiers.extend(other.identifiers);
    }

    fn source_label(&self) -> String {
        self.sources.iter().cloned().collect::<Vec<_>>().join("+")
    }

    fn match_inputs(&self) -> impl Iterator<Item = &str> {
        std::iter::once(self.name.as_str()).chain(self.identifiers.iter().map(String::as_str))
    }
}

fn detect_installed_apps(os_name: &str) -> Vec<RawApp> {
    let mut apps = Vec::new();

    match os_name {
        n if n.contains("indows") || n == "Windows" => {
            if let Some(output) = run_cmd(
                "winget",
                &["list", "--accept-source-agreements", "--disable-interactivity"],
            ) {
                apps.extend(parse_winget_list(&output));
            }
            for dir in &["C:\\Program Files", "C:\\Program Files (x86)"] {
                if let Ok(entries) = std::fs::read_dir(dir) {
                    for entry in entries.flatten() {
                        if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                            apps.push(RawApp::new(
                                entry.file_name().to_string_lossy().into_owned(),
                                "program_files",
                            ));
                        }
                    }
                }
            }
        }
        "macOS" => {
            if let Some(output) = run_cmd("brew", &["list", "--formula"]) {
                for line in output.lines() {
                    let name = line.trim();
                    if !name.is_empty() {
                        apps.push(RawApp::new(name, "brew"));
                    }
                }
            }
            if let Some(output) = run_cmd("brew", &["list", "--cask"]) {
                for line in output.lines() {
                    let name = line.trim();
                    if !name.is_empty() {
                        apps.push(RawApp::new(name, "brew_cask"));
                    }
                }
            }
            if let Ok(entries) = std::fs::read_dir("/Applications") {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().into_owned();
                    if name.ends_with(".app") {
                        apps.push(RawApp::new(name.trim_end_matches(".app"), "applications"));
                    }
                }
            }
        }
        _ => {
            if let Some(output) = run_cmd("dpkg", &["--get-selections"]) {
                for line in output.lines() {
                    if let Some(name) = line.split_whitespace().next() {
                        if line.contains("install") {
                            apps.push(RawApp::new(name, "dpkg"));
                        }
                    }
                }
            }
            if let Some(output) = run_cmd("pacman", &["-Qq"]) {
                for line in output.lines() {
                    let name = line.trim();
                    if !name.is_empty() {
                        apps.push(RawApp::new(name, "pacman"));
                    }
                }
            }
            if let Some(output) = run_cmd("rpm", &["-qa", "--qf", "%{NAME}\n"]) {
                for line in output.lines() {
                    let name = line.trim();
                    if !name.is_empty() {
                        apps.push(RawApp::new(name, "rpm"));
                    }
                }
            }
            if let Some(output) = run_cmd("flatpak", &["list", "--app", "--columns=application"]) {
                for line in output.lines() {
                    let name = line.trim();
                    if !name.is_empty() && name.contains('.') {
                        apps.push(RawApp::new(name, "flatpak"));
                    }
                }
            }
            if let Some(output) = run_cmd("snap", &["list"]) {
                for line in output.lines().skip(1) {
                    if let Some(name) = line.split_whitespace().next() {
                        apps.push(RawApp::new(name, "snap"));
                    }
                }
            }
            if let Some(output) = run_cmd(
                "sh",
                &["-c", "nixos-option environment.systemPackages 2>/dev/null | head -50"],
            ) {
                for pkg in output.split_whitespace() {
                    if pkg.contains("nixpkgs") || pkg.contains("pkgs.") {
                        let name = pkg.rsplit('.').next().unwrap_or(pkg).trim();
                        if !name.is_empty() {
                            apps.push(RawApp::new(name, "nixos"));
                        }
                    }
                }
            }
            if let Some(output) = run_cmd("nix", &["profile", "list"]) {
                for line in output.lines() {
                    if let Some(name) = line.strip_prefix("Name:") {
                        let name = name.replace("\x1b[1m", "").replace("\x1b[0m", "");
                        let name = name.trim();
                        if !name.is_empty() {
                            apps.push(RawApp::new(name, "nix_profile"));
                        }
                    }
                }
            }
            if let Some(output) = run_cmd("nix-env", &["-q"]) {
                for line in output.lines() {
                    let parts: Vec<&str> = line.rsplitn(2, '-').collect();
                    let name = if parts.len() == 2 { parts[1] } else { line };
                    if !name.is_empty() {
                        apps.push(RawApp::new(name, "nix_env"));
                    }
                }
            }
        }
    }

    merge_raw_apps(apps)
}

fn merge_raw_apps(apps: Vec<RawApp>) -> Vec<RawApp> {
    let mut merged: BTreeMap<String, RawApp> = BTreeMap::new();
    for app in apps {
        let key = normalize_name(&app.name);
        if key.is_empty() {
            continue;
        }
        match merged.get_mut(&key) {
            Some(existing) => existing.merge(app),
            None => {
                merged.insert(key, app);
            }
        }
    }
    merged.into_values().collect()
}

/// Parse WinGet's human-readable table using header column offsets rather than
/// assuming the first three whitespace tokens are the application name.
///
/// If the expected `Id`/`Version` columns cannot be located, no rows are emitted:
/// missing evidence is preferable to manufacturing an incorrect app identity.
fn parse_winget_list(output: &str) -> Vec<RawApp> {
    let lines: Vec<&str> = output.lines().collect();
    let Some(separator_index) = lines.iter().position(|line| {
        let trimmed = line.trim();
        trimmed.len() >= 3 && trimmed.chars().all(|c| c == '-' || c.is_whitespace())
    }) else {
        return Vec::new();
    };
    let Some(header) = lines[..separator_index]
        .iter()
        .rev()
        .find(|line| !line.trim().is_empty())
        .copied()
    else {
        return Vec::new();
    };

    let Some(id_start) = char_index_of(header, "Id") else {
        return Vec::new();
    };
    let Some(version_start) = char_index_of(header, "Version") else {
        return Vec::new();
    };
    if id_start == 0 || version_start <= id_start {
        return Vec::new();
    }

    lines[separator_index + 1..]
        .iter()
        .filter_map(|line| {
            let name = char_slice(line, 0, id_start).trim().to_string();
            let identifier = char_slice(line, id_start, version_start).trim().to_string();
            if name.is_empty() || identifier.is_empty() {
                return None;
            }
            Some(RawApp::new(name, "winget").with_identifier(identifier))
        })
        .collect()
}

fn char_index_of(haystack: &str, needle: &str) -> Option<usize> {
    let byte_index = haystack.find(needle)?;
    Some(haystack[..byte_index].chars().count())
}

fn char_slice(value: &str, start: usize, end: usize) -> String {
    value.chars().skip(start).take(end.saturating_sub(start)).collect()
}

// ═══════════════════════════════════════════════════════
// Deterministic app identity resolution
// ═══════════════════════════════════════════════════════

type MatchRecord = (&'static str, &'static str, MatchQuality, &'static str);

fn match_apps(db: &AppDatabase, raw: &[RawApp]) -> Vec<DetectedApp> {
    let mut alias_index: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut details: BTreeMap<String, MatchRecord> = BTreeMap::new();

    for entry in db.entries() {
        let canonical = entry.name.to_string();
        details.insert(
            canonical.clone(),
            (
                entry.name,
                entry.primary.nix_pkg,
                entry.primary.quality,
                entry.primary.display_name,
            ),
        );

        for alias in entry
            .windows_names
            .iter()
            .chain(entry.macos_names.iter())
            .chain(entry.linux_names.iter())
            .chain(entry.flatpak_ids.iter())
            .chain(entry.snap_names.iter())
            .chain(entry.winget_ids.iter())
            .chain(entry.brew_names.iter())
            .copied()
            .chain(std::iter::once(entry.name))
        {
            alias_index
                .entry(normalize_name(alias))
                .or_default()
                .insert(canonical.clone());
        }
    }

    raw.iter()
        .map(|app| {
            let matched = resolve_app(&alias_index, &details, app);
            DetectedApp {
                detected_name: app.name.clone(),
                source: app.source_label(),
                canonical_name: matched.map(|m| m.0.to_string()),
                nix_package: matched.map(|m| m.1.to_string()),
                nix_display: matched.map(|m| m.3.to_string()),
                quality: matched.map(|m| m.2.label().to_string()),
                confidence: matched.map(|m| (m.2.confidence() * 100.0) as u32),
            }
        })
        .collect()
}

fn resolve_app(
    alias_index: &BTreeMap<String, BTreeSet<String>>,
    details: &BTreeMap<String, MatchRecord>,
    app: &RawApp,
) -> Option<MatchRecord> {
    let inputs: Vec<String> = app
        .match_inputs()
        .map(normalize_name)
        .filter(|value| !value.is_empty())
        .collect();

    // Exact aliases and package identifiers have priority. If exact identities
    // disagree, preserve ambiguity rather than selecting one arbitrarily.
    let mut exact_candidates = BTreeSet::new();
    for input in &inputs {
        if let Some(candidates) = alias_index.get(input) {
            exact_candidates.extend(candidates.iter().cloned());
        }
    }
    match exact_candidates.len() {
        0 => {}
        1 => return details.get(exact_candidates.iter().next()?).copied(),
        _ => return None,
    }

    // Last-resort substring matching is set-valued and deterministic. Only a
    // single unique canonical candidate may be promoted to a match.
    let mut fuzzy_candidates = BTreeSet::new();
    for input in &inputs {
        for (alias, candidates) in alias_index {
            if alias.len() > 3 && input.len() > 3 && (input.contains(alias) || alias.contains(input)) {
                fuzzy_candidates.extend(candidates.iter().cloned());
            }
        }
    }
    if fuzzy_candidates.len() != 1 {
        return None;
    }
    details.get(fuzzy_candidates.iter().next()?).copied()
}

fn normalize_name(value: &str) -> String {
    value.trim().to_lowercase()
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
        assert!(
            !result.installed_apps.is_empty(),
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
    fn winget_parser_preserves_name_and_package_id() {
        let output = "Name                   Id                Version     Source\n\
                      --------------------------------------------------------\n\
                      Google Chrome          Google.Chrome     120.0       winget\n\
                      Firefox                Mozilla.Firefox   121.0       winget\n";
        let apps = parse_winget_list(output);
        assert_eq!(apps.len(), 2);
        assert_eq!(apps[0].name, "Google Chrome");
        assert!(apps[0].identifiers.contains("Google.Chrome"));
        assert_eq!(apps[1].name, "Firefox");
        assert!(apps[1].identifiers.contains("Mozilla.Firefox"));
    }

    #[test]
    fn malformed_winget_table_fails_closed() {
        let output = "Name Version\n----------------\nGoogle Chrome 120.0\n";
        assert!(parse_winget_list(output).is_empty());
    }

    #[test]
    fn windows_hardware_json_parser_preserves_disk_and_memory() {
        let probe = parse_windows_hardware_probe(
            r#"{"cpu_model":"AMD Ryzen","memory_bytes":17179869184,"gpu":"NVIDIA RTX","disk_total_bytes":1000000000000,"disk_free_bytes":500000000000}"#,
        )
        .unwrap();
        assert_eq!(probe.cpu_model.as_deref(), Some("AMD Ryzen"));
        assert_eq!(probe.memory_bytes, Some(17_179_869_184));
        assert_eq!(probe.disk_total_bytes, Some(1_000_000_000_000));
        assert_eq!(probe.disk_free_bytes, Some(500_000_000_000));
    }

    #[test]
    fn ambiguous_fuzzy_candidates_are_not_promoted() {
        let mut aliases = BTreeMap::new();
        aliases.insert(
            "studio".into(),
            BTreeSet::from(["Alpha Studio".into(), "Beta Studio".into()]),
        );
        let details: BTreeMap<String, MatchRecord> = BTreeMap::from([
            (
                "Alpha Studio".into(),
                ("Alpha Studio", "alpha", MatchQuality::Native, "Alpha"),
            ),
            (
                "Beta Studio".into(),
                ("Beta Studio", "beta", MatchQuality::Native, "Beta"),
            ),
        ]);
        let app = RawApp::new("Studio", "fixture");
        assert!(resolve_app(&aliases, &details, &app).is_none());
    }

    #[test]
    fn independent_sources_are_merged_not_discarded() {
        let merged = merge_raw_apps(vec![
            RawApp::new("Google Chrome", "winget").with_identifier("Google.Chrome"),
            RawApp::new("google chrome", "program_files"),
        ]);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].source_label(), "program_files+winget");
        assert!(merged[0].identifiers.contains("Google.Chrome"));
    }
}
