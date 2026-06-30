// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Unified App Database — The single source of truth for app migration
//!
//! Maps applications across Windows (winget), macOS (brew), Linux (dpkg/pacman/
//! flatpak/snap) to nixpkgs equivalents. Supports opinionated defaults with
//! transparent alternatives, confidence scoring, bundle detection, and
//! justification strings for every recommendation.
//!
//! Compiles to both native (for SSH scan) and WASM (for browser-side matching).
//! No filesystem operations — pure data and computation.

pub mod aliases;
pub mod config_gen;
pub mod package_healer;
pub mod semantic_search;
pub mod validation;

use std::collections::HashMap;

// ═══════════════════════════════════════════════════════
// Core Types
// ═══════════════════════════════════════════════════════

/// How well a nixpkgs package replaces the original app.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum MatchQuality {
    /// Exact same app in nixpkgs (firefox → firefox). Confidence: 95-100%
    Native,
    /// Official Linux build exists (VS Code, Discord). Confidence: 90-95%
    OfficialLinux,
    /// Strong open-source alternative (Office → LibreOffice). Confidence: 70-85%
    StrongAlternative,
    /// Partial alternative, different workflow (Photoshop → GIMP). Confidence: 50-70%
    PartialAlternative,
    /// Runs via Wine/Proton with good results. Confidence: 60-80%
    WineCompatible,
    /// Web-based alternative. Confidence: 50-70%
    WebApp,
    /// No real equivalent. Confidence: 0-15%
    NoEquivalent,
}

impl MatchQuality {
    pub fn confidence(&self) -> f32 {
        match self {
            Self::Native => 0.97,
            Self::OfficialLinux => 0.92,
            Self::StrongAlternative => 0.78,
            Self::PartialAlternative => 0.55,
            Self::WineCompatible => 0.65,
            Self::WebApp => 0.60,
            Self::NoEquivalent => 0.05,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Native => "Native",
            Self::OfficialLinux => "Official Linux",
            Self::StrongAlternative => "Strong Alternative",
            Self::PartialAlternative => "Partial Alternative",
            Self::WineCompatible => "Wine/Proton",
            Self::WebApp => "Web App",
            Self::NoEquivalent => "No Equivalent",
        }
    }
}

/// Application category for bundle detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AppCategory {
    Browser,
    Email,
    Office,
    Notes,
    Editor,
    IDE,
    Terminal,
    VersionControl,
    Container,
    DevTools,
    Creative2D,
    Creative3D,
    Audio,
    Video,
    Photo,
    Gaming,
    GamingTools,
    Communication,
    Streaming,
    MediaPlayer,
    FileManager,
    Archive,
    Security,
    VPN,
    Backup,
    SystemUtil,
    Virtualization,
    Science,
    Finance,
}

/// A single nixpkgs recommendation.
#[derive(Debug, Clone)]
pub struct NixRecommendation {
    /// nixpkgs attribute name (e.g., "gimp")
    pub nix_pkg: &'static str,
    /// Human-readable name
    pub display_name: &'static str,
    /// How well it replaces the original
    pub quality: MatchQuality,
    /// WHY this is recommended — the user reads this
    pub justification: &'static str,
    /// What's missing compared to the original
    pub trade_offs: &'static [&'static str],
}

/// An application entry in the database.
#[derive(Debug, Clone)]
pub struct AppEntry {
    /// Canonical display name (e.g., "Adobe Photoshop")
    pub name: &'static str,
    /// Category for bundle detection
    pub category: AppCategory,
    /// Is this a proprietary app?
    pub proprietary: bool,

    // ── Detection: how to recognize this app across platforms ──
    /// Windows: names as they appear in Add/Remove Programs or winget
    pub windows_names: &'static [&'static str],
    /// macOS: .app names or brew formula/cask names
    pub macos_names: &'static [&'static str],
    /// Linux: dpkg/pacman/dnf package names
    pub linux_names: &'static [&'static str],
    /// Flatpak application IDs
    pub flatpak_ids: &'static [&'static str],
    /// Snap package names
    pub snap_names: &'static [&'static str],
    /// winget package IDs
    pub winget_ids: &'static [&'static str],
    /// Homebrew formula or cask names
    pub brew_names: &'static [&'static str],

    // ── Recommendation: what to use on NixOS ──
    /// The PRIMARY recommendation — what Symthaea suggests by default
    pub primary: NixRecommendation,
    /// Alternative recommendations — shown when user clicks "Alternatives"
    pub alternatives: &'static [NixRecommendation],

    /// nixpkgs channel this package name was last verified against (e.g., "25.05")
    pub verified_channel: &'static str,
}

// ═══════════════════════════════════════════════════════
// Bundle System
// ═══════════════════════════════════════════════════════

/// A curated bundle of related packages + NixOS config.
#[derive(Debug, Clone)]
pub struct AppBundle {
    pub name: &'static str,
    pub description: &'static str,
    /// Categories that trigger this bundle
    pub trigger_categories: &'static [AppCategory],
    /// Minimum number of matching apps to suggest this bundle
    pub trigger_threshold: usize,
    /// Packages included in this bundle
    pub packages: &'static [&'static str],
    /// NixOS config options this bundle enables
    pub nix_options: &'static [(&'static str, &'static str)],
    /// Human-readable explanation
    pub explanation: &'static str,
}

// ═══════════════════════════════════════════════════════
// Migration Report
// ═══════════════════════════════════════════════════════

/// Result of matching a user's app list.
#[derive(Debug, Clone)]
pub struct MigrationReport {
    pub total_apps: usize,
    pub matched: Vec<MatchedApp>,
    pub unmatched: Vec<String>,
    pub readiness_score: f32,
    pub suggested_bundles: Vec<&'static AppBundle>,
    pub summary: String,
}

#[derive(Debug, Clone)]
pub struct MatchedApp {
    pub source_name: String,
    pub entry: &'static AppEntry,
}

// ═══════════════════════════════════════════════════════
// The Database
// ═══════════════════════════════════════════════════════

/// The unified app database. Constructed once, used for all matching.
pub struct AppDatabase {
    entries: Vec<&'static AppEntry>,
    bundles: Vec<&'static AppBundle>,
    /// Index: lowercase name → entry index (for fast lookup)
    name_index: HashMap<String, usize>,
}

impl AppDatabase {
    /// Build the database with all known apps.
    pub fn new() -> Self {
        let entries: Vec<&'static AppEntry> = APPS.iter().collect();
        let bundles: Vec<&'static AppBundle> = BUNDLES.iter().collect();

        // Build name index from all detection fields
        let mut name_index = HashMap::new();
        for (i, entry) in entries.iter().enumerate() {
            // Index by canonical name
            name_index.insert(entry.name.to_lowercase(), i);
            // Index by all platform-specific names
            for name in entry
                .windows_names
                .iter()
                .chain(entry.macos_names.iter())
                .chain(entry.linux_names.iter())
                .chain(entry.flatpak_ids.iter())
                .chain(entry.snap_names.iter())
                .chain(entry.winget_ids.iter())
                .chain(entry.brew_names.iter())
            {
                name_index.insert(name.to_lowercase(), i);
            }
        }

        Self {
            entries,
            bundles,
            name_index,
        }
    }

    /// Match a single app name against the database.
    pub fn match_app(&self, name: &str) -> Option<&'static AppEntry> {
        let lower = name.to_lowercase().trim().to_string();

        // Exact match
        if let Some(&idx) = self.name_index.get(&lower) {
            return Some(self.entries[idx]);
        }

        // Fuzzy: try removing version numbers, publishers, etc.
        let cleaned = lower
            .split(|c: char| c == '(' || c == '[' || c == '-')
            .next()
            .unwrap_or(&lower)
            .trim();
        if let Some(&idx) = self.name_index.get(cleaned) {
            return Some(self.entries[idx]);
        }

        // Substring match (last resort)
        for (key, &idx) in &self.name_index {
            if key.len() > 3 && (lower.contains(key.as_str()) || key.contains(cleaned)) {
                return Some(self.entries[idx]);
            }
        }

        None
    }

    /// Match a list of app names (from paste or scan).
    pub fn match_list(&self, names: &[String]) -> MigrationReport {
        let mut matched = Vec::new();
        let mut unmatched = Vec::new();

        for name in names {
            if let Some(entry) = self.match_app(name) {
                matched.push(MatchedApp {
                    source_name: name.clone(),
                    entry,
                });
            } else if !name.trim().is_empty() {
                unmatched.push(name.clone());
            }
        }

        // Calculate readiness score
        let total = matched.len() + unmatched.len();
        let readiness = if total > 0 {
            let score_sum: f32 = matched
                .iter()
                .map(|m| m.entry.primary.quality.confidence())
                .sum();
            score_sum / total as f32
        } else {
            0.0
        };

        // Detect bundles
        let mut category_counts: HashMap<AppCategory, usize> = HashMap::new();
        for m in &matched {
            *category_counts.entry(m.entry.category).or_insert(0) += 1;
        }
        let suggested_bundles: Vec<&'static AppBundle> = self
            .bundles
            .iter()
            .filter(|b| {
                let count: usize = b
                    .trigger_categories
                    .iter()
                    .map(|c| category_counts.get(c).copied().unwrap_or(0))
                    .sum();
                count >= b.trigger_threshold
            })
            .copied()
            .collect();

        // Generate summary
        let native_count = matched
            .iter()
            .filter(|m| {
                m.entry.primary.quality == MatchQuality::Native
                    || m.entry.primary.quality == MatchQuality::OfficialLinux
            })
            .count();
        let alt_count = matched
            .iter()
            .filter(|m| {
                m.entry.primary.quality == MatchQuality::StrongAlternative
                    || m.entry.primary.quality == MatchQuality::PartialAlternative
            })
            .count();

        let summary = if readiness > 0.85 {
            format!(
                "{} of {} apps work natively on NixOS. Your system is highly compatible!",
                native_count, total
            )
        } else if readiness > 0.65 {
            format!(
                "{} apps work natively, {} have good alternatives. Review the alternatives below.",
                native_count, alt_count
            )
        } else if readiness > 0.45 {
            format!(
                "{} apps need alternatives. Consider dual-boot for full compatibility.",
                alt_count + unmatched.len()
            )
        } else {
            format!(
                "Many apps lack Linux equivalents ({} unmatched). Dual-boot strongly recommended.",
                unmatched.len()
            )
        };

        MigrationReport {
            total_apps: total,
            matched,
            unmatched,
            readiness_score: readiness,
            suggested_bundles,
            summary,
        }
    }

    /// Parse a raw text paste (from winget, brew, dpkg, etc.) into app names.
    pub fn parse_app_list(&self, raw_text: &str) -> Vec<String> {
        let mut apps = Vec::new();
        let lines: Vec<&str> = raw_text.lines().collect();

        if lines.is_empty() {
            return apps;
        }

        // Detect format
        if raw_text.contains("---APPS---") || raw_text.contains("---BREW---") {
            // macOS companion script format
            for line in &lines {
                if line.starts_with("---") {
                    continue;
                }
                let name = line.trim();
                if !name.is_empty() {
                    apps.push(name.to_string());
                }
            }
        } else if lines
            .iter()
            .any(|l| l.contains("winget") || l.contains("Microsoft."))
        {
            // winget list output: skip header, parse name column
            let mut past_header = false;
            for line in &lines {
                if line.contains("----") {
                    past_header = true;
                    continue;
                }
                if !past_header {
                    continue;
                }
                // winget list: Name | Id | Version | Source
                let name = line
                    .split_whitespace()
                    .take(3)
                    .collect::<Vec<_>>()
                    .join(" ");
                if !name.is_empty() {
                    apps.push(name.trim().to_string());
                }
            }
            // Also try extracting winget IDs
            for line in &lines {
                if let Some(id) = extract_winget_id(line) {
                    apps.push(id);
                }
            }
        } else if lines
            .iter()
            .any(|l| l.starts_with("ii ") || l.contains("/stable"))
        {
            // dpkg --list output
            for line in &lines {
                if line.starts_with("ii ") {
                    if let Some(pkg) = line.split_whitespace().nth(1) {
                        // Remove architecture suffix
                        let name = pkg.split(':').next().unwrap_or(pkg);
                        apps.push(name.to_string());
                    }
                }
            }
        } else if lines
            .iter()
            .any(|l| l.contains("extra/") || l.contains("core/"))
        {
            // pacman -Q output
            for line in &lines {
                if let Some(pkg) = line.split_whitespace().next() {
                    apps.push(pkg.to_string());
                }
            }
        } else if lines
            .iter()
            .any(|l| l.contains(".desktop") || l.contains("org."))
        {
            // flatpak list output
            for line in &lines {
                let id = line.trim();
                if id.contains('.') && !id.starts_with('#') {
                    apps.push(id.to_string());
                }
            }
        } else {
            // Generic: one app per line
            for line in &lines {
                let name = line.trim();
                if !name.is_empty() && !name.starts_with('#') && !name.starts_with("---") {
                    apps.push(name.to_string());
                }
            }
        }

        apps.sort();
        apps.dedup();
        apps
    }

    /// Get all entries.
    pub fn entries(&self) -> &[&'static AppEntry] {
        &self.entries
    }

    /// Get all bundles.
    pub fn bundles(&self) -> &[&'static AppBundle] {
        &self.bundles
    }
}

fn extract_winget_id(line: &str) -> Option<String> {
    // Look for Publisher.App pattern
    for word in line.split_whitespace() {
        if word.contains('.')
            && word.chars().filter(|c| *c == '.').count() >= 1
            && word
                .chars()
                .next()
                .map(|c| c.is_uppercase())
                .unwrap_or(false)
        {
            return Some(word.to_string());
        }
    }
    None
}

// ═══════════════════════════════════════════════════════
// Static Data: Apps
// ═══════════════════════════════════════════════════════

macro_rules! app {
    ($name:expr_2021, $cat:expr_2021, $prop:expr_2021,
     win: [$($w:expr_2021),*], mac: [$($m:expr_2021),*], linux: [$($l:expr_2021),*],
     flatpak: [$($fp:expr_2021),*], snap: [$($sn:expr_2021),*],
     winget: [$($wg:expr_2021),*], brew: [$($br:expr_2021),*],
     primary: ($pkg:expr_2021, $dname:expr_2021, $quality:expr_2021, $just:expr_2021, [$($tradeoff:expr_2021),*]),
     alts: [$(($apkg:expr_2021, $adname:expr_2021, $aquality:expr_2021, $ajust:expr_2021, [$($atradeoff:expr_2021),*])),*]
    ) => {
        AppEntry {
            name: $name,
            category: $cat,
            proprietary: $prop,
            windows_names: &[$($w),*],
            macos_names: &[$($m),*],
            linux_names: &[$($l),*],
            flatpak_ids: &[$($fp),*],
            snap_names: &[$($sn),*],
            winget_ids: &[$($wg),*],
            brew_names: &[$($br),*],
            primary: NixRecommendation {
                nix_pkg: $pkg, display_name: $dname, quality: $quality,
                justification: $just, trade_offs: &[$($tradeoff),*],
            },
            alternatives: &[$(NixRecommendation {
                nix_pkg: $apkg, display_name: $adname, quality: $aquality,
                justification: $ajust, trade_offs: &[$($atradeoff),*],
            }),*],
            verified_channel: "25.05",
        }
    };
}

/// The nixpkgs channel that all entries in this database were verified against.
pub const CURRENT_VERIFIED_CHANNEL: &str = "25.05";

/// Check if a package entry might be stale (verified against a different channel
/// than the target system is running). Returns true if the channels differ.
pub fn is_potentially_stale(entry: &AppEntry, current_channel: &str) -> bool {
    entry.verified_channel != current_channel
}

/// Extract the major.minor channel version from a full NixOS version string.
/// e.g., "25.05.20260401.abc1234" -> "25.05", "24.11" -> "24.11"
pub fn parse_channel_version(version_str: &str) -> &str {
    // NixOS versions look like "25.05.20260401.abc1234" or just "25.05"
    let trimmed = version_str.trim();
    // Find the second dot (after major.minor)
    let mut dots = 0;
    for (i, c) in trimmed.char_indices() {
        if c == '.' {
            dots += 1;
            if dots == 2 {
                return &trimmed[..i];
            }
        }
    }
    trimmed
}

static APPS: &[AppEntry] = &[
    // ── Browsers ──
    app!("Firefox", AppCategory::Browser, false,
        win: ["Mozilla Firefox", "Firefox"], mac: ["Firefox"], linux: ["firefox", "firefox-esr"],
        flatpak: ["org.mozilla.firefox"], snap: ["firefox"],
        winget: ["Mozilla.Firefox"], brew: ["firefox"],
        primary: ("firefox", "Firefox", MatchQuality::Native, "Same browser, same profile sync, same extensions.", []),
        alts: []
    ),
    app!("Google Chrome", AppCategory::Browser, true,
        win: ["Google Chrome", "Chrome"], mac: ["Google Chrome"], linux: ["google-chrome-stable"],
        flatpak: ["com.google.Chrome"], snap: [],
        winget: ["Google.Chrome"], brew: ["google-chrome"],
        primary: ("google-chrome", "Google Chrome", MatchQuality::Native, "Official Chrome for Linux. Same sync, same extensions.", ["Proprietary"]),
        alts: [("chromium", "Chromium", MatchQuality::StrongAlternative, "Open-source base of Chrome. No Google sync built in.", ["No Google account sync", "No Widevine DRM by default"])]
    ),
    app!("Microsoft Edge", AppCategory::Browser, true,
        win: ["Microsoft Edge"], mac: ["Microsoft Edge"], linux: ["microsoft-edge-stable"],
        flatpak: [], snap: [],
        winget: ["Microsoft.Edge"], brew: ["microsoft-edge"],
        primary: ("microsoft-edge", "Microsoft Edge", MatchQuality::OfficialLinux, "Official Edge for Linux. Same sync and features.", ["Proprietary"]),
        alts: [("firefox", "Firefox", MatchQuality::StrongAlternative, "Open-source, privacy-focused, excellent on Linux.", [])]
    ),
    app!("Brave", AppCategory::Browser, false,
        win: ["Brave"], mac: ["Brave Browser"], linux: ["brave"],
        flatpak: ["com.brave.Browser"], snap: ["brave"],
        winget: ["Brave.Brave"], brew: ["brave-browser"],
        primary: ("brave", "Brave", MatchQuality::Native, "Same browser on Linux.", []),
        alts: []
    ),
    // ── Office ──
    app!("Microsoft Office", AppCategory::Office, true,
        win: ["Microsoft Office", "Microsoft Word", "Microsoft Excel", "Microsoft PowerPoint", "Microsoft 365"],
        mac: ["Microsoft Word", "Microsoft Excel", "Microsoft PowerPoint"],
        linux: [],
        flatpak: [], snap: [],
        winget: ["Microsoft.Office"], brew: [],
        primary: ("onlyoffice-bin", "OnlyOffice", MatchQuality::StrongAlternative,
            "Best .docx/.xlsx/.pptx compatibility of any Linux office suite. Familiar ribbon interface.",
            ["Some advanced macros won't work", "No Outlook equivalent (use Thunderbird)"]),
        alts: [
            ("libreoffice", "LibreOffice", MatchQuality::StrongAlternative,
             "The open-source gold standard. Huge community. Slightly different formatting on complex documents.",
             ["Complex .docx formatting may shift", "Different UI paradigm"]),
            ("wps-office", "WPS Office", MatchQuality::StrongAlternative,
             "Very similar look to MS Office. Good compatibility. Freemium model.",
             ["Proprietary", "Some features require payment"])
        ]
    ),
    app!("Microsoft Word", AppCategory::Office, true,
        win: ["Microsoft Word", "Word"], mac: ["Microsoft Word"], linux: [],
        flatpak: [], snap: [],
        winget: ["Microsoft.Office"], brew: [],
        primary: ("onlyoffice-bin", "OnlyOffice Writer", MatchQuality::StrongAlternative,
            "Best Word compatibility. Your .docx files will look right.", ["No advanced macros"]),
        alts: [("libreoffice-still", "LibreOffice Writer", MatchQuality::StrongAlternative,
             "Rock-solid word processor. May reformat complex documents slightly.", ["Formatting differences on complex docs"])]
    ),
    // ── Development ──
    app!("Visual Studio Code", AppCategory::IDE, false,
        win: ["Visual Studio Code", "VS Code", "Code"], mac: ["Visual Studio Code"],
        linux: ["code", "code-oss"],
        flatpak: ["com.visualstudio.code"], snap: ["code"],
        winget: ["Microsoft.VisualStudioCode"], brew: ["visual-studio-code"],
        primary: ("vscode", "VS Code", MatchQuality::Native,
            "Same VS Code, same extensions, same settings sync. Your keybindings and themes carry over.", []),
        alts: [
            ("vscodium", "VSCodium", MatchQuality::StrongAlternative,
             "VS Code without Microsoft telemetry. Same extensions, fully open-source.",
             ["No Microsoft account sync", "Some extensions not in Open VSX registry"]),
            ("helix", "Helix", MatchQuality::PartialAlternative,
             "Modern terminal editor with LSP support. Very fast, zero config. Different paradigm.",
             ["Terminal-only", "Different keybindings", "No extension marketplace"])
        ]
    ),
    app!("Docker Desktop", AppCategory::Container, false,
        win: ["Docker Desktop"], mac: ["Docker Desktop", "Docker"], linux: ["docker-ce"],
        flatpak: [], snap: ["docker"],
        winget: ["Docker.DockerDesktop"], brew: ["docker"],
        primary: ("docker", "Docker Engine", MatchQuality::Native,
            "Docker runs natively on Linux — no VM needed. Faster than Docker Desktop.", ["No GUI dashboard (use Portainer or lazydocker)"]),
        alts: [
            ("podman", "Podman", MatchQuality::StrongAlternative,
             "Drop-in Docker replacement that runs rootless. Same CLI commands. More secure by default.",
             ["Some docker-compose files need podman-compose", "Slight CLI differences"])
        ]
    ),
    app!("Git", AppCategory::VersionControl, false,
        win: ["Git", "Git for Windows"], mac: ["git"], linux: ["git"],
        flatpak: [], snap: ["git"],
        winget: ["Git.Git"], brew: ["git"],
        primary: ("git", "Git", MatchQuality::Native, "Same git. Your config and keys transfer directly.", []),
        alts: []
    ),
    app!("JetBrains IntelliJ IDEA", AppCategory::IDE, true,
        win: ["IntelliJ IDEA", "JetBrains IntelliJ"], mac: ["IntelliJ IDEA"],
        linux: ["intellij-idea-ultimate"],
        flatpak: ["com.jetbrains.IntelliJ-IDEA-Ultimate"], snap: ["intellij-idea-ultimate"],
        winget: ["JetBrains.IntelliJIDEA.Ultimate"], brew: ["intellij-idea"],
        primary: ("jetbrains.idea-ultimate", "IntelliJ IDEA", MatchQuality::Native,
            "Same IDE, same plugins, same settings sync via JetBrains account.", ["Proprietary, requires license"]),
        alts: [("vscode", "VS Code", MatchQuality::StrongAlternative,
             "Free, extensible, excellent Java support via extensions.", ["Different UI", "No built-in database tools"])]
    ),
    // ── Creative ──
    app!("Adobe Photoshop", AppCategory::Creative2D, true,
        win: ["Adobe Photoshop", "Photoshop"], mac: ["Adobe Photoshop"],
        linux: [],
        flatpak: [], snap: [],
        winget: ["Adobe.Photoshop"], brew: [],
        primary: ("krita", "Krita", MatchQuality::PartialAlternative,
            "Professional digital painting. Best for illustration and concept art. Excellent brush engine.",
            ["Different UI paradigm", "Weaker photo manipulation than Photoshop", "No RAW processing"]),
        alts: [
            ("gimp", "GIMP", MatchQuality::PartialAlternative,
             "Closest to Photoshop for photo editing. Supports PSD files. Steep learning curve for PS users.",
             ["Very different UI", "No CMYK editing", "Slower for some operations"]),
            ("darktable", "darktable", MatchQuality::PartialAlternative,
             "If you mainly use Photoshop for RAW photo editing, darktable is the answer.",
             ["RAW editing only, not general image editing"])
        ]
    ),
    app!("Adobe Illustrator", AppCategory::Creative2D, true,
        win: ["Adobe Illustrator", "Illustrator"], mac: ["Adobe Illustrator"],
        linux: [],
        flatpak: [], snap: [],
        winget: ["Adobe.Illustrator"], brew: [],
        primary: ("inkscape", "Inkscape", MatchQuality::StrongAlternative,
            "Professional vector graphics. Opens/saves SVG natively. Imports AI files.",
            ["Different tools layout", "Some advanced AI features missing"]),
        alts: []
    ),
    // ── Audio ──
    app!("Spotify", AppCategory::Streaming, true,
        win: ["Spotify"], mac: ["Spotify"], linux: ["spotify-client"],
        flatpak: ["com.spotify.Client"], snap: ["spotify"],
        winget: ["Spotify.Spotify"], brew: ["spotify"],
        primary: ("spotify", "Spotify", MatchQuality::Native,
            "Official Spotify client for Linux. Same library, same playlists.", ["Proprietary"]),
        alts: [("spotifyd", "spotifyd", MatchQuality::PartialAlternative,
             "Lightweight Spotify daemon. No GUI — use spotify-tui or your phone as remote.",
             ["Terminal only", "Requires Spotify Premium"])]
    ),
    app!("Ableton Live", AppCategory::Audio, true,
        win: ["Ableton Live", "Ableton"], mac: ["Ableton Live"],
        linux: [],
        flatpak: [], snap: [],
        winget: ["Ableton.Live"], brew: [],
        primary: ("ardour", "Ardour", MatchQuality::PartialAlternative,
            "Professional DAW. Records, edits, mixes. JACK/PipeWire for low latency. Different workflow from Ableton.",
            ["Linear workflow (not session view)", "Different plugin format (LV2 vs VST)", "Steeper learning curve"]),
        alts: [
            ("lmms", "LMMS", MatchQuality::PartialAlternative,
             "Closer to Ableton's pattern-based workflow. Good for electronic music. Free.",
             ["Less professional than Ardour", "Limited mixing capabilities"]),
            ("bitwig-studio", "Bitwig Studio", MatchQuality::StrongAlternative,
             "The closest to Ableton on Linux. Made by ex-Ableton devs. Native Linux support.",
             ["Proprietary, expensive", "Smaller community than Ableton"])
        ]
    ),
    // ── Communication ──
    app!("Discord", AppCategory::Communication, true,
        win: ["Discord"], mac: ["Discord"], linux: ["discord"],
        flatpak: ["com.discordapp.Discord"], snap: ["discord"],
        winget: ["Discord.Discord"], brew: ["discord"],
        primary: ("discord", "Discord", MatchQuality::Native,
            "Official Discord client. Same servers, same voice chat.", ["Proprietary"]),
        alts: [("webcord", "WebCord", MatchQuality::StrongAlternative,
             "Discord web client wrapped with better privacy. Same features.",
             ["No official support", "Occasional feature lag"])]
    ),
    app!("Slack", AppCategory::Communication, true,
        win: ["Slack"], mac: ["Slack"], linux: ["slack"],
        flatpak: ["com.slack.Slack"], snap: ["slack"],
        winget: ["SlackTechnologies.Slack"], brew: ["slack"],
        primary: ("slack", "Slack", MatchQuality::Native,
            "Official Slack client for Linux.", ["Proprietary", "Electron-based (uses more RAM)"]),
        alts: []
    ),
    app!("Zoom", AppCategory::Communication, true,
        win: ["Zoom", "Zoom Meetings"], mac: ["Zoom", "zoom.us"],
        linux: ["zoom-us"],
        flatpak: ["us.zoom.Zoom"], snap: ["zoom-client"],
        winget: ["Zoom.Zoom"], brew: ["zoom"],
        primary: ("zoom-us", "Zoom", MatchQuality::Native,
            "Official Zoom client. Same meetings, same screen sharing.", ["Proprietary"]),
        alts: []
    ),
    app!("Signal", AppCategory::Communication, false,
        win: ["Signal"], mac: ["Signal"], linux: ["signal-desktop"],
        flatpak: ["org.signal.Signal"], snap: ["signal-desktop"],
        winget: ["OpenWhisperSystems.Signal"], brew: ["signal"],
        primary: ("signal-desktop", "Signal", MatchQuality::Native,
            "Same Signal, same encryption, same contacts.", []),
        alts: []
    ),
    app!("iMessage", AppCategory::Communication, true,
        win: [], mac: ["Messages"], linux: [],
        flatpak: [], snap: [],
        winget: [], brew: [],
        primary: ("signal-desktop", "Signal", MatchQuality::NoEquivalent,
            "iMessage is Apple-exclusive. Signal is the most similar: end-to-end encrypted, clean design.",
            ["Different network — your iMessage contacts need Signal too", "No iCloud integration"]),
        alts: [("element-desktop", "Element", MatchQuality::NoEquivalent,
             "Decentralized, self-hostable, end-to-end encrypted via Matrix protocol.",
             ["Different network", "More complex than iMessage"])]
    ),
    // ── Gaming ──
    app!("Steam", AppCategory::Gaming, true,
        win: ["Steam"], mac: ["Steam"], linux: ["steam"],
        flatpak: ["com.valvesoftware.Steam"], snap: [],
        winget: ["Valve.Steam"], brew: ["steam"],
        primary: ("steam", "Steam", MatchQuality::Native,
            "Steam runs natively on Linux. Proton plays most Windows games. Your library is the same.",
            ["Some anti-cheat games don't work (check ProtonDB)", "Performance varies per game"]),
        alts: []
    ),
    app!("Epic Games Launcher", AppCategory::Gaming, true,
        win: ["Epic Games Launcher", "Epic Games"], mac: ["Epic Games Launcher"],
        linux: [],
        flatpak: ["com.heroicgameslauncher.hgl"], snap: [],
        winget: ["EpicGames.EpicGamesLauncher"], brew: [],
        primary: ("heroic", "Heroic Games Launcher", MatchQuality::StrongAlternative,
            "Open-source launcher for Epic and GOG games. Runs Windows games via Proton.",
            ["Not official", "Some games may need manual setup"]),
        alts: [("lutris", "Lutris", MatchQuality::StrongAlternative,
             "Universal game launcher. Supports Epic, GOG, Battle.net, and more.",
             ["Manual configuration sometimes needed"])]
    ),
    // ── System / Utilities ──
    app!("7-Zip", AppCategory::Archive, false,
        win: ["7-Zip", "7zip"], mac: [], linux: ["p7zip"],
        flatpak: [], snap: [],
        winget: ["7zip.7zip"], brew: ["p7zip"],
        primary: ("p7zip", "7-Zip", MatchQuality::Native, "Same 7-Zip compression. Same formats.", []),
        alts: []
    ),
    app!("WinRAR", AppCategory::Archive, true,
        win: ["WinRAR"], mac: [], linux: [],
        flatpak: [], snap: [],
        winget: ["RARLab.WinRAR"], brew: [],
        primary: ("p7zip", "7-Zip", MatchQuality::StrongAlternative,
            "Handles RAR and every other archive format. Free and open-source.",
            []),
        alts: []
    ),
    app!("VLC", AppCategory::MediaPlayer, false,
        win: ["VLC media player", "VLC"], mac: ["VLC"], linux: ["vlc"],
        flatpak: ["org.videolan.VLC"], snap: ["vlc"],
        winget: ["VideoLAN.VLC"], brew: ["vlc"],
        primary: ("vlc", "VLC", MatchQuality::Native, "Same VLC. Plays everything.", []),
        alts: [("mpv", "mpv", MatchQuality::StrongAlternative,
             "Lighter, faster, keyboard-driven. Power users prefer it.",
             ["No GUI by default (minimal controls)", "Configured via text file"])]
    ),
    app!("OBS Studio", AppCategory::Video, false,
        win: ["OBS Studio", "OBS"], mac: ["OBS"], linux: ["obs-studio"],
        flatpak: ["com.obsproject.Studio"], snap: [],
        winget: ["OBSProject.OBSStudio"], brew: ["obs"],
        primary: ("obs-studio", "OBS Studio", MatchQuality::Native,
            "Same OBS. Same scenes, same plugins. Wayland capture works.", []),
        alts: []
    ),
    app!("Bitwarden", AppCategory::Security, false,
        win: ["Bitwarden"], mac: ["Bitwarden"], linux: ["bitwarden"],
        flatpak: ["com.bitwarden.desktop"], snap: ["bitwarden"],
        winget: ["Bitwarden.Bitwarden"], brew: ["bitwarden"],
        primary: ("bitwarden-desktop", "Bitwarden", MatchQuality::Native,
            "Same vault, same sync, same browser extension.", []),
        alts: []
    ),
    app!("KeePassXC", AppCategory::Security, false,
        win: ["KeePassXC", "KeePass"], mac: ["KeePassXC"], linux: ["keepassxc"],
        flatpak: ["org.keepassxc.KeePassXC"], snap: ["keepassxc"],
        winget: ["KeePassXCTeam.KeePassXC"], brew: ["keepassxc"],
        primary: ("keepassxc", "KeePassXC", MatchQuality::Native,
            "Same database, same format. Your .kdbx file just works.", []),
        alts: []
    ),
    // ── More Browsers ──
    app!("Vivaldi", AppCategory::Browser, true,
        win: ["Vivaldi"], mac: ["Vivaldi"], linux: ["vivaldi-stable"],
        flatpak: [], snap: [],
        winget: ["VivaldiTechnologies.Vivaldi"], brew: ["vivaldi"],
        primary: ("vivaldi", "Vivaldi", MatchQuality::Native, "Same browser, same sync.", ["Proprietary"]),
        alts: []
    ),
    app!("Opera", AppCategory::Browser, true,
        win: ["Opera"], mac: ["Opera"], linux: ["opera"],
        flatpak: [], snap: [],
        winget: ["Opera.Opera"], brew: ["opera"],
        primary: ("opera", "Opera", MatchQuality::Native, "Official Opera for Linux.", ["Proprietary"]),
        alts: []
    ),
    // ── More Communication ──
    app!("Telegram", AppCategory::Communication, false,
        win: ["Telegram Desktop", "Telegram"], mac: ["Telegram"], linux: ["telegram-desktop"],
        flatpak: ["org.telegram.desktop"], snap: ["telegram-desktop"],
        winget: ["Telegram.TelegramDesktop"], brew: ["telegram"],
        primary: ("telegram-desktop", "Telegram", MatchQuality::Native, "Same app, same chats.", []),
        alts: []
    ),
    app!("Microsoft Teams", AppCategory::Communication, true,
        win: ["Microsoft Teams", "Teams"], mac: ["Microsoft Teams"], linux: [],
        flatpak: ["com.github.niclas.teams-for-linux"], snap: [],
        winget: ["Microsoft.Teams"], brew: ["microsoft-teams"],
        primary: ("teams-for-linux", "Teams for Linux", MatchQuality::OfficialLinux,
            "Community-maintained Teams client. Same meetings, same chat.", ["Not official Microsoft build", "Occasional feature lag"]),
        alts: []
    ),
    app!("WhatsApp", AppCategory::Communication, true,
        win: ["WhatsApp"], mac: ["WhatsApp"], linux: [],
        flatpak: [], snap: [],
        winget: ["WhatsApp.WhatsApp"], brew: [],
        primary: ("whatsapp-for-linux", "WhatsApp for Linux", MatchQuality::OfficialLinux,
            "Community WhatsApp wrapper. Same account, same chats.", ["Unofficial wrapper", "Web-based"]),
        alts: []
    ),
    // ── More Dev Tools ──
    app!("Neovim", AppCategory::Editor, false,
        win: ["Neovim"], mac: ["nvim", "neovim"], linux: ["neovim", "nvim"],
        flatpak: [], snap: ["nvim"],
        winget: ["Neovim.Neovim"], brew: ["neovim"],
        primary: ("neovim", "Neovim", MatchQuality::Native, "Same Neovim. Your config transfers directly.", []),
        alts: []
    ),
    app!("Sublime Text", AppCategory::Editor, true,
        win: ["Sublime Text", "Sublime"], mac: ["Sublime Text"], linux: ["sublime-text"],
        flatpak: ["com.sublimetext.three"], snap: ["sublime-text"],
        winget: ["SublimeHQ.SublimeText.4"], brew: ["sublime-text"],
        primary: ("sublime4", "Sublime Text", MatchQuality::Native, "Same editor, same license.", ["Proprietary"]),
        alts: []
    ),
    app!("Postman", AppCategory::DevTools, true,
        win: ["Postman"], mac: ["Postman"], linux: ["postman"],
        flatpak: ["com.getpostman.Postman"], snap: ["postman"],
        winget: ["Postman.Postman"], brew: ["postman"],
        primary: ("postman", "Postman", MatchQuality::Native, "Same API testing tool.", ["Proprietary"]),
        alts: [("insomnia", "Insomnia", MatchQuality::StrongAlternative,
             "Open-source API client. Clean, fast, supports GraphQL.", ["Different UI", "Fewer integrations"])]
    ),
    app!("Node.js", AppCategory::DevTools, false,
        win: ["Node.js"], mac: ["node"], linux: ["nodejs"],
        flatpak: [], snap: ["node"],
        winget: ["OpenJS.NodeJS"], brew: ["node"],
        primary: ("nodejs", "Node.js", MatchQuality::Native, "Same Node. Use nix develop for per-project versions instead of nvm.", []),
        alts: []
    ),
    app!("Python", AppCategory::DevTools, false,
        win: ["Python", "Python 3"], mac: ["python3", "python"], linux: ["python3"],
        flatpak: [], snap: [],
        winget: ["Python.Python.3.12", "Python.Python.3.11"], brew: ["python"],
        primary: ("python3", "Python 3", MatchQuality::Native, "Same Python. Use nix develop for per-project versions instead of pyenv.", []),
        alts: []
    ),
    // ── More Creative ──
    app!("Blender", AppCategory::Creative3D, false,
        win: ["Blender"], mac: ["Blender"], linux: ["blender"],
        flatpak: ["org.blender.Blender"], snap: ["blender"],
        winget: ["BlenderFoundation.Blender"], brew: ["blender"],
        primary: ("blender", "Blender", MatchQuality::Native, "Same Blender. Same files, same addons.", []),
        alts: []
    ),
    app!("GIMP", AppCategory::Creative2D, false,
        win: ["GIMP"], mac: ["GIMP"], linux: ["gimp"],
        flatpak: ["org.gimp.GIMP"], snap: ["gimp"],
        winget: ["GIMP.GIMP"], brew: ["gimp"],
        primary: ("gimp", "GIMP", MatchQuality::Native, "Same GIMP. Same plugins, same files.", []),
        alts: []
    ),
    app!("Inkscape", AppCategory::Creative2D, false,
        win: ["Inkscape"], mac: ["Inkscape"], linux: ["inkscape"],
        flatpak: ["org.inkscape.Inkscape"], snap: ["inkscape"],
        winget: ["Inkscape.Inkscape"], brew: ["inkscape"],
        primary: ("inkscape", "Inkscape", MatchQuality::Native, "Same Inkscape. SVG editing is identical.", []),
        alts: []
    ),
    app!("Audacity", AppCategory::Audio, false,
        win: ["Audacity"], mac: ["Audacity"], linux: ["audacity"],
        flatpak: ["org.audacityteam.Audacity"], snap: ["audacity"],
        winget: ["Audacity.Audacity"], brew: ["audacity"],
        primary: ("audacity", "Audacity", MatchQuality::Native, "Same Audacity. Same project files.", []),
        alts: []
    ),
    app!("Kdenlive", AppCategory::Video, false,
        win: ["Kdenlive"], mac: ["Kdenlive"], linux: ["kdenlive"],
        flatpak: ["org.kde.kdenlive"], snap: ["kdenlive"],
        winget: ["KDE.Kdenlive"], brew: ["kdenlive"],
        primary: ("kdenlive", "Kdenlive", MatchQuality::Native, "Same video editor. Same project files.", []),
        alts: []
    ),
    app!("HandBrake", AppCategory::Video, false,
        win: ["HandBrake"], mac: ["HandBrake"], linux: ["handbrake"],
        flatpak: ["fr.handbrake.ghb"], snap: ["handbrake-jz"],
        winget: ["HandBrake.HandBrake"], brew: ["handbrake"],
        primary: ("handbrake", "HandBrake", MatchQuality::Native, "Same transcoder.", []),
        alts: []
    ),
    // ── More Media ──
    app!("Thunderbird", AppCategory::Email, false,
        win: ["Mozilla Thunderbird", "Thunderbird"], mac: ["Thunderbird"], linux: ["thunderbird"],
        flatpak: ["org.mozilla.Thunderbird"], snap: ["thunderbird"],
        winget: ["Mozilla.Thunderbird"], brew: ["thunderbird"],
        primary: ("thunderbird", "Thunderbird", MatchQuality::Native,
            "Same email client. Your profile transfers directly — emails, accounts, calendar.", []),
        alts: []
    ),
    app!("qBittorrent", AppCategory::FileManager, false,
        win: ["qBittorrent"], mac: ["qBittorrent"], linux: ["qbittorrent"],
        flatpak: ["org.qbittorrent.qBittorrent"], snap: [],
        winget: ["qBittorrent.qBittorrent"], brew: ["qbittorrent"],
        primary: ("qbittorrent", "qBittorrent", MatchQuality::Native, "Same torrent client.", []),
        alts: []
    ),
    app!("FileZilla", AppCategory::FileManager, false,
        win: ["FileZilla"], mac: ["FileZilla"], linux: ["filezilla"],
        flatpak: ["org.filezilla-project.Filezilla"], snap: [],
        winget: ["TimKosse.FileZilla.Client"], brew: ["filezilla"],
        primary: ("filezilla", "FileZilla", MatchQuality::Native, "Same FTP client.", []),
        alts: []
    ),
    // ── More System ──
    app!("Notepad++", AppCategory::Editor, false,
        win: ["Notepad++", "Notepad Plus Plus"], mac: [], linux: [],
        flatpak: [], snap: [],
        winget: ["Notepad++.Notepad++"], brew: [],
        primary: ("vscode", "VS Code", MatchQuality::StrongAlternative,
            "VS Code is the most popular Notepad++ alternative. Far more powerful, with extensions.",
            ["Much heavier than Notepad++", "Different keybindings"]),
        alts: [
            ("kate", "Kate", MatchQuality::StrongAlternative,
             "KDE's text editor. Lightweight, fast, syntax highlighting. Closest feel to Notepad++.",
             ["KDE dependency"]),
            ("gedit", "gedit", MatchQuality::StrongAlternative,
             "GNOME's text editor. Simple, clean, fast. Good for quick edits.",
             ["Fewer features than Notepad++"])
        ]
    ),
    app!("PuTTY", AppCategory::SystemUtil, false,
        win: ["PuTTY"], mac: [], linux: [],
        flatpak: [], snap: [],
        winget: ["SimonTatham.PuTTY"], brew: [],
        primary: ("openssh", "OpenSSH (built-in)", MatchQuality::Native,
            "Linux has SSH built into the terminal. No need for PuTTY. Just type: ssh user@host",
            []),
        alts: []
    ),
    // ── macOS-specific ──
    app!("Xcode", AppCategory::IDE, true,
        win: [], mac: ["Xcode"], linux: [],
        flatpak: [], snap: [],
        winget: [], brew: [],
        primary: ("vscode", "VS Code", MatchQuality::PartialAlternative,
            "VS Code with language-specific extensions. Apple platform SDKs are macOS-only.",
            ["No iOS/macOS app development", "No Interface Builder", "No Instruments"]),
        alts: []
    ),
    app!("Final Cut Pro", AppCategory::Video, true,
        win: [], mac: ["Final Cut Pro"], linux: [],
        flatpak: [], snap: [],
        winget: [], brew: [],
        primary: ("kdenlive", "Kdenlive", MatchQuality::PartialAlternative,
            "Professional video editor. Timeline-based like Final Cut. Free.",
            ["Different UI", "Fewer effects", "No magnetic timeline"]),
        alts: [
            ("davinci-resolve", "DaVinci Resolve", MatchQuality::StrongAlternative,
             "Industry-standard color grading + editing. Professional quality. Free version available.",
             ["Proprietary", "Heavy resource usage", "Free version limited to H.264"])
        ]
    ),
    app!("GarageBand", AppCategory::Audio, true,
        win: [], mac: ["GarageBand"], linux: [],
        flatpak: [], snap: [],
        winget: [], brew: [],
        primary: ("lmms", "LMMS", MatchQuality::StrongAlternative,
            "Music production similar to GarageBand. Built-in instruments, beat sequencer, effects.",
            ["Different interface", "Fewer built-in loops"]),
        alts: [("ardour", "Ardour", MatchQuality::PartialAlternative,
             "Professional DAW. More powerful than GarageBand but steeper learning curve.",
             ["More complex", "Requires audio knowledge"])]
    ),
    // ── More Browsers ──
    app!("Chromium", AppCategory::Browser, false,
        win: ["Chromium"], mac: ["Chromium"], linux: ["chromium", "chromium-browser"],
        flatpak: ["org.chromium.Chromium"], snap: ["chromium"],
        winget: [], brew: ["chromium"],
        primary: ("chromium", "Chromium", MatchQuality::Native,
            "Same open-source browser. No Google sync, but all Chrome extensions work.", []),
        alts: []
    ),
    app!("Tor Browser", AppCategory::Browser, false,
        win: ["Tor Browser"], mac: ["Tor Browser"], linux: ["tor-browser"],
        flatpak: ["com.github.nickvdp.torbrowser-launcher"], snap: [],
        winget: ["TorProject.TorBrowser"], brew: ["tor-browser"],
        primary: ("tor-browser", "Tor Browser", MatchQuality::Native,
            "Same Tor Browser for anonymous browsing. Identical privacy guarantees.", []),
        alts: []
    ),
    app!("Waterfox", AppCategory::Browser, false,
        win: ["Waterfox"], mac: ["Waterfox"], linux: ["waterfox"],
        flatpak: [], snap: [],
        winget: ["Waterfox.Waterfox"], brew: ["waterfox"],
        primary: ("waterfox", "Waterfox", MatchQuality::Native,
            "Same privacy-focused Firefox fork. Supports legacy extensions.", []),
        alts: [("firefox", "Firefox", MatchQuality::StrongAlternative,
             "Mainstream Firefox with broader extension support and faster updates.",
             ["No legacy XUL extension support"])]
    ),
    // ── Office / Productivity ──
    app!("LibreOffice", AppCategory::Office, false,
        win: ["LibreOffice"], mac: ["LibreOffice"], linux: ["libreoffice", "libreoffice-still", "libreoffice-fresh"],
        flatpak: ["org.libreoffice.LibreOffice"], snap: ["libreoffice"],
        winget: ["TheDocumentFoundation.LibreOffice"], brew: ["libreoffice"],
        primary: ("libreoffice", "LibreOffice", MatchQuality::Native,
            "Same LibreOffice suite. Writer, Calc, Impress, Draw — all your files carry over.", []),
        alts: []
    ),
    app!("Obsidian", AppCategory::Notes, true,
        win: ["Obsidian"], mac: ["Obsidian"], linux: ["obsidian"],
        flatpak: ["md.obsidian.Obsidian"], snap: ["obsidian"],
        winget: ["Obsidian.Obsidian"], brew: ["obsidian"],
        primary: ("obsidian", "Obsidian", MatchQuality::Native,
            "Same note-taking app. Your vault is just a folder of Markdown files — works everywhere.",
            ["Proprietary"]),
        alts: [("logseq", "Logseq", MatchQuality::StrongAlternative,
             "Open-source knowledge management. Outliner-based, local-first, Markdown/Org-mode files.",
             ["Different UI paradigm (outliner vs freeform)", "Smaller plugin ecosystem"])]
    ),
    app!("Logseq", AppCategory::Notes, false,
        win: ["Logseq"], mac: ["Logseq"], linux: ["logseq"],
        flatpak: ["com.logseq.Logseq"], snap: [],
        winget: ["Logseq.Logseq"], brew: ["logseq"],
        primary: ("logseq", "Logseq", MatchQuality::Native,
            "Same knowledge management tool. Your Markdown/Org-mode graph works identically.", []),
        alts: []
    ),
    app!("Notion", AppCategory::Notes, true,
        win: ["Notion"], mac: ["Notion"], linux: [],
        flatpak: [], snap: [],
        winget: ["Notion.Notion"], brew: ["notion"],
        primary: ("notion-app-enhanced", "Notion (Web)", MatchQuality::WebApp,
            "Notion works via the browser or community wrapper on Linux. Same workspace, same collaboration.",
            ["No official desktop client for Linux", "Web-based wrapper"]),
        alts: [("obsidian", "Obsidian", MatchQuality::StrongAlternative,
             "Local-first note taking with offline support and a plugin ecosystem.",
             ["No real-time collaboration", "Different paradigm (files vs database)"])]
    ),
    app!("Calibre", AppCategory::Office, false,
        win: ["calibre", "Calibre"], mac: ["calibre"], linux: ["calibre"],
        flatpak: ["com.calibre_ebook.calibre"], snap: ["calibre"],
        winget: ["calibre.calibre"], brew: ["calibre"],
        primary: ("calibre", "Calibre", MatchQuality::Native,
            "Same e-book manager. Library, converter, reader — everything transfers.", []),
        alts: []
    ),
    app!("Zotero", AppCategory::Science, false,
        win: ["Zotero"], mac: ["Zotero"], linux: ["zotero"],
        flatpak: ["org.zotero.Zotero"], snap: [],
        winget: ["Zotero.Zotero"], brew: ["zotero"],
        primary: ("zotero", "Zotero", MatchQuality::Native,
            "Same reference manager. Your library syncs via Zotero account.", []),
        alts: []
    ),
    // ── More Development ──
    app!("Vim", AppCategory::Editor, false,
        win: ["Vim", "gVim"], mac: ["vim", "macvim"], linux: ["vim", "vim-gtk3", "gvim"],
        flatpak: [], snap: [],
        winget: ["vim.vim"], brew: ["vim", "macvim"],
        primary: ("vim", "Vim", MatchQuality::Native, "Same Vim. Your .vimrc transfers directly.", []),
        alts: [("neovim", "Neovim", MatchQuality::StrongAlternative,
             "Modern rewrite of Vim with Lua config, built-in LSP, and treesitter. Reads your .vimrc.",
             ["Some Vim plugins need adaptation"])]
    ),
    app!("Emacs", AppCategory::Editor, false,
        win: ["Emacs", "GNU Emacs"], mac: ["Emacs"], linux: ["emacs", "emacs-gtk", "emacs-nox"],
        flatpak: ["org.gnu.emacs"], snap: ["emacs"],
        winget: ["GNU.Emacs"], brew: ["emacs"],
        primary: ("emacs", "Emacs", MatchQuality::Native,
            "Same Emacs. Your .emacs.d or Doom/Spacemacs config carries over unchanged.", []),
        alts: []
    ),
    app!("Helix", AppCategory::Editor, false,
        win: ["Helix"], mac: ["helix"], linux: ["helix"],
        flatpak: [], snap: [],
        winget: [], brew: ["helix"],
        primary: ("helix", "Helix", MatchQuality::Native,
            "Same Helix editor. Built-in LSP, treesitter, and multi-cursor out of the box.", []),
        alts: []
    ),
    app!("Zed", AppCategory::IDE, false,
        win: ["Zed"], mac: ["Zed"], linux: ["zed"],
        flatpak: ["dev.zed.Zed"], snap: [],
        winget: [], brew: ["zed"],
        primary: ("zed-editor", "Zed", MatchQuality::Native,
            "Same GPU-accelerated editor. Fast, collaborative, with built-in AI assistant.", []),
        alts: []
    ),
    app!("Cursor", AppCategory::IDE, true,
        win: ["Cursor"], mac: ["Cursor"], linux: ["cursor"],
        flatpak: [], snap: [],
        winget: [], brew: ["cursor"],
        primary: ("vscode", "VS Code", MatchQuality::StrongAlternative,
            "VS Code is the base Cursor is built on. Same extensions, with AI via Copilot or Continue.",
            ["No built-in Cursor AI features", "Need separate AI extension"]),
        alts: [("zed-editor", "Zed", MatchQuality::StrongAlternative,
             "GPU-accelerated editor with built-in AI assistant. Native Linux support.",
             ["Different keybindings", "Smaller extension ecosystem"])]
    ),
    app!("Rust/rustup", AppCategory::DevTools, false,
        win: ["rustup", "Rust"], mac: ["rustup", "rust"], linux: ["rustup", "rustc", "cargo"],
        flatpak: [], snap: [],
        winget: ["Rustlang.Rustup"], brew: ["rustup-init"],
        primary: ("rustup", "Rust (rustup)", MatchQuality::Native,
            "Same rustup toolchain manager. Your toolchains and components carry over. Even better: use nix develop for per-project Rust versions.", []),
        alts: []
    ),
    app!("Go", AppCategory::DevTools, false,
        win: ["Go Programming Language", "Go"], mac: ["go", "golang"], linux: ["golang", "golang-go"],
        flatpak: [], snap: ["go"],
        winget: ["GoLang.Go"], brew: ["go"],
        primary: ("go", "Go", MatchQuality::Native,
            "Same Go toolchain. Use nix develop for per-project Go versions.", []),
        alts: []
    ),
    app!("Java/JDK", AppCategory::DevTools, false,
        win: ["Java", "JDK", "OpenJDK", "Oracle JDK"], mac: ["java", "openjdk"], linux: ["openjdk", "default-jdk"],
        flatpak: [], snap: [],
        winget: ["EclipseAdoptium.Temurin.21.JDK", "Oracle.JDK.21"], brew: ["openjdk"],
        primary: ("jdk", "OpenJDK", MatchQuality::Native,
            "Same JDK. Nix makes it easy to have multiple JDK versions per project via nix develop.", []),
        alts: []
    ),
    app!("Ruby", AppCategory::DevTools, false,
        win: ["Ruby", "RubyInstaller"], mac: ["ruby"], linux: ["ruby", "ruby-full"],
        flatpak: [], snap: [],
        winget: ["RubyInstallerTeam.Ruby.3.3"], brew: ["ruby"],
        primary: ("ruby", "Ruby", MatchQuality::Native,
            "Same Ruby interpreter. Use nix develop for per-project Ruby versions instead of rbenv.", []),
        alts: []
    ),
    app!("Alacritty", AppCategory::Terminal, false,
        win: ["Alacritty"], mac: ["Alacritty"], linux: ["alacritty"],
        flatpak: ["org.alacritty.Alacritty"], snap: ["alacritty"],
        winget: ["Alacritty.Alacritty"], brew: ["alacritty"],
        primary: ("alacritty", "Alacritty", MatchQuality::Native,
            "Same GPU-accelerated terminal. Your alacritty.toml config transfers directly.", []),
        alts: []
    ),
    app!("Kitty", AppCategory::Terminal, false,
        win: [], mac: ["Kitty", "kitty"], linux: ["kitty"],
        flatpak: [], snap: [],
        winget: [], brew: ["kitty"],
        primary: ("kitty", "Kitty", MatchQuality::Native,
            "Same GPU-accelerated terminal with image rendering. Config carries over.", []),
        alts: []
    ),
    app!("WezTerm", AppCategory::Terminal, false,
        win: ["WezTerm"], mac: ["WezTerm"], linux: ["wezterm"],
        flatpak: ["org.wezfurlong.wezterm"], snap: [],
        winget: ["wez.wezterm"], brew: ["wezterm"],
        primary: ("wezterm", "WezTerm", MatchQuality::Native,
            "Same terminal with Lua config, multiplexing, and ligature support.", []),
        alts: []
    ),
    app!("GitHub Desktop", AppCategory::VersionControl, true,
        win: ["GitHub Desktop"], mac: ["GitHub Desktop"], linux: [],
        flatpak: [], snap: [],
        winget: ["GitHub.GitHubDesktop"], brew: ["github"],
        primary: ("lazygit", "lazygit", MatchQuality::StrongAlternative,
            "GitHub Desktop has no Linux version. lazygit is a fast terminal Git client with visual staging, committing, and branching.",
            ["Terminal-based", "Different UI paradigm"]),
        alts: [("gittyup", "Gittyup", MatchQuality::StrongAlternative,
             "Graphical git client for Linux. Visual commit history, staging, and branching.",
             ["Smaller community", "Fewer integrations than GitHub Desktop"])]
    ),
    app!("Podman", AppCategory::Container, false,
        win: ["Podman", "Podman Desktop"], mac: ["Podman Desktop"], linux: ["podman"],
        flatpak: ["io.podman_desktop.PodmanDesktop"], snap: [],
        winget: ["RedHat.Podman"], brew: ["podman"],
        primary: ("podman", "Podman", MatchQuality::Native,
            "Same rootless container engine. Drop-in Docker replacement. Even better on Linux — no VM needed.", []),
        alts: []
    ),
    app!("Vagrant", AppCategory::DevTools, false,
        win: ["Vagrant", "HashiCorp Vagrant"], mac: ["vagrant"], linux: ["vagrant"],
        flatpak: [], snap: [],
        winget: ["Hashicorp.Vagrant"], brew: ["vagrant"],
        primary: ("vagrant", "Vagrant", MatchQuality::Native,
            "Same Vagrant. Your Vagrantfiles work identically.", []),
        alts: []
    ),
    app!("Terraform", AppCategory::DevTools, false,
        win: ["Terraform", "HashiCorp Terraform"], mac: ["terraform"], linux: ["terraform"],
        flatpak: [], snap: [],
        winget: ["Hashicorp.Terraform"], brew: ["terraform"],
        primary: ("terraform", "Terraform", MatchQuality::Native,
            "Same Terraform. Your .tf files and state work identically.", []),
        alts: [("opentofu", "OpenTofu", MatchQuality::StrongAlternative,
             "Open-source fork of Terraform. Drop-in compatible, community-governed.",
             ["Slight feature lag behind Terraform"])]
    ),
    app!("Ansible", AppCategory::DevTools, false,
        win: [], mac: ["ansible"], linux: ["ansible"],
        flatpak: [], snap: [],
        winget: [], brew: ["ansible"],
        primary: ("ansible", "Ansible", MatchQuality::Native,
            "Same Ansible. Playbooks, roles, and inventory work identically. Even better on Linux — native SSH.", []),
        alts: []
    ),
    // ── More Creative ──
    app!("Krita", AppCategory::Creative2D, false,
        win: ["Krita"], mac: ["Krita"], linux: ["krita"],
        flatpak: ["org.kde.krita"], snap: ["krita"],
        winget: ["KDE.Krita"], brew: ["krita"],
        primary: ("krita", "Krita", MatchQuality::Native,
            "Same digital painting app. Same brushes, same files.", []),
        alts: []
    ),
    app!("DaVinci Resolve", AppCategory::Video, true,
        win: ["DaVinci Resolve"], mac: ["DaVinci Resolve"], linux: ["davinci-resolve"],
        flatpak: [], snap: [],
        winget: ["Blackmagic.DaVinciResolve"], brew: [],
        primary: ("davinci-resolve", "DaVinci Resolve", MatchQuality::Native,
            "Official Linux build. Industry-standard color grading, editing, VFX, and audio post.",
            ["Proprietary", "Free version limited to H.264 and 4K", "Requires AMD/NVIDIA GPU"]),
        alts: [("kdenlive", "Kdenlive", MatchQuality::StrongAlternative,
             "Fully open-source video editor. Simpler but capable for most editing tasks.",
             ["No color grading panel", "Fewer effects"])]
    ),
    app!("darktable", AppCategory::Photo, false,
        win: ["darktable"], mac: ["darktable"], linux: ["darktable"],
        flatpak: ["org.darktable.Darktable"], snap: ["darktable"],
        winget: ["darktable.darktable"], brew: ["darktable"],
        primary: ("darktable", "darktable", MatchQuality::Native,
            "Same RAW photo editor. Your library and edits carry over.", []),
        alts: []
    ),
    app!("RawTherapee", AppCategory::Photo, false,
        win: ["RawTherapee"], mac: ["RawTherapee"], linux: ["rawtherapee"],
        flatpak: ["com.rawtherapee.RawTherapee"], snap: ["rawtherapee"],
        winget: ["RawTherapee.RawTherapee"], brew: ["rawtherapee"],
        primary: ("rawtherapee", "RawTherapee", MatchQuality::Native,
            "Same RAW processor. Non-destructive editing pipeline carries over.", []),
        alts: []
    ),
    app!("Shotcut", AppCategory::Video, false,
        win: ["Shotcut"], mac: ["Shotcut"], linux: ["shotcut"],
        flatpak: ["org.shotcut.Shotcut"], snap: ["shotcut"],
        winget: ["Meltytech.Shotcut"], brew: ["shotcut"],
        primary: ("shotcut", "Shotcut", MatchQuality::Native,
            "Same cross-platform video editor. Timeline and filters work identically.", []),
        alts: []
    ),
    app!("Natron", AppCategory::Video, false,
        win: ["Natron"], mac: ["Natron"], linux: ["natron"],
        flatpak: ["fr.natron.Natron"], snap: [],
        winget: [], brew: ["natron"],
        primary: ("natron", "Natron", MatchQuality::Native,
            "Open-source compositing. Node-based workflow similar to Nuke/After Effects.", []),
        alts: []
    ),
    app!("Ardour", AppCategory::Audio, false,
        win: ["Ardour"], mac: ["Ardour"], linux: ["ardour"],
        flatpak: ["org.ardour.Ardour"], snap: [],
        winget: [], brew: ["ardour"],
        primary: ("ardour", "Ardour", MatchQuality::Native,
            "Same professional DAW. Records, edits, mixes. JACK/PipeWire for pro audio.", []),
        alts: []
    ),
    app!("LMMS", AppCategory::Audio, false,
        win: ["LMMS"], mac: ["LMMS"], linux: ["lmms"],
        flatpak: ["io.lmms.LMMS"], snap: ["lmms"],
        winget: ["LMMS.LMMS"], brew: ["lmms"],
        primary: ("lmms", "LMMS", MatchQuality::Native,
            "Same beat-making and music production. Built-in synths, sampler, beat sequencer.", []),
        alts: []
    ),
    app!("Reaper", AppCategory::Audio, true,
        win: ["REAPER", "Reaper"], mac: ["REAPER"], linux: ["reaper"],
        flatpak: [], snap: [],
        winget: ["Cockos.REAPER"], brew: ["reaper"],
        primary: ("reaper", "REAPER", MatchQuality::Native,
            "Official Linux build. Same DAW, same project files, same extensions.",
            ["Proprietary (generous evaluation license)"]),
        alts: [("ardour", "Ardour", MatchQuality::StrongAlternative,
             "Fully open-source professional DAW. Records, edits, mixes with JACK/PipeWire.",
             ["Different workflow", "Different plugin format preferences"])]
    ),
    app!("MuseScore", AppCategory::Audio, false,
        win: ["MuseScore", "MuseScore 4"], mac: ["MuseScore 4"], linux: ["musescore"],
        flatpak: ["org.musescore.MuseScore"], snap: ["musescore"],
        winget: ["Musescore.Musescore"], brew: ["musescore"],
        primary: ("musescore", "MuseScore", MatchQuality::Native,
            "Same music notation software. Scores, parts, and MIDI playback all carry over.", []),
        alts: []
    ),
    // ── More Communication ──
    app!("Element", AppCategory::Communication, false,
        win: ["Element"], mac: ["Element"], linux: ["element-desktop"],
        flatpak: ["im.riot.Riot"], snap: ["element-desktop"],
        winget: ["Element.Element"], brew: ["element"],
        primary: ("element-desktop", "Element", MatchQuality::Native,
            "Same Matrix client. End-to-end encrypted, decentralized, self-hostable.", []),
        alts: []
    ),
    app!("Mumble", AppCategory::Communication, false,
        win: ["Mumble"], mac: ["Mumble"], linux: ["mumble"],
        flatpak: [], snap: [],
        winget: ["Mumble.Mumble"], brew: ["mumble"],
        primary: ("mumble", "Mumble", MatchQuality::Native,
            "Same low-latency voice chat. Same server connections.", []),
        alts: []
    ),
    app!("Jitsi Meet", AppCategory::Communication, false,
        win: ["Jitsi Meet"], mac: ["Jitsi Meet"], linux: [],
        flatpak: [], snap: [],
        winget: [], brew: [],
        primary: ("jitsi-meet", "Jitsi Meet (Web)", MatchQuality::WebApp,
            "Jitsi is browser-based — no app needed. Self-hostable video conferencing.",
            ["Browser-only, no desktop client"]),
        alts: []
    ),
    app!("Mattermost", AppCategory::Communication, false,
        win: ["Mattermost"], mac: ["Mattermost"], linux: ["mattermost-desktop"],
        flatpak: ["com.mattermost.Desktop"], snap: ["mattermost-desktop"],
        winget: ["Mattermost.MattermostDesktop"], brew: ["mattermost"],
        primary: ("mattermost-desktop", "Mattermost", MatchQuality::Native,
            "Same team chat client. Self-hosted Slack alternative. Same server, same channels.", []),
        alts: []
    ),
    // ── More Gaming ──
    app!("Lutris", AppCategory::GamingTools, false,
        win: [], mac: [], linux: ["lutris"],
        flatpak: ["net.lutris.Lutris"], snap: [],
        winget: [], brew: [],
        primary: ("lutris", "Lutris", MatchQuality::Native,
            "Universal game launcher for Linux. Manages Wine, Proton, emulators, and native games.", []),
        alts: []
    ),
    app!("Wine", AppCategory::GamingTools, false,
        win: [], mac: ["wine-stable"], linux: ["wine", "wine-stable", "wine64"],
        flatpak: [], snap: [],
        winget: [], brew: ["wine-stable"],
        primary: ("wine", "Wine", MatchQuality::Native,
            "Same Windows compatibility layer. Run Windows apps directly on Linux.", []),
        alts: [("bottles", "Bottles", MatchQuality::StrongAlternative,
             "Modern Wine frontend with per-app prefix management and dependency handling.",
             ["GUI wrapper — adds overhead"])]
    ),
    app!("ProtonGE", AppCategory::GamingTools, false,
        win: [], mac: [], linux: [],
        flatpak: [], snap: [],
        winget: [], brew: [],
        primary: ("protonup-qt", "ProtonUp-Qt", MatchQuality::Native,
            "Manages Proton-GE and Wine-GE installations for Steam and Lutris. Latest compatibility patches.", []),
        alts: []
    ),
    app!("Minecraft", AppCategory::Gaming, true,
        win: ["Minecraft Launcher", "Minecraft"], mac: ["Minecraft"], linux: ["minecraft-launcher"],
        flatpak: ["com.mojang.Minecraft"], snap: [],
        winget: ["Mojang.MinecraftLauncher"], brew: ["minecraft"],
        primary: ("prismlauncher", "Prism Launcher", MatchQuality::StrongAlternative,
            "Open-source Minecraft launcher. Manages instances, mods, and resource packs. Supports Microsoft accounts.",
            ["Unofficial launcher", "Requires owning Minecraft"]),
        alts: [("minecraft-server", "Minecraft Server", MatchQuality::Native,
             "Official dedicated server in nixpkgs. Host your own world.",
             ["Server only, not the client"])]
    ),
    app!("RetroArch", AppCategory::Gaming, false,
        win: ["RetroArch"], mac: ["RetroArch"], linux: ["retroarch"],
        flatpak: ["org.libretro.RetroArch"], snap: ["retroarch"],
        winget: ["Libretro.RetroArch"], brew: ["retroarch"],
        primary: ("retroarch", "RetroArch", MatchQuality::Native,
            "Same multi-system emulator frontend. Core library, shaders, and saves carry over.", []),
        alts: []
    ),
    // ── More System / Utility ──
    app!("htop", AppCategory::SystemUtil, false,
        win: [], mac: ["htop"], linux: ["htop"],
        flatpak: [], snap: [],
        winget: [], brew: ["htop"],
        primary: ("htop", "htop", MatchQuality::Native,
            "Same interactive process viewer. Even better: try btop for a modern alternative.", []),
        alts: [("btop", "btop++", MatchQuality::StrongAlternative,
             "Modern resource monitor with CPU, memory, disk, network, and GPU graphs.",
             ["Different UI (more graphical)"])]
    ),
    app!("Timeshift", AppCategory::Backup, false,
        win: [], mac: [], linux: ["timeshift"],
        flatpak: [], snap: [],
        winget: [], brew: [],
        primary: ("timeshift", "Timeshift", MatchQuality::Native,
            "Same system snapshot tool. Works with btrfs and rsync backends.",
            ["NixOS has built-in rollback via generations — Timeshift is optional"]),
        alts: []
    ),
    app!("BleachBit", AppCategory::SystemUtil, false,
        win: ["BleachBit"], mac: [], linux: ["bleachbit"],
        flatpak: ["org.bleachbit.BleachBit"], snap: [],
        winget: ["BleachBit.BleachBit"], brew: [],
        primary: ("bleachbit", "BleachBit", MatchQuality::Native,
            "Same disk cleaner. Frees cache, logs, and temp files.", []),
        alts: []
    ),
    app!("Syncthing", AppCategory::Backup, false,
        win: ["Syncthing"], mac: ["Syncthing"], linux: ["syncthing"],
        flatpak: [], snap: ["syncthing"],
        winget: ["Syncthing.Syncthing"], brew: ["syncthing"],
        primary: ("syncthing", "Syncthing", MatchQuality::Native,
            "Same decentralized file sync. NixOS has a built-in module: services.syncthing.enable = true.", []),
        alts: []
    ),
    app!("Nextcloud Client", AppCategory::Backup, false,
        win: ["Nextcloud", "Nextcloud Desktop"], mac: ["Nextcloud"], linux: ["nextcloud-desktop"],
        flatpak: ["com.nextcloud.desktopclient.nextcloud"], snap: ["nextcloud-desktop-client"],
        winget: ["Nextcloud.NextcloudDesktop"], brew: ["nextcloud"],
        primary: ("nextcloud-client", "Nextcloud Client", MatchQuality::Native,
            "Same sync client. Your Nextcloud server connection transfers directly.", []),
        alts: []
    ),
    app!("WireGuard", AppCategory::VPN, false,
        win: ["WireGuard"], mac: ["WireGuard"], linux: ["wireguard-tools"],
        flatpak: [], snap: [],
        winget: ["WireGuard.WireGuard"], brew: ["wireguard-tools"],
        primary: ("wireguard-tools", "WireGuard", MatchQuality::Native,
            "Same VPN. NixOS has first-class WireGuard support: networking.wireguard.interfaces.", []),
        alts: []
    ),
    app!("Mullvad VPN", AppCategory::VPN, true,
        win: ["Mullvad VPN"], mac: ["Mullvad VPN"], linux: ["mullvad-vpn"],
        flatpak: [], snap: [],
        winget: ["MullvadVPN.MullvadVPN"], brew: ["mullvadvpn"],
        primary: ("mullvad-vpn", "Mullvad VPN", MatchQuality::Native,
            "Official Linux client. Same account, same servers, same privacy.", ["Proprietary client"]),
        alts: []
    ),
    app!("ProtonVPN", AppCategory::VPN, true,
        win: ["ProtonVPN", "Proton VPN"], mac: ["ProtonVPN"], linux: ["protonvpn"],
        flatpak: [], snap: [],
        winget: ["Proton.ProtonVPN"], brew: ["protonvpn"],
        primary: ("protonvpn-gui", "ProtonVPN", MatchQuality::Native,
            "Official Linux client. Same account, same Secure Core servers.", ["Proprietary client"]),
        alts: []
    ),
    app!("Remmina", AppCategory::SystemUtil, false,
        win: [], mac: [], linux: ["remmina"],
        flatpak: ["org.remmina.Remmina"], snap: ["remmina"],
        winget: [], brew: [],
        primary: ("remmina", "Remmina", MatchQuality::Native,
            "Remote desktop client supporting RDP, VNC, SSH, and SPICE. The Linux go-to for remote access.", []),
        alts: []
    ),
    app!("VirtualBox", AppCategory::Virtualization, false,
        win: ["Oracle VM VirtualBox", "VirtualBox"], mac: ["VirtualBox"], linux: ["virtualbox"],
        flatpak: [], snap: [],
        winget: ["Oracle.VirtualBox"], brew: ["virtualbox"],
        primary: ("virtualbox", "VirtualBox", MatchQuality::Native,
            "Same VM hypervisor. NixOS module: virtualisation.virtualbox.host.enable = true.",
            ["Kernel module needs NixOS configuration"]),
        alts: [("virt-manager", "virt-manager (QEMU/KVM)", MatchQuality::StrongAlternative,
             "KVM-based virtualization — faster than VirtualBox. Native Linux hypervisor.",
             ["Different UI", "No direct .vdi import"])]
    ),
    app!("QEMU/virt-manager", AppCategory::Virtualization, false,
        win: [], mac: ["qemu"], linux: ["qemu", "virt-manager", "qemu-kvm"],
        flatpak: [], snap: [],
        winget: [], brew: ["qemu"],
        primary: ("virt-manager", "virt-manager", MatchQuality::Native,
            "QEMU/KVM with a graphical manager. Near-native VM performance via hardware virtualization.", []),
        alts: []
    ),
    // ── More Media ──
    app!("mpv", AppCategory::MediaPlayer, false,
        win: ["mpv"], mac: ["mpv"], linux: ["mpv"],
        flatpak: ["io.mpv.Mpv"], snap: ["mpv"],
        winget: ["mpv.net"], brew: ["mpv"],
        primary: ("mpv", "mpv", MatchQuality::Native,
            "Same lightweight, scriptable media player. Hardware decoding, Lua scripting, minimal UI.", []),
        alts: []
    ),
    app!("Plex Media Server", AppCategory::MediaPlayer, true,
        win: ["Plex Media Server", "Plex"], mac: ["Plex Media Server"],
        linux: ["plexmediaserver"],
        flatpak: [], snap: ["plexmediaserver"],
        winget: ["Plex.PlexMediaServer"], brew: ["plex-media-server"],
        primary: ("plex", "Plex Media Server", MatchQuality::Native,
            "Official Linux build. NixOS module: services.plex.enable = true. Same library, same clients.",
            ["Proprietary"]),
        alts: [("jellyfin", "Jellyfin", MatchQuality::StrongAlternative,
             "Fully open-source media server. No account required, no premium features locked.",
             ["Smaller app ecosystem", "Fewer automatic metadata agents"])]
    ),
    app!("Jellyfin", AppCategory::MediaPlayer, false,
        win: ["Jellyfin", "Jellyfin Server"], mac: ["Jellyfin"], linux: ["jellyfin"],
        flatpak: [], snap: [],
        winget: ["Jellyfin.JellyfinServer"], brew: ["jellyfin"],
        primary: ("jellyfin", "Jellyfin", MatchQuality::Native,
            "Same open-source media server. NixOS module: services.jellyfin.enable = true.", []),
        alts: []
    ),
    app!("Kodi", AppCategory::MediaPlayer, false,
        win: ["Kodi"], mac: ["Kodi"], linux: ["kodi"],
        flatpak: ["tv.kodi.Kodi"], snap: ["kodi"],
        winget: ["XBMCFoundation.Kodi"], brew: ["kodi"],
        primary: ("kodi", "Kodi", MatchQuality::Native,
            "Same media center. Addons, skins, and library data all transfer.", []),
        alts: []
    ),
];

// ═══════════════════════════════════════════════════════
// Static Data: Bundles
// ═══════════════════════════════════════════════════════

static BUNDLES: &[AppBundle] = &[
    AppBundle {
        name: "Music Production",
        description: "Low-latency audio, DAW, MIDI support",
        trigger_categories: &[AppCategory::Audio],
        trigger_threshold: 1,
        packages: &[
            "ardour",
            "audacity",
            "lmms",
            "hydrogen",
            "musescore",
            "qjackctl",
        ],
        nix_options: &[
            ("services.pipewire.jack.enable", "true"),
            ("security.rtkit.enable", "true"),
            ("boot.kernelParams", "[ \"threadirqs\" ]"),
        ],
        explanation: "PipeWire with JACK bridge for professional low-latency audio. Real-time thread priority enabled.",
    },
    AppBundle {
        name: "Creative Suite",
        description: "Image editing, vector graphics, video production",
        trigger_categories: &[
            AppCategory::Creative2D,
            AppCategory::Creative3D,
            AppCategory::Video,
            AppCategory::Photo,
        ],
        trigger_threshold: 2,
        packages: &[
            "gimp",
            "inkscape",
            "krita",
            "blender",
            "kdenlive",
            "darktable",
            "obs-studio",
        ],
        nix_options: &[],
        explanation: "Comprehensive creative tools. GIMP for photos, Inkscape for vectors, Blender for 3D, Kdenlive for video.",
    },
    AppBundle {
        name: "Development",
        description: "Editors, containers, version control, build tools",
        trigger_categories: &[
            AppCategory::IDE,
            AppCategory::Editor,
            AppCategory::VersionControl,
            AppCategory::Container,
            AppCategory::DevTools,
        ],
        trigger_threshold: 2,
        packages: &[
            "git",
            "vscode",
            "direnv",
            "nix-direnv",
            "ripgrep",
            "fd",
            "bat",
            "jq",
            "curl",
        ],
        nix_options: &[("programs.direnv.enable", "true")],
        explanation: "Modern development toolkit with direnv for per-project environments. No more version managers.",
    },
    AppBundle {
        name: "Gaming",
        description: "Steam, game launchers, performance tools",
        trigger_categories: &[AppCategory::Gaming, AppCategory::GamingTools],
        trigger_threshold: 1,
        packages: &[
            "steam",
            "lutris",
            "mangohud",
            "gamemode",
            "gamescope",
            "protonup-qt",
            "heroic",
        ],
        nix_options: &[
            ("programs.steam.enable", "true"),
            ("programs.gamemode.enable", "true"),
            ("hardware.graphics.enable32Bit", "true"),
        ],
        explanation: "Steam with Proton for Windows games. MangoHud for FPS overlay. GameMode for performance.",
    },
    AppBundle {
        name: "Privacy & Security",
        description: "VPN, password management, encryption",
        trigger_categories: &[AppCategory::Security, AppCategory::VPN],
        trigger_threshold: 2,
        packages: &[
            "bitwarden-desktop",
            "keepassxc",
            "gnupg",
            "age",
            "tor-browser",
        ],
        nix_options: &[("networking.firewall.enable", "true")],
        explanation: "Privacy-focused tools. Firewall always enabled. Encrypted password storage.",
    },
];

// ═══════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_database_builds() {
        let db = AppDatabase::new();
        assert!(
            db.entries().len() >= 20,
            "Database has {} entries",
            db.entries().len()
        );
        assert!(
            db.bundles().len() >= 4,
            "Database has {} bundles",
            db.bundles().len()
        );
    }

    #[test]
    fn test_match_by_canonical_name() {
        let db = AppDatabase::new();
        assert!(db.match_app("Firefox").is_some());
        assert!(db.match_app("Google Chrome").is_some());
        assert!(db.match_app("Steam").is_some());
    }

    #[test]
    fn test_match_by_winget_id() {
        let db = AppDatabase::new();
        assert!(db.match_app("Mozilla.Firefox").is_some());
        assert!(db.match_app("Microsoft.VisualStudioCode").is_some());
        assert!(db.match_app("Valve.Steam").is_some());
    }

    #[test]
    fn test_match_by_flatpak_id() {
        let db = AppDatabase::new();
        assert!(db.match_app("org.mozilla.firefox").is_some());
        assert!(db.match_app("com.spotify.Client").is_some());
    }

    #[test]
    fn test_match_by_brew_name() {
        let db = AppDatabase::new();
        assert!(db.match_app("firefox").is_some());
        assert!(db.match_app("visual-studio-code").is_some());
    }

    #[test]
    fn test_case_insensitive() {
        let db = AppDatabase::new();
        assert!(db.match_app("FIREFOX").is_some());
        assert!(db.match_app("google chrome").is_some());
        assert!(db.match_app("STEAM").is_some());
    }

    #[test]
    fn test_no_equivalent_has_low_confidence() {
        let db = AppDatabase::new();
        let imessage = db.match_app("iMessage").unwrap();
        assert!(imessage.primary.quality.confidence() < 0.2);
    }

    #[test]
    fn test_native_has_high_confidence() {
        let db = AppDatabase::new();
        let firefox = db.match_app("Firefox").unwrap();
        assert!(firefox.primary.quality.confidence() > 0.9);
    }

    #[test]
    fn test_every_entry_has_justification() {
        let db = AppDatabase::new();
        for entry in db.entries() {
            assert!(
                !entry.primary.justification.is_empty(),
                "App '{}' primary has no justification",
                entry.name
            );
            for alt in entry.alternatives {
                assert!(
                    !alt.justification.is_empty(),
                    "App '{}' alternative '{}' has no justification",
                    entry.name,
                    alt.display_name
                );
            }
        }
    }

    #[test]
    fn test_migration_report() {
        let db = AppDatabase::new();
        let apps = vec![
            "Firefox".into(),
            "Visual Studio Code".into(),
            "Steam".into(),
            "Adobe Photoshop".into(),
            "iMessage".into(),
            "SomeUnknownApp".into(),
        ];
        let report = db.match_list(&apps);

        assert_eq!(report.total_apps, 6);
        assert_eq!(report.matched.len(), 5); // 5 matched
        assert_eq!(report.unmatched.len(), 1); // SomeUnknownApp
        assert!(report.readiness_score > 0.5);
    }

    #[test]
    fn test_gaming_bundle_detected() {
        let db = AppDatabase::new();
        let apps = vec!["Steam".into(), "Discord".into()];
        let report = db.match_list(&apps);
        assert!(
            report.suggested_bundles.iter().any(|b| b.name == "Gaming"),
            "Gaming bundle should be suggested when Steam is present"
        );
    }

    #[test]
    fn test_music_bundle_detected() {
        let db = AppDatabase::new();
        let apps = vec!["Ableton Live".into()];
        let report = db.match_list(&apps);
        assert!(
            report
                .suggested_bundles
                .iter()
                .any(|b| b.name == "Music Production"),
            "Music bundle should be suggested when DAW is present"
        );
    }

    #[test]
    fn test_parse_winget_output() {
        let db = AppDatabase::new();
        let winget_output = "Name                    Id                          Version\n\
                             ---------------------   -------------------------   -------\n\
                             Mozilla Firefox         Mozilla.Firefox             125.0\n\
                             Visual Studio Code      Microsoft.VisualStudioCode  1.88\n\
                             Steam                   Valve.Steam                 latest";
        let apps = db.parse_app_list(winget_output);
        assert!(
            apps.len() >= 3,
            "Parsed {} apps from winget output",
            apps.len()
        );
    }

    #[test]
    fn test_parse_brew_output() {
        let db = AppDatabase::new();
        let brew_output = "---APPS---\nFirefox\nSpotify\n---BREW---\ngit\nhtop\nnvim";
        let apps = db.parse_app_list(brew_output);
        assert!(
            apps.len() >= 5,
            "Parsed {} apps from brew output",
            apps.len()
        );
    }

    #[test]
    fn test_parse_dpkg_output() {
        let db = AppDatabase::new();
        let dpkg_output = "ii  firefox                125.0   amd64   Mozilla Firefox\n\
                           ii  git                    2.44    amd64   Git VCS\n\
                           ii  vim                    9.1     amd64   Vi IMproved";
        let apps = db.parse_app_list(dpkg_output);
        assert_eq!(apps.len(), 3);
        assert!(apps.contains(&"firefox".to_string()));
    }

    #[test]
    fn test_opinionated_defaults_with_alternatives() {
        let db = AppDatabase::new();
        let photoshop = db.match_app("Adobe Photoshop").unwrap();

        // Has a primary recommendation
        assert!(!photoshop.primary.nix_pkg.is_empty());
        assert!(!photoshop.primary.justification.is_empty());

        // Has alternatives
        assert!(!photoshop.alternatives.is_empty());

        // Each alternative has justification
        for alt in photoshop.alternatives {
            assert!(!alt.justification.is_empty());
        }
    }

    #[test]
    fn test_windows_user_office_recommendation() {
        let db = AppDatabase::new();
        let office = db.match_app("Microsoft Office").unwrap();

        // Primary should be OnlyOffice (best compatibility)
        assert_eq!(office.primary.nix_pkg, "onlyoffice-bin");
        assert!(office.primary.justification.contains("compatibility"));

        // LibreOffice should be an alternative
        assert!(
            office
                .alternatives
                .iter()
                .any(|a| a.nix_pkg == "libreoffice")
        );
    }

    #[test]
    fn all_entries_have_verified_channel() {
        let db = AppDatabase::new();
        for entry in db.entries() {
            assert!(
                !entry.verified_channel.is_empty(),
                "Entry '{}' has empty verified_channel",
                entry.name
            );
        }
    }

    #[test]
    fn is_potentially_stale_detects_mismatch() {
        let db = AppDatabase::new();
        let firefox = db.match_app("Firefox").unwrap();

        // Same channel = not stale
        assert!(!super::is_potentially_stale(firefox, "25.05"));

        // Different channel = stale
        assert!(super::is_potentially_stale(firefox, "24.11"));
        assert!(super::is_potentially_stale(firefox, "25.11"));
    }

    #[test]
    fn parse_channel_version_extracts_major_minor() {
        assert_eq!(
            super::parse_channel_version("25.05.20260401.abc1234"),
            "25.05"
        );
        assert_eq!(super::parse_channel_version("24.11"), "24.11");
        assert_eq!(super::parse_channel_version("25.05"), "25.05");
        assert_eq!(super::parse_channel_version("unknown"), "unknown");
        assert_eq!(super::parse_channel_version("  25.05.123  "), "25.05");
    }
}
