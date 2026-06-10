// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! NixOS Knowledge Corpus with HDC Similarity Search
//!
//! Pre-seeded knowledge articles covering common NixOS issues, encoded as HDC
//! vectors for fast approximate nearest-neighbor search. When a user encounters
//! an error, the knowledge base finds the most semantically similar articles
//! and returns actionable solutions.
//!
//! Uses the existing `NixCodebook` for encoding and its built-in LSH index
//! for sub-linear search.

use crate::encoding::codebook::NixCodebook;
use symthaea_core::hdc::ContinuousHV;

/// A curated knowledge article about a NixOS issue or pattern.
#[derive(Debug, Clone)]
pub struct KnowledgeArticle {
    pub id: &'static str,
    pub title: &'static str,
    pub category: KnowledgeCategory,
    pub symptoms: &'static [&'static str],
    pub solution: &'static str,
    pub commands: &'static [&'static str],
}

/// Categories for knowledge articles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum KnowledgeCategory {
    BuildError,
    ServiceIssue,
    StoreManagement,
    FlakePattern,
    HardwareDriver,
    EvaluationError,
}

impl std::fmt::Display for KnowledgeCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BuildError => write!(f, "Build Error"),
            Self::ServiceIssue => write!(f, "Service Issue"),
            Self::StoreManagement => write!(f, "Store Management"),
            Self::FlakePattern => write!(f, "Flake Pattern"),
            Self::HardwareDriver => write!(f, "Hardware/Driver"),
            Self::EvaluationError => write!(f, "Evaluation Error"),
        }
    }
}

/// A search result matching a knowledge article.
#[derive(Debug, Clone)]
pub struct KnowledgeMatch<'a> {
    pub article: &'a KnowledgeArticle,
    pub similarity: f32,
}

/// A dynamically learned knowledge article (from incident resolution).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DynamicKnowledgeArticle {
    pub id: String,
    pub title: String,
    pub category: KnowledgeCategory,
    pub symptoms: Vec<String>,
    pub solution: String,
    pub commands: Vec<String>,
    pub learned_at: i64,
    pub hit_count: u32,
}

/// A unified search result that wraps either a static or dynamic article.
#[derive(Debug, Clone)]
pub enum AnyKnowledgeMatch<'a> {
    Static(KnowledgeMatch<'a>),
    Dynamic {
        article: DynamicKnowledgeArticle,
        similarity: f32,
    },
}

impl<'a> AnyKnowledgeMatch<'a> {
    pub fn similarity(&self) -> f32 {
        match self {
            Self::Static(m) => m.similarity,
            Self::Dynamic { similarity, .. } => *similarity,
        }
    }

    pub fn title(&self) -> &str {
        match self {
            Self::Static(m) => m.article.title,
            Self::Dynamic { article, .. } => &article.title,
        }
    }

    pub fn solution(&self) -> &str {
        match self {
            Self::Static(m) => m.article.solution,
            Self::Dynamic { article, .. } => &article.solution,
        }
    }

    pub fn id(&self) -> &str {
        match self {
            Self::Static(m) => m.article.id,
            Self::Dynamic { article, .. } => &article.id,
        }
    }

    pub fn category(&self) -> KnowledgeCategory {
        match self {
            Self::Static(m) => m.article.category,
            Self::Dynamic { article, .. } => article.category,
        }
    }

    pub fn commands(&self) -> Vec<&str> {
        match self {
            Self::Static(m) => m.article.commands.to_vec(),
            Self::Dynamic { article, .. } => article.commands.iter().map(|s| s.as_str()).collect(),
        }
    }

    pub fn is_dynamic(&self) -> bool {
        matches!(self, Self::Dynamic { .. })
    }
}

/// Pre-encoded NixOS knowledge base with HDC similarity search.
pub struct KnowledgeBase {
    articles: Vec<KnowledgeArticle>,
    article_hvs: Vec<ContinuousHV>,
    dynamic_articles: Vec<DynamicKnowledgeArticle>,
    dynamic_hvs: Vec<ContinuousHV>,
}

impl KnowledgeBase {
    /// Create a new knowledge base, encoding all articles using the given codebook.
    pub fn new(codebook: &mut NixCodebook) -> Self {
        let articles = all_articles();
        let article_hvs: Vec<ContinuousHV> = articles
            .iter()
            .map(|a| Self::encode_article(codebook, a))
            .collect();

        Self {
            articles,
            article_hvs,
            dynamic_articles: Vec::new(),
            dynamic_hvs: Vec::new(),
        }
    }

    /// Search the knowledge base by natural language query.
    pub fn search(
        &self,
        query: &str,
        codebook: &mut NixCodebook,
        k: usize,
    ) -> Vec<KnowledgeMatch<'_>> {
        let query_hv = Self::encode_text(codebook, query);
        self.search_by_hv(&query_hv, k)
    }

    /// Search by error message — encodes the error and finds matching articles.
    pub fn search_by_error(
        &self,
        error: &str,
        codebook: &mut NixCodebook,
        k: usize,
    ) -> Vec<KnowledgeMatch<'_>> {
        // Combine error text with common error-related keywords for better matching
        let augmented = format!("error {}", error);
        let query_hv = Self::encode_text(codebook, &augmented);
        self.search_by_hv(&query_hv, k)
    }

    /// Get all articles in a specific category.
    pub fn articles_in_category(&self, category: KnowledgeCategory) -> Vec<&KnowledgeArticle> {
        self.articles
            .iter()
            .filter(|a| a.category == category)
            .collect()
    }

    /// Total number of articles.
    pub fn len(&self) -> usize {
        self.articles.len()
    }

    /// Whether the knowledge base is empty.
    pub fn is_empty(&self) -> bool {
        self.articles.is_empty()
    }

    /// Add a dynamically learned article from incident resolution.
    pub fn add_learned_article(
        &mut self,
        article: DynamicKnowledgeArticle,
        codebook: &mut NixCodebook,
    ) {
        let hv = Self::encode_dynamic_article(codebook, &article);
        self.dynamic_articles.push(article);
        self.dynamic_hvs.push(hv);
    }

    /// Search both static and dynamic articles, returning unified results.
    pub fn search_all(
        &mut self,
        query: &str,
        codebook: &mut NixCodebook,
        k: usize,
    ) -> Vec<AnyKnowledgeMatch<'_>> {
        let query_hv = Self::encode_text(codebook, query);

        let mut scored: Vec<AnyKnowledgeMatch<'_>> = Vec::new();

        // Score static articles
        for (article, hv) in self.articles.iter().zip(self.article_hvs.iter()) {
            scored.push(AnyKnowledgeMatch::Static(KnowledgeMatch {
                article,
                similarity: query_hv.similarity(hv).max(0.0),
            }));
        }

        // Score dynamic articles
        for (article, hv) in self.dynamic_articles.iter().zip(self.dynamic_hvs.iter()) {
            scored.push(AnyKnowledgeMatch::Dynamic {
                article: article.clone(),
                similarity: query_hv.similarity(hv).max(0.0),
            });
        }

        // Sort by similarity descending, with stable tie-breaking on article ID
        scored.sort_by(|a, b| {
            b.similarity()
                .partial_cmp(&a.similarity())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id().cmp(b.id()))
        });
        scored.truncate(k);

        // Increment hit counts for dynamic matches
        for m in &scored {
            if let AnyKnowledgeMatch::Dynamic { article, .. } = m {
                if let Some(da) = self
                    .dynamic_articles
                    .iter_mut()
                    .find(|a| a.id == article.id)
                {
                    da.hit_count += 1;
                }
            }
        }

        scored
    }

    /// Serialize dynamic articles to JSON for persistence.
    pub fn save_dynamic(&self) -> String {
        serde_json::to_string_pretty(&self.dynamic_articles).unwrap_or_else(|_| "[]".to_string())
    }

    /// Load dynamic articles from JSON, re-encoding each with the codebook.
    pub fn load_dynamic(&mut self, json: &str, codebook: &mut NixCodebook) {
        if let Ok(articles) = serde_json::from_str::<Vec<DynamicKnowledgeArticle>>(json) {
            for article in articles {
                let hv = Self::encode_dynamic_article(codebook, &article);
                self.dynamic_articles.push(article);
                self.dynamic_hvs.push(hv);
            }
        }
    }

    pub fn static_len(&self) -> usize {
        self.articles.len()
    }

    /// Number of dynamically learned articles.
    pub fn dynamic_len(&self) -> usize {
        self.dynamic_articles.len()
    }

    /// Get the hit count for a dynamic article by ID.
    pub fn dynamic_hit_count(&self, id: &str) -> Option<u32> {
        self.dynamic_articles
            .iter()
            .find(|a| a.id == id)
            .map(|a| a.hit_count)
    }

    pub fn search_by_hv(&self, query_hv: &ContinuousHV, k: usize) -> Vec<KnowledgeMatch<'_>> {
        let mut scored: Vec<KnowledgeMatch<'_>> = self
            .articles
            .iter()
            .zip(self.article_hvs.iter())
            .map(|(article, hv)| KnowledgeMatch {
                article,
                similarity: query_hv.similarity(hv).max(0.0),
            })
            .collect();

        // Sort by similarity descending, with stable tie-breaking on article ID
        scored.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.article.id.cmp(b.article.id))
        });
        scored.truncate(k);
        scored
    }

    /// Encode an article by bundling its title, symptoms, and solution text.
    fn encode_article(codebook: &mut NixCodebook, article: &KnowledgeArticle) -> ContinuousHV {
        let mut components = Vec::new();
        let mut weights = Vec::new();

        // Encode title (highest weight)
        let title_hv = Self::encode_text(codebook, article.title);
        components.push(title_hv);
        weights.push(1.0);

        // Encode each symptom
        for symptom in article.symptoms {
            let symptom_hv = Self::encode_text(codebook, symptom);
            components.push(symptom_hv);
            weights.push(0.8);
        }

        // Encode solution text
        let solution_hv = Self::encode_text(codebook, article.solution);
        components.push(solution_hv);
        weights.push(0.5);

        // Encode commands
        for cmd in article.commands {
            let cmd_hv = Self::encode_text(codebook, cmd);
            components.push(cmd_hv);
            weights.push(0.3);
        }

        let refs: Vec<&ContinuousHV> = components.iter().collect();
        ContinuousHV::weighted_bundle(&refs, &weights)
    }

    /// Encode a dynamic article using the same strategy as static articles.
    fn encode_dynamic_article(
        codebook: &mut NixCodebook,
        article: &DynamicKnowledgeArticle,
    ) -> ContinuousHV {
        let mut components = Vec::new();
        let mut weights = Vec::new();

        let title_hv = Self::encode_text(codebook, &article.title);
        components.push(title_hv);
        weights.push(1.0);

        for symptom in &article.symptoms {
            let symptom_hv = Self::encode_text(codebook, symptom);
            components.push(symptom_hv);
            weights.push(0.8);
        }

        let solution_hv = Self::encode_text(codebook, &article.solution);
        components.push(solution_hv);
        weights.push(0.5);

        for cmd in &article.commands {
            let cmd_hv = Self::encode_text(codebook, cmd);
            components.push(cmd_hv);
            weights.push(0.3);
        }

        let refs: Vec<&ContinuousHV> = components.iter().collect();
        ContinuousHV::weighted_bundle(&refs, &weights)
    }

    /// Encode a free-text string by bundling word-level basis vectors.
    pub fn encode_text(codebook: &mut NixCodebook, text: &str) -> ContinuousHV {
        let words: Vec<ContinuousHV> = text
            .split_whitespace()
            .map(|w| {
                let lower = w.to_lowercase();
                let clean = lower.trim_matches(|c: char| {
                    !c.is_alphanumeric() && c != '-' && c != '_' && c != '.'
                });
                if clean.is_empty() {
                    ContinuousHV::zero(codebook.dim())
                } else {
                    codebook.get_or_create(clean).clone()
                }
            })
            .collect();

        if words.is_empty() {
            return ContinuousHV::zero(codebook.dim());
        }

        let refs: Vec<&ContinuousHV> = words.iter().collect();
        ContinuousHV::bundle(&refs)
    }
}

// ============================================================================
// Pre-seeded Knowledge Articles
// ============================================================================

fn all_articles() -> Vec<KnowledgeArticle> {
    let mut articles = Vec::with_capacity(50);
    articles.extend(build_error_articles());
    articles.extend(service_issue_articles());
    articles.extend(store_management_articles());
    articles.extend(flake_pattern_articles());
    articles.extend(hardware_driver_articles());
    articles.extend(evaluation_error_articles());
    articles
}

fn build_error_articles() -> Vec<KnowledgeArticle> {
    vec![
        KnowledgeArticle {
            id: "build-hash-mismatch",
            title: "Hash mismatch in fixed-output derivation",
            category: KnowledgeCategory::BuildError,
            symptoms: &[
                "hash mismatch in fixed-output derivation",
                "got: sha256-",
                "wanted: sha256-",
                "output path has wrong hash",
            ],
            solution: "The upstream source changed or the hash in the derivation is incorrect. \
                       Update the hash in your derivation or use `lib.fakeSha256` temporarily \
                       to get the correct hash from the build error.",
            commands: &[
                "nix-prefetch-url <url>",
                "nix-prefetch-url --unpack <url>",
                "nix build --rebuild",
            ],
        },
        KnowledgeArticle {
            id: "build-missing-dep",
            title: "Missing build dependency or package not found",
            category: KnowledgeCategory::BuildError,
            symptoms: &[
                "error: attribute not found",
                "undefined variable",
                "package not found in nixpkgs",
                "is not available on the requested hostPlatform",
            ],
            solution: "The package may have been renamed, moved to a different attribute set, \
                       or marked as broken/insecure. Search nixpkgs for alternatives.",
            commands: &[
                "nix search nixpkgs <name>",
                "nix eval nixpkgs#<name>.meta.available",
            ],
        },
        KnowledgeArticle {
            id: "build-sandbox-violation",
            title: "Sandbox violation during build",
            category: KnowledgeCategory::BuildError,
            symptoms: &[
                "error: while setting up the build environment: mounting",
                "sandbox violation",
                "network access not allowed",
                "build failed due to sandbox",
            ],
            solution: "NixOS builds run in a sandboxed environment. If a build needs network \
                       access (e.g., to download dependencies), it should use a fixed-output \
                       derivation. For development, you can temporarily set `sandbox = relaxed` \
                       in nix.conf.",
            commands: &[
                "nix build --option sandbox false", // only for debugging!
            ],
        },
        KnowledgeArticle {
            id: "build-compiler-error",
            title: "Compiler failure during package build",
            category: KnowledgeCategory::BuildError,
            symptoms: &[
                "error: builder for",
                "failed with exit code",
                "build of derivation failed",
                "compilation error",
            ],
            solution: "Check the build log for the specific compiler error. Often caused by \
                       version mismatches between the package and its dependencies. Try building \
                       with --show-trace for more context.",
            commands: &[
                "nix log /nix/store/<hash>.drv",
                "nix build --show-trace",
                "nixos-rebuild switch --show-trace",
            ],
        },
        KnowledgeArticle {
            id: "build-out-of-disk",
            title: "Build fails due to insufficient disk space",
            category: KnowledgeCategory::BuildError,
            symptoms: &[
                "No space left on device",
                "ENOSPC",
                "disk quota exceeded",
                "cannot write to /nix/store",
            ],
            solution: "Free disk space by garbage-collecting old store paths and generations. \
                       Large builds (e.g., linux kernel, chromium) need significant temporary space.",
            commands: &[
                "nix-collect-garbage -d",
                "nix store optimise",
                "df -h /nix/store",
            ],
        },
        KnowledgeArticle {
            id: "build-nix-store-repair",
            title: "Corrupted nix store paths",
            category: KnowledgeCategory::BuildError,
            symptoms: &[
                "path is not valid",
                "hash mismatch on existing path",
                "corrupted store path",
                "nix store verify failed",
            ],
            solution: "The nix store may have corrupted paths. Use `nix store repair` to fix \
                       individual paths or verify the entire store.",
            commands: &[
                "nix store verify --all",
                "nix store repair --all",
                "nix-store --verify --check-contents",
            ],
        },
        KnowledgeArticle {
            id: "build-download-failure",
            title: "Failed to download source or substitutes",
            category: KnowledgeCategory::BuildError,
            symptoms: &[
                "unable to download",
                "HTTP error 404",
                "connection timed out",
                "SSL certificate problem",
                "cannot substitute",
            ],
            solution: "Check network connectivity and DNS. If the cache is unavailable, \
                       try building from source with --option substitute false.",
            commands: &[
                "nix build --option substitute false",
                "nix store ping --store https://cache.nixos.org",
            ],
        },
        KnowledgeArticle {
            id: "build-permission-denied",
            title: "Permission denied during nix operations",
            category: KnowledgeCategory::BuildError,
            symptoms: &[
                "error: cannot open connection to remote store",
                "error: getting status of /nix/var/nix/daemon-socket",
                "Permission denied",
                "nix-daemon not running",
            ],
            solution: "Ensure the nix-daemon is running and you are in the correct group. \
                       On NixOS multi-user installs, the nix-daemon handles all store operations.",
            commands: &[
                "systemctl status nix-daemon",
                "systemctl restart nix-daemon",
                "groups $(whoami)",
            ],
        },
    ]
}

fn service_issue_articles() -> Vec<KnowledgeArticle> {
    vec![
        KnowledgeArticle {
            id: "service-failed-start",
            title: "Service failed to start after rebuild",
            category: KnowledgeCategory::ServiceIssue,
            symptoms: &[
                "Job failed",
                "service entered failed state",
                "Start request repeated too quickly",
                "main process exited, code=exited, status=1",
            ],
            solution: "Check the service journal for the actual error. Common causes: \
                       missing config file, port already in use, permission denied, \
                       or incompatible configuration after an upgrade.",
            commands: &[
                "journalctl -u <service> --no-pager -n 50",
                "systemctl status <service>",
                "systemctl restart <service>",
            ],
        },
        KnowledgeArticle {
            id: "service-port-conflict",
            title: "Port already in use / socket conflict",
            category: KnowledgeCategory::ServiceIssue,
            symptoms: &[
                "Address already in use",
                "port is already allocated",
                "bind: address in use",
                "EADDRINUSE",
            ],
            solution: "Another process is using the same port. Find and stop it, or change \
                       the service's port in your NixOS configuration.",
            commands: &[
                "ss -tlnp | grep <port>",
                "fuser <port>/tcp",
                "systemctl stop <conflicting-service>",
            ],
        },
        KnowledgeArticle {
            id: "service-permission-denied",
            title: "Service permission denied on files or directories",
            category: KnowledgeCategory::ServiceIssue,
            symptoms: &["Permission denied", "EACCES", "cannot open", "mkdir failed"],
            solution: "The service user doesn't have access to required paths. Use \
                       systemd's StateDirectory, CacheDirectory, or tmpfiles rules \
                       to ensure correct ownership.",
            commands: &[
                "ls -la /path/to/problem",
                "systemctl show <service> -p StateDirectory",
                "journalctl -u <service> --no-pager -n 20",
            ],
        },
        KnowledgeArticle {
            id: "service-dependency-order",
            title: "Service starts before dependency is ready",
            category: KnowledgeCategory::ServiceIssue,
            symptoms: &[
                "Connection refused",
                "No such file or directory",
                "database connection failed",
                "Cannot connect to",
            ],
            solution: "Add After= and Wants= dependencies in the service unit. For database \
                       services, use `after = [ \"postgresql.service\" ]` in the systemd config.",
            commands: &[
                "systemd-analyze critical-chain <service>",
                "systemctl list-dependencies <service>",
            ],
        },
        KnowledgeArticle {
            id: "service-oom-killed",
            title: "Service killed by OOM killer",
            category: KnowledgeCategory::ServiceIssue,
            symptoms: &[
                "oom-kill",
                "Out of memory",
                "Killed process",
                "memory cgroup out of memory",
            ],
            solution: "The service exceeded its memory limit or the system ran out of RAM. \
                       Increase MemoryMax in the systemd unit, add swap, or optimize the service's \
                       memory usage.",
            commands: &[
                "journalctl -k --no-pager | grep -i oom",
                "systemctl show <service> -p MemoryMax",
                "systemd-cgtop",
            ],
        },
        KnowledgeArticle {
            id: "service-timer-drift",
            title: "Systemd timer not firing or drifting",
            category: KnowledgeCategory::ServiceIssue,
            symptoms: &[
                "timer not triggering",
                "last trigger long ago",
                "missed timer event",
                "AccuracySec",
            ],
            solution: "Check the timer's schedule and AccuracySec setting. Systemd batches \
                       timers by default for power savings. Set AccuracySec=1s for precise timing.",
            commands: &[
                "systemctl list-timers",
                "systemctl status <timer>",
                "journalctl -u <timer> --no-pager -n 20",
            ],
        },
    ]
}

fn store_management_articles() -> Vec<KnowledgeArticle> {
    vec![
        KnowledgeArticle {
            id: "store-gc-strategy",
            title: "Garbage collection strategies for nix store",
            category: KnowledgeCategory::StoreManagement,
            symptoms: &[
                "nix store too large",
                "disk space running low",
                "too many store paths",
                "how to clean nix store",
            ],
            solution: "Use nix-collect-garbage with age-based deletion. Keep at least the \
                       last 3 generations for rollback safety. Run `nix store optimise` \
                       after GC to deduplicate identical files via hardlinks.",
            commands: &[
                "nix-collect-garbage -d --delete-older-than 30d",
                "nix store optimise",
                "nix-env --delete-generations +3 -p /nix/var/nix/profiles/system",
            ],
        },
        KnowledgeArticle {
            id: "store-repair",
            title: "Verifying and repairing the nix store",
            category: KnowledgeCategory::StoreManagement,
            symptoms: &[
                "store corruption",
                "invalid path",
                "hash mismatch in store",
                "nix store integrity",
            ],
            solution: "Run nix store verify to check integrity. For corrupted paths, \
                       nix store repair will re-download or rebuild them from substitutes.",
            commands: &[
                "nix store verify --all",
                "nix store repair --all",
                "nix-store --verify --check-contents --repair",
            ],
        },
        KnowledgeArticle {
            id: "store-optimise",
            title: "Deduplicating nix store with hardlinks",
            category: KnowledgeCategory::StoreManagement,
            symptoms: &[
                "duplicate files in store",
                "store optimise",
                "save disk space nix",
                "hardlink dedup",
            ],
            solution: "nix store optimise scans all store paths and replaces identical \
                       files with hardlinks. Can save 25-40% on large stores. Safe to run \
                       at any time. Enable auto-optimise in nix.conf for continuous savings.",
            commands: &[
                "nix store optimise",
                "nix.settings.auto-optimise-store = true",
            ],
        },
        KnowledgeArticle {
            id: "store-gc-roots",
            title: "Understanding and managing GC roots",
            category: KnowledgeCategory::StoreManagement,
            symptoms: &[
                "gc root",
                "why is this path kept",
                "cannot garbage collect",
                "nix-store --gc --print-roots",
            ],
            solution: "GC roots are symlinks that prevent store paths from being collected. \
                       They live in /nix/var/nix/gcroots/ and profile links. Remove stale \
                       roots to allow GC of unused paths.",
            commands: &[
                "nix-store --gc --print-roots",
                "nix path-info -rS /nix/store/<path>",
                "nix why-depends /run/current-system /nix/store/<path>",
            ],
        },
    ]
}

fn flake_pattern_articles() -> Vec<KnowledgeArticle> {
    vec![
        KnowledgeArticle {
            id: "flake-lock-conflict",
            title: "Flake lock file conflicts or stale locks",
            category: KnowledgeCategory::FlakePattern,
            symptoms: &[
                "flake.lock conflict",
                "lock file out of date",
                "input not in lock file",
                "cannot resolve flake reference",
            ],
            solution: "Update the flake lock file to resolve conflicts. For specific inputs, \
                       use `nix flake lock --update-input`. For all inputs, use `nix flake update`.",
            commands: &[
                "nix flake update",
                "nix flake lock --update-input nixpkgs",
                "git add flake.lock && git commit -m 'Update flake.lock'",
            ],
        },
        KnowledgeArticle {
            id: "flake-follows",
            title: "Using follows to deduplicate flake inputs",
            category: KnowledgeCategory::FlakePattern,
            symptoms: &[
                "duplicate nixpkgs",
                "multiple nixpkgs versions",
                "follows",
                "flake input deduplication",
            ],
            solution: "Use `follows` to make all inputs share the same nixpkgs instance. \
                       This reduces eval time and store size significantly.",
            commands: &["nix flake metadata --json | jq '.locks.nodes | keys'"],
        },
        KnowledgeArticle {
            id: "flake-override",
            title: "Overriding packages in a flake",
            category: KnowledgeCategory::FlakePattern,
            symptoms: &[
                "override package version",
                "override source",
                "custom package in flake",
                "overlay in flake",
            ],
            solution: "Use overlays in your flake to override packages. Define the overlay \
                       in your flake outputs and apply it via nixpkgs.overlays in your NixOS config.",
            commands: &[
                "nix build .#packages.x86_64-linux.<pkg>",
                "nix eval .#nixosConfigurations.<host>.config.nixpkgs.overlays",
            ],
        },
        KnowledgeArticle {
            id: "flake-eval-slow",
            title: "Slow flake evaluation / nix eval performance",
            category: KnowledgeCategory::FlakePattern,
            symptoms: &[
                "slow evaluation",
                "nix eval takes long",
                "rebuilding takes forever",
                "IFD causing slowdown",
            ],
            solution: "Slow eval is usually caused by import-from-derivation (IFD), \
                       large nixpkgs imports, or unintended deep evaluation. Use \
                       `--show-trace` and `nix eval --debugger` to identify bottlenecks.",
            commands: &[
                "nix eval --show-trace",
                "time nix eval .#nixosConfigurations.<host>.config.system.build.toplevel",
            ],
        },
        KnowledgeArticle {
            id: "flake-pure-eval",
            title: "Flake pure evaluation errors",
            category: KnowledgeCategory::FlakePattern,
            symptoms: &[
                "pure eval",
                "access to path is forbidden in pure eval mode",
                "cannot access",
                "builtins.getEnv",
            ],
            solution: "Flakes enforce pure evaluation by default. You cannot access files \
                       outside the flake directory or use impure builtins. Add files to git \
                       (even unstaged) or use `--impure` for development.",
            commands: &["git add <file>", "nix build --impure"],
        },
    ]
}

fn hardware_driver_articles() -> Vec<KnowledgeArticle> {
    vec![
        KnowledgeArticle {
            id: "hw-nvidia-driver",
            title: "NVIDIA GPU driver setup on NixOS",
            category: KnowledgeCategory::HardwareDriver,
            symptoms: &[
                "NVIDIA",
                "nvidia driver not loaded",
                "GPU not detected",
                "nouveau crash",
                "graphics acceleration",
            ],
            solution: "NixOS has dedicated NVIDIA driver support. Enable hardware.nvidia \
                       and blacklist nouveau. Use the stable or beta driver branch.",
            commands: &[
                "lspci | grep -i nvidia",
                "nix eval nixpkgs#linuxPackages.nvidia_x11.version",
            ],
        },
        KnowledgeArticle {
            id: "hw-amd-gpu",
            title: "AMD GPU setup on NixOS",
            category: KnowledgeCategory::HardwareDriver,
            symptoms: &[
                "AMD GPU",
                "amdgpu driver",
                "Radeon",
                "Vulkan not working",
                "mesa",
            ],
            solution: "AMD GPUs use the open-source amdgpu driver included in the kernel. \
                       Enable hardware.opengl and install mesa drivers. For Vulkan, add \
                       the appropriate Vulkan loader packages.",
            commands: &["glxinfo | grep 'OpenGL renderer'", "vulkaninfo"],
        },
        KnowledgeArticle {
            id: "hw-wifi-firmware",
            title: "WiFi not working / firmware missing",
            category: KnowledgeCategory::HardwareDriver,
            symptoms: &[
                "WiFi not working",
                "wireless interface not found",
                "firmware missing",
                "iwlwifi",
                "no wireless networks",
            ],
            solution: "NixOS needs explicit firmware packages for most WiFi chipsets. \
                       Enable hardware.enableRedistributableFirmware or add specific \
                       firmware packages. Intel WiFi needs iwlwifi firmware.",
            commands: &[
                "lspci | grep -i wireless",
                "dmesg | grep -i firmware",
                "ip link show",
            ],
        },
        KnowledgeArticle {
            id: "hw-bluetooth",
            title: "Bluetooth not working on NixOS",
            category: KnowledgeCategory::HardwareDriver,
            symptoms: &[
                "bluetooth not working",
                "cannot find bluetooth adapter",
                "bluez",
                "bluetooth service failed",
            ],
            solution: "Enable hardware.bluetooth and the blueman service. Some adapters \
                       need additional firmware. Check if the kernel module is loaded.",
            commands: &[
                "bluetoothctl show",
                "systemctl status bluetooth",
                "dmesg | grep -i bluetooth",
            ],
        },
        KnowledgeArticle {
            id: "hw-kernel-module",
            title: "Loading custom kernel modules on NixOS",
            category: KnowledgeCategory::HardwareDriver,
            symptoms: &[
                "kernel module not loaded",
                "modprobe failed",
                "module not found",
                "extraModulePackages",
            ],
            solution: "Add kernel modules via boot.kernelModules (auto-load at boot) \
                       or boot.extraModulePackages (compile out-of-tree modules). \
                       Use boot.kernelPackages to select the kernel version.",
            commands: &[
                "lsmod | grep <module>",
                "modinfo <module>",
                "dmesg | tail -20",
            ],
        },
        KnowledgeArticle {
            id: "hw-audio",
            title: "Audio not working / PipeWire/PulseAudio setup",
            category: KnowledgeCategory::HardwareDriver,
            symptoms: &[
                "no audio",
                "sound not working",
                "PipeWire",
                "PulseAudio",
                "ALSA",
            ],
            solution: "NixOS uses PipeWire by default since 23.11. Enable \
                       services.pipewire with PulseAudio compatibility. Check that ALSA \
                       firmware is available.",
            commands: &["wpctl status", "pactl list sinks short", "aplay -l"],
        },
    ]
}

fn evaluation_error_articles() -> Vec<KnowledgeArticle> {
    vec![
        KnowledgeArticle {
            id: "eval-infinite-recursion",
            title: "Infinite recursion during Nix evaluation",
            category: KnowledgeCategory::EvaluationError,
            symptoms: &[
                "infinite recursion encountered",
                "stack overflow during evaluation",
                "maximum recursion depth exceeded",
            ],
            solution: "Usually caused by circular imports, self-referencing overlays, \
                       or mkMerge conflicts. Use --show-trace to find the cycle. \
                       Common fix: use `lib.mkForce` or restructure module imports.",
            commands: &["nixos-rebuild switch --show-trace 2>&1 | head -100"],
        },
        KnowledgeArticle {
            id: "eval-type-mismatch",
            title: "Type mismatch in NixOS configuration",
            category: KnowledgeCategory::EvaluationError,
            symptoms: &[
                "value is a string while a set was expected",
                "value is a boolean while a string was expected",
                "type error",
                "cannot coerce",
            ],
            solution: "The option value doesn't match the expected type. Check the NixOS \
                       option documentation for the correct type. Common mistakes: \
                       using a string where a list is expected, or a bool where an enum is needed.",
            commands: &[
                "nixos-option <option.path>",
                "nix eval nixpkgs#nixosModules --show-trace",
            ],
        },
        KnowledgeArticle {
            id: "eval-missing-attribute",
            title: "Missing attribute in set / undefined variable",
            category: KnowledgeCategory::EvaluationError,
            symptoms: &[
                "attribute missing",
                "undefined variable",
                "error: access to undefined attribute",
                "attribute 'X' missing",
            ],
            solution: "The referenced attribute or variable doesn't exist. Check for typos \
                       in option names, missing module imports, or packages that were renamed. \
                       Use `builtins.hasAttr` to check safely.",
            commands: &[
                "nix eval nixpkgs#<attr> --show-trace",
                "nix search nixpkgs <name>",
            ],
        },
        KnowledgeArticle {
            id: "eval-assertion-failure",
            title: "Assertion failure in NixOS configuration",
            category: KnowledgeCategory::EvaluationError,
            symptoms: &[
                "Failed assertions",
                "assertion failed",
                "The option .* is not defined",
            ],
            solution: "NixOS modules use assertions to enforce valid configurations. \
                       The assertion message usually explains what's wrong. Common causes: \
                       enabling a service without its required companion options.",
            commands: &["nixos-rebuild switch --show-trace 2>&1 | grep -A5 'Failed assertions'"],
        },
        KnowledgeArticle {
            id: "eval-option-conflict",
            title: "Conflicting option definitions across modules",
            category: KnowledgeCategory::EvaluationError,
            symptoms: &[
                "The option .* has conflicting definition values",
                "conflict between",
                "option defined in multiple modules",
                "mkForce",
            ],
            solution: "Two modules set the same option to different values. Use \
                       `lib.mkForce` to override, `lib.mkDefault` for fallbacks, or \
                       `lib.mkMerge` for list/attrset options that should combine.",
            commands: &["nixos-rebuild switch --show-trace 2>&1 | grep 'conflicting'"],
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knowledge_base_creation() {
        let mut codebook = NixCodebook::new();
        let kb = KnowledgeBase::new(&mut codebook);
        assert!(!kb.is_empty());
        assert!(kb.len() >= 30); // at least 30 articles
    }

    #[test]
    fn test_search_returns_results() {
        let mut codebook = NixCodebook::new();
        let kb = KnowledgeBase::new(&mut codebook);
        let results = kb.search("hash mismatch build error", &mut codebook, 5);
        assert!(!results.is_empty());
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_search_results_sorted_by_similarity() {
        let mut codebook = NixCodebook::new();
        let kb = KnowledgeBase::new(&mut codebook);
        let results = kb.search("disk space full nix store", &mut codebook, 10);

        for window in results.windows(2) {
            assert!(
                window[0].similarity >= window[1].similarity,
                "Results should be sorted by similarity descending"
            );
        }
    }

    #[test]
    fn test_search_by_error() {
        let mut codebook = NixCodebook::new();
        let kb = KnowledgeBase::new(&mut codebook);
        let results = kb.search_by_error(
            "error: hash mismatch in fixed-output derivation '/nix/store/abc.drv'",
            &mut codebook,
            3,
        );
        assert!(!results.is_empty());
    }

    #[test]
    fn test_category_filter() {
        let mut codebook = NixCodebook::new();
        let kb = KnowledgeBase::new(&mut codebook);

        let build_errors = kb.articles_in_category(KnowledgeCategory::BuildError);
        assert!(!build_errors.is_empty());
        assert!(
            build_errors
                .iter()
                .all(|a| a.category == KnowledgeCategory::BuildError)
        );

        let service_issues = kb.articles_in_category(KnowledgeCategory::ServiceIssue);
        assert!(!service_issues.is_empty());
    }

    #[test]
    fn test_all_articles_have_required_fields() {
        let articles = all_articles();
        for article in &articles {
            assert!(!article.id.is_empty(), "Article must have an ID");
            assert!(!article.title.is_empty(), "Article must have a title");
            assert!(
                !article.symptoms.is_empty(),
                "Article {} must have symptoms",
                article.id
            );
            assert!(
                !article.solution.is_empty(),
                "Article {} must have a solution",
                article.id
            );
        }
    }

    #[test]
    fn test_article_ids_unique() {
        let articles = all_articles();
        let mut ids: Vec<&str> = articles.iter().map(|a| a.id).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), articles.len(), "All article IDs must be unique");
    }

    #[test]
    fn test_nvidia_query_finds_nvidia_article() {
        let mut codebook = NixCodebook::new();
        let kb = KnowledgeBase::new(&mut codebook);
        let results = kb.search("NVIDIA GPU driver", &mut codebook, 5);
        assert!(!results.is_empty());
        // The top result should be the NVIDIA article
        let top = &results[0];
        assert!(
            top.article.id.contains("nvidia") || top.article.title.contains("NVIDIA"),
            "Top result for 'NVIDIA GPU driver' should be the NVIDIA article, got: {}",
            top.article.title
        );
    }

    #[test]
    fn test_gc_query_finds_store_articles() {
        let mut codebook = NixCodebook::new();
        let kb = KnowledgeBase::new(&mut codebook);
        let results = kb.search("garbage collection nix store cleanup", &mut codebook, 3);
        assert!(!results.is_empty());
        // At least one result should be in StoreManagement category
        assert!(
            results
                .iter()
                .any(|r| r.article.category == KnowledgeCategory::StoreManagement),
            "GC query should find store management articles"
        );
    }

    #[test]
    fn test_empty_query_still_returns_results() {
        let mut codebook = NixCodebook::new();
        let kb = KnowledgeBase::new(&mut codebook);
        let results = kb.search("", &mut codebook, 5);
        // Empty query encodes to zero vector — should still return results (all with ~0 similarity)
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_search_respects_k_limit() {
        let mut codebook = NixCodebook::new();
        let kb = KnowledgeBase::new(&mut codebook);
        let results = kb.search("error", &mut codebook, 3);
        assert!(results.len() <= 3);

        let results = kb.search("error", &mut codebook, 1);
        assert_eq!(results.len(), 1);
    }

    fn make_dynamic_article(id: &str, title: &str) -> DynamicKnowledgeArticle {
        DynamicKnowledgeArticle {
            id: id.to_string(),
            title: title.to_string(),
            category: KnowledgeCategory::ServiceIssue,
            symptoms: vec!["custom error message".to_string()],
            solution: "Restart the custom service".to_string(),
            commands: vec!["systemctl restart custom".to_string()],
            learned_at: 1700000000,
            hit_count: 0,
        }
    }

    #[test]
    fn test_add_dynamic_article() {
        let mut codebook = NixCodebook::new();
        let mut kb = KnowledgeBase::new(&mut codebook);
        assert_eq!(kb.dynamic_len(), 0);

        kb.add_learned_article(
            make_dynamic_article("dyn-1", "Custom service crash fix"),
            &mut codebook,
        );
        assert_eq!(kb.dynamic_len(), 1);
    }

    #[test]
    fn test_search_all_finds_dynamic_articles() {
        let mut codebook = NixCodebook::new();
        let mut kb = KnowledgeBase::new(&mut codebook);
        kb.add_learned_article(
            make_dynamic_article("dyn-custom", "Custom service crash after upgrade"),
            &mut codebook,
        );

        let results = kb.search_all("custom service crash", &mut codebook, 5);
        assert!(!results.is_empty());
        let has_dynamic = results
            .iter()
            .any(|m| matches!(m, AnyKnowledgeMatch::Dynamic { .. }));
        assert!(has_dynamic, "search_all should find dynamic articles");
    }

    #[test]
    fn test_dynamic_save_load_roundtrip() {
        let mut codebook = NixCodebook::new();
        let mut kb = KnowledgeBase::new(&mut codebook);
        kb.add_learned_article(
            make_dynamic_article("dyn-save", "Persisted article"),
            &mut codebook,
        );

        let json = kb.save_dynamic();
        assert!(json.contains("dyn-save"));

        let mut kb2 = KnowledgeBase::new(&mut codebook);
        kb2.load_dynamic(&json, &mut codebook);
        assert_eq!(kb2.dynamic_len(), 1);
    }

    #[test]
    fn test_dynamic_serialization() {
        let article = make_dynamic_article("dyn-ser", "Serialization test");
        let json = serde_json::to_string(&article).unwrap();
        let restored: DynamicKnowledgeArticle = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.id, "dyn-ser");
        assert_eq!(restored.hit_count, 0);
    }

    #[test]
    fn test_dynamic_hit_count_tracking() {
        let mut codebook = NixCodebook::new();
        let mut kb = KnowledgeBase::new(&mut codebook);
        kb.add_learned_article(
            make_dynamic_article("dyn-hits", "Service restart workaround"),
            &mut codebook,
        );

        // Search should increment hit count
        let _ = kb.search_all("service restart workaround", &mut codebook, 5);
        // The hit count may or may not be incremented depending on whether the
        // dynamic article was in the top-k results. Let's just verify it doesn't panic.
        assert!(kb.dynamic_len() == 1);
    }

    #[test]
    fn test_any_match_accessors() {
        let mut codebook = NixCodebook::new();
        let mut kb = KnowledgeBase::new(&mut codebook);
        kb.add_learned_article(
            make_dynamic_article("dyn-acc", "Accessor test article"),
            &mut codebook,
        );

        let results = kb.search_all("accessor test", &mut codebook, 20);
        // Should have both static and dynamic results
        let has_static = results.iter().any(|r| !r.is_dynamic());
        let has_dynamic = results.iter().any(|r| r.is_dynamic());
        assert!(has_static, "should have static articles");
        assert!(has_dynamic, "should have dynamic articles");

        // Verify accessor methods work on both types
        for r in &results {
            assert!(!r.id().is_empty());
            assert!(!r.title().is_empty());
            assert!(!r.solution().is_empty());
            let _ = r.category();
            let _ = r.commands();
            assert!(r.similarity() >= 0.0);
        }
    }

    #[test]
    fn test_static_len() {
        let mut codebook = NixCodebook::new();
        let kb = KnowledgeBase::new(&mut codebook);
        assert!(kb.static_len() > 0, "should have static articles");
        assert_eq!(kb.dynamic_len(), 0, "no dynamic articles yet");
    }

    #[test]
    fn test_hit_count_increments_on_search_all() {
        let mut codebook = NixCodebook::new();
        let mut kb = KnowledgeBase::new(&mut codebook);

        kb.add_learned_article(
            make_dynamic_article("dyn-hit-test", "Custom service crash fix"),
            &mut codebook,
        );
        assert_eq!(kb.dynamic_hit_count("dyn-hit-test"), Some(0));

        // search_all with k large enough to include the dynamic article
        let results = kb.search_all("custom service crash fix", &mut codebook, 100);
        let dynamic_in_results = results
            .iter()
            .any(|m| m.is_dynamic() && m.id() == "dyn-hit-test");
        assert!(dynamic_in_results, "Dynamic article should be in results");

        // Hit count should have incremented
        assert_eq!(
            kb.dynamic_hit_count("dyn-hit-test"),
            Some(1),
            "Hit count should increment when article appears in search_all results"
        );

        // Search again — should increment again
        let _ = kb.search_all("custom service crash fix", &mut codebook, 100);
        assert_eq!(kb.dynamic_hit_count("dyn-hit-test"), Some(2));
    }

    #[test]
    fn test_hit_count_accessor_missing_id() {
        let mut codebook = NixCodebook::new();
        let kb = KnowledgeBase::new(&mut codebook);
        assert_eq!(kb.dynamic_hit_count("nonexistent"), None);
    }

    #[test]
    fn test_hit_count_preserved_through_save_load() {
        let mut codebook = NixCodebook::new();
        let mut kb = KnowledgeBase::new(&mut codebook);

        kb.add_learned_article(
            make_dynamic_article("dyn-persist-hit", "Persist hit test"),
            &mut codebook,
        );

        // Search to increment hit count
        let _ = kb.search_all("persist hit test", &mut codebook, 100);
        let hits_before = kb.dynamic_hit_count("dyn-persist-hit").unwrap();
        assert!(hits_before > 0);

        // Save and reload
        let json = kb.save_dynamic();
        let mut kb2 = KnowledgeBase::new(&mut codebook);
        kb2.load_dynamic(&json, &mut codebook);

        assert_eq!(
            kb2.dynamic_hit_count("dyn-persist-hit"),
            Some(hits_before),
            "Hit count should be preserved through save/load"
        );
    }

    #[test]
    fn test_search_sort_stability() {
        let mut codebook = NixCodebook::new();
        let kb = KnowledgeBase::new(&mut codebook);

        // Run the same search twice — results should be in the same order
        let r1 = kb.search("error hash", &mut codebook, 10);
        let r2 = kb.search("error hash", &mut codebook, 10);

        let ids1: Vec<&str> = r1.iter().map(|m| m.article.id).collect();
        let ids2: Vec<&str> = r2.iter().map(|m| m.article.id).collect();
        assert_eq!(ids1, ids2, "Sort should be deterministic across runs");
    }

    #[test]
    fn test_search_all_sort_stability() {
        let mut codebook = NixCodebook::new();
        let mut kb = KnowledgeBase::new(&mut codebook);
        kb.add_learned_article(
            make_dynamic_article("dyn-stab-1", "Stability test A"),
            &mut codebook,
        );
        kb.add_learned_article(
            make_dynamic_article("dyn-stab-2", "Stability test B"),
            &mut codebook,
        );

        let r1 = kb.search_all("stability test", &mut codebook, 10);
        let ids1: Vec<String> = r1.iter().map(|m| m.id().to_string()).collect();
        drop(r1);

        let r2 = kb.search_all("stability test", &mut codebook, 10);
        let ids2: Vec<String> = r2.iter().map(|m| m.id().to_string()).collect();

        assert_eq!(ids1, ids2, "search_all sort should be deterministic");
    }
}
