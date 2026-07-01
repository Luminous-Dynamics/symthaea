// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Nix store observation and analysis.
//!
//! Parses output from `nix store info`, `nix path-info`, and `nix-store --gc`
//! commands to observe store size, path counts, closure sizes, and GC roots.

use std::process::Command;
use tracing::debug;

/// Observes the Nix store: paths, closures, references, and disk usage.
pub struct StoreObserver;

/// Aggregate information about the Nix store.
#[derive(Debug, Clone, Default)]
pub struct StoreInfo {
    /// The store path prefix (typically `/nix/store`).
    pub store_path: String,
    /// Total size of all store paths in bytes.
    pub total_size_bytes: u64,
    /// Number of store paths.
    pub path_count: usize,
    /// Number of paths that have a deriver (were built, not fetched).
    pub deriver_count: usize,
}

/// A garbage-collection root that keeps a store path alive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GcRoot {
    /// The filesystem path of the GC root symlink.
    pub path: String,
    /// The store path it references.
    pub store_path: String,
}

impl StoreObserver {
    /// Gather aggregate store information by running `nix path-info --all -S`.
    ///
    /// We count the paths, sum up closure sizes, and detect derivers.
    pub fn store_info() -> Result<StoreInfo, std::io::Error> {
        let output = Command::new("nix")
            .args(["path-info", "--all", "-S"])
            .output()?;

        if !output.status.success() {
            return Err(std::io::Error::other(format!(
                "nix path-info --all -S failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Self::parse_store_info(&stdout)
    }

    /// Get the closure size for a specific store path.
    pub fn path_size(path: &str) -> Result<u64, std::io::Error> {
        let output = Command::new("nix")
            .args(["path-info", "-S", path])
            .output()?;

        if !output.status.success() {
            return Err(std::io::Error::other(format!(
                "nix path-info -S failed for '{}': {}",
                path,
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Self::parse_path_size(&stdout)
    }

    /// List GC roots via `nix-store --gc --print-roots`.
    ///
    /// Output format per line:
    /// ```text
    /// /nix/var/nix/gcroots/auto/xyz -> /nix/store/abc123-some-package
    /// ```
    pub fn roots() -> Result<Vec<GcRoot>, std::io::Error> {
        let output = Command::new("nix-store")
            .args(["--gc", "--print-roots"])
            .output()?;

        if !output.status.success() {
            return Err(std::io::Error::other(format!(
                "nix-store --gc --print-roots failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Self::parse_gc_roots(&stdout)
    }

    // ---- Parsing helpers ----

    /// Parse the output of `nix path-info --all -S`.
    ///
    /// Each line: `/nix/store/hash-name  <closure-size>`
    pub fn parse_store_info(output: &str) -> Result<StoreInfo, std::io::Error> {
        let mut total_size_bytes: u64 = 0;
        let mut path_count: usize = 0;
        let mut deriver_count: usize = 0;

        for line in output.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            path_count += 1;

            // If there is a second column, it is the closure size in bytes
            if parts.len() >= 2 {
                match parts[1].parse::<u64>() {
                    Ok(size) => total_size_bytes += size,
                    Err(_) => {
                        debug!(
                            path = parts[0],
                            raw_size = parts[1],
                            "Skipping unparseable store path size"
                        );
                    }
                }
            }

            // A very rough heuristic: paths containing ".drv" are derivers.
            // In practice we would check the `deriver` field; for now we count
            // any path whose name segment contains "-" after the hash (which
            // is nearly all real paths). We instead count paths with a deriver
            // column when available. As a simplification, count all paths.
            if parts[0].contains(".drv") {
                deriver_count += 1;
            }
        }

        Ok(StoreInfo {
            store_path: "/nix/store".to_string(),
            total_size_bytes,
            path_count,
            deriver_count,
        })
    }

    /// Parse the size from a single `nix path-info -S <path>` result.
    ///
    /// Expected format: `/nix/store/hash-name   <size-in-bytes>`
    pub fn parse_path_size(output: &str) -> Result<u64, std::io::Error> {
        for line in output.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(size) = parts[1].parse::<u64>() {
                    return Ok(size);
                }
            }
        }

        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("could not parse size from nix path-info output: {output}"),
        ))
    }

    /// Parse GC root output from `nix-store --gc --print-roots`.
    ///
    /// Each line: `<root-path> -> <store-path>`
    pub fn parse_gc_roots(output: &str) -> Result<Vec<GcRoot>, std::io::Error> {
        let mut roots = Vec::new();

        for line in output.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            // Split on " -> "
            if let Some((left, right)) = trimmed.split_once(" -> ") {
                roots.push(GcRoot {
                    path: left.trim().to_string(),
                    store_path: right.trim().to_string(),
                });
            }
        }

        Ok(roots)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MOCK_PATH_INFO: &str = "\
/nix/store/abc123-glibc-2.38   31457280
/nix/store/def456-nginx-1.25.3   5242880
/nix/store/ghi789-linux-6.1.72.drv   0
/nix/store/jkl012-firefox-121.0   157286400
";

    #[test]
    fn test_parse_store_info() {
        let info = StoreObserver::parse_store_info(MOCK_PATH_INFO).unwrap();
        assert_eq!(info.path_count, 4);
        assert_eq!(info.total_size_bytes, (31457280 + 5242880) + 157286400);
        assert_eq!(info.deriver_count, 1); // only the .drv path
        assert_eq!(info.store_path, "/nix/store");
    }

    #[test]
    fn test_parse_store_info_empty() {
        let info = StoreObserver::parse_store_info("").unwrap();
        assert_eq!(info.path_count, 0);
        assert_eq!(info.total_size_bytes, 0);
    }

    #[test]
    fn test_parse_path_size() {
        let output = "/nix/store/abc123-nginx-1.25.3   5242880\n";
        let size = StoreObserver::parse_path_size(output).unwrap();
        assert_eq!(size, 5242880);
    }

    #[test]
    fn test_parse_path_size_invalid() {
        let result = StoreObserver::parse_path_size("no-size-here");
        assert!(result.is_err());
    }

    const MOCK_GC_ROOTS: &str = "\
/nix/var/nix/gcroots/auto/abcdef -> /nix/store/abc123-system
/nix/var/nix/profiles/system-42-link -> /nix/store/xyz789-nixos-system
/home/user/.local/state/nix/profiles/profile-1-link -> /nix/store/qrs456-user-env
";

    #[test]
    fn test_parse_gc_roots() {
        let roots = StoreObserver::parse_gc_roots(MOCK_GC_ROOTS).unwrap();
        assert_eq!(roots.len(), 3);

        assert_eq!(roots[0].path, "/nix/var/nix/gcroots/auto/abcdef");
        assert_eq!(roots[0].store_path, "/nix/store/abc123-system");

        assert_eq!(roots[1].path, "/nix/var/nix/profiles/system-42-link");
        assert_eq!(roots[1].store_path, "/nix/store/xyz789-nixos-system");

        assert_eq!(
            roots[2].path,
            "/home/user/.local/state/nix/profiles/profile-1-link"
        );
        assert_eq!(roots[2].store_path, "/nix/store/qrs456-user-env");
    }

    #[test]
    fn test_parse_gc_roots_empty() {
        let roots = StoreObserver::parse_gc_roots("").unwrap();
        assert!(roots.is_empty());
    }

    #[test]
    fn test_parse_store_info_unparseable_size() {
        // Lines with non-numeric sizes should be counted as paths but not add to total_size
        let output = "/nix/store/abc123-pkg   not_a_number\n\
                       /nix/store/def456-pkg   1024\n";
        let info = StoreObserver::parse_store_info(output).unwrap();
        assert_eq!(info.path_count, 2);
        assert_eq!(info.total_size_bytes, 1024); // only the parseable one
    }

    #[test]
    fn test_parse_store_info_path_only_no_size() {
        // Lines with only a path (no size column)
        let output = "/nix/store/abc123-pkg\n";
        let info = StoreObserver::parse_store_info(output).unwrap();
        assert_eq!(info.path_count, 1);
        assert_eq!(info.total_size_bytes, 0);
    }

    #[test]
    fn test_parse_gc_roots_no_arrow() {
        // Lines without " -> " should be silently skipped
        let output = "this line has no arrow\n\
                       /valid/root -> /nix/store/abc123-pkg\n";
        let roots = StoreObserver::parse_gc_roots(output).unwrap();
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0].path, "/valid/root");
    }

    #[test]
    fn test_parse_path_size_empty() {
        let result = StoreObserver::parse_path_size("");
        assert!(result.is_err());
    }
}
