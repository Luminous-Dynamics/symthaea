// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fail-closed pre-construction environment isolation for GEOM D0/D1.
//!
//! This crate intentionally runs *before* `CognitiveLoopService::new` so
//! process-global environment channels cannot silently alter the experimental
//! agent before the campaign has recorded its boundary conditions.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Component, Path, PathBuf};
use symthaea::cognitive_loop::CognitiveLoopConfig;

/// Environment variables forbidden in the first offline GEOM campaign.
///
/// Values are never copied into reports; only presence/absence is recorded.
pub const FORBIDDEN_ENVIRONMENT_VARIABLES: [&str; 5] = [
    "SYMTHAEA_THRESHOLD_OVERRIDES_PATH",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "MYCELIX_CONDUCTOR_URL",
    "MYCELIX_APP_ID",
];

/// Presence-only view of one environment variable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvironmentVariablePresence {
    pub name: String,
    pub present: bool,
}

/// Presence-only snapshot used by the preflight.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvironmentSnapshot {
    pub variables: Vec<EnvironmentVariablePresence>,
}

impl EnvironmentSnapshot {
    /// Capture the exact forbidden-variable presence surface from this process.
    pub fn capture() -> Self {
        Self {
            variables: FORBIDDEN_ENVIRONMENT_VARIABLES
                .iter()
                .map(|name| EnvironmentVariablePresence {
                    name: (*name).to_string(),
                    present: std::env::var_os(name).is_some(),
                })
                .collect(),
        }
    }

    /// Construct a deterministic synthetic snapshot, useful for independent tests.
    pub fn from_present_names(present_names: &[&str]) -> Self {
        Self {
            variables: FORBIDDEN_ENVIRONMENT_VARIABLES
                .iter()
                .map(|name| EnvironmentVariablePresence {
                    name: (*name).to_string(),
                    present: present_names.iter().any(|present| present == name),
                })
                .collect(),
        }
    }

    fn has_exact_contract(&self) -> bool {
        self.variables.len() == FORBIDDEN_ENVIRONMENT_VARIABLES.len()
            && FORBIDDEN_ENVIRONMENT_VARIABLES.iter().all(|required| {
                self.variables
                    .iter()
                    .filter(|entry| entry.name == *required)
                    .count()
                    == 1
            })
            && self.variables.iter().all(|entry| {
                FORBIDDEN_ENVIRONMENT_VARIABLES
                    .iter()
                    .any(|required| entry.name == *required)
            })
    }

    fn first_forbidden_present(&self) -> Option<&str> {
        self.variables
            .iter()
            .find(|entry| entry.present)
            .map(|entry| entry.name.as_str())
    }
}

/// Serializable evidence that the pre-construction environment was accepted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeomEnvironmentPreflightReport {
    pub fixed_utc_hour: f64,
    pub genesis_phrase_present: bool,
    pub async_training: bool,
    pub online_learning: bool,
    pub canonical_persistence_root: String,
    pub persistence_root_empty: bool,
    pub aesthetic_memory_path: String,
    pub memory_db_path: Option<String>,
    pub epistemic_auditor_db_path: Option<String>,
    pub environment: EnvironmentSnapshot,
}

/// Fail-closed preflight errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeomEnvironmentPreflightError {
    InvalidFixedUtcHour,
    MissingGenesisPhrase,
    AsyncTrainingEnabled,
    OnlineLearningEnabled,
    InvalidEnvironmentSnapshot,
    ForbiddenEnvironmentVariable { name: String },
    PersistenceRootMissing,
    PersistenceRootNotDirectory,
    PersistenceRootUnreadable,
    PersistenceRootNotEmpty,
    MissingAestheticMemoryPath,
    PersistencePathNotAbsolute { field: &'static str },
    PersistencePathEscapesRoot { field: &'static str },
}

impl fmt::Display for GeomEnvironmentPreflightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFixedUtcHour => {
                write!(f, "fixed UTC hour must be finite and in [0, 24)")
            }
            Self::MissingGenesisPhrase => write!(f, "genesis_phrase must be explicit and non-empty"),
            Self::AsyncTrainingEnabled => write!(f, "async_training must be disabled"),
            Self::OnlineLearningEnabled => write!(f, "enable_online_learning must be disabled"),
            Self::InvalidEnvironmentSnapshot => write!(
                f,
                "environment snapshot must contain each forbidden variable exactly once"
            ),
            Self::ForbiddenEnvironmentVariable { name } => {
                write!(f, "forbidden GEOM environment variable is present: {name}")
            }
            Self::PersistenceRootMissing => write!(f, "campaign persistence root does not exist"),
            Self::PersistenceRootNotDirectory => {
                write!(f, "campaign persistence root is not a directory")
            }
            Self::PersistenceRootUnreadable => {
                write!(f, "campaign persistence root cannot be read/canonicalized")
            }
            Self::PersistenceRootNotEmpty => {
                write!(f, "campaign persistence root must be empty before construction")
            }
            Self::MissingAestheticMemoryPath => write!(
                f,
                "aesthetic_memory_path must be explicit for the primary GEOM campaign"
            ),
            Self::PersistencePathNotAbsolute { field } => {
                write!(f, "{field} must be an absolute path")
            }
            Self::PersistencePathEscapesRoot { field } => {
                write!(f, "{field} must be contained within the campaign persistence root")
            }
        }
    }
}

impl std::error::Error for GeomEnvironmentPreflightError {}

/// Run the production preflight against the current process environment.
pub fn preflight_primary_campaign(
    config: &CognitiveLoopConfig,
    fixed_utc_hour: f64,
    persistence_root: &Path,
) -> Result<GeomEnvironmentPreflightReport, GeomEnvironmentPreflightError> {
    preflight_primary_campaign_with_environment(
        config,
        fixed_utc_hour,
        persistence_root,
        EnvironmentSnapshot::capture(),
    )
}

/// Deterministic form of the preflight with an explicitly supplied
/// presence-only environment snapshot.
pub fn preflight_primary_campaign_with_environment(
    config: &CognitiveLoopConfig,
    fixed_utc_hour: f64,
    persistence_root: &Path,
    environment: EnvironmentSnapshot,
) -> Result<GeomEnvironmentPreflightReport, GeomEnvironmentPreflightError> {
    if !fixed_utc_hour.is_finite() || !(0.0..24.0).contains(&fixed_utc_hour) {
        return Err(GeomEnvironmentPreflightError::InvalidFixedUtcHour);
    }

    if config
        .genesis_phrase
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .is_none()
    {
        return Err(GeomEnvironmentPreflightError::MissingGenesisPhrase);
    }
    if config.async_training {
        return Err(GeomEnvironmentPreflightError::AsyncTrainingEnabled);
    }
    if config.enable_online_learning {
        return Err(GeomEnvironmentPreflightError::OnlineLearningEnabled);
    }

    if !environment.has_exact_contract() {
        return Err(GeomEnvironmentPreflightError::InvalidEnvironmentSnapshot);
    }
    if let Some(name) = environment.first_forbidden_present() {
        return Err(GeomEnvironmentPreflightError::ForbiddenEnvironmentVariable {
            name: name.to_string(),
        });
    }

    if !persistence_root.exists() {
        return Err(GeomEnvironmentPreflightError::PersistenceRootMissing);
    }
    if !persistence_root.is_dir() {
        return Err(GeomEnvironmentPreflightError::PersistenceRootNotDirectory);
    }
    let canonical_root = persistence_root
        .canonicalize()
        .map_err(|_| GeomEnvironmentPreflightError::PersistenceRootUnreadable)?;
    let mut entries = std::fs::read_dir(&canonical_root)
        .map_err(|_| GeomEnvironmentPreflightError::PersistenceRootUnreadable)?;
    if entries.next().is_some() {
        return Err(GeomEnvironmentPreflightError::PersistenceRootNotEmpty);
    }

    let aesthetic = config
        .aesthetic_memory_path
        .as_deref()
        .ok_or(GeomEnvironmentPreflightError::MissingAestheticMemoryPath)?;
    let aesthetic = validate_campaign_path("aesthetic_memory_path", aesthetic, &canonical_root)?;
    let memory_db = validate_optional_campaign_path(
        "memory_db_path",
        config.memory_db_path.as_deref(),
        &canonical_root,
    )?;
    let epistemic_db = validate_optional_campaign_path(
        "epistemic_auditor_db_path",
        config.epistemic_auditor_db_path.as_deref(),
        &canonical_root,
    )?;

    Ok(GeomEnvironmentPreflightReport {
        fixed_utc_hour,
        genesis_phrase_present: true,
        async_training: false,
        online_learning: false,
        canonical_persistence_root: canonical_root.to_string_lossy().into_owned(),
        persistence_root_empty: true,
        aesthetic_memory_path: aesthetic.to_string_lossy().into_owned(),
        memory_db_path: memory_db.map(|path| path.to_string_lossy().into_owned()),
        epistemic_auditor_db_path: epistemic_db
            .map(|path| path.to_string_lossy().into_owned()),
        environment,
    })
}

fn validate_optional_campaign_path(
    field: &'static str,
    value: Option<&str>,
    canonical_root: &Path,
) -> Result<Option<PathBuf>, GeomEnvironmentPreflightError> {
    value
        .map(|path| validate_campaign_path(field, path, canonical_root))
        .transpose()
}

fn validate_campaign_path(
    field: &'static str,
    value: &str,
    canonical_root: &Path,
) -> Result<PathBuf, GeomEnvironmentPreflightError> {
    let path = PathBuf::from(value);
    if !path.is_absolute() {
        return Err(GeomEnvironmentPreflightError::PersistencePathNotAbsolute { field });
    }
    if path
        .components()
        .any(|component| matches!(component, Component::ParentDir))
        || path == canonical_root
        || !path.starts_with(canonical_root)
    {
        return Err(GeomEnvironmentPreflightError::PersistencePathEscapesRoot { field });
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

    fn unique_temp_path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "symthaea_geom_preflight_{label}_{}_{}",
            std::process::id(),
            NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn temp_root(label: &str) -> PathBuf {
        let root = unique_temp_path(label);
        std::fs::create_dir_all(&root).expect("create temp root");
        root.canonicalize().expect("canonical temp root")
    }

    fn clean_config(root: &Path) -> CognitiveLoopConfig {
        let mut config = CognitiveLoopConfig::default();
        config.genesis_phrase = Some("geom-preflight-test-seed".to_string());
        config.async_training = false;
        config.enable_online_learning = false;
        config.aesthetic_memory_path = Some(root.join("aesthetic.json").to_string_lossy().into());
        config.memory_db_path = None;
        config.epistemic_auditor_db_path = None;
        config
    }

    fn no_env() -> EnvironmentSnapshot {
        EnvironmentSnapshot::from_present_names(&[])
    }

    #[test]
    fn clean_primary_campaign_passes() {
        let root = temp_root("clean");
        let config = clean_config(&root);
        let report = preflight_primary_campaign_with_environment(&config, 12.5, &root, no_env())
            .expect("clean environment should pass");
        assert_eq!(report.fixed_utc_hour, 12.5);
        assert!(report.genesis_phrase_present);
        assert!(report.persistence_root_empty);
        assert!(report.environment.variables.iter().all(|entry| !entry.present));
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn every_forbidden_environment_channel_fails() {
        for name in FORBIDDEN_ENVIRONMENT_VARIABLES {
            let root = temp_root("forbidden_env");
            let config = clean_config(&root);
            assert_eq!(
                preflight_primary_campaign_with_environment(
                    &config,
                    12.0,
                    &root,
                    EnvironmentSnapshot::from_present_names(&[name]),
                ),
                Err(GeomEnvironmentPreflightError::ForbiddenEnvironmentVariable {
                    name: name.to_string(),
                })
            );
            std::fs::remove_dir_all(root).ok();
        }
    }

    #[test]
    fn malformed_environment_snapshot_fails_closed() {
        let root = temp_root("env_contract");
        let config = clean_config(&root);
        assert_eq!(
            preflight_primary_campaign_with_environment(
                &config,
                12.0,
                &root,
                EnvironmentSnapshot { variables: Vec::new() },
            ),
            Err(GeomEnvironmentPreflightError::InvalidEnvironmentSnapshot)
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn invalid_fixed_hour_fails() {
        let root = temp_root("hour");
        let config = clean_config(&root);
        for hour in [f64::NAN, f64::INFINITY, -0.1, 24.0] {
            assert_eq!(
                preflight_primary_campaign_with_environment(&config, hour, &root, no_env()),
                Err(GeomEnvironmentPreflightError::InvalidFixedUtcHour)
            );
        }
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn missing_genesis_and_adaptive_training_fail() {
        let root = temp_root("config");
        let mut config = clean_config(&root);
        config.genesis_phrase = None;
        assert_eq!(
            preflight_primary_campaign_with_environment(&config, 12.0, &root, no_env()),
            Err(GeomEnvironmentPreflightError::MissingGenesisPhrase)
        );

        let mut config = clean_config(&root);
        config.async_training = true;
        assert_eq!(
            preflight_primary_campaign_with_environment(&config, 12.0, &root, no_env()),
            Err(GeomEnvironmentPreflightError::AsyncTrainingEnabled)
        );

        let mut config = clean_config(&root);
        config.enable_online_learning = true;
        assert_eq!(
            preflight_primary_campaign_with_environment(&config, 12.0, &root, no_env()),
            Err(GeomEnvironmentPreflightError::OnlineLearningEnabled)
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn persistence_root_must_exist_be_directory_and_be_empty() {
        let missing = unique_temp_path("missing");
        let mut missing_config = CognitiveLoopConfig::default();
        missing_config.genesis_phrase = Some("geom-preflight-test-seed".to_string());
        missing_config.async_training = false;
        missing_config.enable_online_learning = false;
        missing_config.aesthetic_memory_path =
            Some(missing.join("aesthetic.json").to_string_lossy().into_owned());
        assert_eq!(
            preflight_primary_campaign_with_environment(
                &missing_config,
                12.0,
                &missing,
                no_env(),
            ),
            Err(GeomEnvironmentPreflightError::PersistenceRootMissing)
        );

        let root = temp_root("dirty");
        let config = clean_config(&root);
        std::fs::write(root.join("historical-state"), b"not clean").unwrap();
        assert_eq!(
            preflight_primary_campaign_with_environment(&config, 12.0, &root, no_env()),
            Err(GeomEnvironmentPreflightError::PersistenceRootNotEmpty)
        );
        std::fs::remove_dir_all(root).ok();

        let parent = temp_root("file_parent");
        let file = parent.join("not-a-directory");
        std::fs::write(&file, b"x").unwrap();
        let config = clean_config(&parent);
        assert_eq!(
            preflight_primary_campaign_with_environment(&config, 12.0, &file, no_env()),
            Err(GeomEnvironmentPreflightError::PersistenceRootNotDirectory)
        );
        std::fs::remove_dir_all(parent).ok();
    }

    #[test]
    fn aesthetic_path_is_mandatory_and_all_persistence_paths_must_stay_under_root() {
        let root = temp_root("paths");
        let mut config = clean_config(&root);
        config.aesthetic_memory_path = None;
        assert_eq!(
            preflight_primary_campaign_with_environment(&config, 12.0, &root, no_env()),
            Err(GeomEnvironmentPreflightError::MissingAestheticMemoryPath)
        );

        let mut config = clean_config(&root);
        config.aesthetic_memory_path = Some("relative/aesthetic.json".to_string());
        assert_eq!(
            preflight_primary_campaign_with_environment(&config, 12.0, &root, no_env()),
            Err(GeomEnvironmentPreflightError::PersistencePathNotAbsolute {
                field: "aesthetic_memory_path",
            })
        );

        let mut config = clean_config(&root);
        config.aesthetic_memory_path = Some(
            root.join("nested")
                .join("..")
                .join("aesthetic.json")
                .to_string_lossy()
                .into_owned(),
        );
        assert_eq!(
            preflight_primary_campaign_with_environment(&config, 12.0, &root, no_env()),
            Err(GeomEnvironmentPreflightError::PersistencePathEscapesRoot {
                field: "aesthetic_memory_path",
            })
        );

        let outside = unique_temp_path("outside").with_extension("sqlite");
        let mut config = clean_config(&root);
        config.memory_db_path = Some(outside.to_string_lossy().into_owned());
        assert_eq!(
            preflight_primary_campaign_with_environment(&config, 12.0, &root, no_env()),
            Err(GeomEnvironmentPreflightError::PersistencePathEscapesRoot {
                field: "memory_db_path",
            })
        );

        let mut config = clean_config(&root);
        config.epistemic_auditor_db_path =
            Some(root.join("audit.duckdb").to_string_lossy().into_owned());
        assert!(
            preflight_primary_campaign_with_environment(&config, 12.0, &root, no_env()).is_ok()
        );
        std::fs::remove_dir_all(root).ok();
    }
}
