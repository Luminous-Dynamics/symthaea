// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic, fail-closed application identity resolution.
//!
//! `symthaea-app-db` owns the canonical migration knowledge corpus. This crate
//! owns the separate theorem for resolving observed names/identifiers into that
//! corpus without allowing hash iteration order, namespace confusion, or
//! ambiguous fuzzy candidates to manufacture canonical application identity.
//!
//! Resolution order for an untyped observation:
//!
//! 1. exact alias/identifier match;
//! 2. normalized exact match;
//! 3. complete substring candidate collection;
//! 4. promote only when exactly one canonical application remains.
//!
//! Structured observations can additionally retain typed WinGet/Flatpak/Snap/
//! package-manager identities. Typed identifiers are resolved only inside their
//! declared namespace and can expose contradictions with display-name evidence.
//!
//! `Ambiguous != Resolved` and `NamespaceMismatch != Match` are core invariants.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use symthaea_app_db::{AppDatabase, AppEntry};

/// Why one application identity was accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolutionBasis {
    /// Byte-insensitive/case-insensitive exact alias or package identifier.
    Exact,
    /// Exact after conservative punctuation/whitespace normalization.
    NormalizedExact,
    /// Exactly one canonical application remained after collecting every
    /// substring candidate from the normalized alias corpus.
    UniqueSubstring,
    /// One or more typed identifiers uniquely identified the same application.
    TypedIdentifiers,
}

/// Result of resolving one observed application identity.
#[derive(Debug, Clone)]
pub enum AppResolution {
    Resolved {
        entry: &'static AppEntry,
        basis: ResolutionBasis,
    },
    Ambiguous {
        query: String,
        candidates: Vec<&'static AppEntry>,
    },
    Unknown {
        query: String,
    },
}

impl AppResolution {
    /// Return an application only when resolution is unique.
    pub fn resolved_entry(&self) -> Option<&'static AppEntry> {
        match self {
            Self::Resolved { entry, .. } => Some(*entry),
            Self::Ambiguous { .. } | Self::Unknown { .. } => None,
        }
    }

    pub fn basis(&self) -> Option<ResolutionBasis> {
        match self {
            Self::Resolved { basis, .. } => Some(*basis),
            Self::Ambiguous { .. } | Self::Unknown { .. } => None,
        }
    }

    pub fn is_ambiguous(&self) -> bool {
        matches!(self, Self::Ambiguous { .. })
    }
}

/// Namespace of one observed application identifier.
///
/// These values are deliberately not interchangeable. For example, a WinGet ID
/// is never looked up in the Flatpak namespace merely because the text happens
/// to match an alias there.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ApplicationIdentifierKind {
    CanonicalName,
    WindowsName,
    MacosName,
    LinuxPackage,
    FlatpakId,
    SnapName,
    WingetId,
    BrewName,
}

/// One exact observed identifier with an explicit namespace.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ObservedAppIdentifierV1 {
    kind: ApplicationIdentifierKind,
    value: String,
}

impl ObservedAppIdentifierV1 {
    pub fn new(
        kind: ApplicationIdentifierKind,
        value: impl Into<String>,
    ) -> Result<Self, IdentityObservationError> {
        let value = checked_identity_text(value.into())?;
        Ok(Self { kind, value })
    }

    pub fn kind(&self) -> ApplicationIdentifierKind {
        self.kind
    }

    pub fn value(&self) -> &str {
        &self.value
    }
}

/// Structured application observation preserving display-name and typed IDs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservedApplicationIdentityV1 {
    display_name: Option<String>,
    identifiers: Vec<ObservedAppIdentifierV1>,
}

impl ObservedApplicationIdentityV1 {
    pub fn new(
        display_name: Option<String>,
        mut identifiers: Vec<ObservedAppIdentifierV1>,
    ) -> Result<Self, IdentityObservationError> {
        let display_name = match display_name {
            Some(value) if !value.trim().is_empty() => Some(checked_identity_text(value)?),
            _ => None,
        };
        identifiers.sort();
        identifiers.dedup();
        if display_name.is_none() && identifiers.is_empty() {
            return Err(IdentityObservationError::EmptyObservation);
        }
        Ok(Self {
            display_name,
            identifiers,
        })
    }

    pub fn display_name(&self) -> Option<&str> {
        self.display_name.as_deref()
    }

    pub fn identifiers(&self) -> &[ObservedAppIdentifierV1] {
        &self.identifiers
    }
}

/// Structured resolution preserving typed identifiers that were not recognized.
#[derive(Debug, Clone)]
pub struct ObservedAppResolution {
    resolution: AppResolution,
    matched_identifier_count: usize,
    unresolved_identifiers: Vec<ObservedAppIdentifierV1>,
}

impl ObservedAppResolution {
    pub fn resolution(&self) -> &AppResolution {
        &self.resolution
    }

    pub fn resolved_entry(&self) -> Option<&'static AppEntry> {
        self.resolution.resolved_entry()
    }

    pub fn matched_identifier_count(&self) -> usize {
        self.matched_identifier_count
    }

    pub fn unresolved_identifiers(&self) -> &[ObservedAppIdentifierV1] {
        &self.unresolved_identifiers
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IdentityObservationError {
    EmptyIdentifier,
    IdentifierTooLong,
    ControlCharacters,
    EmptyObservation,
}

/// One alias collision already present in the migration corpus.
#[derive(Debug, Clone)]
pub struct AliasCollision {
    pub alias: String,
    pub candidates: Vec<&'static AppEntry>,
}

/// Deterministic resolver built from the canonical `symthaea-app-db` corpus.
pub struct AppResolver {
    entries: Vec<&'static AppEntry>,
    exact_index: BTreeMap<String, BTreeSet<usize>>,
    normalized_index: BTreeMap<String, BTreeSet<usize>>,
    typed_index:
        BTreeMap<ApplicationIdentifierKind, BTreeMap<String, BTreeSet<usize>>>,
}

impl AppResolver {
    pub fn new(database: &AppDatabase) -> Self {
        let entries = database.entries().to_vec();
        let mut exact_index = BTreeMap::<String, BTreeSet<usize>>::new();
        let mut normalized_index = BTreeMap::<String, BTreeSet<usize>>::new();
        let mut typed_index = BTreeMap::<
            ApplicationIdentifierKind,
            BTreeMap<String, BTreeSet<usize>>,
        >::new();

        for (index, entry) in entries.iter().enumerate() {
            for alias in entry_aliases(entry) {
                insert_alias(&mut exact_index, exact_key(alias), index);
                let normalized = normalize_identity(alias);
                if !normalized.is_empty() {
                    insert_alias(&mut normalized_index, normalized, index);
                }
            }

            insert_typed_alias(
                &mut typed_index,
                ApplicationIdentifierKind::CanonicalName,
                entry.name,
                index,
            );
            for value in entry.windows_names {
                insert_typed_alias(
                    &mut typed_index,
                    ApplicationIdentifierKind::WindowsName,
                    value,
                    index,
                );
            }
            for value in entry.macos_names {
                insert_typed_alias(
                    &mut typed_index,
                    ApplicationIdentifierKind::MacosName,
                    value,
                    index,
                );
            }
            for value in entry.linux_names {
                insert_typed_alias(
                    &mut typed_index,
                    ApplicationIdentifierKind::LinuxPackage,
                    value,
                    index,
                );
            }
            for value in entry.flatpak_ids {
                insert_typed_alias(
                    &mut typed_index,
                    ApplicationIdentifierKind::FlatpakId,
                    value,
                    index,
                );
            }
            for value in entry.snap_names {
                insert_typed_alias(
                    &mut typed_index,
                    ApplicationIdentifierKind::SnapName,
                    value,
                    index,
                );
            }
            for value in entry.winget_ids {
                insert_typed_alias(
                    &mut typed_index,
                    ApplicationIdentifierKind::WingetId,
                    value,
                    index,
                );
            }
            for value in entry.brew_names {
                insert_typed_alias(
                    &mut typed_index,
                    ApplicationIdentifierKind::BrewName,
                    value,
                    index,
                );
            }
        }

        Self {
            entries,
            exact_index,
            normalized_index,
            typed_index,
        }
    }

    /// Resolve one untyped observed display name or package identifier.
    ///
    /// No ambiguous set is ever reduced by iteration order. A caller that needs
    /// a compatibility-style `Option` should use [`Self::resolve_unique`], which
    /// returns `None` for both ambiguous and unknown observations.
    pub fn resolve(&self, query: &str) -> AppResolution {
        let trimmed = query.trim();
        if trimmed.is_empty() {
            return AppResolution::Unknown {
                query: String::new(),
            };
        }

        let exact = exact_key(trimmed);
        if let Some(indices) = self.exact_index.get(&exact) {
            return self.classify(trimmed, indices, ResolutionBasis::Exact);
        }

        let normalized = normalize_identity(trimmed);
        if normalized.is_empty() {
            return AppResolution::Unknown {
                query: trimmed.to_string(),
            };
        }

        if let Some(indices) = self.normalized_index.get(&normalized) {
            return self.classify(trimmed, indices, ResolutionBasis::NormalizedExact);
        }

        // Last resort: collect the COMPLETE candidate set. Never return the
        // first map hit. Very short strings are too collision-prone for fuzzy
        // identity resolution and remain unknown.
        if normalized.chars().count() < 4 {
            return AppResolution::Unknown {
                query: trimmed.to_string(),
            };
        }

        let mut candidates = BTreeSet::new();
        for (alias, indices) in &self.normalized_index {
            if alias.chars().count() < 4 {
                continue;
            }
            if normalized.contains(alias) || alias.contains(&normalized) {
                candidates.extend(indices.iter().copied());
            }
        }

        if candidates.is_empty() {
            AppResolution::Unknown {
                query: trimmed.to_string(),
            }
        } else {
            self.classify(trimmed, &candidates, ResolutionBasis::UniqueSubstring)
        }
    }

    /// Resolve a structured observation while retaining identifier namespaces.
    ///
    /// Known typed identifiers are treated as exact identity evidence. If they
    /// point at more than one canonical application, the observation is
    /// ambiguous. If they uniquely identify one app but an exact/normalized
    /// display name identifies a different app, the observation is also
    /// ambiguous instead of silently choosing one source.
    pub fn resolve_observation(
        &self,
        observation: &ObservedApplicationIdentityV1,
    ) -> ObservedAppResolution {
        let mut typed_candidates = BTreeSet::new();
        let mut matched_identifier_count = 0usize;
        let mut unresolved_identifiers = Vec::new();

        for identifier in observation.identifiers() {
            let result = self
                .typed_index
                .get(&identifier.kind())
                .and_then(|index| index.get(&exact_key(identifier.value())));
            match result {
                Some(indices) => {
                    matched_identifier_count += 1;
                    typed_candidates.extend(indices.iter().copied());
                }
                None => unresolved_identifiers.push(identifier.clone()),
            }
        }

        let resolution = match typed_candidates.len() {
            0 => observation
                .display_name()
                .map(|name| self.resolve(name))
                .unwrap_or_else(|| AppResolution::Unknown {
                    query: observation_query(observation),
                }),
            1 => {
                let typed_index = *typed_candidates.iter().next().expect("length checked");
                if let Some(display_name) = observation.display_name() {
                    if let AppResolution::Resolved { entry, basis } = self.resolve(display_name) {
                        let display_index = self.entry_index(entry);
                        if display_index != Some(typed_index)
                            && matches!(basis, ResolutionBasis::Exact | ResolutionBasis::NormalizedExact)
                        {
                            let mut conflict = typed_candidates.clone();
                            if let Some(display_index) = display_index {
                                conflict.insert(display_index);
                            }
                            return ObservedAppResolution {
                                resolution: self.classify(
                                    &observation_query(observation),
                                    &conflict,
                                    ResolutionBasis::TypedIdentifiers,
                                ),
                                matched_identifier_count,
                                unresolved_identifiers,
                            };
                        }
                    }
                }
                AppResolution::Resolved {
                    entry: self.entries[typed_index],
                    basis: ResolutionBasis::TypedIdentifiers,
                }
            }
            _ => self.classify(
                &observation_query(observation),
                &typed_candidates,
                ResolutionBasis::TypedIdentifiers,
            ),
        };

        ObservedAppResolution {
            resolution,
            matched_identifier_count,
            unresolved_identifiers,
        }
    }

    /// Compatibility helper: return only uniquely resolved application identity.
    pub fn resolve_unique(&self, query: &str) -> Option<&'static AppEntry> {
        self.resolve(query).resolved_entry()
    }

    /// Report exact aliases/package identifiers that currently map to more than
    /// one canonical application. The list is deterministic and diagnostic; a
    /// collision remains fail-closed during resolution.
    pub fn exact_alias_collisions(&self) -> Vec<AliasCollision> {
        self.exact_index
            .iter()
            .filter(|(_, indices)| indices.len() > 1)
            .map(|(alias, indices)| AliasCollision {
                alias: alias.clone(),
                candidates: self.entries_for(indices),
            })
            .collect()
    }

    fn classify(
        &self,
        query: &str,
        indices: &BTreeSet<usize>,
        basis: ResolutionBasis,
    ) -> AppResolution {
        match indices.len() {
            0 => AppResolution::Unknown {
                query: query.to_string(),
            },
            1 => {
                let index = *indices.iter().next().expect("length checked");
                AppResolution::Resolved {
                    entry: self.entries[index],
                    basis,
                }
            }
            _ => AppResolution::Ambiguous {
                query: query.to_string(),
                candidates: self.entries_for(indices),
            },
        }
    }

    fn entries_for(&self, indices: &BTreeSet<usize>) -> Vec<&'static AppEntry> {
        let mut entries: Vec<_> = indices.iter().map(|index| self.entries[*index]).collect();
        entries.sort_by(|left, right| {
            left.name
                .to_ascii_lowercase()
                .cmp(&right.name.to_ascii_lowercase())
                .then_with(|| left.name.cmp(right.name))
        });
        entries
    }

    fn entry_index(&self, needle: &'static AppEntry) -> Option<usize> {
        self.entries
            .iter()
            .position(|entry| std::ptr::eq(*entry, needle))
    }
}

fn entry_aliases(entry: &AppEntry) -> Vec<&str> {
    std::iter::once(entry.name)
        .chain(entry.windows_names.iter().copied())
        .chain(entry.macos_names.iter().copied())
        .chain(entry.linux_names.iter().copied())
        .chain(entry.flatpak_ids.iter().copied())
        .chain(entry.snap_names.iter().copied())
        .chain(entry.winget_ids.iter().copied())
        .chain(entry.brew_names.iter().copied())
        .collect()
}

fn insert_alias(
    index: &mut BTreeMap<String, BTreeSet<usize>>,
    alias: String,
    entry_index: usize,
) {
    if alias.is_empty() {
        return;
    }
    index.entry(alias).or_default().insert(entry_index);
}

fn insert_typed_alias(
    typed_index: &mut BTreeMap<
        ApplicationIdentifierKind,
        BTreeMap<String, BTreeSet<usize>>,
    >,
    kind: ApplicationIdentifierKind,
    alias: &str,
    entry_index: usize,
) {
    let alias = exact_key(alias);
    if alias.is_empty() {
        return;
    }
    typed_index
        .entry(kind)
        .or_default()
        .entry(alias)
        .or_default()
        .insert(entry_index);
}

fn observation_query(observation: &ObservedApplicationIdentityV1) -> String {
    observation
        .display_name()
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| {
            observation
                .identifiers()
                .iter()
                .map(|identifier| format!("{:?}:{}", identifier.kind(), identifier.value()))
                .collect::<Vec<_>>()
                .join("+")
        })
}

fn checked_identity_text(value: String) -> Result<String, IdentityObservationError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(IdentityObservationError::EmptyIdentifier);
    }
    if trimmed.len() > 1024 {
        return Err(IdentityObservationError::IdentifierTooLong);
    }
    if trimmed.chars().any(char::is_control) {
        return Err(IdentityObservationError::ControlCharacters);
    }
    Ok(trimmed.to_string())
}

fn exact_key(value: &str) -> String {
    value.trim().to_lowercase()
}

/// Conservative identity normalization.
///
/// Punctuation and whitespace become separators; semantic tokens are never
/// deleted, reordered, stemmed, version-stripped, or guessed. More aggressive
/// interpretation belongs in explicit candidate generation, not exact identity.
pub fn normalize_identity(value: &str) -> String {
    let mut out = String::new();
    let mut pending_separator = false;

    for ch in value.trim().chars().flat_map(char::to_lowercase) {
        if ch.is_alphanumeric() {
            if pending_separator && !out.is_empty() {
                out.push(' ');
            }
            out.push(ch);
            pending_separator = false;
        } else {
            pending_separator = !out.is_empty();
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_winget_identifier_resolves() {
        let db = AppDatabase::new();
        let resolver = AppResolver::new(&db);
        let result = resolver.resolve("Google.Chrome");
        let entry = result.resolved_entry().expect("Google.Chrome should resolve");
        assert_eq!(entry.name, "Google Chrome");
        assert_eq!(result.basis(), Some(ResolutionBasis::Exact));
    }

    #[test]
    fn normalization_is_deterministic_and_conservative() {
        assert_eq!(normalize_identity("  Google.Chrome  "), "google chrome");
        assert_eq!(normalize_identity("Visual_Studio-Code"), "visual studio code");
        assert_eq!(normalize_identity("v1.2.3"), "v1 2 3");
    }

    #[test]
    fn normalized_exact_can_resolve_without_exact_alias() {
        let db = AppDatabase::new();
        let resolver = AppResolver::new(&db);
        let result = resolver.resolve("GOOGLE   CHROME");
        let entry = result.resolved_entry().expect("normalized name should resolve");
        assert_eq!(entry.name, "Google Chrome");
        assert_eq!(result.basis(), Some(ResolutionBasis::NormalizedExact));
    }

    #[test]
    fn short_fuzzy_queries_never_manufacture_identity() {
        let db = AppDatabase::new();
        let resolver = AppResolver::new(&db);
        assert!(matches!(
            resolver.resolve("go"),
            AppResolution::Unknown { .. }
        ));
    }

    #[test]
    fn candidate_classification_preserves_ambiguity() {
        let db = AppDatabase::new();
        let resolver = AppResolver::new(&db);
        assert!(resolver.entries.len() >= 2);
        let mut candidates = BTreeSet::new();
        candidates.insert(0);
        candidates.insert(1);
        let result = resolver.classify(
            "synthetic-collision",
            &candidates,
            ResolutionBasis::UniqueSubstring,
        );
        match result {
            AppResolution::Ambiguous { candidates, .. } => {
                assert_eq!(candidates.len(), 2);
                assert_ne!(candidates[0].name, candidates[1].name);
            }
            _ => panic!("multiple canonical candidates must remain ambiguous"),
        }
    }

    #[test]
    fn compatibility_helper_refuses_ambiguity_and_unknown() {
        let db = AppDatabase::new();
        let resolver = AppResolver::new(&db);
        assert!(resolver.resolve_unique("").is_none());
        assert!(resolver
            .resolve_unique("definitely-not-a-real-application-xyz")
            .is_none());
    }

    #[test]
    fn alias_collision_report_is_sorted_and_stable() {
        let db = AppDatabase::new();
        let resolver = AppResolver::new(&db);
        let collisions = resolver.exact_alias_collisions();
        for window in collisions.windows(2) {
            assert!(window[0].alias <= window[1].alias);
        }
        for collision in collisions {
            assert!(collision.candidates.len() > 1);
        }
    }

    #[test]
    fn typed_winget_id_resolves_without_display_name() {
        let db = AppDatabase::new();
        let resolver = AppResolver::new(&db);
        let observation = ObservedApplicationIdentityV1::new(
            None,
            vec![ObservedAppIdentifierV1::new(
                ApplicationIdentifierKind::WingetId,
                "Google.Chrome",
            )
            .unwrap()],
        )
        .unwrap();
        let result = resolver.resolve_observation(&observation);
        assert_eq!(
            result.resolved_entry().expect("typed WinGet ID should resolve").name,
            "Google Chrome"
        );
        assert_eq!(
            result.resolution().basis(),
            Some(ResolutionBasis::TypedIdentifiers)
        );
        assert_eq!(result.matched_identifier_count(), 1);
        assert!(result.unresolved_identifiers().is_empty());
    }

    #[test]
    fn namespace_mismatch_does_not_cross_match() {
        let db = AppDatabase::new();
        let resolver = AppResolver::new(&db);
        let observation = ObservedApplicationIdentityV1::new(
            None,
            vec![ObservedAppIdentifierV1::new(
                ApplicationIdentifierKind::FlatpakId,
                "Google.Chrome",
            )
            .unwrap()],
        )
        .unwrap();
        let result = resolver.resolve_observation(&observation);
        assert!(matches!(result.resolution(), AppResolution::Unknown { .. }));
        assert_eq!(result.matched_identifier_count(), 0);
        assert_eq!(result.unresolved_identifiers().len(), 1);
    }

    #[test]
    fn contradictory_strong_id_and_display_name_are_ambiguous() {
        let db = AppDatabase::new();
        let resolver = AppResolver::new(&db);
        let observation = ObservedApplicationIdentityV1::new(
            Some("Firefox".into()),
            vec![ObservedAppIdentifierV1::new(
                ApplicationIdentifierKind::WingetId,
                "Google.Chrome",
            )
            .unwrap()],
        )
        .unwrap();
        let result = resolver.resolve_observation(&observation);
        match result.resolution() {
            AppResolution::Ambiguous { candidates, .. } => {
                let names: BTreeSet<_> = candidates.iter().map(|entry| entry.name).collect();
                assert!(names.contains("Google Chrome"));
                assert!(names.contains("Firefox"));
            }
            _ => panic!("contradictory exact identity evidence must stay ambiguous"),
        }
    }

    #[test]
    fn unknown_typed_identifier_is_retained_beside_resolved_name() {
        let db = AppDatabase::new();
        let resolver = AppResolver::new(&db);
        let observation = ObservedApplicationIdentityV1::new(
            Some("Google Chrome".into()),
            vec![ObservedAppIdentifierV1::new(
                ApplicationIdentifierKind::WingetId,
                "Vendor.UnknownPackage",
            )
            .unwrap()],
        )
        .unwrap();
        let result = resolver.resolve_observation(&observation);
        assert_eq!(
            result.resolved_entry().expect("display name should still resolve").name,
            "Google Chrome"
        );
        assert_eq!(result.unresolved_identifiers().len(), 1);
    }
}
