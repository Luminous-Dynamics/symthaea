//! Deterministic composition of style and functional garment presets.
//!
//! Presets provide suggestions, never user hard constraints. Conflicting
//! values are returned to the caller for an explicit user decision; this
//! module never uses last-write-wins or sums competing preference weights.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum PresetRole {
    Style,
    Function,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PresetLayer {
    pub preset_id: String,
    pub revision: u64,
    pub sha256: String,
    pub role: PresetRole,
    pub style_tags: BTreeSet<String>,
    pub silhouette: Option<String>,
    pub component_defaults: BTreeMap<String, String>,
    pub objective_defaults: BTreeMap<String, u16>,
    /// A preset is not allowed to create a hard user requirement.
    pub hard_constraints: Vec<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PresetOverrides {
    pub style_tags: BTreeSet<String>,
    pub silhouette: Option<String>,
    pub components: BTreeMap<String, String>,
    pub objective_weights: BTreeMap<String, u16>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PresetConflictKind {
    EmptyStack,
    EmptyPresetId,
    ZeroRevision,
    InvalidDigest,
    DuplicatePreset,
    PresetContainsHardConstraint,
    EmptyStyleTag,
    EmptyFieldPath,
    EmptyFieldValue,
    ConflictingSilhouette,
    ConflictingComponentValue,
    ConflictingObjectiveWeight,
    ObjectiveWeightOutOfRange,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PresetConflict {
    pub kind: PresetConflictKind,
    pub field: String,
    pub source_ids: Vec<String>,
    pub detail: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourcedValue<T> {
    pub value: T,
    pub preset_sources: BTreeSet<String>,
    pub user_override: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PresetRevision {
    pub preset_id: String,
    pub revision: u64,
    pub sha256: String,
    pub role: PresetRole,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComposedPreset {
    pub source_revisions: Vec<PresetRevision>,
    pub style_tags: BTreeSet<String>,
    pub silhouette: Option<SourcedValue<String>>,
    pub components: BTreeMap<String, SourcedValue<String>>,
    pub objective_weights: BTreeMap<String, SourcedValue<u16>>,
}

pub fn compose_preset_stack(
    layers: &[PresetLayer],
    overrides: &PresetOverrides,
) -> Result<ComposedPreset, Vec<PresetConflict>> {
    let mut conflicts = Vec::new();
    if layers.is_empty() {
        conflicts.push(PresetConflict {
            kind: PresetConflictKind::EmptyStack,
            field: "layers".to_owned(),
            source_ids: Vec::new(),
            detail: "at least one preset layer is required".to_owned(),
        });
        return Err(conflicts);
    }

    let mut seen_ids = BTreeSet::new();
    let mut source_revisions = Vec::new();
    let mut style_tags = BTreeSet::new();
    let mut silhouette_sources: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut component_sources: BTreeMap<
        String,
        BTreeMap<String, BTreeSet<String>>,
    > = BTreeMap::new();
    let mut objective_sources: BTreeMap<
        String,
        BTreeMap<u16, BTreeSet<String>>,
    > = BTreeMap::new();

    for layer in layers {
        let id = layer.preset_id.as_str();
        if id.trim().is_empty() {
            conflicts.push(conflict(
                PresetConflictKind::EmptyPresetId,
                "preset_id",
                Vec::new(),
                "preset identifier must be non-empty",
            ));
            continue;
        }
        if !seen_ids.insert(id.to_owned()) {
            conflicts.push(conflict(
                PresetConflictKind::DuplicatePreset,
                "preset_id",
                vec![id.to_owned()],
                "the same preset cannot occur twice in a stack",
            ));
        }
        if layer.revision == 0 {
            conflicts.push(conflict(
                PresetConflictKind::ZeroRevision,
                "revision",
                vec![id.to_owned()],
                "preset revision must be greater than zero",
            ));
        }
        if !is_sha256_hex(&layer.sha256) {
            conflicts.push(conflict(
                PresetConflictKind::InvalidDigest,
                "sha256",
                vec![id.to_owned()],
                "expected exactly 64 hexadecimal characters",
            ));
        }
        if !layer.hard_constraints.is_empty() {
            conflicts.push(conflict(
                PresetConflictKind::PresetContainsHardConstraint,
                "hard_constraints",
                vec![id.to_owned()],
                "presets may suggest preferences but cannot impose user hard constraints",
            ));
        }

        source_revisions.push(PresetRevision {
            preset_id: id.to_owned(),
            revision: layer.revision,
            sha256: layer.sha256.clone(),
            role: layer.role,
        });

        for tag in &layer.style_tags {
            if tag.trim().is_empty() {
                conflicts.push(conflict(
                    PresetConflictKind::EmptyStyleTag,
                    "style_tags",
                    vec![id.to_owned()],
                    "style tags must be non-empty",
                ));
            } else {
                style_tags.insert(tag.clone());
            }
        }

        if let Some(silhouette) = &layer.silhouette {
            if silhouette.trim().is_empty() {
                conflicts.push(conflict(
                    PresetConflictKind::EmptyFieldValue,
                    "silhouette",
                    vec![id.to_owned()],
                    "silhouette suggestion must be non-empty",
                ));
            } else {
                silhouette_sources
                    .entry(silhouette.clone())
                    .or_default()
                    .insert(id.to_owned());
            }
        }

        for (field, value) in &layer.component_defaults {
            if field.trim().is_empty() {
                conflicts.push(conflict(
                    PresetConflictKind::EmptyFieldPath,
                    "component_defaults",
                    vec![id.to_owned()],
                    "component path must be non-empty",
                ));
            } else if value.trim().is_empty() {
                conflicts.push(conflict(
                    PresetConflictKind::EmptyFieldValue,
                    field,
                    vec![id.to_owned()],
                    "component suggestion must be non-empty",
                ));
            } else {
                component_sources
                    .entry(field.clone())
                    .or_default()
                    .entry(value.clone())
                    .or_default()
                    .insert(id.to_owned());
            }
        }

        for (metric, weight) in &layer.objective_defaults {
            if metric.trim().is_empty() {
                conflicts.push(conflict(
                    PresetConflictKind::EmptyFieldPath,
                    "objective_defaults",
                    vec![id.to_owned()],
                    "objective metric must be non-empty",
                ));
            } else if *weight > 100 {
                conflicts.push(conflict(
                    PresetConflictKind::ObjectiveWeightOutOfRange,
                    metric,
                    vec![id.to_owned()],
                    "preset objective weight must be between 0 and 100",
                ));
            } else {
                objective_sources
                    .entry(metric.clone())
                    .or_default()
                    .entry(*weight)
                    .or_default()
                    .insert(id.to_owned());
            }
        }
    }

    for tag in &overrides.style_tags {
        if tag.trim().is_empty() {
            conflicts.push(conflict(
                PresetConflictKind::EmptyStyleTag,
                "user_overrides.style_tags",
                Vec::new(),
                "user style tags must be non-empty",
            ));
        } else {
            style_tags.insert(tag.clone());
        }
    }

    if !conflicts.is_empty() {
        return Err(conflicts);
    }

    let silhouette = resolve_text(
        "silhouette",
        silhouette_sources,
        overrides.silhouette.as_ref(),
        PresetConflictKind::ConflictingSilhouette,
        &mut conflicts,
    );

    let mut components = BTreeMap::new();
    let mut all_component_fields: BTreeSet<String> =
        component_sources.keys().cloned().collect();
    all_component_fields.extend(overrides.components.keys().cloned());

    for field in all_component_fields {
        if field.trim().is_empty() {
            conflicts.push(conflict(
                PresetConflictKind::EmptyFieldPath,
                "user_overrides.components",
                Vec::new(),
                "component path must be non-empty",
            ));
            continue;
        }
        let values = component_sources.remove(&field).unwrap_or_default();
        if let Some(value) = resolve_text(
            &field,
            values,
            overrides.components.get(&field),
            PresetConflictKind::ConflictingComponentValue,
            &mut conflicts,
        ) {
            components.insert(field, value);
        }
    }

    let mut objective_weights = BTreeMap::new();
    let mut all_objectives: BTreeSet<String> =
        objective_sources.keys().cloned().collect();
    all_objectives.extend(overrides.objective_weights.keys().cloned());

    for metric in all_objectives {
        if metric.trim().is_empty() {
            conflicts.push(conflict(
                PresetConflictKind::EmptyFieldPath,
                "user_overrides.objective_weights",
                Vec::new(),
                "objective metric must be non-empty",
            ));
            continue;
        }
        let values = objective_sources.remove(&metric).unwrap_or_default();
        if let Some(weight) = overrides.objective_weights.get(&metric) {
            if *weight > 100 {
                conflicts.push(conflict(
                    PresetConflictKind::ObjectiveWeightOutOfRange,
                    &metric,
                    Vec::new(),
                    "user objective weight must be between 0 and 100",
                ));
                continue;
            }
            objective_weights.insert(
                metric.clone(),
                SourcedValue {
                    value: *weight,
                    preset_sources: all_sources_u16(&values),
                    user_override: true,
                },
            );
        } else if values.len() > 1 {
            conflicts.push(conflict(
                PresetConflictKind::ConflictingObjectiveWeight,
                &metric,
                all_sources_u16(&values),
                "competing objective weights require an explicit user choice; do not sum",
                                or normalize them",
            ));
        } else if let Some((weight, sources)) = values.into_iter().next() {
            objective_weights.insert(
                metric,
                SourcedValue {
                    value: weight,
                    preset_sources: sources,
                    user_override: false,
                },
            );
        }
    }

    if !conflicts.is_empty() {
        return Err(conflicts);
    }

    source_revisions.sort_by(|left, right| left.preset_id.cmp(&right.preset_id));
    Ok(ComposedPreset {
        source_revisions,
        style_tags,
        silhouette,
        components,
        objective_weights,
    })
}

fn resolve_text(
    field: &str,
    values: BTreeMap<String, BTreeSet<String>>,
    user_override: Option<&String>,
    conflict_kind: PresetConflictKind,
    conflicts: &mut Vec<PresetConflict>,
) -> Option<SourcedValue<String>> {
    let sources = all_sources_text(&values);
    if let Some(value) = user_override {
        if value.trim().is_empty() {
            conflicts.push(conflict(
                PresetConflictKind::EmptyFieldValue,
                field,
                sources,
                "user override must be non-empty",
            ));
            return None;
        }
        return Some(SourcedValue {
            value: value.clone(),
            preset_sources: values.values().flat_map(|v| v.iter().cloned()).collect(),
            user_override: true,
        });
    }

    if values.len() > 1 {
        conflicts.push(conflict(
            conflict_kind,
            field,
            sources,
            "conflicting preset values require an explicit user decision",
        ));
        return None;
    }

    values.into_iter().next().map(|(value, sources)| SourcedValue {
        value,
        preset_sources: sources,
        user_override: false,
    })
}

fn all_sources_text(values: &BTreeMap<String, BTreeSet<String>>) -> Vec<String> {
    values
        .values()
        .flat_map(|sources| sources.iter().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn all_sources_u16(values: &BTreeMap<u16, BTreeSet<String>>) -> Vec<String> {
    values
        .values()
        .flat_map(|sources| sources.iter().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn conflict(
    kind: PresetConflictKind,
    field: impl Into<String>,
    source_ids: Vec<String>,
    detail: impl Into<String>,
) -> PresetConflict {
    PresetConflict {
        kind,
        field: field.into(),
        source_ids,
        detail: detail.into(),
    }
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.chars().all(|character| character.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer(
        preset_id: &str,
        role: PresetRole,
        silhouette: Option<&str>,
        components: &[(&str, &str)],
        objectives: &[(&str, u16)],
    ) -> PresetLayer {
        PresetLayer {
            preset_id: preset_id.to_owned(),
            revision: 1,
            sha256: "a".repeat(64),
            role,
            style_tags: BTreeSet::from([preset_id.to_owned()]),
            silhouette: silhouette.map(str::to_owned),
            component_defaults: components
                .iter()
                .map(|(field, value)| ((*field).to_owned(), (*value).to_owned()))
                .collect(),
            objective_defaults: objectives
                .iter()
                .map(|(metric, weight)| ((*metric).to_owned(), *weight))
                .collect(),
            hard_constraints: Vec::new(),
        }
    }

    #[test]
    fn style_and_function_layers_compose_without_losing_provenance() {
        let style = layer(
            "classic-tailoring",
            PresetRole::Style,
            Some("structured"),
            &[("closure.type", "buttons")],
            &[("appearance_preference", 90)],
        );
        let function = layer(
            "adaptive-ease",
            PresetRole::Function,
            None,
            &[("closure.access", "one-hand")],
            &[("comfort_preference", 100)],
        );
        let result = compose_preset_stack(&[style, function], &PresetOverrides::default()).unwrap();
        assert_eq!(result.source_revisions.len(), 2);
        assert!(result.style_tags.contains("classic-tailoring"));
        assert!(result.style_tags.contains("adaptive-ease"));
        assert_eq!(result.components["closure.type"].value, "buttons");
        assert_eq!(
            result.components["closure.access"].value,
            "one-hand"
        );
        assert_eq!(
            result.objective_weights["comfort_preference"].value,
            100
        );
    }

    #[test]
    fn conflicting_component_values_require_user_override() {
        let a = layer(
            "streetwear",
            PresetRole::Style,
            None,
            &[("pocket.layout", "patch")],
            &[],
        );
        let b = layer(
            "tailoring",
            PresetRole::Style,
            None,
            &[("pocket.layout", "welt")],
            &[],
        );
        let error = compose_preset_stack(&[a.clone(), b.clone()], &PresetOverrides::default())
            .unwrap_err();
        assert!(error.iter().any(|item| {
            item.kind == PresetConflictKind::ConflictingComponentValue
        }));

        let overrides = PresetOverrides {
            components: BTreeMap::from([("pocket.layout".to_owned(), "patch".to_owned())]),
            ..PresetOverrides::default()
        };
        let result = compose_preset_stack(&[a, b], &overrides).unwrap();
        assert!(result.components["pocket.layout"].user_override);
    }

    #[test]
    fn competing_objective_weights_are_not_summed() {
        let a = layer(
            "everyday",
            PresetRole::Function,
            None,
            &[],
            &[("comfort_preference", 80)],
        );
        let b = layer(
            "heat-adaptive",
            PresetRole::Function,
            None,
            &[],
            &[("comfort_preference", 100)],
        );
        let error = compose_preset_stack(&[a.clone(), b.clone()], &PresetOverrides::default())
            .unwrap_err();
        assert!(error.iter().any(|item| {
            item.kind == PresetConflictKind::ConflictingObjectiveWeight
        }));

        let overrides = PresetOverrides {
            objective_weights: BTreeMap::from([("comfort_preference".to_owned(), 95)]),
            ..PresetOverrides::default()
        };
        let result = compose_preset_stack(&[a, b], &overrides).unwrap();
        assert_eq!(result.objective_weights["comfort_preference"].value, 95);
        assert!(result.objective_weights["comfort_preference"].user_override);
    }

    #[test]
    fn presets_cannot_inject_hard_constraints() {
        let mut a = layer("repair-first", PresetRole::Function, None, &[], &[]);
        a.hard_constraints.push("price < 1000".to_owned());
        let error = compose_preset_stack(&[a], &PresetOverrides::default()).unwrap_err();
        assert!(error.iter().any(|item| {
            item.kind == PresetConflictKind::PresetContainsHardConstraint
        }));
    }

    #[test]
    fn duplicate_preset_and_invalid_digest_are_rejected() {
        let mut a = layer("everyday-core", PresetRole::Function, None, &[], &[]);
        a.sha256 = "not-a-digest".to_owned();
        let error = compose_preset_stack(&[a.clone(), a], &PresetOverrides::default()).unwrap_err();
        assert!(error.iter().any(|item| item.kind == PresetConflictKind::DuplicatePreset));
        assert!(error.iter().any(|item| item.kind == PresetConflictKind::InvalidDigest));
    }

    #[test]
    fn preset_layer_with_zero_revision_is_rejected() {
        let mut a = layer("repair-first", PresetRole::Function, None, &[], &[]);
        a.revision = 0;
        let error = compose_preset_stack(&[a], &PresetOverrides::default()).unwrap_err();
        assert!(error.iter().any(|item| item.kind == PresetConflictKind::ZeroRevision));
    }

    #[test]
    fn conflicting_silhouettes_are_rejected() {
        let a = layer("oversized-streetwear", PresetRole::Style, Some("oversized"), &[], &[]);
        let b = layer("tailored-fit", PresetRole::Style, Some("tailored"), &[], &[]);
        let error = compose_preset_stack(&[a, b], &PresetOverrides::default()).unwrap_err();
        assert!(error.iter().any(|item| {
            item.kind == PresetConflictKind::ConflictingSilhouette
        }));
    }

    #[test]
    fn user_silhouette_override_resolves_conflict() {
        let a = layer("oversized-streetwear", PresetRole::Style, Some("oversized"), &[], &[]);
        let b = layer("tailored-fit", PresetRole::Style, Some("tailored"), &[], &[]);
        let overrides = PresetOverrides {
            silhouette: Some("relaxed-tailored".to_owned()),
            ..PresetOverrides::default()
        };
        let result = compose_preset_stack(&[a, b], &overrides).unwrap();
        assert_eq!(
            result.silhouette.unwrap().value,
            "relaxed-tailored"
        );
    }

    #[test]
    fn objective_weights_are_not_normalized() {
        let a = layer("style-a", PresetRole::Style, None, &[], &[("appearance_preference", 70)]);
        let b = layer("function-a", PresetRole::Function, None, &[], &[("comfort_preference", 90)]);
        let result = compose_preset_stack(&[a, b], &PresetOverrides::default()).unwrap();
        assert_eq!(result.objective_weights["appearance_preference"].value, 70);
        assert_eq!(result.objective_weights["comfort_preference"].value, 90);
    }

    #[test]
    fn empty_stack_fails_explicitly() {
        let error = compose_preset_stack(&[], &PresetOverrides::default()).unwrap_err();
        assert_eq!(error[0].kind, PresetConflictKind::EmptyStack);
    }
}
