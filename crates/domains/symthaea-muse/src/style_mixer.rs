// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic, provenance-bearing compiler for layered hybrid styles.
//!
//! The structural owner remains responsible for form, meter, motif banks and
//! long-range obligations. Other styles may donate only the named musical
//! layer. The compiler does not claim that a hybrid is a historical genre,
//! does not optimize for listener preference, and does not silently apply
//! renderer/production metadata that has no qualified adapter.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use symthaea_muse_protocol::catalog::{
    CanonicalStyle, HybridStyleSpec, StyleStatus, catalog_entry,
};
use symthaea_music_theory::{
    CompositionSpec, MusicalIntent, Score, Style, MUSIC_THEORY_ENGINE_VERSION,
    compose_with_grammar_plan,
};

pub const HYBRID_STYLE_COMPILER_SCHEMA: &str = "muse-hybrid-style-compiler-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StyleMixLayer {
    Structure,
    Melody,
    Rhythm,
    Harmony,
    Ensemble,
    Performance,
    Production,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StyleMixSource {
    pub layer: StyleMixLayer,
    pub catalog_id: u16,
    pub canonical_name: String,
    pub composer_style: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnappliedStyleLayer {
    pub layer: StyleMixLayer,
    pub catalog_id: u16,
    pub reason_code: String,
    pub declared_environment: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StyleMixProvenance {
    pub schema: String,
    pub compiler_version: String,
    pub source_recipe_sha256: String,
    pub display_label: String,
    pub sources: Vec<StyleMixSource>,
    pub unapplied_layers: Vec<UnappliedStyleLayer>,
    /// Always false: a composite recipe is a new work, not a historical
    /// tradition claim, even when all of its component sources are named.
    pub historical_genre_claim: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompiledHybridStyle {
    /// Retains the structural owner's dispatch family so dedicated engines
    /// (groove-cycle, additive process, modal arc, fugue, etc.) stay active.
    pub structural_style: Style,
    pub spec: CompositionSpec,
    pub provenance: StyleMixProvenance,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StyleMixIssue {
    pub code: String,
    pub layer: Option<StyleMixLayer>,
    pub catalog_id: Option<u16>,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HybridComposition {
    pub score: Score,
    pub compiled: CompiledHybridStyle,
}

/// Compile an explicit catalogue-layer recipe to the actual composer's
/// declarative spec. Every applied layer is copied from an executable
/// source style; unsupported meter transfers or unknown/unimplemented
/// composer routes fail closed.
pub fn compile_hybrid_style(
    hybrid: &HybridStyleSpec,
) -> Result<CompiledHybridStyle, Vec<StyleMixIssue>> {
    let mut issues = hybrid
        .validate()
        .into_iter()
        .map(|issue| StyleMixIssue {
            code: issue.code,
            layer: None,
            catalog_id: None,
            message: issue.message,
        })
        .collect::<Vec<_>>();

    let structural = resolve_executable_style(
        hybrid.structural_style_id,
        StyleMixLayer::Structure,
        &mut issues,
    );

    let melody = hybrid.melodic_style_id.and_then(|id| {
        resolve_executable_style(id, StyleMixLayer::Melody, &mut issues)
    });
    let rhythm = hybrid.rhythmic_style_id.and_then(|id| {
        resolve_executable_style(id, StyleMixLayer::Rhythm, &mut issues)
    });
    let harmony = hybrid.harmonic_style_id.and_then(|id| {
        resolve_executable_style(id, StyleMixLayer::Harmony, &mut issues)
    });
    let ensemble = hybrid.ensemble_style_id.and_then(|id| {
        resolve_executable_style(id, StyleMixLayer::Ensemble, &mut issues)
    });
    let performance = hybrid.performance_style_id.and_then(|id| {
        resolve_executable_style(id, StyleMixLayer::Performance, &mut issues)
    });

    let production = hybrid.production_style_id.and_then(|id| {
        match catalog_entry(id) {
            Some(entry) => Some(entry),
            None => {
                issues.push(unknown_style_issue(id, StyleMixLayer::Production));
                None
            }
        }
    });

    let Some((structural_entry, structural_style)) = structural else {
        return Err(issues);
    };

    let mut spec = structural_style.spec();

    // A rhythm donor can contribute its accompaniment/cycle engine only
    // when its bar length is compatible with the structural owner's score.
    // The owner retains meter and its exact motif-duration contract. A future
    // meter adapter may support cross-meter hybrids; this v1 compiler does not
    // silently stretch or truncate rhythmic cells.
    if let Some((entry, style)) = rhythm {
        let donor = style.spec();
        if donor.meter != spec.meter {
            issues.push(StyleMixIssue {
                code: "rhythm_meter_mismatch".into(),
                layer: Some(StyleMixLayer::Rhythm),
                catalog_id: Some(entry.id),
                message: format!(
                    "rhythm donor {} uses {}/4-style quarter-note beats per bar, but structural owner {} uses {}; an explicit meter/cycle adapter is required",
                    entry.display_name, donor.meter, structural_entry.display_name, spec.meter
                ),
            });
        } else {
            spec.accompaniment_pool = donor.accompaniment_pool;
            spec.texture.drums = donor.texture.drums;
        }
    }

    if let Some((_entry, style)) = melody {
        // Melodic DNA is independent from the structural owner's form and
        // section plan. Keep the owner's motif banks and meter obligations;
        // only replace the hook contour/rhythm/ornament vocabulary.
        spec.melody = style.spec().melody;
    }

    if let Some((_entry, style)) = harmony {
        let donor = style.spec();
        spec.progression = donor.progression;
        spec.mode = donor.mode;
        spec.mode_pool = donor.mode_pool;
        spec.texture.deceptive_close = donor.texture.deceptive_close;
        spec.texture.drone = donor.texture.drone;
        spec.texture.planing = donor.texture.planing;
        spec.texture.suspension_rate = donor.texture.suspension_rate;
        spec.texture.seventh_chords = donor.texture.seventh_chords;
        spec.texture.harmonic_sequence = donor.texture.harmonic_sequence;
        spec.texture.harmonic_stasis = donor.texture.harmonic_stasis;
        spec.texture.full_drone = donor.texture.full_drone;
    }

    if let Some((_entry, style)) = ensemble {
        let donor = style.spec();
        spec.ensemble_pool = donor.ensemble_pool;
        spec.counter_instrument = donor.counter_instrument;
    }

    if let Some((_entry, style)) = performance {
        let donor = style.spec();
        spec.tempo_range = donor.tempo_range;
        spec.rhetoric = donor.rhetoric;
        spec.texture.swing = donor.texture.swing;
        spec.texture.intro_bars = donor.texture.intro_bars;
        spec.texture.staged_entrances = donor.texture.staged_entrances;
        spec.texture.held_arrivals = donor.texture.held_arrivals;
        spec.texture.damage = donor.texture.damage;
        spec.texture.return_color = donor.texture.return_color;
    }

    spec.name = hybrid.honest_label();

    if let Err(validation) = spec.validate() {
        issues.extend(validation.into_iter().map(|message| StyleMixIssue {
            code: "compiled_spec_invalid".into(),
            layer: None,
            catalog_id: None,
            message,
        }));
    }

    let mut sources = vec![source_record(structural_entry, StyleMixLayer::Structure)];
    if let Some((entry, _)) = melody {
        sources.push(source_record(entry, StyleMixLayer::Melody));
    }
    if let Some((entry, _)) = rhythm {
        sources.push(source_record(entry, StyleMixLayer::Rhythm));
    }
    if let Some((entry, _)) = harmony {
        sources.push(source_record(entry, StyleMixLayer::Harmony));
    }
    if let Some((entry, _)) = ensemble {
        sources.push(source_record(entry, StyleMixLayer::Ensemble));
    }
    if let Some((entry, _)) = performance {
        sources.push(source_record(entry, StyleMixLayer::Performance));
    }

    let mut unapplied_layers = Vec::new();
    if let Some(entry) = production {
        // CompositionSpec has no production/renderer field. Preserve the
        // declared environmental source as provenance, but never claim it
        // changed the sound. A qualified renderer adapter is a later stage.
        unapplied_layers.push(UnappliedStyleLayer {
            layer: StyleMixLayer::Production,
            catalog_id: entry.id,
            reason_code: "renderer_adapter_not_implemented".into(),
            declared_environment: entry.anatomy().production_environment.into(),
        });
        sources.push(source_record(entry, StyleMixLayer::Production));
    }

    if !issues.is_empty() {
        return Err(issues);
    }

    let recipe_bytes = serde_json::to_vec(hybrid).map_err(|error| {
        vec![StyleMixIssue {
            code: "recipe_serialization_failed".into(),
            layer: None,
            catalog_id: None,
            message: error.to_string(),
        }]
    })?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea:muse-hybrid-style-recipe:v1\n");
    hasher.update(recipe_bytes);
    let source_recipe_sha256 = format!("{:x}", hasher.finalize());

    Ok(CompiledHybridStyle {
        structural_style,
        spec,
        provenance: StyleMixProvenance {
            schema: HYBRID_STYLE_COMPILER_SCHEMA.into(),
            compiler_version: MUSIC_THEORY_ENGINE_VERSION.into(),
            source_recipe_sha256,
            display_label: hybrid.honest_label(),
            sources,
            unapplied_layers,
            historical_genre_claim: false,
        },
    })
}

/// Compose a symbolic Score from a compiled hybrid. Rendering remains a
/// separate, renderer-qualified operation in Muse.
pub fn compose_hybrid_style(
    intent: &MusicalIntent,
    hybrid: &HybridStyleSpec,
) -> Result<HybridComposition, Vec<StyleMixIssue>> {
    let compiled = compile_hybrid_style(hybrid)?;
    // Keep the structural owner's grammar-family dispatch. In particular,
    // dedicated engines must not be bypassed merely because the recipe is a
    // hybrid CompositionSpec.
    let score = compose_with_grammar_plan(
        compiled.structural_style.grammar_profile(),
        intent,
        &compiled.spec,
    )
    .score;
    Ok(HybridComposition { score, compiled })
}

fn resolve_executable_style(
    id: u16,
    layer: StyleMixLayer,
    issues: &mut Vec<StyleMixIssue>,
) -> Option<(&'static CanonicalStyle, Style)> {
    let Some(entry) = catalog_entry(id) else {
        issues.push(unknown_style_issue(id, layer));
        return None;
    };
    let Some(name) = entry.composer_style else {
        issues.push(StyleMixIssue {
            code: "source_style_not_implemented".into(),
            layer: Some(layer),
            catalog_id: Some(id),
            message: format!(
                "{} has no executable composer route; research catalogue entries cannot donate executable grammar",
                entry.display_name
            ),
        });
        return None;
    };
    let Some(style) = Style::ALL
        .into_iter()
        .find(|style| format!("{style:?}") == name)
    else {
        issues.push(StyleMixIssue {
            code: "composer_route_unresolved".into(),
            layer: Some(layer),
            catalog_id: Some(id),
            message: format!(
                "catalogue composer route {name:?} for {} does not resolve to Style::ALL",
                entry.display_name
            ),
        });
        return None;
    };
    if entry.status == StyleStatus::Research || !entry.is_composable() {
        issues.push(StyleMixIssue {
            code: "source_style_not_composable".into(),
            layer: Some(layer),
            catalog_id: Some(id),
            message: format!(
                "{} is not in a composable catalogue state",
                entry.display_name
            ),
        });
        return None;
    }
    Some((entry, style))
}

fn unknown_style_issue(id: u16, layer: StyleMixLayer) -> StyleMixIssue {
    StyleMixIssue {
        code: "unknown_catalogue_style".into(),
        layer: Some(layer),
        catalog_id: Some(id),
        message: format!("catalogue style id {id} does not exist"),
    }
}

fn source_record(entry: &CanonicalStyle, layer: StyleMixLayer) -> StyleMixSource {
    StyleMixSource {
        layer,
        catalog_id: entry.id,
        canonical_name: entry.display_name.into(),
        composer_style: entry.composer_style.map(str::to_string),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_muse_protocol::catalog::catalog_entry;

    fn id_for_composer_style(name: &str) -> u16 {
        (1..=152)
            .find(|id| catalog_entry(*id).is_some_and(|entry| entry.composer_style == Some(name)))
            .unwrap_or_else(|| panic!("missing composer style {name}"))
    }

    fn first_unimplemented_id() -> u16 {
        (1..=152)
            .find(|id| catalog_entry(*id).is_some_and(|entry| entry.composer_style.is_none()))
            .expect("catalogue should keep research-only entries")
    }

    #[test]
    fn compiles_independent_layers_into_a_real_composition_spec() {
        let hybrid = HybridStyleSpec {
            structural_style_id: id_for_composer_style("Classical"),
            melodic_style_id: Some(id_for_composer_style("Blues")),
            rhythmic_style_id: Some(id_for_composer_style("Tango")),
            harmonic_style_id: Some(id_for_composer_style("Blues")),
            ensemble_style_id: Some(id_for_composer_style("Folk")),
            performance_style_id: Some(id_for_composer_style("Cinematic")),
            production_style_id: Some(id_for_composer_style("Ambient")),
            culturally_qualified_sources_acknowledged: false,
        };

        let compiled = compile_hybrid_style(&hybrid).expect("compatible four-beat hybrid");
        let structural = Style::Classical.spec();
        let melody = Style::Blues.spec();
        let rhythm = Style::Tango.spec();
        let harmony = Style::Blues.spec();
        let ensemble = Style::Folk.spec();
        let performance = Style::Cinematic.spec();

        assert_eq!(compiled.spec.meter, structural.meter);
        assert_eq!(compiled.spec.form_pool, structural.form_pool);
        assert_eq!(compiled.spec.motifs_calm, structural.motifs_calm);
        assert_eq!(compiled.spec.melody, melody.melody);
        assert_eq!(compiled.spec.accompaniment_pool, rhythm.accompaniment_pool);
        assert_eq!(compiled.spec.progression, harmony.progression);
        assert_eq!(compiled.spec.mode, harmony.mode);
        assert_eq!(compiled.spec.ensemble_pool, ensemble.ensemble_pool);
        assert_eq!(compiled.spec.tempo_range, performance.tempo_range);
        assert!(compiled.spec.validate().is_ok());
        assert!(!compiled.provenance.historical_genre_claim);
        assert_eq!(compiled.provenance.sources.len(), 7);
        assert_eq!(compiled.provenance.unapplied_layers.len(), 1);
        assert_eq!(
            compiled.provenance.unapplied_layers[0].reason_code,
            "renderer_adapter_not_implemented"
        );
        assert_eq!(compiled.provenance.source_recipe_sha256.len(), 64);
        assert!(compiled.provenance.display_label.starts_with("Hybrid — "));
    }

    #[test]
    fn incompatible_rhythm_meter_is_rejected_instead_of_rescaled() {
        let hybrid = HybridStyleSpec {
            structural_style_id: id_for_composer_style("Classical"),
            melodic_style_id: None,
            rhythmic_style_id: Some(id_for_composer_style("Waltz")),
            harmonic_style_id: None,
            ensemble_style_id: None,
            performance_style_id: None,
            production_style_id: None,
            culturally_qualified_sources_acknowledged: false,
        };
        let issues = compile_hybrid_style(&hybrid).unwrap_err();
        assert!(issues.iter().any(|issue| issue.code == "rhythm_meter_mismatch"));
    }

    #[test]
    fn research_only_style_cannot_be_used_as_a_generator_parent() {
        let hybrid = HybridStyleSpec {
            structural_style_id: first_unimplemented_id(),
            melodic_style_id: None,
            rhythmic_style_id: None,
            harmonic_style_id: None,
            ensemble_style_id: None,
            performance_style_id: None,
            production_style_id: None,
            culturally_qualified_sources_acknowledged: false,
        };
        let issues = compile_hybrid_style(&hybrid).unwrap_err();
        assert!(issues.iter().any(|issue| {
            issue.code == "source_style_not_implemented"
                || issue.code == "structural_owner_not_implemented"
        }));
    }

    #[test]
    fn hybrid_composition_uses_compiled_spec_without_preference_ranking() {
        let hybrid = HybridStyleSpec {
            structural_style_id: id_for_composer_style("Classical"),
            melodic_style_id: Some(id_for_composer_style("Blues")),
            rhythmic_style_id: Some(id_for_composer_style("Tango")),
            harmonic_style_id: None,
            ensemble_style_id: Some(id_for_composer_style("Folk")),
            performance_style_id: None,
            production_style_id: None,
            culturally_qualified_sources_acknowledged: false,
        };
        let intent = MusicalIntent::default();
        let output = compose_hybrid_style(&intent, &hybrid).expect("valid hybrid composition");
        assert!(!output.score.notes.is_empty());
        assert_eq!(output.compiled.spec.name, hybrid.honest_label());
        assert!(!output.compiled.provenance.historical_genre_claim);
    }

    #[test]
    fn source_recipe_hash_and_compiled_spec_are_deterministic() {
        let hybrid = HybridStyleSpec {
            structural_style_id: id_for_composer_style("Classical"),
            melodic_style_id: Some(id_for_composer_style("Blues")),
            rhythmic_style_id: Some(id_for_composer_style("Tango")),
            harmonic_style_id: None,
            ensemble_style_id: None,
            performance_style_id: None,
            production_style_id: None,
            culturally_qualified_sources_acknowledged: false,
        };
        let left = compile_hybrid_style(&hybrid).unwrap();
        let right = compile_hybrid_style(&hybrid).unwrap();
        assert_eq!(left.spec, right.spec);
        assert_eq!(
            left.provenance.source_recipe_sha256,
            right.provenance.source_recipe_sha256
        );
    }
}
