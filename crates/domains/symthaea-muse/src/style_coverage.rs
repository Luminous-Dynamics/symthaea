// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic source-generated coverage matrix for Muse's canonical styles.
//!
//! This report distinguishes catalogue hypotheses, registered composer routes,
//! fields declared in a CompositionSpec, and instrument names that resolve.
//! None of those lower evidence levels is promoted into sample, renderer,
//! cultural-authenticity, or listener-qualification claims.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

use symthaea_muse_protocol::catalog::{
    CanonicalStyle, Constellation, catalog_for_constellation,
};
use symthaea_music_theory::{
    MUSIC_THEORY_ENGINE_VERSION, ProgressionSpec, Style, TextureSpec,
    spec::MelodicDna,
};

use crate::instruments::Instrument;

pub const STYLE_COVERAGE_SCHEMA: &str = "muse-style-coverage-matrix-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StyleCoverageAxis {
    CorpusAndProvenance,
    Structure,
    Melody,
    Rhythm,
    HarmonyAndTuning,
    TextureAndCounterpoint,
    InstrumentationAndTechnique,
    Performance,
    Production,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StyleCoverageState {
    /// No source-backed capability or hypothesis was found for this axis.
    Missing,
    /// Only a constellation-level research description exists.
    FamilyHypothesisOnly,
    /// The style declares a field in its CompositionSpec. This is not proof
    /// that the field is idiomatic, source-validated or perceptually useful.
    DeclaredInSpec,
    /// The catalogue route resolves to one member of Style::ALL.
    RegisteredComposerRoute,
    /// An ensemble instrument name resolves, but range, sample and rendering
    /// capability have not thereby been qualified.
    NameResolvesOnly,
    /// A declared instrument name does not resolve through Instrument::from_name.
    UnresolvedInstrumentName,
    /// The axis currently relies on a shared engine default rather than
    /// a style-specific CompositionSpec value.
    SharedEngineDefault,
    /// A source-specific score/audio corpus has been checked and pinned.
    SourceVerified,
    /// Declared sample assets have been decoded and audited.
    AssetDecoded,
    /// A specific pitch/velocity/technique/renderer cell has been rendered
    /// and mechanically verified.
    RenderCellVerified,
    /// Independent perceptual or knowledgeable-practitioner qualification
    /// exists for the exact scoped claim.
    PerceptuallyOrExpertQualified,
    /// A source/corpus lineage and scope has been independently verified.
    /// Not emitted by the current collector until a source registry is bound.
    SourceVerified,
    /// A specific sample asset has been decoded and its content digest checked.
    /// Not emitted by this collector; sample evidence belongs to #7464/#7554.
    AssetDecoded,
    /// An exact instrument × articulation × pitch × velocity × renderer cell
    /// has passed a render check. Not emitted until render receipts are joined.
    RenderCellVerified,
    /// Perceptual identity or specialist review has qualified the stated scope.
    /// Not emitted by code declaration or preference score alone.
    PerceptuallyOrExpertQualified,
    /// No qualified per-style renderer / production recipe is present here.
    NotRendererQualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StyleAxisCoverage {
    pub axis: StyleCoverageAxis,
    pub state: StyleCoverageState,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StyleAnatomySnapshot {
    pub grammar_focus: String,
    pub phrase_behavior: String,
    pub harmonic_system: String,
    pub rhythm: String,
    pub melodic_language: String,
    pub ensemble: String,
    pub performance_dialect: String,
    pub production_environment: String,
    /// These values are family-level research gaps from the canonical
    /// catalogue, not a style-specific source bibliography.
    pub research_gaps: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstrumentNameResolution {
    pub declared_name: String,
    pub resolved: bool,
    pub canonical_variant: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StyleRouteCoverage {
    pub composer_style: String,
    pub grammar_family: String,
    pub specification_valid: bool,
    pub specification_errors: Vec<String>,
    pub meter_quarter_beats_per_bar: u8,
    pub tempo_range_bpm: [f32; 2],
    pub form_pool: Vec<String>,
    pub motif_bank_counts: [usize; 3],
    pub melodic_dna_style_specific: bool,
    pub progression_spec: ProgressionSpec,
    pub accompaniment_pool: Vec<String>,
    pub melodic_dna: MelodicDna,
    pub meter_pool: Vec<u8>,
    pub mode: Option<String>,
    pub mode_pool: Vec<Option<String>>,
    pub ensemble_pool: Vec<[String; 3]>,
    pub counter_instrument: Option<String>,
    pub ensemble_instrument_resolution: Vec<InstrumentNameResolution>,
    pub phrase_rhetoric: String,
    pub development_dna: String,
    pub texture: TextureSpec,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CanonicalStyleCoverage {
    pub catalog_id: u16,
    pub canonical_name: String,
    pub display_name: String,
    pub constellation: String,
    pub catalog_status: String,
    pub requires_expert_review: bool,
    pub composer_route_declared: Option<String>,
    pub anatomy: StyleAnatomySnapshot,
    pub route: Option<StyleRouteCoverage>,
    pub axes: Vec<StyleAxisCoverage>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StyleCoverageStateCount {
    pub axis: StyleCoverageAxis,
    pub state: StyleCoverageState,
    pub count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConstellationCoverageCount {
    pub constellation: String,
    pub style_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogStatusCoverageCount {
    pub status: String,
    pub style_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StyleCoverageSummary {
    pub catalog_entry_count: usize,
    pub constellation_count: usize,
    pub styles_by_constellation: Vec<ConstellationCoverageCount>,
    pub styles_by_catalog_status: Vec<CatalogStatusCoverageCount>,
    pub catalog_routes_declared: usize,
    pub composer_routes_resolved: usize,
    pub catalog_entries_without_routes: usize,
    pub unresolved_composer_routes: usize,
    pub invalid_composition_specs: usize,
    pub declared_ensemble_names: usize,
    pub resolved_ensemble_names: usize,
    pub unresolved_ensemble_names: usize,
    pub axes_by_state: Vec<StyleCoverageStateCount>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StyleCoverageReport {
    pub schema: String,
    pub music_theory_engine_version: String,
    pub catalogue_contract: String,
    pub style_registry_count: usize,
    pub summary: StyleCoverageSummary,
    pub styles: Vec<CanonicalStyleCoverage>,
    /// Digest of the report with this field empty. It binds the exact
    /// generated matrix bytes without relying on timestamps or human taste.
    pub report_sha256: String,
}

pub fn collect_style_coverage() -> StyleCoverageReport {
    let mut entries: Vec<&'static CanonicalStyle> = Constellation::ALL
        .into_iter()
        .flat_map(catalog_for_constellation)
        .collect();
    entries.sort_by_key(|entry| entry.id);

    let mut registry = BTreeMap::<String, Style>::new();
    for style in Style::ALL {
        registry.insert(format!("{style:?}"), style);
    }

    let mut styles = Vec::with_capacity(entries.len());
    let mut catalog_routes_declared = 0usize;
    let mut composer_routes_resolved = 0usize;
    let mut unresolved_composer_routes = 0usize;
    let mut invalid_composition_specs = 0usize;
    let mut declared_ensemble_names = 0usize;
    let mut resolved_ensemble_names = 0usize;
    let mut unresolved_ensemble_names = 0usize;

    for entry in entries {
        let anatomy = entry.anatomy();
        let route = entry.composer_style.map(|route_name| {
            catalog_routes_declared += 1;
            match registry.get(route_name).copied() {
                Some(style) => {
                    composer_routes_resolved += 1;
                    let spec = style.spec();
                    let validation = spec.validate();
                    let specification_errors = validation.err().unwrap_or_default();
                    let specification_valid = specification_errors.is_empty();
                    if !specification_valid {
                        invalid_composition_specs += 1;
                    }

                    let mut ensemble_instrument_resolution = Vec::new();
                    for ensemble in &spec.ensemble_pool {
                        for declared_name in ensemble {
                            declared_ensemble_names += 1;
                            let resolved = Instrument::from_name(declared_name);
                            if resolved.is_some() {
                                resolved_ensemble_names += 1;
                            } else {
                                unresolved_ensemble_names += 1;
                            }
                            ensemble_instrument_resolution.push(InstrumentNameResolution {
                                declared_name: declared_name.clone(),
                                resolved: resolved.is_some(),
                                canonical_variant: resolved.map(|instrument| format!("{instrument:?}")),
                            });
                        }
                    }
                    if let Some(counter) = &spec.counter_instrument {
                        declared_ensemble_names += 1;
                        let resolved = Instrument::from_name(counter);
                        if resolved.is_some() {
                            resolved_ensemble_names += 1;
                        } else {
                            unresolved_ensemble_names += 1;
                        }
                        ensemble_instrument_resolution.push(InstrumentNameResolution {
                            declared_name: counter.clone(),
                            resolved: resolved.is_some(),
                            canonical_variant: resolved.map(|instrument| format!("{instrument:?}")),
                        });
                    }

                    StyleRouteCoverage {
                        composer_style: route_name.into(),
                        grammar_family: format!("{:?}", style.grammar_family()),
                        specification_valid,
                        specification_errors,
                        meter_quarter_beats_per_bar: spec.meter,
                        tempo_range_bpm: [spec.tempo_range.0, spec.tempo_range.1],
                        form_pool: spec.form_pool.iter().map(|form| format!("{form:?}")).collect(),
                        motif_bank_counts: [
                            spec.motifs_calm.len(),
                            spec.motifs_medium.len(),
                            spec.motifs_busy.len(),
                        ],
                        melodic_dna_style_specific: !spec.melody.hook_rhythms.is_empty()
                            || !spec.melody.hook_contours.is_empty()
                            || spec.melody.appoggiatura_rate > 0.0
                            || spec.melody.ornament_rate > 0.0
                            || spec.melody.blue_note_rate > 0.0
                            || spec.melody.use_procedural_foundry,
                        progression_spec: spec.progression.clone(),
                        accompaniment_pool: spec
                            .accompaniment_pool
                            .iter()
                            .map(|item| format!("{item:?}"))
                            .collect(),
                        melodic_dna: spec.melody.clone(),
                        meter_pool: spec.meter_pool.clone(),
                        mode: spec.mode.map(|mode| format!("{mode:?}")),
                        mode_pool: spec
                            .mode_pool
                            .iter()
                            .map(|mode| mode.as_ref().map(|value| format!("{value:?}")))
                            .collect(),
                        ensemble_pool: spec.ensemble_pool.clone(),
                        counter_instrument: spec.counter_instrument.clone(),
                        ensemble_instrument_resolution,
                        phrase_rhetoric: format!("{:?}", spec.rhetoric),
                        development_dna: format!("{:?}", spec.development),
                        texture: spec.texture.clone(),
                    }
                }
                None => {
                    unresolved_composer_routes += 1;
                    StyleRouteCoverage {
                        composer_style: route_name.into(),
                        grammar_family: "unresolved".into(),
                        specification_valid: false,
                        specification_errors: vec![
                            "catalogue composer_style does not resolve to Style::ALL".into(),
                        ],
                        meter_quarter_beats_per_bar: 0,
                        tempo_range_bpm: [0.0, 0.0],
                        form_pool: Vec::new(),
                        motif_bank_counts: [0, 0, 0],
                        melodic_dna_style_specific: false,
                        progression_spec: ProgressionSpec::Grammar,
                        accompaniment_pool: Vec::new(),
                        melodic_dna: MelodicDna::default(),
                        meter_pool: Vec::new(),
                        mode: None,
                        mode_pool: Vec::new(),
                        ensemble_pool: Vec::new(),
                        counter_instrument: None,
                        ensemble_instrument_resolution: Vec::new(),
                        phrase_rhetoric: "unresolved".into(),
                        development_dna: "unresolved".into(),
                        texture: Style::Classical.spec().texture,
                    }
                }
            }
        });

        let axes = build_axes(entry, route.as_ref());
        styles.push(CanonicalStyleCoverage {
            catalog_id: entry.id,
            canonical_name: entry.name.into(),
            display_name: entry.display_name.into(),
            constellation: entry.constellation.name().into(),
            catalog_status: entry.status.label().into(),
            requires_expert_review: entry.requires_expert_review,
            composer_route_declared: entry.composer_style.map(str::to_string),
            anatomy: StyleAnatomySnapshot {
                grammar_focus: anatomy.grammar.into(),
                phrase_behavior: anatomy.phrase_behavior.into(),
                harmonic_system: anatomy.harmonic_system.into(),
                rhythm: anatomy.rhythm.into(),
                melodic_language: anatomy.melodic_language.into(),
                ensemble: anatomy.ensemble.into(),
                performance_dialect: anatomy.performance_dialect.into(),
                production_environment: anatomy.production_environment.into(),
                research_gaps: entry.research_gaps().iter().map(|gap| (*gap).into()).collect(),
            },
            route,
            axes,
        });
    }

    let axes_by_state = summarize_axes(&styles);
    let mut constellation_counts = BTreeMap::<String, usize>::new();
    let mut status_counts = BTreeMap::<String, usize>::new();
    for style in &styles {
        *constellation_counts.entry(style.constellation.clone()).or_default() += 1;
        *status_counts.entry(style.catalog_status.clone()).or_default() += 1;
    }
    let styles_by_constellation = constellation_counts
        .into_iter()
        .map(|(constellation, style_count)| ConstellationCoverageCount {
            constellation,
            style_count,
        })
        .collect();
    let styles_by_catalog_status = status_counts
        .into_iter()
        .map(|(status, style_count)| CatalogStatusCoverageCount {
            status,
            style_count,
        })
        .collect();
    let report = StyleCoverageReport {
        schema: STYLE_COVERAGE_SCHEMA.into(),
        music_theory_engine_version: MUSIC_THEORY_ENGINE_VERSION.into(),
        catalogue_contract: "152 entries; 19 constellations; 8 entries per constellation".into(),
        style_registry_count: Style::ALL.len(),
        summary: StyleCoverageSummary {
            catalog_entry_count: styles.len(),
            constellation_count: Constellation::ALL.len(),
            styles_by_constellation,
            styles_by_catalog_status,
            catalog_routes_declared,
            composer_routes_resolved,
            catalog_entries_without_routes: styles.len().saturating_sub(catalog_routes_declared),
            unresolved_composer_routes,
            invalid_composition_specs,
            declared_ensemble_names,
            resolved_ensemble_names,
            unresolved_ensemble_names,
            axes_by_state,
        },
        styles,
        report_sha256: String::new(),
    };
    with_report_digest(report)
}

fn build_axes(
    entry: &CanonicalStyle,
    route: Option<&StyleRouteCoverage>,
) -> Vec<StyleAxisCoverage> {
    let anatomy = entry.anatomy();
    let Some(route) = route else {
        return vec![
            axis(StyleCoverageAxis::CorpusAndProvenance, StyleCoverageState::FamilyHypothesisOnly, "no style-specific source corpus is linked from the canonical catalogue"),
            axis(StyleCoverageAxis::Structure, StyleCoverageState::FamilyHypothesisOnly, anatomy.grammar),
            axis(StyleCoverageAxis::Melody, StyleCoverageState::FamilyHypothesisOnly, anatomy.melodic_language),
            axis(StyleCoverageAxis::Rhythm, StyleCoverageState::FamilyHypothesisOnly, anatomy.rhythm),
            axis(StyleCoverageAxis::HarmonyAndTuning, StyleCoverageState::FamilyHypothesisOnly, anatomy.harmonic_system),
            axis(StyleCoverageAxis::TextureAndCounterpoint, StyleCoverageState::FamilyHypothesisOnly, anatomy.phrase_behavior),
            axis(StyleCoverageAxis::InstrumentationAndTechnique, StyleCoverageState::FamilyHypothesisOnly, anatomy.ensemble),
            axis(StyleCoverageAxis::Performance, StyleCoverageState::FamilyHypothesisOnly, anatomy.performance_dialect),
            axis(StyleCoverageAxis::Production, StyleCoverageState::FamilyHypothesisOnly, anatomy.production_environment),
        ];
    };

    if route.grammar_family == "unresolved" {
        return vec![
            axis(StyleCoverageAxis::CorpusAndProvenance, StyleCoverageState::FamilyHypothesisOnly, "no style-specific source corpus is linked from the canonical catalogue"),
            axis(StyleCoverageAxis::Structure, StyleCoverageState::FamilyHypothesisOnly, "composer route is declared but does not resolve to Style::ALL"),
            axis(StyleCoverageAxis::Melody, StyleCoverageState::FamilyHypothesisOnly, anatomy.melodic_language),
            axis(StyleCoverageAxis::Rhythm, StyleCoverageState::FamilyHypothesisOnly, anatomy.rhythm),
            axis(StyleCoverageAxis::HarmonyAndTuning, StyleCoverageState::FamilyHypothesisOnly, anatomy.harmonic_system),
            axis(StyleCoverageAxis::TextureAndCounterpoint, StyleCoverageState::FamilyHypothesisOnly, anatomy.phrase_behavior),
            axis(StyleCoverageAxis::InstrumentationAndTechnique, StyleCoverageState::FamilyHypothesisOnly, anatomy.ensemble),
            axis(StyleCoverageAxis::Performance, StyleCoverageState::FamilyHypothesisOnly, anatomy.performance_dialect),
            axis(StyleCoverageAxis::Production, StyleCoverageState::FamilyHypothesisOnly, anatomy.production_environment),
        ];
    }

    let instrument_state = if !route.ensemble_instrument_resolution.is_empty()
        && route.ensemble_instrument_resolution.iter().all(|instrument| instrument.resolved)
    {
        StyleCoverageState::NameResolvesOnly
    } else if route.ensemble_instrument_resolution.iter().any(|instrument| !instrument.resolved) {
        StyleCoverageState::UnresolvedInstrumentName
    } else {
        StyleCoverageState::DeclaredInSpec
    };

    // Every route owns motif banks in its CompositionSpec. The separate
    // MelodicDna field may still be empty and use shared hook defaults; report
    // that distinction in detail rather than downgrading the entire melodic
    // axis to "missing".
    let melody_state = if route.motif_bank_counts.iter().all(|count| *count > 0) {
        StyleCoverageState::DeclaredInSpec
    } else {
        StyleCoverageState::SharedEngineDefault
    };
    let has_style_specific_harmony = !matches!(route.progression_spec, ProgressionSpec::Grammar)
        || route.mode.is_some()
        || !route.mode_pool.is_empty();
    let harmony_state = if has_style_specific_harmony {
        StyleCoverageState::DeclaredInSpec
    } else {
        StyleCoverageState::SharedEngineDefault
    };
    vec![
        axis(StyleCoverageAxis::CorpusAndProvenance, StyleCoverageState::FamilyHypothesisOnly, "source-specific corpus lineage is not recorded in the canonical style catalogue"),
        axis(StyleCoverageAxis::Structure, StyleCoverageState::RegisteredComposerRoute, &format!("{} / {}; form pool: {:?}", route.composer_style, route.grammar_family, route.form_pool)),
        axis(StyleCoverageAxis::Melody, melody_state, &format!("motif banks {:?}; melodic DNA style-specific={}; {} hook rhythms, {} hook contours; appoggiatura {:.3}, ornament {:.3}, blue-note {:.3}", route.motif_bank_counts, route.melodic_dna_style_specific, route.melodic_dna.hook_rhythms.len(), route.melodic_dna.hook_contours.len(), route.melodic_dna.appoggiatura_rate, route.melodic_dna.ornament_rate, route.melodic_dna.blue_note_rate)),
        axis(StyleCoverageAxis::Rhythm, StyleCoverageState::DeclaredInSpec, &format!("meter {} quarter-note beats; accompaniment {:?}; meter alternatives {:?}", route.meter_quarter_beats_per_bar, route.accompaniment_pool, route.meter_pool)),
        axis(StyleCoverageAxis::HarmonyAndTuning, harmony_state, &format!("progression {:?}; mode {:?}; mode pool {:?}", route.progression_spec, route.mode, route.mode_pool)),
        axis(StyleCoverageAxis::TextureAndCounterpoint, StyleCoverageState::DeclaredInSpec, &format!("rhetoric {}; development {}; texture fields recorded", route.phrase_rhetoric, route.development_dna)),
        axis(StyleCoverageAxis::InstrumentationAndTechnique, instrument_state, &format!("{} ensemble name cells; {} resolve; name resolution is not range/sample/technique qualification", route.ensemble_instrument_resolution.len(), route.ensemble_instrument_resolution.iter().filter(|cell| cell.resolved).count())),
        axis(StyleCoverageAxis::Performance, StyleCoverageState::DeclaredInSpec, &format!("tempo range {:?}; swing {}; intro bars {}; staged entrances {}; held arrivals {}", route.tempo_range_bpm, route.texture.swing, route.texture.intro_bars, route.texture.staged_entrances, route.texture.held_arrivals)),
        axis(StyleCoverageAxis::Production, StyleCoverageState::NotRendererQualified, "no per-style qualified renderer/patch/production receipt is represented in this style registry"),
    ]
}

fn axis(
    axis: StyleCoverageAxis,
    state: StyleCoverageState,
    detail: &str,
) -> StyleAxisCoverage {
    StyleAxisCoverage {
        axis,
        state,
        detail: detail.into(),
    }
}

fn summarize_axes(styles: &[CanonicalStyleCoverage]) -> Vec<StyleCoverageStateCount> {
    let mut counts = BTreeMap::<(StyleCoverageAxis, StyleCoverageState), usize>::new();
    for style in styles {
        for axis in &style.axes {
            *counts.entry((axis.axis, axis.state)).or_default() += 1;
        }
    }
    counts
        .into_iter()
        .map(|((axis, state), count)| StyleCoverageStateCount { axis, state, count })
        .collect()
}

fn with_report_digest(mut report: StyleCoverageReport) -> StyleCoverageReport {
    report.report_sha256.clear();
    let bytes = serde_json::to_vec(&report).expect("coverage report DTO must serialize");
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea:muse-style-coverage-report:v1\n");
    hasher.update(bytes);
    report.report_sha256 = format!("{:x}", hasher.finalize());
    report
}

impl StyleCoverageReport {
    pub fn verify_digest(&self) -> bool {
        let mut copy = self.clone();
        let expected = copy.report_sha256.clone();
        copy.report_sha256.clear();
        let bytes = match serde_json::to_vec(&copy) {
            Ok(bytes) => bytes,
            Err(_) => return false,
        };
        let mut hasher = Sha256::new();
        hasher.update(b"symthaea:muse-style-coverage-report:v1\n");
        hasher.update(bytes);
        format!("{:x}", hasher.finalize()) == expected
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_muse_protocol::catalog::implemented_catalog;

    #[test]
    fn report_accounts_for_all_152_catalogue_styles_and_29_routes() {
        let report = collect_style_coverage();
        assert_eq!(report.summary.catalog_entry_count, 152);
        assert_eq!(report.summary.constellation_count, 19);
        assert_eq!(report.summary.catalog_routes_declared, 29);
        assert_eq!(report.summary.composer_routes_resolved, 29);
        assert_eq!(report.summary.catalog_entries_without_routes, 123);
        assert_eq!(report.summary.unresolved_composer_routes, 0);
        assert_eq!(report.style_registry_count, 29);
        assert_eq!(implemented_catalog().count(), 29);
    }

    #[test]
    fn each_canonical_constellation_has_exactly_eight_styles() {
        let report = collect_style_coverage();
        for constellation in Constellation::ALL {
            assert_eq!(
                report.styles.iter()
                    .filter(|style| style.constellation == constellation.name())
                    .count(),
                8,
                "{}",
                constellation.name()
            );
        }
    }

    #[test]
    fn canonical_ids_are_unique_sorted_and_complete() {
        let report = collect_style_coverage();
        assert_eq!(
            report.styles.iter().map(|style| style.catalog_id).collect::<Vec<_>>(),
            (1_u16..=152).collect::<Vec<_>>()
        );
    }

    #[test]
    fn route_mappings_are_unique_and_spec_validation_is_accounted_for() {
        let report = collect_style_coverage();
        let mut names = std::collections::BTreeSet::new();
        let mut invalid = 0usize;
        for style in &report.styles {
            if let Some(route) = &style.route {
                assert!(names.insert(route.composer_style.clone()));
                if !route.specification_valid {
                    invalid += 1;
                    assert!(!route.specification_errors.is_empty());
                }
            }
        }
        assert_eq!(names.len(), 29);
        assert_eq!(invalid, report.summary.invalid_composition_specs);
    }

    #[test]
    fn unqualified_axes_are_never_promoted_from_declared_fields() {
        let report = collect_style_coverage();
        for style in &report.styles {
            let production = style.axes.iter().find(|axis| axis.axis == StyleCoverageAxis::Production).unwrap();
            if style.route.is_some() {
                assert_eq!(production.state, StyleCoverageState::NotRendererQualified);
            } else {
                assert_eq!(production.state, StyleCoverageState::FamilyHypothesisOnly);
            }
            let instruments = style.axes.iter().find(|axis| axis.axis == StyleCoverageAxis::InstrumentationAndTechnique).unwrap();
            if let Some(route) = &style.route {
                let expected = if route.grammar_family == "unresolved" {
                    StyleCoverageState::FamilyHypothesisOnly
                } else if route.ensemble_instrument_resolution.iter().any(|cell| !cell.resolved) {
                    StyleCoverageState::UnresolvedInstrumentName
                } else if route.ensemble_instrument_resolution.is_empty() {
                    StyleCoverageState::DeclaredInSpec
                } else {
                    StyleCoverageState::NameResolvesOnly
                };
                assert_eq!(instruments.state, expected);
            }
        }
    }

    #[test]
    fn advanced_qualification_states_are_not_inferred_from_source_declarations() {
        let report = collect_style_coverage();
        for style in &report.styles {
            assert!(
                !style.axes.iter().any(|axis| matches!(
                    axis.state,
                    StyleCoverageState::SourceVerified
                        | StyleCoverageState::AssetDecoded
                        | StyleCoverageState::RenderCellVerified
                        | StyleCoverageState::PerceptuallyOrExpertQualified
                )),
                "style {} was promoted without joined evidence",
                style.canonical_name
            );
        }
    }

    #[test]
    fn report_bytes_and_digest_are_deterministic() {
        let first = collect_style_coverage();
        let second = collect_style_coverage();
        assert_eq!(first, second);
        assert!(first.verify_digest());
        assert_eq!(
            serde_json::to_vec(&first).unwrap(),
            serde_json::to_vec(&second).unwrap()
        );
        assert_eq!(first.report_sha256.len(), 64);
    }
}
