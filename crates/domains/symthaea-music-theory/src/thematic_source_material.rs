// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact declaration-time material for independent FORM-002 identities.
//!
//! [`crate::ThematicIdentityGraphV1`] intentionally records genealogy rather
//! than concrete motif bytes. That separation is useful, but an ordered
//! development program needs a durable answer to a simpler provenance
//! question: *what exact symbolic material did this independent identity start
//! from?*
//!
//! V1 binds every independent identity to one exact scale-degree [`Motif`]. It
//! deliberately refuses bindings for derived/synthesis identities. Those must
//! remain products of declared transformations and later realization/evidence;
//! accepting them here would turn a source-material registry into a way to
//! bypass genealogy.

use crate::motif::Motif;
use crate::thematic_identity::{
    ThematicGraphErrorV1, ThematicIdentityGraphV1, ThematicOriginV1,
};
use crate::work_plan::HierarchicalWorkPlanV1;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const THEMATIC_SOURCE_MATERIAL_PLAN_VERSION: &str =
    "melothaea-thematic-source-material-plan-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThematicSourceMaterialV1 {
    /// Exact symbolic source material. Scale degree and rhythm are declaration
    /// authority; this is not score-side evidence that the listener later hears
    /// the identity.
    pub motif: Motif,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThematicSourceMaterialPlanV1 {
    pub version: String,
    /// Identity ID -> exact authored source material. BTreeMap preserves one
    /// deterministic identity order in serialized/canonical consumers.
    pub sources: BTreeMap<String, ThematicSourceMaterialV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThematicSourceMaterialErrorV1 {
    WrongVersion { found: String },
    InvalidThematicGraph(ThematicGraphErrorV1),
    EmptyIdentityId,
    UnknownIdentity { identity_id: String },
    NonIndependentIdentity {
        identity_id: String,
        origin: ThematicOriginV1,
    },
    MissingIndependentIdentity { identity_id: String },
    EmptyMotif { identity_id: String },
    RestOnlyMotif { identity_id: String },
    InvalidDuration {
        identity_id: String,
        event_index: usize,
    },
}

impl Default for ThematicSourceMaterialPlanV1 {
    fn default() -> Self {
        Self {
            version: THEMATIC_SOURCE_MATERIAL_PLAN_VERSION.into(),
            sources: BTreeMap::new(),
        }
    }
}

impl ThematicSourceMaterialPlanV1 {
    pub fn insert(
        &mut self,
        identity_id: impl Into<String>,
        motif: Motif,
    ) -> Result<(), ThematicSourceMaterialErrorV1> {
        let identity_id = identity_id.into();
        if identity_id.trim().is_empty() {
            return Err(ThematicSourceMaterialErrorV1::EmptyIdentityId);
        }
        self.sources
            .insert(identity_id, ThematicSourceMaterialV1 { motif });
        Ok(())
    }

    /// Fail-closed validation against the already-valid work/genealogy layer.
    ///
    /// Completeness is intentional: every `Independent` FORM-002 identity must
    /// have exact source material, while no derived/synthesis identity may be
    /// represented as an authored source.
    pub fn validate(
        &self,
        work_plan: &HierarchicalWorkPlanV1,
        thematic_graph: &ThematicIdentityGraphV1,
    ) -> Result<(), ThematicSourceMaterialErrorV1> {
        if self.version != THEMATIC_SOURCE_MATERIAL_PLAN_VERSION {
            return Err(ThematicSourceMaterialErrorV1::WrongVersion {
                found: self.version.clone(),
            });
        }
        thematic_graph
            .validate(work_plan)
            .map_err(ThematicSourceMaterialErrorV1::InvalidThematicGraph)?;

        for (identity_id, source) in &self.sources {
            if identity_id.trim().is_empty() {
                return Err(ThematicSourceMaterialErrorV1::EmptyIdentityId);
            }
            let identity = thematic_graph.identities.get(identity_id).ok_or_else(|| {
                ThematicSourceMaterialErrorV1::UnknownIdentity {
                    identity_id: identity_id.clone(),
                }
            })?;
            if identity.origin != ThematicOriginV1::Independent {
                return Err(ThematicSourceMaterialErrorV1::NonIndependentIdentity {
                    identity_id: identity_id.clone(),
                    origin: identity.origin,
                });
            }
            validate_motif(identity_id, &source.motif)?;
        }

        let bound: BTreeSet<&str> = self.sources.keys().map(String::as_str).collect();
        for (identity_id, identity) in &thematic_graph.identities {
            if identity.origin == ThematicOriginV1::Independent
                && !bound.contains(identity_id.as_str())
            {
                return Err(ThematicSourceMaterialErrorV1::MissingIndependentIdentity {
                    identity_id: identity_id.clone(),
                });
            }
        }
        Ok(())
    }

    pub fn source(
        &self,
        identity_id: &str,
    ) -> Option<&ThematicSourceMaterialV1> {
        self.sources.get(identity_id)
    }
}

fn validate_motif(
    identity_id: &str,
    motif: &Motif,
) -> Result<(), ThematicSourceMaterialErrorV1> {
    if motif.notes.is_empty() {
        return Err(ThematicSourceMaterialErrorV1::EmptyMotif {
            identity_id: identity_id.into(),
        });
    }
    if !motif.notes.iter().any(|note| note.degree.is_some()) {
        return Err(ThematicSourceMaterialErrorV1::RestOnlyMotif {
            identity_id: identity_id.into(),
        });
    }
    for (event_index, note) in motif.notes.iter().enumerate() {
        if note.duration.num() <= 0 || note.duration.den() <= 0 {
            return Err(ThematicSourceMaterialErrorV1::InvalidDuration {
                identity_id: identity_id.into(),
                event_index,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Duration, FormalFunctionV1, MotifNote, ThematicDerivationV1,
        ThematicIdentityV1, ThematicTransformationClassV1, WorkNodeKindV1,
        WorkNodeV1,
    };

    fn work_plan() -> HierarchicalWorkPlanV1 {
        let mut work = HierarchicalWorkPlanV1::new("work", Duration::new(8, 1)).unwrap();
        work.insert_node(
            "A",
            WorkNodeV1 {
                parent_id: Some("work".into()),
                label: Some("A".into()),
                kind: WorkNodeKindV1::Section,
                start: Duration::zero(),
                end: Duration::new(4, 1),
                functions: vec![FormalFunctionV1::Establish],
            },
        )
        .unwrap();
        work.insert_node(
            "B",
            WorkNodeV1 {
                parent_id: Some("work".into()),
                label: Some("B".into()),
                kind: WorkNodeKindV1::Section,
                start: Duration::new(4, 1),
                end: Duration::new(8, 1),
                functions: vec![FormalFunctionV1::Develop],
            },
        )
        .unwrap();
        work
    }

    fn graph() -> ThematicIdentityGraphV1 {
        let work = work_plan();
        let mut graph = ThematicIdentityGraphV1::default();
        graph
            .insert_identity(
                "P",
                ThematicIdentityV1 {
                    label: Some("Primary".into()),
                    origin: ThematicOriginV1::Independent,
                    introduced_in: "A".into(),
                },
            )
            .unwrap();
        graph
            .insert_identity(
                "P-dev",
                ThematicIdentityV1 {
                    label: Some("Developed P".into()),
                    origin: ThematicOriginV1::Derived,
                    introduced_in: "B".into(),
                },
            )
            .unwrap();
        graph
            .insert_derivation(
                "derive-P",
                ThematicDerivationV1 {
                    source_id: "P".into(),
                    target_id: "P-dev".into(),
                    transformations: vec![ThematicTransformationClassV1::Inversion],
                },
            )
            .unwrap();
        graph.validate(&work).unwrap();
        graph
    }

    fn motif() -> Motif {
        Motif::from_degrees(&[
            (1, Duration::quarter()),
            (2, Duration::quarter()),
            (5, Duration::new(2, 1)),
        ])
    }

    #[test]
    fn every_independent_identity_can_be_bound_to_exact_material() {
        let work = work_plan();
        let graph = graph();
        let mut sources = ThematicSourceMaterialPlanV1::default();
        sources.insert("P", motif()).unwrap();
        sources.validate(&work, &graph).unwrap();
        assert_eq!(sources.source("P").unwrap().motif, motif());
    }

    #[test]
    fn missing_independent_identity_fails_completeness() {
        let error = ThematicSourceMaterialPlanV1::default()
            .validate(&work_plan(), &graph())
            .unwrap_err();
        assert_eq!(
            error,
            ThematicSourceMaterialErrorV1::MissingIndependentIdentity {
                identity_id: "P".into()
            }
        );
    }

    #[test]
    fn derived_identity_cannot_be_smuggled_in_as_source_material() {
        let work = work_plan();
        let graph = graph();
        let mut sources = ThematicSourceMaterialPlanV1::default();
        sources.insert("P", motif()).unwrap();
        sources.insert("P-dev", motif().invert(1)).unwrap();
        assert_eq!(
            sources.validate(&work, &graph),
            Err(ThematicSourceMaterialErrorV1::NonIndependentIdentity {
                identity_id: "P-dev".into(),
                origin: ThematicOriginV1::Derived,
            })
        );
    }

    #[test]
    fn empty_and_rest_only_material_fail_closed() {
        let work = work_plan();
        let graph = graph();

        let mut empty = ThematicSourceMaterialPlanV1::default();
        empty.insert("P", Motif::default()).unwrap();
        assert_eq!(
            empty.validate(&work, &graph),
            Err(ThematicSourceMaterialErrorV1::EmptyMotif {
                identity_id: "P".into()
            })
        );

        let mut rests = ThematicSourceMaterialPlanV1::default();
        rests
            .insert(
                "P",
                Motif::new(vec![MotifNote::rest(Duration::quarter())]),
            )
            .unwrap();
        assert_eq!(
            rests.validate(&work, &graph),
            Err(ThematicSourceMaterialErrorV1::RestOnlyMotif {
                identity_id: "P".into()
            })
        );
    }

    #[test]
    fn malformed_duration_is_rejected_after_deserialization_boundary() {
        let work = work_plan();
        let graph = graph();
        let mut sources = ThematicSourceMaterialPlanV1::default();
        sources
            .insert(
                "P",
                Motif::new(vec![MotifNote::new(1, Duration::new(0, 1))]),
            )
            .unwrap();
        assert_eq!(
            sources.validate(&work, &graph),
            Err(ThematicSourceMaterialErrorV1::InvalidDuration {
                identity_id: "P".into(),
                event_index: 0,
            })
        );
    }

    #[test]
    fn upstream_genealogy_error_is_preserved() {
        let work = work_plan();
        let mut graph = graph();
        graph.version = "wrong-graph-version".into();
        let mut sources = ThematicSourceMaterialPlanV1::default();
        sources.insert("P", motif()).unwrap();
        assert!(matches!(
            sources.validate(&work, &graph),
            Err(ThematicSourceMaterialErrorV1::InvalidThematicGraph(
                ThematicGraphErrorV1::WrongVersion { .. }
            ))
        ));
    }
}
