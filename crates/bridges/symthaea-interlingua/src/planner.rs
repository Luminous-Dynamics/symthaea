// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Per-message transfer planning for SCIP cognitive state synchronization.
//!
//! A session negotiates capabilities, but each message should choose the cheapest
//! grounded semantic synchronization available. HDC projection delivery is a
//! separate bandwidth-for-compute choice.

use crate::HdcWireEncoding;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticTransferMode {
    /// Receiver already owns the exact target graph; send only its content address.
    SemanticReference,
    /// Receiver owns the exact base graph; send an exact content-addressed delta.
    GraphDelta,
    /// Send the complete canonical grounded graph.
    GroundedGraph,
    /// Presentation/degraded fallback; not equivalent to grounded machine sync.
    HumanTextFallback,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectionAttachment {
    None,
    Hdc(HdcWireEncoding),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectionCandidate {
    pub encoding: HdcWireEncoding,
    pub bytes: usize,
    pub cosine_similarity: f32,
    pub exact: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TransferPlanningInput {
    /// `Some(bytes)` only when the receiver has the exact target graph cached.
    pub semantic_reference_bytes: Option<usize>,
    /// `Some(bytes)` only when the receiver owns the exact required base graph.
    pub graph_delta_bytes: Option<usize>,
    pub grounded_graph_bytes: usize,
    pub human_text_bytes: Option<usize>,
    pub projection_candidates: Vec<ProjectionCandidate>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ProjectionPolicy {
    /// Bandwidth-first default. Receiver deterministically reconstructs HDC.
    ReconstructLocally,
    /// Attach the smallest projection satisfying the requested fidelity.
    AttachSmallest {
        minimum_cosine: f32,
        require_exact: bool,
        max_extra_bytes: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransferPolicy {
    /// Machine-to-machine default: text may not replace grounded semantics.
    pub require_grounded_semantics: bool,
    pub projection: ProjectionPolicy,
}

impl Default for TransferPolicy {
    fn default() -> Self {
        Self {
            require_grounded_semantics: true,
            projection: ProjectionPolicy::ReconstructLocally,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransferPlan {
    pub semantic: SemanticTransferMode,
    pub semantic_bytes: usize,
    pub projection: ProjectionAttachment,
    pub projection_bytes: usize,
    pub total_bytes: usize,
}

/// Select the smallest applicable semantic transfer and, independently, an
/// optional HDC projection attachment.
pub fn plan_transfer(
    input: &TransferPlanningInput,
    policy: TransferPolicy,
) -> Option<TransferPlan> {
    let mut semantic_candidates = Vec::with_capacity(4);

    if let Some(bytes) = input.semantic_reference_bytes {
        semantic_candidates.push((SemanticTransferMode::SemanticReference, bytes));
    }
    if let Some(bytes) = input.graph_delta_bytes {
        semantic_candidates.push((SemanticTransferMode::GraphDelta, bytes));
    }
    semantic_candidates.push((
        SemanticTransferMode::GroundedGraph,
        input.grounded_graph_bytes,
    ));
    if !policy.require_grounded_semantics {
        if let Some(bytes) = input.human_text_bytes {
            semantic_candidates.push((SemanticTransferMode::HumanTextFallback, bytes));
        }
    }

    semantic_candidates.sort_by_key(|(_, bytes)| *bytes);
    let (semantic, semantic_bytes) = *semantic_candidates.first()?;

    let (projection, projection_bytes) = match policy.projection {
        ProjectionPolicy::ReconstructLocally => (ProjectionAttachment::None, 0),
        ProjectionPolicy::AttachSmallest {
            minimum_cosine,
            require_exact,
            max_extra_bytes,
        } => {
            if !minimum_cosine.is_finite() || !(-1.0..=1.0).contains(&minimum_cosine) {
                return None;
            }

            let mut candidates = input
                .projection_candidates
                .iter()
                .copied()
                .filter(|candidate| {
                    candidate.bytes <= max_extra_bytes
                        && candidate.cosine_similarity.is_finite()
                        && candidate.cosine_similarity >= minimum_cosine
                        && (!require_exact || candidate.exact)
                })
                .collect::<Vec<_>>();

            candidates.sort_by(|left, right| {
                left.bytes.cmp(&right.bytes).then_with(|| {
                    right
                        .cosine_similarity
                        .total_cmp(&left.cosine_similarity)
                })
            });

            if let Some(candidate) = candidates.first() {
                (
                    ProjectionAttachment::Hdc(candidate.encoding),
                    candidate.bytes,
                )
            } else {
                (ProjectionAttachment::None, 0)
            }
        }
    };

    Some(TransferPlan {
        semantic,
        semantic_bytes,
        projection,
        projection_bytes,
        total_bytes: semantic_bytes.saturating_add(projection_bytes),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input() -> TransferPlanningInput {
        TransferPlanningInput {
            semantic_reference_bytes: None,
            graph_delta_bytes: Some(350),
            grounded_graph_bytes: 5_000,
            human_text_bytes: Some(120),
            projection_candidates: vec![
                ProjectionCandidate {
                    encoding: HdcWireEncoding::Q8SymmetricV1,
                    bytes: 16_384,
                    cosine_similarity: 0.9995,
                    exact: false,
                },
                ProjectionCandidate {
                    encoding: HdcWireEncoding::F32LeV1,
                    bytes: 65_536,
                    cosine_similarity: 1.0,
                    exact: true,
                },
            ],
        }
    }

    #[test]
    fn grounded_sync_prefers_delta_over_tiny_text() {
        let plan = plan_transfer(&input(), TransferPolicy::default()).unwrap();
        assert_eq!(plan.semantic, SemanticTransferMode::GraphDelta);
        assert_eq!(plan.semantic_bytes, 350);
        assert_eq!(plan.projection, ProjectionAttachment::None);
    }

    #[test]
    fn semantic_reference_wins_when_target_is_cached() {
        let mut input = input();
        input.semantic_reference_bytes = Some(72);
        let plan = plan_transfer(&input, TransferPolicy::default()).unwrap();
        assert_eq!(plan.semantic, SemanticTransferMode::SemanticReference);
        assert_eq!(plan.semantic_bytes, 72);
    }

    #[test]
    fn projection_attachment_is_independent_of_semantic_sync() {
        let policy = TransferPolicy {
            require_grounded_semantics: true,
            projection: ProjectionPolicy::AttachSmallest {
                minimum_cosine: 0.999,
                require_exact: false,
                max_extra_bytes: 20_000,
            },
        };
        let plan = plan_transfer(&input(), policy).unwrap();
        assert_eq!(plan.semantic, SemanticTransferMode::GraphDelta);
        assert_eq!(
            plan.projection,
            ProjectionAttachment::Hdc(HdcWireEncoding::Q8SymmetricV1)
        );
        assert_eq!(plan.total_bytes, 350 + 16_384);
    }

    #[test]
    fn exact_policy_does_not_silently_use_lossy_projection() {
        let policy = TransferPolicy {
            require_grounded_semantics: true,
            projection: ProjectionPolicy::AttachSmallest {
                minimum_cosine: 1.0,
                require_exact: true,
                max_extra_bytes: 20_000,
            },
        };
        let plan = plan_transfer(&input(), policy).unwrap();
        assert_eq!(plan.projection, ProjectionAttachment::None);
    }
}
