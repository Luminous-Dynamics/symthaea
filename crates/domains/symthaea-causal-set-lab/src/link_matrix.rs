// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact finite-causal-set link matrix and two-link path diagnostics.
//!
//! For events x < y, the directed relation is a link iff the open order
//! interval I(x,y) contains no event. The optimized constructor first builds
//! predecessor/successor bitsets for the causal order, then tests whether
//! successor(x) intersect predecessor(y) is empty.
//!
//! This module is an exact finite-poset primitive. It does not assign physical
//! propagator weights, continuum normalizations, mass-scattering signs or
//! curvature couplings. Those must be taken from the relevant paper equations
//! in the separate causal-propagator research workstream.

use crate::{CausalSet2D, Event2D};
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

/// Maximum events accepted by the optimized bitset constructor.
pub const MAX_LINK_MATRIX_EVENTS: usize = 16_384;
/// Maximum events accepted by the cubic brute-force reference.
pub const MAX_LINK_ORACLE_EVENTS: usize = 512;
/// Hard bound for the two causal-order bitset matrices plus link bitsets.
pub const MAX_LINK_MATRIX_BYTES: usize = 64 * 1024 * 1024;

/// Invalid input or resource limit in a link-matrix construction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LinkMatrixError {
    TooManyEvents {
        count: usize,
        maximum: usize,
    },
    OracleLimitExceeded {
        count: usize,
        maximum: usize,
    },
    MemoryBudgetExceeded {
        required_bytes: usize,
        maximum_bytes: usize,
    },
    UnknownEventId {
        event_id: usize,
    },
    NumericalOverflow,
}

impl fmt::Display for LinkMatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyEvents { count, maximum } => {
                write!(f, "link matrix has {count} events; maximum is {maximum}")
            }
            Self::OracleLimitExceeded { count, maximum } => {
                write!(f, "brute-force link oracle has {count} events; maximum is {maximum}")
            }
            Self::MemoryBudgetExceeded {
                required_bytes,
                maximum_bytes,
            } => write!(
                f,
                "link matrix requires {required_bytes} bytes; budget is {maximum_bytes}"
            ),
            Self::UnknownEventId { event_id } => {
                write!(f, "event ID {event_id} is not in this link matrix")
            }
            Self::NumericalOverflow => write!(f, "link matrix resource calculation overflowed"),
        }
    }
}

impl Error for LinkMatrixError {}

/// Directed link matrix indexed by a deterministic topological event order.
///
/// Row i contains outgoing links from event_ids[i]. Column membership is
/// mirrored in incoming_rows so two-link path counts can be computed by
/// bitset intersection. The stored graph is acyclic by construction because
/// all links point forward in the causal topological order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinkMatrix {
    event_ids: Vec<usize>,
    event_index: BTreeMap<usize, usize>,
    outgoing_rows: Vec<Vec<u64>>,
    incoming_rows: Vec<Vec<u64>>,
    link_count: u64,
}

impl LinkMatrix {
    /// Construct a link matrix using a cubic brute-force empty-interval oracle.
    ///
    /// This path is intentionally capped at MAX_LINK_ORACLE_EVENTS and is used
    /// only for differential testing and small exact references.
    pub fn from_causet_bruteforce(causet: &CausalSet2D) -> Result<Self, LinkMatrixError> {
        let events = topological_events(causet);
        let n = events.len();
        if n > MAX_LINK_ORACLE_EVENTS {
            return Err(LinkMatrixError::OracleLimitExceeded {
                count: n,
                maximum: MAX_LINK_ORACLE_EVENTS,
            });
        }
        let words = words_for(n);
        check_memory_budget(n, words)?;

        let mut outgoing_rows = vec![vec![0_u64; words]; n];
        let mut incoming_rows = vec![vec![0_u64; words]; n];
        let mut link_count = 0_u64;

        for source in 0..n {
            for target in (source + 1)..n {
                if !events[source].causally_precedes(events[target]) {
                    continue;
                }

                let mut has_intermediate = false;
                for middle in (source + 1)..target {
                    if events[source].causally_precedes(events[middle])
                        && events[middle].causally_precedes(events[target])
                    {
                        has_intermediate = true;
                        break;
                    }
                }

                if !has_intermediate {
                    set_bit(&mut outgoing_rows[source], target);
                    set_bit(&mut incoming_rows[target], source);
                    link_count = link_count
                        .checked_add(1)
                        .ok_or(LinkMatrixError::NumericalOverflow)?;
                }
            }
        }

        Self::from_parts(events, outgoing_rows, incoming_rows, link_count)
    }

    /// Construct a link matrix from causal predecessor/successor bitsets.
    ///
    /// Time complexity is O(N² + R ceil(N/64)), where R is the number of
    /// related event pairs. Memory is O(N²) bits with a strict 64 MiB budget.
    /// For each related pair x<y, it is a link exactly when
    /// successors(x) & predecessors(y) is empty.
    pub fn from_causet_bitset(causet: &CausalSet2D) -> Result<Self, LinkMatrixError> {
        let events = topological_events(causet);
        let n = events.len();
        if n > MAX_LINK_MATRIX_EVENTS {
            return Err(LinkMatrixError::TooManyEvents {
                count: n,
                maximum: MAX_LINK_MATRIX_EVENTS,
            });
        }
        let words = words_for(n);
        check_memory_budget(n, words)?;

        let mut causal_successors = vec![vec![0_u64; words]; n];
        let mut causal_predecessors = vec![vec![0_u64; words]; n];

        for source in 0..n {
            for target in (source + 1)..n {
                if events[source].causally_precedes(events[target]) {
                    set_bit(&mut causal_successors[source], target);
                    set_bit(&mut causal_predecessors[target], source);
                }
            }
        }

        let mut outgoing_rows = vec![vec![0_u64; words]; n];
        let mut incoming_rows = vec![vec![0_u64; words]; n];
        let mut link_count = 0_u64;

        for source in 0..n {
            for target in (source + 1)..n {
                if !get_bit(&causal_successors[source], target) {
                    continue;
                }

                let has_intermediate = causal_successors[source]
                    .iter()
                    .zip(&causal_predecessors[target])
                    .any(|(successors, predecessors)| successors & predecessors != 0);

                if !has_intermediate {
                    set_bit(&mut outgoing_rows[source], target);
                    set_bit(&mut incoming_rows[target], source);
                    link_count = link_count
                        .checked_add(1)
                        .ok_or(LinkMatrixError::NumericalOverflow)?;
                }
            }
        }

        Self::from_parts(events, outgoing_rows, incoming_rows, link_count)
    }

    fn from_parts(
        events: Vec<Event2D>,
        outgoing_rows: Vec<Vec<u64>>,
        incoming_rows: Vec<Vec<u64>>,
        link_count: u64,
    ) -> Result<Self, LinkMatrixError> {
        let event_ids: Vec<usize> = events.iter().map(|event| event.id).collect();
        let event_index: BTreeMap<usize, usize> = event_ids
            .iter()
            .copied()
            .enumerate()
            .map(|(index, event_id)| (event_id, index))
            .collect();
        if event_index.len() != event_ids.len() {
            return Err(LinkMatrixError::NumericalOverflow);
        }
        Ok(Self {
            event_ids,
            event_index,
            outgoing_rows,
            incoming_rows,
            link_count,
        })
    }

    pub fn len(&self) -> usize {
        self.event_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.event_ids.is_empty()
    }

    pub fn link_count(&self) -> u64 {
        self.link_count
    }

    /// IDs in the deterministic topological order used by the matrix.
    pub fn event_ids(&self) -> &[usize] {
        &self.event_ids
    }

    /// Whether source -> target is a direct link.
    pub fn has_link(&self, source_event_id: usize, target_event_id: usize) -> Result<bool, LinkMatrixError> {
        let source = self.event_index.get(&source_event_id).copied().ok_or(
            LinkMatrixError::UnknownEventId {
                event_id: source_event_id,
            },
        )?;
        let target = self.event_index.get(&target_event_id).copied().ok_or(
            LinkMatrixError::UnknownEventId {
                event_id: target_event_id,
            },
        )?;
        if source >= target {
            return Ok(false);
        }
        Ok(get_bit(&self.outgoing_rows[source], target))
    }

    /// Number of distinct two-link paths source -> middle -> target.
    ///
    /// The calculation is an exact bitset intersection over intermediate
    /// events. It returns zero for non-related or unknown-direction pairs.
    pub fn two_link_path_count(
        &self,
        source_event_id: usize,
        target_event_id: usize,
    ) -> Result<u64, LinkMatrixError> {
        let source = self.event_index.get(&source_event_id).copied().ok_or(
            LinkMatrixError::UnknownEventId {
                event_id: source_event_id,
            },
        )?;
        let target = self.event_index.get(&target_event_id).copied().ok_or(
            LinkMatrixError::UnknownEventId {
                event_id: target_event_id,
            },
        )?;
        Ok(self.outgoing_rows[source]
            .iter()
            .zip(&self.incoming_rows[target])
            .map(|(outgoing, incoming)| (outgoing & incoming).count_ones() as u64)
            .sum())
    }

    /// Total number of length-two directed paths in the causal set.
    pub fn total_two_link_paths(&self) -> u64 {
        let mut total = 0_u64;
        for source in 0..self.len() {
            for target in (source + 1)..self.len() {
                total = total.saturating_add(
                    self.outgoing_rows[source]
                        .iter()
                        .zip(&self.incoming_rows[target])
                        .map(|(outgoing, incoming)| (outgoing & incoming).count_ones() as u64)
                        .sum::<u64>(),
                );
            }
        }
        total
    }

    /// Longest directed path in the link DAG, measured in link edges.
    pub fn longest_link_path_length(&self) -> usize {
        let n = self.len();
        let mut distance = vec![0_usize; n];
        let mut best = 0_usize;
        for source in 0..n {
            let next_distance = distance[source].saturating_add(1);
            best = best.max(distance[source]);
            for target in (source + 1)..n {
                if get_bit(&self.outgoing_rows[source], target) {
                    distance[target] = distance[target].max(next_distance);
                }
            }
        }
        distance.into_iter().fold(best, usize::max)
    }

    /// Check that all nonzero links point strictly forward in stored order.
    pub fn is_acyclic(&self) -> bool {
        for source in 0..self.len() {
            for target in 0..=source {
                if get_bit(&self.outgoing_rows[source], target) {
                    return false;
                }
            }
        }
        true
    }

    /// Return all (source ID, target ID) link pairs in deterministic order.
    pub fn link_pairs(&self) -> Vec<(usize, usize)> {
        let mut pairs = Vec::with_capacity(self.link_count as usize);
        for source in 0..self.len() {
            for target in (source + 1)..self.len() {
                if get_bit(&self.outgoing_rows[source], target) {
                    pairs.push((self.event_ids[source], self.event_ids[target]));
                }
            }
        }
        pairs
    }
}

fn topological_events(causet: &CausalSet2D) -> Vec<Event2D> {
    let mut events = causet.events().to_vec();
    events.sort_by(|left, right| {
        left.u
            .total_cmp(&right.u)
            .then_with(|| left.v.total_cmp(&right.v))
            .then_with(|| left.id.cmp(&right.id))
    });
    events
}

fn words_for(event_count: usize) -> usize {
    event_count.div_ceil(64)
}

fn check_memory_budget(event_count: usize, word_count: usize) -> Result<(), LinkMatrixError> {
    let matrix_words = event_count
        .checked_mul(word_count)
        .and_then(|value| value.checked_mul(4))
        .ok_or(LinkMatrixError::NumericalOverflow)?;
    let required_bytes = matrix_words
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or(LinkMatrixError::NumericalOverflow)?;
    if required_bytes > MAX_LINK_MATRIX_BYTES {
        return Err(LinkMatrixError::MemoryBudgetExceeded {
            required_bytes,
            maximum_bytes: MAX_LINK_MATRIX_BYTES,
        });
    }
    Ok(())
}

fn set_bit(row: &mut [u64], index: usize) {
    row[index / 64] |= 1_u64 << (index % 64);
}

fn get_bit(row: &[u64], index: usize) -> bool {
    row.get(index / 64)
        .is_some_and(|word| word & (1_u64 << (index % 64)) != 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn causet_from_points(points: &[(usize, f64, f64)]) -> CausalSet2D {
        CausalSet2D::new(
            points
                .iter()
                .map(|(id, u, v)| Event2D {
                    id: *id,
                    u: *u,
                    v: *v,
                })
                .collect(),
            10.0,
            10.0,
        )
        .unwrap()
    }

    #[test]
    fn total_order_links_only_adjacent_events() {
        let causet = causet_from_points(&[
            (10, 0.0, 0.0),
            (20, 1.0, 1.0),
            (30, 2.0, 2.0),
            (40, 3.0, 3.0),
        ]);
        let matrix = LinkMatrix::from_causet_bitset(&causet).unwrap();

        assert_eq!(matrix.link_count(), 3);
        assert_eq!(matrix.link_pairs(), vec![(10, 20), (20, 30), (30, 40)]);
        assert_eq!(matrix.two_link_path_count(10, 30).unwrap(), 1);
        assert_eq!(matrix.two_link_path_count(10, 40).unwrap(), 0);
        assert_eq!(matrix.total_two_link_paths(), 2);
        assert_eq!(matrix.longest_link_path_length(), 3);
        assert!(matrix.is_acyclic());
    }

    #[test]
    fn antichain_has_no_links_or_two_link_paths() {
        let causet = causet_from_points(&[
            (0, 0.0, 3.0),
            (1, 1.0, 2.0),
            (2, 2.0, 1.0),
            (3, 3.0, 0.0),
        ]);
        let matrix = LinkMatrix::from_causet_bitset(&causet).unwrap();
        assert_eq!(matrix.link_count(), 0);
        assert_eq!(matrix.total_two_link_paths(), 0);
        assert_eq!(matrix.longest_link_path_length(), 0);
    }

    #[test]
    fn bitset_constructor_matches_bruteforce_on_seeded_sprinklings() {
        for seed in 0..64_u64 {
            let causet =
                CausalSet2D::sprinkle_fixed_cardinality(seed * 7919 + 17, 48, 3.0).unwrap();
            let oracle = LinkMatrix::from_causet_bruteforce(&causet).unwrap();
            let optimized = LinkMatrix::from_causet_bitset(&causet).unwrap();
            assert_eq!(optimized, oracle, "link matrix mismatch at seed={seed}");
        }
    }

    #[test]
    fn relabelling_events_preserves_link_structure_after_id_mapping() {
        let points = [
            (100, 0.1, 0.2),
            (200, 0.2, 0.8),
            (300, 0.7, 0.4),
            (400, 1.1, 1.2),
            (500, 1.5, 2.0),
            (600, 2.0, 1.6),
        ];
        let original = causet_from_points(&points);
        let relabelled = causet_from_points(&[
            (50, 0.1, 0.2),
            (11, 0.2, 0.8),
            (900, 0.7, 0.4),
            (77, 1.1, 1.2),
            (4, 1.5, 2.0),
            (81, 2.0, 1.6),
        ]);
        let left = LinkMatrix::from_causet_bitset(&original).unwrap();
        let right = LinkMatrix::from_causet_bitset(&relabelled).unwrap();

        let mut original_coordinate_links: Vec<((u64, u64), (u64, u64))> = left
            .link_pairs()
            .into_iter()
            .map(|(source, target)| {
                let a = original.events().iter().find(|event| event.id == source).unwrap();
                let b = original.events().iter().find(|event| event.id == target).unwrap();
                ((a.u.to_bits(), a.v.to_bits()), (b.u.to_bits(), b.v.to_bits()))
            })
            .collect();
        let mut relabelled_coordinate_links: Vec<((u64, u64), (u64, u64))> = right
            .link_pairs()
            .into_iter()
            .map(|(source, target)| {
                let a = relabelled.events().iter().find(|event| event.id == source).unwrap();
                let b = relabelled.events().iter().find(|event| event.id == target).unwrap();
                ((a.u.to_bits(), a.v.to_bits()), (b.u.to_bits(), b.v.to_bits()))
            })
            .collect();
        original_coordinate_links.sort_unstable();
        relabelled_coordinate_links.sort_unstable();
        assert_eq!(original_coordinate_links, relabelled_coordinate_links);
    }

    #[test]
    fn duplicate_coordinates_are_not_links_or_intermediate_events() {
        let causet = causet_from_points(&[
            (0, 0.0, 0.0),
            (1, 1.0, 1.0),
            (2, 1.0, 1.0),
            (3, 2.0, 2.0),
        ]);
        let matrix = LinkMatrix::from_causet_bitset(&causet).unwrap();
        let oracle = LinkMatrix::from_causet_bruteforce(&causet).unwrap();
        assert_eq!(matrix, oracle);
        assert_eq!(matrix.link_count(), 4);
    }

    #[test]
    fn empty_and_singleton_posets_are_valid() {
        let empty = CausalSet2D::new(vec![], 1.0, 1.0).unwrap();
        let singleton = CausalSet2D::new(
            vec![Event2D { id: 7, u: 0.5, v: 0.5 }],
            1.0,
            1.0,
        )
        .unwrap();
        for causet in [&empty, &singleton] {
            let matrix = LinkMatrix::from_causet_bitset(causet).unwrap();
            assert_eq!(matrix.link_count(), 0);
            assert!(matrix.is_acyclic());
            assert_eq!(matrix.longest_link_path_length(), 0);
        }
    }

    #[test]
    fn unknown_ids_and_oracle_limit_fail_closed() {
        let causet = CausalSet2D::sprinkle_fixed_cardinality(7, 32, 2.0).unwrap();
        let matrix = LinkMatrix::from_causet_bitset(&causet).unwrap();
        assert_eq!(
            matrix.has_link(999, 1),
            Err(LinkMatrixError::UnknownEventId { event_id: 999 })
        );

        let large = CausalSet2D::sprinkle_fixed_cardinality(8, MAX_LINK_ORACLE_EVENTS + 1, 2.0)
            .unwrap();
        assert!(matches!(
            LinkMatrix::from_causet_bruteforce(&large),
            Err(LinkMatrixError::OracleLimitExceeded { .. })
        ));
    }
}
