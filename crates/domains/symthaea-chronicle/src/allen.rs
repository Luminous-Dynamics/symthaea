// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Allen's interval algebra — the 13 jointly-exhaustive, pairwise-disjoint
//! relations that can hold between two time intervals (Allen 1983).
//!
//! This is the vocabulary the main crate's temporal system speaks, so computing
//! it here lets a [`Chronicle`](crate::Chronicle) hand a precise interval
//! relation across the integration boundary rather than the coarse
//! `precedes`/`contemporaneous` booleans.
//!
//! Intervals are `[start, end]` with `start <= end`. Point events
//! (`start == end`) are supported; they simply classify into the boundary
//! relations (`Meets`/`Equals`/…) that their coincident endpoints imply.

use crate::chronicle::Event;

/// One of Allen's 13 interval relations, describing `A` relative to `B`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllenRelation {
    /// A entirely before B (`a.end < b.start`).
    Before,
    /// A entirely after B.
    After,
    /// A's end coincides with B's start.
    Meets,
    /// A's start coincides with B's end.
    MetBy,
    /// A starts before B and they overlap, A ending inside B.
    Overlaps,
    /// The inverse of `Overlaps`.
    OverlappedBy,
    /// A and B start together, A ending first.
    Starts,
    /// A and B start together, B ending first.
    StartedBy,
    /// A falls strictly inside B.
    During,
    /// A strictly contains B.
    Contains,
    /// A and B end together, A starting later.
    Finishes,
    /// A and B end together, B starting later.
    FinishedBy,
    /// A and B are the same interval.
    Equals,
}

impl AllenRelation {
    /// The relation of `B` to `A` (converse). `x.relate(y).inverse()` equals
    /// `y.relate(x)`.
    pub fn inverse(self) -> AllenRelation {
        use AllenRelation::*;
        match self {
            Before => After,
            After => Before,
            Meets => MetBy,
            MetBy => Meets,
            Overlaps => OverlappedBy,
            OverlappedBy => Overlaps,
            Starts => StartedBy,
            StartedBy => Starts,
            During => Contains,
            Contains => During,
            Finishes => FinishedBy,
            FinishedBy => Finishes,
            Equals => Equals,
        }
    }

    /// A short human-readable label.
    pub fn label(self) -> &'static str {
        use AllenRelation::*;
        match self {
            Before => "before",
            After => "after",
            Meets => "meets",
            MetBy => "met by",
            Overlaps => "overlaps",
            OverlappedBy => "overlapped by",
            Starts => "starts",
            StartedBy => "started by",
            During => "during",
            Contains => "contains",
            Finishes => "finishes",
            FinishedBy => "finished by",
            Equals => "equal to",
        }
    }
}

/// Classify two intervals into one of Allen's 13 relations.
///
/// The relations are computed from the four endpoint comparisons and are
/// mutually exclusive and exhaustive for proper intervals.
pub fn relation(a: &Event, b: &Event) -> AllenRelation {
    use AllenRelation::*;
    use std::cmp::Ordering::*;

    let (a1, a2, b1, b2) = (a.start, a.end, b.start, b.end);

    // Disjoint / touching cases first.
    if a2 < b1 {
        return Before;
    }
    if a2 == b1 {
        return Meets;
    }
    if a1 > b2 {
        return After;
    }
    if a1 == b2 {
        return MetBy;
    }

    // Overlapping cases: classify by how the starts and ends compare.
    match (a1.cmp(&b1), a2.cmp(&b2)) {
        (Equal, Equal) => Equals,
        (Equal, Less) => Starts,
        (Equal, Greater) => StartedBy,
        (Greater, Equal) => Finishes,
        (Less, Equal) => FinishedBy,
        (Greater, Less) => During,
        (Less, Greater) => Contains,
        (Less, Less) => Overlaps,
        (Greater, Greater) => OverlappedBy,
    }
}

#[cfg(test)]
mod tests {
    use super::AllenRelation::*;
    use super::*;

    fn ev(start: i64, end: i64) -> Event {
        Event {
            name: "x".to_string(),
            start,
            end,
        }
    }

    #[test]
    fn the_seven_forward_relations() {
        assert_eq!(relation(&ev(0, 1), &ev(5, 6)), Before);
        assert_eq!(relation(&ev(0, 5), &ev(5, 9)), Meets);
        assert_eq!(relation(&ev(0, 5), &ev(3, 9)), Overlaps);
        assert_eq!(relation(&ev(0, 9), &ev(0, 5)), StartedBy);
        assert_eq!(relation(&ev(0, 5), &ev(0, 9)), Starts);
        assert_eq!(relation(&ev(3, 5), &ev(0, 9)), During);
        assert_eq!(relation(&ev(0, 9), &ev(3, 5)), Contains);
        assert_eq!(relation(&ev(3, 9), &ev(0, 9)), Finishes);
        assert_eq!(relation(&ev(0, 9), &ev(3, 9)), FinishedBy);
        assert_eq!(relation(&ev(2, 7), &ev(2, 7)), Equals);
    }

    #[test]
    fn inverses_are_consistent() {
        // relation(a,b).inverse() must equal relation(b,a) for every pair.
        let intervals = [(0, 1), (0, 5), (3, 9), (5, 6), (2, 7), (0, 9), (3, 5)];
        for &(a1, a2) in &intervals {
            for &(b1, b2) in &intervals {
                let a = ev(a1, a2);
                let b = ev(b1, b2);
                assert_eq!(
                    relation(&a, &b).inverse(),
                    relation(&b, &a),
                    "pair {:?} vs {:?}",
                    (a1, a2),
                    (b1, b2)
                );
            }
        }
    }

    #[test]
    fn labels_are_stable() {
        assert_eq!(Overlaps.label(), "overlaps");
        assert_eq!(Contains.label(), "contains");
    }
}
