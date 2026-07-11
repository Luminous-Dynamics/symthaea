// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Event chronology, causal chains, and anachronism detection.
//!
//! Years are integers (negative = BCE). Temporal reasoning here is intentionally
//! minimal (before/overlap by comparison) — the rich Allen-interval algebra lives
//! in the main crate and should be used when this is integrated.

use std::collections::{HashMap, HashSet};

/// A dated historical event. `end` defaults to `start` (a point event).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Event {
    pub name: String,
    pub start: i64,
    pub end: i64,
}

/// An entity's lifespan (for anachronism checks).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lifespan {
    pub start: i64,
    pub end: i64,
}

/// A collection of events, causal links, and entity lifespans.
#[derive(Debug, Clone, Default)]
pub struct Chronicle {
    events: HashMap<String, Event>,
    causes: Vec<(String, String)>,
    lifespans: HashMap<String, Lifespan>,
}

impl Chronicle {
    pub fn new() -> Chronicle {
        Chronicle::default()
    }

    /// Add a point or interval event. Pass `end = None` for a point event.
    pub fn event(&mut self, name: &str, start: i64, end: Option<i64>) -> &mut Chronicle {
        self.events.insert(
            name.to_string(),
            Event {
                name: name.to_string(),
                start,
                end: end.unwrap_or(start),
            },
        );
        self
    }

    /// Record that `cause` causally contributed to `effect`.
    pub fn causation(&mut self, cause: &str, effect: &str) -> &mut Chronicle {
        self.causes.push((cause.to_string(), effect.to_string()));
        self
    }

    /// Register an entity's lifespan.
    pub fn entity(&mut self, name: &str, start: i64, end: i64) -> &mut Chronicle {
        self.lifespans
            .insert(name.to_string(), Lifespan { start, end });
        self
    }

    /// Event names in chronological order (by start year, then name).
    pub fn chronological(&self) -> Vec<String> {
        let mut evs: Vec<&Event> = self.events.values().collect();
        evs.sort_by(|a, b| a.start.cmp(&b.start).then(a.name.cmp(&b.name)));
        evs.into_iter().map(|e| e.name.clone()).collect()
    }

    /// Whether event `a` ends no later than `b` starts (strictly precedes).
    /// `None` if either event is unknown.
    pub fn precedes(&self, a: &str, b: &str) -> Option<bool> {
        let (ea, eb) = (self.events.get(a)?, self.events.get(b)?);
        Some(ea.end <= eb.start)
    }

    /// Whether two events' intervals overlap (contemporaneous).
    pub fn contemporaneous(&self, a: &str, b: &str) -> Option<bool> {
        let (ea, eb) = (self.events.get(a)?, self.events.get(b)?);
        Some(ea.start <= eb.end && eb.start <= ea.end)
    }

    /// Whether `cause` causally leads to `effect` (transitive reachability over
    /// the causal graph).
    pub fn causally_leads_to(&self, cause: &str, effect: &str) -> bool {
        let mut stack = vec![cause.to_string()];
        let mut seen = HashSet::new();
        while let Some(node) = stack.pop() {
            if node == effect && node != cause {
                return true;
            }
            if !seen.insert(node.clone()) {
                continue;
            }
            for (from, to) in &self.causes {
                if *from == node {
                    if to == effect {
                        return true;
                    }
                    stack.push(to.clone());
                }
            }
        }
        false
    }

    /// A causal chain (path of event names) from `cause` to `effect`, if any.
    pub fn causal_chain(&self, cause: &str, effect: &str) -> Option<Vec<String>> {
        let mut path = vec![cause.to_string()];
        let mut seen = HashSet::new();
        if self.dfs_chain(cause, effect, &mut path, &mut seen) {
            Some(path)
        } else {
            None
        }
    }

    fn dfs_chain(
        &self,
        node: &str,
        target: &str,
        path: &mut Vec<String>,
        seen: &mut HashSet<String>,
    ) -> bool {
        if node == target && path.len() > 1 {
            return true;
        }
        if !seen.insert(node.to_string()) {
            return false;
        }
        for (from, to) in &self.causes {
            if from == node {
                path.push(to.clone());
                if to == target || self.dfs_chain(to, target, path, seen) {
                    return true;
                }
                path.pop();
            }
        }
        false
    }

    /// Whether referencing `entity` at `year` is an anachronism (outside its
    /// lifespan). `None` if the entity is unknown.
    pub fn is_anachronistic(&self, entity: &str, year: i64) -> Option<bool> {
        let span = self.lifespans.get(entity)?;
        Some(year < span.start || year > span.end)
    }

    /// Look up an event's full record (name, start, end) by name. Lets a
    /// consumer (e.g. the main crate's temporal bridge) read interval bounds.
    pub fn get_event(&self, name: &str) -> Option<&Event> {
        self.events.get(name)
    }

    /// The precise Allen-interval relation of event `a` to event `b` (e.g.
    /// `Overlaps`, `During`, `Meets`). This is the vocabulary to hand across the
    /// boundary to the main crate's temporal system. `None` if either event is
    /// unknown. See [`crate::allen`].
    pub fn allen_relation(&self, a: &str, b: &str) -> Option<crate::allen::AllenRelation> {
        let (ea, eb) = (self.events.get(a)?, self.events.get(b)?);
        Some(crate::allen::relation(ea, eb))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn history() -> Chronicle {
        let mut c = Chronicle::new();
        c.event("printing_press", 1440, None)
            .event("reformation", 1517, Some(1648))
            .event("enlightenment", 1685, Some(1815))
            .causation("printing_press", "reformation")
            .causation("reformation", "enlightenment")
            .entity("napoleon", 1769, 1821);
        c
    }

    #[test]
    fn chronological_order() {
        assert_eq!(
            history().chronological(),
            vec!["printing_press", "reformation", "enlightenment"]
        );
    }

    #[test]
    fn allen_relation_by_name() {
        use crate::allen::AllenRelation;
        let c = history();
        // reformation [1517,1648] vs enlightenment [1685,1815] → strictly before.
        assert_eq!(
            c.allen_relation("reformation", "enlightenment"),
            Some(AllenRelation::Before)
        );
        // Converse is After.
        assert_eq!(
            c.allen_relation("enlightenment", "reformation"),
            Some(AllenRelation::After)
        );
        assert_eq!(c.allen_relation("reformation", "unknown"), None);
    }

    #[test]
    fn precedence_and_contemporaneity() {
        let c = history();
        assert_eq!(c.precedes("printing_press", "reformation"), Some(true));
        assert_eq!(c.precedes("reformation", "printing_press"), Some(false));
        // Reformation (1517-1648) and Enlightenment (1685-1815) don't overlap.
        assert_eq!(
            c.contemporaneous("reformation", "enlightenment"),
            Some(false)
        );
        assert_eq!(c.precedes("printing_press", "unknown"), None);
    }

    #[test]
    fn causal_reachability_is_transitive() {
        let c = history();
        // press → reformation → enlightenment: press causally leads to enlightenment.
        assert!(c.causally_leads_to("printing_press", "enlightenment"));
        assert!(!c.causally_leads_to("enlightenment", "printing_press"));
        let chain = c.causal_chain("printing_press", "enlightenment").unwrap();
        assert_eq!(
            chain,
            vec!["printing_press", "reformation", "enlightenment"]
        );
    }

    #[test]
    fn anachronism_detection() {
        let c = history();
        // Napoleon (1769-1821) with a smartphone (2007) → anachronism.
        assert_eq!(c.is_anachronistic("napoleon", 2007), Some(true));
        assert_eq!(c.is_anachronistic("napoleon", 1800), Some(false));
        assert_eq!(c.is_anachronistic("napoleon", 1750), Some(true)); // before birth
    }
}
