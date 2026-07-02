// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! CRDT (Conflict-free Replicated Data Types) implementations

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use uuid::Uuid;

/// Lamport timestamp for ordering
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct LamportTimestamp {
    pub counter: u64,
    pub node_id: Uuid,
}

impl LamportTimestamp {
    pub fn new(node_id: Uuid) -> Self {
        Self { counter: 0, node_id }
    }

    pub fn increment(&mut self) -> Self {
        self.counter += 1;
        *self
    }

    pub fn update(&mut self, other: &Self) {
        self.counter = self.counter.max(other.counter) + 1;
    }
}

/// G-Counter (Grow-only Counter)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GCounter {
    counts: BTreeMap<Uuid, u64>,
}

impl GCounter {
    pub fn new() -> Self {
        Self {
            counts: BTreeMap::new(),
        }
    }

    pub fn increment(&mut self, node_id: Uuid) {
        *self.counts.entry(node_id).or_insert(0) += 1;
    }

    pub fn value(&self) -> u64 {
        self.counts.values().sum()
    }

    pub fn merge(&mut self, other: &Self) {
        for (node_id, &count) in &other.counts {
            let entry = self.counts.entry(*node_id).or_insert(0);
            *entry = (*entry).max(count);
        }
    }
}

impl Default for GCounter {
    fn default() -> Self {
        Self::new()
    }
}

/// PN-Counter (Positive-Negative Counter)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PNCounter {
    increments: GCounter,
    decrements: GCounter,
}

impl PNCounter {
    pub fn new() -> Self {
        Self {
            increments: GCounter::new(),
            decrements: GCounter::new(),
        }
    }

    pub fn increment(&mut self, node_id: Uuid) {
        self.increments.increment(node_id);
    }

    pub fn decrement(&mut self, node_id: Uuid) {
        self.decrements.increment(node_id);
    }

    pub fn value(&self) -> i64 {
        self.increments.value() as i64 - self.decrements.value() as i64
    }

    pub fn merge(&mut self, other: &Self) {
        self.increments.merge(&other.increments);
        self.decrements.merge(&other.decrements);
    }
}

impl Default for PNCounter {
    fn default() -> Self {
        Self::new()
    }
}

/// G-Set (Grow-only Set)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GSet<T: Eq + std::hash::Hash + Clone> {
    elements: HashSet<T>,
}

impl<T: Eq + std::hash::Hash + Clone> GSet<T> {
    pub fn new() -> Self {
        Self {
            elements: HashSet::new(),
        }
    }

    pub fn add(&mut self, element: T) {
        self.elements.insert(element);
    }

    pub fn contains(&self, element: &T) -> bool {
        self.elements.contains(element)
    }

    pub fn elements(&self) -> impl Iterator<Item = &T> {
        self.elements.iter()
    }

    pub fn merge(&mut self, other: &Self) {
        for element in &other.elements {
            self.elements.insert(element.clone());
        }
    }

    pub fn len(&self) -> usize {
        self.elements.len()
    }

    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }
}

impl<T: Eq + std::hash::Hash + Clone> Default for GSet<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// 2P-Set (Two-Phase Set - add and remove)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwoPhaseSet<T: Eq + std::hash::Hash + Clone> {
    added: GSet<T>,
    removed: GSet<T>,
}

impl<T: Eq + std::hash::Hash + Clone> TwoPhaseSet<T> {
    pub fn new() -> Self {
        Self {
            added: GSet::new(),
            removed: GSet::new(),
        }
    }

    pub fn add(&mut self, element: T) {
        if !self.removed.contains(&element) {
            self.added.add(element);
        }
    }

    pub fn remove(&mut self, element: T) {
        if self.added.contains(&element) {
            self.removed.add(element);
        }
    }

    pub fn contains(&self, element: &T) -> bool {
        self.added.contains(element) && !self.removed.contains(element)
    }

    pub fn elements(&self) -> Vec<T> {
        self.added
            .elements()
            .filter(|e| !self.removed.contains(e))
            .cloned()
            .collect()
    }

    pub fn merge(&mut self, other: &Self) {
        self.added.merge(&other.added);
        self.removed.merge(&other.removed);
    }
}

impl<T: Eq + std::hash::Hash + Clone> Default for TwoPhaseSet<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// LWW-Register (Last-Writer-Wins Register)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LWWRegister<T: Clone> {
    value: Option<T>,
    timestamp: LamportTimestamp,
}

impl<T: Clone> LWWRegister<T> {
    pub fn new(node_id: Uuid) -> Self {
        Self {
            value: None,
            timestamp: LamportTimestamp::new(node_id),
        }
    }

    pub fn set(&mut self, value: T) {
        self.value = Some(value);
        self.timestamp.increment();
    }

    pub fn get(&self) -> Option<&T> {
        self.value.as_ref()
    }

    pub fn merge(&mut self, other: &Self) {
        if other.timestamp > self.timestamp {
            self.value = other.value.clone();
            self.timestamp = other.timestamp;
        }
    }
}

/// Ordered list element
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ListElement<T: Clone> {
    pub id: LamportTimestamp,
    pub value: T,
    pub deleted: bool,
}

/// RGA (Replicated Growable Array) - ordered list
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RGAList<T: Clone + Eq> {
    elements: Vec<ListElement<T>>,
    node_id: Uuid,
    counter: u64,
}

impl<T: Clone + Eq> RGAList<T> {
    pub fn new(node_id: Uuid) -> Self {
        Self {
            elements: Vec::new(),
            node_id,
            counter: 0,
        }
    }

    fn next_id(&mut self) -> LamportTimestamp {
        self.counter += 1;
        LamportTimestamp {
            counter: self.counter,
            node_id: self.node_id,
        }
    }

    pub fn insert(&mut self, index: usize, value: T) {
        let id = self.next_id();
        let element = ListElement {
            id,
            value,
            deleted: false,
        };

        // Find actual position considering deleted elements
        let mut actual_index = 0;
        let mut visible_index = 0;
        for el in &self.elements {
            if !el.deleted {
                if visible_index == index {
                    break;
                }
                visible_index += 1;
            }
            actual_index += 1;
        }

        self.elements.insert(actual_index, element);
    }

    pub fn push(&mut self, value: T) {
        let id = self.next_id();
        self.elements.push(ListElement {
            id,
            value,
            deleted: false,
        });
    }

    pub fn remove(&mut self, index: usize) -> bool {
        let mut actual_index = 0;
        let mut visible_index = 0;

        for el in &mut self.elements {
            if !el.deleted {
                if visible_index == index {
                    el.deleted = true;
                    return true;
                }
                visible_index += 1;
            }
            actual_index += 1;
        }
        false
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        let mut visible_index = 0;
        for el in &self.elements {
            if !el.deleted {
                if visible_index == index {
                    return Some(&el.value);
                }
                visible_index += 1;
            }
        }
        None
    }

    pub fn to_vec(&self) -> Vec<T> {
        self.elements
            .iter()
            .filter(|e| !e.deleted)
            .map(|e| e.value.clone())
            .collect()
    }

    pub fn len(&self) -> usize {
        self.elements.iter().filter(|e| !e.deleted).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn merge(&mut self, other: &Self) {
        for other_el in &other.elements {
            // Find if element exists
            if let Some(el) = self.elements.iter_mut().find(|e| e.id == other_el.id) {
                // Mark as deleted if either has it deleted
                el.deleted = el.deleted || other_el.deleted;
            } else {
                // Insert at correct position based on ID
                let insert_pos = self.elements.iter().position(|e| e.id > other_el.id)
                    .unwrap_or(self.elements.len());
                self.elements.insert(insert_pos, other_el.clone());
            }
        }

        // Update our counter
        self.counter = self.counter.max(other.counter);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_g_counter() {
        let mut counter1 = GCounter::new();
        let mut counter2 = GCounter::new();
        let node1 = Uuid::new_v4();
        let node2 = Uuid::new_v4();

        counter1.increment(node1);
        counter1.increment(node1);
        counter2.increment(node2);

        counter1.merge(&counter2);
        assert_eq!(counter1.value(), 3);
    }

    #[test]
    fn test_rga_list() {
        let node_id = Uuid::new_v4();
        let mut list = RGAList::new(node_id);

        list.push("a".to_string());
        list.push("b".to_string());
        list.push("c".to_string());

        assert_eq!(list.len(), 3);
        assert_eq!(list.get(1), Some(&"b".to_string()));

        list.remove(1);
        assert_eq!(list.len(), 2);
        assert_eq!(list.to_vec(), vec!["a".to_string(), "c".to_string()]);
    }
}
