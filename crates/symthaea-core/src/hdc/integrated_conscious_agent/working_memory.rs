// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Working Memory - Global Workspace Theory Implementation

use super::super::unified_hv::ContinuousHV;
use std::collections::VecDeque;

/// Working memory with limited capacity (inspired by Baddeley's model)
#[derive(Clone, Debug)]
pub struct WorkingMemory {
    /// Central executive - controls attention allocation
    pub(crate) central_executive_load: f64,
    /// Phonological loop - verbal/acoustic information
    phonological_buffer: VecDeque<ContinuousHV>,
    /// Visuospatial sketchpad - visual/spatial information
    visuospatial_buffer: VecDeque<ContinuousHV>,
    /// Episodic buffer - integrates information from multiple sources
    pub(crate) episodic_buffer: VecDeque<WorkingMemoryItem>,
    /// Maximum capacity per buffer (Miller's 7±2)
    capacity: usize,
    /// Decay rate for items in working memory
    decay_rate: f64,
}

/// An item in working memory
#[derive(Clone, Debug)]
pub struct WorkingMemoryItem {
    /// The content vector
    pub content: ContinuousHV,
    /// When this item was added
    pub timestamp: usize,
    /// Current activation level (0-1)
    pub activation: f64,
    /// Source of this item
    pub source: MemorySource,
    /// Relevance to current goals
    pub goal_relevance: f64,
}

/// Source of a working memory item
#[derive(Clone, Debug, PartialEq)]
pub enum MemorySource {
    Perception,
    LongTermMemory,
    InternalGeneration,
    GoalActivation,
}

impl WorkingMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            central_executive_load: 0.0,
            phonological_buffer: VecDeque::with_capacity(capacity),
            visuospatial_buffer: VecDeque::with_capacity(capacity),
            episodic_buffer: VecDeque::with_capacity(capacity),
            capacity,
            decay_rate: 0.1,
        }
    }

    /// Add item to episodic buffer (the integration hub)
    pub fn add_to_episodic(
        &mut self,
        content: ContinuousHV,
        source: MemorySource,
        goal_relevance: f64,
        timestamp: usize,
    ) {
        // If at capacity, remove least activated item
        if self.episodic_buffer.len() >= self.capacity {
            // Find and remove lowest activation item
            if let Some(min_idx) = self
                .episodic_buffer
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.activation.total_cmp(&b.1.activation))
                .map(|(i, _)| i)
            {
                self.episodic_buffer.remove(min_idx);
            }
        }

        self.episodic_buffer.push_back(WorkingMemoryItem {
            content,
            timestamp,
            activation: 1.0,
            source,
            goal_relevance,
        });
    }

    /// Update working memory (decay + rehearsal)
    pub fn update(&mut self, current_focus: Option<&ContinuousHV>) {
        for item in self.episodic_buffer.iter_mut() {
            // Natural decay
            item.activation *= 1.0 - self.decay_rate;

            // Rehearsal boost if similar to current focus
            if let Some(focus) = current_focus {
                let similarity = item.content.similarity(focus).max(0.0) as f64;
                if similarity > 0.5 {
                    item.activation = (item.activation + similarity * 0.2).min(1.0);
                }
            }
        }

        // Remove items below threshold
        self.episodic_buffer.retain(|item| item.activation > 0.1);

        // Update central executive load
        self.central_executive_load = self.episodic_buffer.len() as f64 / self.capacity as f64;
    }

    /// Get most activated item
    pub fn most_active(&self) -> Option<&WorkingMemoryItem> {
        self.episodic_buffer
            .iter()
            .max_by(|a, b| a.activation.total_cmp(&b.activation))
    }

    /// Get working memory load (0-1)
    pub fn load(&self) -> f64 {
        self.central_executive_load
    }

    /// Get average activation level
    pub fn average_activation(&self) -> f64 {
        if self.episodic_buffer.is_empty() {
            return 0.0;
        }
        self.episodic_buffer
            .iter()
            .map(|i| i.activation)
            .sum::<f64>()
            / self.episodic_buffer.len() as f64
    }

    /// Check if working memory is overloaded
    pub fn is_overloaded(&self) -> bool {
        self.central_executive_load > 0.9
    }

    // =========================================================================
    // Phonological Loop - Verbal/Acoustic Information (Baddeley's Model)
    // =========================================================================

    /// Add item to phonological loop (verbal/acoustic information)
    ///
    /// The phonological loop handles verbal-linguistic information through
    /// subvocal rehearsal. Items decay quickly without active rehearsal.
    pub fn add_to_phonological(&mut self, content: ContinuousHV) {
        if self.phonological_buffer.len() >= self.capacity {
            self.phonological_buffer.pop_front();
        }
        self.phonological_buffer.push_back(content);
    }

    /// Rehearse phonological loop (prevents decay)
    ///
    /// Subvocal rehearsal maintains items in the phonological loop.
    /// Returns the rehearsed items bundled together.
    pub fn rehearse_phonological(&self) -> Option<ContinuousHV> {
        if self.phonological_buffer.is_empty() {
            return None;
        }
        let owned: Vec<ContinuousHV> = self.phonological_buffer.iter().cloned().collect();
        Some(ContinuousHV::bundle_owned(&owned))
    }

    /// Get phonological buffer contents
    pub fn phonological_contents(&self) -> &VecDeque<ContinuousHV> {
        &self.phonological_buffer
    }

    // =========================================================================
    // Visuospatial Sketchpad - Visual/Spatial Information (Baddeley's Model)
    // =========================================================================

    /// Add item to visuospatial sketchpad (visual/spatial information)
    ///
    /// The visuospatial sketchpad handles visual imagery and spatial
    /// relationships. It supports mental imagery and spatial reasoning.
    pub fn add_to_visuospatial(&mut self, content: ContinuousHV) {
        if self.visuospatial_buffer.len() >= self.capacity {
            self.visuospatial_buffer.pop_front();
        }
        self.visuospatial_buffer.push_back(content);
    }

    /// Manipulate visuospatial contents (mental rotation, transformation)
    ///
    /// Applies a transformation vector to all items in the sketchpad.
    /// This models mental manipulation of visual/spatial representations.
    pub fn transform_visuospatial(&mut self, transformation: &ContinuousHV) {
        for item in self.visuospatial_buffer.iter_mut() {
            *item = item.bind(transformation);
        }
    }

    /// Get combined visuospatial representation
    ///
    /// Returns a single vector representing the current spatial scene.
    pub fn visuospatial_scene(&self) -> Option<ContinuousHV> {
        if self.visuospatial_buffer.is_empty() {
            return None;
        }
        let owned: Vec<ContinuousHV> = self.visuospatial_buffer.iter().cloned().collect();
        Some(ContinuousHV::bundle_owned(&owned))
    }

    /// Get visuospatial buffer contents
    pub fn visuospatial_contents(&self) -> &VecDeque<ContinuousHV> {
        &self.visuospatial_buffer
    }

    // =========================================================================
    // Integration Methods (Central Executive Coordination)
    // =========================================================================

    /// Get total working memory utilization across all buffers
    pub fn total_utilization(&self) -> f64 {
        let phonological_load = self.phonological_buffer.len() as f64 / self.capacity as f64;
        let visuospatial_load = self.visuospatial_buffer.len() as f64 / self.capacity as f64;
        let episodic_load = self.episodic_buffer.len() as f64 / self.capacity as f64;

        // Weighted average with episodic buffer weighted more (it's the integration hub)
        (phonological_load + visuospatial_load + episodic_load * 2.0) / 4.0
    }

    /// Integrate contents from all buffers into episodic buffer
    ///
    /// The episodic buffer serves as the integration hub that combines
    /// information from the phonological and visuospatial subsystems.
    pub fn integrate_to_episodic(&mut self, timestamp: usize) {
        // Bundle phonological contents
        if let Some(phonological) = self.rehearse_phonological() {
            self.add_to_episodic(
                phonological,
                MemorySource::InternalGeneration,
                0.5, // Moderate goal relevance
                timestamp,
            );
        }

        // Bundle visuospatial contents
        if let Some(visuospatial) = self.visuospatial_scene() {
            self.add_to_episodic(
                visuospatial,
                MemorySource::InternalGeneration,
                0.5, // Moderate goal relevance
                timestamp,
            );
        }
    }
}