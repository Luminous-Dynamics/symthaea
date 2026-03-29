// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Segment management and caching

use crate::{Segment, StreamingResult};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Segment cache for efficient retrieval
pub struct SegmentCache {
    segments: RwLock<HashMap<String, Arc<Segment>>>,
    max_size: usize,
    current_size: RwLock<usize>,
}

impl SegmentCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            segments: RwLock::new(HashMap::new()),
            max_size,
            current_size: RwLock::new(0),
        }
    }

    pub async fn get(&self, key: &str) -> Option<Arc<Segment>> {
        let segments = self.segments.read().await;
        segments.get(key).cloned()
    }

    pub async fn put(&self, key: String, segment: Segment) -> StreamingResult<()> {
        let segment_size = segment.data.len();

        // Check if we need to evict
        let mut current = self.current_size.write().await;
        if *current + segment_size > self.max_size {
            // Simple eviction: clear everything (would use LRU in production)
            let mut segments = self.segments.write().await;
            segments.clear();
            *current = 0;
        }

        let mut segments = self.segments.write().await;
        segments.insert(key, Arc::new(segment));
        *current += segment_size;

        Ok(())
    }

    pub async fn remove(&self, key: &str) -> Option<Arc<Segment>> {
        let mut segments = self.segments.write().await;
        if let Some(segment) = segments.remove(key) {
            let mut current = self.current_size.write().await;
            *current = current.saturating_sub(segment.data.len());
            Some(segment)
        } else {
            None
        }
    }

    pub async fn clear(&self) {
        let mut segments = self.segments.write().await;
        segments.clear();
        let mut current = self.current_size.write().await;
        *current = 0;
    }

    pub async fn len(&self) -> usize {
        let segments = self.segments.read().await;
        segments.len()
    }

    pub async fn is_empty(&self) -> bool {
        self.len().await == 0
    }
}

/// Segment index for fast lookup
pub struct SegmentIndex {
    entries: RwLock<Vec<SegmentEntry>>,
}

#[derive(Clone)]
pub struct SegmentEntry {
    pub sequence: u32,
    pub time_offset: f64,
    pub duration: f32,
    pub key: String,
}

impl SegmentIndex {
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(Vec::new()),
        }
    }

    pub async fn add(&self, entry: SegmentEntry) {
        let mut entries = self.entries.write().await;
        entries.push(entry);
    }

    pub async fn find_by_time(&self, time: f64) -> Option<SegmentEntry> {
        let entries = self.entries.read().await;
        entries.iter().find(|e| {
            time >= e.time_offset && time < e.time_offset + e.duration as f64
        }).cloned()
    }

    pub async fn find_by_sequence(&self, sequence: u32) -> Option<SegmentEntry> {
        let entries = self.entries.read().await;
        entries.iter().find(|e| e.sequence == sequence).cloned()
    }

    pub async fn range(&self, start_time: f64, end_time: f64) -> Vec<SegmentEntry> {
        let entries = self.entries.read().await;
        entries.iter().filter(|e| {
            let segment_end = e.time_offset + e.duration as f64;
            e.time_offset < end_time && segment_end > start_time
        }).cloned().collect()
    }
}

impl Default for SegmentIndex {
    fn default() -> Self {
        Self::new()
    }
}
