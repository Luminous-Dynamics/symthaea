//! Lightweight working memory for benchmark tasks.
//!
//! A minimal reimplementation of Symthaea's `ContinuousMind` working memory
//! semantics: FIFO eviction at capacity, tick counting, and eviction tracking.
//! This avoids depending on the full symthaea crate.

use symthaea_core::hdc::ContinuousHV;

/// Configuration for the lightweight working memory.
#[derive(Debug, Clone)]
pub struct WmConfig {
    /// HDC dimension.
    pub dimension: usize,
    /// Maximum items in working memory.
    pub capacity: usize,
}

impl Default for WmConfig {
    fn default() -> Self {
        Self {
            dimension: 512,
            capacity: 7,
        }
    }
}

/// Lightweight working memory with FIFO eviction.
///
/// Mirrors `ContinuousMind`'s working memory behavior:
/// items enter via `perceive()`, oldest items are evicted when capacity
/// is exceeded, and evicted items are tracked for downstream processing.
#[derive(Debug)]
pub struct WorkingMemory {
    /// Working memory contents.
    items: Vec<ContinuousHV>,
    /// Arrival tick for each item (parallel array).
    arrival_ticks: Vec<u64>,
    /// Current tick counter.
    tick: u64,
    /// Configuration.
    config: WmConfig,
    /// Items evicted since last `take_evicted()` call.
    evicted: Vec<(ContinuousHV, u64)>,
}

impl WorkingMemory {
    /// Create a new working memory.
    pub fn new(config: WmConfig) -> Self {
        Self {
            items: Vec::new(),
            arrival_ticks: Vec::new(),
            tick: 0,
            config,
            evicted: Vec::new(),
        }
    }

    /// Add an item to working memory, evicting the oldest if at capacity.
    pub fn perceive(&mut self, hv: ContinuousHV) {
        while self.items.len() >= self.config.capacity {
            let evicted_hv = self.items.remove(0);
            let arrival = self.arrival_ticks.remove(0);
            let steps_survived = self.tick.saturating_sub(arrival);
            self.evicted.push((evicted_hv, steps_survived));
        }
        self.items.push(hv);
        self.arrival_ticks.push(self.tick);
    }

    /// Advance one tick (no processing, just increment counter).
    pub fn tick(&mut self) {
        self.tick += 1;
    }

    /// Get current working memory contents.
    pub fn contents(&self) -> &[ContinuousHV] {
        &self.items
    }

    /// Current tick count.
    pub fn current_tick(&self) -> u64 {
        self.tick
    }

    /// Drain evicted items since last call.
    pub fn take_evicted(&mut self) -> Vec<(ContinuousHV, u64)> {
        std::mem::take(&mut self.evicted)
    }

    /// Number of items currently in WM.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether WM is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fifo_eviction() {
        let mut wm = WorkingMemory::new(WmConfig {
            dimension: 64,
            capacity: 3,
        });

        for i in 0..5 {
            wm.perceive(ContinuousHV::random(64, i + 1));
            wm.tick();
        }

        assert_eq!(wm.len(), 3);
        let evicted = wm.take_evicted();
        assert_eq!(evicted.len(), 2);
    }

    #[test]
    fn test_empty_wm() {
        let wm = WorkingMemory::new(WmConfig::default());
        assert!(wm.is_empty());
        assert_eq!(wm.len(), 0);
    }

    #[test]
    fn test_eviction_tracking() {
        let mut wm = WorkingMemory::new(WmConfig {
            dimension: 64,
            capacity: 2,
        });

        wm.perceive(ContinuousHV::random(64, 1));
        wm.tick();
        wm.tick();
        wm.perceive(ContinuousHV::random(64, 2));
        wm.tick();
        // This should evict the first item
        wm.perceive(ContinuousHV::random(64, 3));

        let evicted = wm.take_evicted();
        assert_eq!(evicted.len(), 1);
        // First item was added at tick 0, evicted at tick 3 → survived 3 ticks
        assert_eq!(evicted[0].1, 3);
    }
}
