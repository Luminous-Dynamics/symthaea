//! Lightweight working memory — self-contained FIFO with activation decay.
//!
//! Activation starts at 1.0 on `perceive()` and decays by
//! `decay_rate` (default 0.95) each `tick()`. This produces
//! a measurable recency bias matching human working memory.

use symthaea_core::hdc::ContinuousHV;

/// Default activation decay rate per tick.
const DEFAULT_DECAY_RATE: f32 = 0.95;

/// Configuration for the lightweight working memory.
#[derive(Debug, Clone)]
pub struct WmConfig {
    /// HDC dimension.
    pub dimension: usize,
    /// Maximum items in working memory.
    pub capacity: usize,
    /// Activation decay rate per tick (default 0.95).
    pub decay_rate: f32,
}

impl Default for WmConfig {
    fn default() -> Self {
        Self {
            dimension: 512,
            capacity: 7,
            decay_rate: DEFAULT_DECAY_RATE,
        }
    }
}

/// Lightweight working memory with FIFO eviction and activation decay.
///
/// Mirrors `ContinuousMind`'s working memory behavior:
/// items enter via `perceive()`, oldest items are evicted when capacity
/// is exceeded, and evicted items are tracked for downstream processing.
///
/// Each item tracks an activation level (initially 1.0) that decays
/// multiplicatively each tick. Use `activation_weighted_similarity()`
/// or `activations()` to incorporate recency into retrieval.
#[derive(Debug)]
pub struct WorkingMemory {
    items: Vec<ContinuousHV>,
    arrival_ticks: Vec<u64>,
    activations: Vec<f32>,
    tick: u64,
    config: WmConfig,
    evicted: Vec<(ContinuousHV, u64)>,
}

impl WorkingMemory {
    /// Create a new working memory.
    pub fn new(config: WmConfig) -> Self {
        Self {
            items: Vec::new(),
            arrival_ticks: Vec::new(),
            activations: Vec::new(),
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
            self.activations.remove(0);
            let steps_survived = self.tick.saturating_sub(arrival);
            self.evicted.push((evicted_hv, steps_survived));
        }
        self.items.push(hv);
        self.arrival_ticks.push(self.tick);
        self.activations.push(1.0);
    }

    /// Advance one tick: decay all activations.
    pub fn tick(&mut self) {
        self.tick += 1;
        let rate = self.config.decay_rate;
        for act in &mut self.activations {
            *act *= rate;
        }
    }

    /// Get current working memory contents.
    pub fn contents(&self) -> &[ContinuousHV] {
        &self.items
    }

    /// Get activation levels for each item (parallel to `contents()`).
    pub fn activations(&self) -> &[f32] {
        &self.activations
    }

    /// Compute the maximum activation-weighted similarity to `probe`.
    ///
    /// For each item, computes `cosine_similarity(item, probe) * activation`.
    /// Returns the maximum across all items, or 0.0 if WM is empty.
    pub fn activation_weighted_similarity(&self, probe: &ContinuousHV) -> f32 {
        self.items
            .iter()
            .zip(&self.activations)
            .map(|(item, &act)| item.similarity(probe) * act)
            .fold(0.0f32, f32::max)
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
            ..Default::default()
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
            ..Default::default()
        });

        wm.perceive(ContinuousHV::random(64, 1));
        wm.tick();
        wm.tick();
        wm.perceive(ContinuousHV::random(64, 2));
        wm.tick();
        wm.perceive(ContinuousHV::random(64, 3));

        let evicted = wm.take_evicted();
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].1, 3);
    }

    #[test]
    fn test_activation_decay() {
        let mut wm = WorkingMemory::new(WmConfig {
            dimension: 64,
            capacity: 7,
            decay_rate: 0.9,
        });

        let hv = ContinuousHV::random(64, 42);
        wm.perceive(hv);
        assert!((wm.activations()[0] - 1.0).abs() < 1e-6);

        wm.tick();
        assert!((wm.activations()[0] - 0.9).abs() < 1e-6);

        wm.tick();
        assert!((wm.activations()[0] - 0.81).abs() < 1e-4);
    }

    #[test]
    fn test_recency_weighted_similarity() {
        let mut wm = WorkingMemory::new(WmConfig {
            dimension: 256,
            capacity: 7,
            ..Default::default()
        });

        let older = ContinuousHV::random(256, 1);
        let newer = ContinuousHV::random(256, 2);

        wm.perceive(older.clone());
        for _ in 0..5 {
            wm.tick();
        }
        wm.perceive(newer.clone());
        wm.tick();

        let older_ws = wm.activation_weighted_similarity(&older);
        let newer_ws = wm.activation_weighted_similarity(&newer);
        assert!(
            newer_ws > older_ws,
            "newer ({}) should beat older ({}) in weighted similarity",
            newer_ws,
            older_ws
        );
    }
}
