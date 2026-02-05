//! # Exploration Strategies for Symthaea
//!
//! This module provides various exploration strategies based on the Free Energy Principle (FEP).
//!
//! ## Modules
//!
//! - [`surprise_driven`]: Surprise-driven exploration using prediction errors
//!
//! ## Overview
//!
//! Exploration in Symthaea is guided by the Free Energy Principle, which posits that
//! biological systems minimize surprise (prediction errors). When predictions consistently
//! fail, the system triggers exploration to find states where predictions are more accurate.
//!
//! ## Key Concepts
//!
//! - **Surprise**: Measured as prediction error (MSE or KL divergence)
//! - **Adaptive Threshold**: Dynamically adjusts based on running statistics
//! - **Exploration Actions**: Perturbations to current state when surprise is high
//! - **Success Tracking**: Monitors which explorations successfully reduce surprise
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::exploration::{SurpriseTracker, SurpriseTrackerConfig};
//!
//! // Create tracker with default config
//! let mut tracker = SurpriseTracker::new(SurpriseTrackerConfig::default());
//!
//! // In cognitive loop:
//! let surprise = tracker.compute_surprise(&predicted, &actual);
//! tracker.record_surprise(surprise);
//!
//! if tracker.should_explore(surprise) {
//!     let action = tracker.generate_exploration_action(&current_state);
//!     let new_state = tracker.apply_exploration(&current_state, &action);
//!     // Use new_state...
//! }
//! ```

pub mod surprise_driven;

// Re-export main types for convenience
pub use surprise_driven::{
    ExplorationEvent, SurpriseExplorationBridge, SurpriseStats, SurpriseTracker,
    SurpriseTrackerConfig, SurpriseTrackerSummary,
};
