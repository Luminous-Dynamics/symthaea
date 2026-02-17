//! Aggregation algorithms for federated learning.
//!
//! Provides FedAvg, Krum, Median, and TrimmedMean aggregation methods,
//! each delegating to mycelix-fl-core for the canonical implementation.

pub mod fedavg;
pub mod krum;
pub mod median;
pub mod trimmed_mean;

pub use fedavg::fedavg;
pub use krum::krum;
pub use median::coordinate_median;
pub use trimmed_mean::trimmed_mean;
