/*!
 * Null Observer - No-op implementation for production use
 *
 * Use this when you don't want observability overhead.
 */

use super::{SymthaeaObserver, types::*, ObserverStats};
use anyhow::Result;

/// Observer that does nothing (zero overhead)
///
/// Use in production when observability is not needed.
/// All methods are no-ops that compile to nothing.
#[derive(Debug, Clone, Default)]
pub struct NullObserver;

impl NullObserver {
    pub fn new() -> Self {
        Self
    }
}

impl SymthaeaObserver for NullObserver {
    // All default implementations (no-ops)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_null_observer() {
        let mut observer = NullObserver::new();

        // All operations should succeed as no-ops
        assert!(observer.record_router_selection(RouterSelectionEvent::default()).is_ok());
        assert!(observer.record_phi_measurement(PhiMeasurementEvent::default()).is_ok());
        assert!(observer.finalize().is_ok());

        // Stats should be default
        let stats = observer.get_stats().unwrap();
        assert_eq!(stats.total_events, 0);
    }
}
