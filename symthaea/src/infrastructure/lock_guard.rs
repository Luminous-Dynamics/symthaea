// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Resilient Lock Guards
//!
//! Provides lock acquisition that recovers from poisoned mutexes instead of panicking.
//! This prevents cascading failures in the consciousness system when a single thread
//! panics while holding a lock.
//!
//! # Lock Types
//!
//! This module provides two approaches to safe locking:
//!
//! ## 1. `parking_lot` (Recommended for new code)
//!
//! The `parking_lot` crate provides mutexes that **never poison**. This is the
//! preferred approach for new code as it eliminates the problem entirely.
//!
//! ```rust
//! use symthaea::infrastructure::lock_guard::{FastMutex, FastRwLock};
//!
//! let mutex = FastMutex::new(42);
//! let guard = mutex.lock();  // Never panics, never poisons
//! ```
//!
//! ## 2. Resilient std::sync (For legacy compatibility)
//!
//! For existing code using `std::sync::Mutex`, the `ResilientMutex` trait
//! provides poison recovery:
//!
//! ```rust
//! use std::sync::Mutex;
//! use symthaea::infrastructure::lock_guard::ResilientMutex;
//!
//! let data = Mutex::new(42);
//! let guard = data.lock_resilient("state_update");  // Recovers from poison
//! ```
//!
//! # Performance
//!
//! `parking_lot` mutexes are 2-3x faster than `std::sync::Mutex` and have
//! a smaller memory footprint (1 byte vs 40+ bytes on Linux).

use std::sync::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};
use tracing::warn;

// ============================================================================
// PARKING_LOT RE-EXPORTS (Recommended for new code)
// ============================================================================

/// Fast, non-poisoning mutex from parking_lot.
///
/// Use this for new code instead of `std::sync::Mutex`.
/// Benefits:
/// - **Never poisons** (panics don't affect other threads)
/// - 2-3x faster than std::sync::Mutex
/// - Smaller memory footprint (1 byte vs 40+ bytes)
/// - Fair locking option available
pub use parking_lot::Mutex as FastMutex;

/// Fast, non-poisoning RwLock from parking_lot.
///
/// Use this for new code instead of `std::sync::RwLock`.
pub use parking_lot::RwLock as FastRwLock;

/// MutexGuard from parking_lot.
pub use parking_lot::MutexGuard as FastMutexGuard;

/// RwLockReadGuard from parking_lot.
pub use parking_lot::RwLockReadGuard as FastRwLockReadGuard;

/// RwLockWriteGuard from parking_lot.
pub use parking_lot::RwLockWriteGuard as FastRwLockWriteGuard;

/// Fair mutex that prevents starvation.
pub use parking_lot::FairMutex;

/// Condition variable compatible with parking_lot mutexes.
pub use parking_lot::Condvar as FastCondvar;

/// Once cell for lazy initialization.
pub use parking_lot::Once as FastOnce;

// ============================================================================
// RESILIENT STD::SYNC (For legacy compatibility)
// ============================================================================

/// Extension trait for resilient mutex locking.
///
/// Recovers from poisoned mutexes by extracting the inner data,
/// logging a warning, and continuing operation.
pub trait ResilientMutex<T> {
    /// Acquire the lock, recovering from poison if necessary.
    ///
    /// # Arguments
    /// * `context` - A string describing what operation is being protected,
    ///              used in warning logs when recovering from poison.
    ///
    /// # Returns
    /// A `MutexGuard` allowing access to the protected data.
    fn lock_resilient(&self, context: &str) -> MutexGuard<'_, T>;

    /// Try to acquire the lock, returning None if it would block.
    /// Recovers from poison if the lock was poisoned.
    fn try_lock_resilient(&self, context: &str) -> Option<MutexGuard<'_, T>>;
}

impl<T> ResilientMutex<T> for Mutex<T> {
    fn lock_resilient(&self, context: &str) -> MutexGuard<'_, T> {
        match self.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                // Log warning but recover the data
                warn!(
                    context = context,
                    "Mutex poisoned - recovering data. A previous thread panicked \
                     while holding this lock. System continuing in degraded state."
                );
                poisoned.into_inner()
            }
        }
    }

    fn try_lock_resilient(&self, context: &str) -> Option<MutexGuard<'_, T>> {
        match self.try_lock() {
            Ok(guard) => Some(guard),
            Err(std::sync::TryLockError::Poisoned(poisoned)) => {
                warn!(
                    context = context,
                    "Mutex poisoned (try_lock) - recovering data."
                );
                Some(poisoned.into_inner())
            }
            Err(std::sync::TryLockError::WouldBlock) => None,
        }
    }
}

/// Extension: lock_resilient with pain channel reporting.
///
/// When a lock poison is detected, reports the error through the
/// pain channel so the organism *feels* the infrastructure damage.
pub trait ResilientMutexWithPain<T> {
    /// Acquire the lock, recovering from poison and reporting pain.
    fn lock_with_pain(
        &self,
        context: &'static str,
        pain_tx: &super::somatic_error_bridge::PainSender,
    ) -> MutexGuard<'_, T>;
}

impl<T> ResilientMutexWithPain<T> for Mutex<T> {
    fn lock_with_pain(
        &self,
        context: &'static str,
        pain_tx: &super::somatic_error_bridge::PainSender,
    ) -> MutexGuard<'_, T> {
        match self.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                warn!(
                    context = context,
                    "Mutex poisoned - recovering data and reporting pain."
                );
                let _ = pain_tx.send(
                    super::somatic_error_bridge::InfrastructureError::LockPoisoned {
                        subsystem: context,
                    },
                );
                poisoned.into_inner()
            }
        }
    }
}

/// Extension trait for resilient RwLock locking.
pub trait ResilientRwLock<T> {
    /// Acquire read lock, recovering from poison if necessary.
    fn read_resilient(&self, context: &str) -> RwLockReadGuard<'_, T>;

    /// Acquire write lock, recovering from poison if necessary.
    fn write_resilient(&self, context: &str) -> RwLockWriteGuard<'_, T>;
}

impl<T> ResilientRwLock<T> for RwLock<T> {
    fn read_resilient(&self, context: &str) -> RwLockReadGuard<'_, T> {
        match self.read() {
            Ok(guard) => guard,
            Err(poisoned) => {
                warn!(
                    context = context,
                    lock_type = "read",
                    "RwLock poisoned - recovering data."
                );
                poisoned.into_inner()
            }
        }
    }

    fn write_resilient(&self, context: &str) -> RwLockWriteGuard<'_, T> {
        match self.write() {
            Ok(guard) => guard,
            Err(poisoned) => {
                warn!(
                    context = context,
                    lock_type = "write",
                    "RwLock poisoned - recovering data."
                );
                poisoned.into_inner()
            }
        }
    }
}

/// Macro for resilient lock acquisition with automatic context from location.
///
/// # Usage
///
/// ```rust,ignore
/// // Simple form - uses file:line as context
/// let guard = lock!(mutex);
///
/// // With custom context
/// let guard = lock!(mutex, "processing_input");
/// ```
#[macro_export]
macro_rules! lock {
    ($mutex:expr_2021) => {
        $crate::infrastructure::lock_guard::ResilientMutex::lock_resilient(
            &$mutex,
            concat!(file!(), ":", line!()),
        )
    };
    ($mutex:expr_2021, $context:expr_2021) => {
        $crate::infrastructure::lock_guard::ResilientMutex::lock_resilient(&$mutex, $context)
    };
}

/// Macro for resilient read lock acquisition.
#[macro_export]
macro_rules! read_lock {
    ($rwlock:expr_2021) => {
        $crate::infrastructure::lock_guard::ResilientRwLock::read_resilient(
            &$rwlock,
            concat!(file!(), ":", line!()),
        )
    };
    ($rwlock:expr_2021, $context:expr_2021) => {
        $crate::infrastructure::lock_guard::ResilientRwLock::read_resilient(&$rwlock, $context)
    };
}

/// Macro for resilient write lock acquisition.
#[macro_export]
macro_rules! write_lock {
    ($rwlock:expr_2021) => {
        $crate::infrastructure::lock_guard::ResilientRwLock::write_resilient(
            &$rwlock,
            concat!(file!(), ":", line!()),
        )
    };
    ($rwlock:expr_2021, $context:expr_2021) => {
        $crate::infrastructure::lock_guard::ResilientRwLock::write_resilient(&$rwlock, $context)
    };
}

/// Macro for lock acquisition with pain channel reporting.
///
/// # Usage
/// ```rust,ignore
/// let guard = lock_with_pain!(mutex, "sqlite_client", &pain_tx);
/// ```
#[macro_export]
macro_rules! lock_with_pain {
    ($mutex:expr_2021, $context:expr_2021, $pain_tx:expr_2021) => {
        $crate::infrastructure::lock_guard::ResilientMutexWithPain::lock_with_pain(
            &$mutex, $context, $pain_tx,
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_resilient_mutex_normal() {
        let mutex = Mutex::new(42);
        let guard = mutex.lock_resilient("test");
        assert_eq!(*guard, 42);
    }

    #[test]
    fn test_resilient_mutex_poisoned() {
        let mutex = Arc::new(Mutex::new(42));
        let mutex_clone = mutex.clone();

        // Spawn thread that will panic while holding lock
        let handle = thread::spawn(move || {
            let _guard = mutex_clone.lock().unwrap();
            panic!("intentional panic to poison mutex");
        });

        // Wait for thread to finish (and panic)
        let _ = handle.join();

        // Normal lock would panic here, but resilient recovers
        let guard = mutex.lock_resilient("recovery_test");
        assert_eq!(*guard, 42);
    }

    #[test]
    fn test_resilient_rwlock_normal() {
        let rwlock = RwLock::new("hello");
        let guard = rwlock.read_resilient("test");
        assert_eq!(*guard, "hello");
    }

    #[test]
    fn test_try_lock_resilient() {
        let mutex = Mutex::new(123);

        // Should succeed when lock is available
        {
            let guard = mutex.try_lock_resilient("test");
            assert!(guard.is_some());
            if let Some(g) = guard {
                assert_eq!(*g, 123);
            }
        }
    }

    #[test]
    fn test_lock_macro() {
        let mutex = Mutex::new(vec![1, 2, 3]);
        {
            let guard = lock!(mutex);
            assert_eq!(guard.len(), 3);
        }
        {
            let guard = lock!(mutex, "custom_context");
            assert_eq!(guard.len(), 3);
        }
    }
}
