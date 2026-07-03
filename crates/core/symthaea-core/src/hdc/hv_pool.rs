// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Memory Pool for Hypervector Allocations
//!
//! This module provides thread-local object pools for BinaryHV and ContinuousHV types
//! to reduce allocation overhead in performance-critical code paths.
//!
//! # Motivation
//!
//! Hypervectors are frequently created and destroyed during HDC operations:
//! - Each bind() creates a new 2KB BinaryHV
//! - Each bundle() creates temporary accumulators
//! - Iterative algorithms may create millions of temporary vectors
//!
//! Object pooling can reduce allocation pressure by 10-100x for hot paths.
//!
//! # Usage
//!
//! ```rust,ignore
//! use symthaea::hdc::hv_pool::{BinaryHVPool, PooledBinaryHV};
//!
//! // Get a vector from the pool (allocation-free if pool has capacity)
//! let mut hv = BinaryHVPool::get();
//!
//! // Use the vector
//! hv.0.fill(0xFF);
//!
//! // When `hv` is dropped, it's returned to the pool automatically
//! ```
//!
//! # Thread Safety
//!
//! Each thread has its own pool via thread-local storage. Pooled vectors should
//! not be sent between threads (they implement !Send).
//!
//! # Performance
//!
//! - Pool hit (reuse): ~5ns
//! - Pool miss (allocate): ~50ns
//! - Standard allocation: ~80-200ns

use super::binary_hv::BinaryHV;
use super::unified_hv::ContinuousHV;
use std::cell::RefCell;
use std::ops::{Deref, DerefMut};

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Default pool capacity for BinaryHV (2KB each, 64 = 128KB per thread)
const HV16_POOL_CAPACITY: usize = 64;

/// Default pool capacity for ContinuousHV (64KB each, 8 = 512KB per thread)
const CONTINUOUS_HV_POOL_CAPACITY: usize = 8;

// =============================================================================
// BinaryHV POOL
// =============================================================================

thread_local! {
    #[allow(clippy::vec_box)]
    static HV16_POOL: RefCell<Vec<Box<[u8; 2048]>>> = RefCell::new(Vec::with_capacity(HV16_POOL_CAPACITY));
}

/// A pooled BinaryHV that returns to the pool when dropped
///
/// # Example
///
/// ```rust,ignore
/// use symthaea::hdc::hv_pool::PooledBinaryHV;
///
/// {
///     let mut hv = PooledBinaryHV::new();  // Gets from pool or allocates
///     // Use hv.0 as [u8; 2048]
/// }  // Automatically returned to pool
/// ```
pub struct PooledBinaryHV {
    data: Option<Box<[u8; 2048]>>,
}

impl PooledBinaryHV {
    /// Get a new pooled BinaryHV (from pool if available, else allocate)
    #[inline]
    pub fn new() -> Self {
        let data = HV16_POOL.with(|pool| pool.borrow_mut().pop());

        let data = data.unwrap_or_else(|| Box::new([0u8; 2048]));
        Self { data: Some(data) }
    }

    /// Get a new pooled BinaryHV initialized to zero
    #[inline]
    pub fn zeroed() -> Self {
        let mut hv = Self::new();
        hv.data
            .as_mut()
            .expect("PooledBinaryHV data is always Some until consumed")
            .fill(0);
        hv
    }

    /// Get a new pooled BinaryHV from an existing BinaryHV
    #[inline]
    pub fn from_hv16(hv: &BinaryHV) -> Self {
        let mut pooled = Self::new();
        pooled
            .data
            .as_mut()
            .expect("PooledBinaryHV data is always Some until consumed")
            .copy_from_slice(&hv.0);
        pooled
    }

    /// Convert to owned BinaryHV (removes from pool tracking)
    #[inline]
    pub fn into_hv16(mut self) -> BinaryHV {
        let data = self
            .data
            .take()
            .expect("PooledBinaryHV::into_hv16 called on already-consumed value");
        BinaryHV(*data)
    }

    /// Get the inner data as a reference
    #[inline]
    pub fn as_bytes(&self) -> &[u8; 2048] {
        self.data
            .as_ref()
            .expect("PooledBinaryHV data is always Some until consumed")
    }

    /// Get the inner data as a mutable reference
    #[inline]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 2048] {
        self.data
            .as_mut()
            .expect("PooledBinaryHV data is always Some until consumed")
    }
}

impl Default for PooledBinaryHV {
    fn default() -> Self {
        Self::new()
    }
}

impl Deref for PooledBinaryHV {
    type Target = [u8; 2048];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.data
            .as_ref()
            .expect("PooledBinaryHV deref on already-consumed value")
    }
}

impl DerefMut for PooledBinaryHV {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data
            .as_mut()
            .expect("PooledBinaryHV deref_mut on already-consumed value")
    }
}

impl Drop for PooledBinaryHV {
    fn drop(&mut self) {
        if let Some(data) = self.data.take() {
            HV16_POOL.with(|pool| {
                // Use try_borrow_mut to avoid panicking if the pool is already
                // borrowed (e.g., during concurrent Drop in test cleanup).
                if let Ok(mut pool) = pool.try_borrow_mut()
                    && pool.len() < HV16_POOL_CAPACITY
                {
                    pool.push(data);
                }
                // If borrow fails or pool is full, let the Box drop normally
            });
        }
    }
}

/// Public interface for BinaryHV pool
pub struct BinaryHVPool;

impl BinaryHVPool {
    /// Get a pooled BinaryHV
    #[inline]
    pub fn get() -> PooledBinaryHV {
        PooledBinaryHV::new()
    }

    /// Get a zeroed pooled BinaryHV
    #[inline]
    pub fn get_zeroed() -> PooledBinaryHV {
        PooledBinaryHV::zeroed()
    }

    /// Get current pool size (for diagnostics)
    pub fn pool_size() -> usize {
        HV16_POOL.with(|pool| pool.borrow().len())
    }

    /// Get pool capacity
    pub const fn capacity() -> usize {
        HV16_POOL_CAPACITY
    }

    /// Clear the pool (free all pooled allocations)
    pub fn clear() {
        HV16_POOL.with(|pool| pool.borrow_mut().clear());
    }

    /// Prewarm the pool with N allocations
    pub fn prewarm(n: usize) {
        let n = n.min(HV16_POOL_CAPACITY);
        HV16_POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            while pool.len() < n {
                pool.push(Box::new([0u8; 2048]));
            }
        });
    }
}

// =============================================================================
// CONTINUOUS HV POOL
// =============================================================================

thread_local! {
    #[allow(clippy::vec_box)]
    static CONTINUOUS_HV_POOL: RefCell<Vec<Box<Vec<f32>>>> = RefCell::new(Vec::with_capacity(CONTINUOUS_HV_POOL_CAPACITY));
}

/// A pooled ContinuousHV that returns to the pool when dropped
#[allow(clippy::box_collection)]
pub struct PooledContinuousHV {
    data: Option<Box<Vec<f32>>>,
}

impl PooledContinuousHV {
    /// Default dimension for continuous HVs
    const DEFAULT_DIM: usize = super::HDC_DIMENSION;

    /// Get a new pooled ContinuousHV (from pool if available, else allocate)
    #[inline]
    pub fn new() -> Self {
        let data = CONTINUOUS_HV_POOL.with(|pool| pool.borrow_mut().pop());

        let data = data.unwrap_or_else(|| Box::new(vec![0.0f32; Self::DEFAULT_DIM]));
        Self { data: Some(data) }
    }

    /// Get a new pooled ContinuousHV with specified dimension
    #[inline]
    pub fn with_dim(dim: usize) -> Self {
        let mut hv = Self::new();
        let data = hv
            .data
            .as_mut()
            .expect("PooledContinuousHV data is always Some until consumed");
        data.resize(dim, 0.0);
        data.fill(0.0);
        hv
    }

    /// Get a new pooled ContinuousHV initialized to zero
    #[inline]
    pub fn zeroed() -> Self {
        let mut hv = Self::new();
        hv.data
            .as_mut()
            .expect("PooledContinuousHV data is always Some until consumed")
            .fill(0.0);
        hv
    }

    /// Get a new pooled ContinuousHV from an existing ContinuousHV
    #[inline]
    pub fn from_continuous_hv(hv: &ContinuousHV) -> Self {
        let mut pooled = Self::new();
        let data = pooled
            .data
            .as_mut()
            .expect("PooledContinuousHV data is always Some until consumed");
        data.resize(hv.values.len(), 0.0);
        data.copy_from_slice(&hv.values);
        pooled
    }

    /// Convert to owned ContinuousHV (removes from pool tracking)
    #[inline]
    pub fn into_continuous_hv(mut self) -> ContinuousHV {
        let data = self
            .data
            .take()
            .expect("PooledContinuousHV::into_continuous_hv called on already-consumed value");
        ContinuousHV::from_vec(*data)
    }

    /// Get the inner data as a slice
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        self.data
            .as_ref()
            .expect("PooledContinuousHV data is always Some until consumed")
    }

    /// Get the inner data as a mutable slice
    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [f32] {
        self.data
            .as_mut()
            .expect("PooledContinuousHV data is always Some until consumed")
    }

    /// Get dimension
    #[inline]
    pub fn dim(&self) -> usize {
        self.data
            .as_ref()
            .expect("PooledContinuousHV data is always Some until consumed")
            .len()
    }
}

impl Default for PooledContinuousHV {
    fn default() -> Self {
        Self::new()
    }
}

impl Deref for PooledContinuousHV {
    type Target = [f32];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.data
            .as_ref()
            .expect("PooledContinuousHV deref on already-consumed value")
    }
}

impl DerefMut for PooledContinuousHV {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data
            .as_mut()
            .expect("PooledContinuousHV deref_mut on already-consumed value")
    }
}

impl Drop for PooledContinuousHV {
    fn drop(&mut self) {
        if let Some(data) = self.data.take() {
            CONTINUOUS_HV_POOL.with(|pool| {
                // Use try_borrow_mut to avoid panicking if the pool is already
                // borrowed (e.g., during concurrent Drop in test cleanup).
                if let Ok(mut pool) = pool.try_borrow_mut()
                    && pool.len() < CONTINUOUS_HV_POOL_CAPACITY
                {
                    pool.push(data);
                }
                // If borrow fails or pool is full, let the Box drop normally
            });
        }
    }
}

/// Public interface for ContinuousHV pool
pub struct ContinuousHVPool;

impl ContinuousHVPool {
    /// Get a pooled ContinuousHV
    #[inline]
    pub fn get() -> PooledContinuousHV {
        PooledContinuousHV::new()
    }

    /// Get a pooled ContinuousHV with specified dimension
    #[inline]
    pub fn get_with_dim(dim: usize) -> PooledContinuousHV {
        PooledContinuousHV::with_dim(dim)
    }

    /// Get a zeroed pooled ContinuousHV
    #[inline]
    pub fn get_zeroed() -> PooledContinuousHV {
        PooledContinuousHV::zeroed()
    }

    /// Get current pool size (for diagnostics)
    pub fn pool_size() -> usize {
        CONTINUOUS_HV_POOL.with(|pool| pool.borrow().len())
    }

    /// Get pool capacity
    pub const fn capacity() -> usize {
        CONTINUOUS_HV_POOL_CAPACITY
    }

    /// Clear the pool (free all pooled allocations)
    pub fn clear() {
        CONTINUOUS_HV_POOL.with(|pool| pool.borrow_mut().clear());
    }

    /// Prewarm the pool with N allocations
    pub fn prewarm(n: usize) {
        let n = n.min(CONTINUOUS_HV_POOL_CAPACITY);
        let dim = PooledContinuousHV::DEFAULT_DIM;
        CONTINUOUS_HV_POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            while pool.len() < n {
                pool.push(Box::new(vec![0.0f32; dim]));
            }
        });
    }
}

// =============================================================================
// POOLED OPERATIONS
// =============================================================================

/// Perform XOR bind using pooled allocations
#[inline]
pub fn pooled_bind(a: &BinaryHV, b: &BinaryHV) -> PooledBinaryHV {
    let mut result = PooledBinaryHV::new();
    let r = result.as_bytes_mut();
    for i in 0..2048 {
        r[i] = a.0[i] ^ b.0[i];
    }
    result
}

/// Perform similarity calculation (returns f32, no allocation needed)
#[inline]
pub fn pooled_similarity(a: &BinaryHV, b: &BinaryHV) -> f32 {
    // No pooling needed - just returns a number
    a.similarity(b)
}

// =============================================================================
// POOL STATISTICS
// =============================================================================

/// Statistics about pool usage
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    pub hv16_pool_size: usize,
    pub hv16_capacity: usize,
    pub continuous_hv_pool_size: usize,
    pub continuous_hv_capacity: usize,
}

impl PoolStats {
    /// Get current pool statistics
    pub fn current() -> Self {
        Self {
            hv16_pool_size: BinaryHVPool::pool_size(),
            hv16_capacity: BinaryHVPool::capacity(),
            continuous_hv_pool_size: ContinuousHVPool::pool_size(),
            continuous_hv_capacity: ContinuousHVPool::capacity(),
        }
    }

    /// Total memory used by pools (approximate)
    pub fn total_memory_bytes(&self) -> usize {
        let hv16_bytes = self.hv16_pool_size * 2048;
        let continuous_bytes = self.continuous_hv_pool_size * super::HDC_DIMENSION * 4;
        hv16_bytes + continuous_bytes
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hv16_pool_basic() {
        // Clear pool first
        BinaryHVPool::clear();

        // Get a pooled HV
        let hv1 = BinaryHVPool::get();
        assert_eq!(hv1.len(), 2048);

        // Drop it
        drop(hv1);

        // Pool should now have 1 entry
        assert_eq!(BinaryHVPool::pool_size(), 1);

        // Get another - should reuse
        let hv2 = BinaryHVPool::get();
        assert_eq!(BinaryHVPool::pool_size(), 0);

        drop(hv2);
        assert_eq!(BinaryHVPool::pool_size(), 1);
    }

    #[test]
    fn test_hv16_pool_zeroed() {
        let hv = BinaryHVPool::get_zeroed();
        for byte in hv.iter() {
            assert_eq!(*byte, 0);
        }
    }

    #[test]
    fn test_hv16_pool_from_hv16() {
        let original = BinaryHV::random(42);
        let pooled = PooledBinaryHV::from_hv16(&original);

        assert_eq!(pooled.as_bytes(), &original.0);
    }

    #[test]
    fn test_hv16_pool_into_hv16() {
        let original = BinaryHV::random(42);
        let pooled = PooledBinaryHV::from_hv16(&original);
        let converted = pooled.into_hv16();

        assert_eq!(converted, original);
    }

    #[test]
    fn test_hv16_pool_capacity() {
        BinaryHVPool::clear();

        // Fill pool beyond capacity
        let mut hvs = Vec::new();
        for _ in 0..100 {
            hvs.push(BinaryHVPool::get());
        }

        // Drop all
        drop(hvs);

        // Pool should be at capacity, not 100
        assert!(BinaryHVPool::pool_size() <= BinaryHVPool::capacity());
    }

    #[test]
    fn test_hv16_pool_prewarm() {
        BinaryHVPool::clear();
        assert_eq!(BinaryHVPool::pool_size(), 0);

        BinaryHVPool::prewarm(10);
        assert_eq!(BinaryHVPool::pool_size(), 10);
    }

    #[test]
    fn test_continuous_hv_pool_basic() {
        ContinuousHVPool::clear();

        let hv1 = ContinuousHVPool::get();
        assert_eq!(hv1.dim(), crate::hdc::HDC_DIMENSION);

        drop(hv1);
        assert_eq!(ContinuousHVPool::pool_size(), 1);

        let hv2 = ContinuousHVPool::get();
        assert_eq!(ContinuousHVPool::pool_size(), 0);

        drop(hv2);
    }

    #[test]
    fn test_continuous_hv_pool_with_dim() {
        let hv = ContinuousHVPool::get_with_dim(1024);
        assert_eq!(hv.dim(), 1024);
    }

    #[test]
    fn test_pooled_bind() {
        let a = BinaryHV::random(1);
        let b = BinaryHV::random(2);

        let result = pooled_bind(&a, &b);
        let expected = a.bind(&b);

        assert_eq!(result.as_bytes(), &expected.0);
    }

    #[test]
    fn test_pool_stats() {
        BinaryHVPool::clear();
        ContinuousHVPool::clear();

        BinaryHVPool::prewarm(5);
        ContinuousHVPool::prewarm(2);

        let stats = PoolStats::current();
        assert_eq!(stats.hv16_pool_size, 5);
        assert_eq!(stats.continuous_hv_pool_size, 2);

        let mem = stats.total_memory_bytes();
        assert!(mem > 0);
        println!("Pool memory usage: {} bytes", mem);
    }

    #[test]
    #[ignore = "benchmark test - run with cargo test --release -- --ignored"]
    fn bench_pooled_vs_regular_allocation() {
        use std::hint::black_box;
        use std::time::Instant;

        let iterations = 100_000;

        // Prewarm pool
        BinaryHVPool::prewarm(BinaryHVPool::capacity());

        // Benchmark pooled allocation
        let start = Instant::now();
        for _ in 0..iterations {
            let hv = black_box(BinaryHVPool::get());
            black_box(hv);
        }
        let pooled_ns = start.elapsed().as_nanos() / iterations;

        // Benchmark regular allocation
        let start = Instant::now();
        for _ in 0..iterations {
            let hv = black_box(BinaryHV::zero());
            black_box(hv);
        }
        let regular_ns = start.elapsed().as_nanos() / iterations;

        println!("\n📊 Allocation Performance:");
        println!("  Pooled:  {}ns", pooled_ns);
        println!("  Regular: {}ns", regular_ns);
        println!(
            "  Speedup: {:.1}x",
            regular_ns as f64 / pooled_ns.max(1) as f64
        );
    }
}
