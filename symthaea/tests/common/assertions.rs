#![allow(dead_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Custom Assertions
//!
//! Custom assertion macros and functions for common test patterns.

use symthaea::databases::MemoryRecord;
use symthaea::hdc::binary_hv::BinaryHV;

// ============================================================================
// Float Assertions
// ============================================================================

/// Assert two f32 values are approximately equal
pub fn assert_f32_eq(actual: f32, expected: f32, epsilon: f32, msg: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff < epsilon,
        "{}: expected {} but got {} (diff: {}, epsilon: {})",
        msg,
        expected,
        actual,
        diff,
        epsilon
    );
}

/// Assert two f64 values are approximately equal
pub fn assert_f64_eq(actual: f64, expected: f64, epsilon: f64, msg: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff < epsilon,
        "{}: expected {} but got {} (diff: {}, epsilon: {})",
        msg,
        expected,
        actual,
        diff,
        epsilon
    );
}

/// Default epsilon for f32 comparisons
pub const F32_EPSILON: f32 = 0.001;

/// Default epsilon for f64 comparisons
pub const F64_EPSILON: f64 = 0.0001;

// ============================================================================
// Memory Record Assertions
// ============================================================================

/// Assert two MemoryRecords are equal (with tolerance for floats)
pub fn assert_memory_eq(actual: &MemoryRecord, expected: &MemoryRecord) {
    assert_eq!(actual.id, expected.id, "ID mismatch");
    assert_eq!(actual.content, expected.content, "Content mismatch");
    assert_eq!(
        actual.memory_type, expected.memory_type,
        "Memory type mismatch"
    );
    assert_eq!(actual.topics, expected.topics, "Topics mismatch");
    assert_eq!(
        actual.timestamp_ms, expected.timestamp_ms,
        "Timestamp mismatch"
    );

    assert_f32_eq(actual.valence, expected.valence, F32_EPSILON, "Valence");
    assert_f32_eq(actual.arousal, expected.arousal, F32_EPSILON, "Arousal");
    assert_f64_eq(actual.psi, expected.psi, F64_EPSILON, "Phi");
}

/// Assert MemoryRecord IDs match
pub fn assert_memory_id(record: &MemoryRecord, expected_id: &str) {
    assert_eq!(
        record.id, expected_id,
        "Memory ID mismatch: expected '{}', got '{}'",
        expected_id, record.id
    );
}

/// Assert MemoryRecord has expected phi value
pub fn assert_memory_phi(record: &MemoryRecord, expected_phi: f64, epsilon: f64) {
    assert_f64_eq(
        record.psi,
        expected_phi,
        epsilon,
        &format!("Memory '{}' phi", record.id),
    );
}

// ============================================================================
// Encoding Assertions
// ============================================================================

/// Assert two BinaryHV encodings are identical
pub fn assert_encoding_eq(actual: &BinaryHV, expected: &BinaryHV, msg: &str) {
    assert_eq!(actual.0, expected.0, "{}: Encoding mismatch", msg);
}

/// Assert encoding roundtrip integrity
pub fn assert_encoding_roundtrip(original: &BinaryHV, recovered: &BinaryHV) {
    assert_encoding_eq(original, recovered, "Encoding roundtrip failed");
}

// ============================================================================
// Collection Assertions
// ============================================================================

/// Assert a vector has expected length
pub fn assert_vec_len<T>(vec: &[T], expected_len: usize, msg: &str) {
    assert_eq!(
        vec.len(),
        expected_len,
        "{}: expected {} items, got {}",
        msg,
        expected_len,
        vec.len()
    );
}

/// Assert a vector is not empty
pub fn assert_not_empty<T>(vec: &[T], msg: &str) {
    assert!(
        !vec.is_empty(),
        "{}: expected non-empty collection, got empty",
        msg
    );
}

/// Assert a vector is empty
pub fn assert_empty<T>(vec: &[T], msg: &str) {
    assert!(
        vec.is_empty(),
        "{}: expected empty collection, got {} items",
        msg,
        vec.len()
    );
}

// ============================================================================
// Result Assertions
// ============================================================================

/// Assert a Result is Ok and return the value
pub fn assert_ok<T, E: std::fmt::Debug>(result: Result<T, E>, msg: &str) -> T {
    match result {
        Ok(v) => v,
        Err(e) => panic!("{}: expected Ok, got Err({:?})", msg, e),
    }
}

/// Assert a Result is Err
pub fn assert_err<T: std::fmt::Debug, E>(result: Result<T, E>, msg: &str) {
    if let Ok(v) = result {
        panic!("{}: expected Err, got Ok({:?})", msg, v);
    }
}

/// Assert an Option is Some and return the value
pub fn assert_some<T>(option: Option<T>, msg: &str) -> T {
    match option {
        Some(v) => v,
        None => panic!("{}: expected Some, got None", msg),
    }
}

/// Assert an Option is None
pub fn assert_none<T: std::fmt::Debug>(option: Option<T>, msg: &str) {
    if let Some(v) = option {
        panic!("{}: expected None, got Some({:?})", msg, v);
    }
}

// ============================================================================
// Timing Assertions
// ============================================================================

/// Assert an operation completes within a time limit
pub fn assert_completes_within_ms<F, R>(f: F, max_ms: u128, msg: &str) -> R
where
    F: FnOnce() -> R,
{
    let start = std::time::Instant::now();
    let result = f();
    let elapsed = start.elapsed().as_millis();
    assert!(
        elapsed < max_ms,
        "{}: took {}ms, expected < {}ms",
        msg,
        elapsed,
        max_ms
    );
    result
}

/// Assert an async operation completes within a time limit
#[allow(dead_code)]
pub async fn assert_async_completes_within_ms<F, Fut, R>(f: F, max_ms: u128, msg: &str) -> R
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = R>,
{
    let start = std::time::Instant::now();
    let result = f().await;
    let elapsed = start.elapsed().as_millis();
    assert!(
        elapsed < max_ms,
        "{}: took {}ms, expected < {}ms",
        msg,
        elapsed,
        max_ms
    );
    result
}

// ============================================================================
// Assertion Macros
// ============================================================================

/// Assert approximate equality for floats
#[macro_export]
macro_rules! assert_approx_eq {
    ($actual:expr, $expected:expr, $epsilon:expr) => {
        let diff = ($actual - $expected).abs();
        assert!(
            diff < $epsilon,
            "assertion failed: {} !≈ {} (diff: {}, epsilon: {})",
            $actual,
            $expected,
            diff,
            $epsilon
        );
    };
    ($actual:expr, $expected:expr, $epsilon:expr, $($arg:tt)+) => {
        let diff = ($actual - $expected).abs();
        assert!(
            diff < $epsilon,
            "assertion failed: {} !≈ {} (diff: {}, epsilon: {}): {}",
            $actual,
            $expected,
            diff,
            $epsilon,
            format!($($arg)+)
        );
    };
}

/// Assert a value is within a range
#[macro_export]
macro_rules! assert_in_range {
    ($value:expr, $min:expr, $max:expr) => {
        assert!(
            $value >= $min && $value <= $max,
            "assertion failed: {} not in range [{}, {}]",
            $value,
            $min,
            $max
        );
    };
    ($value:expr, $min:expr, $max:expr, $($arg:tt)+) => {
        assert!(
            $value >= $min && $value <= $max,
            "assertion failed: {} not in range [{}, {}]: {}",
            $value,
            $min,
            $max,
            format!($($arg)+)
        );
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_eq() {
        assert_f32_eq(0.5, 0.5001, 0.001, "Should be approximately equal");
    }

    #[test]
    #[should_panic(expected = "expected")]
    fn test_f32_eq_fails() {
        assert_f32_eq(0.5, 0.6, 0.001, "Should fail");
    }

    #[test]
    fn test_vec_len() {
        let v = vec![1, 2, 3];
        assert_vec_len(&v, 3, "Vector length");
    }

    #[test]
    fn test_assert_ok() {
        let result: Result<i32, &str> = Ok(42);
        let value = assert_ok(result, "Should be Ok");
        assert_eq!(value, 42);
    }

    #[test]
    fn test_assert_some() {
        let opt = Some(42);
        let value = assert_some(opt, "Should be Some");
        assert_eq!(value, 42);
    }

    #[test]
    fn test_completes_within() {
        let result = assert_completes_within_ms(|| 42, 1000, "Simple operation");
        assert_eq!(result, 42);
    }

    #[test]
    fn test_approx_eq_macro() {
        assert_approx_eq!(0.5f64, 0.500001, 0.001);
    }

    #[test]
    fn test_in_range_macro() {
        assert_in_range!(5, 0, 10);
        assert_in_range!(0.5f64, 0.0, 1.0);
    }
}