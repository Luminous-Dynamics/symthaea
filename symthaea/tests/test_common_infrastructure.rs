// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tests for the shared test infrastructure
//!
//! Verifies that the common test utilities work correctly.

mod common;

use common::prelude::*;
use symthaea::databases::MemoryType;

#[test]
fn test_memory_builder_works() {
    let record = MemoryRecordBuilder::new("infra-test-1")
        .with_seed(999)
        .with_type(MemoryType::Episodic)
        .with_psi(0.85)
        .with_content("Infrastructure test content")
        .build();

    assert_eq!(record.id, "infra-test-1");
    assert_eq!(record.memory_type, MemoryType::Episodic);
    assert!((record.psi - 0.85).abs() < 0.001);
}

#[test]
fn test_create_test_memory_shorthand() {
    let record = create_test_memory("quick-test", 123, MemoryType::Working);

    assert_eq!(record.id, "quick-test");
    assert_eq!(record.memory_type, MemoryType::Working);
}

#[test]
fn test_memory_batch_creation() {
    let batch = create_memory_batch("batch", 500, 5, MemoryType::Semantic);

    assert_eq!(batch.len(), 5);
    for (i, record) in batch.iter().enumerate() {
        assert_eq!(record.id, format!("batch-{}", i));
        assert_eq!(record.memory_type, MemoryType::Semantic);
    }
}

#[test]
fn test_temp_test_dir() {
    let temp = TempTestDir::new().expect("Should create temp dir");

    let file_path = temp
        .write_file("test.txt", "Hello, test!")
        .expect("Should write file");
    assert!(file_path.exists());

    let content = std::fs::read_to_string(&file_path).expect("Should read file");
    assert_eq!(content, "Hello, test!");
}

#[test]
fn test_unique_socket_paths() {
    let path1 = unique_socket_path();
    let path2 = unique_socket_path();
    let path3 = unique_socket_path();

    assert_ne!(path1, path2);
    assert_ne!(path2, path3);
    assert_ne!(path1, path3);
}

#[test]
fn test_temp_socket_creates_unique_path() {
    let socket1 = TempSocket::new();
    let socket2 = TempSocket::new();

    assert_ne!(socket1.path(), socket2.path());
}

#[test]
fn test_call_recorder() {
    let recorder = CallRecorder::new();

    recorder.record("init");
    recorder.record_with_args("process", "arg1, arg2");
    recorder.record("cleanup");

    assert!(recorder.was_called("init"));
    assert!(recorder.was_called("process"));
    assert!(recorder.was_called("cleanup"));
    assert!(!recorder.was_called("unknown"));

    assert_eq!(recorder.call_count("init"), 1);
    assert_eq!(recorder.calls().len(), 3);
}

#[test]
fn test_fake_clock() {
    let clock = FakeClock::new(1000);

    assert_eq!(clock.now_ms(), 1000);

    clock.advance_ms(500);
    assert_eq!(clock.now_ms(), 1500);

    clock.set_time_ms(3000);
    assert_eq!(clock.now_ms(), 3000);
}

#[test]
fn test_float_assertions() {
    assert_f32_eq(0.333, 0.334, 0.01, "Should be close enough");
    assert_f64_eq(0.666666, 0.666667, 0.0001, "Should be close enough");
}

#[test]
fn test_collection_assertions() {
    let non_empty = vec![1, 2, 3];
    let empty: Vec<i32> = vec![];

    assert_vec_len(&non_empty, 3, "Should have 3 items");
    assert_not_empty(&non_empty, "Should not be empty");
    assert_empty(&empty, "Should be empty");
}

#[test]
fn test_result_assertions() {
    let ok_result: Result<i32, &str> = Ok(42);
    let err_result: Result<i32, &str> = Err("error");

    let value = assert_ok(ok_result, "Should be Ok");
    assert_eq!(value, 42);

    assert_err(err_result, "Should be Err");
}

#[test]
fn test_option_assertions() {
    let some_val = Some(42);
    let none_val: Option<i32> = None;

    let value = assert_some(some_val, "Should be Some");
    assert_eq!(value, 42);

    assert_none(none_val, "Should be None");
}

#[test]
fn test_timing_assertion() {
    let result = assert_completes_within_ms(
        || {
            // Simple fast operation
            let mut sum = 0;
            for i in 0..1000 {
                sum += i;
            }
            sum
        },
        1000, // Should complete way under 1 second
        "Fast operation",
    );

    assert!(result > 0);
}

#[test]
fn test_test_timer() {
    let timer = TestTimer::start("timer_test");
    std::thread::sleep(std::time::Duration::from_millis(5));
    let elapsed = timer.elapsed_ms();
    assert!(elapsed >= 5, "Should have measured at least 5ms");
}