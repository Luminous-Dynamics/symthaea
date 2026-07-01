#![allow(dead_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mock Implementations
//!
//! Mock servers, clients, and other test doubles for integration testing.

use parking_lot::Mutex;
use std::sync::Arc;

#[cfg(feature = "shell")]
use std::collections::VecDeque;
#[cfg(feature = "shell")]
use std::path::PathBuf;

// ============================================================================
// Mock IPC Server (for shell feature)
// ============================================================================

#[cfg(feature = "shell")]
pub mod ipc {
    use super::*;
    use symthaea::shell::ipc_client::{RequestEnvelope, Response, ResponseEnvelope};
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use tokio::net::UnixListener;

    /// Mock IPC server that responds with predefined responses
    ///
    /// # Example
    /// ```rust,ignore
    /// let responses = vec![
    ///     Response::Pong { timestamp_ms: 12345 },
    /// ];
    /// let server = MockIpcServer::new(socket_path, responses);
    /// tokio::spawn(server.run());
    /// ```
    pub struct MockIpcServer {
        socket_path: PathBuf,
        responses: Arc<Mutex<VecDeque<Response>>>,
    }

    impl MockIpcServer {
        /// Create a new mock server
        pub fn new(socket_path: PathBuf, responses: Vec<Response>) -> Self {
            Self {
                socket_path,
                responses: Arc::new(Mutex::new(responses.into())),
            }
        }

        /// Run the mock server (blocks until client disconnects)
        pub async fn run(self) {
            // Clean up any existing socket
            let _ = std::fs::remove_file(&self.socket_path);

            let listener = match UnixListener::bind(&self.socket_path) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("[MockServer] Failed to bind: {}", e);
                    return;
                }
            };

            if let Ok((stream, _)) = listener.accept().await {
                let (read_half, mut write_half) = stream.into_split();
                let mut reader = BufReader::new(read_half);

                loop {
                    let mut line = String::new();
                    match reader.read_line(&mut line).await {
                        Ok(0) => break, // EOF
                        Ok(_) => {
                            if let Ok(envelope) = serde_json::from_str::<RequestEnvelope>(&line) {
                                // Get next response or default Pong
                                let response =
                                    self.responses.lock().pop_front().unwrap_or(Response::Pong {
                                        timestamp_ms: std::time::SystemTime::now()
                                            .duration_since(std::time::UNIX_EPOCH)
                                            .unwrap()
                                            .as_millis()
                                            as u64,
                                    });

                                let response_envelope = ResponseEnvelope {
                                    id: envelope.id,
                                    latency_ms: Some(1),
                                    response,
                                };

                                let json =
                                    serde_json::to_string(&response_envelope).unwrap() + "\n";
                                if write_half.write_all(json.as_bytes()).await.is_err() {
                                    break;
                                }
                            }
                        }
                        Err(_) => break,
                    }
                }
            }

            // Cleanup
            let _ = std::fs::remove_file(&self.socket_path);
        }
    }

    /// Create a simple echo server that responds with Pong
    pub fn echo_server(socket_path: PathBuf) -> MockIpcServer {
        MockIpcServer::new(socket_path, vec![])
    }
}

// ============================================================================
// Mock Response Builders
// ============================================================================

#[cfg(feature = "shell")]
pub mod responses {
    use symthaea::shell::ipc_client::Response;

    /// Create a Pong response
    pub fn pong() -> Response {
        Response::Pong {
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        }
    }

    /// Create an Error response
    pub fn error(code: i32, message: &str) -> Response {
        Response::Error {
            code,
            message: message.to_string(),
        }
    }
}

// ============================================================================
// Mock Call Recorder
// ============================================================================

/// Records calls to mock objects for verification
#[derive(Default, Clone)]
pub struct CallRecorder {
    calls: Arc<Mutex<Vec<String>>>,
}

impl CallRecorder {
    /// Create a new call recorder
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a call
    pub fn record(&self, call: &str) {
        self.calls.lock().push(call.to_string());
    }

    /// Record a call with arguments
    pub fn record_with_args(&self, method: &str, args: &str) {
        self.calls.lock().push(format!("{}({})", method, args));
    }

    /// Get all recorded calls
    pub fn calls(&self) -> Vec<String> {
        self.calls.lock().clone()
    }

    /// Check if a specific call was made
    pub fn was_called(&self, method: &str) -> bool {
        self.calls.lock().iter().any(|c| c.starts_with(method))
    }

    /// Get the number of times a method was called
    pub fn call_count(&self, method: &str) -> usize {
        self.calls
            .lock()
            .iter()
            .filter(|c| c.starts_with(method))
            .count()
    }

    /// Clear recorded calls
    pub fn clear(&self) {
        self.calls.lock().clear();
    }

    /// Assert that a call was made
    pub fn assert_called(&self, method: &str) {
        assert!(
            self.was_called(method),
            "Expected {} to be called, but it wasn't. Calls: {:?}",
            method,
            self.calls()
        );
    }

    /// Assert call count
    pub fn assert_call_count(&self, method: &str, expected: usize) {
        let actual = self.call_count(method);
        assert_eq!(
            actual, expected,
            "Expected {} to be called {} times, but was called {} times",
            method, expected, actual
        );
    }
}

// ============================================================================
// Fake Clock
// ============================================================================

/// A fake clock for time-dependent tests
#[derive(Clone)]
pub struct FakeClock {
    current_time_ms: Arc<Mutex<u64>>,
}

impl FakeClock {
    /// Create a new fake clock starting at the given time
    pub fn new(start_time_ms: u64) -> Self {
        Self {
            current_time_ms: Arc::new(Mutex::new(start_time_ms)),
        }
    }

    /// Create a clock starting at Unix epoch + 1 billion ms (for realistic timestamps)
    pub fn default_test_time() -> Self {
        Self::new(1700000000000) // ~2023-11-14
    }

    /// Get current time
    pub fn now_ms(&self) -> u64 {
        *self.current_time_ms.lock()
    }

    /// Advance time by the given milliseconds
    pub fn advance_ms(&self, ms: u64) {
        *self.current_time_ms.lock() += ms;
    }

    /// Set the current time
    pub fn set_time_ms(&self, time_ms: u64) {
        *self.current_time_ms.lock() = time_ms;
    }
}

impl Default for FakeClock {
    fn default() -> Self {
        Self::default_test_time()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_call_recorder() {
        let recorder = CallRecorder::new();

        recorder.record("method1");
        recorder.record_with_args("method2", "arg1, arg2");

        assert!(recorder.was_called("method1"));
        assert!(recorder.was_called("method2"));
        assert!(!recorder.was_called("method3"));

        assert_eq!(recorder.call_count("method1"), 1);
    }

    #[test]
    fn test_fake_clock() {
        let clock = FakeClock::new(1000);

        assert_eq!(clock.now_ms(), 1000);

        clock.advance_ms(500);
        assert_eq!(clock.now_ms(), 1500);

        clock.set_time_ms(2000);
        assert_eq!(clock.now_ms(), 2000);
    }
}