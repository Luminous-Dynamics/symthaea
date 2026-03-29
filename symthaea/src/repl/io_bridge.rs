// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! I/O Bridge - Input/Output abstraction for REPL
//!
//! Provides unified I/O interface for different input sources:
//! - stdin (interactive terminal)
//! - IPC (Unix socket from shell/GUI clients)
//! - Programmatic (library embedding)

use std::collections::VecDeque;
use std::io::{self, BufRead, BufReader, Write};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::Duration;

use anyhow::Result;
use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════════
// I/O EVENTS
// ═══════════════════════════════════════════════════════════════════════════════

/// Events flowing through the I/O bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IOEvent {
    /// User input received
    Input(String),
    /// Response to send
    Output(String),
    /// Consciousness metrics update
    MetricsUpdate(MetricsEvent),
    /// Action execution request
    ActionRequest(ActionEvent),
    /// Error occurred
    Error(String),
    /// Shutdown signal
    Shutdown,
}

/// Metrics update event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsEvent {
    pub phi: f32,
    pub coherence: f32,
    pub in_flow: bool,
    pub depth: String,
    pub valence: f32,
    pub arousal: f32,
}

/// Action execution event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionEvent {
    pub command: String,
    pub executed: bool,
    pub output: Option<String>,
    pub blocked_reason: Option<String>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// INPUT BRIDGE TRAIT
// ═══════════════════════════════════════════════════════════════════════════════

/// Trait for input sources
pub trait InputBridge: Send {
    /// Read next input (blocking)
    fn read(&mut self) -> Result<Option<String>>;

    /// Read with timeout
    fn read_timeout(&mut self, timeout: Duration) -> Result<Option<String>>;

    /// Check if more input is available
    fn has_input(&self) -> bool;

    /// Close the input source
    fn close(&mut self);
}

/// Standard input bridge
pub struct StdinBridge {
    reader: BufReader<io::Stdin>,
    closed: bool,
}

impl StdinBridge {
    pub fn new() -> Self {
        Self {
            reader: BufReader::new(io::stdin()),
            closed: false,
        }
    }
}

impl Default for StdinBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl InputBridge for StdinBridge {
    fn read(&mut self) -> Result<Option<String>> {
        if self.closed {
            return Ok(None);
        }

        let mut line = String::new();
        match self.reader.read_line(&mut line) {
            Ok(0) => Ok(None), // EOF
            Ok(_) => Ok(Some(line.trim().to_string())),
            Err(e) => Err(e.into()),
        }
    }

    fn read_timeout(&mut self, _timeout: Duration) -> Result<Option<String>> {
        // Standard stdin doesn't support timeout easily
        // In a full implementation, we'd use platform-specific APIs
        self.read()
    }

    fn has_input(&self) -> bool {
        !self.closed
    }

    fn close(&mut self) {
        self.closed = true;
    }
}

/// Channel-based input bridge (for testing/embedding)
pub struct ChannelInputBridge {
    receiver: Receiver<String>,
    buffer: VecDeque<String>,
}

impl ChannelInputBridge {
    pub fn new(receiver: Receiver<String>) -> Self {
        Self {
            receiver,
            buffer: VecDeque::new(),
        }
    }

    /// Create a paired sender/bridge
    pub fn create_pair() -> (Sender<String>, Self) {
        let (tx, rx) = mpsc::channel();
        (tx, Self::new(rx))
    }
}

impl InputBridge for ChannelInputBridge {
    fn read(&mut self) -> Result<Option<String>> {
        // Check buffer first
        if let Some(line) = self.buffer.pop_front() {
            return Ok(Some(line));
        }

        // Block on receiver
        match self.receiver.recv() {
            Ok(line) => Ok(Some(line)),
            Err(_) => Ok(None), // Channel closed
        }
    }

    fn read_timeout(&mut self, timeout: Duration) -> Result<Option<String>> {
        // Check buffer first
        if let Some(line) = self.buffer.pop_front() {
            return Ok(Some(line));
        }

        match self.receiver.recv_timeout(timeout) {
            Ok(line) => Ok(Some(line)),
            Err(mpsc::RecvTimeoutError::Timeout) => Ok(None),
            Err(mpsc::RecvTimeoutError::Disconnected) => Ok(None),
        }
    }

    fn has_input(&self) -> bool {
        !self.buffer.is_empty()
    }

    fn close(&mut self) {
        // Drain the receiver
        while self.receiver.try_recv().is_ok() {}
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// OUTPUT BRIDGE TRAIT
// ═══════════════════════════════════════════════════════════════════════════════

/// Trait for output destinations
pub trait OutputBridge: Send {
    /// Write output
    fn write(&mut self, output: &str) -> Result<()>;

    /// Write with newline
    fn writeln(&mut self, output: &str) -> Result<()>;

    /// Write metrics update
    fn write_metrics(&mut self, metrics: &MetricsEvent) -> Result<()>;

    /// Write error
    fn write_error(&mut self, error: &str) -> Result<()>;

    /// Flush output
    fn flush(&mut self) -> Result<()>;

    /// Close the output destination
    fn close(&mut self);
}

/// Standard output bridge
pub struct StdoutBridge {
    closed: bool,
}

impl StdoutBridge {
    pub fn new() -> Self {
        Self { closed: false }
    }
}

impl Default for StdoutBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputBridge for StdoutBridge {
    fn write(&mut self, output: &str) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        print!("{output}");
        Ok(())
    }

    fn writeln(&mut self, output: &str) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        println!("{output}");
        Ok(())
    }

    fn write_metrics(&mut self, _metrics: &MetricsEvent) -> Result<()> {
        // In terminal mode, we typically don't spam metrics
        // They're shown in the prompt or on request
        Ok(())
    }

    fn write_error(&mut self, error: &str) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        eprintln!("Error: {error}");
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        io::stdout().flush()?;
        Ok(())
    }

    fn close(&mut self) {
        self.closed = true;
    }
}

/// Channel-based output bridge (for testing/embedding)
pub struct ChannelOutputBridge {
    sender: Sender<IOEvent>,
}

impl ChannelOutputBridge {
    pub fn new(sender: Sender<IOEvent>) -> Self {
        Self { sender }
    }

    /// Create a paired bridge/receiver
    pub fn create_pair() -> (Self, Receiver<IOEvent>) {
        let (tx, rx) = mpsc::channel();
        (Self::new(tx), rx)
    }
}

impl OutputBridge for ChannelOutputBridge {
    fn write(&mut self, output: &str) -> Result<()> {
        let _ = self.sender.send(IOEvent::Output(output.to_string()));
        Ok(())
    }

    fn writeln(&mut self, output: &str) -> Result<()> {
        let _ = self.sender.send(IOEvent::Output(format!("{output}\n")));
        Ok(())
    }

    fn write_metrics(&mut self, metrics: &MetricsEvent) -> Result<()> {
        let _ = self.sender.send(IOEvent::MetricsUpdate(metrics.clone()));
        Ok(())
    }

    fn write_error(&mut self, error: &str) -> Result<()> {
        let _ = self.sender.send(IOEvent::Error(error.to_string()));
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        // Channels are unbuffered
        Ok(())
    }

    fn close(&mut self) {
        let _ = self.sender.send(IOEvent::Shutdown);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMBINED I/O BRIDGE
// ═══════════════════════════════════════════════════════════════════════════════

/// Combined input/output bridge
pub struct IOBridge {
    input: Box<dyn InputBridge>,
    output: Box<dyn OutputBridge>,
}

impl IOBridge {
    /// Create a new I/O bridge
    pub fn new(input: Box<dyn InputBridge>, output: Box<dyn OutputBridge>) -> Self {
        Self { input, output }
    }

    /// Create stdin/stdout bridge
    pub fn stdio() -> Self {
        Self {
            input: Box::new(StdinBridge::new()),
            output: Box::new(StdoutBridge::new()),
        }
    }

    /// Read input
    pub fn read(&mut self) -> Result<Option<String>> {
        self.input.read()
    }

    /// Read with timeout
    pub fn read_timeout(&mut self, timeout: Duration) -> Result<Option<String>> {
        self.input.read_timeout(timeout)
    }

    /// Write output
    pub fn write(&mut self, output: &str) -> Result<()> {
        self.output.write(output)
    }

    /// Write line
    pub fn writeln(&mut self, output: &str) -> Result<()> {
        self.output.writeln(output)
    }

    /// Write metrics
    pub fn write_metrics(&mut self, metrics: &MetricsEvent) -> Result<()> {
        self.output.write_metrics(metrics)
    }

    /// Write error
    pub fn write_error(&mut self, error: &str) -> Result<()> {
        self.output.write_error(error)
    }

    /// Flush
    pub fn flush(&mut self) -> Result<()> {
        self.output.flush()
    }

    /// Close bridge
    pub fn close(&mut self) {
        self.input.close();
        self.output.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_bridge() {
        let (tx, bridge) = ChannelInputBridge::create_pair();

        // Send input
        tx.send("hello".to_string()).unwrap();
        tx.send("world".to_string()).unwrap();

        // Read
        let mut bridge = bridge;
        assert_eq!(bridge.read().unwrap(), Some("hello".to_string()));
        assert_eq!(bridge.read().unwrap(), Some("world".to_string()));
    }

    #[test]
    fn test_output_bridge() {
        let (bridge, rx) = ChannelOutputBridge::create_pair();
        let mut bridge = bridge;

        bridge.writeln("test output").unwrap();

        match rx.recv().unwrap() {
            IOEvent::Output(s) => assert!(s.contains("test output")),
            _ => panic!("Expected Output event"),
        }
    }
}
