// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hardware abstraction traits for EEG devices
//!
//! Defines the common interface that all EEG device adapters must implement.

use std::time::Duration;
use thiserror::Error;

/// Error types for device operations
#[derive(Debug, Error)]
pub enum DeviceError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Device not found at {0}")]
    DeviceNotFound(String),

    #[error("Communication error: {0}")]
    CommunicationError(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Device not streaming")]
    NotStreaming,

    #[error("Device already streaming")]
    AlreadyStreaming,

    #[error("Timeout after {0:?}")]
    Timeout(Duration),

    #[error("Bluetooth error: {0}")]
    BluetoothError(String),

    #[error("Serial port error: {0}")]
    SerialError(String),

    #[error("Unsupported operation: {0}")]
    Unsupported(String),
}

/// Device state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceState {
    /// Device is disconnected
    Disconnected,
    /// Device is connected but not streaming
    Connected,
    /// Device is actively streaming data
    Streaming,
    /// Device encountered an error
    Error,
}

/// Sample rate options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleRate {
    Hz125,
    Hz200,
    Hz250,
    Hz256,
    Hz500,
    Hz1000,
    Custom(u32),
}

impl SampleRate {
    /// Get the numeric value in Hz
    pub fn as_hz(&self) -> u32 {
        match self {
            SampleRate::Hz125 => 125,
            SampleRate::Hz200 => 200,
            SampleRate::Hz250 => 250,
            SampleRate::Hz256 => 256,
            SampleRate::Hz500 => 500,
            SampleRate::Hz1000 => 1000,
            SampleRate::Custom(hz) => *hz,
        }
    }

    /// Get as f32 for calculations
    pub fn as_f32(&self) -> f32 {
        self.as_hz() as f32
    }
}

/// Information about a single channel
#[derive(Debug, Clone)]
pub struct ChannelInfo {
    /// Channel index (0-based)
    pub index: usize,
    /// Channel name (e.g., "Fp1", "Fp2")
    pub name: String,
    /// Whether the channel is enabled
    pub enabled: bool,
    /// Gain setting
    pub gain: f32,
    /// Channel type (EEG, EMG, ECG, etc.)
    pub channel_type: ChannelType,
}

/// Type of channel
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelType {
    Eeg,
    Emg,
    Ecg,
    Eog,
    Accelerometer,
    Gyroscope,
    Auxiliary,
    Unknown,
}

/// Device information
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device name
    pub name: String,
    /// Device manufacturer
    pub manufacturer: String,
    /// Device model
    pub model: String,
    /// Firmware version (if available)
    pub firmware_version: Option<String>,
    /// Serial number (if available)
    pub serial_number: Option<String>,
    /// Number of EEG channels
    pub num_channels: usize,
    /// Supported sample rates
    pub supported_sample_rates: Vec<SampleRate>,
    /// Default sample rate
    pub default_sample_rate: SampleRate,
    /// Resolution in bits
    pub resolution_bits: u8,
    /// Input range in microvolts
    pub input_range_uv: f32,
}

/// Device configuration
#[derive(Debug, Clone)]
pub struct DeviceConfig {
    /// Sample rate
    pub sample_rate: SampleRate,
    /// Enabled channels
    pub enabled_channels: Vec<usize>,
    /// Buffer size (samples per channel per callback)
    pub buffer_size: usize,
    /// Enable bias (DRL) if available
    pub enable_bias: bool,
    /// Enable SRB1 if available
    pub enable_srb1: bool,
    /// Enable SRB2 if available
    pub enable_srb2: bool,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            sample_rate: SampleRate::Hz250,
            enabled_channels: vec![0, 1, 2, 3, 4, 5, 6, 7],
            buffer_size: 256,
            enable_bias: true,
            enable_srb1: true,
            enable_srb2: true,
        }
    }
}

/// A single sample from the EEG device
#[derive(Debug, Clone)]
pub struct EegSample {
    /// Timestamp in milliseconds since stream start
    pub timestamp_ms: f64,
    /// Sample index (monotonically increasing)
    pub sample_index: u64,
    /// Channel data in microvolts
    pub channels: Vec<f32>,
    /// Accelerometer data (if available) - [x, y, z]
    pub accel: Option<[f32; 3]>,
    /// Auxiliary data (if available)
    pub aux: Option<Vec<f32>>,
    /// Whether this sample has valid data
    pub valid: bool,
}

impl EegSample {
    /// Create a new sample with default values
    pub fn new(num_channels: usize) -> Self {
        Self {
            timestamp_ms: 0.0,
            sample_index: 0,
            channels: vec![0.0; num_channels],
            accel: None,
            aux: None,
            valid: true,
        }
    }

    /// Get a specific channel value
    pub fn get_channel(&self, index: usize) -> Option<f32> {
        self.channels.get(index).copied()
    }
}

/// Main trait for EEG devices
pub trait EegDevice: Send {
    /// Connect to the device
    fn connect(&mut self) -> Result<(), DeviceError>;

    /// Disconnect from the device
    fn disconnect(&mut self) -> Result<(), DeviceError>;

    /// Start streaming data
    fn start_streaming(&mut self) -> Result<(), DeviceError>;

    /// Stop streaming data
    fn stop_streaming(&mut self) -> Result<(), DeviceError>;

    /// Read a single sample (blocking)
    fn read_sample(&mut self) -> Result<Option<EegSample>, DeviceError>;

    /// Read multiple samples (blocking)
    fn read_samples(&mut self, num_samples: usize) -> Result<Vec<EegSample>, DeviceError> {
        let mut samples = Vec::with_capacity(num_samples);
        for _ in 0..num_samples {
            if let Some(sample) = self.read_sample()? {
                samples.push(sample);
            }
        }
        Ok(samples)
    }

    /// Get device information
    fn get_info(&self) -> &DeviceInfo;

    /// Get current configuration
    fn get_config(&self) -> &DeviceConfig;

    /// Set configuration
    fn set_config(&mut self, config: DeviceConfig) -> Result<(), DeviceError>;

    /// Get current device state
    fn get_state(&self) -> DeviceState;

    /// Check if device is streaming
    fn is_streaming(&self) -> bool {
        self.get_state() == DeviceState::Streaming
    }

    /// Check if device is connected
    fn is_connected(&self) -> bool {
        matches!(
            self.get_state(),
            DeviceState::Connected | DeviceState::Streaming
        )
    }

    /// Get channel information
    fn get_channels(&self) -> Vec<ChannelInfo>;

    /// Enable/disable a channel
    fn set_channel_enabled(&mut self, channel: usize, enabled: bool) -> Result<(), DeviceError>;

    /// Set channel gain
    fn set_channel_gain(&mut self, channel: usize, gain: f32) -> Result<(), DeviceError>;

    /// Get current sample rate
    fn get_sample_rate(&self) -> SampleRate {
        self.get_config().sample_rate
    }

    /// Set sample rate
    fn set_sample_rate(&mut self, rate: SampleRate) -> Result<(), DeviceError>;

    /// Get the number of channels
    fn num_channels(&self) -> usize {
        self.get_info().num_channels
    }

    /// Collect samples for a duration (convenience method)
    fn collect_duration(&mut self, duration: Duration) -> Result<Vec<f32>, DeviceError> {
        let sample_rate = self.get_sample_rate().as_f32();
        let num_samples = (duration.as_secs_f32() * sample_rate) as usize;

        let samples = self.read_samples(num_samples)?;

        // Flatten to single channel (channel 0) for simplicity
        // In practice, you'd want to handle multi-channel data differently
        Ok(samples
            .iter()
            .filter_map(|s| s.channels.first().copied())
            .collect())
    }
}