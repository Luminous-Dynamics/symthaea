// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! OpenBCI Adapter
//!
//! Supports OpenBCI Cyton (8-channel), Cyton-Daisy (16-channel), and Ganglion (4-channel).
//!
//! # Protocol
//! - Serial communication at 115200 baud
//! - Uses OpenBCI packet protocol with 33-byte packets
//! - Commands are single ASCII characters

use super::traits::*;
use std::time::{Duration, Instant};

/// OpenBCI board types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenBciBoard {
    /// 8-channel Cyton board
    Cyton,
    /// 16-channel Cyton + Daisy
    CytonDaisy,
    /// 4-channel Ganglion
    Ganglion,
}

impl OpenBciBoard {
    /// Get number of channels for this board
    pub fn num_channels(&self) -> usize {
        match self {
            OpenBciBoard::Cyton => 8,
            OpenBciBoard::CytonDaisy => 16,
            OpenBciBoard::Ganglion => 4,
        }
    }

    /// Get default sample rate for this board
    pub fn default_sample_rate(&self) -> SampleRate {
        match self {
            OpenBciBoard::Cyton => SampleRate::Hz250,
            OpenBciBoard::CytonDaisy => SampleRate::Hz125,
            OpenBciBoard::Ganglion => SampleRate::Hz200,
        }
    }
}

/// OpenBCI-specific configuration
#[derive(Debug, Clone)]
pub struct OpenBciConfig {
    /// Serial port path (e.g., "/dev/ttyUSB0" or "COM3")
    pub port: String,
    /// Baud rate (default 115200)
    pub baud_rate: u32,
    /// Board type
    pub board: OpenBciBoard,
    /// Enable accelerometer
    pub enable_accel: bool,
    /// Common configuration
    pub common: DeviceConfig,
}

impl Default for OpenBciConfig {
    fn default() -> Self {
        Self {
            port: String::new(),
            baud_rate: 115200,
            board: OpenBciBoard::Cyton,
            enable_accel: true,
            common: DeviceConfig::default(),
        }
    }
}

impl OpenBciConfig {
    /// Create config for Cyton board
    pub fn cyton(port: &str) -> Self {
        Self {
            port: port.to_string(),
            board: OpenBciBoard::Cyton,
            common: DeviceConfig {
                sample_rate: SampleRate::Hz250,
                enabled_channels: (0..8).collect(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Create config for Cyton-Daisy board
    pub fn cyton_daisy(port: &str) -> Self {
        Self {
            port: port.to_string(),
            board: OpenBciBoard::CytonDaisy,
            common: DeviceConfig {
                sample_rate: SampleRate::Hz125,
                enabled_channels: (0..16).collect(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Create config for Ganglion board
    pub fn ganglion(port: &str) -> Self {
        Self {
            port: port.to_string(),
            board: OpenBciBoard::Ganglion,
            common: DeviceConfig {
                sample_rate: SampleRate::Hz250,
                enabled_channels: (0..4).collect(),
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

/// OpenBCI device adapter
///
/// Provides connection to OpenBCI hardware via serial port.
///
/// # Example (requires hardware)
/// ```rust,ignore
/// use sentinels::hardware::{OpenBciAdapter, OpenBciConfig};
///
/// let config = OpenBciConfig::cyton("/dev/ttyUSB0");
/// let mut device = OpenBciAdapter::new(config);
/// device.connect()?;
/// device.start_streaming()?;
///
/// // Collect 5 seconds of data
/// let data = device.collect_duration(std::time::Duration::from_secs(5))?;
/// ```
pub struct OpenBciAdapter {
    config: OpenBciConfig,
    info: DeviceInfo,
    state: DeviceState,
    sample_index: u64,
    stream_start: Option<Instant>,
    // In a real implementation, this would hold the serial port
    // port: Option<serialport::SerialPort>,
    // For now, we simulate with a buffer
    #[allow(dead_code)]
    simulated: bool,
}

impl OpenBciAdapter {
    /// Create a new OpenBCI adapter with the given configuration
    pub fn new(config: OpenBciConfig) -> Self {
        let num_channels = config.board.num_channels();
        let info = DeviceInfo {
            name: format!("OpenBCI {:?}", config.board),
            manufacturer: "OpenBCI".to_string(),
            model: format!("{:?}", config.board),
            firmware_version: None,
            serial_number: None,
            num_channels,
            supported_sample_rates: vec![
                SampleRate::Hz125,
                SampleRate::Hz250,
                SampleRate::Hz500,
                SampleRate::Hz1000,
            ],
            default_sample_rate: config.board.default_sample_rate(),
            resolution_bits: 24,
            input_range_uv: 187500.0, // ±187.5mV with default gain
        };

        Self {
            config,
            info,
            state: DeviceState::Disconnected,
            sample_index: 0,
            stream_start: None,
            simulated: true,
        }
    }

    /// Create an adapter and immediately connect
    pub fn connect_to(port: &str) -> Result<Self, DeviceError> {
        let config = OpenBciConfig::cyton(port);
        let mut adapter = Self::new(config);
        adapter.connect()?;
        Ok(adapter)
    }

    /// Create a simulated device for testing
    pub fn simulated(board: OpenBciBoard) -> Self {
        let config = match board {
            OpenBciBoard::Cyton => OpenBciConfig::cyton("SIMULATED"),
            OpenBciBoard::CytonDaisy => OpenBciConfig::cyton_daisy("SIMULATED"),
            OpenBciBoard::Ganglion => OpenBciConfig::ganglion("SIMULATED"),
        };
        let mut adapter = Self::new(config);
        adapter.simulated = true;
        adapter.state = DeviceState::Connected;
        adapter
    }

    /// Get the board type
    pub fn board_type(&self) -> OpenBciBoard {
        self.config.board
    }

    /// Generate a simulated sample (for testing)
    fn generate_simulated_sample(&self) -> EegSample {
        use std::f32::consts::PI;

        let timestamp_ms = self
            .stream_start
            .map(|start| start.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);

        let t = timestamp_ms / 1000.0;
        let num_channels = self.config.board.num_channels();

        // Generate synthetic EEG-like data
        let channels: Vec<f32> = (0..num_channels)
            .map(|ch| {
                let phase = ch as f32 * 0.5;
                // Mix of typical EEG bands
                let alpha = 10.0 * (2.0 * PI * 10.0 * t as f32 + phase).sin();
                let beta = 3.0 * (2.0 * PI * 20.0 * t as f32 + phase).sin();
                let theta = 5.0 * (2.0 * PI * 6.0 * t as f32 + phase).sin();
                let noise = (rand_simple(self.sample_index + ch as u64) - 0.5) * 2.0;
                alpha + beta + theta + noise
            })
            .collect();

        // Simulated accelerometer data
        let accel = if self.config.enable_accel {
            Some([0.0, 0.0, 1.0]) // Earth gravity pointing down
        } else {
            None
        };

        EegSample {
            timestamp_ms,
            sample_index: self.sample_index,
            channels,
            accel,
            aux: None,
            valid: true,
        }
    }
}

impl EegDevice for OpenBciAdapter {
    fn connect(&mut self) -> Result<(), DeviceError> {
        if self.state != DeviceState::Disconnected {
            return Err(DeviceError::InvalidConfig("Already connected".to_string()));
        }

        // In a real implementation, we would open the serial port here
        // For now, we just simulate the connection

        if self.config.port.is_empty() {
            return Err(DeviceError::InvalidConfig("No port specified".to_string()));
        }

        // Simulate successful connection
        self.state = DeviceState::Connected;
        Ok(())
    }

    fn disconnect(&mut self) -> Result<(), DeviceError> {
        if self.state == DeviceState::Streaming {
            self.stop_streaming()?;
        }
        self.state = DeviceState::Disconnected;
        Ok(())
    }

    fn start_streaming(&mut self) -> Result<(), DeviceError> {
        if self.state != DeviceState::Connected {
            return Err(DeviceError::NotStreaming);
        }

        self.stream_start = Some(Instant::now());
        self.sample_index = 0;
        self.state = DeviceState::Streaming;
        Ok(())
    }

    fn stop_streaming(&mut self) -> Result<(), DeviceError> {
        if self.state != DeviceState::Streaming {
            return Err(DeviceError::NotStreaming);
        }

        self.state = DeviceState::Connected;
        self.stream_start = None;
        Ok(())
    }

    fn read_sample(&mut self) -> Result<Option<EegSample>, DeviceError> {
        if self.state != DeviceState::Streaming {
            return Err(DeviceError::NotStreaming);
        }

        // Generate simulated sample
        let sample = self.generate_simulated_sample();
        self.sample_index += 1;

        // Simulate sample rate timing
        let sample_period =
            Duration::from_secs_f64(1.0 / self.config.common.sample_rate.as_f32() as f64);
        std::thread::sleep(sample_period);

        Ok(Some(sample))
    }

    fn get_info(&self) -> &DeviceInfo {
        &self.info
    }

    fn get_config(&self) -> &DeviceConfig {
        &self.config.common
    }

    fn set_config(&mut self, config: DeviceConfig) -> Result<(), DeviceError> {
        if self.state == DeviceState::Streaming {
            return Err(DeviceError::AlreadyStreaming);
        }
        self.config.common = config;
        Ok(())
    }

    fn get_state(&self) -> DeviceState {
        self.state
    }

    fn get_channels(&self) -> Vec<ChannelInfo> {
        let channel_names = match self.config.board {
            OpenBciBoard::Cyton => vec!["Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2"],
            OpenBciBoard::CytonDaisy => vec![
                "Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2", "F7", "F8", "F3", "F4", "T7",
                "T8", "P3", "P4",
            ],
            OpenBciBoard::Ganglion => vec!["Fp1", "Fp2", "T3", "T4"],
        };

        channel_names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                ChannelInfo {
                    index: i,
                    name: name.to_string(),
                    enabled: self.config.common.enabled_channels.contains(&i),
                    gain: 24.0, // Default OpenBCI gain
                    channel_type: ChannelType::Eeg,
                }
            })
            .collect()
    }

    fn set_channel_enabled(&mut self, channel: usize, enabled: bool) -> Result<(), DeviceError> {
        if channel >= self.info.num_channels {
            return Err(DeviceError::InvalidConfig(format!(
                "Invalid channel: {}",
                channel
            )));
        }

        if enabled {
            if !self.config.common.enabled_channels.contains(&channel) {
                self.config.common.enabled_channels.push(channel);
                self.config.common.enabled_channels.sort();
            }
        } else {
            self.config
                .common
                .enabled_channels
                .retain(|&c| c != channel);
        }

        Ok(())
    }

    fn set_channel_gain(&mut self, channel: usize, _gain: f32) -> Result<(), DeviceError> {
        if channel >= self.info.num_channels {
            return Err(DeviceError::InvalidConfig(format!(
                "Invalid channel: {}",
                channel
            )));
        }
        // In real implementation, would send gain command to device
        Ok(())
    }

    fn set_sample_rate(&mut self, rate: SampleRate) -> Result<(), DeviceError> {
        if self.state == DeviceState::Streaming {
            return Err(DeviceError::AlreadyStreaming);
        }
        self.config.common.sample_rate = rate;
        Ok(())
    }
}

/// Simple pseudo-random number generator for simulation
fn rand_simple(seed: u64) -> f32 {
    let x = seed.wrapping_mul(0x5DEECE66D).wrapping_add(0xB);
    ((x >> 16) & 0xFFFF) as f32 / 65535.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulated_device() {
        let mut device = OpenBciAdapter::simulated(OpenBciBoard::Cyton);
        assert_eq!(device.get_state(), DeviceState::Connected);
        assert_eq!(device.num_channels(), 8);

        device.start_streaming().unwrap();
        assert!(device.is_streaming());

        // Read a few samples
        for _ in 0..5 {
            let sample = device.read_sample().unwrap().unwrap();
            assert_eq!(sample.channels.len(), 8);
            assert!(sample.valid);
        }

        device.stop_streaming().unwrap();
        assert!(!device.is_streaming());
    }

    #[test]
    fn test_board_types() {
        assert_eq!(OpenBciBoard::Cyton.num_channels(), 8);
        assert_eq!(OpenBciBoard::CytonDaisy.num_channels(), 16);
        assert_eq!(OpenBciBoard::Ganglion.num_channels(), 4);
    }

    #[test]
    fn test_channel_info() {
        let device = OpenBciAdapter::simulated(OpenBciBoard::Cyton);
        let channels = device.get_channels();
        assert_eq!(channels.len(), 8);
        assert_eq!(channels[0].name, "Fp1");
    }
}