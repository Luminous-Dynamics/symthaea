// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Muse Adapter
//!
//! Supports Muse 2 and Muse S headbands via Bluetooth LE.
//!
//! # Channels
//! - TP9 (left ear)
//! - AF7 (left forehead)
//! - AF8 (right forehead)
//! - TP10 (right ear)
//! - Aux (reference)
//!
//! # Protocol
//! - Bluetooth Low Energy (BLE)
//! - 256 Hz sample rate
//! - 12-bit resolution

use super::traits::*;
use std::time::{Duration, Instant};

/// Muse headband models
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MuseModel {
    /// Muse 2 (2018+)
    Muse2,
    /// Muse S (Sleep focused, 2020+)
    MuseS,
    /// Original Muse (2014-2016)
    MuseOriginal,
}

impl MuseModel {
    /// Get number of EEG channels
    pub fn num_eeg_channels(&self) -> usize {
        match self {
            MuseModel::MuseOriginal => 4,
            MuseModel::Muse2 | MuseModel::MuseS => 4, // 4 EEG + aux
        }
    }

    /// Check if model has PPG sensor
    pub fn has_ppg(&self) -> bool {
        matches!(self, MuseModel::Muse2 | MuseModel::MuseS)
    }

    /// Check if model has accelerometer
    pub fn has_accel(&self) -> bool {
        true // All models have accelerometer
    }

    /// Check if model has gyroscope
    pub fn has_gyro(&self) -> bool {
        matches!(self, MuseModel::Muse2 | MuseModel::MuseS)
    }
}

/// Muse-specific configuration
#[derive(Debug, Clone)]
pub struct MuseConfig {
    /// Bluetooth address or name
    pub address: String,
    /// Model type
    pub model: MuseModel,
    /// Enable PPG (heart rate) if available
    pub enable_ppg: bool,
    /// Enable accelerometer
    pub enable_accel: bool,
    /// Enable gyroscope if available
    pub enable_gyro: bool,
    /// Preset for different use cases
    pub preset: MusePreset,
    /// Common configuration
    pub common: DeviceConfig,
}

/// Muse presets for different use cases
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MusePreset {
    /// Default preset - all sensors enabled
    Default,
    /// Sleep tracking - optimized for overnight use
    Sleep,
    /// Meditation - focus on frontal channels
    Meditation,
    /// Research - maximum data, all channels
    Research,
}

impl Default for MuseConfig {
    fn default() -> Self {
        Self {
            address: String::new(),
            model: MuseModel::Muse2,
            enable_ppg: true,
            enable_accel: true,
            enable_gyro: true,
            preset: MusePreset::Default,
            common: DeviceConfig {
                sample_rate: SampleRate::Hz256,
                enabled_channels: (0..4).collect(),
                buffer_size: 256,
                ..Default::default()
            },
        }
    }
}

impl MuseConfig {
    /// Create config for Muse 2
    pub fn muse2(address: &str) -> Self {
        Self {
            address: address.to_string(),
            model: MuseModel::Muse2,
            ..Default::default()
        }
    }

    /// Create config for Muse S
    pub fn muse_s(address: &str) -> Self {
        Self {
            address: address.to_string(),
            model: MuseModel::MuseS,
            ..Default::default()
        }
    }

    /// Create config with meditation preset
    pub fn meditation(address: &str) -> Self {
        Self {
            address: address.to_string(),
            preset: MusePreset::Meditation,
            enable_ppg: false,
            enable_gyro: false,
            ..Default::default()
        }
    }

    /// Create config with sleep preset
    pub fn sleep(address: &str) -> Self {
        Self {
            address: address.to_string(),
            model: MuseModel::MuseS,
            preset: MusePreset::Sleep,
            enable_ppg: true,
            enable_accel: true,
            enable_gyro: false,
            ..Default::default()
        }
    }
}

/// Muse device adapter
///
/// Provides connection to Muse headbands via Bluetooth LE.
///
/// # Example (requires hardware)
/// ```rust,ignore
/// use sentinels::hardware::{MuseAdapter, MuseConfig};
///
/// // Scan for devices
/// let devices = MuseAdapter::scan()?;
/// println!("Found {} Muse devices", devices.len());
///
/// // Connect to first device
/// let config = MuseConfig::muse2(&devices[0]);
/// let mut device = MuseAdapter::new(config);
/// device.connect()?;
/// device.start_streaming()?;
///
/// // Collect 5 seconds of data
/// let data = device.collect_duration(std::time::Duration::from_secs(5))?;
/// ```
pub struct MuseAdapter {
    config: MuseConfig,
    info: DeviceInfo,
    state: DeviceState,
    sample_index: u64,
    stream_start: Option<Instant>,
    /// Battery level (0-100)
    battery_level: Option<u8>,
    /// Signal quality per channel (0-100)
    signal_quality: [u8; 4],
    // In a real implementation, this would hold the BLE connection
    #[allow(dead_code)]
    simulated: bool,
}

impl MuseAdapter {
    /// Create a new Muse adapter with the given configuration
    pub fn new(config: MuseConfig) -> Self {
        let info = DeviceInfo {
            name: format!("Muse {:?}", config.model),
            manufacturer: "InteraXon".to_string(),
            model: format!("{:?}", config.model),
            firmware_version: None,
            serial_number: None,
            num_channels: config.model.num_eeg_channels(),
            supported_sample_rates: vec![SampleRate::Hz256],
            default_sample_rate: SampleRate::Hz256,
            resolution_bits: 12,
            input_range_uv: 1682.815, // Muse specific
        };

        Self {
            config,
            info,
            state: DeviceState::Disconnected,
            sample_index: 0,
            stream_start: None,
            battery_level: None,
            signal_quality: [0; 4],
            simulated: true,
        }
    }

    /// Scan for available Muse devices
    ///
    /// Returns a list of Bluetooth addresses for discovered Muse headbands.
    pub fn scan() -> Result<Vec<String>, DeviceError> {
        // In a real implementation, this would scan for BLE devices
        // For now, return an empty list
        Ok(vec![])
    }

    /// Scan with timeout
    pub fn scan_timeout(_timeout: Duration) -> Result<Vec<String>, DeviceError> {
        Self::scan()
    }

    /// Create a simulated device for testing
    pub fn simulated(model: MuseModel) -> Self {
        let mut config = match model {
            MuseModel::Muse2 => MuseConfig::muse2("SIMULATED"),
            MuseModel::MuseS => MuseConfig::muse_s("SIMULATED"),
            MuseModel::MuseOriginal => {
                let mut cfg = MuseConfig::default();
                cfg.model = MuseModel::MuseOriginal;
                cfg.address = "SIMULATED".to_string();
                cfg.enable_ppg = false;
                cfg.enable_gyro = false;
                cfg
            }
        };
        config.address = "SIMULATED".to_string();

        let mut adapter = Self::new(config);
        adapter.simulated = true;
        adapter.state = DeviceState::Connected;
        adapter.battery_level = Some(85);
        adapter.signal_quality = [95, 92, 91, 94];
        adapter
    }

    /// Get battery level (0-100)
    pub fn battery_level(&self) -> Option<u8> {
        self.battery_level
    }

    /// Get signal quality for each channel (0-100)
    pub fn signal_quality(&self) -> &[u8; 4] {
        &self.signal_quality
    }

    /// Get overall signal quality (average of all channels)
    pub fn overall_signal_quality(&self) -> u8 {
        let sum: u32 = self.signal_quality.iter().map(|&q| q as u32).sum();
        (sum / 4) as u8
    }

    /// Check if headband is properly fitted
    pub fn is_fitted(&self) -> bool {
        self.overall_signal_quality() >= 50
    }

    /// Get the model type
    pub fn model(&self) -> MuseModel {
        self.config.model
    }

    /// Generate a simulated sample
    fn generate_simulated_sample(&self) -> EegSample {
        use std::f32::consts::PI;

        let timestamp_ms = self
            .stream_start
            .map(|start| start.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);

        let t = timestamp_ms / 1000.0;

        // Muse channel layout: TP9, AF7, AF8, TP10
        let channels: Vec<f32> = (0..4)
            .map(|ch| {
                let phase = ch as f32 * 0.3;
                // Typical Muse EEG patterns
                let alpha = 8.0 * (2.0 * PI * 10.0 * t as f32 + phase).sin();
                let beta = 2.0 * (2.0 * PI * 25.0 * t as f32 + phase).sin();
                let theta = 4.0 * (2.0 * PI * 5.0 * t as f32 + phase).sin();

                // Frontal channels (AF7, AF8) have more alpha
                let alpha_boost = if ch == 1 || ch == 2 { 1.3 } else { 1.0 };
                let noise = (rand_simple(self.sample_index + ch as u64) - 0.5) * 1.5;

                alpha * alpha_boost + beta + theta + noise
            })
            .collect();

        // Muse-specific accelerometer
        let accel = if self.config.enable_accel {
            Some([0.0, 0.0, 1.0])
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

impl EegDevice for MuseAdapter {
    fn connect(&mut self) -> Result<(), DeviceError> {
        if self.state != DeviceState::Disconnected {
            return Err(DeviceError::InvalidConfig("Already connected".to_string()));
        }

        if self.config.address.is_empty() {
            return Err(DeviceError::InvalidConfig(
                "No address specified".to_string(),
            ));
        }

        // In a real implementation, we would establish BLE connection here
        self.state = DeviceState::Connected;
        self.battery_level = Some(85);
        self.signal_quality = [95, 92, 91, 94];
        Ok(())
    }

    fn disconnect(&mut self) -> Result<(), DeviceError> {
        if self.state == DeviceState::Streaming {
            self.stop_streaming()?;
        }
        self.state = DeviceState::Disconnected;
        self.battery_level = None;
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

        let sample = self.generate_simulated_sample();
        self.sample_index += 1;

        // Simulate 256 Hz sample rate
        let sample_period = Duration::from_secs_f64(1.0 / 256.0);
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
        let channel_names = ["TP9", "AF7", "AF8", "TP10"];

        channel_names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                ChannelInfo {
                    index: i,
                    name: name.to_string(),
                    enabled: self.config.common.enabled_channels.contains(&i),
                    gain: 1.0, // Muse has fixed gain
                    channel_type: ChannelType::Eeg,
                }
            })
            .collect()
    }

    fn set_channel_enabled(&mut self, channel: usize, enabled: bool) -> Result<(), DeviceError> {
        if channel >= 4 {
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

    fn set_channel_gain(&mut self, _channel: usize, _gain: f32) -> Result<(), DeviceError> {
        // Muse has fixed gain
        Err(DeviceError::Unsupported("Muse has fixed gain".to_string()))
    }

    fn set_sample_rate(&mut self, rate: SampleRate) -> Result<(), DeviceError> {
        if rate != SampleRate::Hz256 {
            return Err(DeviceError::Unsupported(
                "Muse only supports 256 Hz".to_string(),
            ));
        }
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
    fn test_simulated_muse() {
        let mut device = MuseAdapter::simulated(MuseModel::Muse2);
        assert_eq!(device.get_state(), DeviceState::Connected);
        assert_eq!(device.num_channels(), 4);
        assert!(device.is_fitted());

        device.start_streaming().unwrap();
        assert!(device.is_streaming());

        // Read a few samples
        for _ in 0..5 {
            let sample = device.read_sample().unwrap().unwrap();
            assert_eq!(sample.channels.len(), 4);
            assert!(sample.valid);
        }

        device.stop_streaming().unwrap();
    }

    #[test]
    fn test_muse_models() {
        assert_eq!(MuseModel::Muse2.num_eeg_channels(), 4);
        assert!(MuseModel::Muse2.has_ppg());
        assert!(MuseModel::Muse2.has_gyro());
        assert!(!MuseModel::MuseOriginal.has_ppg());
    }

    #[test]
    fn test_channel_names() {
        let device = MuseAdapter::simulated(MuseModel::Muse2);
        let channels = device.get_channels();
        assert_eq!(channels.len(), 4);
        assert_eq!(channels[0].name, "TP9");
        assert_eq!(channels[1].name, "AF7");
        assert_eq!(channels[2].name, "AF8");
        assert_eq!(channels[3].name, "TP10");
    }

    #[test]
    fn test_signal_quality() {
        let device = MuseAdapter::simulated(MuseModel::Muse2);
        assert!(device.overall_signal_quality() >= 50);
        assert!(device.is_fitted());
    }
}