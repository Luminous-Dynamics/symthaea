//! Hardware Abstraction Layer for EEG Devices
//!
//! This module provides a unified interface for connecting to various EEG devices.
//!
//! ## Supported Devices
//! - OpenBCI (Cyton, Ganglion, Cyton-Daisy)
//! - Muse (Muse 2, Muse S)
//!
//! ## Usage
//! ```rust,ignore
//! use sentinels::hardware::{EegDevice, OpenBciAdapter, MuseAdapter};
//!
//! // Connect to OpenBCI
//! let mut device = OpenBciAdapter::connect("/dev/ttyUSB0")?;
//! device.start_streaming()?;
//!
//! // Read samples
//! while let Some(sample) = device.read_sample()? {
//!     println!("Channel 0: {}", sample.channels[0]);
//! }
//! ```

mod traits;
mod openbci;
mod muse;

pub use traits::{
    EegDevice, EegSample, DeviceConfig, DeviceInfo, DeviceError,
    ChannelInfo, SampleRate, DeviceState,
};
pub use openbci::{OpenBciAdapter, OpenBciConfig, OpenBciBoard};
pub use muse::{MuseAdapter, MuseConfig, MuseModel};
