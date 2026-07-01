// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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

mod muse;
mod openbci;
mod traits;

pub use muse::{MuseAdapter, MuseConfig, MuseModel};
pub use openbci::{OpenBciAdapter, OpenBciBoard, OpenBciConfig};
pub use traits::{
    ChannelInfo, DeviceConfig, DeviceError, DeviceInfo, DeviceState, EegDevice, EegSample,
    SampleRate,
};