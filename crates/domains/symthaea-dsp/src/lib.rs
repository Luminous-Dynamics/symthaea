// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-dsp
//!
//! Digital signal processing for Symthaea — spectra, convolution, filtering,
//! and sampling theory. Connects to the audio work (Broca, muse), which
//! synthesize/analyze sound but had no shared DSP primitives.
//!
//! Pure `std`, zero dependencies, no `symthaea-core` link. Checked against known
//! transforms.
//!
//! ## Scope
//!
//! - [`dft`]: DFT and magnitude spectrum.
//! - [`signal`]: convolution, moving-average filter, Nyquist/aliasing.
//!
//! ## Example
//!
//! ```
//! use symthaea_dsp::{dft::dft, signal::convolve};
//! assert_eq!(convolve(&[1.0, 1.0], &[1.0, 1.0]), vec![1.0, 2.0, 1.0]);
//! assert!((dft(&[1.0, 1.0, 1.0, 1.0])[0].0 - 4.0).abs() < 1e-9); // DC bin
//! ```

pub mod dft;
pub mod signal;

pub use dft::{dft, magnitude};
pub use signal::{convolve, moving_average, nyquist_frequency, will_alias};
