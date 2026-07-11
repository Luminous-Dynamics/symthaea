// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-complex
//!
//! Complex numbers and the FFT — `std` has no complex type, and the domain
//! ecosystem needs one (`symthaea-dsp` currently carries a naive O(n²) DFT over
//! bare `(re, im)` tuples; control-theory reasons about poles in the complex
//! plane; quantum/signal work needs it).
//!
//! Pure `std`, zero dependencies, no `symthaea-core` link. Checked against
//! closed forms (Euler's identity, roots of unity, FFT of known signals).
//!
//! ## Contents
//! - [`complex::Complex`] — operator-overloaded arithmetic, polar form,
//!   `exp`/`ln`/`sqrt`, roots of unity
//! - [`fft`] — radix-2 Cooley-Tukey `fft`/`ifft` (power-of-two lengths)
//!
//! ## Example
//!
//! ```
//! use symthaea_complex::Complex;
//! use std::f64::consts::PI;
//! // Euler's identity: e^{iπ} = −1.
//! let z = (Complex::i() * Complex::real(PI)).exp();
//! assert!(z.approx_eq(Complex::real(-1.0), 1e-12));
//! ```

pub mod complex;
pub mod fft;

pub use complex::Complex;
pub use fft::{fft, fft_real, ifft};
