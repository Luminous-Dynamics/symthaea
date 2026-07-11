// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-finite-field
//!
//! Finite-field arithmetic — foundational to coding theory and cryptography.
//! Two fields: the prime field `GF(p)` and the binary field `GF(2⁸)` used by
//! AES and Reed-Solomon.
//!
//! Pure `std`, zero dependencies, no `symthaea-core` link. Checked against known
//! values (GF(7) inverses, the FIPS-197 GF(2⁸) multiplication example).
//!
//! ## Contents
//! - [`prime::PrimeField`] — `GF(p)`: add/sub/mul/pow and inverse (Fermat)
//! - [`binary`] — `GF(2⁸)`: XOR add, AES-polynomial mul, pow, inverse
//!
//! ## Example
//!
//! ```
//! use symthaea_finite_field::prime::PrimeField;
//! let f = PrimeField::new(7).unwrap();
//! // In GF(7), 3·5 = 1, so 3⁻¹ = 5.
//! assert_eq!(f.mul(3, 5), 1);
//! assert_eq!(f.inverse(3), Some(5));
//! ```

pub mod binary;
pub mod prime;

pub use prime::PrimeField;
