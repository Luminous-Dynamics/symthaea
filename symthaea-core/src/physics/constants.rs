//! Physical Constants
//!
//! Standard physical constants used throughout the physics modules.

/// Speed of light in vacuum (m/s)
pub const C: f64 = 299_792_458.0;

/// Gravitational constant (m³/(kg·s²))
pub const G: f64 = 6.674_30e-11;

/// Planck constant (J·s)
pub const H: f64 = 6.626_070_15e-34;

/// Reduced Planck constant (J·s)
pub const HBAR: f64 = 1.054_571_817e-34;

/// Boltzmann constant (J/K)
pub const K_BOLTZMANN: f64 = 1.380_649e-23;

/// Elementary charge (C)
pub const E_CHARGE: f64 = 1.602_176_634e-19;

/// Electron mass (kg)
pub const M_ELECTRON: f64 = 9.109_383_7015e-31;

/// Proton mass (kg)
pub const M_PROTON: f64 = 1.672_621_923_69e-27;

/// Vacuum permittivity (F/m)
pub const EPSILON_0: f64 = 8.854_187_8128e-12;

/// Vacuum permeability (H/m)
pub const MU_0: f64 = 1.256_637_062_12e-6;

/// Gas constant (J/(mol·K))
pub const R_GAS: f64 = 8.314_462_618;

/// Avogadro constant (1/mol)
pub const N_AVOGADRO: f64 = 6.022_140_76e23;

/// Room temperature (K) - standard 25°C
pub const T_ROOM: f64 = 298.15;

/// Standard atmosphere (Pa)
pub const P_STANDARD: f64 = 101_325.0;
