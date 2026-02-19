//! # Symthaea Signal Processing
//!
//! Pure mathematical signal processing and numerical analysis primitives for
//! Symthaea.  This crate has zero dependency on HDC, consciousness, or CfC
//! internals — it provides the mathematical substrate that the cognitive
//! architecture builds upon.
//!
//! ## Modules
//!
//! - [`ode_solvers`]: Euler, RK4, Dormand-Prince, Implicit Midpoint, Backward Euler,
//!   Exponential Integrator, Newton solver
//! - [`spectral_analysis`]: Cooley-Tukey FFT, Welch PSD, coherence, band power,
//!   spectral entropy
//! - [`stability_analysis`]: Jacobian, eigenvalue analysis (QR), Lyapunov exponents,
//!   fixed point detection, bifurcation tracking
//! - [`stochastic_dynamics`]: SDE solvers (Euler-Maruyama, Milstein, Stochastic RK),
//!   Ornstein-Uhlenbeck, Fokker-Planck, Langevin dynamics
//! - [`hmm`]: Forward-Backward, Viterbi decoding, Baum-Welch EM, pre-configured
//!   sleep staging and anesthesia HMMs
//! - [`wavelet`]: DWT (Mallat), CWT (Morlet), wavelet families (Haar, Daubechies,
//!   Symlet), spindle detection, wavelet entropy
//! - [`phase_amplitude_coupling`]: PAC analysis, Modulation Index, Mean Vector Length,
//!   surrogate testing, comodulogram

#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::type_complexity)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::new_without_default)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::ptr_arg)]
#![allow(clippy::unnecessary_unwrap)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::should_implement_trait)]

pub mod hmm;
pub mod ode_solvers;
pub mod phase_amplitude_coupling;
pub mod spectral_analysis;
pub mod stability_analysis;
pub mod stochastic_dynamics;
pub mod wavelet;
