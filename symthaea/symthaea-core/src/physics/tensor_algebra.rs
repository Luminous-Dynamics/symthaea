// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Tensor Algebra for General Relativity
//!
//! This module implements tensor calculus for spacetime geometry including
//! metric tensors, Christoffel symbols, curvature tensors, and geodesic solvers.
//!
//! ## Key Concepts
//!
//! - **Metric Tensor**: g_μν defines spacetime geometry and distances
//! - **Christoffel Symbols**: Γ^λ_μν describe connection (parallel transport)
//! - **Riemann Tensor**: R^ρ_σμν measures spacetime curvature
//! - **Geodesics**: Paths of freely falling particles
//!
//! ## Conventions
//!
//! - Greek indices (μ, ν, ...) run from 0 to 3 (spacetime)
//! - Index 0 is timelike, indices 1-3 are spacelike
//! - Signature convention is configurable (mostly-plus or mostly-minus)
//!
//! ## References
//!
//! - Misner, Thorne, Wheeler (1973) - "Gravitation"
//! - Wald (1984) - "General Relativity"

use super::constants::{C, G};
use super::standard_model::PHYSICS_DIM;
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// Metric signature convention
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Signature {
    /// (-,+,+,+) mostly plus convention (spacelike positive)
    #[default]
    MostlyPlus,
    /// (+,-,-,-) mostly minus convention (timelike positive)
    MostlyMinus,
}

/// 4x4 matrix for 4D spacetime tensors
pub type Matrix4 = [[f64; 4]; 4];

/// Metric tensor g_μν
///
/// The metric tensor defines the geometry of spacetime:
/// ds² = g_μν dx^μ dx^ν
#[derive(Debug, Clone)]
pub struct MetricTensor {
    /// Covariant components g_μν
    components: Matrix4,
    /// Metric signature
    signature: Signature,
}

impl MetricTensor {
    /// Create a new metric tensor
    pub fn new(components: Matrix4, signature: Signature) -> Self {
        Self {
            components,
            signature,
        }
    }

    /// Minkowski (flat spacetime) metric
    ///
    /// η_μν = diag(-1, 1, 1, 1) for mostly-plus
    /// η_μν = diag(1, -1, -1, -1) for mostly-minus
    pub fn minkowski() -> Self {
        Self::minkowski_with_signature(Signature::MostlyPlus)
    }

    /// Minkowski metric with specified signature
    pub fn minkowski_with_signature(signature: Signature) -> Self {
        let (t, s) = match signature {
            Signature::MostlyPlus => (-1.0, 1.0),
            Signature::MostlyMinus => (1.0, -1.0),
        };

        let components = [
            [t, 0.0, 0.0, 0.0],
            [0.0, s, 0.0, 0.0],
            [0.0, 0.0, s, 0.0],
            [0.0, 0.0, 0.0, s],
        ];

        Self {
            components,
            signature,
        }
    }

    /// Schwarzschild metric (non-rotating black hole)
    ///
    /// ds² = -(1 - r_s/r)c²dt² + (1 - r_s/r)⁻¹dr² + r²(dθ² + sin²θ dφ²)
    ///
    /// where r_s = 2GM/c² is the Schwarzschild radius
    ///
    /// # Arguments
    /// * `m` - Mass of the black hole (kg)
    /// * `r` - Radial coordinate (m)
    /// * `theta` - Polar angle (radians)
    pub fn schwarzschild(m: f64, r: f64, theta: f64) -> Self {
        let r_s = 2.0 * G * m / (C * C); // Schwarzschild radius
        let f = 1.0 - r_s / r;

        if r <= r_s {
            // Inside horizon - metric components swap sign
            return Self::schwarzschild_interior(m, r, theta);
        }

        let sin_theta = theta.sin();
        let sin2_theta = sin_theta * sin_theta;

        let components = [
            [-f * C * C, 0.0, 0.0, 0.0],
            [0.0, 1.0 / f, 0.0, 0.0],
            [0.0, 0.0, r * r, 0.0],
            [0.0, 0.0, 0.0, r * r * sin2_theta],
        ];

        Self {
            components,
            signature: Signature::MostlyPlus,
        }
    }

    /// Schwarzschild interior (r < r_s)
    fn schwarzschild_interior(m: f64, r: f64, theta: f64) -> Self {
        let r_s = 2.0 * G * m / (C * C);
        let f = r_s / r - 1.0; // Note: positive inside horizon

        let sin_theta = theta.sin();
        let sin2_theta = sin_theta * sin_theta;

        // Inside horizon, r becomes timelike and t becomes spacelike
        let components = [
            [f * C * C, 0.0, 0.0, 0.0],
            [0.0, -1.0 / f, 0.0, 0.0],
            [0.0, 0.0, r * r, 0.0],
            [0.0, 0.0, 0.0, r * r * sin2_theta],
        ];

        Self {
            components,
            signature: Signature::MostlyPlus,
        }
    }

    /// Kerr metric (rotating black hole)
    ///
    /// Uses Boyer-Lindquist coordinates (t, r, θ, φ)
    ///
    /// # Arguments
    /// * `m` - Mass (kg)
    /// * `a` - Spin parameter a = J/(Mc) (m)
    /// * `r` - Radial coordinate (m)
    /// * `theta` - Polar angle (radians)
    pub fn kerr(m: f64, a: f64, r: f64, theta: f64) -> Self {
        let r_s = 2.0 * G * m / (C * C);

        // Kerr metric functions
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();
        let sin2_theta = sin_theta * sin_theta;
        let cos2_theta = cos_theta * cos_theta;

        // Σ = r² + a²cos²θ
        let sigma = r * r + a * a * cos2_theta;

        // Δ = r² - r_s*r + a²
        let delta = r * r - r_s * r + a * a;

        // A = (r² + a²)² - a²Δsin²θ
        let r2_a2 = r * r + a * a;
        let big_a = r2_a2 * r2_a2 - a * a * delta * sin2_theta;

        let c2 = C * C;

        let components = [
            [
                -(1.0 - r_s * r / sigma) * c2,
                0.0,
                0.0,
                -r_s * r * a * sin2_theta * C / sigma,
            ],
            [0.0, sigma / delta, 0.0, 0.0],
            [0.0, 0.0, sigma, 0.0],
            [
                -r_s * r * a * sin2_theta * C / sigma,
                0.0,
                0.0,
                big_a * sin2_theta / sigma,
            ],
        ];

        Self {
            components,
            signature: Signature::MostlyPlus,
        }
    }

    /// FLRW metric (cosmological, expanding universe)
    ///
    /// ds² = -c²dt² + a(t)²[dr²/(1-kr²) + r²(dθ² + sin²θ dφ²)]
    ///
    /// # Arguments
    /// * `scale_factor` - a(t) the cosmic scale factor
    /// * `k` - Curvature parameter: 1 (closed), 0 (flat), -1 (open)
    /// * `r` - Comoving radial coordinate
    /// * `theta` - Polar angle
    pub fn flrw(scale_factor: f64, k: f64, r: f64, theta: f64) -> Self {
        let a2 = scale_factor * scale_factor;
        let sin_theta = theta.sin();
        let sin2_theta = sin_theta * sin_theta;

        let g_rr = a2 / (1.0 - k * r * r);

        let components = [
            [-C * C, 0.0, 0.0, 0.0],
            [0.0, g_rr, 0.0, 0.0],
            [0.0, 0.0, a2 * r * r, 0.0],
            [0.0, 0.0, 0.0, a2 * r * r * sin2_theta],
        ];

        Self {
            components,
            signature: Signature::MostlyPlus,
        }
    }

    /// Get component g_μν
    pub fn get(&self, mu: usize, nu: usize) -> f64 {
        self.components[mu][nu]
    }

    /// Get signature
    pub fn signature(&self) -> Signature {
        self.signature
    }

    /// Compute inverse metric g^μν
    pub fn inverse(&self) -> MetricTensor {
        let inv = matrix_inverse_4x4(&self.components);
        MetricTensor {
            components: inv,
            signature: self.signature,
        }
    }

    /// Compute determinant det(g)
    pub fn determinant(&self) -> f64 {
        matrix_determinant_4x4(&self.components)
    }

    /// Compute line element ds² = g_μν dx^μ dx^ν
    pub fn line_element(&self, dx: &[f64; 4]) -> f64 {
        let mut ds2 = 0.0;
        for mu in 0..4 {
            for nu in 0..4 {
                ds2 += self.components[mu][nu] * dx[mu] * dx[nu];
            }
        }
        ds2
    }

    /// Lower an index: v_μ = g_μν v^ν
    pub fn lower_index(&self, v_up: &[f64; 4]) -> [f64; 4] {
        let mut v_down = [0.0; 4];
        for mu in 0..4 {
            for nu in 0..4 {
                v_down[mu] += self.components[mu][nu] * v_up[nu];
            }
        }
        v_down
    }

    /// Raise an index: v^μ = g^μν v_ν
    pub fn raise_index(&self, v_down: &[f64; 4]) -> [f64; 4] {
        let g_inv = self.inverse();
        let mut v_up = [0.0; 4];
        for mu in 0..4 {
            for nu in 0..4 {
                v_up[mu] += g_inv.components[mu][nu] * v_down[nu];
            }
        }
        v_up
    }
}

/// Christoffel symbols Γ^λ_μν
///
/// Connection coefficients for parallel transport:
/// Γ^λ_μν = ½ g^λσ (∂_μ g_νσ + ∂_ν g_μσ - ∂_σ g_μν)
#[derive(Debug, Clone)]
pub struct ChristoffelSymbols {
    /// Components Γ^λ_μν indexed as [lambda][mu][nu]
    components: [[[f64; 4]; 4]; 4],
}

impl ChristoffelSymbols {
    /// Create Christoffel symbols from a metric tensor
    ///
    /// Uses numerical differentiation with step size h
    pub fn from_metric<F>(metric_fn: F, coords: &[f64; 4], h: f64) -> Self
    where
        F: Fn(&[f64; 4]) -> MetricTensor,
    {
        let g = metric_fn(coords);
        let g_inv = g.inverse();

        // Compute metric derivatives numerically
        let mut dg = [[[0.0f64; 4]; 4]; 4]; // dg[sigma][mu][nu] = ∂_σ g_μν

        for sigma in 0..4 {
            let mut coords_plus = *coords;
            let mut coords_minus = *coords;
            coords_plus[sigma] += h;
            coords_minus[sigma] -= h;

            let g_plus = metric_fn(&coords_plus);
            let g_minus = metric_fn(&coords_minus);

            for mu in 0..4 {
                for nu in 0..4 {
                    dg[sigma][mu][nu] = (g_plus.get(mu, nu) - g_minus.get(mu, nu)) / (2.0 * h);
                }
            }
        }

        // Compute Christoffel symbols
        // Γ^λ_μν = ½ g^λσ (∂_μ g_νσ + ∂_ν g_μσ - ∂_σ g_μν)
        // dg[alpha][mu][nu] = ∂_α g_μν
        let mut components = [[[0.0; 4]; 4]; 4];

        for lambda in 0..4 {
            for mu in 0..4 {
                for nu in 0..4 {
                    let mut sum = 0.0;
                    for sigma in 0..4 {
                        // ∂_μ g_νσ + ∂_ν g_μσ - ∂_σ g_μν
                        sum += g_inv.get(lambda, sigma)
                            * (dg[mu][nu][sigma] + dg[nu][mu][sigma] - dg[sigma][mu][nu]);
                    }
                    components[lambda][mu][nu] = 0.5 * sum;
                }
            }
        }

        Self { components }
    }

    /// Create Christoffel symbols for Schwarzschild metric (analytical)
    pub fn schwarzschild(m: f64, r: f64, theta: f64) -> Self {
        let r_s = 2.0 * G * m / (C * C);
        let f = 1.0 - r_s / r;

        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        let mut components = [[[0.0; 4]; 4]; 4];

        // Non-zero Christoffel symbols for Schwarzschild
        // Γ^t_tr = Γ^t_rt = r_s/(2r(r-r_s))
        components[0][0][1] = r_s * C * C / (2.0 * r * r * f);
        components[0][1][0] = components[0][0][1];

        // Γ^r_tt = c²r_s(r-r_s)/(2r³)
        components[1][0][0] = r_s * C * C * f / (2.0 * r * r);

        // Γ^r_rr = -r_s/(2r(r-r_s))
        components[1][1][1] = -r_s / (2.0 * r * r * f);

        // Γ^r_θθ = -(r-r_s)
        components[1][2][2] = -(r - r_s);

        // Γ^r_φφ = -(r-r_s)sin²θ
        components[1][3][3] = -(r - r_s) * sin_theta * sin_theta;

        // Γ^θ_rθ = Γ^θ_θr = 1/r
        components[2][1][2] = 1.0 / r;
        components[2][2][1] = 1.0 / r;

        // Γ^θ_φφ = -sinθ cosθ
        components[2][3][3] = -sin_theta * cos_theta;

        // Γ^φ_rφ = Γ^φ_φr = 1/r
        components[3][1][3] = 1.0 / r;
        components[3][3][1] = 1.0 / r;

        // Γ^φ_θφ = Γ^φ_φθ = cotθ
        components[3][2][3] = cos_theta / sin_theta;
        components[3][3][2] = cos_theta / sin_theta;

        Self { components }
    }

    /// Get component Γ^λ_μν
    pub fn get(&self, lambda: usize, mu: usize, nu: usize) -> f64 {
        self.components[lambda][mu][nu]
    }
}

/// Riemann curvature tensor R^ρ_σμν
#[derive(Debug, Clone)]
pub struct RiemannTensor {
    /// Components R^ρ_σμν indexed as [rho][sigma][mu][nu]
    components: [[[[f64; 4]; 4]; 4]; 4],
}

impl RiemannTensor {
    /// Create Riemann tensor from Christoffel symbols
    ///
    /// R^ρ_σμν = ∂_μΓ^ρ_νσ - ∂_νΓ^ρ_μσ + Γ^ρ_μλΓ^λ_νσ - Γ^ρ_νλΓ^λ_μσ
    pub fn from_christoffel<F>(christoffel_fn: F, coords: &[f64; 4], h: f64) -> Self
    where
        F: Fn(&[f64; 4]) -> ChristoffelSymbols,
    {
        let gamma = christoffel_fn(coords);

        // Compute Christoffel derivatives numerically
        let mut dgamma = [[[[0.0f64; 4]; 4]; 4]; 4]; // dgamma[alpha][lambda][mu][nu] = ∂_α Γ^λ_μν

        for alpha in 0..4 {
            let mut coords_plus = *coords;
            let mut coords_minus = *coords;
            coords_plus[alpha] += h;
            coords_minus[alpha] -= h;

            let gamma_plus = christoffel_fn(&coords_plus);
            let gamma_minus = christoffel_fn(&coords_minus);

            for lambda in 0..4 {
                for mu in 0..4 {
                    for nu in 0..4 {
                        dgamma[alpha][lambda][mu][nu] = (gamma_plus.get(lambda, mu, nu)
                            - gamma_minus.get(lambda, mu, nu))
                            / (2.0 * h);
                    }
                }
            }
        }

        // Compute Riemann tensor
        let mut components = [[[[0.0; 4]; 4]; 4]; 4];

        for rho in 0..4 {
            for sigma in 0..4 {
                for mu in 0..4 {
                    for nu in 0..4 {
                        // R^ρ_σμν = ∂_μΓ^ρ_νσ - ∂_νΓ^ρ_μσ
                        let mut r = dgamma[mu][rho][nu][sigma] - dgamma[nu][rho][mu][sigma];

                        // + Γ^ρ_μλΓ^λ_νσ - Γ^ρ_νλΓ^λ_μσ
                        for lambda in 0..4 {
                            r += gamma.get(rho, mu, lambda) * gamma.get(lambda, nu, sigma);
                            r -= gamma.get(rho, nu, lambda) * gamma.get(lambda, mu, sigma);
                        }

                        components[rho][sigma][mu][nu] = r;
                    }
                }
            }
        }

        Self { components }
    }

    /// Get component R^ρ_σμν
    pub fn get(&self, rho: usize, sigma: usize, mu: usize, nu: usize) -> f64 {
        self.components[rho][sigma][mu][nu]
    }

    /// Compute Ricci tensor R_μν = R^ρ_μρν
    pub fn ricci_tensor(&self) -> RicciTensor {
        let mut components = [[0.0; 4]; 4];

        for mu in 0..4 {
            for nu in 0..4 {
                for rho in 0..4 {
                    components[mu][nu] += self.components[rho][mu][rho][nu];
                }
            }
        }

        RicciTensor { components }
    }

    /// Compute Ricci scalar R = g^μν R_μν
    pub fn ricci_scalar(&self, metric: &MetricTensor) -> f64 {
        let ricci = self.ricci_tensor();
        let g_inv = metric.inverse();

        let mut r = 0.0;
        for mu in 0..4 {
            for nu in 0..4 {
                r += g_inv.get(mu, nu) * ricci.get(mu, nu);
            }
        }
        r
    }

    /// Compute Kretschmann scalar K = R_μνρσ R^μνρσ
    ///
    /// This is useful for detecting singularities (diverges at r=0 for Schwarzschild)
    pub fn kretschmann_scalar(&self, metric: &MetricTensor) -> f64 {
        let g_inv = metric.inverse();

        // R_μνρσ = g_μα R^α_νρσ (lower first index)
        // K = R_μνρσ R^μνρσ = g^μα g^νβ g^ργ g^σδ R_αβγδ R_μνρσ

        // For simplicity, compute K = R^ρ_σμν R^σ_ρ^μν with appropriate contractions
        let mut k = 0.0;

        for rho in 0..4 {
            for sigma in 0..4 {
                for mu in 0..4 {
                    for nu in 0..4 {
                        // Contract with metric
                        for alpha in 0..4 {
                            for beta in 0..4 {
                                for gamma in 0..4 {
                                    for delta in 0..4 {
                                        k += g_inv.get(rho, alpha)
                                            * metric.get(sigma, beta)
                                            * g_inv.get(mu, gamma)
                                            * g_inv.get(nu, delta)
                                            * self.components[alpha][beta][gamma][delta]
                                            * self.components[rho][sigma][mu][nu];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        k
    }
}

/// Ricci tensor R_μν
#[derive(Debug, Clone)]
pub struct RicciTensor {
    components: Matrix4,
}

impl RicciTensor {
    /// Get component R_μν
    pub fn get(&self, mu: usize, nu: usize) -> f64 {
        self.components[mu][nu]
    }
}

/// Geodesic solver
///
/// Solves the geodesic equation:
/// d²x^μ/dτ² + Γ^μ_νρ (dx^ν/dτ)(dx^ρ/dτ) = 0
#[derive(Debug, Clone)]
pub struct GeodesicSolver {
    mass: f64,
    h: f64, // Step size for Christoffel computation
}

impl GeodesicSolver {
    /// Create solver for a given mass
    pub fn new(mass: f64) -> Self {
        Self { mass, h: 1.0 }
    }

    /// Set step size for numerical differentiation
    pub fn with_h(mut self, h: f64) -> Self {
        self.h = h;
        self
    }

    /// Solve geodesic for Schwarzschild spacetime
    ///
    /// # Arguments
    /// * `initial_pos` - Initial 4-position [ct, r, θ, φ]
    /// * `initial_vel` - Initial 4-velocity [d(ct)/dτ, dr/dτ, dθ/dτ, dφ/dτ]
    /// * `tau_max` - Maximum proper time
    /// * `steps` - Number of integration steps
    ///
    /// # Returns
    /// Vector of 4-positions along the geodesic
    pub fn solve_schwarzschild(
        &self,
        initial_pos: [f64; 4],
        initial_vel: [f64; 4],
        tau_max: f64,
        steps: usize,
    ) -> Vec<[f64; 4]> {
        let dtau = tau_max / steps as f64;
        let mut trajectory = Vec::with_capacity(steps + 1);

        let mut x = initial_pos;
        let mut v = initial_vel;

        trajectory.push(x);

        for _ in 0..steps {
            // Get Christoffel symbols at current position
            let r = x[1];
            let theta = x[2];
            let gamma = ChristoffelSymbols::schwarzschild(self.mass, r, theta);

            // Compute acceleration: a^μ = -Γ^μ_νρ v^ν v^ρ
            let mut a = [0.0; 4];
            for mu in 0..4 {
                for nu in 0..4 {
                    for rho in 0..4 {
                        a[mu] -= gamma.get(mu, nu, rho) * v[nu] * v[rho];
                    }
                }
            }

            // Simple Euler integration
            for mu in 0..4 {
                x[mu] += v[mu] * dtau;
                v[mu] += a[mu] * dtau;
            }

            trajectory.push(x);
        }

        trajectory
    }

    /// Compute conserved quantities for Schwarzschild geodesic
    ///
    /// Returns (energy per unit mass, angular momentum per unit mass)
    pub fn schwarzschild_constants(&self, pos: &[f64; 4], vel: &[f64; 4]) -> (f64, f64) {
        let r = pos[1];
        let theta = pos[2];

        let r_s = 2.0 * G * self.mass / (C * C);
        let f = 1.0 - r_s / r;

        // E/m = (1 - r_s/r) c (dt/dτ)
        let energy = f * C * vel[0];

        // L/m = r² sin²θ (dφ/dτ)
        let sin_theta = theta.sin();
        let angular_momentum = r * r * sin_theta * sin_theta * vel[3];

        (energy, angular_momentum)
    }
}

/// HDC representation of spacetime geometry
#[derive(Debug, Clone)]
pub struct SpacetimeEncoder {
    genesis: GenesisSeed,
    /// Flat spacetime concept
    pub flat: ContinuousHV,
    /// Curved spacetime concept
    pub curved: ContinuousHV,
    /// Singularity concept
    pub singularity: ContinuousHV,
    /// Horizon concept
    pub horizon: ContinuousHV,
    /// Geodesic concept
    pub geodesic: ContinuousHV,
}

impl SpacetimeEncoder {
    /// Create a new spacetime encoder from genesis
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        Self {
            genesis: genesis.clone(),
            flat: genesis.hv("spacetime::flat", PHYSICS_DIM),
            curved: genesis.hv("spacetime::curved", PHYSICS_DIM),
            singularity: genesis.hv("spacetime::singularity", PHYSICS_DIM),
            horizon: genesis.hv("spacetime::horizon", PHYSICS_DIM),
            geodesic: genesis.hv("spacetime::geodesic", PHYSICS_DIM),
        }
    }

    /// Encode curvature level (0 = flat, 1 = highly curved)
    pub fn encode_curvature(&self, curvature: f64) -> ContinuousHV {
        let curvature = curvature.clamp(0.0, 1.0) as f32;
        ContinuousHV::weighted_bundle(&[&self.flat, &self.curved], &[1.0 - curvature, curvature])
    }

    /// Encode spacetime type
    pub fn encode_spacetime_type(&self, metric_type: &str) -> ContinuousHV {
        let type_hv = self
            .genesis
            .hv(&format!("metric::{metric_type}"), PHYSICS_DIM);
        self.curved.bind(&type_hv)
    }
}

// ============================================================================
// Matrix utilities
// ============================================================================

/// 4x4 matrix inverse
fn matrix_inverse_4x4(m: &Matrix4) -> Matrix4 {
    let det = matrix_determinant_4x4(m);
    if det.abs() < 1e-20 {
        return *m; // Return original if singular
    }

    let mut adj = [[0.0; 4]; 4];

    // Compute adjugate matrix
    for i in 0..4 {
        for j in 0..4 {
            let minor = minor_3x3(m, i, j);
            let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
            adj[j][i] = sign * minor;
        }
    }

    // Divide by determinant
    let mut inv = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            inv[i][j] = adj[i][j] / det;
        }
    }

    inv
}

/// 3x3 minor of 4x4 matrix (determinant of submatrix excluding row i, col j)
fn minor_3x3(m: &Matrix4, exclude_row: usize, exclude_col: usize) -> f64 {
    let mut sub = [[0.0; 3]; 3];
    let mut si = 0;
    for i in 0..4 {
        if i == exclude_row {
            continue;
        }
        let mut sj = 0;
        for j in 0..4 {
            if j == exclude_col {
                continue;
            }
            sub[si][sj] = m[i][j];
            sj += 1;
        }
        si += 1;
    }

    // 3x3 determinant
    sub[0][0] * (sub[1][1] * sub[2][2] - sub[1][2] * sub[2][1])
        - sub[0][1] * (sub[1][0] * sub[2][2] - sub[1][2] * sub[2][0])
        + sub[0][2] * (sub[1][0] * sub[2][1] - sub[1][1] * sub[2][0])
}

/// 4x4 matrix determinant
fn matrix_determinant_4x4(m: &Matrix4) -> f64 {
    let mut det = 0.0;
    for j in 0..4 {
        let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
        det += sign * m[0][j] * minor_3x3(m, 0, j);
    }
    det
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minkowski_metric() {
        let eta = MetricTensor::minkowski();

        // Check signature
        assert_eq!(eta.get(0, 0), -1.0);
        assert_eq!(eta.get(1, 1), 1.0);
        assert_eq!(eta.get(2, 2), 1.0);
        assert_eq!(eta.get(3, 3), 1.0);

        // Off-diagonal should be zero
        assert_eq!(eta.get(0, 1), 0.0);
        assert_eq!(eta.get(1, 2), 0.0);
    }

    #[test]
    fn test_minkowski_inverse() {
        let eta = MetricTensor::minkowski();
        let eta_inv = eta.inverse();

        // For Minkowski, inverse equals itself
        for mu in 0..4 {
            for nu in 0..4 {
                assert!((eta.get(mu, nu) - eta_inv.get(mu, nu)).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_minkowski_determinant() {
        let eta = MetricTensor::minkowski();
        let det = eta.determinant();
        assert!((det - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_schwarzschild_far_field() {
        // Far from black hole, should approach Minkowski
        let m = 2e30; // ~1 solar mass
        let r = 1e12; // Very far
        let g = MetricTensor::schwarzschild(m, r, std::f64::consts::PI / 2.0);

        let r_s = 2.0 * G * m / (C * C);
        let f = 1.0 - r_s / r;

        // g_tt should be approximately -c²
        assert!((g.get(0, 0) / (-C * C) - 1.0).abs() < 0.01);

        // g_rr should be approximately 1
        assert!((g.get(1, 1) - 1.0 / f).abs() < 0.01);
    }

    #[test]
    fn test_schwarzschild_horizon() {
        let m = 2e30;
        let r_s = 2.0 * G * m / (C * C);
        let r = r_s * 1.001; // Just outside horizon

        let g = MetricTensor::schwarzschild(m, r, std::f64::consts::PI / 2.0);

        // g_tt should be very small (approaching 0)
        assert!(g.get(0, 0).abs() < 0.01 * C * C);

        // g_rr should be large
        assert!(g.get(1, 1) > 100.0);
    }

    #[test]
    fn test_flrw_flat() {
        let a = 1.0; // Scale factor
        let g = MetricTensor::flrw(a, 0.0, 1.0, std::f64::consts::PI / 2.0);

        // For flat FLRW, g_rr = a²
        assert!((g.get(1, 1) - a * a).abs() < 1e-10);
    }

    #[test]
    fn test_christoffel_schwarzschild() {
        let m = 2e30;
        let r = 1e10;
        let theta = std::f64::consts::PI / 2.0;

        let gamma = ChristoffelSymbols::schwarzschild(m, r, theta);

        // Γ^r_θθ = -(r - r_s) at equator
        let r_s = 2.0 * G * m / (C * C);
        assert!((gamma.get(1, 2, 2) - (-(r - r_s))).abs() < 1e-5);

        // Γ^θ_rθ = 1/r
        assert!((gamma.get(2, 1, 2) - 1.0 / r).abs() < 1e-15);
    }

    #[test]
    fn test_geodesic_solver() {
        let m = 2e30; // Solar mass
        let solver = GeodesicSolver::new(m);

        // Test geodesic solver produces trajectory output
        let r_s = 2.0 * G * m / (C * C);
        let r = 100.0 * r_s; // Far from horizon for numerical stability

        // Simple radial infall test - just verify trajectory is computed
        let initial_pos = [0.0, r, std::f64::consts::PI / 2.0, 0.0];
        let initial_vel = [C, 0.0, 0.0, 0.0]; // Pure radial motion

        let trajectory = solver.solve_schwarzschild(initial_pos, initial_vel, 1e-4, 10);

        // Verify trajectory was computed with correct number of points
        assert_eq!(trajectory.len(), 11);

        // Verify positions are finite (no NaN/Inf)
        for pos in &trajectory {
            assert!(pos[1].is_finite(), "Radius should be finite");
        }

        // For radial infall, radius should decrease (particle falling in)
        let r_final = trajectory[10][1];
        assert!(r_final < r, "Radial infall should decrease radius");
    }

    #[test]
    fn test_spacetime_encoder() {
        let genesis = GenesisSeed::from_phrase("test");
        let encoder = SpacetimeEncoder::from_genesis(&genesis);

        let curvature_hv = encoder.encode_curvature(0.5);
        assert_eq!(curvature_hv.values.len(), PHYSICS_DIM);

        let type_hv = encoder.encode_spacetime_type("schwarzschild");
        assert_eq!(type_hv.values.len(), PHYSICS_DIM);
    }

    #[test]
    fn test_matrix_inverse() {
        let m = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
        ];

        let inv = matrix_inverse_4x4(&m);

        assert!((inv[0][0] - 1.0).abs() < 1e-10);
        assert!((inv[1][1] - 0.5).abs() < 1e-10);
        assert!((inv[2][2] - 1.0 / 3.0).abs() < 1e-10);
        assert!((inv[3][3] - 0.25).abs() < 1e-10);
    }
}
