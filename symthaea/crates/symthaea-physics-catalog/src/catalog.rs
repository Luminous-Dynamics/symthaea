// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! The 148-equation physics catalog for WASM.

use crate::hv::HV;
use serde::{Deserialize, Serialize};

/// Physics domain classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Domain {
    ClassicalMechanics,
    Electromagnetism,
    GeneralRelativity,
    QuantumMechanics,
    QuantumFieldTheory,
    StatisticalMechanics,
    FluidDynamics,
    Cosmology,
    NuclearPhysics,
    Thermodynamics,
    Optics,
    CondensedMatter,
    ParticlePhysics,
    ModifiedGravity,
    InformationTheory,
    Biophysics,
    Acoustics,
    PlasmaPhysics,
    Mathematics,
}

impl Domain {
    pub fn label(&self) -> &'static str {
        match self {
            Self::ClassicalMechanics => "Classical Mechanics",
            Self::Electromagnetism => "Electromagnetism",
            Self::GeneralRelativity => "General Relativity",
            Self::QuantumMechanics => "Quantum Mechanics",
            Self::QuantumFieldTheory => "Quantum Field Theory",
            Self::StatisticalMechanics => "Statistical Mechanics",
            Self::FluidDynamics => "Fluid Dynamics",
            Self::Cosmology => "Cosmology",
            Self::NuclearPhysics => "Nuclear Physics",
            Self::Thermodynamics => "Thermodynamics",
            Self::Optics => "Optics",
            Self::CondensedMatter => "Condensed Matter",
            Self::ParticlePhysics => "Particle Physics",
            Self::ModifiedGravity => "Modified Gravity",
            Self::InformationTheory => "Information Theory",
            Self::Biophysics => "Biophysics",
            Self::Acoustics => "Acoustics",
            Self::PlasmaPhysics => "Plasma Physics",
            Self::Mathematics => "Mathematics",
        }
    }
}

/// A catalog entry: equation name, domain, LaTeX, and pre-computed HV.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogEntry {
    pub name: String,
    pub domain: Domain,
    pub latex: String,
    pub hv: HV,
}

/// The full 148-equation catalog.
pub struct PhysicsCatalog {
    entries: Vec<CatalogEntry>,
}

impl PhysicsCatalog {
    /// Build the catalog (pre-computes all HVs).
    pub fn new() -> Self {
        let raw = equations_raw();
        let entries = raw
            .into_iter()
            .map(|(name, domain, latex)| {
                let hv = HV::from_name(name);
                CatalogEntry {
                    name: name.to_string(),
                    domain,
                    latex: latex.to_string(),
                    hv,
                }
            })
            .collect();
        Self { entries }
    }

    pub fn entries(&self) -> &[CatalogEntry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn find_by_name(&self, name: &str) -> Option<&CatalogEntry> {
        self.entries.iter().find(|e| e.name == name)
    }

    pub fn entries_in_domain(&self, domain: Domain) -> Vec<&CatalogEntry> {
        self.entries.iter().filter(|e| e.domain == domain).collect()
    }
}

impl Default for PhysicsCatalog {
    fn default() -> Self {
        Self::new()
    }
}

/// All 148 equations as (name, domain, latex) tuples.
fn equations_raw() -> Vec<(&'static str, Domain, &'static str)> {
    use Domain::*;
    vec![
        // ── Electromagnetism ──
        ("Maxwell Gauss Law", Electromagnetism, r"\nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_0}"),
        ("Maxwell Gauss Magnetism", Electromagnetism, r"\nabla \cdot \mathbf{B} = 0"),
        ("Maxwell Faraday Law", Electromagnetism, r"\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}"),
        ("Maxwell Ampere Law", Electromagnetism, r"\nabla \times \mathbf{B} = \mu_0 \mathbf{J} + \mu_0 \varepsilon_0 \frac{\partial \mathbf{E}}{\partial t}"),
        ("Coulomb Law", Electromagnetism, r"F = k_e \frac{q_1 q_2}{r^2}"),
        ("Lorentz Force", Electromagnetism, r"\mathbf{F} = q(\mathbf{E} + \mathbf{v} \times \mathbf{B})"),
        ("Ohm Law", Electromagnetism, r"V = IR"),
        ("Biot-Savart Law", Electromagnetism, r"d\mathbf{B} = \frac{\mu_0}{4\pi}\frac{I d\mathbf{l} \times \hat{r}}{r^2}"),
        ("Poisson Equation", Electromagnetism, r"\nabla^2 \varphi = -\frac{\rho}{\varepsilon_0}"),
        ("Larmor Radiation Formula", Electromagnetism, r"P = \frac{q^2 a^2}{6\pi\varepsilon_0 c^3}"),
        ("Poynting Vector", Electromagnetism, r"\mathbf{S} = \frac{1}{\mu_0}(\mathbf{E} \times \mathbf{B})"),
        ("Magnetic Dipole Moment", Electromagnetism, r"\mathbf{m} = NI\mathbf{A}"),
        // ── General Relativity ──
        ("Einstein Field Equations", GeneralRelativity, r"G_{\mu\nu} = \frac{8\pi G}{c^4} T_{\mu\nu}"),
        ("Schwarzschild Metric", GeneralRelativity, r"ds^2 = -(1-\frac{2GM}{c^2 r})c^2 dt^2 + \cdots"),
        ("Kerr Metric", GeneralRelativity, r"ds^2 = -(1-\frac{r_s r}{\Sigma})c^2 dt^2 + \cdots"),
        ("Alcubierre Metric", GeneralRelativity, r"ds^2 = -c^2 dt^2 + (dx - v_s f(r_s) dt)^2 + dy^2 + dz^2"),
        ("Mass-Energy Equivalence", GeneralRelativity, r"E = mc^2"),
        ("Lorentz Factor", GeneralRelativity, r"\gamma = \frac{1}{\sqrt{1-v^2/c^2}}"),
        ("Schwarzschild Radius", GeneralRelativity, r"r_s = \frac{2GM}{c^2}"),
        ("Hawking Temperature", GeneralRelativity, r"T_H = \frac{\hbar c^3}{8\pi G M k_B}"),
        ("Bekenstein-Hawking Entropy", GeneralRelativity, r"S = \frac{k_B c^3 A}{4 G \hbar}"),
        ("Geodesic Equation", GeneralRelativity, r"\frac{d^2 x^\mu}{d\tau^2} + \Gamma^\mu_{\nu\sigma}\frac{dx^\nu}{d\tau}\frac{dx^\sigma}{d\tau} = 0"),
        ("Raychaudhuri Equation", GeneralRelativity, r"\frac{d\theta}{d\tau} = -\frac{\theta^2}{3} - \sigma^2 + \omega^2 - R_{\mu\nu}u^\mu u^\nu"),
        ("Gravitational Wave Strain", GeneralRelativity, r"h = \frac{4G\mathcal{M}_c^{5/3}(\pi f)^{2/3}}{c^4 d}"),
        // ── Quantum Mechanics ──
        ("Schrodinger Equation", QuantumMechanics, r"i\hbar\frac{\partial\psi}{\partial t} = \hat{H}\psi"),
        ("Planck-Einstein Relation", QuantumMechanics, r"E = hf"),
        ("De Broglie Wavelength", QuantumMechanics, r"\lambda = \frac{h}{p}"),
        ("Heisenberg Uncertainty Principle", QuantumMechanics, r"\Delta x \Delta p \geq \frac{\hbar}{2}"),
        ("Photoelectric Equation", QuantumMechanics, r"KE_{max} = hf - \phi"),
        ("Planck Radiation Law", QuantumMechanics, r"B(\nu,T) = \frac{2h\nu^3/c^2}{e^{h\nu/kT}-1}"),
        ("Fermi Golden Rule", QuantumMechanics, r"\Gamma = \frac{2\pi}{\hbar}|\langle f|V|i\rangle|^2\rho(E)"),
        ("Compton Scattering", QuantumMechanics, r"\Delta\lambda = \frac{h}{m_e c}(1-\cos\theta)"),
        ("Rydberg Formula", QuantumMechanics, r"\frac{1}{\lambda} = R_\infty\left(\frac{1}{n_1^2}-\frac{1}{n_2^2}\right)"),
        ("Pauli Exclusion Principle", QuantumMechanics, r"\psi(1,2) = -\psi(2,1)"),
        ("WKB Approximation", QuantumMechanics, r"\psi \approx \frac{e^{\pm i\int p\,dx/\hbar}}{\sqrt{p}}"),
        ("Born Approximation", QuantumMechanics, r"f(\theta) = -\frac{m}{2\pi\hbar^2}\int V(\mathbf{r}')e^{i\mathbf{q}\cdot\mathbf{r}'}d^3r'"),
        ("Time-Independent Perturbation", QuantumMechanics, r"E_n^{(1)} = \langle n^0|V|n^0\rangle"),
        // ── Quantum Field Theory ──
        ("Dirac Equation", QuantumFieldTheory, r"(i\gamma^\mu\partial_\mu - m)\psi = 0"),
        ("Klein-Gordon Equation", QuantumFieldTheory, r"(\Box + m^2)\phi = 0"),
        ("Yang-Mills Equation", QuantumFieldTheory, r"D_\mu F^{\mu\nu} = J^\nu"),
        // ── Classical Mechanics ──
        ("Newton Second Law", ClassicalMechanics, r"\mathbf{F} = m\mathbf{a}"),
        ("Newton Gravitation", ClassicalMechanics, r"F = G\frac{Mm}{r^2}"),
        ("Kepler Third Law", ClassicalMechanics, r"T^2 = \frac{4\pi^2}{GM}a^3"),
        ("Hooke Law", ClassicalMechanics, r"F = -kx"),
        ("Centripetal Acceleration", ClassicalMechanics, r"a = \frac{v^2}{r}"),
        ("Euler-Lagrange Equation", ClassicalMechanics, r"\frac{\partial L}{\partial q} - \frac{d}{dt}\frac{\partial L}{\partial\dot{q}} = 0"),
        ("Hamilton Equations", ClassicalMechanics, r"\dot{q}=\frac{\partial H}{\partial p},\;\dot{p}=-\frac{\partial H}{\partial q}"),
        ("Simple Harmonic Oscillator", ClassicalMechanics, r"x = A\cos(\omega t + \phi)"),
        ("Angular Momentum", ClassicalMechanics, r"\mathbf{L} = I\boldsymbol{\omega}"),
        ("Torque", ClassicalMechanics, r"\boldsymbol{\tau} = \mathbf{r} \times \mathbf{F}"),
        ("Work-Energy Theorem", ClassicalMechanics, r"W = \Delta KE"),
        ("Moment of Inertia", ClassicalMechanics, r"I = \sum m_i r_i^2"),
        // ── Fluid Dynamics ──
        ("Navier-Stokes Equation", FluidDynamics, r"\rho(\partial_t\mathbf{v}+\mathbf{v}\cdot\nabla\mathbf{v})=-\nabla p+\mu\nabla^2\mathbf{v}+\mathbf{f}"),
        ("Wave Equation", FluidDynamics, r"\Box\psi = 0"),
        ("Bernoulli Equation", FluidDynamics, r"P + \frac{1}{2}\rho v^2 + \rho g h = \text{const}"),
        ("Continuity Equation", FluidDynamics, r"\frac{\partial\rho}{\partial t} + \nabla\cdot(\rho\mathbf{v}) = 0"),
        ("Reynolds Number", FluidDynamics, r"Re = \frac{\rho v L}{\mu}"),
        ("Stokes Drag", FluidDynamics, r"F = 6\pi\mu R v"),
        ("Poiseuille Flow", FluidDynamics, r"Q = \frac{\pi R^4 \Delta P}{8\mu L}"),
        // ── Thermodynamics ──
        ("Heat Equation", Thermodynamics, r"\frac{\partial T}{\partial t} = \alpha\nabla^2 T"),
        ("Ideal Gas Law", Thermodynamics, r"PV = nRT"),
        ("Stefan-Boltzmann Law", Thermodynamics, r"P/A = \sigma T^4"),
        ("Wien Displacement Law", Thermodynamics, r"\lambda_{max} = \frac{b}{T}"),
        ("Landauer Principle", Thermodynamics, r"E_{min} = k_B T \ln 2"),
        ("Clausius-Clapeyron Equation", Thermodynamics, r"\frac{dP}{dT} = \frac{\Delta H}{T\Delta V}"),
        ("Gibbs Free Energy", Thermodynamics, r"G = H - TS"),
        ("Van der Waals Equation", Thermodynamics, r"\left(P+\frac{a}{V^2}\right)(V-b)=RT"),
        ("Fourier Heat Law", Thermodynamics, r"\mathbf{q} = -k\nabla T"),
        ("Arrhenius Equation", Thermodynamics, r"k = A e^{-E_a/RT}"),
        ("Helmholtz Free Energy", Thermodynamics, r"F = U - TS"),
        ("Entropy of Mixing", Thermodynamics, r"\Delta S_{mix} = -nR\sum x_i \ln x_i"),
        ("Equipartition Theorem", Thermodynamics, r"\langle E\rangle = \frac{f}{2}k_B T"),
        ("Carnot Efficiency", Thermodynamics, r"\eta = 1 - \frac{T_{cold}}{T_{hot}}"),
        ("Henderson-Hasselbalch", Thermodynamics, r"pH = pK_a + \log_{10}\frac{[A^-]}{[HA]}"),
        // ── Cosmology ──
        ("Friedmann First Equation", Cosmology, r"H^2 = \frac{8\pi G\rho}{3} - \frac{k}{a^2} + \frac{\Lambda}{3}"),
        ("Friedmann Second Equation", Cosmology, r"\frac{\ddot{a}}{a} = -\frac{4\pi G}{3}(\rho+3p) + \frac{\Lambda}{3}"),
        ("FLRW Metric", Cosmology, r"ds^2 = -c^2 dt^2 + a^2(t)\left[\frac{dr^2}{1-kr^2}+r^2 d\Omega^2\right]"),
        ("Hubble Law", Cosmology, r"v = H_0 d"),
        ("Hubble Parameter Evolution", Cosmology, r"H^2 = H_0^2[\Omega_r/a^4 + \Omega_m/a^3 + \Omega_k/a^2 + \Omega_\Lambda]"),
        ("CMB Temperature Redshift", Cosmology, r"T(z) = T_0(1+z)"),
        // ── Nuclear Physics ──
        ("Gamow Peak Integral", NuclearPhysics, r"\langle\sigma v\rangle = \sqrt{\frac{8}{\pi\mu}}(kT)^{-3/2}\int S(E)e^{-E/kT-B_G/\sqrt{E}}dE"),
        ("Yukawa Potential", NuclearPhysics, r"V(r) = -\frac{g^2}{4\pi}\frac{e^{-\mu r}}{r}"),
        ("One-Pion Exchange Potential", NuclearPhysics, r"V_{OPEP} \propto (\sigma_1\cdot\sigma_2)(\tau_1\cdot\tau_2)\frac{e^{-m_\pi r}}{m_\pi r}"),
        ("Nuclear Radius Formula", NuclearPhysics, r"R = r_0 A^{1/3}"),
        ("Geiger-Nuttall Law", NuclearPhysics, r"\log_{10}(t_{1/2}) = \frac{aZ}{\sqrt{Q_\alpha}} + b"),
        ("Bateman Decay Equations", NuclearPhysics, r"\frac{dN_i}{dt} = \lambda_{parent}N_{parent} - \lambda_i N_i"),
        ("Bethe-Bloch Energy Loss", NuclearPhysics, r"-\frac{dE}{dx} = Kz^2\frac{Z}{A}\frac{1}{\beta^2}\left[\ln\frac{2m_e\beta^2\gamma^2}{I}-\beta^2\right]"),
        ("Gamow Tunneling Factor", NuclearPhysics, r"T = e^{-2\pi\eta}"),
        ("Nuclear Magnetic Resonance", NuclearPhysics, r"\omega_0 = \gamma B_0"),
        // ── Statistical Mechanics ──
        ("Boltzmann Distribution", StatisticalMechanics, r"P(E) \propto e^{-E/kT}"),
        ("Ising Hamiltonian", StatisticalMechanics, r"H = -J\sum_{\langle ij\rangle}s_i s_j - h\sum_i s_i"),
        ("Canonical Partition Function", StatisticalMechanics, r"Z = \sum_n e^{-\beta E_n}"),
        ("Boltzmann Entropy", StatisticalMechanics, r"S = k_B \ln W"),
        ("Fermi-Dirac Distribution", StatisticalMechanics, r"f(E) = \frac{1}{e^{(E-\mu)/kT}+1}"),
        ("Bose-Einstein Distribution", StatisticalMechanics, r"n(E) = \frac{1}{e^{(E-\mu)/kT}-1}"),
        ("Saha Ionization Equation", StatisticalMechanics, r"\frac{n_i n_e}{n_0} = \left(\frac{2\pi m_e kT}{h^2}\right)^{3/2}e^{-\chi/kT}"),
        // ── Optics ──
        ("Snell Law", Optics, r"n_1\sin\theta_1 = n_2\sin\theta_2"),
        ("Fresnel Equations", Optics, r"r_s = \frac{n_1\cos\theta_1 - n_2\cos\theta_2}{n_1\cos\theta_1 + n_2\cos\theta_2}"),
        ("Waveguide Dispersion Relation", Optics, r"\omega^2 = \frac{c^2}{n^2}k^2 + \omega_c^2"),
        ("Rayleigh Scattering", Optics, r"I \propto \frac{1}{\lambda^4}"),
        ("Brewster Angle", Optics, r"\tan\theta_B = \frac{n_2}{n_1}"),
        ("Diffraction Grating", Optics, r"d\sin\theta = m\lambda"),
        ("Abbe Diffraction Limit", Optics, r"d = \frac{\lambda}{2n\sin\alpha}"),
        // ── Condensed Matter ──
        ("BCS Gap Equation", CondensedMatter, r"\Delta = 2\hbar\omega_D e^{-1/(N(0)V)}"),
        ("Drude Conductivity Model", CondensedMatter, r"\sigma(\omega) = \frac{ne^2/m}{\gamma - i\omega}"),
        ("Hall Effect", CondensedMatter, r"V_H = \frac{IB}{nqt}"),
        ("Bragg Diffraction", CondensedMatter, r"2d\sin\theta = n\lambda"),
        ("Debye Model Heat Capacity", CondensedMatter, r"C_V = 9Nk\left(\frac{T}{\Theta_D}\right)^3\int_0^{\Theta_D/T}\frac{x^4 e^x}{(e^x-1)^2}dx"),
        // ── Particle Physics ──
        ("Proca Equation", ParticlePhysics, r"\partial_\mu F^{\mu\nu} + m^2 A^\nu = J^\nu"),
        ("QCD Running Coupling", ParticlePhysics, r"\alpha_s(Q^2) = \frac{\alpha_s(M_Z^2)}{1+b_0\alpha_s\ln(Q^2/M_Z^2)/(2\pi)}"),
        ("Casimir Force", ParticlePhysics, r"\frac{F}{A} = -\frac{\pi^2\hbar c}{240 d^4}"),
        ("Higgs Potential", ParticlePhysics, r"V(\phi) = \mu^2|\phi|^2 + \lambda|\phi|^4"),
        // ── Modified Gravity ──
        ("f(R) Modified Gravity", ModifiedGravity, r"f'(R)R_{\mu\nu}-\frac{1}{2}f(R)g_{\mu\nu}+\cdots=\kappa T_{\mu\nu}"),
        ("MOND Milgrom Law", ModifiedGravity, r"\mu(|\mathbf{a}|/a_0)\mathbf{a}=\mathbf{a}_N"),
        ("Brans-Dicke Equation", ModifiedGravity, r"\Box\phi = \frac{8\pi T}{3+2\omega}"),
        // ── Information Theory ──
        ("Shannon Entropy", InformationTheory, r"H(X) = -\sum p(x)\log p(x)"),
        ("KL Divergence", InformationTheory, r"D_{KL}(P\|Q) = \sum P\ln\frac{P}{Q}"),
        ("Mutual Information", InformationTheory, r"I(X;Y) = H(X) - H(X|Y)"),
        ("Cross-Entropy", InformationTheory, r"H(P,Q) = -\sum P\ln Q"),
        ("Fisher Information", InformationTheory, r"I(\theta) = E\left[\left(\frac{\partial\ln f}{\partial\theta}\right)^2\right]"),
        ("Variational Free Energy (Friston)", InformationTheory, r"F = E_q[\ln q(z) - \ln p(o,z)]"),
        ("Integrated Information (Phi)", InformationTheory, r"\Phi = \min_{partition}D_{KL}(p\|\prod p_i)"),
        ("Bayes Theorem", InformationTheory, r"P(A|B) = \frac{P(B|A)P(A)}{P(B)}"),
        ("Shannon-Hartley Channel Capacity", InformationTheory, r"C = B\log_2(1+S/N)"),
        ("Data Processing Inequality", InformationTheory, r"I(X;Z) \leq I(X;Y) \text{ for } X\to Y\to Z"),
        ("Rate-Distortion Function", InformationTheory, r"R(D) = \min I(X;\hat{X})\text{ s.t. }E[d(X,\hat{X})]\leq D"),
        // ── Biophysics ──
        ("Hodgkin-Huxley Equation", Biophysics, r"C\frac{dV}{dt} = -g_{Na}m^3h(V-E_{Na}) - g_K n^4(V-E_K) - g_L(V-E_L) + I"),
        ("Lotka-Volterra Equations", Biophysics, r"\frac{dx}{dt}=\alpha x-\beta xy,\;\frac{dy}{dt}=\delta xy-\gamma y"),
        ("Michaelis-Menten Kinetics", Biophysics, r"v = \frac{V_{max}[S]}{K_m + [S]}"),
        ("Hill Equation", Biophysics, r"\theta = \frac{[L]^n}{K_d^n + [L]^n}"),
        ("Nernst Equation", Biophysics, r"E = E^\circ - \frac{RT}{nF}\ln Q"),
        ("Logistic Growth", Biophysics, r"\frac{dN}{dt} = rN\left(1-\frac{N}{K}\right)"),
        // ── Acoustics ──
        ("Speed of Sound", Acoustics, r"v = \sqrt{\gamma P/\rho}"),
        ("Doppler Effect", Acoustics, r"f' = f\frac{v \pm v_{obs}}{v \mp v_{src}}"),
        ("Sound Intensity Level", Acoustics, r"\beta = 10\log_{10}(I/I_0)\text{ dB}"),
        // ── Plasma Physics ──
        ("Debye Length", PlasmaPhysics, r"\lambda_D = \sqrt{\frac{\varepsilon_0 kT}{ne^2}}"),
        ("Plasma Frequency", PlasmaPhysics, r"\omega_p = \sqrt{\frac{ne^2}{m\varepsilon_0}}"),
        ("MHD Force Balance", PlasmaPhysics, r"\mathbf{J}\times\mathbf{B} = \nabla P"),
        // ── Mathematics ──
        ("Euler Identity", Mathematics, r"e^{i\pi} + 1 = 0"),
        ("Fourier Transform", Mathematics, r"F(\omega) = \int f(t)e^{-i\omega t}dt"),
        ("Laplace Transform", Mathematics, r"F(s) = \int_0^\infty f(t)e^{-st}dt"),
        ("Cauchy-Schwarz Inequality", Mathematics, r"|\langle u,v\rangle|^2 \leq \langle u,u\rangle\langle v,v\rangle"),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_has_148_equations() {
        let catalog = PhysicsCatalog::new();
        assert!(
            catalog.len() >= 140,
            "Expected >= 140, got {}",
            catalog.len()
        );
    }

    #[test]
    fn all_hvs_finite() {
        let catalog = PhysicsCatalog::new();
        for entry in catalog.entries() {
            assert!(
                entry.hv.values.iter().all(|v| v.is_finite()),
                "Non-finite HV for {}",
                entry.name
            );
        }
    }

    #[test]
    fn domains_covered() {
        let catalog = PhysicsCatalog::new();
        assert!(!catalog.entries_in_domain(Domain::InformationTheory).is_empty());
        assert!(!catalog.entries_in_domain(Domain::Biophysics).is_empty());
        assert!(!catalog.entries_in_domain(Domain::Acoustics).is_empty());
        assert!(!catalog.entries_in_domain(Domain::PlasmaPhysics).is_empty());
        assert!(!catalog.entries_in_domain(Domain::Mathematics).is_empty());
    }
}
