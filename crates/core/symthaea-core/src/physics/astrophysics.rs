// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Astrophysics
//!
//! This module encodes astrophysical concepts from stellar classification to cosmology.
//!
//! ## Stellar Classification
//!
//! Stars are classified by spectral class (O, B, A, F, G, K, M) based on temperature,
//! and luminosity class (Ia to VII) based on intrinsic brightness.
//!
//! ## Stellar Nucleosynthesis
//!
//! Elements heavier than hydrogen are created in stars through various processes:
//! - **pp-chain**: 4H → He (main sequence, like the Sun)
//! - **CNO cycle**: 4H → He (catalyzed by C, N, O - hot massive stars)
//! - **Triple-alpha**: 3He → C (red giants)
//! - **s-process**: Slow neutron capture (AGB stars)
//! - **r-process**: Rapid neutron capture (supernovae, neutron star mergers)
//!
//! ## Compact Objects
//!
//! - **White dwarfs**: Electron degenerate matter
//! - **Neutron stars**: Neutron degenerate matter (~10^57 neutrons)
//! - **Black holes**: Spacetime singularity (no hair theorem: mass, spin, charge)

use super::hadrons::Hadrons;
use super::nuclear::NuclearPhysics;
use super::periodic_table::PeriodicTable;
use super::standard_model::PHYSICS_DIM;
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// Stellar spectral class (Harvard classification)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpectralClass {
    /// Hot blue stars (>30,000 K)
    O,
    /// Blue-white stars (10,000-30,000 K)
    B,
    /// White stars (7,500-10,000 K)
    A,
    /// Yellow-white stars (6,000-7,500 K)
    F,
    /// Yellow stars like the Sun (5,200-6,000 K)
    G,
    /// Orange stars (3,700-5,200 K)
    K,
    /// Red stars (<3,700 K)
    M,
}

impl SpectralClass {
    fn domain_label(&self) -> &'static str {
        match self {
            SpectralClass::O => "stellar::spectral_o",
            SpectralClass::B => "stellar::spectral_b",
            SpectralClass::A => "stellar::spectral_a",
            SpectralClass::F => "stellar::spectral_f",
            SpectralClass::G => "stellar::spectral_g",
            SpectralClass::K => "stellar::spectral_k",
            SpectralClass::M => "stellar::spectral_m",
        }
    }

    /// Typical temperature for this spectral class
    pub fn typical_temperature(&self) -> f64 {
        match self {
            SpectralClass::O => 40000.0,
            SpectralClass::B => 20000.0,
            SpectralClass::A => 8500.0,
            SpectralClass::F => 6750.0,
            SpectralClass::G => 5600.0,
            SpectralClass::K => 4500.0,
            SpectralClass::M => 3000.0,
        }
    }
}

/// Luminosity class (Yerkes classification)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LuminosityClass {
    /// Luminous supergiant
    Ia,
    /// Less luminous supergiant
    Ib,
    /// Bright giant
    II,
    /// Giant
    III,
    /// Subgiant
    IV,
    /// Main sequence (dwarf)
    V,
    /// Subdwarf
    VI,
    /// White dwarf
    VII,
}

impl LuminosityClass {
    fn domain_label(&self) -> &'static str {
        match self {
            LuminosityClass::Ia => "stellar::luminosity_ia",
            LuminosityClass::Ib => "stellar::luminosity_ib",
            LuminosityClass::II => "stellar::luminosity_ii",
            LuminosityClass::III => "stellar::luminosity_iii",
            LuminosityClass::IV => "stellar::luminosity_iv",
            LuminosityClass::V => "stellar::luminosity_v",
            LuminosityClass::VI => "stellar::luminosity_vi",
            LuminosityClass::VII => "stellar::luminosity_vii",
        }
    }
}

/// A star
#[derive(Debug, Clone)]
pub struct Star {
    /// Name or designation
    pub name: String,
    /// Spectral class
    pub spectral: SpectralClass,
    /// Luminosity class
    pub luminosity: LuminosityClass,
    /// Mass in solar masses
    pub mass_solar: f64,
    /// Radius in solar radii
    pub radius_solar: f64,
    /// Surface temperature in Kelvin
    pub temperature_k: f64,
    /// Age in billions of years
    pub age_gyr: f64,
    /// Vector representation
    pub vector: ContinuousHV,
}

/// Stellar encoder
#[derive(Debug, Clone)]
pub struct StellarEncoder {
    // Spectral class vectors
    pub class_o: ContinuousHV,
    pub class_b: ContinuousHV,
    pub class_a: ContinuousHV,
    pub class_f: ContinuousHV,
    pub class_g: ContinuousHV,
    pub class_k: ContinuousHV,
    pub class_m: ContinuousHV,

    // Luminosity class vectors
    pub lum_ia: ContinuousHV,
    pub lum_ib: ContinuousHV,
    pub lum_ii: ContinuousHV,
    pub lum_iii: ContinuousHV,
    pub lum_iv: ContinuousHV,
    pub lum_v: ContinuousHV,
    pub lum_vi: ContinuousHV,
    pub lum_vii: ContinuousHV,

    // Evolution stages
    pub main_sequence: ContinuousHV,
    pub red_giant: ContinuousHV,
    pub white_dwarf: ContinuousHV,
    pub neutron_star: ContinuousHV,
    pub black_hole: ContinuousHV,

    // Property concepts
    pub mass: ContinuousHV,
    pub radius: ContinuousHV,
    pub temperature: ContinuousHV,
    pub luminosity: ContinuousHV,
    pub spin_vector: ContinuousHV,
}

impl StellarEncoder {
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        Self {
            class_o: genesis.hv(SpectralClass::O.domain_label(), PHYSICS_DIM),
            class_b: genesis.hv(SpectralClass::B.domain_label(), PHYSICS_DIM),
            class_a: genesis.hv(SpectralClass::A.domain_label(), PHYSICS_DIM),
            class_f: genesis.hv(SpectralClass::F.domain_label(), PHYSICS_DIM),
            class_g: genesis.hv(SpectralClass::G.domain_label(), PHYSICS_DIM),
            class_k: genesis.hv(SpectralClass::K.domain_label(), PHYSICS_DIM),
            class_m: genesis.hv(SpectralClass::M.domain_label(), PHYSICS_DIM),

            lum_ia: genesis.hv(LuminosityClass::Ia.domain_label(), PHYSICS_DIM),
            lum_ib: genesis.hv(LuminosityClass::Ib.domain_label(), PHYSICS_DIM),
            lum_ii: genesis.hv(LuminosityClass::II.domain_label(), PHYSICS_DIM),
            lum_iii: genesis.hv(LuminosityClass::III.domain_label(), PHYSICS_DIM),
            lum_iv: genesis.hv(LuminosityClass::IV.domain_label(), PHYSICS_DIM),
            lum_v: genesis.hv(LuminosityClass::V.domain_label(), PHYSICS_DIM),
            lum_vi: genesis.hv(LuminosityClass::VI.domain_label(), PHYSICS_DIM),
            lum_vii: genesis.hv(LuminosityClass::VII.domain_label(), PHYSICS_DIM),

            main_sequence: genesis.hv("stellar::main_sequence", PHYSICS_DIM),
            red_giant: genesis.hv("stellar::red_giant", PHYSICS_DIM),
            white_dwarf: genesis.hv("stellar::white_dwarf", PHYSICS_DIM),
            neutron_star: genesis.hv("stellar::neutron_star", PHYSICS_DIM),
            black_hole: genesis.hv("stellar::black_hole", PHYSICS_DIM),

            mass: genesis.hv("stellar::mass", PHYSICS_DIM),
            radius: genesis.hv("stellar::radius", PHYSICS_DIM),
            temperature: genesis.hv("stellar::temperature", PHYSICS_DIM),
            luminosity: genesis.hv("stellar::luminosity", PHYSICS_DIM),
            spin_vector: genesis.hv("stellar::spin", PHYSICS_DIM),
        }
    }

    /// Create StellarEncoder from nuclear physics
    ///
    /// This constructor grounds stellar concepts in nuclear energy scales,
    /// connecting stellar evolution to nuclear fusion processes.
    pub fn from_nuclear(nuclear: &NuclearPhysics, genesis: &GenesisSeed) -> Self {
        // Main sequence is powered by nuclear fusion
        let main_sequence = nuclear
            .nuclear_energy
            .bind(&genesis.hv("stellar::main_sequence", PHYSICS_DIM));

        // White dwarf: electron degeneracy, sub-nuclear energy scale
        let white_dwarf = ContinuousHV::bundle(&[
            &nuclear.chemical_energy,
            &genesis.hv("stellar::white_dwarf", PHYSICS_DIM),
        ]);

        // Neutron star: nuclear degeneracy
        let neutron_star = nuclear
            .nuclear_energy
            .bind(&genesis.hv("stellar::neutron_star", PHYSICS_DIM));

        // Black hole: relativistic energy scale
        let black_hole = nuclear
            .relativistic_energy
            .bind(&genesis.hv("stellar::black_hole", PHYSICS_DIM));

        Self {
            class_o: genesis.hv(SpectralClass::O.domain_label(), PHYSICS_DIM),
            class_b: genesis.hv(SpectralClass::B.domain_label(), PHYSICS_DIM),
            class_a: genesis.hv(SpectralClass::A.domain_label(), PHYSICS_DIM),
            class_f: genesis.hv(SpectralClass::F.domain_label(), PHYSICS_DIM),
            class_g: genesis.hv(SpectralClass::G.domain_label(), PHYSICS_DIM),
            class_k: genesis.hv(SpectralClass::K.domain_label(), PHYSICS_DIM),
            class_m: genesis.hv(SpectralClass::M.domain_label(), PHYSICS_DIM),

            lum_ia: genesis.hv(LuminosityClass::Ia.domain_label(), PHYSICS_DIM),
            lum_ib: genesis.hv(LuminosityClass::Ib.domain_label(), PHYSICS_DIM),
            lum_ii: genesis.hv(LuminosityClass::II.domain_label(), PHYSICS_DIM),
            lum_iii: genesis.hv(LuminosityClass::III.domain_label(), PHYSICS_DIM),
            lum_iv: genesis.hv(LuminosityClass::IV.domain_label(), PHYSICS_DIM),
            lum_v: genesis.hv(LuminosityClass::V.domain_label(), PHYSICS_DIM),
            lum_vi: genesis.hv(LuminosityClass::VI.domain_label(), PHYSICS_DIM),
            lum_vii: genesis.hv(LuminosityClass::VII.domain_label(), PHYSICS_DIM),

            main_sequence,
            red_giant: genesis.hv("stellar::red_giant", PHYSICS_DIM),
            white_dwarf,
            neutron_star,
            black_hole,

            mass: genesis.hv("stellar::mass", PHYSICS_DIM),
            radius: genesis.hv("stellar::radius", PHYSICS_DIM),
            temperature: genesis.hv("stellar::temperature", PHYSICS_DIM),
            luminosity: genesis.hv("stellar::luminosity", PHYSICS_DIM),
            spin_vector: genesis.hv("stellar::spin", PHYSICS_DIM),
        }
    }

    fn spectral_vector(&self, class: SpectralClass) -> &ContinuousHV {
        match class {
            SpectralClass::O => &self.class_o,
            SpectralClass::B => &self.class_b,
            SpectralClass::A => &self.class_a,
            SpectralClass::F => &self.class_f,
            SpectralClass::G => &self.class_g,
            SpectralClass::K => &self.class_k,
            SpectralClass::M => &self.class_m,
        }
    }

    fn luminosity_vector(&self, class: LuminosityClass) -> &ContinuousHV {
        match class {
            LuminosityClass::Ia => &self.lum_ia,
            LuminosityClass::Ib => &self.lum_ib,
            LuminosityClass::II => &self.lum_ii,
            LuminosityClass::III => &self.lum_iii,
            LuminosityClass::IV => &self.lum_iv,
            LuminosityClass::V => &self.lum_v,
            LuminosityClass::VI => &self.lum_vi,
            LuminosityClass::VII => &self.lum_vii,
        }
    }

    /// Encode a star from its properties
    #[allow(clippy::too_many_arguments)]
    pub fn encode_star(
        &self,
        name: &str,
        spectral: SpectralClass,
        luminosity: LuminosityClass,
        mass_solar: f64,
        radius_solar: f64,
        temperature_k: f64,
        age_gyr: f64,
    ) -> Star {
        let spec_vec = self.spectral_vector(spectral);
        let lum_vec = self.luminosity_vector(luminosity);

        // Scale property vectors
        let mass_scaled = self.mass.scale(mass_solar.ln() as f32 + 1.0);
        let radius_scaled = self.radius.scale(radius_solar.ln() as f32 + 1.0);
        let temp_scaled = self.temperature.scale((temperature_k / 5800.0) as f32);

        // Bundle all properties
        let vector = ContinuousHV::weighted_bundle(
            &[
                spec_vec,
                lum_vec,
                &mass_scaled,
                &radius_scaled,
                &temp_scaled,
            ],
            &[1.0, 1.0, 0.5, 0.5, 0.5],
        );

        Star {
            name: name.to_string(),
            spectral,
            luminosity,
            mass_solar,
            radius_solar,
            temperature_k,
            age_gyr,
            vector,
        }
    }

    /// The Sun: G2V star
    pub fn sun(&self) -> Star {
        self.encode_star(
            "Sun",
            SpectralClass::G,
            LuminosityClass::V,
            1.0,
            1.0,
            5778.0,
            4.6,
        )
    }
}

/// Nucleosynthesis process
#[derive(Debug, Clone)]
pub struct NucleosynthesisProcess {
    /// Process name
    pub name: String,
    /// Reactant vectors
    pub reactants: Vec<ContinuousHV>,
    /// Product vectors
    pub products: Vec<ContinuousHV>,
    /// Temperature required (Kelvin)
    pub temperature_k: f64,
    /// Energy released (MeV)
    pub energy_mev: f64,
    /// Vector representation
    pub vector: ContinuousHV,
}

impl StellarEncoder {
    /// Proton-proton chain: 4H → He + 2e⁺ + 2ν + 26.7 MeV
    pub fn pp_chain(&self, table: &PeriodicTable) -> NucleosynthesisProcess {
        let h = table
            .element(1)
            .map(|e| e.vector.clone())
            .unwrap_or_else(|| ContinuousHV::zero(PHYSICS_DIM));
        let he = table
            .element(2)
            .map(|e| e.vector.clone())
            .unwrap_or_else(|| ContinuousHV::zero(PHYSICS_DIM));

        let vector = h.scale(4.0).bind(&he);

        NucleosynthesisProcess {
            name: "pp-chain".to_string(),
            reactants: vec![h],
            products: vec![he],
            temperature_k: 15e6,
            energy_mev: 26.7,
            vector,
        }
    }

    /// CNO cycle: 4H → He (catalyzed by C, N, O)
    pub fn cno_cycle(&self, table: &PeriodicTable) -> NucleosynthesisProcess {
        let h = table
            .element(1)
            .map(|e| e.vector.clone())
            .unwrap_or_else(|| ContinuousHV::zero(PHYSICS_DIM));
        let he = table
            .element(2)
            .map(|e| e.vector.clone())
            .unwrap_or_else(|| ContinuousHV::zero(PHYSICS_DIM));
        let c = table
            .element(6)
            .map(|e| e.vector.clone())
            .unwrap_or_else(|| ContinuousHV::zero(PHYSICS_DIM));

        // Catalyzed process: C acts as catalyst
        let vector = h.scale(4.0).bind(&he).bind(&c);

        NucleosynthesisProcess {
            name: "CNO-cycle".to_string(),
            reactants: vec![h],
            products: vec![he],
            temperature_k: 17e6,
            energy_mev: 26.7,
            vector,
        }
    }

    /// Triple-alpha process: 3He → C + 7.27 MeV
    pub fn triple_alpha(&self, table: &PeriodicTable) -> NucleosynthesisProcess {
        let he = table
            .element(2)
            .map(|e| e.vector.clone())
            .unwrap_or_else(|| ContinuousHV::zero(PHYSICS_DIM));
        let c = table
            .element(6)
            .map(|e| e.vector.clone())
            .unwrap_or_else(|| ContinuousHV::zero(PHYSICS_DIM));

        let vector = he.scale(3.0).bind(&c);

        NucleosynthesisProcess {
            name: "triple-alpha".to_string(),
            reactants: vec![he],
            products: vec![c],
            temperature_k: 100e6,
            energy_mev: 7.27,
            vector,
        }
    }

    /// s-process: slow neutron capture (AGB stars)
    pub fn s_process(&self, seed: &ContinuousHV, neutron: &ContinuousHV) -> ContinuousHV {
        // Slow capture: seed + n → heavier element
        seed.bind(neutron).normalize()
    }

    /// r-process: rapid neutron capture (supernovae, neutron star mergers)
    pub fn r_process(&self, seed: &ContinuousHV, neutron: &ContinuousHV) -> ContinuousHV {
        // Rapid capture: many neutrons captured before decay
        let multi_neutron = neutron.scale(10.0);
        seed.bind(&multi_neutron).normalize()
    }
}

/// Neutron star
#[derive(Debug, Clone)]
pub struct NeutronStar {
    /// Name
    pub name: String,
    /// Mass in solar masses
    pub mass_solar: f64,
    /// Radius in km
    pub radius_km: f64,
    /// Rotation period in milliseconds
    pub period_ms: f64,
    /// Magnetic field in Gauss
    pub magnetic_field_gauss: f64,
    /// Is it a pulsar?
    pub is_pulsar: bool,
    /// Vector representation
    pub vector: ContinuousHV,
}

impl StellarEncoder {
    /// Encode a neutron star
    pub fn encode_neutron_star(
        &self,
        name: &str,
        mass_solar: f64,
        hadrons: &Hadrons,
    ) -> NeutronStar {
        // Neutron star is composed primarily of neutrons at nuclear density
        // ~10^57 neutrons per solar mass
        let neutron_count = mass_solar * 1.2e57;
        let core = hadrons.neutron.scale(neutron_count.ln() as f32);

        // Radius from TOV equation approximation
        let radius_km = 10.0 * (1.4 / mass_solar).powf(1.0 / 3.0);

        let degeneracy = self.neutron_star.bind(&core);

        NeutronStar {
            name: name.to_string(),
            mass_solar,
            radius_km,
            period_ms: 1.0, // Default millisecond pulsar
            magnetic_field_gauss: 1e12,
            is_pulsar: true,
            vector: degeneracy,
        }
    }
}

/// Black hole
#[derive(Debug, Clone)]
pub struct BlackHole {
    /// Name
    pub name: String,
    /// Mass in solar masses
    pub mass_solar: f64,
    /// Kerr spin parameter (a/M), range [0, 1]
    pub spin: f64,
    /// Charge (usually 0)
    pub charge: f64,
    /// Schwarzschild radius in km
    pub schwarzschild_radius_km: f64,
    /// Hawking temperature in Kelvin
    pub hawking_temperature_k: f64,
    /// Vector representation
    pub vector: ContinuousHV,
}

impl StellarEncoder {
    /// Encode a black hole
    ///
    /// No hair theorem: fully described by mass, spin, charge
    pub fn encode_black_hole(&self, name: &str, mass_solar: f64, spin: f64) -> BlackHole {
        // Schwarzschild radius: r_s = 2GM/c² ≈ 2.95 km per solar mass
        let rs = 2.95 * mass_solar;

        // Hawking temperature: T = ℏc³/(8πGMk_B) ≈ 6.17e-8 K / (M/M_sun)
        let t_hawking = 6.17e-8 / mass_solar;

        // Vector: mass-scaled black hole concept + spin
        let vector = self
            .black_hole
            .scale(mass_solar.ln() as f32 + 1.0)
            .bind(&self.spin_vector.scale(spin as f32));

        BlackHole {
            name: name.to_string(),
            mass_solar,
            spin,
            charge: 0.0,
            schwarzschild_radius_km: rs,
            hawking_temperature_k: t_hawking,
            vector,
        }
    }

    /// Stellar mass black hole (3-100 solar masses)
    pub fn stellar_black_hole(&self, mass_solar: f64) -> BlackHole {
        self.encode_black_hole("Stellar BH", mass_solar.clamp(3.0, 100.0), 0.7)
    }

    /// Supermassive black hole (10^6 - 10^10 solar masses)
    pub fn supermassive_black_hole(&self, mass_solar: f64) -> BlackHole {
        self.encode_black_hole("SMBH", mass_solar, 0.9)
    }
}

/// Galaxy morphology
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GalaxyType {
    Spiral,
    Elliptical,
    Irregular,
    Lenticular,
}

/// Galaxy
#[derive(Debug, Clone)]
pub struct Galaxy {
    /// Name
    pub name: String,
    /// Morphology type
    pub morphology: GalaxyType,
    /// Total mass in solar masses
    pub mass_solar: f64,
    /// Approximate star count
    pub star_count: f64,
    /// Dark matter fraction
    pub dark_matter_fraction: f64,
    /// Vector representation
    pub vector: ContinuousHV,
}

/// Cosmological encoder
#[derive(Debug, Clone)]
pub struct CosmologicalEncoder {
    // Epochs
    pub big_bang: ContinuousHV,
    pub inflation: ContinuousHV,
    pub recombination: ContinuousHV,
    pub reionization: ContinuousHV,

    // Components
    pub dark_matter: ContinuousHV,
    pub dark_energy: ContinuousHV,
    pub baryonic_matter: ContinuousHV,

    // Observables
    pub cosmic_microwave_background: ContinuousHV,
    pub redshift: ContinuousHV,
    pub hubble_constant: ContinuousHV,

    // Galaxy types
    pub spiral: ContinuousHV,
    pub elliptical: ContinuousHV,
    pub irregular: ContinuousHV,
    pub lenticular: ContinuousHV,
}

impl CosmologicalEncoder {
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        Self {
            big_bang: genesis.hv("cosmo::big_bang", PHYSICS_DIM),
            inflation: genesis.hv("cosmo::inflation", PHYSICS_DIM),
            recombination: genesis.hv("cosmo::recombination", PHYSICS_DIM),
            reionization: genesis.hv("cosmo::reionization", PHYSICS_DIM),
            dark_matter: genesis.hv("cosmo::dark_matter", PHYSICS_DIM),
            dark_energy: genesis.hv("cosmo::dark_energy", PHYSICS_DIM),
            baryonic_matter: genesis.hv("cosmo::baryonic_matter", PHYSICS_DIM),
            cosmic_microwave_background: genesis.hv("cosmo::cmb", PHYSICS_DIM),
            redshift: genesis.hv("cosmo::redshift", PHYSICS_DIM),
            hubble_constant: genesis.hv("cosmo::hubble", PHYSICS_DIM),
            spiral: genesis.hv("galaxy::spiral", PHYSICS_DIM),
            elliptical: genesis.hv("galaxy::elliptical", PHYSICS_DIM),
            irregular: genesis.hv("galaxy::irregular", PHYSICS_DIM),
            lenticular: genesis.hv("galaxy::lenticular", PHYSICS_DIM),
        }
    }

    fn galaxy_type_vector(&self, gtype: GalaxyType) -> &ContinuousHV {
        match gtype {
            GalaxyType::Spiral => &self.spiral,
            GalaxyType::Elliptical => &self.elliptical,
            GalaxyType::Irregular => &self.irregular,
            GalaxyType::Lenticular => &self.lenticular,
        }
    }

    /// Encode a galaxy
    pub fn encode_galaxy(
        &self,
        name: &str,
        morphology: GalaxyType,
        mass_solar: f64,
        star_count: f64,
        dark_matter_fraction: f64,
    ) -> Galaxy {
        let type_vec = self.galaxy_type_vector(morphology);
        let dm_vec = self.dark_matter.scale(dark_matter_fraction as f32);
        let bary_vec = self
            .baryonic_matter
            .scale((1.0 - dark_matter_fraction) as f32);

        let vector =
            ContinuousHV::weighted_bundle(&[type_vec, &dm_vec, &bary_vec], &[1.0, 0.5, 0.5])
                .scale(mass_solar.log10() as f32 / 12.0);

        Galaxy {
            name: name.to_string(),
            morphology,
            mass_solar,
            star_count,
            dark_matter_fraction,
            vector,
        }
    }

    /// The Milky Way
    pub fn milky_way(&self) -> Galaxy {
        self.encode_galaxy(
            "Milky Way",
            GalaxyType::Spiral,
            1.5e12, // 1.5 trillion solar masses (including dark matter)
            2e11,   // 200 billion stars
            0.84,   // ~84% dark matter
        )
    }

    /// Encode cosmic epoch by redshift
    pub fn encode_epoch(&self, redshift: f64) -> ContinuousHV {
        // z = 0: present
        // z ~ 1100: recombination (CMB)
        // z ~ 6-20: reionization
        // z > 1100: pre-recombination

        let z_scaled = (redshift.log10() + 1.0) as f32 / 4.0;

        if redshift > 1100.0 {
            self.big_bang.scale(z_scaled)
        } else if redshift > 6.0 {
            self.recombination
                .scale(z_scaled)
                .bind(&self.cosmic_microwave_background)
        } else {
            self.reionization.scale(z_scaled)
        }
    }

    /// Universe composition fractions (ΛCDM model)
    pub fn universe_composition(&self) -> ContinuousHV {
        // ~68% dark energy, ~27% dark matter, ~5% baryonic
        ContinuousHV::weighted_bundle(
            &[&self.dark_energy, &self.dark_matter, &self.baryonic_matter],
            &[0.68, 0.27, 0.05],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (StellarEncoder, CosmologicalEncoder, GenesisSeed) {
        let genesis = GenesisSeed::from_phrase("astrophysics test");
        let stellar = StellarEncoder::from_genesis(&genesis);
        let cosmo = CosmologicalEncoder::from_genesis(&genesis);
        (stellar, cosmo, genesis)
    }

    #[test]
    fn test_sun_encoding() {
        let (encoder, _, _) = setup();

        let sun = encoder.sun();

        assert_eq!(sun.spectral, SpectralClass::G);
        assert_eq!(sun.luminosity, LuminosityClass::V);
        assert!((sun.mass_solar - 1.0).abs() < 0.001);
        assert!((sun.temperature_k - 5778.0).abs() < 1.0);
    }

    #[test]
    fn test_stellar_classification_similarity() {
        let (encoder, _, _) = setup();

        // Similar stars should have similar vectors
        let sun = encoder.sun();
        let alpha_cen = encoder.encode_star(
            "Alpha Centauri A",
            SpectralClass::G,
            LuminosityClass::V,
            1.1,
            1.2,
            5790.0,
            5.3,
        );

        // Both G2V stars should be similar
        let sim = sun.vector.similarity(&alpha_cen.vector);
        assert!(
            sim > 0.8,
            "Similar stars should have similar vectors: {}",
            sim
        );

        // O star should be different
        let hot_star = encoder.encode_star(
            "Hot Star",
            SpectralClass::O,
            LuminosityClass::V,
            30.0,
            10.0,
            40000.0,
            0.001,
        );
        let sim_hot = sun.vector.similarity(&hot_star.vector);
        assert!(
            sim_hot < sim,
            "Different spectral classes should be less similar"
        );
    }

    #[test]
    fn test_nucleosynthesis() {
        let genesis = GenesisSeed::from_phrase("nucleosynthesis test");
        let stellar = StellarEncoder::from_genesis(&genesis);
        let model = super::super::StandardModel::from_genesis(&genesis);
        let hadrons = Hadrons::from_model(&model, &genesis);
        let table = PeriodicTable::from_model(&model, &hadrons, &genesis);

        let pp = stellar.pp_chain(&table);
        assert_eq!(pp.name, "pp-chain");
        assert!((pp.temperature_k - 15e6).abs() < 1e6);
        assert!((pp.energy_mev - 26.7).abs() < 0.1);

        let triple = stellar.triple_alpha(&table);
        assert!(triple.temperature_k > pp.temperature_k);
    }

    #[test]
    fn test_neutron_star() {
        let genesis = GenesisSeed::from_phrase("neutron star test");
        let stellar = StellarEncoder::from_genesis(&genesis);
        let model = super::super::StandardModel::from_genesis(&genesis);
        let hadrons = Hadrons::from_model(&model, &genesis);

        let ns = stellar.encode_neutron_star("PSR B1919+21", 1.4, &hadrons);

        assert!((ns.mass_solar - 1.4).abs() < 0.01);
        assert!(ns.radius_km > 8.0 && ns.radius_km < 15.0);
    }

    #[test]
    fn test_black_hole() {
        let (encoder, _, _) = setup();

        let stellar_bh = encoder.stellar_black_hole(10.0);
        assert!((stellar_bh.schwarzschild_radius_km - 29.5).abs() < 0.1);

        let smbh = encoder.supermassive_black_hole(4e6); // Sgr A*
        assert!(smbh.schwarzschild_radius_km > 1e7);

        // Hawking temperature: stellar BH should be hotter than SMBH
        assert!(stellar_bh.hawking_temperature_k > smbh.hawking_temperature_k);
    }

    #[test]
    fn test_milky_way() {
        let (_, cosmo, _) = setup();

        let mw = cosmo.milky_way();

        assert_eq!(mw.morphology, GalaxyType::Spiral);
        assert!(mw.mass_solar > 1e12);
        assert!(mw.dark_matter_fraction > 0.8);
    }

    #[test]
    fn test_universe_composition() {
        let (_, cosmo, _) = setup();

        let composition = cosmo.universe_composition();
        assert_eq!(composition.dim(), PHYSICS_DIM);
        assert!(composition.norm() > 0.0);
    }

    #[test]
    fn test_cosmic_epochs() {
        let (_, cosmo, _) = setup();

        let present = cosmo.encode_epoch(0.0);
        let cmb = cosmo.encode_epoch(1100.0);
        let early = cosmo.encode_epoch(10000.0);

        // All should have valid dimension
        assert_eq!(present.dim(), PHYSICS_DIM);
        assert_eq!(cmb.dim(), PHYSICS_DIM);
        assert_eq!(early.dim(), PHYSICS_DIM);
    }

    #[test]
    fn test_determinism() {
        let genesis = GenesisSeed::from_phrase("determinism test");
        let enc1 = StellarEncoder::from_genesis(&genesis);
        let enc2 = StellarEncoder::from_genesis(&genesis);

        assert!(
            enc1.class_g.similarity(&enc2.class_g) > 0.9999,
            "Stellar encoder should be deterministic"
        );
    }
}
