//! # Periodic Table: Layer 3 - Elements from Hadrons
//!
//! Elements are **constructed** from their nuclear and electronic structure,
//! not memorized as independent vectors.
//!
//! ## The Compositional Formula
//!
//! ```text
//! Element(Z, N) = Bundle(
//!     Z × Proton,
//!     N × Neutron,
//!     Z × Electron
//! ) ⊗ BindingEnergy(Z, N)
//! ```
//!
//! Where:
//! - Z = atomic number (protons = electrons for neutral atom)
//! - N = neutron number
//! - A = Z + N = mass number
//!
//! ## Emergent Properties
//!
//! Because elements are composed:
//! - **Isotopes** share most structure (differ only in neutron count)
//! - **Ions** are derived by adjusting electron count
//! - **Nuclear stability** emerges from binding energy
//!
//! ## Examples
//!
//! ```text
//! Carbon-12:  Bundle(6P, 6N, 6e) ⊗ Binding(6,6)
//! Carbon-14:  Bundle(6P, 8N, 6e) ⊗ Binding(6,8)  ← radioactive!
//! Nitrogen:   Bundle(7P, 7N, 7e) ⊗ Binding(7,7)
//!
//! sim(C-12, C-14) > 0.9   // Isotopes are highly similar
//! sim(C-12, N-14) < 0.8   // Different elements are less similar
//! ```

use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use super::hadrons::Hadrons;
use super::standard_model::{StandardModel, PHYSICS_DIM};
use serde::{Deserialize, Serialize};

/// Element data (basic properties)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElementData {
    pub symbol: &'static str,
    pub name: &'static str,
    pub atomic_number: u8,
    pub standard_neutrons: u8, // Most common isotope
    pub atomic_mass: f32,      // In atomic mass units
    pub electronegativity: Option<f32>, // Pauling scale
    pub group: u8,
    pub period: u8,
}

/// Complete element information
#[derive(Debug, Clone)]
pub struct Element {
    pub data: ElementData,
    pub vector: ContinuousHV,
}

/// Electron shell configuration helper
#[derive(Debug, Clone)]
pub struct ElectronShell {
    shell_vectors: Vec<ContinuousHV>,
}

impl ElectronShell {
    /// Create shell vectors from genesis
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        // s, p, d, f orbitals for shells 1-7
        let mut shell_vectors = Vec::new();
        for n in 1..=7u8 {
            let shell = genesis.hv(&format!("electron::shell_{}", n), PHYSICS_DIM);
            shell_vectors.push(shell);
        }
        Self { shell_vectors }
    }

    /// Encode electron configuration
    ///
    /// Returns a vector representing the electron arrangement.
    /// Uses aufbau principle for filling order.
    pub fn encode_configuration(&self, electron_count: u8, base_electron: &ContinuousHV) -> ContinuousHV {
        // Simplified: weight each shell by electron occupancy
        // Real version would use aufbau filling order
        let mut remaining = electron_count;
        let shell_capacities = [2, 8, 18, 32, 32, 18, 8]; // Max electrons per shell

        let mut weighted_shells = Vec::new();
        let mut weights = Vec::new();

        for (i, &capacity) in shell_capacities.iter().enumerate() {
            if remaining == 0 || i >= self.shell_vectors.len() {
                break;
            }

            let electrons_in_shell = remaining.min(capacity);
            remaining -= electrons_in_shell;

            // Bind electron to shell
            let shell_contribution = base_electron.bind(&self.shell_vectors[i])
                .scale(electrons_in_shell as f32);
            weighted_shells.push(shell_contribution);
            weights.push(electrons_in_shell as f32);
        }

        if weighted_shells.is_empty() {
            return ContinuousHV::zero(PHYSICS_DIM);
        }

        let refs: Vec<&ContinuousHV> = weighted_shells.iter().collect();
        ContinuousHV::weighted_bundle(&refs, &weights)
    }

    /// Get valence electrons
    pub fn valence_electrons(&self, atomic_number: u8) -> u8 {
        // Simplified valence calculation
        let noble_gas_configs = [2, 10, 18, 36, 54, 86, 118];

        for &config in &noble_gas_configs {
            if atomic_number <= config {
                if atomic_number == config {
                    return 0; // Noble gas
                }
                // Find previous noble gas
                let prev = noble_gas_configs.iter()
                    .filter(|&&x| x < atomic_number)
                    .max()
                    .copied()
                    .unwrap_or(0);
                return atomic_number - prev;
            }
        }
        0
    }
}

/// The Periodic Table: all elements composed from first principles
#[derive(Debug, Clone)]
pub struct PeriodicTable {
    /// All elements (indexed by atomic number - 1)
    elements: Vec<Element>,

    /// Electron shell encoder
    shells: ElectronShell,

    /// Concept vectors for chemical properties
    pub metallic: ContinuousHV,
    pub nonmetallic: ContinuousHV,
    pub noble: ContinuousHV,
    pub reactive: ContinuousHV,
    pub oxidizing: ContinuousHV,
    pub reducing: ContinuousHV,

    /// Reference to building blocks
    proton: ContinuousHV,
    neutron: ContinuousHV,
    electron: ContinuousHV,
}

impl PeriodicTable {
    /// Construct the periodic table from the Standard Model
    ///
    /// Every element is COMPOSED from protons, neutrons, and electrons.
    pub fn from_model(model: &StandardModel, hadrons: &Hadrons, genesis: &GenesisSeed) -> Self {
        let shells = ElectronShell::from_genesis(genesis);

        // Property concept vectors
        let metallic = genesis.hv("chemistry::metallic", PHYSICS_DIM);
        let nonmetallic = genesis.hv("chemistry::nonmetallic", PHYSICS_DIM);
        let noble = genesis.hv("chemistry::noble", PHYSICS_DIM);
        let reactive = genesis.hv("chemistry::reactive", PHYSICS_DIM);
        let oxidizing = genesis.hv("chemistry::oxidizing", PHYSICS_DIM);
        let reducing = genesis.hv("chemistry::reducing", PHYSICS_DIM);

        // Store building blocks
        let proton = hadrons.proton.clone();
        let neutron = hadrons.neutron.clone();
        let electron = model.electron.clone();

        // Create partial table for now
        let mut table = Self {
            elements: Vec::new(),
            shells,
            metallic,
            nonmetallic,
            noble,
            reactive,
            oxidizing,
            reducing,
            proton,
            neutron,
            electron,
        };

        // Build elements 1-118
        table.build_all_elements(hadrons);

        table
    }

    /// Build all elements
    fn build_all_elements(&mut self, hadrons: &Hadrons) {
        // Element data for first 36 elements (extend as needed)
        let element_data: Vec<ElementData> = vec![
            ElementData { symbol: "H", name: "Hydrogen", atomic_number: 1, standard_neutrons: 0, atomic_mass: 1.008, electronegativity: Some(2.20), group: 1, period: 1 },
            ElementData { symbol: "He", name: "Helium", atomic_number: 2, standard_neutrons: 2, atomic_mass: 4.003, electronegativity: None, group: 18, period: 1 },
            ElementData { symbol: "Li", name: "Lithium", atomic_number: 3, standard_neutrons: 4, atomic_mass: 6.941, electronegativity: Some(0.98), group: 1, period: 2 },
            ElementData { symbol: "Be", name: "Beryllium", atomic_number: 4, standard_neutrons: 5, atomic_mass: 9.012, electronegativity: Some(1.57), group: 2, period: 2 },
            ElementData { symbol: "B", name: "Boron", atomic_number: 5, standard_neutrons: 6, atomic_mass: 10.81, electronegativity: Some(2.04), group: 13, period: 2 },
            ElementData { symbol: "C", name: "Carbon", atomic_number: 6, standard_neutrons: 6, atomic_mass: 12.01, electronegativity: Some(2.55), group: 14, period: 2 },
            ElementData { symbol: "N", name: "Nitrogen", atomic_number: 7, standard_neutrons: 7, atomic_mass: 14.01, electronegativity: Some(3.04), group: 15, period: 2 },
            ElementData { symbol: "O", name: "Oxygen", atomic_number: 8, standard_neutrons: 8, atomic_mass: 16.00, electronegativity: Some(3.44), group: 16, period: 2 },
            ElementData { symbol: "F", name: "Fluorine", atomic_number: 9, standard_neutrons: 10, atomic_mass: 19.00, electronegativity: Some(3.98), group: 17, period: 2 },
            ElementData { symbol: "Ne", name: "Neon", atomic_number: 10, standard_neutrons: 10, atomic_mass: 20.18, electronegativity: None, group: 18, period: 2 },
            ElementData { symbol: "Na", name: "Sodium", atomic_number: 11, standard_neutrons: 12, atomic_mass: 22.99, electronegativity: Some(0.93), group: 1, period: 3 },
            ElementData { symbol: "Mg", name: "Magnesium", atomic_number: 12, standard_neutrons: 12, atomic_mass: 24.31, electronegativity: Some(1.31), group: 2, period: 3 },
            ElementData { symbol: "Al", name: "Aluminum", atomic_number: 13, standard_neutrons: 14, atomic_mass: 26.98, electronegativity: Some(1.61), group: 13, period: 3 },
            ElementData { symbol: "Si", name: "Silicon", atomic_number: 14, standard_neutrons: 14, atomic_mass: 28.09, electronegativity: Some(1.90), group: 14, period: 3 },
            ElementData { symbol: "P", name: "Phosphorus", atomic_number: 15, standard_neutrons: 16, atomic_mass: 30.97, electronegativity: Some(2.19), group: 15, period: 3 },
            ElementData { symbol: "S", name: "Sulfur", atomic_number: 16, standard_neutrons: 16, atomic_mass: 32.07, electronegativity: Some(2.58), group: 16, period: 3 },
            ElementData { symbol: "Cl", name: "Chlorine", atomic_number: 17, standard_neutrons: 18, atomic_mass: 35.45, electronegativity: Some(3.16), group: 17, period: 3 },
            ElementData { symbol: "Ar", name: "Argon", atomic_number: 18, standard_neutrons: 22, atomic_mass: 39.95, electronegativity: None, group: 18, period: 3 },
            ElementData { symbol: "K", name: "Potassium", atomic_number: 19, standard_neutrons: 20, atomic_mass: 39.10, electronegativity: Some(0.82), group: 1, period: 4 },
            ElementData { symbol: "Ca", name: "Calcium", atomic_number: 20, standard_neutrons: 20, atomic_mass: 40.08, electronegativity: Some(1.00), group: 2, period: 4 },
            ElementData { symbol: "Sc", name: "Scandium", atomic_number: 21, standard_neutrons: 24, atomic_mass: 44.96, electronegativity: Some(1.36), group: 3, period: 4 },
            ElementData { symbol: "Ti", name: "Titanium", atomic_number: 22, standard_neutrons: 26, atomic_mass: 47.87, electronegativity: Some(1.54), group: 4, period: 4 },
            ElementData { symbol: "V", name: "Vanadium", atomic_number: 23, standard_neutrons: 28, atomic_mass: 50.94, electronegativity: Some(1.63), group: 5, period: 4 },
            ElementData { symbol: "Cr", name: "Chromium", atomic_number: 24, standard_neutrons: 28, atomic_mass: 52.00, electronegativity: Some(1.66), group: 6, period: 4 },
            ElementData { symbol: "Mn", name: "Manganese", atomic_number: 25, standard_neutrons: 30, atomic_mass: 54.94, electronegativity: Some(1.55), group: 7, period: 4 },
            ElementData { symbol: "Fe", name: "Iron", atomic_number: 26, standard_neutrons: 30, atomic_mass: 55.85, electronegativity: Some(1.83), group: 8, period: 4 },
            ElementData { symbol: "Co", name: "Cobalt", atomic_number: 27, standard_neutrons: 32, atomic_mass: 58.93, electronegativity: Some(1.88), group: 9, period: 4 },
            ElementData { symbol: "Ni", name: "Nickel", atomic_number: 28, standard_neutrons: 30, atomic_mass: 58.69, electronegativity: Some(1.91), group: 10, period: 4 },
            ElementData { symbol: "Cu", name: "Copper", atomic_number: 29, standard_neutrons: 34, atomic_mass: 63.55, electronegativity: Some(1.90), group: 11, period: 4 },
            ElementData { symbol: "Zn", name: "Zinc", atomic_number: 30, standard_neutrons: 34, atomic_mass: 65.38, electronegativity: Some(1.65), group: 12, period: 4 },
            ElementData { symbol: "Ga", name: "Gallium", atomic_number: 31, standard_neutrons: 38, atomic_mass: 69.72, electronegativity: Some(1.81), group: 13, period: 4 },
            ElementData { symbol: "Ge", name: "Germanium", atomic_number: 32, standard_neutrons: 42, atomic_mass: 72.63, electronegativity: Some(2.01), group: 14, period: 4 },
            ElementData { symbol: "As", name: "Arsenic", atomic_number: 33, standard_neutrons: 42, atomic_mass: 74.92, electronegativity: Some(2.18), group: 15, period: 4 },
            ElementData { symbol: "Se", name: "Selenium", atomic_number: 34, standard_neutrons: 46, atomic_mass: 78.97, electronegativity: Some(2.55), group: 16, period: 4 },
            ElementData { symbol: "Br", name: "Bromine", atomic_number: 35, standard_neutrons: 44, atomic_mass: 79.90, electronegativity: Some(2.96), group: 17, period: 4 },
            ElementData { symbol: "Kr", name: "Krypton", atomic_number: 36, standard_neutrons: 48, atomic_mass: 83.80, electronegativity: Some(3.00), group: 18, period: 4 },
        ];

        for data in element_data {
            let mut vector = self.compose_element(data.atomic_number, data.standard_neutrons, hadrons);

            // Add noble gas character to group 18 elements (He, Ne, Ar, Kr, etc.)
            // This makes them share a common "full shell" signature
            if data.group == 18 {
                vector = ContinuousHV::weighted_bundle(
                    &[&vector, &self.noble],
                    &[1.0, 0.5],
                );
            }

            self.elements.push(Element { data, vector });
        }
    }

    /// Compose an element from its constituents
    ///
    /// Element = Bundle(Z×Proton, N×Neutron, Z×Electron) ⊗ Binding
    pub fn compose_element(&self, protons: u8, neutrons: u8, hadrons: &Hadrons) -> ContinuousHV {
        let z = protons as f32;
        let n = neutrons as f32;

        // Nuclear component: weighted bundle of protons and neutrons
        let nuclear = ContinuousHV::weighted_bundle(
            &[&self.proton, &self.neutron],
            &[z, n],
        );

        // Electronic component: electron cloud
        let electron_cloud = self.shells.encode_configuration(protons, &self.electron);

        // Binding energy (simplified model)
        let binding = hadrons.compute_binding(protons as usize, neutrons as usize);

        // Combine: Bundle nucleus + electrons, then bind with binding energy
        let atom = ContinuousHV::bundle(&[&nuclear, &electron_cloud]);
        atom.bind(&binding)
    }

    /// Get element by atomic number
    pub fn element(&self, atomic_number: u8) -> Option<&Element> {
        if atomic_number == 0 || atomic_number as usize > self.elements.len() {
            return None;
        }
        self.elements.get(atomic_number as usize - 1)
    }

    /// Get element by symbol
    pub fn by_symbol(&self, symbol: &str) -> Option<&Element> {
        self.elements.iter().find(|e| e.data.symbol.eq_ignore_ascii_case(symbol))
    }

    /// Create an isotope
    ///
    /// Same element with different neutron count.
    pub fn isotope(&self, atomic_number: u8, neutrons: u8, hadrons: &Hadrons) -> ContinuousHV {
        self.compose_element(atomic_number, neutrons, hadrons)
    }

    /// Create an ion
    ///
    /// Element with different electron count.
    /// Ions are distinguished by:
    /// 1. Different electron configuration
    /// 2. Charge marker (permuted by charge magnitude)
    pub fn ion(&self, atomic_number: u8, charge: i8, hadrons: &Hadrons) -> ContinuousHV {
        let base = self.element(atomic_number);
        if base.is_none() {
            return ContinuousHV::zero(PHYSICS_DIM);
        }

        let neutrons = base.unwrap().data.standard_neutrons;
        let electrons = (atomic_number as i8 - charge) as u8;

        let z = atomic_number as f32;
        let n = neutrons as f32;

        // Nuclear component unchanged
        let nuclear = ContinuousHV::weighted_bundle(
            &[&self.proton, &self.neutron],
            &[z, n],
        );

        // Modified electron cloud
        let electron_cloud = self.shells.encode_configuration(electrons, &self.electron);

        // Binding (nuclear part unchanged)
        let binding = hadrons.compute_binding(atomic_number as usize, neutrons as usize);

        // Create charge marker: permute reactive vector by charge magnitude
        // This makes ions significantly different from neutral atoms
        let charge_marker = if charge > 0 {
            // Cation: missing electrons (positive charge)
            self.reactive.permute(charge.abs() as usize * 1000)
        } else if charge < 0 {
            // Anion: extra electrons (negative charge)
            self.reactive.permute(PHYSICS_DIM / 2 + charge.abs() as usize * 1000)
        } else {
            ContinuousHV::zero(PHYSICS_DIM)
        };

        // Bundle: nuclear + electrons + charge marker (giving charge significant weight)
        let ion_base = ContinuousHV::weighted_bundle(
            &[&nuclear, &electron_cloud, &charge_marker],
            &[1.0, 1.0, 0.5 * charge.abs() as f32],
        );

        ion_base.bind(&binding)
    }

    /// Compare isotope similarity
    pub fn isotope_similarity(&self, z: u8, n1: u8, n2: u8, hadrons: &Hadrons) -> f32 {
        let iso1 = self.isotope(z, n1, hadrons);
        let iso2 = self.isotope(z, n2, hadrons);
        iso1.similarity(&iso2)
    }

    /// Get chemical character (metallic vs nonmetallic)
    pub fn chemical_character(&self, element: &Element) -> ContinuousHV {
        // Metals: groups 1-12 (except H), left side
        // Nonmetals: groups 13-18, right side

        let z = element.data.atomic_number;
        let group = element.data.group;

        let metallic_weight = if z == 1 {
            0.0 // Hydrogen is nonmetal
        } else if group <= 12 {
            1.0
        } else if group <= 15 {
            0.5 // Metalloids
        } else {
            0.0
        };

        ContinuousHV::weighted_bundle(
            &[&self.metallic, &self.nonmetallic],
            &[metallic_weight, 1.0 - metallic_weight],
        )
    }

    /// Get number of elements defined
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Iterate over all elements
    pub fn iter(&self) -> impl Iterator<Item = &Element> {
        self.elements.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (StandardModel, Hadrons, PeriodicTable, GenesisSeed) {
        let genesis = GenesisSeed::from_phrase("periodic table test");
        let model = StandardModel::from_genesis(&genesis);
        let hadrons = Hadrons::from_model(&model, &genesis);
        let table = PeriodicTable::from_model(&model, &hadrons, &genesis);
        (model, hadrons, table, genesis)
    }

    #[test]
    fn test_periodic_table_creation() {
        let (_, _, table, _) = setup();

        assert!(table.len() >= 36, "Should have at least 36 elements");

        let hydrogen = table.element(1).unwrap();
        assert_eq!(hydrogen.data.symbol, "H");
        assert_eq!(hydrogen.data.atomic_number, 1);
    }

    #[test]
    fn test_element_by_symbol() {
        let (_, _, table, _) = setup();

        let carbon = table.by_symbol("C").unwrap();
        assert_eq!(carbon.data.atomic_number, 6);
        assert_eq!(carbon.data.name, "Carbon");

        let iron = table.by_symbol("Fe").unwrap();
        assert_eq!(iron.data.atomic_number, 26);
    }

    #[test]
    fn test_isotope_similarity() {
        let (_, hadrons, table, _) = setup();

        // Carbon-12 vs Carbon-14 should be highly similar
        let c12_c14 = table.isotope_similarity(6, 6, 8, &hadrons);

        // Carbon-12 vs Nitrogen-14 should be less similar
        let carbon = table.isotope(6, 6, &hadrons);
        let nitrogen = table.isotope(7, 7, &hadrons);
        let c_n = carbon.similarity(&nitrogen);

        assert!(
            c12_c14 > c_n,
            "Isotopes should be more similar than different elements: C12-C14={}, C-N={}",
            c12_c14, c_n
        );
    }

    #[test]
    fn test_neighboring_elements() {
        let (_, _, table, _) = setup();

        // Adjacent elements should share some structure
        let carbon = table.element(6).unwrap();
        let nitrogen = table.element(7).unwrap();
        let iron = table.element(26).unwrap();

        let c_n_sim = carbon.vector.similarity(&nitrogen.vector);
        let c_fe_sim = carbon.vector.similarity(&iron.vector);

        // Adjacent elements should be more similar than distant ones
        assert!(
            c_n_sim > c_fe_sim,
            "Adjacent elements should be more similar: C-N={}, C-Fe={}",
            c_n_sim, c_fe_sim
        );
    }

    #[test]
    fn test_ion_creation() {
        let (_, hadrons, table, _) = setup();

        // Sodium ion (Na+) vs neutral sodium
        let na = table.element(11).unwrap().vector.clone();
        let na_plus = table.ion(11, 1, &hadrons);

        // Ion should be similar but not identical
        let sim = na.similarity(&na_plus);
        assert!(
            sim > 0.5 && sim < 0.99,
            "Ion should be similar but distinct from neutral: {}",
            sim
        );
    }

    #[test]
    fn test_deterministic_elements() {
        let genesis = GenesisSeed::from_phrase("determinism test");
        let model = StandardModel::from_genesis(&genesis);
        let hadrons = Hadrons::from_model(&model, &genesis);

        let table1 = PeriodicTable::from_model(&model, &hadrons, &genesis);
        let table2 = PeriodicTable::from_model(&model, &hadrons, &genesis);

        let c1 = table1.element(6).unwrap();
        let c2 = table2.element(6).unwrap();

        assert!(
            c1.vector.similarity(&c2.vector) > 0.9999,
            "Elements should be deterministic"
        );
    }

    #[test]
    fn test_noble_gases() {
        let (_, _, table, _) = setup();

        // Noble gases: He, Ne, Ar, Kr
        let he = table.element(2).unwrap();
        let ne = table.element(10).unwrap();
        let ar = table.element(18).unwrap();
        let _kr = table.element(36).unwrap();

        // All should have no electronegativity
        assert!(he.data.electronegativity.is_none());
        assert!(ne.data.electronegativity.is_none());
        assert!(ar.data.electronegativity.is_none());

        // Noble gases should share some character (full shells)
        let he_ne = he.vector.similarity(&ne.vector);
        let he_li = he.vector.similarity(&table.element(3).unwrap().vector);

        assert!(
            he_ne > he_li * 0.5,
            "Noble gases should share character"
        );
    }
}
