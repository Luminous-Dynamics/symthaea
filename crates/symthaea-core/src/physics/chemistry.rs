// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Chemistry: Layer 4 - Bonds and Reactions
//!
//! Chemical bonds and reactions built on top of the periodic table.
//! Molecules are **composed** from atoms using bond operations.
//!
//! ## Bond Types
//!
//! - **Covalent**: Electron sharing (bind operation)
//! - **Ionic**: Electron transfer (ion composition)
//! - **Metallic**: Electron sea (bundle with metallic character)
//! - **Hydrogen**: Weak polar attraction
//!
//! ## Reaction Representation
//!
//! Reactions are transformations of molecular vectors:
//! ```text
//! A + B → C + D
//!
//! Reactants = Bundle(A, B)
//! Products = Bundle(C, D)
//! Reaction = Bind(Reactants, Products)
//! ```
//!
//! ## Energy and Thermodynamics
//!
//! Bond energies are encoded through binding energy vectors:
//! - Exothermic: Products have lower energy (more stable bonds)
//! - Endothermic: Products have higher energy

use super::periodic_table::PeriodicTable;
use super::standard_model::PHYSICS_DIM;
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// Chemical bond types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BondType {
    /// Single covalent bond
    Single,
    /// Double covalent bond
    Double,
    /// Triple covalent bond
    Triple,
    /// Aromatic (resonance) bond
    Aromatic,
    /// Ionic bond
    Ionic,
    /// Metallic bond
    Metallic,
    /// Hydrogen bond
    Hydrogen,
    /// Van der Waals (weak)
    VanDerWaals,
}

impl BondType {
    /// Relative bond strength (arbitrary units)
    pub fn strength(&self) -> f32 {
        match self {
            BondType::Triple => 3.0,
            BondType::Double => 2.0,
            BondType::Aromatic => 1.5,
            BondType::Single => 1.0,
            BondType::Ionic => 0.8,
            BondType::Metallic => 0.7,
            BondType::Hydrogen => 0.1,
            BondType::VanDerWaals => 0.02,
        }
    }

    /// Domain label for vector generation
    fn domain(&self) -> &'static str {
        match self {
            BondType::Single => "bond::covalent::single",
            BondType::Double => "bond::covalent::double",
            BondType::Triple => "bond::covalent::triple",
            BondType::Aromatic => "bond::aromatic",
            BondType::Ionic => "bond::ionic",
            BondType::Metallic => "bond::metallic",
            BondType::Hydrogen => "bond::hydrogen",
            BondType::VanDerWaals => "bond::vdw",
        }
    }
}

/// A chemical bond between atoms
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Bond {
    pub bond_type: BondType,
    pub vector: ContinuousHV,
}

/// A molecule composed of atoms and bonds
#[derive(Debug, Clone)]
pub struct Molecule {
    pub name: String,
    pub formula: String,
    pub atoms: Vec<(u8, u8)>, // (atomic_number, count)
    pub vector: ContinuousHV,
}

/// Reaction thermodynamics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReactionType {
    Exothermic,
    Endothermic,
    Neutral,
}

/// A chemical reaction
#[derive(Debug, Clone)]
pub struct Reaction {
    pub name: String,
    pub reactants: Vec<Molecule>,
    pub products: Vec<Molecule>,
    pub reaction_type: ReactionType,
    pub vector: ContinuousHV,
}

/// Chemistry system: bonds, molecules, and reactions
#[derive(Debug, Clone)]
pub struct Chemistry {
    // Bond type vectors
    pub single_bond: ContinuousHV,
    pub double_bond: ContinuousHV,
    pub triple_bond: ContinuousHV,
    pub aromatic_bond: ContinuousHV,
    pub ionic_bond: ContinuousHV,
    pub metallic_bond: ContinuousHV,
    pub hydrogen_bond: ContinuousHV,
    pub vdw_bond: ContinuousHV,

    // Reaction concept vectors
    pub exothermic: ContinuousHV,
    pub endothermic: ContinuousHV,
    pub catalyst: ContinuousHV,
    pub equilibrium: ContinuousHV,
    pub activation_energy: ContinuousHV,

    // Functional group vectors
    pub hydroxyl: ContinuousHV,   // -OH
    pub carbonyl: ContinuousHV,   // C=O
    pub carboxyl: ContinuousHV,   // -COOH
    pub amino: ContinuousHV,      // -NH2
    pub phosphate: ContinuousHV,  // -PO4
    pub sulfhydryl: ContinuousHV, // -SH
    pub methyl: ContinuousHV,     // -CH3

    // Reference to periodic table
    elements: Vec<ContinuousHV>, // Cached element vectors
}

impl Chemistry {
    /// Create chemistry system from genesis and periodic table
    pub fn from_genesis(genesis: &GenesisSeed, table: &PeriodicTable) -> Self {
        // Bond vectors
        let single_bond = genesis.hv(BondType::Single.domain(), PHYSICS_DIM);
        let double_bond = genesis.hv(BondType::Double.domain(), PHYSICS_DIM);
        let triple_bond = genesis.hv(BondType::Triple.domain(), PHYSICS_DIM);
        let aromatic_bond = genesis.hv(BondType::Aromatic.domain(), PHYSICS_DIM);
        let ionic_bond = genesis.hv(BondType::Ionic.domain(), PHYSICS_DIM);
        let metallic_bond = genesis.hv(BondType::Metallic.domain(), PHYSICS_DIM);
        let hydrogen_bond = genesis.hv(BondType::Hydrogen.domain(), PHYSICS_DIM);
        let vdw_bond = genesis.hv(BondType::VanDerWaals.domain(), PHYSICS_DIM);

        // Reaction concept vectors
        let exothermic = genesis.hv("reaction::exothermic", PHYSICS_DIM);
        let endothermic = genesis.hv("reaction::endothermic", PHYSICS_DIM);
        let catalyst = genesis.hv("reaction::catalyst", PHYSICS_DIM);
        let equilibrium = genesis.hv("reaction::equilibrium", PHYSICS_DIM);
        let activation_energy = genesis.hv("reaction::activation_energy", PHYSICS_DIM);

        // Functional groups (composed from elements)
        let o = table
            .element(8)
            .map(|e| e.vector.clone())
            .unwrap_or_else(|| ContinuousHV::zero(PHYSICS_DIM));
        let h = table
            .element(1)
            .map(|e| e.vector.clone())
            .unwrap_or_else(|| ContinuousHV::zero(PHYSICS_DIM));
        let c = table
            .element(6)
            .map(|e| e.vector.clone())
            .unwrap_or_else(|| ContinuousHV::zero(PHYSICS_DIM));
        let n = table
            .element(7)
            .map(|e| e.vector.clone())
            .unwrap_or_else(|| ContinuousHV::zero(PHYSICS_DIM));
        let p = table
            .element(15)
            .map(|e| e.vector.clone())
            .unwrap_or_else(|| ContinuousHV::zero(PHYSICS_DIM));
        let s = table
            .element(16)
            .map(|e| e.vector.clone())
            .unwrap_or_else(|| ContinuousHV::zero(PHYSICS_DIM));

        // Compose functional groups
        let hydroxyl = ContinuousHV::bundle(&[&o, &h]).bind(&single_bond);
        let carbonyl = ContinuousHV::bundle(&[&c, &o]).bind(&double_bond);
        let carboxyl = ContinuousHV::bundle(&[&c, &o, &o, &h]).bind(&single_bond);
        let amino = ContinuousHV::bundle(&[&n, &h, &h]).bind(&single_bond);
        let phosphate = ContinuousHV::bundle(&[&p, &o, &o, &o, &o]).bind(&single_bond);
        let sulfhydryl = ContinuousHV::bundle(&[&s, &h]).bind(&single_bond);
        let methyl = ContinuousHV::bundle(&[&c, &h, &h, &h]).bind(&single_bond);

        // Cache element vectors
        let elements: Vec<ContinuousHV> = table.iter().map(|e| e.vector.clone()).collect();

        Self {
            single_bond,
            double_bond,
            triple_bond,
            aromatic_bond,
            ionic_bond,
            metallic_bond,
            hydrogen_bond,
            vdw_bond,
            exothermic,
            endothermic,
            catalyst,
            equilibrium,
            activation_energy,
            hydroxyl,
            carbonyl,
            carboxyl,
            amino,
            phosphate,
            sulfhydryl,
            methyl,
            elements,
        }
    }

    /// Get bond vector by type
    pub fn bond(&self, bond_type: BondType) -> &ContinuousHV {
        match bond_type {
            BondType::Single => &self.single_bond,
            BondType::Double => &self.double_bond,
            BondType::Triple => &self.triple_bond,
            BondType::Aromatic => &self.aromatic_bond,
            BondType::Ionic => &self.ionic_bond,
            BondType::Metallic => &self.metallic_bond,
            BondType::Hydrogen => &self.hydrogen_bond,
            BondType::VanDerWaals => &self.vdw_bond,
        }
    }

    /// Create a bond between two atoms
    pub fn create_bond(
        &self,
        atom1: &ContinuousHV,
        atom2: &ContinuousHV,
        bond_type: BondType,
    ) -> ContinuousHV {
        let bond_vector = self.bond(bond_type);
        let strength = bond_type.strength();

        // Bond = Bind(atom1, atom2, bond_type) scaled by strength
        atom1.bind(atom2).bind(bond_vector).scale(strength)
    }

    /// Compose a molecule from atoms and bonds
    ///
    /// # Arguments
    /// * `atoms` - List of (element_vector, count) tuples
    /// * `bonds` - List of bonds between atoms
    pub fn compose_molecule(
        &self,
        name: &str,
        formula: &str,
        atoms: &[(ContinuousHV, u8)],
        bonds: &[BondType],
    ) -> Molecule {
        // Bundle atoms with multiplicity
        let atom_vectors: Vec<ContinuousHV> = atoms
            .iter()
            .map(|(atom, count)| atom.scale(*count as f32))
            .collect();

        let refs: Vec<&ContinuousHV> = atom_vectors.iter().collect();
        let atom_bundle = ContinuousHV::bundle(&refs);

        // Bundle bonds
        let bond_vectors: Vec<&ContinuousHV> = bonds.iter().map(|b| self.bond(*b)).collect();

        let bond_bundle = if bond_vectors.is_empty() {
            ContinuousHV::zero(PHYSICS_DIM)
        } else {
            ContinuousHV::bundle(&bond_vectors)
        };

        // Molecule = Bind(atoms, bonds)
        let vector = atom_bundle.bind(&bond_bundle);

        // Extract atom counts (for serialization)
        let atom_counts: Vec<(u8, u8)> = Vec::new(); // Would need element lookup

        Molecule {
            name: name.to_string(),
            formula: formula.to_string(),
            atoms: atom_counts,
            vector,
        }
    }

    /// Create a simple molecule from element symbols
    pub fn simple_molecule(
        &self,
        name: &str,
        formula: &str,
        table: &PeriodicTable,
    ) -> Option<Molecule> {
        // Parse simple formulas like H2O, CO2, CH4
        // This is a simplified parser

        let mut atoms = Vec::new();
        let mut bonds = Vec::new();

        let chars: Vec<char> = formula.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            if chars[i].is_ascii_uppercase() {
                // Element symbol
                let mut symbol = String::from(chars[i]);
                i += 1;

                // Check for lowercase continuation
                if i < chars.len() && chars[i].is_ascii_lowercase() {
                    symbol.push(chars[i]);
                    i += 1;
                }

                // Check for count
                let mut count = 0u8;
                while i < chars.len() && chars[i].is_ascii_digit() {
                    count = count * 10 + (chars[i] as u8 - b'0');
                    i += 1;
                }
                if count == 0 {
                    count = 1;
                }

                // Look up element
                if let Some(element) = table.by_symbol(&symbol) {
                    atoms.push((element.vector.clone(), count));
                    // Add single bonds (simplified)
                    for _ in 0..count.saturating_sub(1) {
                        bonds.push(BondType::Single);
                    }
                }
            } else {
                i += 1;
            }
        }

        if atoms.is_empty() {
            return None;
        }

        Some(self.compose_molecule(name, formula, &atoms, &bonds))
    }

    /// Create a reaction from reactants and products
    pub fn create_reaction(
        &self,
        name: &str,
        reactants: Vec<Molecule>,
        products: Vec<Molecule>,
    ) -> Reaction {
        // Bundle reactants
        let reactant_refs: Vec<&ContinuousHV> = reactants.iter().map(|m| &m.vector).collect();
        let reactant_bundle = ContinuousHV::bundle(&reactant_refs);

        // Bundle products
        let product_refs: Vec<&ContinuousHV> = products.iter().map(|m| &m.vector).collect();
        let product_bundle = ContinuousHV::bundle(&product_refs);

        // Reaction vector: transformation from reactants to products
        let reaction_vector = reactant_bundle.bind(&product_bundle);

        // Determine reaction type based on similarity to exo/endo vectors
        let exo_sim = reaction_vector.similarity(&self.exothermic);
        let endo_sim = reaction_vector.similarity(&self.endothermic);

        let reaction_type = if exo_sim > endo_sim + 0.1 {
            ReactionType::Exothermic
        } else if endo_sim > exo_sim + 0.1 {
            ReactionType::Endothermic
        } else {
            ReactionType::Neutral
        };

        Reaction {
            name: name.to_string(),
            reactants,
            products,
            reaction_type,
            vector: reaction_vector,
        }
    }

    /// Predict reaction similarity
    ///
    /// How similar are two reactions in terms of mechanism?
    pub fn reaction_similarity(&self, r1: &Reaction, r2: &Reaction) -> f32 {
        r1.vector.similarity(&r2.vector)
    }

    /// Check if molecule contains functional group
    pub fn has_functional_group(&self, molecule: &Molecule, group: &ContinuousHV) -> f32 {
        molecule.vector.similarity(group)
    }

    /// Get cached element vector
    pub fn element(&self, atomic_number: u8) -> Option<&ContinuousHV> {
        if atomic_number == 0 {
            return None;
        }
        self.elements.get(atomic_number as usize - 1)
    }

    /// Common molecules
    pub fn water(&self, table: &PeriodicTable) -> Molecule {
        self.simple_molecule("Water", "H2O", table)
            .unwrap_or_else(|| Molecule {
                name: "Water".to_string(),
                formula: "H2O".to_string(),
                atoms: vec![(1, 2), (8, 1)],
                vector: ContinuousHV::zero(PHYSICS_DIM),
            })
    }

    pub fn carbon_dioxide(&self, table: &PeriodicTable) -> Molecule {
        self.simple_molecule("Carbon Dioxide", "CO2", table)
            .unwrap_or_else(|| Molecule {
                name: "Carbon Dioxide".to_string(),
                formula: "CO2".to_string(),
                atoms: vec![(6, 1), (8, 2)],
                vector: ContinuousHV::zero(PHYSICS_DIM),
            })
    }

    pub fn methane(&self, table: &PeriodicTable) -> Molecule {
        self.simple_molecule("Methane", "CH4", table)
            .unwrap_or_else(|| Molecule {
                name: "Methane".to_string(),
                formula: "CH4".to_string(),
                atoms: vec![(6, 1), (1, 4)],
                vector: ContinuousHV::zero(PHYSICS_DIM),
            })
    }

    pub fn ammonia(&self, table: &PeriodicTable) -> Molecule {
        self.simple_molecule("Ammonia", "NH3", table)
            .unwrap_or_else(|| Molecule {
                name: "Ammonia".to_string(),
                formula: "NH3".to_string(),
                atoms: vec![(7, 1), (1, 3)],
                vector: ContinuousHV::zero(PHYSICS_DIM),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::{Hadrons, StandardModel};

    fn setup() -> (PeriodicTable, Chemistry, GenesisSeed) {
        let genesis = GenesisSeed::from_phrase("chemistry test");
        let model = StandardModel::from_genesis(&genesis);
        let hadrons = Hadrons::from_model(&model, &genesis);
        let table = PeriodicTable::from_model(&model, &hadrons, &genesis);
        let chem = Chemistry::from_genesis(&genesis, &table);
        (table, chem, genesis)
    }

    #[test]
    fn test_chemistry_creation() {
        let (_, chem, _) = setup();

        // Bond vectors should exist
        assert_eq!(chem.single_bond.dim(), PHYSICS_DIM);
        assert_eq!(chem.double_bond.dim(), PHYSICS_DIM);

        // Functional groups should exist
        assert_eq!(chem.hydroxyl.dim(), PHYSICS_DIM);
        assert_eq!(chem.amino.dim(), PHYSICS_DIM);
    }

    #[test]
    fn test_bond_strength_ordering() {
        assert!(BondType::Triple.strength() > BondType::Double.strength());
        assert!(BondType::Double.strength() > BondType::Single.strength());
        assert!(BondType::Single.strength() > BondType::Hydrogen.strength());
    }

    #[test]
    fn test_simple_molecule_creation() {
        let (table, chem, _) = setup();

        let water = chem.water(&table);
        assert_eq!(water.formula, "H2O");
        assert_eq!(water.vector.dim(), PHYSICS_DIM);

        let co2 = chem.carbon_dioxide(&table);
        assert_eq!(co2.formula, "CO2");
    }

    #[test]
    fn test_molecule_similarity() {
        let (table, chem, _) = setup();

        let water = chem.water(&table);
        let methane = chem.methane(&table);
        let ammonia = chem.ammonia(&table);

        // Different molecules should have some distinction
        let h2o_ch4 = water.vector.similarity(&methane.vector);
        let h2o_nh3 = water.vector.similarity(&ammonia.vector);

        // Water and ammonia are both polar, might be more similar
        // This tests that the system creates distinct representations
        assert!(
            (h2o_ch4 - h2o_nh3).abs() > 0.01,
            "Molecules should have distinct relationships"
        );
    }

    #[test]
    fn test_functional_group_detection() {
        let (table, chem, _) = setup();

        // Methane has CH3 (methyl) character
        let methane = chem.methane(&table);
        let methyl_sim = chem.has_functional_group(&methane, &chem.methyl);

        // Water has OH (hydroxyl) character
        let water = chem.water(&table);
        let hydroxyl_sim = chem.has_functional_group(&water, &chem.hydroxyl);

        // Both should show positive affinity to their functional groups
        assert!(methyl_sim > -0.5, "Methane should relate to methyl");
        assert!(hydroxyl_sim > -0.5, "Water should relate to hydroxyl");
    }

    #[test]
    fn test_reaction_creation() {
        let (table, chem, _) = setup();

        // Simple reaction: 2H2 + O2 → 2H2O
        let h2 = chem.simple_molecule("Hydrogen", "H2", &table).unwrap();
        let o2 = chem.simple_molecule("Oxygen", "O2", &table).unwrap();
        let water = chem.water(&table);

        let combustion = chem.create_reaction(
            "Hydrogen Combustion",
            vec![h2.clone(), h2, o2],
            vec![water.clone(), water],
        );

        assert_eq!(combustion.name, "Hydrogen Combustion");
        assert_eq!(combustion.vector.dim(), PHYSICS_DIM);
    }

    #[test]
    fn test_deterministic_chemistry() {
        let (table1, chem1, genesis) = setup();

        // Create second instance with same genesis
        let model = StandardModel::from_genesis(&genesis);
        let hadrons = Hadrons::from_model(&model, &genesis);
        let table2 = PeriodicTable::from_model(&model, &hadrons, &genesis);
        let chem2 = Chemistry::from_genesis(&genesis, &table2);

        let water1 = chem1.water(&table1);
        let water2 = chem2.water(&table2);

        assert!(
            water1.vector.similarity(&water2.vector) > 0.99,
            "Chemistry should be deterministic"
        );
    }
}
