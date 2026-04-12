// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Molecular geometry representation.
//!
//! Defines atoms and molecules in Bohr (atomic units). Provides XYZ parsing,
//! nuclear repulsion energy, and basic molecular properties.

use crate::constants::ANGSTROM_TO_BOHR;
use serde::{Deserialize, Serialize};

/// Element symbols indexed by atomic number (1-based)
const ELEMENT_SYMBOLS: &[&str] = &[
    "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
    "Cl", "Ar", "K", "Ca",
];

/// An atom: nuclear charge + position in 3D space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Atom {
    /// Atomic number (1 = H, 6 = C, 8 = O, ...)
    pub atomic_number: u8,
    /// Position in Bohr (atomic units)
    pub position: [f64; 3],
}

impl Atom {
    /// Create a new atom at position given in Bohr.
    pub fn new(atomic_number: u8, x: f64, y: f64, z: f64) -> Self {
        Self {
            atomic_number,
            position: [x, y, z],
        }
    }

    /// Create a new atom at position given in Angstrom (converts to Bohr).
    pub fn from_angstrom(atomic_number: u8, x: f64, y: f64, z: f64) -> Self {
        Self {
            atomic_number,
            position: [
                x * ANGSTROM_TO_BOHR,
                y * ANGSTROM_TO_BOHR,
                z * ANGSTROM_TO_BOHR,
            ],
        }
    }

    /// Element symbol (e.g., "H", "O", "C")
    pub fn symbol(&self) -> &'static str {
        if (self.atomic_number as usize) < ELEMENT_SYMBOLS.len() {
            ELEMENT_SYMBOLS[self.atomic_number as usize]
        } else {
            "??"
        }
    }

    /// Distance to another atom in Bohr
    pub fn distance_to(&self, other: &Atom) -> f64 {
        let dx = self.position[0] - other.position[0];
        let dy = self.position[1] - other.position[1];
        let dz = self.position[2] - other.position[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// A molecule: collection of atoms with charge and spin multiplicity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Molecule {
    pub atoms: Vec<Atom>,
    /// Total charge (0 = neutral, +1 = cation, -1 = anion)
    pub charge: i32,
    /// Spin multiplicity (1 = singlet, 2 = doublet, 3 = triplet)
    pub multiplicity: u32,
}

impl Molecule {
    /// Create a neutral singlet molecule.
    pub fn new(atoms: Vec<Atom>) -> Self {
        Self {
            atoms,
            charge: 0,
            multiplicity: 1,
        }
    }

    /// Create a molecule with specified charge and multiplicity.
    pub fn with_charge(atoms: Vec<Atom>, charge: i32, multiplicity: u32) -> Self {
        Self {
            atoms,
            charge,
            multiplicity,
        }
    }

    /// Total number of electrons.
    pub fn n_electrons(&self) -> usize {
        let nuclear_charge: i32 = self.atoms.iter().map(|a| a.atomic_number as i32).sum();
        (nuclear_charge - self.charge) as usize
    }

    /// Number of occupied orbitals (RHF: n_electrons / 2).
    pub fn n_occupied(&self) -> usize {
        self.n_electrons() / 2
    }

    /// Number of atoms.
    pub fn n_atoms(&self) -> usize {
        self.atoms.len()
    }

    /// Nuclear repulsion energy in Hartree.
    /// V_nn = Σ_{A>B} Z_A * Z_B / |R_A - R_B|
    pub fn nuclear_repulsion_energy(&self) -> f64 {
        let mut energy = 0.0;
        for i in 0..self.atoms.len() {
            for j in (i + 1)..self.atoms.len() {
                let za = self.atoms[i].atomic_number as f64;
                let zb = self.atoms[j].atomic_number as f64;
                let r = self.atoms[i].distance_to(&self.atoms[j]);
                if r > 1e-14 {
                    energy += za * zb / r;
                }
            }
        }
        energy
    }

    /// Parse an XYZ-format string. Positions are in Angstrom, converted to Bohr.
    ///
    /// Format:
    /// ```text
    /// n_atoms
    /// comment line
    /// Symbol x y z
    /// Symbol x y z
    /// ...
    /// ```
    pub fn from_xyz(input: &str) -> Result<Self, String> {
        let lines: Vec<&str> = input.lines().collect();
        if lines.len() < 3 {
            return Err("XYZ needs at least 3 lines (count, comment, atoms)".into());
        }

        let n_atoms: usize = lines[0]
            .trim()
            .parse()
            .map_err(|_| "First line must be atom count".to_string())?;

        if lines.len() < 2 + n_atoms {
            return Err(format!("Expected {} atom lines, got {}", n_atoms, lines.len() - 2));
        }

        let mut atoms = Vec::with_capacity(n_atoms);
        for line in &lines[2..2 + n_atoms] {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 4 {
                return Err(format!("Bad atom line: '{}'", line));
            }

            let atomic_number = symbol_to_z(parts[0])
                .ok_or_else(|| format!("Unknown element: {}", parts[0]))?;
            let x: f64 = parts[1].parse().map_err(|_| "Bad x coordinate")?;
            let y: f64 = parts[2].parse().map_err(|_| "Bad y coordinate")?;
            let z: f64 = parts[3].parse().map_err(|_| "Bad z coordinate")?;

            atoms.push(Atom::from_angstrom(atomic_number, x, y, z));
        }

        Ok(Self::new(atoms))
    }
}

/// Convert element symbol to atomic number.
fn symbol_to_z(symbol: &str) -> Option<u8> {
    match symbol.to_uppercase().as_str() {
        "H" => Some(1),
        "HE" => Some(2),
        "LI" => Some(3),
        "BE" => Some(4),
        "B" => Some(5),
        "C" => Some(6),
        "N" => Some(7),
        "O" => Some(8),
        "F" => Some(9),
        "NE" => Some(10),
        "NA" => Some(11),
        "MG" => Some(12),
        "AL" => Some(13),
        "SI" => Some(14),
        "P" => Some(15),
        "S" => Some(16),
        "CL" => Some(17),
        "AR" => Some(18),
        _ => None,
    }
}

// ── Convenience constructors for benchmark molecules ────────────────────────

impl Molecule {
    /// H₂ at R = 1.4 Bohr (equilibrium-ish).
    pub fn h2() -> Self {
        Self::new(vec![
            Atom::new(1, 0.0, 0.0, 0.0),
            Atom::new(1, 0.0, 0.0, 1.4),
        ])
    }

    /// HeH⁺ at R = 1.4632 Bohr.
    pub fn heh_plus() -> Self {
        Self::with_charge(
            vec![
                Atom::new(2, 0.0, 0.0, 0.0),
                Atom::new(1, 0.0, 0.0, 1.4632),
            ],
            1,
            1,
        )
    }

    /// Glycine (simplest amino acid): H₂N-CH₂-COOH
    /// 3D coordinates from PubChem CID 750 (Angstrom → Bohr).
    pub fn glycine() -> Self {
        Self::new(vec![
            Atom::from_angstrom(8, -1.6487, 0.6571, -0.0104), // O (carbonyl)
            Atom::from_angstrom(8, -0.4837, -1.2934, -0.0005), // O (hydroxyl)
            Atom::from_angstrom(7, 1.9006, -0.0812, -0.0090),  // N (amino)
            Atom::from_angstrom(6, 0.7341, 0.7867, 0.0079),    // C (alpha)
            Atom::from_angstrom(6, -0.5023, -0.0691, 0.0120),   // C (carbonyl)
            Atom::from_angstrom(1, 0.7326, 1.4215, -0.8824),   // H
            Atom::from_angstrom(1, 0.7464, 1.4088, 0.9069),    // H
            Atom::from_angstrom(1, 1.8743, -0.6844, -0.8301),  // H
            Atom::from_angstrom(1, 1.8887, -0.6969, 0.8031),   // H
            Atom::from_angstrom(1, -2.4447, 0.0839, -0.0260),  // H (OH)
        ])
    }

    /// Formaldehyde (H₂CO): simplest carbonyl compound.
    /// 3D coordinates from PubChem CID 712.
    pub fn formaldehyde() -> Self {
        Self::new(vec![
            Atom::from_angstrom(8, 0.6123, 0.0, 0.0),
            Atom::from_angstrom(6, -0.6123, 0.0, 0.0),
            Atom::from_angstrom(1, -1.2000, 0.2426, -0.8998),
            Atom::from_angstrom(1, -1.2000, -0.2424, 0.8998),
        ])
    }

    /// H₂O at experimental geometry (Angstrom → Bohr).
    /// O-H = 0.9572 Å, H-O-H = 104.52°
    pub fn water() -> Self {
        let oh_dist = 0.9572; // Angstrom
        let angle = 104.52_f64.to_radians();

        Self::new(vec![
            Atom::from_angstrom(8, 0.0, 0.0, 0.0),
            Atom::from_angstrom(1, oh_dist, 0.0, 0.0),
            Atom::from_angstrom(
                1,
                oh_dist * angle.cos(),
                oh_dist * angle.sin(),
                0.0,
            ),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_h2_electrons() {
        let h2 = Molecule::h2();
        assert_eq!(h2.n_electrons(), 2);
        assert_eq!(h2.n_occupied(), 1);
    }

    #[test]
    fn test_heh_plus_electrons() {
        let heh = Molecule::heh_plus();
        assert_eq!(heh.n_electrons(), 2); // He(2) + H(1) - charge(1) = 2
        assert_eq!(heh.n_occupied(), 1);
    }

    #[test]
    fn test_water_electrons() {
        let water = Molecule::water();
        assert_eq!(water.n_electrons(), 10);
        assert_eq!(water.n_occupied(), 5);
        assert_eq!(water.n_atoms(), 3);
    }

    #[test]
    fn test_h2_nuclear_repulsion() {
        let h2 = Molecule::h2();
        let v_nn = h2.nuclear_repulsion_energy();
        // Z_H * Z_H / 1.4 = 1/1.4 ≈ 0.7143
        assert!((v_nn - 1.0 / 1.4).abs() < 1e-10);
    }

    #[test]
    fn test_xyz_parsing() {
        let xyz = "2\nH2 molecule\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n";
        let mol = Molecule::from_xyz(xyz).unwrap();
        assert_eq!(mol.n_atoms(), 2);
        assert_eq!(mol.n_electrons(), 2);
        // 0.74 Angstrom → Bohr
        let r = mol.atoms[0].distance_to(&mol.atoms[1]);
        assert!((r - 0.74 * ANGSTROM_TO_BOHR).abs() < 1e-10);
    }
}
