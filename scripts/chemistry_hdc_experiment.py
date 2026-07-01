#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Symthaea Chemistry Experiment: HDC for Molecular Representation

Tests whether Symthaea's consciousness-inspired HDC architecture can
effectively represent and discriminate chemical concepts.

Key hypotheses:
1. HDC binding can compositionally encode molecular structure
2. Chemical class discrimination (organic/inorganic, toxic/benign) is measurable
3. Topological features (rings, chains) emerge in HDC space

Usage:
    cd /srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb
    nix-shell -p "python313.withPackages(ps: with ps; [numpy scipy])" \
        --run "python3 scripts/chemistry_hdc_experiment.py"
"""

import json
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import numpy as np
from scipy import stats


# =============================================================================
# HDC Core (matching Symthaea's 16,384-bit vectors)
# =============================================================================

DIM = 16384


def deterministic_hv(seed: str) -> np.ndarray:
    """Generate deterministic HV from string seed."""
    h = hashlib.sha256(seed.encode()).hexdigest()
    seed_int = int(h[:16], 16)
    rng = np.random.Generator(np.random.PCG64(seed_int))
    return rng.integers(0, 2, size=DIM, dtype=np.uint8)


def bind(*vectors: np.ndarray) -> np.ndarray:
    """Bind multiple vectors via XOR (molecular bonding)."""
    result = vectors[0].copy()
    for v in vectors[1:]:
        result = np.logical_xor(result, v).astype(np.uint8)
    return result


def bundle(vectors: List[np.ndarray]) -> np.ndarray:
    """Bundle vectors via majority vote (superposition/mixture)."""
    if len(vectors) == 0:
        return np.zeros(DIM, dtype=np.uint8)
    counts = np.sum(vectors, axis=0)
    return (counts > len(vectors) / 2).astype(np.uint8)


def permute(hv: np.ndarray, shift: int) -> np.ndarray:
    """Permute HV (for sequence/position encoding)."""
    return np.roll(hv, shift)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine-like similarity (1 - normalized Hamming)."""
    return 1.0 - np.sum(a != b) / DIM


# =============================================================================
# Atomic Basis Vectors
# =============================================================================

class Element:
    """Atomic element with HDC representation."""

    # Class-level cache for element vectors
    _cache: Dict[str, np.ndarray] = {}

    def __init__(self, symbol: str, atomic_number: int,
                 electronegativity: float = 0.0,
                 group: str = "other"):
        self.symbol = symbol
        self.atomic_number = atomic_number
        self.electronegativity = electronegativity
        self.group = group  # organic, metal, halogen, etc.

        # Generate deterministic basis vector
        if symbol not in Element._cache:
            Element._cache[symbol] = deterministic_hv(f"element_{symbol}_{atomic_number}")
        self.hv = Element._cache[symbol]

    def __repr__(self):
        return f"Element({self.symbol})"


# Common elements
ELEMENTS = {
    # Organic
    "H": Element("H", 1, 2.20, "organic"),
    "C": Element("C", 6, 2.55, "organic"),
    "N": Element("N", 7, 3.04, "organic"),
    "O": Element("O", 8, 3.44, "organic"),
    "S": Element("S", 16, 2.58, "organic"),
    "P": Element("P", 15, 2.19, "organic"),

    # Halogens
    "F": Element("F", 9, 3.98, "halogen"),
    "Cl": Element("Cl", 17, 3.16, "halogen"),
    "Br": Element("Br", 35, 2.96, "halogen"),
    "I": Element("I", 53, 2.66, "halogen"),

    # Metals
    "Na": Element("Na", 11, 0.93, "metal"),
    "K": Element("K", 19, 0.82, "metal"),
    "Ca": Element("Ca", 20, 1.00, "metal"),
    "Fe": Element("Fe", 26, 1.83, "metal"),
    "Cu": Element("Cu", 29, 1.90, "metal"),
    "Zn": Element("Zn", 30, 1.65, "metal"),
    "Mg": Element("Mg", 12, 1.31, "metal"),
    "Al": Element("Al", 13, 1.61, "metal"),

    # Noble gases
    "He": Element("He", 2, 0.0, "noble"),
    "Ne": Element("Ne", 10, 0.0, "noble"),
    "Ar": Element("Ar", 18, 0.0, "noble"),
}


# Bond type vectors
BOND_TYPES = {
    "single": deterministic_hv("bond_single"),
    "double": deterministic_hv("bond_double"),
    "triple": deterministic_hv("bond_triple"),
    "aromatic": deterministic_hv("bond_aromatic"),
    "ionic": deterministic_hv("bond_ionic"),
}


# =============================================================================
# Molecular Encoding
# =============================================================================

@dataclass
class Bond:
    """Chemical bond between atoms."""
    atom1_idx: int
    atom2_idx: int
    bond_type: str = "single"


@dataclass
class Molecule:
    """Molecular structure with HDC encoding."""
    name: str
    formula: str
    atoms: List[str]  # Element symbols
    bonds: List[Bond]
    properties: Dict[str, any] = field(default_factory=dict)
    _hv: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def hv(self) -> np.ndarray:
        """Compute HDC representation (cached)."""
        if self._hv is None:
            self._hv = self._encode()
        return self._hv

    def _encode(self) -> np.ndarray:
        """
        Encode molecule using compositional HDC algebra.

        Strategy:
        1. Each atom gets position-permuted basis vector
        2. Each bond binds connected atoms with bond-type vector
        3. Final representation bundles all bonds
        """
        if not self.atoms:
            return np.zeros(DIM, dtype=np.uint8)

        # Get atom vectors with position encoding
        atom_hvs = []
        for i, symbol in enumerate(self.atoms):
            if symbol in ELEMENTS:
                atom_hv = permute(ELEMENTS[symbol].hv, i * 100)
                atom_hvs.append(atom_hv)

        if not atom_hvs:
            return np.zeros(DIM, dtype=np.uint8)

        # If no bonds, just bundle atoms
        if not self.bonds:
            return bundle(atom_hvs)

        # Encode each bond: bind(atom1, bond_type, atom2)
        bond_hvs = []
        for bond in self.bonds:
            if bond.atom1_idx < len(atom_hvs) and bond.atom2_idx < len(atom_hvs):
                bond_type_hv = BOND_TYPES.get(bond.bond_type, BOND_TYPES["single"])
                bond_hv = bind(atom_hvs[bond.atom1_idx], bond_type_hv, atom_hvs[bond.atom2_idx])
                bond_hvs.append(bond_hv)

        # Bundle all bonds + atom context
        all_hvs = atom_hvs + bond_hvs
        return bundle(all_hvs)

    def similarity_to(self, other: 'Molecule') -> float:
        """Compute similarity to another molecule."""
        return similarity(self.hv, other.hv)


# =============================================================================
# Molecule Database
# =============================================================================

def create_molecule_database() -> Dict[str, List[Molecule]]:
    """Create database of molecules organized by class."""

    db = {
        "organic_safe": [],
        "organic_toxic": [],
        "inorganic_safe": [],
        "inorganic_toxic": [],
        "drugs": [],
        "solvents": [],
    }

    # === ORGANIC SAFE ===

    # Methane CH4
    db["organic_safe"].append(Molecule(
        name="methane", formula="CH4",
        atoms=["C", "H", "H", "H", "H"],
        bonds=[Bond(0,1), Bond(0,2), Bond(0,3), Bond(0,4)],
        properties={"toxic": False, "organic": True}
    ))

    # Ethane C2H6
    db["organic_safe"].append(Molecule(
        name="ethane", formula="C2H6",
        atoms=["C", "C", "H", "H", "H", "H", "H", "H"],
        bonds=[Bond(0,1), Bond(0,2), Bond(0,3), Bond(0,4), Bond(1,5), Bond(1,6), Bond(1,7)],
        properties={"toxic": False, "organic": True}
    ))

    # Ethanol C2H5OH
    db["organic_safe"].append(Molecule(
        name="ethanol", formula="C2H5OH",
        atoms=["C", "C", "O", "H", "H", "H", "H", "H", "H"],
        bonds=[Bond(0,1), Bond(1,2), Bond(0,3), Bond(0,4), Bond(0,5), Bond(1,6), Bond(1,7), Bond(2,8)],
        properties={"toxic": False, "organic": True}
    ))

    # Acetic acid CH3COOH
    db["organic_safe"].append(Molecule(
        name="acetic_acid", formula="CH3COOH",
        atoms=["C", "C", "O", "O", "H", "H", "H", "H"],
        bonds=[Bond(0,1), Bond(1,2,"double"), Bond(1,3), Bond(0,4), Bond(0,5), Bond(0,6), Bond(3,7)],
        properties={"toxic": False, "organic": True}
    ))

    # Glucose C6H12O6 (simplified)
    db["organic_safe"].append(Molecule(
        name="glucose", formula="C6H12O6",
        atoms=["C"]*6 + ["O"]*6 + ["H"]*12,
        bonds=[Bond(i, i+1) for i in range(5)] + [Bond(5, 0)],  # Ring
        properties={"toxic": False, "organic": True, "ring": True}
    ))

    # Benzene C6H6 (aromatic)
    db["organic_safe"].append(Molecule(
        name="benzene", formula="C6H6",
        atoms=["C"]*6 + ["H"]*6,
        bonds=[Bond(i, (i+1)%6, "aromatic") for i in range(6)] + [Bond(i, i+6) for i in range(6)],
        properties={"toxic": False, "organic": True, "aromatic": True}
    ))

    # === ORGANIC TOXIC ===

    # Formaldehyde CH2O
    db["organic_toxic"].append(Molecule(
        name="formaldehyde", formula="CH2O",
        atoms=["C", "O", "H", "H"],
        bonds=[Bond(0,1,"double"), Bond(0,2), Bond(0,3)],
        properties={"toxic": True, "organic": True}
    ))

    # Methanol CH3OH
    db["organic_toxic"].append(Molecule(
        name="methanol", formula="CH3OH",
        atoms=["C", "O", "H", "H", "H", "H"],
        bonds=[Bond(0,1), Bond(0,2), Bond(0,3), Bond(0,4), Bond(1,5)],
        properties={"toxic": True, "organic": True}
    ))

    # Chloroform CHCl3
    db["organic_toxic"].append(Molecule(
        name="chloroform", formula="CHCl3",
        atoms=["C", "H", "Cl", "Cl", "Cl"],
        bonds=[Bond(0,1), Bond(0,2), Bond(0,3), Bond(0,4)],
        properties={"toxic": True, "organic": True}
    ))

    # Carbon tetrachloride CCl4
    db["organic_toxic"].append(Molecule(
        name="carbon_tetrachloride", formula="CCl4",
        atoms=["C", "Cl", "Cl", "Cl", "Cl"],
        bonds=[Bond(0,1), Bond(0,2), Bond(0,3), Bond(0,4)],
        properties={"toxic": True, "organic": True}
    ))

    # Cyanide HCN
    db["organic_toxic"].append(Molecule(
        name="hydrogen_cyanide", formula="HCN",
        atoms=["H", "C", "N"],
        bonds=[Bond(0,1), Bond(1,2,"triple")],
        properties={"toxic": True, "organic": True}
    ))

    # Phosgene COCl2
    db["organic_toxic"].append(Molecule(
        name="phosgene", formula="COCl2",
        atoms=["C", "O", "Cl", "Cl"],
        bonds=[Bond(0,1,"double"), Bond(0,2), Bond(0,3)],
        properties={"toxic": True, "organic": True}
    ))

    # === INORGANIC SAFE ===

    # Water H2O
    db["inorganic_safe"].append(Molecule(
        name="water", formula="H2O",
        atoms=["O", "H", "H"],
        bonds=[Bond(0,1), Bond(0,2)],
        properties={"toxic": False, "organic": False}
    ))

    # Sodium chloride NaCl
    db["inorganic_safe"].append(Molecule(
        name="sodium_chloride", formula="NaCl",
        atoms=["Na", "Cl"],
        bonds=[Bond(0,1,"ionic")],
        properties={"toxic": False, "organic": False}
    ))

    # Calcium carbonate CaCO3
    db["inorganic_safe"].append(Molecule(
        name="calcium_carbonate", formula="CaCO3",
        atoms=["Ca", "C", "O", "O", "O"],
        bonds=[Bond(0,1,"ionic"), Bond(1,2,"double"), Bond(1,3), Bond(1,4)],
        properties={"toxic": False, "organic": False}
    ))

    # Magnesium oxide MgO
    db["inorganic_safe"].append(Molecule(
        name="magnesium_oxide", formula="MgO",
        atoms=["Mg", "O"],
        bonds=[Bond(0,1,"ionic")],
        properties={"toxic": False, "organic": False}
    ))

    # Ammonia NH3
    db["inorganic_safe"].append(Molecule(
        name="ammonia", formula="NH3",
        atoms=["N", "H", "H", "H"],
        bonds=[Bond(0,1), Bond(0,2), Bond(0,3)],
        properties={"toxic": False, "organic": False}
    ))

    # === INORGANIC TOXIC ===

    # Lead oxide PbO (simplified - no Pb in our elements, use proxy)
    db["inorganic_toxic"].append(Molecule(
        name="heavy_metal_oxide", formula="MO",
        atoms=["Fe", "O"],  # Proxy for heavy metal
        bonds=[Bond(0,1,"ionic")],
        properties={"toxic": True, "organic": False}
    ))

    # Hydrogen sulfide H2S
    db["inorganic_toxic"].append(Molecule(
        name="hydrogen_sulfide", formula="H2S",
        atoms=["S", "H", "H"],
        bonds=[Bond(0,1), Bond(0,2)],
        properties={"toxic": True, "organic": False}
    ))

    # Chlorine gas Cl2
    db["inorganic_toxic"].append(Molecule(
        name="chlorine_gas", formula="Cl2",
        atoms=["Cl", "Cl"],
        bonds=[Bond(0,1)],
        properties={"toxic": True, "organic": False}
    ))

    # Nitrogen dioxide NO2
    db["inorganic_toxic"].append(Molecule(
        name="nitrogen_dioxide", formula="NO2",
        atoms=["N", "O", "O"],
        bonds=[Bond(0,1,"double"), Bond(0,2)],
        properties={"toxic": True, "organic": False}
    ))

    # Carbon monoxide CO
    db["inorganic_toxic"].append(Molecule(
        name="carbon_monoxide", formula="CO",
        atoms=["C", "O"],
        bonds=[Bond(0,1,"triple")],
        properties={"toxic": True, "organic": False}
    ))

    # === DRUGS (pharmaceutical) ===

    # Aspirin C9H8O4 (simplified)
    db["drugs"].append(Molecule(
        name="aspirin", formula="C9H8O4",
        atoms=["C"]*9 + ["O"]*4 + ["H"]*8,
        bonds=[Bond(i, (i+1)%6, "aromatic") for i in range(6)],  # Benzene ring
        properties={"drug": True, "organic": True}
    ))

    # Caffeine C8H10N4O2 (simplified)
    db["drugs"].append(Molecule(
        name="caffeine", formula="C8H10N4O2",
        atoms=["C"]*8 + ["N"]*4 + ["O"]*2 + ["H"]*10,
        bonds=[Bond(0,1), Bond(1,2), Bond(2,3), Bond(3,0)],  # Imidazole-like
        properties={"drug": True, "organic": True}
    ))

    # Ibuprofen C13H18O2 (simplified)
    db["drugs"].append(Molecule(
        name="ibuprofen", formula="C13H18O2",
        atoms=["C"]*13 + ["O"]*2 + ["H"]*18,
        bonds=[Bond(i, (i+1)%6, "aromatic") for i in range(6)],
        properties={"drug": True, "organic": True}
    ))

    # === SOLVENTS ===

    # Acetone C3H6O
    db["solvents"].append(Molecule(
        name="acetone", formula="C3H6O",
        atoms=["C", "C", "C", "O", "H", "H", "H", "H", "H", "H"],
        bonds=[Bond(0,1), Bond(1,2), Bond(1,3,"double"), Bond(0,4), Bond(0,5), Bond(0,6), Bond(2,7), Bond(2,8), Bond(2,9)],
        properties={"solvent": True, "organic": True}
    ))

    # Diethyl ether C4H10O
    db["solvents"].append(Molecule(
        name="diethyl_ether", formula="C4H10O",
        atoms=["C", "C", "O", "C", "C"] + ["H"]*10,
        bonds=[Bond(0,1), Bond(1,2), Bond(2,3), Bond(3,4)],
        properties={"solvent": True, "organic": True}
    ))

    # DMSO C2H6OS
    db["solvents"].append(Molecule(
        name="dmso", formula="C2H6OS",
        atoms=["C", "C", "S", "O"] + ["H"]*6,
        bonds=[Bond(0,2), Bond(1,2), Bond(2,3,"double")],
        properties={"solvent": True, "organic": True}
    ))

    return db


# =============================================================================
# Analysis Functions
# =============================================================================

def compute_fisher_criterion(class1_hvs: List[np.ndarray],
                             class2_hvs: List[np.ndarray]) -> Dict:
    """Compute Fisher's criterion for class separation."""

    if not class1_hvs or not class2_hvs:
        return {"fisher": 0, "centroid_distance": 0, "avg_within": 0}

    # Convert to float for centroid computation
    c1 = np.array([hv.astype(float) for hv in class1_hvs])
    c2 = np.array([hv.astype(float) for hv in class2_hvs])

    # Centroids
    centroid1 = np.mean(c1, axis=0)
    centroid2 = np.mean(c2, axis=0)

    # Between-class distance
    centroid_distance = np.linalg.norm(centroid1 - centroid2)

    # Within-class variance
    within1 = np.mean([np.linalg.norm(hv - centroid1) for hv in c1])
    within2 = np.mean([np.linalg.norm(hv - centroid2) for hv in c2])
    avg_within = (within1 + within2) / 2

    # Fisher's criterion
    fisher = centroid_distance / avg_within if avg_within > 0 else 0

    # Cosine similarity of centroids
    cos_sim = np.dot(centroid1, centroid2) / (np.linalg.norm(centroid1) * np.linalg.norm(centroid2) + 1e-10)
    angular_sep = 1 - cos_sim

    return {
        "fisher": fisher,
        "centroid_distance": centroid_distance,
        "avg_within": avg_within,
        "angular_separation": angular_sep,
    }


def analyze_molecular_topology(mol: Molecule) -> Dict:
    """Analyze topological features of molecule's HDC representation."""
    hv = mol.hv

    # Density (balance)
    density = np.mean(hv)

    # Block structure (emergent patterns)
    block_size = 256
    n_blocks = DIM // block_size
    block_densities = [np.mean(hv[i*block_size:(i+1)*block_size]) for i in range(n_blocks)]
    block_variance = np.var(block_densities)

    # Autocorrelation (self-similarity)
    bp = 2 * hv.astype(float) - 1
    autocorr_64 = np.corrcoef(bp, np.roll(bp, 64))[0, 1]
    autocorr_256 = np.corrcoef(bp, np.roll(bp, 256))[0, 1]

    return {
        "density": density,
        "block_variance": block_variance,
        "autocorr_64": autocorr_64 if not np.isnan(autocorr_64) else 0,
        "autocorr_256": autocorr_256 if not np.isnan(autocorr_256) else 0,
        "n_atoms": len(mol.atoms),
        "n_bonds": len(mol.bonds),
    }


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment():
    print("\n" + "=" * 70)
    print("   SYMTHAEA CHEMISTRY EXPERIMENT")
    print("   HDC for Molecular Representation")
    print("=" * 70)

    # Create database
    db = create_molecule_database()

    total = sum(len(mols) for mols in db.values())
    print(f"\nMolecule database: {total} molecules")
    for category, mols in db.items():
        print(f"  {category}: {len(mols)}")

    # ==========================================================================
    # Experiment 1: Organic vs Inorganic Discrimination
    # ==========================================================================

    print("\n" + "=" * 70)
    print("   EXPERIMENT 1: Organic vs Inorganic")
    print("=" * 70)

    organic_hvs = [m.hv for m in db["organic_safe"] + db["organic_toxic"]]
    inorganic_hvs = [m.hv for m in db["inorganic_safe"] + db["inorganic_toxic"]]

    fisher_org = compute_fisher_criterion(organic_hvs, inorganic_hvs)

    print(f"\n  Organic molecules: {len(organic_hvs)}")
    print(f"  Inorganic molecules: {len(inorganic_hvs)}")
    print(f"\n  Fisher's criterion: {fisher_org['fisher']:.4f}")
    print(f"  Centroid distance: {fisher_org['centroid_distance']:.2f}")
    print(f"  Angular separation: {fisher_org['angular_separation']:.4f}")

    # ==========================================================================
    # Experiment 2: Toxic vs Safe Discrimination
    # ==========================================================================

    print("\n" + "=" * 70)
    print("   EXPERIMENT 2: Toxic vs Safe")
    print("=" * 70)

    toxic_hvs = [m.hv for m in db["organic_toxic"] + db["inorganic_toxic"]]
    safe_hvs = [m.hv for m in db["organic_safe"] + db["inorganic_safe"]]

    fisher_tox = compute_fisher_criterion(toxic_hvs, safe_hvs)

    print(f"\n  Toxic molecules: {len(toxic_hvs)}")
    print(f"  Safe molecules: {len(safe_hvs)}")
    print(f"\n  Fisher's criterion: {fisher_tox['fisher']:.4f}")
    print(f"  Centroid distance: {fisher_tox['centroid_distance']:.2f}")
    print(f"  Angular separation: {fisher_tox['angular_separation']:.4f}")

    # ==========================================================================
    # Experiment 3: Similarity Analysis
    # ==========================================================================

    print("\n" + "=" * 70)
    print("   EXPERIMENT 3: Molecular Similarity")
    print("=" * 70)

    # Similar molecules should have high similarity
    methane = db["organic_safe"][0]  # CH4
    ethane = db["organic_safe"][1]   # C2H6
    water = db["inorganic_safe"][0]  # H2O

    print(f"\n  Methane ↔ Ethane (similar): {methane.similarity_to(ethane):.4f}")
    print(f"  Methane ↔ Water (different): {methane.similarity_to(water):.4f}")

    # Cross-class similarities
    all_organic = db["organic_safe"] + db["organic_toxic"]
    all_inorganic = db["inorganic_safe"] + db["inorganic_toxic"]

    within_organic = []
    for i, m1 in enumerate(all_organic):
        for m2 in all_organic[i+1:]:
            within_organic.append(m1.similarity_to(m2))

    within_inorganic = []
    for i, m1 in enumerate(all_inorganic):
        for m2 in all_inorganic[i+1:]:
            within_inorganic.append(m2.similarity_to(m2))

    between_class = []
    for m1 in all_organic:
        for m2 in all_inorganic:
            between_class.append(m1.similarity_to(m2))

    print(f"\n  Within-organic similarity: {np.mean(within_organic):.4f} ± {np.std(within_organic):.4f}")
    print(f"  Within-inorganic similarity: {np.mean(within_inorganic):.4f} ± {np.std(within_inorganic):.4f}")
    print(f"  Between-class similarity: {np.mean(between_class):.4f} ± {np.std(between_class):.4f}")

    # Statistical test
    t_stat, p_value = stats.ttest_ind(within_organic, between_class)
    print(f"\n  Within vs Between t-test: t={t_stat:.3f}, p={p_value:.4f}")

    # ==========================================================================
    # Experiment 4: Compositional Algebra Test
    # ==========================================================================

    print("\n" + "=" * 70)
    print("   EXPERIMENT 4: Compositional Algebra")
    print("=" * 70)

    # Test: Can we "unbind" to recover components?
    C = ELEMENTS["C"].hv
    H = ELEMENTS["H"].hv
    O = ELEMENTS["O"].hv

    # Bind C-H (like in CH4)
    CH_bond = bind(C, BOND_TYPES["single"], H)

    # Unbind to recover H
    recovered_H = bind(CH_bond, C, BOND_TYPES["single"])

    recovery_sim = similarity(recovered_H, H)
    print(f"\n  CH bond encoded, then unbound")
    print(f"  Recovered H similarity to original H: {recovery_sim:.4f}")

    # Test: Molecular analogy
    # ethanol - methanol ≈ CH2 (extra carbon)
    ethanol = db["organic_safe"][2]  # C2H5OH
    methanol = db["organic_toxic"][1]  # CH3OH

    diff = bind(ethanol.hv, methanol.hv)  # XOR = symmetric difference

    # This should be similar to a CH2 group
    CH2 = bind(C, permute(H, 100), permute(H, 200))

    analogy_sim = similarity(diff, CH2)
    print(f"\n  Molecular analogy: ethanol ⊕ methanol")
    print(f"  Similarity to CH2 unit: {analogy_sim:.4f}")

    # ==========================================================================
    # Experiment 5: Topological Analysis
    # ==========================================================================

    print("\n" + "=" * 70)
    print("   EXPERIMENT 5: Topological Features")
    print("=" * 70)

    # Analyze topology of different molecule types
    print("\n  Molecule topological signatures:")
    print(f"  {'Name':<20} {'Atoms':<8} {'Bonds':<8} {'Density':<10} {'BlockVar':<10}")
    print("  " + "-" * 56)

    for category in ["organic_safe", "organic_toxic", "inorganic_safe"]:
        for mol in db[category][:3]:
            topo = analyze_molecular_topology(mol)
            print(f"  {mol.name:<20} {topo['n_atoms']:<8} {topo['n_bonds']:<8} "
                  f"{topo['density']:.4f}    {topo['block_variance']:.6f}")

    # ==========================================================================
    # Summary
    # ==========================================================================

    print("\n" + "=" * 70)
    print("   SUMMARY")
    print("=" * 70)

    print(f"""
  1. ORGANIC/INORGANIC DISCRIMINATION
     Fisher's criterion: {fisher_org['fisher']:.4f}
     → {'GOOD' if fisher_org['fisher'] > 1.0 else 'MODERATE' if fisher_org['fisher'] > 0.5 else 'WEAK'} class separation

  2. TOXIC/SAFE DISCRIMINATION
     Fisher's criterion: {fisher_tox['fisher']:.4f}
     → {'GOOD' if fisher_tox['fisher'] > 1.0 else 'MODERATE' if fisher_tox['fisher'] > 0.5 else 'WEAK'} class separation

  3. SIMILARITY STRUCTURE
     Within-class > Between-class: p = {p_value:.4f}
     → {'CONFIRMED' if p_value < 0.05 else 'NOT CONFIRMED'}

  4. COMPOSITIONAL ALGEBRA
     Unbinding recovery: {recovery_sim:.4f}
     → {'WORKING' if recovery_sim > 0.9 else 'PARTIAL' if recovery_sim > 0.7 else 'WEAK'}
""")

    # Key finding
    print("  KEY FINDING:")
    if fisher_org['fisher'] > 0.8 and fisher_tox['fisher'] > 0.5:
        finding = "SUCCESS"
        print("  ✓ HDC compositional encoding successfully discriminates")
        print("    chemical classes (organic/inorganic, toxic/safe)")
    elif fisher_org['fisher'] > 0.5 or fisher_tox['fisher'] > 0.5:
        finding = "PARTIAL"
        print("  ~ Partial success - some discriminations work better than others")
    else:
        finding = "NEEDS_WORK"
        print("  ✗ Current encoding needs refinement")

    # Save results
    results = {
        "experiment": "chemistry_hdc",
        "n_molecules": total,
        "organic_inorganic_discrimination": fisher_org,
        "toxic_safe_discrimination": fisher_tox,
        "similarity_analysis": {
            "within_organic_mean": float(np.mean(within_organic)),
            "between_class_mean": float(np.mean(between_class)),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
        },
        "compositional_algebra": {
            "unbinding_recovery": float(recovery_sim),
            "analogy_similarity": float(analogy_sim),
        },
        "finding": finding,
    }

    output_path = "/srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb/data/chemistry_hdc_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {output_path}")

    print("\n" + "=" * 70 + "\n")

    return results


if __name__ == "__main__":
    run_experiment()
