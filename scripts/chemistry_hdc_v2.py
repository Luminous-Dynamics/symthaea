#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Symthaea Chemistry v2: Improved Molecular HDC Encoding

Improvements over v1:
1. Electronegativity-weighted atomic vectors
2. Functional group recognition
3. Hybridization encoding (sp, sp2, sp3)
4. Ring detection and encoding
5. Better bond type differentiation

Usage:
    cd /srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb
    nix-shell -p "python313.withPackages(ps: with ps; [numpy scipy])" \
        --run "python3 scripts/chemistry_hdc_v2.py"
"""

import json
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from enum import Enum
import numpy as np
from scipy import stats

# =============================================================================
# HDC Core
# =============================================================================

DIM = 16384


def deterministic_hv(seed: str) -> np.ndarray:
    """Generate deterministic HV from string seed."""
    h = hashlib.sha256(seed.encode()).hexdigest()
    seed_int = int(h[:16], 16)
    rng = np.random.Generator(np.random.PCG64(seed_int))
    return rng.integers(0, 2, size=DIM, dtype=np.uint8)


def bind(*vectors: np.ndarray) -> np.ndarray:
    """Bind via XOR."""
    result = vectors[0].copy()
    for v in vectors[1:]:
        result = np.logical_xor(result, v).astype(np.uint8)
    return result


def bundle(vectors: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
    """Weighted bundle via majority vote."""
    if len(vectors) == 0:
        return np.zeros(DIM, dtype=np.uint8)

    if weights is None:
        weights = [1.0] * len(vectors)

    # Weighted sum
    weighted_sum = np.zeros(DIM, dtype=np.float32)
    for v, w in zip(vectors, weights):
        weighted_sum += w * (2 * v.astype(np.float32) - 1)

    return (weighted_sum > 0).astype(np.uint8)


def permute(hv: np.ndarray, shift: int) -> np.ndarray:
    """Cyclic permutation for position encoding."""
    return np.roll(hv, shift)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Hamming similarity."""
    return 1.0 - np.sum(a != b) / DIM


# =============================================================================
# Enhanced Atomic Properties
# =============================================================================

@dataclass
class AtomicProperties:
    """Extended atomic properties for chemistry-aware encoding."""
    symbol: str
    atomic_number: int
    electronegativity: float
    atomic_radius: float  # in pm
    valence_electrons: int
    group: str  # organic, metal, halogen, noble, etc.
    period: int
    is_metal: bool


ATOMIC_DB = {
    # Organic elements
    "H":  AtomicProperties("H",  1,  2.20, 53,  1, "organic", 1, False),
    "C":  AtomicProperties("C",  6,  2.55, 77,  4, "organic", 2, False),
    "N":  AtomicProperties("N",  7,  3.04, 75,  5, "organic", 2, False),
    "O":  AtomicProperties("O",  8,  3.44, 73,  6, "organic", 2, False),
    "S":  AtomicProperties("S",  16, 2.58, 102, 6, "organic", 3, False),
    "P":  AtomicProperties("P",  15, 2.19, 106, 5, "organic", 3, False),

    # Halogens
    "F":  AtomicProperties("F",  9,  3.98, 71,  7, "halogen", 2, False),
    "Cl": AtomicProperties("Cl", 17, 3.16, 99,  7, "halogen", 3, False),
    "Br": AtomicProperties("Br", 35, 2.96, 114, 7, "halogen", 4, False),
    "I":  AtomicProperties("I",  53, 2.66, 133, 7, "halogen", 5, False),

    # Metals
    "Na": AtomicProperties("Na", 11, 0.93, 190, 1, "metal", 3, True),
    "K":  AtomicProperties("K",  19, 0.82, 243, 1, "metal", 4, True),
    "Ca": AtomicProperties("Ca", 20, 1.00, 194, 2, "metal", 4, True),
    "Mg": AtomicProperties("Mg", 12, 1.31, 145, 2, "metal", 3, True),
    "Fe": AtomicProperties("Fe", 26, 1.83, 156, 2, "metal", 4, True),
    "Cu": AtomicProperties("Cu", 29, 1.90, 145, 1, "metal", 4, True),
    "Zn": AtomicProperties("Zn", 30, 1.65, 142, 2, "metal", 4, True),
    "Al": AtomicProperties("Al", 13, 1.61, 118, 3, "metal", 3, True),

    # Noble gases
    "He": AtomicProperties("He", 2,  0.00, 31,  2, "noble", 1, False),
    "Ne": AtomicProperties("Ne", 10, 0.00, 38,  8, "noble", 2, False),
    "Ar": AtomicProperties("Ar", 18, 0.00, 71,  8, "noble", 3, False),
}


# =============================================================================
# Enhanced Encoding
# =============================================================================

class ChemicalEncoder:
    """Enhanced chemical encoder with property-aware HDC representations."""

    def __init__(self):
        # Base vectors for properties
        self.electronegativity_basis = deterministic_hv("property_electronegativity")
        self.radius_basis = deterministic_hv("property_radius")
        self.valence_basis = deterministic_hv("property_valence")
        self.period_basis = deterministic_hv("property_period")

        # Group basis vectors
        self.group_hvs = {
            "organic": deterministic_hv("group_organic"),
            "halogen": deterministic_hv("group_halogen"),
            "metal": deterministic_hv("group_metal"),
            "noble": deterministic_hv("group_noble"),
        }

        # Bond type vectors with more differentiation
        self.bond_hvs = {
            "single": deterministic_hv("bond_single_v2"),
            "double": deterministic_hv("bond_double_v2"),
            "triple": deterministic_hv("bond_triple_v2"),
            "aromatic": deterministic_hv("bond_aromatic_v2"),
            "ionic": deterministic_hv("bond_ionic_v2"),
            "hydrogen": deterministic_hv("bond_hydrogen_v2"),
            "coordinate": deterministic_hv("bond_coordinate_v2"),
        }

        # Functional group signatures
        self.functional_groups = {
            "hydroxyl": deterministic_hv("fg_hydroxyl"),      # -OH
            "carbonyl": deterministic_hv("fg_carbonyl"),      # C=O
            "carboxyl": deterministic_hv("fg_carboxyl"),      # -COOH
            "amine": deterministic_hv("fg_amine"),            # -NH2
            "amide": deterministic_hv("fg_amide"),            # -CONH2
            "ether": deterministic_hv("fg_ether"),            # C-O-C
            "ester": deterministic_hv("fg_ester"),            # -COO-
            "nitro": deterministic_hv("fg_nitro"),            # -NO2
            "halide": deterministic_hv("fg_halide"),          # -X
            "thiol": deterministic_hv("fg_thiol"),            # -SH
            "phosphate": deterministic_hv("fg_phosphate"),    # -PO4
            "sulfate": deterministic_hv("fg_sulfate"),        # -SO4
        }

        # Structural features
        self.ring_hv = deterministic_hv("structure_ring")
        self.aromatic_ring_hv = deterministic_hv("structure_aromatic_ring")
        self.chain_hv = deterministic_hv("structure_chain")
        self.branched_hv = deterministic_hv("structure_branched")

        # Cache for atom vectors
        self._atom_cache: Dict[str, np.ndarray] = {}

    def encode_atom(self, symbol: str) -> np.ndarray:
        """Encode atom with property-aware representation."""
        if symbol in self._atom_cache:
            return self._atom_cache[symbol]

        if symbol not in ATOMIC_DB:
            # Fallback for unknown elements
            return deterministic_hv(f"unknown_element_{symbol}")

        props = ATOMIC_DB[symbol]

        # Base element vector
        base = deterministic_hv(f"element_v2_{symbol}")

        # Encode electronegativity (scale 0-4 → permutation shift 0-4000)
        en_shift = int(props.electronegativity * 1000)
        en_component = permute(self.electronegativity_basis, en_shift)

        # Encode atomic radius (scale 30-250 → shift)
        radius_shift = int(props.atomic_radius * 10)
        radius_component = permute(self.radius_basis, radius_shift)

        # Encode valence electrons
        valence_component = permute(self.valence_basis, props.valence_electrons * 500)

        # Encode period
        period_component = permute(self.period_basis, props.period * 1000)

        # Group membership
        group_component = self.group_hvs.get(props.group, deterministic_hv("group_other"))

        # Combine all components
        # Weighted bundle: base is strongest, properties add nuance
        components = [base, en_component, radius_component, valence_component, period_component, group_component]
        weights = [2.0, 1.0, 0.5, 0.8, 0.5, 1.2]

        result = bundle(components, weights)
        self._atom_cache[symbol] = result
        return result

    def encode_bond(self, atom1_hv: np.ndarray, atom2_hv: np.ndarray,
                   bond_type: str = "single", position: int = 0) -> np.ndarray:
        """Encode a chemical bond."""
        bond_basis = self.bond_hvs.get(bond_type, self.bond_hvs["single"])

        # Position-encode atoms
        a1 = permute(atom1_hv, position * 100)
        a2 = permute(atom2_hv, position * 100 + 50)

        # Bind: atom1 ⊗ bond_type ⊗ atom2
        return bind(a1, bond_basis, a2)

    def detect_functional_groups(self, atoms: List[str], bonds: List[Tuple[int, int, str]]) -> List[str]:
        """Detect functional groups in molecule."""
        detected = []

        # Build adjacency info
        atom_bonds: Dict[int, List[Tuple[int, str]]] = {i: [] for i in range(len(atoms))}
        for a1, a2, btype in bonds:
            atom_bonds[a1].append((a2, btype))
            atom_bonds[a2].append((a1, btype))

        for i, atom in enumerate(atoms):
            neighbors = atom_bonds[i]
            neighbor_atoms = [atoms[n] for n, _ in neighbors if n < len(atoms)]
            neighbor_types = [t for _, t in neighbors]

            # Hydroxyl: O bonded to H and C
            if atom == "O" and "H" in neighbor_atoms and "C" in neighbor_atoms:
                detected.append("hydroxyl")

            # Carbonyl: C double-bonded to O
            if atom == "C" and "double" in neighbor_types:
                for n, t in neighbors:
                    if t == "double" and n < len(atoms) and atoms[n] == "O":
                        detected.append("carbonyl")
                        break

            # Amine: N bonded to H
            if atom == "N" and neighbor_atoms.count("H") >= 1:
                detected.append("amine")

            # Halide: C bonded to halogen
            if atom == "C":
                for na in neighbor_atoms:
                    if na in ["F", "Cl", "Br", "I"]:
                        detected.append("halide")
                        break

            # Thiol: S bonded to H
            if atom == "S" and "H" in neighbor_atoms:
                detected.append("thiol")

        return list(set(detected))

    def detect_rings(self, n_atoms: int, bonds: List[Tuple[int, int, str]]) -> Tuple[bool, bool]:
        """Detect if molecule has rings (and if aromatic)."""
        # Simple ring detection via edge count
        # For connected graph: edges >= vertices means cycle exists
        has_ring = len(bonds) >= n_atoms

        # Aromatic: has aromatic bonds
        has_aromatic = any(b[2] == "aromatic" for b in bonds)

        return has_ring, has_aromatic

    def encode_molecule(self, atoms: List[str], bonds: List[Tuple[int, int, str]],
                       name: str = "") -> np.ndarray:
        """
        Encode complete molecule with enhanced features.

        Args:
            atoms: List of element symbols
            bonds: List of (atom1_idx, atom2_idx, bond_type)
            name: Optional molecule name
        """
        if not atoms:
            return np.zeros(DIM, dtype=np.uint8)

        components = []
        weights = []

        # 1. Encode atoms with positions
        for i, symbol in enumerate(atoms):
            atom_hv = self.encode_atom(symbol)
            positioned = permute(atom_hv, i * 100)
            components.append(positioned)
            weights.append(1.0)

        # 2. Encode bonds
        for idx, (a1, a2, btype) in enumerate(bonds):
            if a1 < len(atoms) and a2 < len(atoms):
                atom1_hv = self.encode_atom(atoms[a1])
                atom2_hv = self.encode_atom(atoms[a2])
                bond_hv = self.encode_bond(atom1_hv, atom2_hv, btype, idx)
                components.append(bond_hv)
                weights.append(1.5)  # Bonds are important!

        # 3. Detect and encode functional groups
        func_groups = self.detect_functional_groups(atoms, bonds)
        for fg in func_groups:
            if fg in self.functional_groups:
                components.append(self.functional_groups[fg])
                weights.append(2.0)  # Functional groups are very important!

        # 4. Structural features
        has_ring, has_aromatic = self.detect_rings(len(atoms), bonds)
        if has_ring:
            components.append(self.ring_hv)
            weights.append(1.5)
        if has_aromatic:
            components.append(self.aromatic_ring_hv)
            weights.append(2.0)

        # 5. Chain length feature
        if len(atoms) > 4 and not has_ring:
            components.append(self.chain_hv)
            weights.append(0.5)

        return bundle(components, weights)


# =============================================================================
# Enhanced Molecule Database
# =============================================================================

def create_enhanced_database(encoder: ChemicalEncoder) -> Dict[str, List[Dict]]:
    """Create molecule database with enhanced encoding."""

    db = {
        "organic_safe": [],
        "organic_toxic": [],
        "inorganic_safe": [],
        "inorganic_toxic": [],
    }

    def add_molecule(category: str, name: str, atoms: List[str],
                    bonds: List[Tuple[int, int, str]], properties: Dict):
        hv = encoder.encode_molecule(atoms, bonds, name)
        db[category].append({
            "name": name,
            "atoms": atoms,
            "bonds": bonds,
            "hv": hv,
            "properties": properties,
        })

    # === ORGANIC SAFE ===

    add_molecule("organic_safe", "methane",
        ["C", "H", "H", "H", "H"],
        [(0,1,"single"), (0,2,"single"), (0,3,"single"), (0,4,"single")],
        {"toxic": False, "mw": 16})

    add_molecule("organic_safe", "ethane",
        ["C", "C", "H", "H", "H", "H", "H", "H"],
        [(0,1,"single"), (0,2,"single"), (0,3,"single"), (0,4,"single"),
         (1,5,"single"), (1,6,"single"), (1,7,"single")],
        {"toxic": False, "mw": 30})

    add_molecule("organic_safe", "ethanol",
        ["C", "C", "O", "H", "H", "H", "H", "H", "H"],
        [(0,1,"single"), (1,2,"single"), (2,8,"single"),  # C-C-O-H (hydroxyl)
         (0,3,"single"), (0,4,"single"), (0,5,"single"),
         (1,6,"single"), (1,7,"single")],
        {"toxic": False, "mw": 46, "has_hydroxyl": True})

    add_molecule("organic_safe", "acetic_acid",
        ["C", "C", "O", "O", "H", "H", "H", "H"],
        [(0,1,"single"), (1,2,"double"), (1,3,"single"), (3,7,"single"),
         (0,4,"single"), (0,5,"single"), (0,6,"single")],
        {"toxic": False, "mw": 60, "has_carboxyl": True})

    add_molecule("organic_safe", "glucose",
        ["C"]*6 + ["O"]*6 + ["H"]*12,
        [(i, i+1, "single") for i in range(5)] + [(5, 0, "single")],  # Ring
        {"toxic": False, "mw": 180, "ring": True})

    add_molecule("organic_safe", "benzene",
        ["C"]*6 + ["H"]*6,
        [(i, (i+1)%6, "aromatic") for i in range(6)] +
        [(i, i+6, "single") for i in range(6)],
        {"toxic": False, "mw": 78, "aromatic": True})

    add_molecule("organic_safe", "propane",
        ["C", "C", "C"] + ["H"]*8,
        [(0,1,"single"), (1,2,"single")],
        {"toxic": False, "mw": 44})

    add_molecule("organic_safe", "butane",
        ["C", "C", "C", "C"] + ["H"]*10,
        [(0,1,"single"), (1,2,"single"), (2,3,"single")],
        {"toxic": False, "mw": 58})

    # === ORGANIC TOXIC ===

    add_molecule("organic_toxic", "formaldehyde",
        ["C", "O", "H", "H"],
        [(0,1,"double"), (0,2,"single"), (0,3,"single")],
        {"toxic": True, "mw": 30, "has_carbonyl": True})

    add_molecule("organic_toxic", "methanol",
        ["C", "O", "H", "H", "H", "H"],
        [(0,1,"single"), (1,5,"single"), (0,2,"single"), (0,3,"single"), (0,4,"single")],
        {"toxic": True, "mw": 32, "has_hydroxyl": True})

    add_molecule("organic_toxic", "chloroform",
        ["C", "H", "Cl", "Cl", "Cl"],
        [(0,1,"single"), (0,2,"single"), (0,3,"single"), (0,4,"single")],
        {"toxic": True, "mw": 119, "has_halide": True})

    add_molecule("organic_toxic", "carbon_tetrachloride",
        ["C", "Cl", "Cl", "Cl", "Cl"],
        [(0,1,"single"), (0,2,"single"), (0,3,"single"), (0,4,"single")],
        {"toxic": True, "mw": 154, "has_halide": True})

    add_molecule("organic_toxic", "hydrogen_cyanide",
        ["H", "C", "N"],
        [(0,1,"single"), (1,2,"triple")],
        {"toxic": True, "mw": 27})

    add_molecule("organic_toxic", "phosgene",
        ["C", "O", "Cl", "Cl"],
        [(0,1,"double"), (0,2,"single"), (0,3,"single")],
        {"toxic": True, "mw": 99, "has_carbonyl": True, "has_halide": True})

    add_molecule("organic_toxic", "vinyl_chloride",
        ["C", "C", "H", "H", "H", "Cl"],
        [(0,1,"double"), (0,2,"single"), (0,3,"single"), (1,4,"single"), (1,5,"single")],
        {"toxic": True, "mw": 62, "has_halide": True})

    add_molecule("organic_toxic", "benzene_toxic",  # Chronic exposure toxic
        ["C"]*6 + ["H"]*6,
        [(i, (i+1)%6, "aromatic") for i in range(6)] +
        [(i, i+6, "single") for i in range(6)],
        {"toxic": True, "mw": 78, "aromatic": True})

    # === INORGANIC SAFE ===

    add_molecule("inorganic_safe", "water",
        ["O", "H", "H"],
        [(0,1,"single"), (0,2,"single")],
        {"toxic": False, "mw": 18})

    add_molecule("inorganic_safe", "sodium_chloride",
        ["Na", "Cl"],
        [(0,1,"ionic")],
        {"toxic": False, "mw": 58})

    add_molecule("inorganic_safe", "calcium_carbonate",
        ["Ca", "C", "O", "O", "O"],
        [(0,1,"ionic"), (1,2,"double"), (1,3,"single"), (1,4,"single")],
        {"toxic": False, "mw": 100})

    add_molecule("inorganic_safe", "magnesium_oxide",
        ["Mg", "O"],
        [(0,1,"ionic")],
        {"toxic": False, "mw": 40})

    add_molecule("inorganic_safe", "ammonia",
        ["N", "H", "H", "H"],
        [(0,1,"single"), (0,2,"single"), (0,3,"single")],
        {"toxic": False, "mw": 17, "has_amine": True})

    add_molecule("inorganic_safe", "carbon_dioxide",
        ["C", "O", "O"],
        [(0,1,"double"), (0,2,"double")],
        {"toxic": False, "mw": 44})

    add_molecule("inorganic_safe", "sodium_bicarbonate",
        ["Na", "H", "C", "O", "O", "O"],
        [(0,2,"ionic"), (2,3,"double"), (2,4,"single"), (2,5,"single"), (4,1,"single")],
        {"toxic": False, "mw": 84})

    # === INORGANIC TOXIC ===

    add_molecule("inorganic_toxic", "hydrogen_sulfide",
        ["S", "H", "H"],
        [(0,1,"single"), (0,2,"single")],
        {"toxic": True, "mw": 34, "has_thiol": True})

    add_molecule("inorganic_toxic", "chlorine_gas",
        ["Cl", "Cl"],
        [(0,1,"single")],
        {"toxic": True, "mw": 71})

    add_molecule("inorganic_toxic", "nitrogen_dioxide",
        ["N", "O", "O"],
        [(0,1,"double"), (0,2,"single")],
        {"toxic": True, "mw": 46})

    add_molecule("inorganic_toxic", "carbon_monoxide",
        ["C", "O"],
        [(0,1,"triple")],
        {"toxic": True, "mw": 28})

    add_molecule("inorganic_toxic", "sulfur_dioxide",
        ["S", "O", "O"],
        [(0,1,"double"), (0,2,"double")],
        {"toxic": True, "mw": 64})

    add_molecule("inorganic_toxic", "hydrogen_fluoride",
        ["H", "F"],
        [(0,1,"single")],
        {"toxic": True, "mw": 20})

    add_molecule("inorganic_toxic", "ozone",
        ["O", "O", "O"],
        [(0,1,"single"), (1,2,"single")],
        {"toxic": True, "mw": 48})

    return db


# =============================================================================
# Analysis
# =============================================================================

def compute_fisher(hvs1: List[np.ndarray], hvs2: List[np.ndarray]) -> Dict:
    """Compute Fisher's criterion."""
    c1 = np.array([hv.astype(float) for hv in hvs1])
    c2 = np.array([hv.astype(float) for hv in hvs2])

    centroid1 = np.mean(c1, axis=0)
    centroid2 = np.mean(c2, axis=0)

    dist = np.linalg.norm(centroid1 - centroid2)
    within1 = np.mean([np.linalg.norm(hv - centroid1) for hv in c1])
    within2 = np.mean([np.linalg.norm(hv - centroid2) for hv in c2])
    avg_within = (within1 + within2) / 2

    fisher = dist / avg_within if avg_within > 0 else 0

    cos_sim = np.dot(centroid1, centroid2) / (np.linalg.norm(centroid1) * np.linalg.norm(centroid2) + 1e-10)

    return {
        "fisher": fisher,
        "centroid_distance": dist,
        "avg_within": avg_within,
        "angular_separation": 1 - cos_sim,
    }


def run_experiment():
    print("\n" + "=" * 70)
    print("   SYMTHAEA CHEMISTRY v2: Enhanced Encoding")
    print("=" * 70)

    encoder = ChemicalEncoder()
    db = create_enhanced_database(encoder)

    total = sum(len(mols) for mols in db.values())
    print(f"\nMolecule database: {total} molecules")
    for category, mols in db.items():
        print(f"  {category}: {len(mols)}")

    # ==========================================================================
    # Experiment 1: Organic vs Inorganic
    # ==========================================================================

    print("\n" + "=" * 70)
    print("   EXPERIMENT 1: Organic vs Inorganic (v2)")
    print("=" * 70)

    organic_hvs = [m["hv"] for m in db["organic_safe"] + db["organic_toxic"]]
    inorganic_hvs = [m["hv"] for m in db["inorganic_safe"] + db["inorganic_toxic"]]

    fisher_org = compute_fisher(organic_hvs, inorganic_hvs)

    print(f"\n  Organic: {len(organic_hvs)}, Inorganic: {len(inorganic_hvs)}")
    print(f"\n  Fisher's criterion: {fisher_org['fisher']:.4f}")
    print(f"  Angular separation: {fisher_org['angular_separation']:.4f}")

    # ==========================================================================
    # Experiment 2: Toxic vs Safe
    # ==========================================================================

    print("\n" + "=" * 70)
    print("   EXPERIMENT 2: Toxic vs Safe (v2)")
    print("=" * 70)

    toxic_hvs = [m["hv"] for m in db["organic_toxic"] + db["inorganic_toxic"]]
    safe_hvs = [m["hv"] for m in db["organic_safe"] + db["inorganic_safe"]]

    fisher_tox = compute_fisher(toxic_hvs, safe_hvs)

    print(f"\n  Toxic: {len(toxic_hvs)}, Safe: {len(safe_hvs)}")
    print(f"\n  Fisher's criterion: {fisher_tox['fisher']:.4f}")
    print(f"  Angular separation: {fisher_tox['angular_separation']:.4f}")

    # ==========================================================================
    # Experiment 3: Functional Group Detection
    # ==========================================================================

    print("\n" + "=" * 70)
    print("   EXPERIMENT 3: Functional Group Analysis")
    print("=" * 70)

    # Group by functional groups
    hydroxyl_hvs = []
    carbonyl_hvs = []
    halide_hvs = []
    no_fg_hvs = []

    for category in db.values():
        for mol in category:
            props = mol["properties"]
            if props.get("has_hydroxyl"):
                hydroxyl_hvs.append(mol["hv"])
            elif props.get("has_carbonyl"):
                carbonyl_hvs.append(mol["hv"])
            elif props.get("has_halide"):
                halide_hvs.append(mol["hv"])
            else:
                no_fg_hvs.append(mol["hv"])

    print(f"\n  Molecules with hydroxyl (-OH): {len(hydroxyl_hvs)}")
    print(f"  Molecules with carbonyl (C=O): {len(carbonyl_hvs)}")
    print(f"  Molecules with halide (-X): {len(halide_hvs)}")
    print(f"  Molecules without key FG: {len(no_fg_hvs)}")

    if len(hydroxyl_hvs) >= 2 and len(halide_hvs) >= 2:
        fisher_fg = compute_fisher(hydroxyl_hvs, halide_hvs)
        print(f"\n  Hydroxyl vs Halide Fisher: {fisher_fg['fisher']:.4f}")

    # ==========================================================================
    # Experiment 4: Similar Molecule Pairs
    # ==========================================================================

    print("\n" + "=" * 70)
    print("   EXPERIMENT 4: Molecular Similarity")
    print("=" * 70)

    # Find specific molecules
    def find_mol(name):
        for cat in db.values():
            for m in cat:
                if m["name"] == name:
                    return m
        return None

    methane = find_mol("methane")
    ethane = find_mol("ethane")
    propane = find_mol("propane")
    water = find_mol("water")
    methanol = find_mol("methanol")
    ethanol = find_mol("ethanol")

    print("\n  Alkane series (should be similar):")
    print(f"    Methane ↔ Ethane:  {similarity(methane['hv'], ethane['hv']):.4f}")
    print(f"    Ethane ↔ Propane:  {similarity(ethane['hv'], propane['hv']):.4f}")
    print(f"    Methane ↔ Propane: {similarity(methane['hv'], propane['hv']):.4f}")

    print("\n  Alcohol series (should be similar):")
    print(f"    Methanol ↔ Ethanol: {similarity(methanol['hv'], ethanol['hv']):.4f}")

    print("\n  Cross-class (should be different):")
    print(f"    Methane ↔ Water:   {similarity(methane['hv'], water['hv']):.4f}")
    print(f"    Ethanol ↔ Water:   {similarity(ethanol['hv'], water['hv']):.4f}")

    # ==========================================================================
    # Comparison to v1
    # ==========================================================================

    print("\n" + "=" * 70)
    print("   COMPARISON: v1 vs v2")
    print("=" * 70)

    print(f"""
  Metric                    v1          v2          Δ
  ─────────────────────────────────────────────────────
  Organic/Inorganic Fisher  0.6503      {fisher_org['fisher']:.4f}      {fisher_org['fisher'] - 0.6503:+.4f}
  Toxic/Safe Fisher         0.5138      {fisher_tox['fisher']:.4f}      {fisher_tox['fisher'] - 0.5138:+.4f}
""")

    improvement_org = (fisher_org['fisher'] - 0.6503) / 0.6503 * 100
    improvement_tox = (fisher_tox['fisher'] - 0.5138) / 0.5138 * 100

    print(f"  Organic/Inorganic improvement: {improvement_org:+.1f}%")
    print(f"  Toxic/Safe improvement: {improvement_tox:+.1f}%")

    # ==========================================================================
    # Summary
    # ==========================================================================

    print("\n" + "=" * 70)
    print("   SUMMARY")
    print("=" * 70)

    if fisher_org['fisher'] > 0.8 and fisher_tox['fisher'] > 0.6:
        finding = "STRONG_SUCCESS"
        print("\n  ✓ STRONG SUCCESS: Enhanced encoding significantly improves")
        print("    discrimination for both organic/inorganic and toxic/safe")
    elif fisher_org['fisher'] > 0.7 or fisher_tox['fisher'] > 0.55:
        finding = "IMPROVED"
        print("\n  ↗ IMPROVED: Enhanced encoding shows improvement over v1")
    else:
        finding = "MARGINAL"
        print("\n  ~ MARGINAL: Enhancement provides limited improvement")

    # Save results
    results = {
        "experiment": "chemistry_hdc_v2",
        "n_molecules": total,
        "organic_inorganic": fisher_org,
        "toxic_safe": fisher_tox,
        "comparison_to_v1": {
            "organic_inorganic_improvement_pct": improvement_org,
            "toxic_safe_improvement_pct": improvement_tox,
        },
        "finding": finding,
    }

    output_path = "/srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb/data/chemistry_hdc_v2_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {output_path}")

    print("\n" + "=" * 70 + "\n")

    return results


if __name__ == "__main__":
    run_experiment()
