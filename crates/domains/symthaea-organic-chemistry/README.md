# symthaea-organic-chemistry

A self-contained **structural organic-chemistry** layer for Symthaea: parse
SMILES into a molecular graph, then derive formula, weight, structure, and
cheminformatic properties. Complements the ab-initio electronic-structure crate
`symthaea-quantum-chemistry` (Hartree-Fock, DFT), which had no structural layer.

Pure `std`, zero dependencies, no `symthaea-core` link.

## Capabilities

| Area | API |
|------|-----|
| SMILES parsing | `Molecule::from_smiles` |
| Formula / weight | `molecular_formula` (Hill), `molecular_weight`, `mass_composition` |
| Structure | `groups::detect`, `ring_count`, `degree_of_unsaturation` |
| Cheminformatics | `hbond_donors`, `hbond_acceptors`, `lipinski` (Rule of Five) |

SMILES supports the organic subset + bracket atoms, single/double/triple/
aromatic bonds, branches, and ring closures. Unsupported features
(stereochemistry, isotopes, disconnected structures) **error** rather than
mis-parse.

## Example

```rust
use symthaea_organic_chemistry::{Molecule, lipinski};

let aspirin = Molecule::from_smiles("CC(=O)Oc1ccccc1C(=O)O").unwrap();
assert_eq!(aspirin.molecular_formula(), "C9H8O4");
assert!((aspirin.molecular_weight() - 180.16).abs() < 0.1);
assert!(lipinski(&aspirin).drug_like);
```

## Validation

All derived quantities are unit-tested against known values (ethanol, benzene,
CO₂, acetic acid, aspirin, caffeine, …). Run:

```bash
cargo test -p symthaea-organic-chemistry
```

## Not yet

logP, stereochemistry, reaction mechanisms / retrosynthesis.
