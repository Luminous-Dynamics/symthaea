"""
Phase A.3 probe: how many USPTO-50K records are shaped like this crate's
two supported templates (Fischer esterification, C-C hydrogenation)?

TDC's RetroSyn framing: 'input' = product molecule, 'output' = reactant
set (dot-separated) -- the reverse of a normal reactants->product record.
"""
import pandas as pd
from rdkit import Chem

CARBOXYLIC_ACID = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
ESTER = Chem.MolFromSmarts("[CX3](=O)[OX2H0][#6]")
# Aliphatic/aromatic -OH that is NOT part of a carboxylic acid.
ALCOHOL_OR_PHENOL_OH = Chem.MolFromSmarts("[#6][OX2H1]")
CC_UNSATURATION = Chem.MolFromSmarts("[#6]=[#6]")
CC_TRIPLE = Chem.MolFromSmarts("[#6]#[#6]")


def parse(smiles):
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def is_carboxylic_acid(mol):
    return mol is not None and mol.HasSubstructMatch(CARBOXYLIC_ACID)


def is_alcohol_not_acid(mol):
    if mol is None:
        return False
    if mol.HasSubstructMatch(CARBOXYLIC_ACID):
        return False
    return mol.HasSubstructMatch(ALCOHOL_OR_PHENOL_OH)


def unsaturation_count(mol):
    if mol is None:
        return 0
    return len(mol.GetSubstructMatches(CC_UNSATURATION)) + len(
        mol.GetSubstructMatches(CC_TRIPLE)
    )


def esterification_shaped(reactant_mols, product_mol):
    if len(reactant_mols) != 2:
        return False
    a, b = reactant_mols
    has_acid_and_alcohol = (is_carboxylic_acid(a) and is_alcohol_not_acid(b)) or (
        is_carboxylic_acid(b) and is_alcohol_not_acid(a)
    )
    if not has_acid_and_alcohol:
        return False
    return product_mol is not None and product_mol.HasSubstructMatch(ESTER)


def hydrogenation_shaped(reactant_mols, product_mol):
    if len(reactant_mols) != 1:
        return False
    r = reactant_mols[0]
    if r is None or product_mol is None:
        return False
    return unsaturation_count(r) > unsaturation_count(product_mol)


def main():
    df = pd.read_csv("uspto50k.csv")
    ester_hits = []
    hydro_hits = []
    parse_failures = 0

    for idx, row in df.iterrows():
        product_smiles = row["input"]
        reactant_smiles_list = row["output"].split(".")
        product_mol = parse(product_smiles)
        reactant_mols = [parse(s) for s in reactant_smiles_list]
        if product_mol is None or any(m is None for m in reactant_mols):
            parse_failures += 1
            continue
        if esterification_shaped(reactant_mols, product_mol):
            ester_hits.append((row["output"], row["input"]))
        elif hydrogenation_shaped(reactant_mols, product_mol):
            hydro_hits.append((row["output"], row["input"]))

    print(f"total records: {len(df)}")
    print(f"parse failures: {parse_failures}")
    print(f"esterification-shaped candidates: {len(ester_hits)}")
    print(f"hydrogenation-shaped candidates: {len(hydro_hits)}")
    print()
    print("=== sample esterification-shaped (up to 10) ===")
    for reactants, product in ester_hits[:10]:
        print(f"  {reactants} >> {product}")
    print()
    print("=== sample hydrogenation-shaped (up to 10) ===")
    for reactants, product in hydro_hits[:10]:
        print(f"  {reactants} >> {product}")

    with open("ester_candidates.tsv", "w") as f:
        f.write("reactants\tproduct\n")
        for reactants, product in ester_hits:
            f.write(f"{reactants}\t{product}\n")
    with open("hydro_candidates.tsv", "w") as f:
        f.write("reactants\tproduct\n")
        for reactants, product in hydro_hits:
            f.write(f"{reactants}\t{product}\n")


if __name__ == "__main__":
    main()
