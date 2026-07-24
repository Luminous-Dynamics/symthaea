"""
Phase A.3: scaffold-grouped dev/validation/holdout split of the
esterification/hydrogenation candidate sets, so near-duplicate patent
reactions (same Bemis-Murcko scaffold) can't leak across splits.

Deterministic: split assignment is a hash of the scaffold SMILES, not a
random draw, so this is fully reproducible from the frozen inputs.
"""
import csv
import hashlib

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def scaffold_key(product_smiles):
    mol = Chem.MolFromSmiles(product_smiles)
    if mol is None:
        return "UNPARSEABLE"
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold) if scaffold is not None else ""
    except Exception:
        scaffold_smiles = ""
    if not scaffold_smiles:
        # No ring scaffold (fully acyclic product) -- fall back to the
        # product's own canonical SMILES as the grouping key so acyclic
        # products still group with their exact duplicates, at minimum.
        scaffold_smiles = Chem.MolToSmiles(mol)
    return scaffold_smiles


def split_for_scaffold(scaffold, dev_frac=0.5, val_frac=0.25):
    # Deterministic hash-based bucketing -- same scaffold always lands in
    # the same split, no RNG/seed to manage.
    h = int(hashlib.sha256(scaffold.encode()).hexdigest(), 16)
    frac = (h % 10_000) / 10_000.0
    if frac < dev_frac:
        return "dev"
    elif frac < dev_frac + val_frac:
        return "validation"
    else:
        return "holdout"


def process(name):
    rows = []
    with open(f"{name}_candidates.tsv") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)

    scaffold_to_split = {}
    out_rows = []
    for row in rows:
        scaffold = scaffold_key(row["product"])
        if scaffold not in scaffold_to_split:
            scaffold_to_split[scaffold] = split_for_scaffold(scaffold)
        split = scaffold_to_split[scaffold]
        out_rows.append({**row, "scaffold": scaffold, "split": split})

    counts = {"dev": 0, "validation": 0, "holdout": 0}
    for r in out_rows:
        counts[r["split"]] += 1
    print(f"{name}: {len(out_rows)} rows, {len(scaffold_to_split)} distinct scaffolds")
    print(f"  split counts: {counts}")

    with open(f"{name}_candidates_split.tsv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["reactants", "product", "scaffold", "split"], delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(out_rows)


if __name__ == "__main__":
    process("ester")
    process("hydro")
