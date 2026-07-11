# Data provenance

These three files are the **unmodified** flavor-network dataset published with:

> Ahn, Y.-Y., Ahnert, S. E., Bagrow, J. P. & Barabási, A.-L.
> "Flavor network and the principles of food pairing."
> *Scientific Reports* **1**, 196 (2011). https://doi.org/10.1038/srep00196

Retrieved 2026-07-09 from the public mirror
`github.com/lingcheng99/Flavor-Network/tree/master/data`.

| File | Original name | Contents |
|------|---------------|----------|
| `ingr_info.tsv` | `ingr_info.tsv` | `id \t ingredient_name \t category` (1530 ingredients) |
| `ingr_comp.tsv` | `ingr_comp.tsv` | `ingredient_id \t compound_id` (36,781 bipartite edges) |
| `recipes.csv` | `srep00196-s3.csv` | `Cuisine,ingredient,ingredient,…` (56,498 recipes, 11 cuisines) |

Files are verbatim (including the leading `#` comment/header lines) so provenance is
auditable. The parsers in `src/data.rs` skip `#`-prefixed lines.

**Why full raw data, not a curated seed:** the crate reproduces Ahn 2011's headline
result (`tests/ahn_2011.rs`) as a genuine falsifiable ground-truth check, not an
approximation. That requires the real ingredient→compound graph and the real
cuisine-labeled recipes.
