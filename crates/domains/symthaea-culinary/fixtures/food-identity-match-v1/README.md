# Food identity match fixtures

The files and test-generated cases are synthetic contract fixtures. The example words and alias language tags demonstrate protocol shape only; no nutrient values or source food composition records are embedded.

- `exact-multilingual-potato.json` demonstrates a possible record-to-record identity assessment with multiple locale aliases and explicit structured identity fields.
- Rust unit/integration tests mutate that fixture to represent raw/cooked conflicts, taxon mismatch/unknown, nutrient denominator mismatch, incorrect source-identity declarations, approximate matching boundaries, duplicate checks and language-tag errors.

These fixtures do not establish actual synonym equivalence for any language, and must not be used as training truth for a food matcher. External ontology IDs, source records, and local synonyms require source rights and expert adjudication. Do not add restricted food-composition rows to the public fixture corpus.
