# Synthetic global regional-pack fixtures

These packs are **protocol test fixtures only**. They do not assert actual food composition, country dietary guidance, prices, regulatory coverage, cultural preferences, market availability, or real-world measurement conventions. Each fixture has an empty source registry and marks all ten verticals `not_configured`.

- `synthetic-za.json` — identifier/locale/currency wiring example.
- `synthetic-in.json` — identifier/locale/currency wiring example.
- `synthetic-br.json` — identifier/locale/currency wiring example.

The examples intentionally avoid real region data so schema and fail-closed behavior can be tested without assuming access rights or smuggling unverified values into the public codebase. Do not change their source/vertical statuses to `supported` unless source references, licensing, coverage and domain review are supplied.
