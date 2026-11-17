# Symthaea Meal Proposal Handoff V1

Status: draft typed contract and synthetic validation cases. No physical meal, inventory or appliance observation is implied.

This package defines a proposal boundary between Symthaea's culinary reasoning and Symtropy's recipe/task compiler.

## Ownership

- Symthaea proposes a user-reviewable meal, exact ingredient quantities where known, culinary task descriptions, user constraints, source-scoped facts and explicit uncertainty.
- Unknown quantities remain unknown and require a reason. A product name, catalogue entry, supplier listing or generic ingredient does not establish that a household lot exists.
- Symtropy grounds natural-language tasks in its action ontology, resolves recipe graph constraints, checks exact module revisions/capabilities, and schedules against resources and power.
- Mycelix owns producer/product/lot/receipt/packaging provenance. Household-local receipt and inventory are not inferred from supplier listings.
- No proposal field selects hardware, resource IDs, appliance settings, or device commands. Human review does not by itself grant appliance actuation authority.

## Quantity and evidence semantics

Quantities are integer coefficients plus decimal scale with explicit mass, volume or count dimensions. Incompatible dimension/unit pairs are invalid. No density-based mass inference is performed.

Each fact carries a claim kind, scope, value, unit/basis, source record/version, evidence class, observation time, optional validity end, market and license references. Runtime freshness is evaluated against an explicit as-of time by the consuming system; schema validation alone does not make a price or availability fact current.

## Validation

- duplicate ingredient/task/fact IDs and unresolved references are rejected;
- price claims require a market, currency/unit and normalization basis; availability requires a market and validity end; allergen declarations require product-SKU or supplier-lot scope;
- task evidence links are fact IDs, not arbitrary unversioned labels;
- dependency cycles and invalid duration windows are rejected;
- unknown quantities need a matching uncertainty record;
- exact hardware assignment is intentionally absent;
- proposal review state is a human workflow marker, not an execution permission.

Run: cargo test -p symthaea-culinary

## Claim ceiling

The proposal establishes only a typed, traceable suggestion. It is not verified household stock, current availability, nutrition/health advice, food-safety approval, equipment compatibility, a complete task graph, schedule or actuation authority.