# Food quantity conversion fixtures

All examples here are synthetic protocol fixtures. The illustrative edible-fraction and boiling-yield numbers are not food-composition data, cooking guidance, or empirical recommendations.

The file estimated-potato-yield.json demonstrates a three-step transformation:
1. purchased raw mass → edible raw portion;
2. edible raw portion → boiled output mass;
3. grams → kilograms using a pinned deterministic unit table.

The expected result is recomputed by the validator. Point estimate: 186.55 g; conservative interval from the illustrative factor bounds: 176.0–197.4 g. Those values only prove contract arithmetic. The source factors are synthetic and the fixture is explicitly calculation_only.

Do not add licensed local food-composition data to public fixtures without confirming redistribution rights. A structurally valid source locator is not proof that the source exists, is licensed, or applies to a specific food/process.
