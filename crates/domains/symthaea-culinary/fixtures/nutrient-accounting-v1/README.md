# Nutrient accounting fixtures

All fixture values are synthetic and are not food composition data or dietary guidance.

`synthetic-rice-pea-recipe.json` contains two nutrient-bearing ingredients (rice and peas), per-100-g edible ingredient protein values, ingredient-specific protein-retention intervals pinned to the recipe process and each ingredient's input state/output food state, plus an independently stated final recipe mass interval. Water used as the cooking medium is intentionally not treated as a nutrient-bearing ingredient in this fixture.

The validator must:
- compute each retained nutrient amount as (nutrient per 100 g edible ingredient × ingredient edible mass / 100 × nutrient retention factor);
- sum retained amounts across ingredients;
- divide the total nutrient amount by the final measured recipe mass to express it per 100 g finished recipe;
- propagate lower/upper bounds conservatively and reject an output that does not match recomputation.

Weight yield is not multiplied into nutrient amount a second time. Missing composition is not interpreted as zero; a supported zero needs an explicit zero-valued composition record. All synthetic evidence remains `calculation_only`.
