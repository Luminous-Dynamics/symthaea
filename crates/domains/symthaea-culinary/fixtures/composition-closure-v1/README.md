# Proximate composition closure fixture

All fixture values are synthetic and are not food-composition data or dietary guidance.

The profile contains a complete seven-component proximate set under the available-carbohydrate-plus-separately-reported-fibre convention. The point sum is 101 g per 100 g; propagated component intervals span 99.6–102.4 g per 100 g. This is a contract fixture for the diagnostic only.

Every component records a source locator, source version, known-at timestamp and evidence status. All evidence is synthetic and must not be promoted to qualified.

The alternative total-carbohydrate-includes-fibre convention uses total_carbohydrate and omits the separate dietary_fibre row from the closure sum. Mixing these conventions would double count fibre and is rejected.


Unavailable-component declarations also carry source/version/time/evidence metadata: a source can document that a component was not analysed or not reported, rather than relying on an untraceable omission. The absence of a measured value remains distinct from a zero.
