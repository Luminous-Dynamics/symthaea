# CETB theory design reference utility

Pure-standard-library reference calculations for categorical expected information gain (EIG) over theory prediction cards. This is an isolated research tool and is not production Symthaea code.

## Run

\`\`\`sh
python3 -m unittest discover -s research/cetb/theory_design -p 'test_*.py' -v
\`\`\`

\`eig.py\` requires normalized prior probabilities over theories and normalized outcome likelihoods for every theory. Each theory must use the same explicit outcome space, including missing, ambiguous and failed outcomes.

It computes mutual information \(I(T;Y\mid d)\) in bits, bounded by prior entropy. It also aggregates EIG over predeclared nuisance scenarios, reports a weighted lower quantile, and reports cost separately and EIG-per-cost. The lower quantile is a robustness summary, not a worst-case guarantee.

The utility does not author theory predictions, establish calibration, choose priors or prove any theory. It is only as trustworthy as the frozen prediction distributions and nuisance assumptions supplied by independent theory owners/evaluators. The accompanying tests use synthetic probability tables only.


## Versioned design manifest

The input contract is defined in `design-run.schema.json`. The synthetic example in `fixtures/synthetic-design-run.json` demonstrates two candidate designs, two nuisance scenarios for one candidate, and an explicit `measurement_unusable` outcome.

Run the ranking:

```sh
python research/cetb/theory_design/eig.py \
  research/cetb/theory_design/fixtures/synthetic-design-run.json \
  --output eig-result.json
```

The ranking sorts by lower-quantile EIG per cost, then expected EIG per cost. The returned result preserves expected EIG, lower-quantile EIG, scenario count and cost separately. A ranking is only as sound as the frozen prior and theory likelihoods; the synthetic fixture is not a human prediction.
