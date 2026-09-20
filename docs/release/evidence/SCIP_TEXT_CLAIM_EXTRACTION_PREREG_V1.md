# SCIP Text-to-Claim Extraction — V18 Preregistration

V18 freezes the experiment required between V17 structured-claim comparison and any future natural-language semantic-fidelity capability.

```text
surface text
-> frozen extractor
-> frozen candidate claim inventory
-> exact canonical match to hidden human inventory
-> reveal hidden source alignment only after freeze
-> V17 comparison
```

Authority is **preregistration-only**. Confirmatory execution, surface fidelity, factual truth, backend identity, authentication, and action authority remain unestablished.

The candidate sees surface text plus the public claim schema only. The grounded source graph, source claims/IDs, human inventory, and V17 result remain hidden during extraction. Candidate output cannot contain source claim IDs. No human/model semantic aligner may repair confirmatory output after ground truth is revealed.

Corpus floor: 528 calibration cases and 832 sealed confirmatory cases, covering all eleven V16 dimensions with balanced positive/negative single-factor cases plus cross-sentence discourse cases. Every case receives two independent annotations and adjudication on disagreement.

All confirmatory gates are conjunctive. Per dimension: positive preservation >= 0.95, negative sensitivity >= 0.95, exact claim precision >= 0.95, exact claim recall >= 0.95, expected V17 verdict accuracy >= 0.95, and indeterminate rate <= 0.05. Unsupported-addition false-positive and required-detail omission rates must each be <= 0.02. Targeted negatives permit zero catastrophic misses for negation, numeric value/unit, attribution/source, and causal strength.

The independent standard-library validator and hostile-input harness pass locally under Python 3.13.5. Frozen semantic preregistration identity:

```text
96e2ec5e1fad213f4261405c22d1f80cb6f1d106fd5621481eb0e95a6cca4530
```

Exact executable/input SHA-256 values:

```text
validator   e9f752911c8a4cec706a3a0b04f7b4143726d6824ea7be70cb4c56a547d0bf6f
harness     ede320c9f07f135033b204b864e6b4456bef58e897aa920a6d4d1240d588106b
prereg JSON 9de306be32e16892beb93a44db81459eca3d1f058b8143b7be56699904ecd084
```
