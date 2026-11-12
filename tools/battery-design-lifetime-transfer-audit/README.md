# battery-design-lifetime-transfer-audit

Dependency-free Rust 1.96 reference audit for early-cycle lifetime prediction on held-out battery design combinations.

Checks that train/validation/test partitions are physical-cell disjoint; leave-one-design-group-out folds keep exactly one material/design combination out of model development; protocol-family holdouts bind and isolate exact protocol groups; feature evidence is available at the declared cutoff; predictions are unique and occur before target events; and event-observed, right-censored, competing-event and unknown outcomes remain distinct.

The point evaluator scores observed end-of-life events only, reporting pooled-cell and design-group-macro MAE/RMSE/relative error plus interval coverage. It reports censored/competing/unknown counts but does not implement survival scoring.

Fourteen unit tests are authored, including explicit protocol-group leakage and exact one-group holdout controls; they require exact-head CI execution. This is a protocol audit, not a trained predictor or physical lifetime result.