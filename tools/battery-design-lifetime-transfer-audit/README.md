# battery-design-lifetime-transfer-audit

Dependency-free Rust 1.96 reference audit for early-cycle lifetime prediction on held-out battery design combinations.

Checks that train/validation/test partitions are physical-cell disjoint; leave-one-design-group-out folds keep entire material/design combinations out of model development; feature evidence is available at the declared cutoff; predictions are unique and occur before target events; and event-observed, right-censored, competing-event and unknown outcomes remain distinct.

The point evaluator scores observed end-of-life events only, reporting pooled-cell and design-group-macro MAE/RMSE/relative error plus interval coverage. It reports censored/competing/unknown counts but does not implement survival scoring.

Unit tests are authored but require exact-head CI execution. This is a protocol audit, not a trained predictor or physical lifetime result.