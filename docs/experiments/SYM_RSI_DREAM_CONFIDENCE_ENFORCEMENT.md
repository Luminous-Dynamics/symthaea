# Dream-confidence enforcement requirement

The current `DreamFeedbackBridge` behavior that boosts confidence merely because a dream-derived prior exists must not be used by SYM-RSI experiments as an epistemic confidence update.

Required behavioral replacement:

- dream priors may influence **action proposal/ranking**;
- dream priors may reduce confidence in contexts where generated failure modes motivate caution;
- dream priors may not increase epistemic confidence until an independent recorded/replay validation event marks the associated hypothesis as empirically validated;
- calibration statistics must distinguish dream-informed proposals from empirically validated confidence updates.

This document records the required behavior while the existing bridge is being patched. Until the code and tests enforce this rule, conditions B and D in SYM-RSI-001 are blocked from measured execution.