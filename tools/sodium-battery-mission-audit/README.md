# sodium-battery-mission-audit

Dependency-free Rust 1.96 reference auditor for mission-conditioned sodium-ion / sodium-metal comparison manifests.

Checks technology-lane/frontier separation, comparator permission, complete-cell mass boundaries, mission condition profiles, specimen/test/measurement references, evidence and rights admission, raw-data digest syntax, and observed/censored/competing lifetime semantics.

The commercial sodium-ion/LFP/NMC frontier is separate from zero-excess sodium-metal research. This prevents a high-risk post-lithium laboratory result from being averaged into a commercial-cell leaderboard.

Unit tests are authored but require the exact-head GitHub Actions workflow. The crate audits declarations only; it does not run an electrochemical model, train an estimator, validate physical test data, or qualify a battery.