# Phase A.3 USPTO Template-Coverage Evaluation -- First Frozen Run

total_candidates=1282 elapsed=41.66ms

## Overall (n=1282)

- certified_exact_transformation: 187 (14.6%)
- structurally_shaped_wrong_transformation: 134 (10.5%)
- unsupported_reaction_context: 186 (14.5%)
- validity_or_conservation_failure: 255 (19.9%)
- policy_or_hazard_rejection: 10 (0.8%)
- representation_or_normalization_failure: 489 (38.1%)
- resource_bounded_uncertainty: 21 (1.6%)
- ambiguous_reactive_site (cross-cutting flag, any category): 77 (6.0%)

## esterification (n=558)

- certified_exact_transformation: 46 (8.2%)
- structurally_shaped_wrong_transformation: 15 (2.7%)
- unsupported_reaction_context: 186 (33.3%)
- validity_or_conservation_failure: 75 (13.4%)
- policy_or_hazard_rejection: 8 (1.4%)
- representation_or_normalization_failure: 218 (39.1%)
- resource_bounded_uncertainty: 10 (1.8%)
- ambiguous_reactive_site (cross-cutting flag, any category): 52 (9.3%)

## hydrogenation (n=724)

- certified_exact_transformation: 141 (19.5%)
- structurally_shaped_wrong_transformation: 119 (16.4%)
- unsupported_reaction_context: 0 (0.0%)
- validity_or_conservation_failure: 180 (24.9%)
- policy_or_hazard_rejection: 2 (0.3%)
- representation_or_normalization_failure: 271 (37.4%)
- resource_bounded_uncertainty: 11 (1.5%)
- ambiguous_reactive_site (cross-cutting flag, any category): 25 (3.5%)

## split=dev (n=541)

- certified_exact_transformation: 78 (14.4%)
- structurally_shaped_wrong_transformation: 55 (10.2%)
- unsupported_reaction_context: 70 (12.9%)
- validity_or_conservation_failure: 106 (19.6%)
- policy_or_hazard_rejection: 2 (0.4%)
- representation_or_normalization_failure: 217 (40.1%)
- resource_bounded_uncertainty: 13 (2.4%)
- ambiguous_reactive_site (cross-cutting flag, any category): 29 (5.4%)

## split=validation (n=456)

- certified_exact_transformation: 79 (17.3%)
- structurally_shaped_wrong_transformation: 42 (9.2%)
- unsupported_reaction_context: 88 (19.3%)
- validity_or_conservation_failure: 79 (17.3%)
- policy_or_hazard_rejection: 5 (1.1%)
- representation_or_normalization_failure: 158 (34.6%)
- resource_bounded_uncertainty: 5 (1.1%)
- ambiguous_reactive_site (cross-cutting flag, any category): 30 (6.6%)

## split=holdout (n=285)

- certified_exact_transformation: 30 (10.5%)
- structurally_shaped_wrong_transformation: 37 (13.0%)
- unsupported_reaction_context: 28 (9.8%)
- validity_or_conservation_failure: 70 (24.6%)
- policy_or_hazard_rejection: 3 (1.1%)
- representation_or_normalization_failure: 114 (40.0%)
- resource_bounded_uncertainty: 3 (1.1%)
- ambiguous_reactive_site (cross-cutting flag, any category): 18 (6.3%)

## Isomorphism module diagnostics (cumulative, process-global)

comparisons_attempted=607 atom_limit_rejections=21 budget_exhaustions=0 worst_steps_used=1305 worst_depth_reached=40
