#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OracleProjection {
    pub causal_status: &'static str,
    pub expected_transfer: &'static str,
    pub expected_action: &'static str,
    pub expected_completion: &'static str,
}

fn confounded_flow(common_cause: bool, _pump_on: bool, _valve_open: bool) -> i32 {
    // The common cause drives both observed controls and the outcome.
    // Valve state is deliberately absent from the structural equation.
    if common_cause { 12 } else { 0 }
}

fn controlled_flow(pump_on: bool, valve_open: bool) -> i32 {
    if !pump_on {
        0
    } else if valve_open {
        12
    } else {
        2
    }
}

fn old_thermal_model(heater_power_kw: i32, cooling_on: bool) -> i32 {
    2 * heater_power_kw - if cooling_on { 4 } else { 0 }
}

pub fn projection_for_family(family: &str) -> Result<OracleProjection, String> {
    match family {
        "observational_confounding" => {
            // Two observations: the common cause changes, and pump/valve co-move.
            // There is no observed intervention that separates valve from pump.
            let observations = [
                (false, false, false, 0),
                (true, true, true, 12),
            ];
            let common_cause_changed = observations[0].0 != observations[1].0;
            let pump_changed = observations[0].1 != observations[1].1;
            let valve_changed = observations[0].2 != observations[1].2;
            let flow_changed = observations[0].3 != observations[1].3;
            let intervention_separates_valve = false;

            // Counterfactual SCM check: holding the common cause and pump fixed,
            // intervening on the valve has no effect because valve is not a parent
            // of the structural flow equation.
            let do_closed = confounded_flow(true, true, false);
            let do_open = confounded_flow(true, true, true);
            let valve_effect = do_open - do_closed;

            if common_cause_changed
                && pump_changed
                && valve_changed
                && flow_changed
                && !intervention_separates_valve
                && valve_effect == 0
            {
                Ok(OracleProjection {
                    causal_status: "not_identified",
                    expected_transfer: "record_association_only",
                    expected_action: "seek_intervention_or_preserve_uncertainty",
                    expected_completion: "not_applicable",
                })
            } else {
                Err("observational-confounding SCM fixture no longer satisfies its contract".to_owned())
            }
        }
        "controlled_intervention" => {
            // Four deterministic replicates per arm under the same pump-on regime.
            // The fixture is explicitly a noiseless simulator contract, not an
            // estimate of a real-world population effect.
            let control = [controlled_flow(true, false); 4];
            let treatment = [controlled_flow(true, true); 4];
            let control_sum: i32 = control.iter().sum();
            let treatment_sum: i32 = treatment.iter().sum();
            let effect = treatment_sum / treatment.len() as i32
                - control_sum / control.len() as i32;
            let all_pairs_consistent = control.iter().all(|x| *x == 2)
                && treatment.iter().all(|x| *x == 12);

            if effect > 0 && all_pairs_consistent {
                Ok(OracleProjection {
                    causal_status: "identified",
                    expected_transfer: "reuse_qualified_transition",
                    expected_action: "reuse_valve_effect_with_preconditions",
                    expected_completion: "not_applicable",
                })
            } else {
                Err("controlled-intervention SCM fixture failed its deterministic effect check".to_owned())
            }
        }
        "regime_shift" => {
            let old_observation_low = old_thermal_model(2, false);
            let old_observation_high = old_thermal_model(3, false);
            let old_effect_per_kw = old_observation_high - old_observation_low;
            let old_prediction_at_current_power = old_thermal_model(3, false);
            let current_regime_observation = old_thermal_model(3, true);

            if old_effect_per_kw == 2
                && old_prediction_at_current_power != current_regime_observation
            {
                Ok(OracleProjection {
                    causal_status: "regime_shifted",
                    expected_transfer: "requalify_before_reuse",
                    expected_action: "request_new_measurement_or_abstain",
                    expected_completion: "not_applicable",
                })
            } else {
                Err("thermal regime-shift SCM fixture no longer changes the prediction".to_owned())
            }
        }
        "outcome_misattribution" => {
            let tool_accepted = true;
            let requested_relay_state = true;
            let observed_relay_state = false;

            if tool_accepted && requested_relay_state != observed_relay_state {
                Ok(OracleProjection {
                    causal_status: "outcome_unverified",
                    expected_transfer: "mark_outcome_unknown",
                    expected_action: "reconcile_external_state_before_reuse_or_retry",
                    expected_completion: "unknown_pending_reconciliation",
                })
            } else {
                Err("relay outcome-misattribution fixture no longer contradicts the tool response".to_owned())
            }
        }
        other => Err(format!("unknown MACE SCM family {other:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_cause_creates_observational_correlation_without_valve_effect() {
        let low = confounded_flow(false, false, false);
        let observed_high = confounded_flow(true, true, true);
        let do_closed = confounded_flow(true, true, false);
        let do_open = confounded_flow(true, true, true);
        assert_eq!(low, 0);
        assert_eq!(observed_high, 12);
        assert_eq!(do_open - do_closed, 0);
        assert_eq!(
            projection_for_family("observational_confounding").unwrap().causal_status,
            "not_identified"
        );
    }

    #[test]
    fn controlled_pump_fixed_interventions_have_positive_effect() {
        assert_eq!(controlled_flow(true, true) - controlled_flow(true, false), 10);
        assert_eq!(
            projection_for_family("controlled_intervention").unwrap().expected_transfer,
            "reuse_qualified_transition"
        );
    }

    #[test]
    fn old_thermal_model_is_not_applicable_after_cooling_regime_change() {
        assert_eq!(old_thermal_model(3, false), 6);
        assert_eq!(old_thermal_model(3, true), 2);
        assert_eq!(
            projection_for_family("regime_shift").unwrap().expected_transfer,
            "requalify_before_reuse"
        );
    }

    #[test]
    fn accepted_relay_write_with_off_observation_is_not_completion() {
        assert_eq!(
            projection_for_family("outcome_misattribution").unwrap().expected_completion,
            "unknown_pending_reconciliation"
        );
    }
}
