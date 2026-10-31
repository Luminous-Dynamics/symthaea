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

fn extract_top_level_json_value<'a>(json: &'a str, key: &str) -> Result<&'a str, String> {
    let marker = format!("\"{key}\":");
    let marker_start = json.find(&marker)
        .ok_or_else(|| format!("agent payload is missing JSON property {key:?}"))?;
    let start = marker_start + marker.len();
    let bytes = json.as_bytes();
    let first = *bytes.get(start).ok_or_else(|| format!("property {key:?} has no value"))?;

    if first == b'{' || first == b'[' {
        let mut depth = 0usize;
        let mut in_string = false;
        let mut escaped = false;
        for index in start..bytes.len() {
            let byte = bytes[index];
            if in_string {
                if escaped {
                    escaped = false;
                } else if byte == b'\\' {
                    escaped = true;
                } else if byte == b'"' {
                    in_string = false;
                }
                continue;
            }
            match byte {
                b'"' => in_string = true,
                b'{' | b'[' => depth += 1,
                b'}' | b']' => {
                    depth = depth.checked_sub(1)
                        .ok_or_else(|| format!("unbalanced JSON value for {key:?}"))?;
                    if depth == 0 {
                        return Ok(&json[start..=index]);
                    }
                }
                _ => {}
            }
        }
        Err(format!("unterminated JSON object/array for {key:?}"))
    } else if first == b'"' {
        let mut escaped = false;
        for index in start + 1..bytes.len() {
            let byte = bytes[index];
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                return Ok(&json[start..=index]);
            }
        }
        Err(format!("unterminated JSON string for {key:?}"))
    } else {
        let end = json[start..]
            .find(|character| matches!(character, ',' | '}' | ']'))
            .map(|offset| start + offset)
            .unwrap_or(json.len());
        if end == start {
            Err(format!("empty JSON value for {key:?}"))
        } else {
            Ok(&json[start..end])
        }
    }
}

pub fn validate_agent_payload(family: &str, payload: &str) -> Result<(), String> {
    let required: &[&str] = match family {
        "observational_confounding" => &[
            "\"event_id\":\"E1\"",
            "\"event_id\":\"E2\"",
            "\"pump\":\"off\"",
            "\"pump\":\"on\"",
            "\"valve\":\"closed\"",
            "\"valve\":\"open\"",
        ],
        "controlled_intervention" => &[
            "\"event_id\":\"E4\"",
            "\"event_id\":\"E5\"",
            "\"event_id\":\"E14\"",
            "\"event_id\":\"E15\"",
            "\"event_id\":\"E16\"",
            "\"event_id\":\"E17\"",
            "\"event_id\":\"E18\"",
            "\"event_id\":\"E19\"",
            "\"assignment\":\"control\"",
            "\"assignment\":\"treatment\"",
            "\"flow_lpm\":2",
            "\"flow_lpm\":12",
            "\"pump\":\"on\"",
        ],
        "regime_shift" => &[
            "\"event_id\":\"E7\"",
            "\"event_id\":\"E8\"",
            "\"old_regime\":{\"cooling\":\"off\"}",
            "\"current_regime\":{\"cooling\":\"on\"}",
        ],
        "outcome_misattribution" => &[
            "\"event_id\":\"E10\"",
            "\"event_id\":\"E11\"",
            "\"event_id\":\"E12\"",
            "\"status\":\"accepted\"",
            "\"relay_state\":\"off\"",
        ],
        other => return Err(format!("unknown MACE agent-input family {other:?}")),
    };

    for token in required {
        if !payload.contains(token) {
            return Err(format!("agent payload for family {family:?} is missing evidence token {token:?}"));
        }
    }
    Ok(())
}

pub fn validate_irrelevant_pair(
    family: &str,
    anchor_payload: &str,
    control_payload: &str,
) -> Result<(), String> {
    for property in ["current_state", "objective", "current_task", "resource_budget"] {
        let anchor_value = extract_top_level_json_value(anchor_payload, property)?;
        let control_value = extract_top_level_json_value(control_payload, property)?;
        if anchor_value != control_value {
            return Err(format!("irrelevant control changed shared property {property:?}"));
        }
    }

    let (context_key, context_value, event_json) = match family {
        "observational_confounding" => (
            "display_theme",
            "blue",
            r#"{"event_id":"E3","session_id":"S2","event_kind":"irrelevant_metadata","payload":{"display_theme":"blue"}}"#,
        ),
        "controlled_intervention" => (
            "operator_note",
            "used standard checklist",
            r#"{"event_id":"E6","session_id":"S4","event_kind":"irrelevant_metadata","payload":{"operator_note":"used standard checklist"}}"#,
        ),
        "regime_shift" => (
            "wall_label",
            "north",
            r#"{"event_id":"E9","session_id":"S6","event_kind":"irrelevant_metadata","payload":{"wall_label":"north"}}"#,
        ),
        "outcome_misattribution" => (
            "dashboard_color",
            "amber",
            r#"{"event_id":"E13","session_id":"S8","event_kind":"irrelevant_metadata","payload":{"dashboard_color":"amber"}}"#,
        ),
        other => return Err(format!("unknown MACE pair family {other:?}")),
    };

    let anchor_context = extract_top_level_json_value(anchor_payload, "scenario_context")?;
    let control_context = extract_top_level_json_value(control_payload, "scenario_context")?;
    let anchor_context_without_close = anchor_context.strip_suffix('}')
        .ok_or_else(|| "anchor scenario_context is not an object".to_owned())?;
    let mut expected_context = anchor_context_without_close.to_owned();
    expected_context.push_str(&format!(
        ",\"{context_key}\":{}",
        json_string(context_value)
    ));
    expected_context.push('}');
    if control_context != expected_context {
        return Err(format!("pair {family:?} control context differs beyond {context_key:?}"));
    }

    let anchor_history = extract_top_level_json_value(anchor_payload, "historical_events")?;
    let control_history = extract_top_level_json_value(control_payload, "historical_events")?;
    let anchor_history_without_close = anchor_history.strip_suffix(']')
        .ok_or_else(|| "anchor historical_events is not an array".to_owned())?;
    let expected_history = format!("{anchor_history_without_close},{event_json}]");
    if control_history != expected_history {
        return Err(format!("pair {family:?} control history must append exactly its declared irrelevant event"));
    }
    Ok(())
}

fn json_string(value: &str) -> String {
    let mut encoded = String::from("\"");
    for character in value.chars() {
        match character {
            '"' => encoded.push_str("\\\""),
            '\\' => encoded.push_str("\\\\"),
            '\n' => encoded.push_str("\\n"),
            '\r' => encoded.push_str("\\r"),
            '\t' => encoded.push_str("\\t"),
            other => encoded.push(other),
        }
    }
    encoded.push('"');
    encoded
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

    #[test]
    fn unknown_family_is_rejected() {
        assert!(projection_for_family("invented_family").is_err());
        assert!(validate_agent_payload("invented_family", "{}").is_err());
    }
}
