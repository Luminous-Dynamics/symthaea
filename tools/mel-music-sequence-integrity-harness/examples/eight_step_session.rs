use mel_music_sequence_integrity_harness::{
    evaluate_sequence_v1, symbolic_state_sha256_v1, MusicalEventV1,
    MusicEditOperationKindV1, SequenceRequestV1, SequenceStepObservationV1,
    StepGateVerdictV1, StepHardGatesV1,
};

fn h(value: u64) -> String {
    format!("{value:064x}")
}

fn event(id: usize, pitch: i32) -> MusicalEventV1 {
    MusicalEventV1 {
        event_id: format!("event-{id:03}"),
        part_id: "piano".to_owned(),
        onset_ticks: (id as u64) * 480,
        duration_ticks: 480,
        pitch_millist: pitch,
        velocity: 88,
        program: 0,
    }
}

fn all_verified() -> StepHardGatesV1 {
    StepHardGatesV1 {
        target_edit: StepGateVerdictV1::Verified,
        protected_preservation: StepGateVerdictV1::Verified,
        rights_admission: StepGateVerdictV1::Verified,
        integrity: StepGateVerdictV1::Verified,
        source_lineage: StepGateVerdictV1::Verified,
        realization_evidence: StepGateVerdictV1::Verified,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let initial: Vec<_> = (0..8).map(|index| event(index, 60_000 + index as i32 * 1_000)).collect();
    let initial_state = symbolic_state_sha256_v1(&initial)?;
    let request = SequenceRequestV1 {
        schema_version: "mel-music-edit-sequence-request-v1".to_owned(),
        session_id: "synthetic-eight-step-preservation-chain".to_owned(),
        source_lineage_sha256: h(1),
        initial_artifact_sha256: h(2),
        initial_state_sha256: initial_state.clone(),
        operation_plan_sha256: h(3),
        expected_step_count: 8,
    };

    let mut previous_events = initial.clone();
    let mut previous_state = initial_state;
    let mut previous_artifact = h(2);
    let mut steps = Vec::new();

    for index in 0..8_usize {
        let mut output = previous_events.clone();
        output[index] = event(index, 61_000 + index as i32 * 1_000);
        let output_state = symbolic_state_sha256_v1(&output)?;
        let step_number = index as u8 + 1;
        let next_artifact = h(100 + u64::from(step_number));

        steps.push(SequenceStepObservationV1 {
            schema_version: "mel-music-edit-sequence-step-v1".to_owned(),
            sequence: step_number,
            step_id: format!("step-{step_number}"),
            operation_id: format!("operation-{step_number}"),
            operation_kind: MusicEditOperationKindV1::Replace,
            history_action: mel_music_sequence_integrity_harness::SequenceHistoryActionV1::Apply,
            undo_target_step_id: None,
            redo_target_step_id: None,
            instruction_sha256: h(10 + u64::from(step_number)),
            source_lineage_sha256: h(1),
            input_artifact_sha256: previous_artifact.clone(),
            output_artifact_sha256: next_artifact.clone(),
            input_state_sha256: previous_state,
            output_state_sha256: output_state.clone(),
            selection_receipt_sha256: h(200 + u64::from(step_number)),
            realization_receipt_sha256: h(300 + u64::from(step_number)),
            verification_report_sha256: h(400 + u64::from(step_number)),
            allowed_event_ids: vec![format!("event-{index:03}")],
            events_after: output.clone(),
            hard_gates: all_verified(),
            candidate_attempt_count: 3,
            failed_candidate_attempt_count: 1,
            render_call_count: 3,
            retry_count: 1,
            manual_intervention_count: 0,
        });

        previous_events = output;
        previous_state = output_state;
        previous_artifact = next_artifact;
    }

    let result = evaluate_sequence_v1(&request, &initial, &steps)?;
    if result.disposition != mel_music_sequence_integrity_harness::SequenceDispositionV1::Accepted {
        return Err("synthetic 8-step known-answer session was not accepted".into());
    }
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
