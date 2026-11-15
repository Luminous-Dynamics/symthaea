use mel_rti_streaming_trace_scorer::{
    PredictedTurnDecisionV1, StreamChunkPredictionV1, StreamChunkTruthV1,
    StreamChunkV1, StreamSessionRequestV1, StreamScorerErrorV1,
    STREAM_CHUNK_SCHEMA_V1, STREAM_PREDICTION_SCHEMA_V1,
    STREAM_SESSION_SCHEMA_V1, STREAM_TRUTH_SCHEMA_V1, TurnActionV1,
    score_streaming_session_v1, verify_stream_score_report_v1,
};
use serde_json::to_string_pretty;

fn h(value: u64) -> String {
    format!("{value:064x}")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let request = StreamSessionRequestV1 {
        schema_version: STREAM_SESSION_SCHEMA_V1.to_owned(),
        session_id: "synthetic-streaming-turntaking-v1".to_owned(),
        source_audio_sha256: h(1),
        task_manifest_sha256: h(2),
        annotation_policy_sha256: h(3),
        immutable_policy_identity_sha256: h(4),
        runner_build_sha256: h(5),
        initial_policy_state_sha256: h(100),
        sample_rate_hz: 48_000,
        channels: 2,
        total_frames: 100,
        expected_chunk_count: 5,
        seed: 7,
    };

    let intervals = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)];
    let chunks: Vec<_> = intervals
        .iter()
        .enumerate()
        .map(|(index, (start, end))| StreamChunkV1 {
            schema_version: STREAM_CHUNK_SCHEMA_V1.to_owned(),
            chunk_index: index as u32,
            chunk_id: format!("chunk-{index}"),
            start_frame: *start,
            end_frame: *end,
        })
        .collect();

    let truth = vec![
        StreamChunkTruthV1 {
            schema_version: STREAM_TRUTH_SCHEMA_V1.to_owned(),
            chunk_index: 0, chunk_id: "chunk-0".to_owned(),
            valid_actions: vec![TurnActionV1::Wait],
            effect_target_frame: None, effect_window: None,
        },
        StreamChunkTruthV1 {
            schema_version: STREAM_TRUTH_SCHEMA_V1.to_owned(),
            chunk_index: 1, chunk_id: "chunk-1".to_owned(),
            valid_actions: vec![TurnActionV1::Respond],
            effect_target_frame: Some(50),
            effect_window: Some(mel_rti_streaming_trace_scorer::FrameWindowV1 {
                start_frame: 40, end_frame: 60,
            }),
        },
        StreamChunkTruthV1 {
            schema_version: STREAM_TRUTH_SCHEMA_V1.to_owned(),
            chunk_index: 2, chunk_id: "chunk-2".to_owned(),
            valid_actions: vec![TurnActionV1::Continue],
            effect_target_frame: None, effect_window: None,
        },
        StreamChunkTruthV1 {
            schema_version: STREAM_TRUTH_SCHEMA_V1.to_owned(),
            chunk_index: 3, chunk_id: "chunk-3".to_owned(),
            valid_actions: vec![TurnActionV1::Yield],
            effect_target_frame: Some(90),
            effect_window: Some(mel_rti_streaming_trace_scorer::FrameWindowV1 {
                start_frame: 80, end_frame: 100,
            }),
        },
        StreamChunkTruthV1 {
            schema_version: STREAM_TRUTH_SCHEMA_V1.to_owned(),
            chunk_index: 4, chunk_id: "chunk-4".to_owned(),
            valid_actions: vec![TurnActionV1::Wait],
            effect_target_frame: None, effect_window: None,
        },
    ];

    let decisions = [
        (PredictedTurnDecisionV1::Wait, None, None),
        (PredictedTurnDecisionV1::Respond, Some(50), Some(220)),
        (PredictedTurnDecisionV1::Continue, None, None),
        (PredictedTurnDecisionV1::Yield, Some(90), Some(420)),
        (PredictedTurnDecisionV1::Wait, None, None),
    ];
    let predictions: Vec<_> = decisions
        .iter()
        .enumerate()
        .map(|(index, (decision, effect_frame, effect_at_ns))| {
            StreamChunkPredictionV1 {
                schema_version: STREAM_PREDICTION_SCHEMA_V1.to_owned(),
                chunk_index: index as u32,
                chunk_id: format!("chunk-{index}"),
                decision: *decision,
                observed_through_frame: (index as u64 + 1) * 20,
                chunk_received_at_ns: (index as u64 + 1) * 100,
                decision_at_ns: (index as u64 + 1) * 100 + 10,
                effect_at_ns: *effect_at_ns,
                effect_frame: *effect_frame,
                input_prefix_sha256: h(200 + index as u64),
                pre_state_sha256: if index == 0 {
                    request.initial_policy_state_sha256.clone()
                } else {
                    h(300 + index as u64)
                },
                post_state_sha256: h(301 + index as u64),
                immutable_policy_identity_sha256: request.immutable_policy_identity_sha256.clone(),
                runner_build_sha256: request.runner_build_sha256.clone(),
                parse_failure_sha256: None,
            }
        })
        .collect();

    let report = score_streaming_session_v1(&request, &chunks, &truth, &predictions)?;
    if report.summary.top1_accuracy_bp != Some(10_000)
        || report.summary.effect_window_success_count != 2
        || report.summary.post_yield_non_wait_leakage_count != 0
    {
        return Err("synthetic known-answer streaming trace failed".into());
    }
    verify_stream_score_report_v1(&request, &chunks, &truth, &predictions, &report)
        .map_err(|error: StreamScorerErrorV1| error)?;
    println!("{}", to_string_pretty(&report)?);
    Ok(())
}
