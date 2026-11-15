use mel_music_audio_continuity_harness::{
    AudioBoundaryKindV1, AudioBoundaryPolicyV1, AudioGateVerdictV1,
    AudioSequenceRequestV1, AudioSequenceStepV1, AudioUpstreamGatesV1,
    FrameWindowV1, PcmAudioArtifactV1, PcmEncodingV1, PcmFormatV1,
    evaluate_audio_sequence_v1, pcm_audio_sha256_v1,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let format = PcmFormatV1 {
        sample_rate_hz: 48_000,
        channels: 2,
        encoding: PcmEncodingV1::SignedInt16LittleEndian,
    };
    let source = PcmAudioArtifactV1 {
        format,
        samples: (0..2048_usize)
            .map(|index| ((index as i32 * 97 % 20_000) - 10_000) as i16)
            .collect(),
    };
    let source_sha = pcm_audio_sha256_v1(&source)?;
    let request = AudioSequenceRequestV1 {
        schema_version: "mel-music-audio-sequence-v1".to_owned(),
        session_id: "synthetic-eight-step-audio-session".to_owned(),
        source_lineage_sha256: format!("{:064x}", 1),
        source_audio_sha256: source_sha.clone(),
        operation_plan_sha256: format!("{:064x}", 2),
        expected_step_count: 8,
        format,
        frame_count: source.frame_count()?,
    };

    let mut current = source.clone();
    let mut steps = Vec::new();
    for sequence in 1..=8_u8 {
        let window = FrameWindowV1 {
            start_frame: 16 + u64::from(sequence) * 48,
            end_frame: 32 + u64::from(sequence) * 48,
        };
        let mut output = current.clone();
        for frame in window.start_frame..window.end_frame {
            for channel in 0..usize::from(format.channels) {
                let index = frame as usize * usize::from(format.channels) + channel;
                output.samples[index] = output.samples[index].saturating_add(100);
            }
        }
        let step = AudioSequenceStepV1 {
            schema_version: "mel-music-audio-step-v1".to_owned(),
            sequence,
            step_id: format!("audio-step-{sequence}"),
            operation_id: format!("audio-op-{sequence}"),
            instruction_sha256: format!("{:064x}", 10 + u64::from(sequence)),
            source_lineage_sha256: format!("{:064x}", 1),
            parent_audio_sha256: pcm_audio_sha256_v1(&current)?,
            output_audio_sha256: pcm_audio_sha256_v1(&output)?,
            edit_regions: vec![window],
            transition_support: Vec::new(),
            boundary_policy: AudioBoundaryPolicyV1 {
                kind: AudioBoundaryKindV1::OperationSpecific,
                measurement_radius_frames: 4,
                maximum_jump_delta_samples: Some(32_767),
                maximum_side_rms_delta_samples: Some(32_767),
            },
            upstream_gates: AudioUpstreamGatesV1 {
                rights_admission: AudioGateVerdictV1::Verified,
                source_lineage: AudioGateVerdictV1::Verified,
                realization_receipt: AudioGateVerdictV1::Verified,
                transaction_admission: AudioGateVerdictV1::Verified,
            },
            candidate_attempt_count: 2,
            failed_candidate_attempt_count: 1,
            render_call_count: 2,
            retry_count: 1,
            manual_intervention_count: 0,
            output: output.clone(),
        };
        steps.push(step);
        current = output;
    }

    let result = evaluate_audio_sequence_v1(&request, &source, &steps)?;
    if result.accepted_step_count != 8 {
        return Err("synthetic 8-step audio contract did not qualify".into());
    }
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
