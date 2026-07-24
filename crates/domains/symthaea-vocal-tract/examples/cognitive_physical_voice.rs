// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_core::genesis::GenesisSeed;
use symthaea_vocal_tract::{
    ArticulatoryQualityRequirements, ArticulatoryTimingConfig, BranchedWaveguideConfig,
    BranchedWaveguideV2, GesturePlanner, IdentityAnatomy, IdentityPhysiology, ProsodyContext,
    VocalTractPipeline, VoiceCognitiveState, analyze_articulatory_trajectory,
};

fn main() -> Result<(), &'static str> {
    let genesis = GenesisSeed::from_phrase("symthaea-cognitive-physical-voice-example");
    let mut pipeline = VocalTractPipeline::new(&genesis);
    let head =
        pipeline.bootstrap_gesture_projection(ArticulatoryTimingConfig::default(), 24, 1e-3)?;
    let mut planner = GesturePlanner::default();
    let anatomy = IdentityAnatomy::velvet();
    let physiology = IdentityPhysiology::default();
    let prosody = ProsodyContext {
        base_f0: 185.0,
        stress: 1,
        is_focus: true,
        ..ProsodyContext::default()
    };
    let cognitive = VoiceCognitiveState {
        emotional_valence: 0.25,
        emotional_arousal: 0.62,
        epistemic_confidence: 0.78,
        integrated_phi: 0.72,
        expected_free_energy: 0.55,
        ..VoiceCognitiveState::default()
    };

    let phonemes = ["M", "AY", "SIL", "N"];
    let frames_per_phoneme = [12usize, 24, 10, 16];
    let mut gestures = Vec::new();
    let mut physical = Vec::new();

    for (index, phoneme) in phonemes.iter().enumerate() {
        let next = phonemes.get(index + 1).copied();
        let frame_count = frames_per_phoneme[index];
        for frame_index in 0..frame_count {
            let motor = pipeline.tick_with_anticipation_physical(
                &cognitive,
                None,
                0.005,
                Some(phoneme),
                next,
                frame_count - frame_index,
                &prosody,
                &head,
                &mut planner,
                &anatomy,
                &physiology,
            )?;
            gestures.push(motor.gesture);
            physical.push(motor.physical);
        }
    }

    let quality = analyze_articulatory_trajectory(&gestures, 200.0)?;
    let gate = quality.gate(&ArticulatoryQualityRequirements {
        require_silence_coverage: true,
        ..ArticulatoryQualityRequirements::default()
    })?;
    if !gate.pass {
        return Err("cognitive physical voice trajectory failed quality gates");
    }

    let mut renderer = BranchedWaveguideV2::try_new(BranchedWaveguideConfig::default())?;
    let stems = renderer.render_frames(&physical, 200.0)?;
    if stems.final_output.is_empty() || stems.final_output.iter().any(|sample| !sample.is_finite())
    {
        return Err("physical waveguide produced invalid PCM");
    }

    println!(
        "rendered {} motor frames into {} PCM samples; max articulator slew {:.3}/s; silence leakage {}",
        physical.len(),
        stems.final_output.len(),
        quality.maximum_coordinate_slew_per_second,
        quality.silence_leakage_frames,
    );
    Ok(())
}
