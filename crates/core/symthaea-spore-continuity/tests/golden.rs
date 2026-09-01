use symthaea_spore_continuity::{
    ContinuityHealth, ContinuityState, LifecycleTransition, MotionProfile, QualityProfile,
};

#[test]
fn canonical_v1_vector_is_byte_stable() {
    let mut state = ContinuityState::new(
        [1u8; 32],
        [2u8; 32],
        LifecycleTransition::BootToGreeter,
    );
    state.visual_plan_digest = Some([3u8; 32]);
    state.phase_micros = 420_000;
    state.world_age_ticks = 12_345;
    state.health = ContinuityHealth::Normal;
    state.quality = QualityProfile::Calm;
    state.motion = MotionProfile::Reduced;

    let encoded = state.encode_json().expect("canonical state must encode");
    let expected = include_bytes!("fixtures/continuity-v1.json");
    assert_eq!(encoded.as_slice(), expected);

    let decoded = ContinuityState::decode_json(expected).expect("canonical vector must decode");
    assert_eq!(decoded, state);
}
