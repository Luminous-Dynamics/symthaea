use proptest::prelude::*;
use symthaea_decision_assessment::event_order_v0_3::{
    canonical_schedule_bytes, canonical_schedule_sha256, generate_delivery_schedule,
    replay_virtual_schedule,
};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn canonical_schedule_is_invariant_under_action_permutation(
        seed in any::<u64>(),
        event_count in 1usize..96,
    ) {
        let schedule = generate_delivery_schedule(seed, event_count).expect("generated schedule");
        let mut permuted = schedule.clone();
        permuted.actions.reverse();
        prop_assert_eq!(
            canonical_schedule_bytes(&schedule).expect("canonical bytes"),
            canonical_schedule_bytes(&permuted).expect("canonical bytes"),
        );
        prop_assert_eq!(
            canonical_schedule_sha256(&schedule).expect("schedule digest"),
            canonical_schedule_sha256(&permuted).expect("schedule digest"),
        );
    }

    #[test]
    fn replay_is_repeatable_for_identical_seed_and_input(
        seed in any::<u64>(),
        event_count in 1usize..96,
    ) {
        let schedule = generate_delivery_schedule(seed, event_count).expect("generated schedule");
        let first = replay_virtual_schedule(&schedule).expect("first replay");
        let second = replay_virtual_schedule(&schedule).expect("second replay");
        prop_assert_eq!(first, second);
    }
}
