// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_boot_protocol::state::BootStateReducer;
use symthaea_boot_protocol::{BootEvent, BootHealth, BootPhase};

#[test]
fn boot_ready_with_unknown_health_stays_unknown() {
    let mut reducer = BootStateReducer::default();
    reducer
        .try_apply(&BootEvent::BootReady {
            sequence: 1,
            elapsed_ms: 10,
            health: BootHealth::Unknown,
        })
        .unwrap();

    let snapshot = reducer.snapshot();
    assert_eq!(snapshot.phase, BootPhase::Ready);
    assert_eq!(snapshot.health, BootHealth::Unknown);
}

#[test]
fn boot_ready_becomes_normal_only_when_normal_is_explicit() {
    let mut reducer = BootStateReducer::default();
    reducer
        .try_apply(&BootEvent::BootReady {
            sequence: 1,
            elapsed_ms: 10,
            health: BootHealth::Normal,
        })
        .unwrap();

    let snapshot = reducer.snapshot();
    assert_eq!(snapshot.phase, BootPhase::Ready);
    assert_eq!(snapshot.health, BootHealth::Normal);
}
