// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

// IF-2 independently compiles the permit boundary before runtime export.
#[path = "../src/language/inference_permit.rs"]
mod inference_permit;

use inference_permit::*;

fn digest(byte: u8) -> BindingDigest {
    BindingDigest::new([byte; 32]).unwrap()
}

#[test]
fn prepared_execution_binds_all_six_authority_inputs() {
    let binding = InferenceExecutionBinding {
        request_digest: digest(1),
        route_digest: digest(2),
        policy_digest: digest(3),
        provider_state_digest: digest(4),
        credential_state_digest: digest(5),
        quota_state_digest: digest(6),
    };
    let mut issuer = InferencePermitIssuer::new();
    let permit = issuer.issue(binding, 50, 5, [42; 32]).unwrap();
    let prepared = issuer.prepare_execution(permit, binding, 51).unwrap();
    assert_eq!(prepared.binding(), &binding);
}
