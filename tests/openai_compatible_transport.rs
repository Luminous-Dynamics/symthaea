// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

// IF-1 independently compiles the transport before runtime module export.
#[path = "../src/language/openai_compatible_transport.rs"]
mod openai_compatible_transport;

use openai_compatible_transport::*;

#[test]
fn transport_can_be_constructed_without_granting_any_execution_authority() {
    let config = OpenAiCompatibleConfig::new(
        "local-compatible",
        "http://127.0.0.1:8000/v1",
        "example-model",
        TransportCredential::None,
    )
    .unwrap();
    let transport = OpenAiCompatibleTransport::new(config).unwrap();
    assert_eq!(transport.config().provider_id(), "local-compatible");
    assert_eq!(transport.config().model(), "example-model");
}
