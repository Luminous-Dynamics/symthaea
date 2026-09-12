wit_bindgen::generate!({
    path: "wit",
    world: "extension-control-v1",
});

use exports::luminous::symthaea_extension::control::{
    ExtensionIdentity, Guest, HealthReport, HealthState,
};

include!(concat!(env!("OUT_DIR"), "/manifest_digest.rs"));

struct HelloControl;

impl Guest for HelloControl {
    fn identity() -> ExtensionIdentity {
        ExtensionIdentity {
            id: "org.example.hello-control".into(),
            version: env!("CARGO_PKG_VERSION").into(),
            abi_major: 1,
            abi_minor: 0,
            manifest_digest: MANIFEST_DIGEST.to_vec(),
        }
    }

    fn health() -> HealthReport {
        HealthReport {
            state: HealthState::Ready,
            message: Some("hello-control is ready".into()),
        }
    }
}

export!(HelloControl);
