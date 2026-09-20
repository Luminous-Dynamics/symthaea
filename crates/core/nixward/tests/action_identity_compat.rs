// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Independent compatibility oracle for pre-service NixActionIntentV1 identities.
//!
//! Do not rewrite these byte encodings to call the production encoder. Their purpose
//! is to catch accidental renumbering/reinterpretation of identities that already
//! existed before typed service actions were added.

use blake3::Hasher;
use nixward::action::authorization::NixActionIntentV1;
use nixward::action::executor::NixOSCommand;

const ACTION_INTENT_DOMAIN: &[u8] = b"nixward-action-intent-v1";

fn put_u8(h: &mut Hasher, value: u8) {
    h.update(&[value]);
}

fn put_u32(h: &mut Hasher, value: u32) {
    h.update(&value.to_be_bytes());
}

fn put_u64(h: &mut Hasher, value: u64) {
    h.update(&value.to_be_bytes());
}

fn put_bool(h: &mut Hasher, value: bool) {
    put_u8(h, u8::from(value));
}

fn put_str(h: &mut Hasher, value: &str) {
    put_u64(h, value.len() as u64);
    h.update(value.as_bytes());
}

fn put_opt_str(h: &mut Hasher, value: Option<&str>) {
    match value {
        Some(value) => {
            put_u8(h, 1);
            put_str(h, value);
        }
        None => put_u8(h, 0),
    }
}

fn put_opt_u32(h: &mut Hasher, value: Option<u32>) {
    match value {
        Some(value) => {
            put_u8(h, 1);
            put_u32(h, value);
        }
        None => put_u8(h, 0),
    }
}

fn put_str_vec(h: &mut Hasher, values: &[&str]) {
    put_u64(h, values.len() as u64);
    for value in values {
        put_str(h, value);
    }
}

fn finish_common_tail(h: &mut Hasher, maximum_scope_tag: u8) {
    put_u8(h, maximum_scope_tag);
    put_str_vec(h, &[]); // preconditions
    put_str_vec(h, &[]); // required postconditions
    put_opt_str(h, None); // rollback/recovery ref
}

#[test]
fn rebuild_switch_keeps_pre_service_v1_identity_encoding() {
    let command = NixOSCommand::RebuildSwitch {
        flake: Some(".#workstation".to_string()),
        extra_args: vec!["--show-trace".to_string()],
    };
    let intent = NixActionIntentV1::from_command(
        "host:workstation",
        Some("generation:42".to_string()),
        &command,
    )
    .unwrap();

    let mut expected = Hasher::new();
    expected.update(ACTION_INTENT_DOMAIN);
    put_str(&mut expected, "host:workstation");
    put_opt_str(&mut expected, Some("generation:42"));

    // Frozen #4929 encoding: RebuildSwitch action tag = 0.
    put_u8(&mut expected, 0);
    put_opt_str(&mut expected, Some(".#workstation"));
    put_str_vec(&mut expected, &["--show-trace"]);

    // Frozen #4929 encoding: SystemCritical scope tag = 3.
    finish_common_tail(&mut expected, 3);

    assert_eq!(intent.digest().unwrap(), expected.finalize().to_hex().to_string());
}

#[test]
fn collect_garbage_keeps_pre_service_v1_identity_encoding() {
    let command = NixOSCommand::CollectGarbage {
        older_than_days: Some(30),
        delete_all: true,
    };
    let intent = NixActionIntentV1::from_command("host:workstation", None, &command).unwrap();

    let mut expected = Hasher::new();
    expected.update(ACTION_INTENT_DOMAIN);
    put_str(&mut expected, "host:workstation");
    put_opt_str(&mut expected, None);

    // Frozen #4929 encoding: CollectGarbage action tag = 16.
    put_u8(&mut expected, 16);
    put_opt_u32(&mut expected, Some(30));
    put_bool(&mut expected, true);

    // Frozen #4929 encoding: Destructive scope tag = 4.
    finish_common_tail(&mut expected, 4);

    assert_eq!(intent.digest().unwrap(), expected.finalize().to_hex().to_string());
}
