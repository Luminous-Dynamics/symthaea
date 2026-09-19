// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use super::*;

fn wasm_action(module_path: PathBuf, function_name: &str) -> ActionIR {
    ActionIR::WasmSandbox {
        module_path,
        function_name: function_name.to_string(),
        input_data: Vec::new(),
    }
}

#[test]
fn wasm_validation_allows_in_sandbox_module_path() {
    let policy = PolicyBundle::restrictive();
    let sandbox = SandboxRoot::new("test_wasm_path_inside").unwrap();
    let module_path = sandbox.root().join("module.wasm");
    std::fs::write(&module_path, b"not-yet-validated-wasm").unwrap();

    let action = wasm_action(module_path, "verify");
    assert!(action.validate(&policy, &sandbox, 1.0).is_ok());
}

#[test]
fn wasm_validation_rejects_module_path_outside_sandbox() {
    let policy = PolicyBundle::restrictive();
    let sandbox = SandboxRoot::new("test_wasm_path_outside").unwrap();
    let action = wasm_action(PathBuf::from("/etc/passwd"), "verify");

    let err = action
        .validate(&policy, &sandbox, 1.0)
        .expect_err("Wasm module path outside SandboxRoot must be rejected");
    assert!(matches!(
        err,
        PolicyViolation::SandboxEscape(_) | PolicyViolation::ReadNotAllowed(_)
    ));
}

#[cfg(unix)]
#[test]
fn wasm_validation_rejects_symlink_escape() {
    use std::os::unix::fs::symlink;

    let policy = PolicyBundle::restrictive();
    let sandbox = SandboxRoot::new("test_wasm_symlink_escape").unwrap();
    let module_path = sandbox.root().join("escape.wasm");
    let _ = std::fs::remove_file(&module_path);
    symlink("/etc/passwd", &module_path).unwrap();

    let err = wasm_action(module_path, "verify")
        .validate(&policy, &sandbox, 1.0)
        .expect_err("symlink escaping SandboxRoot must be rejected after canonicalization");
    assert!(matches!(err, PolicyViolation::SandboxEscape(_)));
}

// (module (func (export "verify") (result i32) i32.const 1))
#[cfg(feature = "wasm-sandbox")]
const RETURN_ONE_WASM: &[u8] = &[
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, // magic + version
    0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f, // type: () -> i32
    0x03, 0x02, 0x01, 0x00, // function section
    0x07, 0x0a, 0x01, 0x06, b'v', b'e', b'r', b'i', b'f', b'y', 0x00, 0x00, // export
    0x0a, 0x06, 0x01, 0x04, 0x00, 0x41, 0x01, 0x0b, // body: i32.const 1
];

// (module
//   (func (export "verify") (result i32)
//     (loop $forever (br $forever))
//     i32.const 1))
#[cfg(feature = "wasm-sandbox")]
const INFINITE_LOOP_WASM: &[u8] = &[
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, // magic + version
    0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f, // type: () -> i32
    0x03, 0x02, 0x01, 0x00, // function section
    0x07, 0x0a, 0x01, 0x06, b'v', b'e', b'r', b'i', b'f', b'y', 0x00, 0x00, // export
    0x0a, 0x0b, 0x01, 0x09, 0x00, 0x03, 0x40, 0x0c, 0x00, 0x0b, 0x41, 0x01, 0x0b,
];

// (module
//   (memory 1)
//   (func (export "verify") (result i32)
//     i32.const 2048
//     memory.grow
//     drop
//     i32.const 1))
//
// 2048 additional 64-KiB pages exceeds the 64-MiB StoreLimits ceiling.
#[cfg(feature = "wasm-sandbox")]
const MEMORY_GROW_WASM: &[u8] = &[
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, // magic + version
    0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f, // type: () -> i32
    0x03, 0x02, 0x01, 0x00, // function section
    0x05, 0x03, 0x01, 0x00, 0x01, // one memory, min 1 page, no declared max
    0x07, 0x0a, 0x01, 0x06, b'v', b'e', b'r', b'i', b'f', b'y', 0x00, 0x00, // export
    0x0a, 0x0c, 0x01, 0x0a, 0x00, 0x41, 0x80, 0x10, 0x40, 0x00, 0x1a, 0x41, 0x01, 0x0b,
];

// (module
//   (import "env" "host" (func $host))
//   (func (export "verify") (result i32)
//     call $host
//     i32.const 1))
#[cfg(feature = "wasm-sandbox")]
const REQUIRES_HOST_IMPORT_WASM: &[u8] = &[
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, // magic + version
    0x01, 0x08, 0x02, 0x60, 0x00, 0x00, 0x60, 0x00, 0x01, 0x7f, // two types
    0x02, 0x0c, 0x01, 0x03, b'e', b'n', b'v', 0x04, b'h', b'o', b's', b't', 0x00, 0x00,
    0x03, 0x02, 0x01, 0x01, // one defined function of type 1
    0x07, 0x0a, 0x01, 0x06, b'v', b'e', b'r', b'i', b'f', b'y', 0x00, 0x01, // export fn 1
    0x0a, 0x08, 0x01, 0x06, 0x00, 0x10, 0x00, 0x41, 0x01, 0x0b, // call import 0
];

#[cfg(feature = "wasm-sandbox")]
fn write_fixture(sandbox: &SandboxRoot, name: &str, bytes: &[u8]) -> PathBuf {
    let path = sandbox.root().join(name);
    std::fs::write(&path, bytes).unwrap();
    path
}

#[cfg(feature = "wasm-sandbox")]
#[test]
fn bounded_wasm_executes_finite_module() {
    let policy = PolicyBundle::restrictive();
    let sandbox = SandboxRoot::new("test_wasm_finite").unwrap();
    let module_path = write_fixture(&sandbox, "return-one.wasm", RETURN_ONE_WASM);
    let action = wasm_action(module_path, "verify");
    let mut executor = SimpleExecutor::with_real_commands();

    let outcome = executor
        .execute(&action, &policy, &sandbox, 1.0)
        .expect("finite no-import Wasm module should execute inside bounded store");

    match outcome.outcome {
        ActionOutcome::WasmResult { output, .. } => assert_eq!(output, vec![1]),
        other => panic!("expected WasmResult, got {other:?}"),
    }
}

#[cfg(feature = "wasm-sandbox")]
#[test]
fn bounded_wasm_exhausts_fuel_in_infinite_loop() {
    let policy = PolicyBundle::restrictive();
    let sandbox = SandboxRoot::new("test_wasm_fuel").unwrap();
    let module_path = write_fixture(&sandbox, "infinite-loop.wasm", INFINITE_LOOP_WASM);
    let action = wasm_action(module_path, "verify");
    let mut executor = SimpleExecutor::with_real_commands();

    let err = executor
        .execute(&action, &policy, &sandbox, 1.0)
        .expect_err("infinite-loop guest must be interrupted by finite fuel");
    let message = err.to_string();
    assert!(
        message.contains("wasm guest trapped") || message.contains("fuel"),
        "expected bounded guest trap/fuel error, got: {message}"
    );
}

#[cfg(feature = "wasm-sandbox")]
#[test]
fn bounded_wasm_traps_memory_growth_beyond_store_limit() {
    let policy = PolicyBundle::restrictive();
    let sandbox = SandboxRoot::new("test_wasm_memory_limit").unwrap();
    let module_path = write_fixture(&sandbox, "memory-grow.wasm", MEMORY_GROW_WASM);
    let action = wasm_action(module_path, "verify");
    let mut executor = SimpleExecutor::with_real_commands();

    let err = executor
        .execute(&action, &policy, &sandbox, 1.0)
        .expect_err("guest memory growth above 64 MiB must trap");
    assert!(
        err.to_string().contains("wasm guest trapped"),
        "unexpected memory-limit error: {err}"
    );
}

#[cfg(feature = "wasm-sandbox")]
#[test]
fn bounded_wasm_rejects_unprovided_host_import() {
    let policy = PolicyBundle::restrictive();
    let sandbox = SandboxRoot::new("test_wasm_no_imports").unwrap();
    let module_path = write_fixture(&sandbox, "requires-host.wasm", REQUIRES_HOST_IMPORT_WASM);
    let action = wasm_action(module_path, "verify");
    let mut executor = SimpleExecutor::with_real_commands();

    let err = executor
        .execute(&action, &policy, &sandbox, 1.0)
        .expect_err("raw ActionIR sandbox must not provide ambient host imports");
    assert!(
        err.to_string().contains("failed to instantiate wasm module"),
        "unexpected import-refusal error: {err}"
    );
}

#[cfg(feature = "wasm-sandbox")]
#[test]
fn bounded_wasm_rejects_malformed_module() {
    let policy = PolicyBundle::restrictive();
    let sandbox = SandboxRoot::new("test_wasm_malformed").unwrap();
    let module_path = write_fixture(&sandbox, "malformed.wasm", b"not-webassembly");
    let action = wasm_action(module_path, "verify");
    let mut executor = SimpleExecutor::with_real_commands();

    let err = executor
        .execute(&action, &policy, &sandbox, 1.0)
        .expect_err("malformed Wasm bytes must fail validation/compilation");
    assert!(
        err.to_string().contains("failed to load wasm module"),
        "unexpected malformed-module error: {err}"
    );
}

#[cfg(feature = "wasm-sandbox")]
#[test]
fn bounded_wasm_missing_export_fails_closed_and_next_guest_still_runs() {
    let policy = PolicyBundle::restrictive();
    let sandbox = SandboxRoot::new("test_wasm_missing_export").unwrap();
    let module_path = write_fixture(&sandbox, "return-one.wasm", RETURN_ONE_WASM);
    let bad_action = wasm_action(module_path.clone(), "not_an_export");
    let good_action = wasm_action(module_path, "verify");
    let mut executor = SimpleExecutor::with_real_commands();

    let err = executor
        .execute(&bad_action, &policy, &sandbox, 1.0)
        .expect_err("unknown Wasm export must fail closed");
    assert!(
        err.to_string().contains("failed to resolve wasm export"),
        "unexpected missing-export error: {err}"
    );

    let outcome = executor
        .execute(&good_action, &policy, &sandbox, 1.0)
        .expect("a guest failure must not poison subsequent bounded execution");
    assert!(matches!(
        outcome.outcome,
        ActionOutcome::WasmResult { ref output, .. } if output == &vec![1]
    ));
}
