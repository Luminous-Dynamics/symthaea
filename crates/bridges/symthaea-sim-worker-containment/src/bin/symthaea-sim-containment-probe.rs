// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::ffi::CString;
use std::io;
use std::ptr;
use symthaea_sim_worker_containment::{
    WORKER_CONTAINMENT_PROFILE_V1, install_worker_containment, seccomp_filter_active,
};

const SYS_CLONE3: libc::c_long = 435;
const SYS_BPF: libc::c_long = 321;
const SYS_IO_URING_SETUP: libc::c_long = 425;

fn main() {
    install_worker_containment().expect("install containment profile");
    assert!(seccomp_filter_active().expect("read seccomp mode"));

    // Legitimate worker timer/compiler threads must remain possible. clone3 is
    // surfaced as ENOSYS so libc falls back to CLONE_THREAD legacy clone.
    let thread_value = std::thread::spawn(|| 42u32)
        .join()
        .expect("pthread-style thread survives containment");
    assert_eq!(thread_value, 42);

    expect_errno("socket", libc::EPERM, || unsafe {
        libc::socket(libc::AF_INET, libc::SOCK_STREAM | libc::SOCK_CLOEXEC, 0) as libc::c_long
    });
    expect_errno("fork", libc::EPERM, || unsafe {
        libc::fork() as libc::c_long
    });
    expect_errno("clone3", libc::ENOSYS, || unsafe {
        libc::syscall(SYS_CLONE3, ptr::null::<libc::c_void>(), 0usize)
    });
    expect_errno("bpf", libc::EPERM, || unsafe {
        libc::syscall(SYS_BPF, 0u32, ptr::null::<libc::c_void>(), 0u32)
    });
    expect_errno("io_uring_setup", libc::EPERM, || unsafe {
        libc::syscall(SYS_IO_URING_SETUP, 1u32, ptr::null_mut::<libc::c_void>())
    });

    let path = CString::new("/bin/true").unwrap();
    let argv: [*const libc::c_char; 2] = [path.as_ptr(), ptr::null()];
    let envp: [*const libc::c_char; 1] = [ptr::null()];
    expect_errno("execve", libc::EPERM, || unsafe {
        libc::execve(path.as_ptr(), argv.as_ptr(), envp.as_ptr()) as libc::c_long
    });

    println!("containment_profile={WORKER_CONTAINMENT_PROFILE_V1}");
    println!("containment_probe=pass");
}

fn expect_errno(name: &str, expected: i32, call: impl FnOnce() -> libc::c_long) {
    let result = call();
    assert_eq!(result, -1, "{name} unexpectedly succeeded");
    let actual = io::Error::last_os_error().raw_os_error();
    assert_eq!(actual, Some(expected), "{name} returned wrong errno");
}
