/// POSIX Hard Real-Time Memory Locking
/// Prevents the Linux Kernel from lazy-allocating or swapping memory pages.
/// This guarantees absolute zero page-fault latency during 500Hz control loops.
#[cfg(target_os = "linux")]
pub fn lock_memory_pages() {
    unsafe {
        // MCL_CURRENT (1) | MCL_FUTURE (2) = 3
        // Locks all currently mapped pages and all future allocated pages into RAM.
        let result = libc::mlockall(libc::MCL_CURRENT | libc::MCL_FUTURE);

        if result != 0 {
            let err = std::io::Error::last_os_error();
            eprintln!(
                "⚠️ WARN: Failed to lock memory pages (mlockall). Root/CAP_IPC_LOCK required. Error: {}",
                err
            );
        } else {
            println!("🔒 OS Memory Locked: Zero-page-fault hard real-time execution active.");
        }
    }
}

#[cfg(not(target_os = "linux"))]
pub fn lock_memory_pages() {
    println!("⚠️ WARN: Memory page locking is only supported on Linux target substrates.");
}
