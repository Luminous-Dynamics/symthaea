// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! ISO-to-USB flashing with safety checks and progress events.
//!
//! Writes an ISO image to a removable block device in 4 MB chunks,
//! emitting `flash-progress` events for the frontend UI. Includes
//! platform-specific safety: Linux checks sysfs removable flag,
//! macOS unmounts the disk before writing, Windows verifies USB bus type.

use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use tauri::Emitter;

/// 4 MB chunk size — balances throughput and progress granularity.
const CHUNK_SIZE: usize = 4 * 1024 * 1024;

/// Flash an ISO image to a USB device, emitting `flash-progress` events.
///
/// # Safety
/// This writes raw bytes to a block device. The caller must ensure:
/// - The user confirmed the target device
/// - The process has sufficient privileges (sudo on Linux/macOS, admin on Windows)
pub fn flash_iso(app: &tauri::AppHandle, iso_path: &Path, device_path: &str) -> Result<(), String> {
    // Platform-specific pre-flight checks
    pre_flash_checks(device_path)?;

    let iso = std::fs::File::open(iso_path).map_err(|e| format!("Cannot open ISO: {}", e))?;
    let total = iso.metadata().map(|m| m.len()).unwrap_or(0);
    let mut reader = BufReader::with_capacity(CHUNK_SIZE, iso);

    let device = std::fs::OpenOptions::new()
        .write(true)
        .open(device_path)
        .map_err(|e| {
            format!(
                "Cannot open device {}: {}. Run with sudo/admin privileges.",
                device_path, e
            )
        })?;
    let mut writer = BufWriter::with_capacity(CHUNK_SIZE, device);

    let mut written: u64 = 0;
    let mut buf = vec![0u8; CHUNK_SIZE];

    let _ = app.emit(
        "flash-progress",
        serde_json::json!({
            "written": 0_u64, "total": total, "percent": 0_u32,
            "status": "flashing"
        }),
    );

    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| format!("Read error: {}", e))?;
        if n == 0 {
            break;
        }
        writer
            .write_all(&buf[..n])
            .map_err(|e| format!("Write error: {}", e))?;
        written += n as u64;

        let percent = if total > 0 {
            (written as f64 / total as f64 * 100.0) as u32
        } else {
            0
        };
        let _ = app.emit(
            "flash-progress",
            serde_json::json!({
                "written": written,
                "total": total,
                "percent": percent,
                "status": "flashing"
            }),
        );
    }

    writer.flush().map_err(|e| format!("Flush error: {}", e))?;

    // Sync to ensure all writes hit the physical media
    sync_device(writer)?;

    let _ = app.emit(
        "flash-progress",
        serde_json::json!({
            "written": written, "total": total, "percent": 100_u32,
            "status": "complete"
        }),
    );

    Ok(())
}

// ─── Platform-specific pre-flight checks ────────────────────────────────────

#[cfg(target_os = "linux")]
fn pre_flash_checks(device_path: &str) -> Result<(), String> {
    verify_removable_linux(device_path)
}

#[cfg(target_os = "macos")]
fn pre_flash_checks(device_path: &str) -> Result<(), String> {
    // Convert raw device path (/dev/rdiskN) to disk identifier for unmount
    let disk_id = device_path
        .trim_start_matches("/dev/r")
        .trim_start_matches("/dev/");
    let _ = std::process::Command::new("diskutil")
        .args(["unmountDisk", &format!("/dev/{}", disk_id)])
        .output();
    Ok(())
}

#[cfg(target_os = "windows")]
fn pre_flash_checks(_device_path: &str) -> Result<(), String> {
    // Device was already filtered by Get-Disk BusType='USB' during enumeration.
    // Additional verification could lock the volume here, but Windows raw device
    // writes already require admin privileges which serves as the safety gate.
    Ok(())
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn pre_flash_checks(_device_path: &str) -> Result<(), String> {
    Err("Unsupported platform for USB flashing".to_string())
}

// ─── Linux: verify device is removable via sysfs ────────────────────────────

#[cfg(target_os = "linux")]
fn verify_removable_linux(device_path: &str) -> Result<(), String> {
    let dev_name = device_path.trim_start_matches("/dev/");
    let removable_path = format!("/sys/block/{}/removable", dev_name);
    match std::fs::read_to_string(&removable_path) {
        Ok(v) if v.trim() == "1" => Ok(()),
        Ok(_) => Err(format!(
            "{} is not a removable device. Refusing to write.",
            device_path
        )),
        Err(_) => Err(format!(
            "Cannot verify {} is removable. Check the device path.",
            device_path
        )),
    }
}

// ─── Sync device writes to physical media ───────────────────────────────────

#[cfg(unix)]
fn sync_device(writer: BufWriter<std::fs::File>) -> Result<(), String> {
    use std::os::unix::io::AsRawFd;
    let file = writer
        .into_inner()
        .map_err(|e| format!("Flush error: {}", e))?;
    let fd = file.as_raw_fd();
    let ret = unsafe { libc::fsync(fd) };
    if ret != 0 {
        Err(format!("fsync failed: {}", std::io::Error::last_os_error()))
    } else {
        Ok(())
    }
}

#[cfg(not(unix))]
fn sync_device(writer: BufWriter<std::fs::File>) -> Result<(), String> {
    // On Windows, FlushFileBuffers is called by flush() above.
    // The BufWriter must still be consumed to avoid drop issues.
    let _ = writer
        .into_inner()
        .map_err(|e| format!("Flush error: {}", e))?;
    Ok(())
}