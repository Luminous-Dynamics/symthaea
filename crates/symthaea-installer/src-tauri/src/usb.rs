// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! USB device detection for NixForHumanity USB creator.
//!
//! Enumerates removable USB storage devices on Linux, macOS, and Windows
//! using platform-native APIs (sysfs, diskutil, PowerShell).

use serde::Serialize;

/// A removable USB storage device suitable for flashing.
#[derive(Debug, Clone, Serialize)]
pub struct UsbDevice {
    /// Block device path (e.g. `/dev/sdb`, `/dev/rdisk2`, `\\.\PhysicalDrive1`)
    pub path: String,
    /// Human-readable device name / model
    pub name: String,
    /// Total capacity in bytes
    pub size_bytes: u64,
    /// Human-readable size (e.g. "16.0 GB")
    pub size_human: String,
    /// Whether the device is removable (always true in returned list)
    pub removable: bool,
}

/// List all removable USB storage devices on the current platform.
pub fn list_devices() -> Vec<UsbDevice> {
    list_devices_platform()
}

fn human_size(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1e9)
    } else if bytes >= 1_000_000 {
        format!("{:.0} MB", bytes as f64 / 1e6)
    } else {
        format!("{} bytes", bytes)
    }
}

// ─── Linux: sysfs enumeration ───────────────────────────────────────────────

#[cfg(target_os = "linux")]
fn list_devices_platform() -> Vec<UsbDevice> {
    let mut devices = Vec::new();

    let block_dir = match std::fs::read_dir("/sys/block") {
        Ok(d) => d,
        Err(_) => return devices,
    };

    for entry in block_dir.flatten() {
        let dev_name = match entry.file_name().into_string() {
            Ok(n) => n,
            Err(_) => continue,
        };

        // Skip virtual devices: loop*, zram*, dm-*, ram*, nbd*
        if dev_name.starts_with("loop")
            || dev_name.starts_with("zram")
            || dev_name.starts_with("dm-")
            || dev_name.starts_with("ram")
            || dev_name.starts_with("nbd")
        {
            continue;
        }

        // Check removable flag
        let removable_path = format!("/sys/block/{}/removable", dev_name);
        let removable = match std::fs::read_to_string(&removable_path) {
            Ok(v) => v.trim() == "1",
            Err(_) => false,
        };

        if !removable {
            continue;
        }

        // Read size in 512-byte sectors
        let size_path = format!("/sys/block/{}/size", dev_name);
        let size_bytes = std::fs::read_to_string(&size_path)
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .map(|sectors| sectors * 512)
            .unwrap_or(0);

        // Skip tiny devices (< 100MB, likely not real USB drives)
        if size_bytes < 100_000_000 {
            continue;
        }

        // Read model name
        let model_path = format!("/sys/block/{}/device/model", dev_name);
        let model = std::fs::read_to_string(&model_path)
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|_| dev_name.clone());

        let path = format!("/dev/{}", dev_name);

        devices.push(UsbDevice {
            path,
            name: model,
            size_bytes,
            size_human: human_size(size_bytes),
            removable: true,
        });
    }

    // Sort by path for stable ordering
    devices.sort_by(|a, b| a.path.cmp(&b.path));
    devices
}

// ─── macOS: diskutil enumeration ────────────────────────────────────────────

#[cfg(target_os = "macos")]
fn list_devices_platform() -> Vec<UsbDevice> {
    let mut devices = Vec::new();

    // diskutil list -plist external physical outputs XML plist
    let output = match std::process::Command::new("diskutil")
        .args(["list", "-plist", "external", "physical"])
        .output()
    {
        Ok(o) => o,
        Err(_) => return devices,
    };

    if !output.status.success() {
        return devices;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Parse disk identifiers from the plist (look for <string>diskN</string>
    // under AllDisksAndPartitions array)
    let disk_ids: Vec<String> = stdout
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if trimmed.starts_with("<string>disk") && trimmed.ends_with("</string>") {
                let inner = trimmed
                    .trim_start_matches("<string>")
                    .trim_end_matches("</string>");
                // Only top-level disks (diskN, not diskNsM)
                if inner.starts_with("disk") && !inner.contains('s') {
                    return Some(inner.to_string());
                }
            }
            None
        })
        .collect();

    for disk_id in disk_ids {
        // Get info for each disk
        let info_output = match std::process::Command::new("diskutil")
            .args(["info", "-plist", &disk_id])
            .output()
        {
            Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).to_string(),
            _ => continue,
        };

        let removable = info_output.contains("<key>RemovableMediaOrExternalDevice</key>")
            && info_output
                .lines()
                .skip_while(|l| !l.contains("RemovableMediaOrExternalDevice"))
                .nth(1)
                .map(|l| l.trim() == "<true/>")
                .unwrap_or(false);

        if !removable {
            continue;
        }

        // Extract size
        let size_bytes = extract_plist_integer(&info_output, "TotalSize").unwrap_or(0);

        if size_bytes < 100_000_000 {
            continue;
        }

        // Extract model name
        let name =
            extract_plist_string(&info_output, "MediaName").unwrap_or_else(|| disk_id.clone());

        // Use raw device for faster writes
        let path = format!("/dev/r{}", disk_id);

        devices.push(UsbDevice {
            path,
            name,
            size_bytes,
            size_human: human_size(size_bytes),
            removable: true,
        });
    }

    devices
}

#[cfg(target_os = "macos")]
fn extract_plist_string(plist: &str, key: &str) -> Option<String> {
    let key_tag = format!("<key>{}</key>", key);
    plist
        .lines()
        .skip_while(|l| !l.contains(&key_tag))
        .nth(1)
        .and_then(|l| {
            let trimmed = l.trim();
            if trimmed.starts_with("<string>") && trimmed.ends_with("</string>") {
                Some(
                    trimmed
                        .trim_start_matches("<string>")
                        .trim_end_matches("</string>")
                        .to_string(),
                )
            } else {
                None
            }
        })
}

#[cfg(target_os = "macos")]
fn extract_plist_integer(plist: &str, key: &str) -> Option<u64> {
    let key_tag = format!("<key>{}</key>", key);
    plist
        .lines()
        .skip_while(|l| !l.contains(&key_tag))
        .nth(1)
        .and_then(|l| {
            let trimmed = l.trim();
            if trimmed.starts_with("<integer>") && trimmed.ends_with("</integer>") {
                trimmed
                    .trim_start_matches("<integer>")
                    .trim_end_matches("</integer>")
                    .parse()
                    .ok()
            } else {
                None
            }
        })
}

// ─── Windows: PowerShell enumeration ────────────────────────────────────────

#[cfg(target_os = "windows")]
fn list_devices_platform() -> Vec<UsbDevice> {
    let mut devices = Vec::new();

    let output = match std::process::Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            "Get-Disk | Where-Object BusType -eq 'USB' | Select-Object Number,FriendlyName,Size | ConvertTo-Json",
        ])
        .output()
    {
        Ok(o) => o,
        Err(_) => return devices,
    };

    if !output.status.success() {
        return devices;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stdout = stdout.trim();
    if stdout.is_empty() {
        return devices;
    }

    // PowerShell returns a single object (not array) when only 1 disk.
    // Normalize to always be an array.
    let json_str = if stdout.starts_with('[') {
        stdout.to_string()
    } else {
        format!("[{}]", stdout)
    };

    let parsed: Vec<serde_json::Value> = match serde_json::from_str(&json_str) {
        Ok(v) => v,
        Err(_) => return devices,
    };

    for disk in parsed {
        let number = disk.get("Number").and_then(|v| v.as_u64()).unwrap_or(0);
        let friendly_name = disk
            .get("FriendlyName")
            .and_then(|v| v.as_str())
            .unwrap_or("USB Disk")
            .to_string();
        let size = disk.get("Size").and_then(|v| v.as_u64()).unwrap_or(0);

        if size < 100_000_000 {
            continue;
        }

        let path = format!(r"\\.\PhysicalDrive{}", number);

        devices.push(UsbDevice {
            path,
            name: friendly_name,
            size_bytes: size,
            size_human: human_size(size),
            removable: true,
        });
    }

    devices
}

// ─── Fallback for unsupported platforms ─────────────────────────────────────

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn list_devices_platform() -> Vec<UsbDevice> {
    Vec::new()
}