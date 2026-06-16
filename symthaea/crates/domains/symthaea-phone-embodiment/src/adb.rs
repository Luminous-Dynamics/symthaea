// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! ADB command wrapper for phone control.

use std::process::Command;

/// ADB command executor for a specific device.
pub struct AdbDevice {
    serial: String,
}

impl AdbDevice {
    pub fn new(serial: impl Into<String>) -> Self {
        Self {
            serial: serial.into(),
        }
    }

    /// Device serial as configured at construction time.
    pub fn serial(&self) -> &str {
        &self.serial
    }

    /// Check if the device is connected and authorized.
    pub fn is_connected(&self) -> bool {
        let output = Command::new("adb").args(["devices"]).output().ok();
        output.map_or(false, |o| {
            String::from_utf8_lossy(&o.stdout).contains(&self.serial)
        })
    }

    /// Capture a screenshot and return the PNG bytes.
    pub fn screenshot(&self) -> Result<Vec<u8>, String> {
        // Capture to device
        let status = Command::new("adb")
            .args([
                "-s",
                &self.serial,
                "shell",
                "screencap",
                "-p",
                "/sdcard/_symthaea_screen.png",
            ])
            .status()
            .map_err(|e| format!("adb screencap failed: {e}"))?;
        if !status.success() {
            return Err("adb screencap returned non-zero".into());
        }

        // Pull to temp file
        let tmp_path = "/tmp/_symthaea_phone_screen.png";
        let status = Command::new("adb")
            .args([
                "-s",
                &self.serial,
                "pull",
                "/sdcard/_symthaea_screen.png",
                tmp_path,
            ])
            .status()
            .map_err(|e| format!("adb pull failed: {e}"))?;
        if !status.success() {
            return Err("adb pull returned non-zero".into());
        }

        std::fs::read(tmp_path).map_err(|e| format!("Failed to read screenshot: {e}"))
    }

    /// Tap at screen coordinates.
    pub fn tap(&self, x: u32, y: u32) -> Result<(), String> {
        let status = Command::new("adb")
            .args([
                "-s",
                &self.serial,
                "shell",
                "input",
                "tap",
                &x.to_string(),
                &y.to_string(),
            ])
            .status()
            .map_err(|e| format!("adb tap failed: {e}"))?;
        if status.success() {
            Ok(())
        } else {
            Err("adb tap non-zero".into())
        }
    }

    /// Swipe from (x1,y1) to (x2,y2) over duration_ms.
    pub fn swipe(
        &self,
        x1: u32,
        y1: u32,
        x2: u32,
        y2: u32,
        duration_ms: u32,
    ) -> Result<(), String> {
        let status = Command::new("adb")
            .args([
                "-s",
                &self.serial,
                "shell",
                "input",
                "swipe",
                &x1.to_string(),
                &y1.to_string(),
                &x2.to_string(),
                &y2.to_string(),
                &duration_ms.to_string(),
            ])
            .status()
            .map_err(|e| format!("adb swipe failed: {e}"))?;
        if status.success() {
            Ok(())
        } else {
            Err("adb swipe non-zero".into())
        }
    }

    /// Type text.
    pub fn input_text(&self, text: &str) -> Result<(), String> {
        // Replace spaces with %s for adb
        let escaped = text.replace(' ', "%s");
        let status = Command::new("adb")
            .args(["-s", &self.serial, "shell", "input", "text", &escaped])
            .status()
            .map_err(|e| format!("adb text failed: {e}"))?;
        if status.success() {
            Ok(())
        } else {
            Err("adb text non-zero".into())
        }
    }

    /// Press back button.
    pub fn back(&self) -> Result<(), String> {
        let status = Command::new("adb")
            .args(["-s", &self.serial, "shell", "input", "keyevent", "4"])
            .status()
            .map_err(|e| format!("adb back failed: {e}"))?;
        if status.success() {
            Ok(())
        } else {
            Err("adb back non-zero".into())
        }
    }

    /// Press home button.
    pub fn home(&self) -> Result<(), String> {
        let status = Command::new("adb")
            .args(["-s", &self.serial, "shell", "input", "keyevent", "3"])
            .status()
            .map_err(|e| format!("adb home failed: {e}"))?;
        if status.success() {
            Ok(())
        } else {
            Err("adb home non-zero".into())
        }
    }

    /// Open a URL in the default browser.
    pub fn open_url(&self, url: &str) -> Result<(), String> {
        let status = Command::new("adb")
            .args([
                "-s",
                &self.serial,
                "shell",
                "am",
                "start",
                "-a",
                "android.intent.action.VIEW",
                "-d",
                url,
            ])
            .status()
            .map_err(|e| format!("adb open_url failed: {e}"))?;
        if status.success() {
            Ok(())
        } else {
            Err("adb open_url non-zero".into())
        }
    }

    /// Search YouTube for a query — uses the YouTube deep link.
    pub fn search_youtube(&self, query: &str) -> Result<(), String> {
        let encoded = query.replace(' ', "+");
        let url = format!("https://www.youtube.com/results?search_query={encoded}");
        self.open_url(&url)
    }

    /// Search the web via Google.
    pub fn search_web(&self, query: &str) -> Result<(), String> {
        let encoded = query.replace(' ', "+");
        let url = format!("https://www.google.com/search?q={encoded}");
        self.open_url(&url)
    }

    /// Launch a specific app by package name.
    pub fn launch_app(&self, package: &str) -> Result<(), String> {
        let status = Command::new("adb")
            .args([
                "-s",
                &self.serial,
                "shell",
                "monkey",
                "-p",
                package,
                "-c",
                "android.intent.category.LAUNCHER",
                "1",
            ])
            .status()
            .map_err(|e| format!("adb launch_app failed: {e}"))?;
        if status.success() {
            Ok(())
        } else {
            Err("adb launch_app non-zero".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adb_device_construction() {
        let dev = AdbDevice::new("test_serial");
        assert_eq!(dev.serial, "test_serial");
    }
}
