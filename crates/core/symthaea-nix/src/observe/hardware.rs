// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hardware and resource observation.
//!
//! Reads from `/proc/cpuinfo`, `/proc/meminfo`, `lspci`, and `df` to build
//! a structured snapshot of the system's hardware state. This feeds the
//! world model's understanding of resource constraints and availability.

use std::process::Command;
use tracing::debug;

/// Observes hardware state: CPU, memory, disk, GPU, and peripheral status.
pub struct HardwareObserver;

/// Aggregate hardware information for the system.
#[derive(Debug, Clone, Default)]
pub struct HardwareInfo {
    /// CPU model name (e.g. "AMD Ryzen 9 7950X").
    pub cpu_model: String,
    /// Number of logical CPU cores.
    pub cpu_cores: usize,
    /// Total physical memory in MiB.
    pub memory_total_mb: u64,
    /// Available memory in MiB.
    pub memory_available_mb: u64,
    /// Detected GPUs.
    pub gpus: Vec<GpuInfo>,
    /// Mounted disks/filesystems.
    pub disks: Vec<DiskInfo>,
    /// CPU load averages [1min, 5min, 15min] from /proc/loadavg.
    pub load_average: [f64; 3],
    /// Total swap in MiB.
    pub swap_total_mb: u64,
    /// Used swap in MiB (total - free).
    pub swap_used_mb: u64,
}

/// Information about a detected GPU.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuInfo {
    /// GPU name/model (e.g. "NVIDIA GeForce RTX 4090").
    pub name: String,
    /// Kernel driver in use, if detectable.
    pub driver: Option<String>,
}

/// Information about a mounted disk or filesystem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiskInfo {
    /// Device path (e.g. "/dev/nvme0n1p2").
    pub device: String,
    /// Mount point (e.g. "/", "/home").
    pub mount_point: String,
    /// Total size in bytes.
    pub total_bytes: u64,
    /// Used size in bytes.
    pub used_bytes: u64,
}

impl HardwareObserver {
    /// Probe the system for hardware information.
    ///
    /// Reads from:
    /// - `/proc/cpuinfo` for CPU model and core count
    /// - `/proc/meminfo` for memory totals
    /// - `lspci` for GPU detection
    /// - `df` for disk usage
    pub fn probe() -> Result<HardwareInfo, std::io::Error> {
        let cpuinfo = std::fs::read_to_string("/proc/cpuinfo")?;
        let meminfo = std::fs::read_to_string("/proc/meminfo")?;

        let (cpu_model, cpu_cores) = Self::parse_cpuinfo(&cpuinfo);
        let (memory_total_mb, memory_available_mb) = Self::parse_meminfo(&meminfo)?;
        let (swap_total_mb, swap_used_mb) = Self::parse_swap(&meminfo);

        // Load average from /proc/loadavg
        let load_average = std::fs::read_to_string("/proc/loadavg")
            .ok()
            .map(|s| Self::parse_loadavg(&s))
            .unwrap_or([0.0; 3]);

        // GPU detection via lspci (may not be available on all systems)
        let gpus = match Command::new("lspci").args(["-mm", "-nn"]).output() {
            Ok(output) if output.status.success() => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                Self::parse_lspci_gpus(&stdout)
            }
            _ => Vec::new(),
        };

        // Disk usage via df
        let disks = match Command::new("df")
            .args(["--output=source,target,size,used", "-B1", "--no-sync"])
            .output()
        {
            Ok(output) if output.status.success() => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                Self::parse_df(&stdout)
            }
            _ => Vec::new(),
        };

        Ok(HardwareInfo {
            cpu_model,
            cpu_cores,
            memory_total_mb,
            memory_available_mb,
            gpus,
            disks,
            load_average,
            swap_total_mb,
            swap_used_mb,
        })
    }

    // ---- Parsing helpers ----

    /// Parse `/proc/cpuinfo` to extract the CPU model name and core count.
    ///
    /// Looks for lines like:
    /// ```text
    /// model name  : AMD Ryzen 9 7950X 16-Core Processor
    /// processor   : 0
    /// ```
    pub fn parse_cpuinfo(content: &str) -> (String, usize) {
        let mut model = String::new();
        let mut core_count: usize = 0;

        for line in content.lines() {
            if line.starts_with("model name") && model.is_empty() {
                if let Some((_, value)) = line.split_once(':') {
                    model = value.trim().to_string();
                }
            }
            if line.starts_with("processor") {
                core_count += 1;
            }
        }

        (model, core_count)
    }

    /// Parse `/proc/meminfo` to extract total and available memory in MiB.
    ///
    /// Looks for:
    /// ```text
    /// MemTotal:       32768000 kB
    /// MemAvailable:   24576000 kB
    /// ```
    pub fn parse_meminfo(content: &str) -> Result<(u64, u64), std::io::Error> {
        let mut total_kb: u64 = 0;
        let mut available_kb: u64 = 0;

        for line in content.lines() {
            if line.starts_with("MemTotal:") {
                total_kb = Self::parse_meminfo_value(line)?;
            } else if line.starts_with("MemAvailable:") {
                available_kb = Self::parse_meminfo_value(line)?;
            }
        }

        Ok((total_kb / 1024, available_kb / 1024))
    }

    /// Parse a single meminfo line value in kB.
    fn parse_meminfo_value(line: &str) -> Result<u64, std::io::Error> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
            parts[1].parse::<u64>().map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("failed to parse meminfo value '{}': {}", parts[1], e),
                )
            })
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unexpected meminfo line format: {line}"),
            ))
        }
    }

    /// Parse `lspci -mm -nn` output to find VGA/3D controllers (GPUs).
    ///
    /// Lines look like:
    /// ```text
    /// 01:00.0 "VGA compatible controller [0300]" "NVIDIA Corporation [10de]" "GA102 [GeForce RTX 3090] [2204]" ...
    /// ```
    pub fn parse_lspci_gpus(output: &str) -> Vec<GpuInfo> {
        let mut gpus = Vec::new();

        for line in output.lines() {
            let trimmed = line.trim();
            // Look for VGA compatible controller or 3D controller
            if trimmed.contains("VGA compatible controller")
                || trimmed.contains("3D controller")
                || trimmed.contains("[0300]")
                || trimmed.contains("[0302]")
            {
                // Extract the quoted segments
                let quoted: Vec<&str> = Self::extract_quoted_segments(trimmed);
                // Typically: [class, vendor, device, ...]
                let name = if quoted.len() >= 3 {
                    format!("{} {}", quoted[1], quoted[2])
                } else if quoted.len() >= 2 {
                    quoted[1].to_string()
                } else {
                    trimmed.to_string()
                };

                gpus.push(GpuInfo {
                    name,
                    driver: None, // Would need `lspci -k` for driver info
                });
            }
        }

        gpus
    }

    /// Extract double-quoted segments from a line.
    fn extract_quoted_segments(line: &str) -> Vec<&str> {
        let mut segments = Vec::new();
        let mut rest = line;
        while let Some(start) = rest.find('"') {
            rest = &rest[start + 1..];
            if let Some(end) = rest.find('"') {
                segments.push(&rest[..end]);
                rest = &rest[end + 1..];
            } else {
                break;
            }
        }
        segments
    }

    /// Parse `/proc/loadavg` to extract 1min, 5min, 15min load averages.
    ///
    /// Format: `0.42 0.35 0.31 1/234 5678`
    pub fn parse_loadavg(content: &str) -> [f64; 3] {
        let parts: Vec<&str> = content.split_whitespace().collect();
        if parts.len() >= 3 {
            [
                parts[0].parse().unwrap_or(0.0),
                parts[1].parse().unwrap_or(0.0),
                parts[2].parse().unwrap_or(0.0),
            ]
        } else {
            [0.0; 3]
        }
    }

    /// Parse swap info from `/proc/meminfo`.
    ///
    /// Returns (swap_total_mb, swap_used_mb).
    pub fn parse_swap(meminfo_content: &str) -> (u64, u64) {
        let mut total_kb: u64 = 0;
        let mut free_kb: u64 = 0;

        for line in meminfo_content.lines() {
            if line.starts_with("SwapTotal:") {
                match Self::parse_meminfo_value(line) {
                    Ok(v) => total_kb = v,
                    Err(e) => debug!(line, error = %e, "Failed to parse SwapTotal"),
                }
            } else if line.starts_with("SwapFree:") {
                match Self::parse_meminfo_value(line) {
                    Ok(v) => free_kb = v,
                    Err(e) => debug!(line, error = %e, "Failed to parse SwapFree"),
                }
            }
        }

        let total_mb = total_kb / 1024;
        let used_mb = total_kb.saturating_sub(free_kb) / 1024;
        (total_mb, used_mb)
    }

    /// Parse `df --output=source,target,size,used -B1` output.
    ///
    /// Format:
    /// ```text
    /// Filesystem     Mounted on          1B-blocks         Used
    /// /dev/nvme0n1p2 /                500107862016 250053931008
    /// tmpfs          /run                16777216     8388608
    /// ```
    pub fn parse_df(output: &str) -> Vec<DiskInfo> {
        let mut disks = Vec::new();

        for (i, line) in output.lines().enumerate() {
            // Skip the header line
            if i == 0 {
                continue;
            }

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() < 4 {
                continue;
            }

            // Skip pseudo-filesystems
            let device = parts[0];
            if device == "tmpfs"
                || device == "devtmpfs"
                || device == "none"
                || device == "overlay"
                || device == "efivarfs"
            {
                continue;
            }

            let mount_point = parts[1].to_string();
            let total_bytes = parts[2].parse::<u64>().unwrap_or(0);
            let used_bytes = parts[3].parse::<u64>().unwrap_or(0);

            disks.push(DiskInfo {
                device: device.to_string(),
                mount_point,
                total_bytes,
                used_bytes,
            });
        }

        disks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MOCK_CPUINFO: &str = "\
processor\t: 0
vendor_id\t: AuthenticAMD
model name\t: AMD Ryzen 9 7950X 16-Core Processor
cpu MHz\t\t: 4500.000

processor\t: 1
vendor_id\t: AuthenticAMD
model name\t: AMD Ryzen 9 7950X 16-Core Processor
cpu MHz\t\t: 4500.000

processor\t: 2
vendor_id\t: AuthenticAMD
model name\t: AMD Ryzen 9 7950X 16-Core Processor
cpu MHz\t\t: 4500.000
";

    #[test]
    fn test_parse_cpuinfo() {
        let (model, cores) = HardwareObserver::parse_cpuinfo(MOCK_CPUINFO);
        assert_eq!(model, "AMD Ryzen 9 7950X 16-Core Processor");
        assert_eq!(cores, 3); // 3 processor entries in mock
    }

    #[test]
    fn test_parse_cpuinfo_empty() {
        let (model, cores) = HardwareObserver::parse_cpuinfo("");
        assert_eq!(model, "");
        assert_eq!(cores, 0);
    }

    const MOCK_MEMINFO: &str = "\
MemTotal:       32768000 kB
MemFree:         8192000 kB
MemAvailable:   24576000 kB
Buffers:         1024000 kB
Cached:         16384000 kB
SwapTotal:       8388608 kB
SwapFree:        8388608 kB
";

    #[test]
    fn test_parse_meminfo() {
        let (total_mb, available_mb) = HardwareObserver::parse_meminfo(MOCK_MEMINFO).unwrap();
        assert_eq!(total_mb, 32768000 / 1024); // 32000 MiB
        assert_eq!(available_mb, 24576000 / 1024); // 24000 MiB
    }

    #[test]
    fn test_parse_meminfo_missing_available() {
        let partial = "MemTotal:       16384000 kB\n";
        let (total_mb, available_mb) = HardwareObserver::parse_meminfo(partial).unwrap();
        assert_eq!(total_mb, 16384000 / 1024);
        assert_eq!(available_mb, 0); // Not found, defaults to 0
    }

    const MOCK_LSPCI: &str = r#"
01:00.0 "VGA compatible controller [0300]" "NVIDIA Corporation [10de]" "GA102 [GeForce RTX 3090] [2204]" "eVga.com. Corp. [3842]" "FTW3 ULTRA GAMING [3897]"
02:00.0 "Ethernet controller [0200]" "Intel Corporation [8086]" "I225-V [15f3]" "" ""
"#;

    #[test]
    fn test_parse_lspci_gpus() {
        let gpus = HardwareObserver::parse_lspci_gpus(MOCK_LSPCI);
        assert_eq!(gpus.len(), 1);
        assert!(gpus[0].name.contains("NVIDIA"));
        assert!(gpus[0].name.contains("GeForce RTX 3090"));
    }

    #[test]
    fn test_parse_lspci_no_gpu() {
        let output = r#"00:00.0 "Host bridge [0600]" "Intel [8086]" "Device [1234]" "" """#;
        let gpus = HardwareObserver::parse_lspci_gpus(output);
        assert!(gpus.is_empty());
    }

    const MOCK_DF: &str = "\
Filesystem     Mounted on          1B-blocks         Used
/dev/nvme0n1p2 /                500107862016 250053931008
/dev/nvme0n1p1 /boot               536870912   104857600
tmpfs          /run                16777216     8388608
";

    #[test]
    fn test_parse_df() {
        let disks = HardwareObserver::parse_df(MOCK_DF);
        // tmpfs should be skipped
        assert_eq!(disks.len(), 2);

        assert_eq!(disks[0].device, "/dev/nvme0n1p2");
        assert_eq!(disks[0].mount_point, "/");
        assert_eq!(disks[0].total_bytes, 500107862016);
        assert_eq!(disks[0].used_bytes, 250053931008);

        assert_eq!(disks[1].device, "/dev/nvme0n1p1");
        assert_eq!(disks[1].mount_point, "/boot");
    }

    #[test]
    fn test_parse_df_empty() {
        let disks = HardwareObserver::parse_df("Filesystem Mounted on 1B-blocks Used\n");
        assert!(disks.is_empty());
    }

    #[test]
    fn test_parse_loadavg() {
        let load = HardwareObserver::parse_loadavg("0.42 0.35 0.31 1/234 5678\n");
        assert!((load[0] - 0.42).abs() < 1e-6);
        assert!((load[1] - 0.35).abs() < 1e-6);
        assert!((load[2] - 0.31).abs() < 1e-6);
    }

    #[test]
    fn test_parse_loadavg_empty() {
        let load = HardwareObserver::parse_loadavg("");
        assert_eq!(load, [0.0; 3]);
    }

    #[test]
    fn test_parse_swap() {
        let (total, used) = HardwareObserver::parse_swap(MOCK_MEMINFO);
        assert_eq!(total, 8388608 / 1024); // 8192 MiB
        assert_eq!(used, 0); // SwapFree == SwapTotal
    }

    #[test]
    fn test_parse_swap_partial_usage() {
        let meminfo = "SwapTotal:       8388608 kB\nSwapFree:        4194304 kB\n";
        let (total, used) = HardwareObserver::parse_swap(meminfo);
        assert_eq!(total, 8388608 / 1024);
        assert_eq!(used, 4194304 / 1024);
    }

    #[test]
    fn test_parse_swap_no_swap() {
        let meminfo = "MemTotal:       32768000 kB\n";
        let (total, used) = HardwareObserver::parse_swap(meminfo);
        assert_eq!(total, 0);
        assert_eq!(used, 0);
    }

    #[test]
    fn test_parse_cpuinfo_no_model_name() {
        let content = "processor\t: 0\nvendor_id\t: GenuineIntel\n";
        let (model, cores) = HardwareObserver::parse_cpuinfo(content);
        assert_eq!(model, "");
        assert_eq!(cores, 1);
    }

    #[test]
    fn test_parse_lspci_3d_controller() {
        let output =
            r#"06:00.0 "3D controller [0302]" "NVIDIA Corporation [10de]" "A100 [2236]" "" """#;
        let gpus = HardwareObserver::parse_lspci_gpus(output);
        assert_eq!(gpus.len(), 1);
        assert!(gpus[0].name.contains("NVIDIA"));
    }

    #[test]
    fn test_parse_lspci_multiple_gpus() {
        let output = "\
01:00.0 \"VGA compatible controller [0300]\" \"NVIDIA [10de]\" \"RTX 3090 [2204]\" \"\" \"\"\n\
02:00.0 \"VGA compatible controller [0300]\" \"AMD [1002]\" \"RX 7900 [744c]\" \"\" \"\"";
        let gpus = HardwareObserver::parse_lspci_gpus(output);
        assert_eq!(gpus.len(), 2);
        assert!(gpus[0].name.contains("NVIDIA"));
        assert!(gpus[1].name.contains("AMD"));
    }

    #[test]
    fn test_parse_df_skips_pseudo_filesystems() {
        let output = "\
Filesystem Mounted on 1B-blocks Used\n\
devtmpfs /dev 16777216 0\n\
none /dev/shm 16777216 0\n\
overlay /run/bla 16777216 0\n\
efivarfs /sys/firmware 131072 65536\n\
/dev/sda1 / 500000000000 250000000000";
        let disks = HardwareObserver::parse_df(output);
        assert_eq!(disks.len(), 1);
        assert_eq!(disks[0].device, "/dev/sda1");
    }

    #[test]
    fn test_parse_df_short_line_skipped() {
        let output = "Filesystem Mounted on 1B-blocks Used\nshort line\n/dev/sda1 / 500 250";
        let disks = HardwareObserver::parse_df(output);
        assert_eq!(disks.len(), 1);
    }

    #[test]
    fn test_parse_meminfo_malformed_value() {
        let bad = "MemTotal:       not_a_number kB\nMemAvailable: 1024 kB\n";
        let result = HardwareObserver::parse_meminfo(bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_meminfo_short_line() {
        let short = "MemTotal:\n";
        let result = HardwareObserver::parse_meminfo(short);
        // Short line should error (no second token)
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_loadavg_malformed() {
        let load = HardwareObserver::parse_loadavg("bad data here");
        assert_eq!(load, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_parse_swap_malformed_values() {
        // Malformed swap values should gracefully return 0 (with debug logging)
        let meminfo = "SwapTotal:       not_a_number kB\nSwapFree:        also_bad kB\n";
        let (total, used) = HardwareObserver::parse_swap(meminfo);
        assert_eq!(total, 0);
        assert_eq!(used, 0);
    }

    #[test]
    fn test_parse_swap_short_lines() {
        // Short lines (missing value) should gracefully return 0
        let meminfo = "SwapTotal:\nSwapFree:\n";
        let (total, used) = HardwareObserver::parse_swap(meminfo);
        assert_eq!(total, 0);
        assert_eq!(used, 0);
    }

    #[test]
    fn test_hardware_info_default() {
        let info = HardwareInfo::default();
        assert_eq!(info.cpu_cores, 0);
        assert_eq!(info.memory_total_mb, 0);
        assert!(info.gpus.is_empty());
        assert!(info.disks.is_empty());
    }
}
