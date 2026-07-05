// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hardware detection and NixOS configuration generation.
//!
//! Since WASM cannot call browser APIs directly, the JavaScript layer collects
//! hardware information via `navigator.hardwareConcurrency`, `navigator.deviceMemory`,
//! WebGPU adapter info, etc., then passes the raw data into Rust for processing.
//!
//! This module:
//! 1. Deserializes browser-collected hardware data into [`HardwareProfile`]
//! 2. Maps the profile to NixOS configuration recommendations ([`NixHardwareConfig`])
//! 3. Generates a valid `hardware-configuration.nix` string

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// HardwareProfile — browser-collected capabilities
// ---------------------------------------------------------------------------

/// Raw hardware capabilities collected from browser APIs by the JS layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HardwareProfile {
    /// Logical CPU cores (from `navigator.hardwareConcurrency`).
    #[serde(default = "default_cpu_cores")]
    pub cpu_cores: u32,

    /// Device memory in GB (from `navigator.deviceMemory`).
    /// May be 0.0 if the API is unavailable (Firefox, Safari).
    #[serde(default)]
    pub device_memory_gb: f32,

    /// GPU vendor string (from WebGPU `GPUAdapterInfo.vendor`).
    #[serde(default)]
    pub gpu_vendor: String,

    /// GPU architecture string (from WebGPU `GPUAdapterInfo.architecture`).
    #[serde(default)]
    pub gpu_architecture: String,

    /// GPU description string (from WebGPU `GPUAdapterInfo.description`).
    #[serde(default)]
    pub gpu_description: String,

    /// Whether `navigator.gpu` is present.
    #[serde(default)]
    pub has_webgpu: bool,

    /// Whether `navigator.usb` is present.
    #[serde(default)]
    pub has_webusb: bool,

    /// Storage quota in bytes (from `navigator.storage.estimate().quota`).
    #[serde(default)]
    pub storage_quota_bytes: u64,

    /// Platform string (from `navigator.userAgentData.platform` or `navigator.platform`).
    #[serde(default)]
    pub platform: String,

    /// Whether the user agent reports a mobile device.
    #[serde(default)]
    pub is_mobile: bool,

    /// Browser name (from `navigator.userAgentData.brands` or user-agent parsing).
    #[serde(default)]
    pub browser_name: String,
}

fn default_cpu_cores() -> u32 {
    1
}

impl Default for HardwareProfile {
    fn default() -> Self {
        Self {
            cpu_cores: 1,
            device_memory_gb: 0.0,
            gpu_vendor: String::new(),
            gpu_architecture: String::new(),
            gpu_description: String::new(),
            has_webgpu: false,
            has_webusb: false,
            storage_quota_bytes: 0,
            platform: String::new(),
            is_mobile: false,
            browser_name: String::new(),
        }
    }
}

impl HardwareProfile {
    /// Classify the GPU vendor into a normalized driver name.
    pub fn gpu_driver_class(&self) -> GpuDriverClass {
        let vendor = self.gpu_vendor.to_lowercase();
        let desc = self.gpu_description.to_lowercase();
        let arch = self.gpu_architecture.to_lowercase();

        let combined = format!("{vendor} {desc} {arch}");

        if combined.contains("nvidia") || combined.contains("geforce") || combined.contains("rtx") {
            GpuDriverClass::Nvidia
        } else if combined.contains("amd")
            || combined.contains("radeon")
            || combined.contains("amdgpu")
        {
            GpuDriverClass::Amd
        } else if combined.contains("intel")
            || combined.contains("iris")
            || combined.contains("uhd")
        {
            GpuDriverClass::Intel
        } else if combined.contains("apple") || combined.contains("m1") || combined.contains("m2") {
            GpuDriverClass::Apple
        } else if combined.contains("qualcomm") || combined.contains("adreno") {
            GpuDriverClass::Qualcomm
        } else if combined.contains("mali") || combined.contains("arm") {
            GpuDriverClass::Arm
        } else if !vendor.is_empty() {
            GpuDriverClass::Unknown
        } else {
            GpuDriverClass::None
        }
    }

    /// Estimate usable RAM in GB. `deviceMemory` is capped by browsers
    /// (e.g. Chrome caps at 8), so we take the reported value as a floor.
    pub fn estimated_ram_gb(&self) -> f32 {
        if self.device_memory_gb > 0.0 {
            self.device_memory_gb
        } else {
            // Conservative fallback: estimate from CPU cores.
            // Single-core devices likely have 1-2 GB; 8-core desktops ~16 GB.
            (self.cpu_cores as f32 * 2.0).clamp(1.0, 64.0)
        }
    }

    /// Estimate storage in GB.
    pub fn storage_gb(&self) -> f64 {
        self.storage_quota_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

/// Normalized GPU driver classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuDriverClass {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Qualcomm,
    Arm,
    Unknown,
    None,
}

// ---------------------------------------------------------------------------
// NixHardwareConfig — generated NixOS recommendations
// ---------------------------------------------------------------------------

/// NixOS hardware configuration recommendations derived from [`HardwareProfile`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NixHardwareConfig {
    /// Recommended filesystem type.
    pub filesystem: String,

    /// GPU driver module to enable.
    pub gpu_driver: String,

    /// Whether to enable CUDA support (NVIDIA).
    pub enable_cuda: bool,

    /// Whether to enable ROCm support (AMD).
    pub enable_rocm: bool,

    /// Recommended swap size in GB.
    pub swap_size_gb: u32,

    /// Whether an NPU (neural processing unit) is likely present.
    pub has_npu: bool,

    /// Recommended Symthaea substrate type name (maps to `SubstrateType` enum).
    pub recommended_substrate: String,

    /// Recommended number of CfC neurons (scales with CPU cores and RAM).
    pub recommended_neurons: u32,

    /// Recommended Spore HDC dimension.
    pub recommended_hdc_dim: u32,

    /// The generated `hardware-configuration.nix` content.
    pub nix_hardware_config: String,
}

// ---------------------------------------------------------------------------
// ProbeResult — combined output
// ---------------------------------------------------------------------------

/// Combined result from hardware probing: the raw profile plus NixOS recommendations.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProbeResult {
    pub profile: HardwareProfile,
    pub nix_config: NixHardwareConfig,
}

// ---------------------------------------------------------------------------
// Generation logic
// ---------------------------------------------------------------------------

impl NixHardwareConfig {
    /// Generate a [`NixHardwareConfig`] from a [`HardwareProfile`].
    pub fn from_profile(profile: &HardwareProfile) -> Self {
        let gpu_class = profile.gpu_driver_class();
        let ram_gb = profile.estimated_ram_gb();
        let storage_gb = profile.storage_gb();

        // --- Filesystem ---
        let filesystem = if storage_gb >= 500.0 {
            "zfs".to_string()
        } else if storage_gb >= 100.0 {
            "btrfs".to_string()
        } else {
            "ext4".to_string()
        };

        // --- GPU driver ---
        let gpu_driver = match gpu_class {
            GpuDriverClass::Nvidia => "nvidia".to_string(),
            GpuDriverClass::Amd => "amdgpu".to_string(),
            GpuDriverClass::Intel => "intel".to_string(),
            GpuDriverClass::Apple => "apple".to_string(),
            GpuDriverClass::Qualcomm => "qualcomm".to_string(),
            GpuDriverClass::Arm => "mali".to_string(),
            GpuDriverClass::Unknown | GpuDriverClass::None => "generic".to_string(),
        };

        // --- Compute acceleration ---
        let enable_cuda = gpu_class == GpuDriverClass::Nvidia;
        let enable_rocm = gpu_class == GpuDriverClass::Amd;

        // --- Swap: 2x RAM for small systems, 1x for large ---
        let swap_size_gb = if ram_gb <= 4.0 {
            (ram_gb * 2.0) as u32
        } else if ram_gb <= 16.0 {
            ram_gb as u32
        } else {
            (ram_gb * 0.5) as u32
        };

        // --- NPU detection heuristic ---
        // Recent Apple Silicon, Qualcomm Snapdragon, Intel Meteor Lake have NPUs.
        let has_npu = gpu_class == GpuDriverClass::Apple
            || gpu_class == GpuDriverClass::Qualcomm
            || (gpu_class == GpuDriverClass::Intel
                && profile.gpu_description.to_lowercase().contains("meteor"));

        // --- Substrate recommendation ---
        let recommended_substrate = if has_npu {
            "NeuromorphicChip".to_string()
        } else {
            // CUDA/ROCm, mobile, and the default case all currently
            // recommend the same substrate -- no distinct heuristic yet.
            "SiliconDigital".to_string()
        };

        // --- Neuron count: scale with resources ---
        // Base: 64 neurons. Double for each: 8+ cores, 16+ GB RAM, GPU compute.
        let mut neurons: u32 = 64;
        if profile.cpu_cores >= 8 {
            neurons *= 2;
        }
        if ram_gb >= 16.0 {
            neurons *= 2;
        }
        if enable_cuda || enable_rocm {
            neurons *= 2;
        }
        let recommended_neurons = neurons.min(1024);

        // --- HDC dimension: scale with RAM ---
        let recommended_hdc_dim = if ram_gb >= 8.0 {
            16_384
        } else if ram_gb >= 4.0 {
            8_192
        } else {
            4_096
        };

        // --- Generate nix config ---
        let nix_hardware_config = generate_hardware_nix(
            profile,
            &gpu_driver,
            &filesystem,
            swap_size_gb,
            enable_cuda,
            enable_rocm,
        );

        Self {
            filesystem,
            gpu_driver,
            enable_cuda,
            enable_rocm,
            swap_size_gb,
            has_npu,
            recommended_substrate,
            recommended_neurons,
            recommended_hdc_dim,
            nix_hardware_config,
        }
    }
}

/// Produce a hardware probe result from browser-collected data.
pub fn probe(profile: HardwareProfile) -> ProbeResult {
    let nix_config = NixHardwareConfig::from_profile(&profile);
    ProbeResult {
        profile,
        nix_config,
    }
}

// ---------------------------------------------------------------------------
// hardware-configuration.nix generation
// ---------------------------------------------------------------------------

fn generate_hardware_nix(
    profile: &HardwareProfile,
    gpu_driver: &str,
    filesystem: &str,
    swap_size_gb: u32,
    enable_cuda: bool,
    enable_rocm: bool,
) -> String {
    let mut nix = String::with_capacity(2048);

    nix.push_str(
        "# Auto-generated by Symthaea Spore hardware probe.\n\
         # Review and adjust before applying to a real NixOS system.\n\
         # Source hardware: browser-detected capabilities (approximate).\n\
         { config, lib, pkgs, modulesPath, ... }:\n\n\
         {\n\
         \x20 imports = [\n\
         \x20   (modulesPath + \"/installer/scan/not-detected.nix\")\n\
         \x20 ];\n\n",
    );

    // --- Boot ---
    nix.push_str(
        "  # Boot\n\
         \x20 boot.initrd.availableKernelModules = [\n\
         \x20   \"xhci_pci\" \"ahci\" \"nvme\" \"usbhid\" \"sd_mod\"\n\
         \x20 ];\n\
         \x20 boot.kernelModules = [ \"kvm-intel\" \"kvm-amd\" ];\n\n",
    );

    // --- Filesystem ---
    nix.push_str(&format!(
        "  # Filesystem: {filesystem}\n\
         \x20 fileSystems.\"/\" = {{\n\
         \x20   device = \"/dev/disk/by-label/nixos\";\n\
         \x20   fsType = \"{filesystem}\";\n\
         \x20 }};\n\n",
    ));

    if filesystem == "btrfs" {
        nix.push_str("  fileSystems.\"/\".options = [ \"compress=zstd\" \"noatime\" ];\n\n");
    }

    // --- Swap ---
    nix.push_str(&format!(
        "  # Swap ({swap_size_gb} GB)\n\
         \x20 swapDevices = [\n\
         \x20   {{ device = \"/dev/disk/by-label/swap\"; size = {swap_gb_mb}; }}\n\
         \x20 ];\n\n",
        swap_gb_mb = swap_size_gb as u64 * 1024,
    ));

    // --- GPU ---
    nix.push_str("  # GPU\n");
    match gpu_driver {
        "nvidia" => {
            nix.push_str(
                "  services.xserver.videoDrivers = [ \"nvidia\" ];\n\
                 \x20 hardware.nvidia = {\n\
                 \x20   modesetting.enable = true;\n\
                 \x20   powerManagement.enable = true;\n\
                 \x20   open = false;\n\
                 \x20   nvidiaSettings = true;\n\
                 \x20 };\n",
            );
        }
        "amdgpu" => {
            nix.push_str(
                "  services.xserver.videoDrivers = [ \"amdgpu\" ];\n\
                 \x20 hardware.amdgpu.initrd.enable = true;\n",
            );
        }
        "intel" => {
            nix.push_str(
                "  services.xserver.videoDrivers = [ \"modesetting\" ];\n\
                 \x20 hardware.graphics.extraPackages = with pkgs; [\n\
                 \x20   intel-media-driver\n\
                 \x20   intel-compute-runtime\n\
                 \x20 ];\n",
            );
        }
        _ => {
            nix.push_str("  services.xserver.videoDrivers = [ \"modesetting\" ];\n");
        }
    }
    nix.push('\n');

    // --- Compute acceleration ---
    if enable_cuda {
        nix.push_str(
            "  # CUDA\n\
             \x20 hardware.graphics.enable = true;\n\
             \x20 environment.systemPackages = with pkgs; [\n\
             \x20   cudaPackages.cudatoolkit\n\
             \x20   cudaPackages.cudnn\n\
             \x20 ];\n\n",
        );
    }
    if enable_rocm {
        nix.push_str(
            "  # ROCm\n\
             \x20 hardware.graphics.enable = true;\n\
             \x20 systemd.tmpfiles.rules = [\n\
             \x20   \"L+ /opt/rocm/hip - - - - ${pkgs.rocmPackages.clr}\"\n\
             \x20 ];\n\n",
        );
    }

    // --- Networking ---
    nix.push_str("  # Networking\n  networking.useDHCP = lib.mkDefault true;\n\n");

    // --- Hardware quirks ---
    let cores = profile.cpu_cores;
    nix.push_str(&format!(
        "  # Detected: {cores} logical cores, ~{ram:.1} GB RAM\n\
         \x20 nix.settings.max-jobs = {max_jobs};\n\
         \x20 nix.settings.cores = {build_cores};\n\n",
        ram = profile.estimated_ram_gb(),
        max_jobs = (cores / 2).max(1),
        build_cores = cores.max(1),
    ));

    // --- Symthaea Spore recommended config (as comment) ---
    nix.push_str(&format!(
        "  # Symthaea Spore recommendations:\n\
         \x20 #   substrate    = \"{substrate}\"\n\
         \x20 #   hdc_dim      = {hdc_dim}\n\
         \x20 #   neurons      = {neurons}\n\
         \x20 #   target_hz    = {hz}\n",
        substrate = "SiliconDigital",
        hdc_dim = if profile.estimated_ram_gb() >= 8.0 {
            16_384
        } else {
            4_096
        },
        neurons = if cores >= 8 { 128 } else { 64 },
        hz = if cores >= 8 { 50 } else { 20 },
    ));

    nix.push_str("}\n");
    nix
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_profile() -> HardwareProfile {
        HardwareProfile {
            cpu_cores: 8,
            device_memory_gb: 16.0,
            gpu_vendor: "NVIDIA".into(),
            gpu_architecture: "Ampere".into(),
            gpu_description: "NVIDIA GeForce RTX 3080".into(),
            has_webgpu: true,
            has_webusb: false,
            storage_quota_bytes: 500 * 1024 * 1024 * 1024, // 500 GB
            platform: "Linux".into(),
            is_mobile: false,
            browser_name: "Chrome".into(),
        }
    }

    #[test]
    fn test_gpu_driver_nvidia() {
        let p = sample_profile();
        assert_eq!(p.gpu_driver_class(), GpuDriverClass::Nvidia);
    }

    #[test]
    fn test_gpu_driver_amd() {
        let mut p = sample_profile();
        p.gpu_vendor = "AMD".into();
        p.gpu_description = "Radeon RX 7900 XTX".into();
        assert_eq!(p.gpu_driver_class(), GpuDriverClass::Amd);
    }

    #[test]
    fn test_gpu_driver_intel() {
        let mut p = sample_profile();
        p.gpu_vendor = "Intel".into();
        p.gpu_description = "Intel UHD Graphics 770".into();
        p.gpu_architecture = String::new();
        assert_eq!(p.gpu_driver_class(), GpuDriverClass::Intel);
    }

    #[test]
    fn test_gpu_driver_apple() {
        let mut p = sample_profile();
        p.gpu_vendor = "Apple".into();
        p.gpu_description = "Apple M2 Pro".into();
        p.gpu_architecture = String::new();
        assert_eq!(p.gpu_driver_class(), GpuDriverClass::Apple);
    }

    #[test]
    fn test_gpu_driver_none() {
        let p = HardwareProfile::default();
        assert_eq!(p.gpu_driver_class(), GpuDriverClass::None);
    }

    #[test]
    fn test_estimated_ram_with_device_memory() {
        let p = sample_profile();
        assert!((p.estimated_ram_gb() - 16.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_estimated_ram_fallback() {
        let p = HardwareProfile {
            cpu_cores: 4,
            device_memory_gb: 0.0,
            ..Default::default()
        };
        assert!((p.estimated_ram_gb() - 8.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_storage_gb_conversion() {
        let p = HardwareProfile {
            storage_quota_bytes: 107_374_182_400, // 100 GB
            ..Default::default()
        };
        assert!((p.storage_gb() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_nix_config_nvidia() {
        let p = sample_profile();
        let config = NixHardwareConfig::from_profile(&p);
        assert_eq!(config.gpu_driver, "nvidia");
        assert!(config.enable_cuda);
        assert!(!config.enable_rocm);
        assert!(config.nix_hardware_config.contains("nvidia"));
        assert!(config.nix_hardware_config.contains("cudaPackages"));
    }

    #[test]
    fn test_nix_config_amd() {
        let mut p = sample_profile();
        p.gpu_vendor = "AMD".into();
        p.gpu_description = "Radeon RX 7900".into();
        let config = NixHardwareConfig::from_profile(&p);
        assert_eq!(config.gpu_driver, "amdgpu");
        assert!(config.enable_rocm);
        assert!(!config.enable_cuda);
        assert!(config.nix_hardware_config.contains("rocm"));
    }

    #[test]
    fn test_nix_config_filesystem_zfs() {
        let p = sample_profile(); // 500 GB
        let config = NixHardwareConfig::from_profile(&p);
        assert_eq!(config.filesystem, "zfs");
    }

    #[test]
    fn test_nix_config_filesystem_btrfs() {
        let mut p = sample_profile();
        p.storage_quota_bytes = 200 * 1024 * 1024 * 1024; // 200 GB
        let config = NixHardwareConfig::from_profile(&p);
        assert_eq!(config.filesystem, "btrfs");
    }

    #[test]
    fn test_nix_config_filesystem_ext4() {
        let mut p = sample_profile();
        p.storage_quota_bytes = 50 * 1024 * 1024 * 1024; // 50 GB
        let config = NixHardwareConfig::from_profile(&p);
        assert_eq!(config.filesystem, "ext4");
    }

    #[test]
    fn test_neuron_scaling_low_end() {
        let p = HardwareProfile {
            cpu_cores: 2,
            device_memory_gb: 2.0,
            ..Default::default()
        };
        let config = NixHardwareConfig::from_profile(&p);
        assert_eq!(config.recommended_neurons, 64);
        assert_eq!(config.recommended_hdc_dim, 4_096);
    }

    #[test]
    fn test_neuron_scaling_high_end() {
        let p = sample_profile(); // 8 cores, 16 GB, NVIDIA
        let config = NixHardwareConfig::from_profile(&p);
        // 64 * 2 (cores) * 2 (RAM) * 2 (CUDA) = 512
        assert_eq!(config.recommended_neurons, 512);
        assert_eq!(config.recommended_hdc_dim, 16_384);
    }

    #[test]
    fn test_swap_small_system() {
        let p = HardwareProfile {
            device_memory_gb: 2.0,
            ..Default::default()
        };
        let config = NixHardwareConfig::from_profile(&p);
        assert_eq!(config.swap_size_gb, 4); // 2x RAM
    }

    #[test]
    fn test_swap_large_system() {
        let p = HardwareProfile {
            device_memory_gb: 32.0,
            ..Default::default()
        };
        let config = NixHardwareConfig::from_profile(&p);
        assert_eq!(config.swap_size_gb, 16); // 0.5x RAM
    }

    #[test]
    fn test_npu_detection_apple() {
        let mut p = sample_profile();
        p.gpu_vendor = "Apple".into();
        p.gpu_description = "Apple M2 Pro".into();
        let config = NixHardwareConfig::from_profile(&p);
        assert!(config.has_npu);
        assert_eq!(config.recommended_substrate, "NeuromorphicChip");
    }

    #[test]
    fn test_probe_roundtrip() {
        let p = sample_profile();
        let result = probe(p.clone());
        assert_eq!(result.profile.cpu_cores, 8);
        assert!(!result.nix_config.nix_hardware_config.is_empty());
    }

    #[test]
    fn test_default_profile() {
        let p = HardwareProfile::default();
        let result = probe(p);
        // Should not panic, should produce valid config
        assert!(!result.nix_config.nix_hardware_config.is_empty());
        assert!(result.nix_config.nix_hardware_config.contains("nixos"));
    }

    #[test]
    fn test_generated_nix_is_valid_structure() {
        let p = sample_profile();
        let config = NixHardwareConfig::from_profile(&p);
        let nix = &config.nix_hardware_config;
        // Must have opening and closing braces
        assert!(nix.contains("{ config, lib, pkgs"));
        assert!(nix.ends_with("}\n"));
        // Must have imports
        assert!(nix.contains("imports ="));
        // Must have fileSystems
        assert!(nix.contains("fileSystems.\"/\""));
        // Must have swap
        assert!(nix.contains("swapDevices"));
    }

    #[test]
    fn test_mobile_profile() {
        let p = HardwareProfile {
            cpu_cores: 4,
            device_memory_gb: 4.0,
            gpu_vendor: "Qualcomm".into(),
            gpu_architecture: "Adreno".into(),
            gpu_description: "Adreno 730".into(),
            has_webgpu: false,
            has_webusb: false,
            storage_quota_bytes: 64 * 1024 * 1024 * 1024,
            platform: "Android".into(),
            is_mobile: true,
            browser_name: "Chrome".into(),
        };
        let config = NixHardwareConfig::from_profile(&p);
        assert!(config.has_npu); // Qualcomm has NPU
        assert_eq!(config.gpu_driver, "qualcomm");
        assert!(!config.enable_cuda);
        assert!(!config.enable_rocm);
    }

    #[test]
    fn test_serde_camel_case() {
        let p = sample_profile();
        let json = serde_json::to_string(&p).unwrap();
        assert!(json.contains("cpuCores"));
        assert!(json.contains("deviceMemoryGb"));
        assert!(json.contains("gpuVendor"));
        assert!(json.contains("hasWebgpu"));
        assert!(json.contains("storageQuotaBytes"));
        assert!(json.contains("isMobile"));
    }

    #[test]
    fn test_deserialize_partial_data() {
        // Browser may only provide partial data
        let json = r#"{"cpuCores": 4, "hasWebgpu": true}"#;
        let p: HardwareProfile = serde_json::from_str(json).unwrap();
        assert_eq!(p.cpu_cores, 4);
        assert!(p.has_webgpu);
        assert_eq!(p.device_memory_gb, 0.0);
        assert!(p.gpu_vendor.is_empty());
    }
}
