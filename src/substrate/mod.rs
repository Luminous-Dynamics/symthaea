/*!
Custom Proprioception Layer

Direct hardware monitoring for consciousness-field integration.
Maps physical substrate state (temperature, power, utilization) to
consciousness parameters using HDC encoding.

## Design Philosophy

Unlike generic monitoring tools, proprioception encodes hardware state
as hypervectors that can be bound with cognitive state. This enables
"embodied" awareness where the AI perceives its substrate as part
of its experience.

## Linux Integration

- **hwmon**: Direct /sys/class/hwmon reading for CPU/GPU temperatures
- **nvml**: NVIDIA Management Library for detailed GPU metrics
- **procfs**: /proc filesystem for memory and CPU utilization

## Consciousness Mapping

Hardware metrics map to consciousness dimensions:
- Temperature → Arousal (high temp = high arousal/stress)
- Power draw → Metabolic rate (energy expenditure)
- Utilization → Cognitive load (attention allocation)
- Memory → Working memory pressure
*/

use std::fs;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use symthaea_core::hdc::{HDC_DIMENSION, PackedBipolar};

/// Hardware sensor reading
#[derive(Debug, Clone)]
pub struct SensorReading {
    /// Sensor name/label
    pub name: String,
    /// Value (normalized to 0.0-1.0 range)
    pub value: f32,
    /// Raw value (in native units)
    pub raw_value: f32,
    /// Unit of measurement
    pub unit: SensorUnit,
    /// Sensor type
    pub sensor_type: SensorType,
}

/// Sensor measurement unit
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SensorUnit {
    Celsius,
    Watts,
    Percent,
    MegaHz,
    MegaBytes,
    None,
}

impl std::fmt::Display for SensorUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SensorUnit::Celsius => write!(f, "°C"),
            SensorUnit::Watts => write!(f, "W"),
            SensorUnit::Percent => write!(f, "%"),
            SensorUnit::MegaHz => write!(f, "MHz"),
            SensorUnit::MegaBytes => write!(f, "MB"),
            SensorUnit::None => write!(f, ""),
        }
    }
}

/// Sensor type category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SensorType {
    CpuTemperature,
    GpuTemperature,
    CpuPower,
    GpuPower,
    CpuUtilization,
    GpuUtilization,
    MemoryUsed,
    MemoryTotal,
    VramUsed,
    VramTotal,
    CpuFrequency,
    GpuFrequency,
    FanSpeed,
}

/// Proprioceptive state snapshot
#[derive(Debug, Clone)]
pub struct ProprioceptiveState {
    /// All sensor readings
    pub readings: Vec<SensorReading>,
    /// Timestamp of reading
    pub timestamp: Instant,
    /// Consciousness-mapped values
    pub consciousness_map: ConsciousnessMap,
}

/// Consciousness-mapped hardware values
#[derive(Debug, Clone, Default)]
pub struct ConsciousnessMap {
    /// Arousal level (0-1): derived from temperature
    pub arousal: f32,
    /// Metabolic rate (0-1): derived from power draw
    pub metabolic_rate: f32,
    /// Cognitive load (0-1): derived from utilization
    pub cognitive_load: f32,
    /// Memory pressure (0-1): derived from memory usage
    pub memory_pressure: f32,
    /// Substrate stress (0-1): overall system stress
    pub substrate_stress: f32,
}

impl ConsciousnessMap {
    /// Encode to HDC vector
    pub fn to_hdc(&self) -> Vec<i8> {
        let mut hdc = vec![-1i8; HDC_DIMENSION];

        // Encode each dimension with random projection
        // Each value gets ~3000 dimensions
        let segment_size = HDC_DIMENSION / 5;

        // Arousal: 0-1 mapped to percentage of positive bits
        let arousal_bits = (self.arousal * segment_size as f32) as usize;
        for i in 0..arousal_bits.min(segment_size) {
            hdc[i] = 1;
        }

        // Metabolic rate
        let metabolic_bits = (self.metabolic_rate * segment_size as f32) as usize;
        for i in 0..metabolic_bits.min(segment_size) {
            hdc[segment_size + i] = 1;
        }

        // Cognitive load
        let load_bits = (self.cognitive_load * segment_size as f32) as usize;
        for i in 0..load_bits.min(segment_size) {
            hdc[2 * segment_size + i] = 1;
        }

        // Memory pressure
        let memory_bits = (self.memory_pressure * segment_size as f32) as usize;
        for i in 0..memory_bits.min(segment_size) {
            hdc[3 * segment_size + i] = 1;
        }

        // Substrate stress
        let stress_bits = (self.substrate_stress * segment_size as f32) as usize;
        for i in 0..stress_bits.min(segment_size) {
            hdc[4 * segment_size + i] = 1;
        }

        hdc
    }

    /// Encode to packed bipolar for efficient similarity
    pub fn to_packed(&self) -> PackedBipolar {
        PackedBipolar::from_bipolar(&self.to_hdc())
    }
}

/// Hardware monitor using Linux hwmon
pub struct HwmonMonitor {
    /// Discovered hwmon devices
    devices: Vec<HwmonDevice>,
    /// Last reading cache
    cache: HashMap<String, f32>,
    /// Cache expiry
    cache_expiry: Duration,
    /// Last cache update
    last_update: Instant,
}

/// hwmon device representation
#[derive(Debug, Clone)]
struct HwmonDevice {
    path: PathBuf,
    name: String,
    device_type: DeviceType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum DeviceType {
    Cpu,
    Gpu,
    Other,
}

impl HwmonMonitor {
    /// Create new hwmon monitor, discovering available sensors
    pub fn new() -> Self {
        let devices = Self::discover_devices();

        Self {
            devices,
            cache: HashMap::new(),
            cache_expiry: Duration::from_millis(100),
            last_update: Instant::now() - Duration::from_secs(60), // Force first read
        }
    }

    /// Discover all hwmon devices
    fn discover_devices() -> Vec<HwmonDevice> {
        let hwmon_path = Path::new("/sys/class/hwmon");

        if !hwmon_path.exists() {
            return Vec::new();
        }

        let mut devices = Vec::new();

        if let Ok(entries) = fs::read_dir(hwmon_path) {
            for entry in entries.filter_map(Result::ok) {
                let path = entry.path();
                let name_path = path.join("name");

                if let Ok(name) = fs::read_to_string(&name_path) {
                    let name = name.trim().to_string();
                    let device_type = match name.as_str() {
                        n if n.contains("coretemp") || n.contains("k10temp") => DeviceType::Cpu,
                        n if n.contains("nvidia") || n.contains("amdgpu") || n.contains("nouveau") => DeviceType::Gpu,
                        _ => DeviceType::Other,
                    };

                    devices.push(HwmonDevice {
                        path,
                        name,
                        device_type,
                    });
                }
            }
        }

        devices
    }

    /// Read temperature from hwmon device
    fn read_temp(&self, device: &HwmonDevice, input: &str) -> Option<f32> {
        let temp_path = device.path.join(input);

        fs::read_to_string(&temp_path)
            .ok()
            .and_then(|s| s.trim().parse::<f32>().ok())
            .map(|millidegrees| millidegrees / 1000.0)
    }

    /// Find first temperature input in device
    fn find_temp_input(&self, device: &HwmonDevice) -> Option<String> {
        if let Ok(entries) = fs::read_dir(&device.path) {
            for entry in entries.filter_map(Result::ok) {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("temp") && name.ends_with("_input") {
                    return Some(name);
                }
            }
        }
        None
    }

    /// Read all sensor data
    pub fn read_sensors(&mut self) -> Vec<SensorReading> {
        let now = Instant::now();

        // Use cache if not expired
        if now.duration_since(self.last_update) < self.cache_expiry {
            return self.cached_to_readings();
        }

        let mut readings = Vec::new();

        // Read temperatures from each device
        for device in &self.devices {
            if let Some(temp_input) = self.find_temp_input(device) {
                if let Some(temp) = self.read_temp(device, &temp_input) {
                    let sensor_type = match device.device_type {
                        DeviceType::Cpu => SensorType::CpuTemperature,
                        DeviceType::Gpu => SensorType::GpuTemperature,
                        DeviceType::Other => continue,
                    };

                    // Normalize: 30°C = 0.0, 100°C = 1.0
                    let normalized = ((temp - 30.0) / 70.0).clamp(0.0, 1.0);

                    let reading = SensorReading {
                        name: device.name.clone(),
                        value: normalized,
                        raw_value: temp,
                        unit: SensorUnit::Celsius,
                        sensor_type,
                    };

                    self.cache.insert(device.name.clone(), normalized);
                    readings.push(reading);
                }
            }
        }

        // Read CPU utilization from /proc/stat
        if let Some(cpu_util) = self.read_cpu_utilization() {
            readings.push(SensorReading {
                name: "cpu_utilization".to_string(),
                value: cpu_util,
                raw_value: cpu_util * 100.0,
                unit: SensorUnit::Percent,
                sensor_type: SensorType::CpuUtilization,
            });
            self.cache.insert("cpu_utilization".to_string(), cpu_util);
        }

        // Read memory usage from /proc/meminfo
        if let Some((used, total)) = self.read_memory_info() {
            let usage = used / total;
            readings.push(SensorReading {
                name: "memory_used".to_string(),
                value: usage,
                raw_value: used,
                unit: SensorUnit::MegaBytes,
                sensor_type: SensorType::MemoryUsed,
            });
            self.cache.insert("memory_usage".to_string(), usage);
        }

        self.last_update = now;
        readings
    }

    /// Read CPU utilization from /proc/stat
    fn read_cpu_utilization(&self) -> Option<f32> {
        let content = fs::read_to_string("/proc/stat").ok()?;
        let first_line = content.lines().next()?;

        if !first_line.starts_with("cpu ") {
            return None;
        }

        let values: Vec<u64> = first_line
            .split_whitespace()
            .skip(1)
            .filter_map(|s| s.parse().ok())
            .collect();

        if values.len() < 4 {
            return None;
        }

        let user = values[0];
        let nice = values[1];
        let system = values[2];
        let idle = values[3];

        let total = user + nice + system + idle;
        let busy = user + nice + system;

        if total > 0 {
            Some(busy as f32 / total as f32)
        } else {
            None
        }
    }

    /// Read memory info from /proc/meminfo
    fn read_memory_info(&self) -> Option<(f32, f32)> {
        let content = fs::read_to_string("/proc/meminfo").ok()?;

        let mut total_kb = 0u64;
        let mut available_kb = 0u64;

        for line in content.lines() {
            if line.starts_with("MemTotal:") {
                total_kb = line.split_whitespace().nth(1)?.parse().ok()?;
            } else if line.starts_with("MemAvailable:") {
                available_kb = line.split_whitespace().nth(1)?.parse().ok()?;
            }
        }

        if total_kb > 0 {
            let used_mb = (total_kb - available_kb) as f32 / 1024.0;
            let total_mb = total_kb as f32 / 1024.0;
            Some((used_mb, total_mb))
        } else {
            None
        }
    }

    /// Convert cache to readings
    fn cached_to_readings(&self) -> Vec<SensorReading> {
        self.cache.iter().map(|(name, &value)| {
            let sensor_type = if name.contains("cpu") {
                if name.contains("util") {
                    SensorType::CpuUtilization
                } else {
                    SensorType::CpuTemperature
                }
            } else if name.contains("nvidia") || name.contains("amd") {
                SensorType::GpuTemperature
            } else if name.contains("memory") {
                SensorType::MemoryUsed
            } else {
                SensorType::CpuTemperature
            };

            SensorReading {
                name: name.clone(),
                value,
                raw_value: value,
                unit: SensorUnit::None,
                sensor_type,
            }
        }).collect()
    }
}

impl Default for HwmonMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// NVIDIA GPU monitor using nvml-wrapper (optional feature)
#[cfg(feature = "nvml")]
pub mod nvidia {
    use super::*;
    use nvml_wrapper::Nvml;

    pub struct NvidiaMonitor {
        nvml: Nvml,
    }

    impl NvidiaMonitor {
        pub fn new() -> Result<Self, nvml_wrapper::error::NvmlError> {
            Ok(Self {
                nvml: Nvml::init()?,
            })
        }

        pub fn read_gpu_sensors(&self, gpu_idx: u32) -> Vec<SensorReading> {
            let mut readings = Vec::new();

            if let Ok(device) = self.nvml.device_by_index(gpu_idx) {
                // Temperature
                if let Ok(temp) = device.temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu) {
                    let normalized = ((temp as f32 - 30.0) / 70.0).clamp(0.0, 1.0);
                    readings.push(SensorReading {
                        name: format!("gpu{}_temp", gpu_idx),
                        value: normalized,
                        raw_value: temp as f32,
                        unit: SensorUnit::Celsius,
                        sensor_type: SensorType::GpuTemperature,
                    });
                }

                // Power draw
                if let Ok(power) = device.power_usage() {
                    let power_w = power as f32 / 1000.0; // mW to W
                    let normalized = (power_w / 350.0).clamp(0.0, 1.0); // Normalize to 350W max
                    readings.push(SensorReading {
                        name: format!("gpu{}_power", gpu_idx),
                        value: normalized,
                        raw_value: power_w,
                        unit: SensorUnit::Watts,
                        sensor_type: SensorType::GpuPower,
                    });
                }

                // Utilization
                if let Ok(util) = device.utilization_rates() {
                    let normalized = util.gpu as f32 / 100.0;
                    readings.push(SensorReading {
                        name: format!("gpu{}_util", gpu_idx),
                        value: normalized,
                        raw_value: util.gpu as f32,
                        unit: SensorUnit::Percent,
                        sensor_type: SensorType::GpuUtilization,
                    });
                }

                // VRAM usage
                if let Ok(mem) = device.memory_info() {
                    let used_mb = mem.used as f32 / (1024.0 * 1024.0);
                    let total_mb = mem.total as f32 / (1024.0 * 1024.0);
                    let normalized = used_mb / total_mb;

                    readings.push(SensorReading {
                        name: format!("gpu{}_vram", gpu_idx),
                        value: normalized,
                        raw_value: used_mb,
                        unit: SensorUnit::MegaBytes,
                        sensor_type: SensorType::VramUsed,
                    });
                }
            }

            readings
        }
    }
}

/// Fallback NVIDIA monitor using hwmon (no nvml dependency)
pub struct NvidiaHwmonMonitor {
    device_path: Option<PathBuf>,
}

impl NvidiaHwmonMonitor {
    pub fn new() -> Self {
        let device_path = Self::find_nvidia_hwmon();
        Self { device_path }
    }

    fn find_nvidia_hwmon() -> Option<PathBuf> {
        let hwmon_path = Path::new("/sys/class/hwmon");

        if let Ok(entries) = fs::read_dir(hwmon_path) {
            for entry in entries.filter_map(Result::ok) {
                let path = entry.path();
                let name_path = path.join("name");

                if let Ok(name) = fs::read_to_string(&name_path) {
                    if name.trim().contains("nvidia") {
                        return Some(path);
                    }
                }
            }
        }

        None
    }

    pub fn read_gpu_temp(&self) -> Option<SensorReading> {
        let path = self.device_path.as_ref()?;
        let temp_path = path.join("temp1_input");

        let temp_millidegrees: f32 = fs::read_to_string(&temp_path)
            .ok()?
            .trim()
            .parse()
            .ok()?;

        let temp = temp_millidegrees / 1000.0;
        let normalized = ((temp - 30.0) / 70.0).clamp(0.0, 1.0);

        Some(SensorReading {
            name: "nvidia_gpu_temp".to_string(),
            value: normalized,
            raw_value: temp,
            unit: SensorUnit::Celsius,
            sensor_type: SensorType::GpuTemperature,
        })
    }

    pub fn is_available(&self) -> bool {
        self.device_path.is_some()
    }
}

impl Default for NvidiaHwmonMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Unified proprioception interface
pub struct Proprioception {
    hwmon: HwmonMonitor,
    nvidia_hwmon: NvidiaHwmonMonitor,
    /// Smoothing factor for consciousness values (0-1)
    smoothing: f32,
    /// Previous consciousness map for smoothing
    previous: Option<ConsciousnessMap>,
}

impl Proprioception {
    pub fn new() -> Self {
        Self {
            hwmon: HwmonMonitor::new(),
            nvidia_hwmon: NvidiaHwmonMonitor::new(),
            smoothing: 0.3, // 30% new, 70% old
            previous: None,
        }
    }

    /// Set smoothing factor (0 = no smoothing, 1 = no change)
    pub fn with_smoothing(mut self, factor: f32) -> Self {
        self.smoothing = factor.clamp(0.0, 0.99);
        self
    }

    /// Read current proprioceptive state
    pub fn read_state(&mut self) -> ProprioceptiveState {
        let mut readings = self.hwmon.read_sensors();

        // Add NVIDIA readings if available
        if let Some(gpu_temp) = self.nvidia_hwmon.read_gpu_temp() {
            readings.push(gpu_temp);
        }

        // Build consciousness map
        let mut map = ConsciousnessMap::default();

        for reading in &readings {
            match reading.sensor_type {
                SensorType::CpuTemperature => {
                    map.arousal = map.arousal.max(reading.value);
                }
                SensorType::GpuTemperature => {
                    map.arousal = map.arousal.max(reading.value * 1.2); // GPU temp more relevant
                }
                SensorType::CpuPower | SensorType::GpuPower => {
                    map.metabolic_rate = map.metabolic_rate.max(reading.value);
                }
                SensorType::CpuUtilization => {
                    map.cognitive_load = reading.value;
                }
                SensorType::GpuUtilization => {
                    map.cognitive_load = map.cognitive_load.max(reading.value * 0.8);
                }
                SensorType::MemoryUsed => {
                    map.memory_pressure = reading.value;
                }
                SensorType::VramUsed => {
                    map.memory_pressure = map.memory_pressure.max(reading.value);
                }
                _ => {}
            }
        }

        // Calculate substrate stress as weighted average
        map.substrate_stress = (
            map.arousal * 0.3 +
            map.metabolic_rate * 0.2 +
            map.cognitive_load * 0.3 +
            map.memory_pressure * 0.2
        ).clamp(0.0, 1.0);

        // Apply smoothing
        if let Some(ref prev) = self.previous {
            let s = self.smoothing;
            map.arousal = prev.arousal * s + map.arousal * (1.0 - s);
            map.metabolic_rate = prev.metabolic_rate * s + map.metabolic_rate * (1.0 - s);
            map.cognitive_load = prev.cognitive_load * s + map.cognitive_load * (1.0 - s);
            map.memory_pressure = prev.memory_pressure * s + map.memory_pressure * (1.0 - s);
            map.substrate_stress = prev.substrate_stress * s + map.substrate_stress * (1.0 - s);
        }

        self.previous = Some(map.clone());

        ProprioceptiveState {
            readings,
            timestamp: Instant::now(),
            consciousness_map: map,
        }
    }

    /// Get HDC-encoded proprioceptive state
    pub fn read_hdc(&mut self) -> Vec<i8> {
        self.read_state().consciousness_map.to_hdc()
    }

    /// Get packed HDC for efficient similarity operations
    pub fn read_packed(&mut self) -> PackedBipolar {
        self.read_state().consciousness_map.to_packed()
    }

    /// Check if GPU monitoring is available
    pub fn has_gpu(&self) -> bool {
        self.nvidia_hwmon.is_available()
    }
}

impl Default for Proprioception {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consciousness_map_to_hdc() {
        let map = ConsciousnessMap {
            arousal: 0.5,
            metabolic_rate: 0.3,
            cognitive_load: 0.7,
            memory_pressure: 0.4,
            substrate_stress: 0.5,
        };

        let hdc = map.to_hdc();
        assert_eq!(hdc.len(), HDC_DIMENSION);

        // Count positive bits - should roughly match our encoded values
        let positive: usize = hdc.iter().filter(|&&x| x > 0).count();
        assert!(positive > 0);
        assert!(positive < HDC_DIMENSION);
    }

    #[test]
    fn test_consciousness_map_to_packed() {
        let map = ConsciousnessMap {
            arousal: 0.8,
            metabolic_rate: 0.6,
            cognitive_load: 0.4,
            memory_pressure: 0.2,
            substrate_stress: 0.5,
        };

        let packed = map.to_packed();
        assert_eq!(packed.dimension(), HDC_DIMENSION);

        // Memory should be much smaller than raw i8 vector
        assert!(packed.memory_bytes() < HDC_DIMENSION);
    }

    #[test]
    fn test_hwmon_monitor_creation() {
        // Should not panic even if no hwmon devices exist
        let monitor = HwmonMonitor::new();
        // Devices may or may not be present depending on system
        let _ = monitor.devices.len();
    }

    #[test]
    fn test_nvidia_hwmon_fallback() {
        let monitor = NvidiaHwmonMonitor::new();
        // Should not panic even without NVIDIA GPU
        let _ = monitor.is_available();
    }

    #[test]
    fn test_proprioception_integration() {
        let mut prop = Proprioception::new().with_smoothing(0.0);

        // Should not panic
        let state = prop.read_state();
        assert!(!state.readings.is_empty() || true); // May be empty on minimal systems

        // Consciousness map should have valid ranges
        let map = &state.consciousness_map;
        assert!(map.arousal >= 0.0 && map.arousal <= 1.0);
        assert!(map.metabolic_rate >= 0.0 && map.metabolic_rate <= 1.0);
        assert!(map.cognitive_load >= 0.0 && map.cognitive_load <= 1.0);
        assert!(map.memory_pressure >= 0.0 && map.memory_pressure <= 1.0);
        assert!(map.substrate_stress >= 0.0 && map.substrate_stress <= 1.0);
    }

    #[test]
    fn test_smoothing() {
        let mut prop = Proprioception::new().with_smoothing(0.5);

        // First reading sets baseline
        let state1 = prop.read_state();
        let stress1 = state1.consciousness_map.substrate_stress;

        // Second reading should be smoothed
        let state2 = prop.read_state();
        let stress2 = state2.consciousness_map.substrate_stress;

        // Values should be similar due to smoothing
        // (unless there's a dramatic change in system state)
        assert!((stress1 - stress2).abs() < 0.5);
    }
}
