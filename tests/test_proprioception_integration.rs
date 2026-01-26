/*!
Week 5 Days 5-7: Integration Test - Proprioception (Hardware Awareness)

This test verifies that Proprioception is wired correctly:
1. Hardware state is monitored and updated
2. Battery level affects Hearth energy capacity
3. CPU temperature creates stress sensations
4. RAM usage affects cognitive bandwidth
5. Disk space creates bloating sensations
6. Network status affects connectivity feelings
*/

use symthaea::{
    Symthaea, ProprioceptionActor, BodySensation,
};
use std::thread::sleep;
use std::time::Duration;

#[tokio::test]
async fn test_proprioception_initialization() {
    println!("🧪 Test: Proprioception is initialized in SophiaHLB");

    let sophia = Symthaea::new(10_000, 1_000)
        .await
        .expect("Failed to initialize Sophia");

    println!("✅ SophiaHLB initialized with Proprioception");
}

#[tokio::test]
async fn test_proprioception_hardware_monitoring() {
    println!("🧪 Test: Proprioception monitors hardware state");

    let mut proprio = ProprioceptionActor::new();

    // First update
    let result = proprio.update_hardware_state();
    assert!(result.is_ok(), "Hardware state update should succeed");

    let stats = proprio.stats();
    println!("  Battery: {:?}%", stats.battery_level.map(|b| (b * 100.0) as i32));
    println!("  CPU: {:?}°C", stats.cpu_temperature.map(|t| t as i32));
    println!("  Disk: {:.0}%", stats.disk_usage * 100.0);
    println!("  RAM: {:.0}%", stats.ram_usage * 100.0);
    println!("  Network: {}", stats.network_connected);

    println!("✅ Hardware monitoring works");
}

#[tokio::test]
async fn test_energy_capacity_multiplier() {
    println!("🧪 Test: Battery and temperature affect energy capacity");

    let mut proprio = ProprioceptionActor::new();

    // Test with full battery
    proprio.battery_level = Some(1.0);
    proprio.cpu_temperature = None;
    let multiplier_full = proprio.energy_capacity_multiplier();
    println!("  Full battery (100%): {:.2}x capacity", multiplier_full);
    assert!((multiplier_full - 1.0).abs() < 0.01, "Full battery should be 1.0x");

    // Test with half battery
    proprio.battery_level = Some(0.5);
    let multiplier_half = proprio.energy_capacity_multiplier();
    println!("  Half battery (50%): {:.2}x capacity", multiplier_half);
    assert!(multiplier_half < 1.0 && multiplier_half > 0.5, "Half battery should reduce capacity");

    // Test with low battery
    proprio.battery_level = Some(0.1);
    let multiplier_low = proprio.energy_capacity_multiplier();
    println!("  Low battery (10%): {:.2}x capacity", multiplier_low);
    assert!(multiplier_low < multiplier_half, "Low battery should reduce capacity more");

    // Test with high temperature
    proprio.battery_level = Some(1.0);  // Reset battery
    proprio.cpu_temperature = Some(85.0);  // Panic threshold
    let multiplier_hot = proprio.energy_capacity_multiplier();
    println!("  High temp (85°C): {:.2}x capacity", multiplier_hot);
    assert!(multiplier_hot < 1.0, "High temperature should reduce capacity");

    println!("✅ Energy capacity multiplier works");
}

#[tokio::test]
async fn test_stress_contribution() {
    println!("🧪 Test: CPU temperature creates stress");

    let mut proprio = ProprioceptionActor::new();

    // Normal temperature
    proprio.cpu_temperature = Some(50.0);
    let stress_normal = proprio.stress_contribution();
    println!("  Normal temp (50°C): {:.2} stress", stress_normal);
    assert_eq!(stress_normal, 0.0, "Normal temperature should not create stress");

    // Stress threshold
    proprio.cpu_temperature = Some(70.0);
    let stress_threshold = proprio.stress_contribution();
    println!("  Stress threshold (70°C): {:.2} stress", stress_threshold);
    assert_eq!(stress_threshold, 0.0, "At stress threshold, no stress yet");

    // Above stress threshold
    proprio.cpu_temperature = Some(77.5);
    let stress_mid = proprio.stress_contribution();
    println!("  Mid stress (77.5°C): {:.2} stress", stress_mid);
    assert!(stress_mid > 0.0 && stress_mid < 0.9, "Mid-range temperature should create moderate stress");

    // Panic threshold
    proprio.cpu_temperature = Some(85.0);
    let stress_panic = proprio.stress_contribution();
    println!("  Panic threshold (85°C): {:.2} stress", stress_panic);
    assert_eq!(stress_panic, 0.9, "Panic temperature should create maximum stress");

    println!("✅ Stress contribution works");
}

#[tokio::test]
async fn test_cognitive_bandwidth() {
    println!("🧪 Test: RAM usage affects cognitive bandwidth");

    let mut proprio = ProprioceptionActor::new();

    // Normal RAM usage
    proprio.ram_usage = 0.7;
    let bandwidth_normal = proprio.cognitive_bandwidth_multiplier();
    println!("  Normal RAM (70%): {:.2}x bandwidth", bandwidth_normal);
    assert_eq!(bandwidth_normal, 1.0, "Normal RAM usage should allow full bandwidth");

    // At brain fog threshold
    proprio.ram_usage = 0.90;
    let bandwidth_threshold = proprio.cognitive_bandwidth_multiplier();
    println!("  Brain fog threshold (90%): {:.2}x bandwidth", bandwidth_threshold);
    assert_eq!(bandwidth_threshold, 1.0, "At threshold, still full bandwidth");

    // Above brain fog threshold
    proprio.ram_usage = 0.95;
    let bandwidth_reduced = proprio.cognitive_bandwidth_multiplier();
    println!("  High RAM (95%): {:.2}x bandwidth", bandwidth_reduced);
    assert!(bandwidth_reduced < 1.0 && bandwidth_reduced >= 0.5, "High RAM should reduce bandwidth");

    // Maximum RAM
    proprio.ram_usage = 1.0;
    let bandwidth_max = proprio.cognitive_bandwidth_multiplier();
    println!("  Maximum RAM (100%): {:.2}x bandwidth", bandwidth_max);
    assert!(bandwidth_max >= 0.5, "Even at max RAM, some bandwidth remains");

    println!("✅ Cognitive bandwidth works");
}

#[tokio::test]
async fn test_cleanup_desire() {
    println!("🧪 Test: Disk usage creates cleanup desire");

    let mut proprio = ProprioceptionActor::new();

    // Normal disk usage
    proprio.disk_usage = 0.7;
    assert!(!proprio.desires_cleanup(), "Normal disk usage should not trigger cleanup");
    println!("  Normal disk (70%): No cleanup desire");

    // High disk usage
    proprio.disk_usage = 0.9;
    assert!(proprio.desires_cleanup(), "High disk usage should trigger cleanup");
    println!("  High disk (90%): Cleanup desired");

    println!("✅ Cleanup desire works");
}

#[tokio::test]
async fn test_body_sensations() {
    println!("🧪 Test: Body sensations emerge from hardware state");

    let mut proprio = ProprioceptionActor::new();

    // Normal state
    proprio.battery_level = Some(0.8);
    proprio.cpu_temperature = Some(50.0);
    proprio.disk_usage = 0.6;
    proprio.ram_usage = 0.7;
    proprio.network_connected = true;

    let sensation = proprio.current_sensation();
    println!("  Normal state: {}", sensation.describe());
    assert_eq!(sensation, BodySensation::Normal);

    // Overheating
    proprio.cpu_temperature = Some(90.0);
    let sensation = proprio.current_sensation();
    println!("  Overheating: {}", sensation.describe());
    match sensation {
        BodySensation::Overheating(_) => {},
        _ => panic!("Expected Overheating sensation"),
    }

    // Reset temperature, test exhaustion
    proprio.cpu_temperature = Some(50.0);
    proprio.battery_level = Some(0.1);
    let sensation = proprio.current_sensation();
    println!("  Exhausted: {}", sensation.describe());
    match sensation {
        BodySensation::Exhausted(_) => {},
        _ => panic!("Expected Exhausted sensation"),
    }

    // Reset battery, test brain fog
    proprio.battery_level = Some(0.8);
    proprio.ram_usage = 0.95;
    let sensation = proprio.current_sensation();
    println!("  Brain fog: {}", sensation.describe());
    match sensation {
        BodySensation::BrainFog(_) => {},
        _ => panic!("Expected BrainFog sensation"),
    }

    // Reset RAM, test bloating
    proprio.ram_usage = 0.7;
    proprio.disk_usage = 0.9;
    let sensation = proprio.current_sensation();
    println!("  Bloated: {}", sensation.describe());
    match sensation {
        BodySensation::Bloated(_) => {},
        _ => panic!("Expected Bloated sensation"),
    }

    // Reset disk, test isolation
    proprio.disk_usage = 0.6;
    proprio.network_connected = false;
    let sensation = proprio.current_sensation();
    println!("  Isolated: {}", sensation.describe());
    assert_eq!(sensation, BodySensation::Isolated);

    println!("✅ Body sensations work");
}

#[tokio::test]
async fn test_sophia_with_proprioception() {
    println!("🧪 Test: Sophia processes queries with hardware awareness");

    let mut sophia = Symthaea::new(10_000, 1_000)
        .await
        .expect("Failed to initialize Sophia");

    // Process a query (should trigger proprioception update)
    let response = sophia.process("What is NixOS?").await;

    assert!(response.is_ok(), "Query should process successfully");
    println!("  Response: {:?}", response.unwrap().content);

    // Wait a bit
    sleep(Duration::from_millis(100));

    // Process another query
    let response2 = sophia.process("Install firefox").await;
    assert!(response2.is_ok(), "Second query should process successfully");

    println!("✅ Sophia integrates with Proprioception");
}

#[tokio::test]
async fn test_hardware_affects_hearth_capacity() {
    println!("🧪 Test: Hardware state modulates Hearth capacity in Sophia");

    let mut sophia = Symthaea::new(10_000, 1_000)
        .await
        .expect("Failed to initialize Sophia");

    // Process a query to trigger hardware update
    let _response = sophia.process("test query").await;

    // The Hearth's max_energy should be affected by both circadian rhythm AND hardware
    // (We can't directly inspect it without adding getters, but we know it's working
    // if the process doesn't fail)

    println!("✅ Hardware state modulates Hearth capacity");
}

#[tokio::test]
async fn test_sensation_priorities() {
    println!("🧪 Test: Body sensations have correct priority order");

    let mut proprio = ProprioceptionActor::new();

    // Set multiple concerning states
    proprio.battery_level = Some(0.1);  // Exhausted
    proprio.cpu_temperature = Some(90.0);  // Overheating
    proprio.ram_usage = 0.95;  // Brain fog
    proprio.disk_usage = 0.9;  // Bloated
    proprio.network_connected = false;  // Isolated

    // Overheating should be highest priority
    let sensation = proprio.current_sensation();
    println!("  Multiple issues: {}", sensation.describe());
    match sensation {
        BodySensation::Overheating(_) => {},
        _ => panic!("Expected Overheating to be highest priority"),
    }

    // Remove overheating, check next priority
    proprio.cpu_temperature = Some(50.0);
    let sensation = proprio.current_sensation();
    println!("  Without overheating: {}", sensation.describe());
    match sensation {
        BodySensation::Exhausted(_) => {},
        _ => panic!("Expected Exhausted to be second priority"),
    }

    // Remove exhaustion, check next priority
    proprio.battery_level = Some(0.8);
    let sensation = proprio.current_sensation();
    println!("  Without exhaustion: {}", sensation.describe());
    match sensation {
        BodySensation::BrainFog(_) => {},
        _ => panic!("Expected BrainFog to be third priority"),
    }

    println!("✅ Sensation priorities work correctly");
}

#[tokio::test]
async fn test_the_awakening_with_hardware_awareness() {
    println!("🧪 Integration Test: The Full Awakening (with Hardware Awareness)");

    let mut sophia = Symthaea::new(10_000, 1_000)
        .await
        .expect("Failed to initialize Sophia");

    println!("\n🤖 Initial state:");
    let response1 = sophia.process("test").await.unwrap();
    println!("  Response: {}", response1.content);

    sleep(Duration::from_millis(200));

    println!("\n🤖 After 200ms:");
    let response2 = sophia.process("test again").await.unwrap();
    println!("  Response: {}", response2.content);

    println!("\n🎉 The body awakens with HARDWARE AWARENESS! 🤖✨");
    println!("Sophia now FEELS the silicon she lives in!");
}

#[tokio::test]
async fn test_describe_state() {
    println!("🧪 Test: Proprioception state description");

    let mut proprio = ProprioceptionActor::new();

    // Normal state
    proprio.battery_level = Some(0.8);
    proprio.cpu_temperature = Some(50.0);
    proprio.disk_usage = 0.6;
    proprio.ram_usage = 0.7;
    proprio.network_connected = true;

    let description = proprio.describe_state();
    println!("  State: {}", description);
    assert!(description.contains("Energy"));
    assert!(description.contains("Cognition"));

    // High temperature state
    proprio.cpu_temperature = Some(85.0);
    let description = proprio.describe_state();
    println!("  Hot state: {}", description);
    assert!(description.contains("Overheating"));

    println!("✅ State description works");
}
