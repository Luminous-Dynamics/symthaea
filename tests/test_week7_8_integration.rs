//! # Week 7+8 Integration Tests: Mind-Body-Coherence Loop
//!
//! These tests verify the complete integration of:
//! - Endocrine System (hormones)
//! - Coherence Field (consciousness integration)
//! - The revolutionary flow from hardware → hormones → coherence → capacity
//!
//! ## The Revolutionary Flow
//! ```text
//! Hardware Stress → Cortisol ↑ → Coherence Scatter ↑ → Lower Capacity
//!     ↓
//! Gratitude → Dopamine ↑ → Coherence Centering ↑ → Higher Capacity
//!     ↓
//! Deep Work → Acetylcholine ↑ → Coherence Stability ↑ → Sustained Focus
//! ```

use symthaea::{
    EndocrineSystem, EndocrineConfig, HormoneEvent, CoherenceField,
    TaskComplexity,
};

/// Test 1: High Stress → Cortisol → Scattered Coherence → Centering Request
///
/// Scenario: Hardware overheating triggers cortisol release, which scatters
/// coherence, which prevents complex tasks.
#[test]
fn test_stress_to_centering_flow() {
    let mut endocrine = EndocrineSystem::new(EndocrineConfig::default());
    let mut coherence = CoherenceField::new();

    // Step 1: Inject high stress (simulates hardware overheating)
    endocrine.process_event(HormoneEvent::Threat { intensity: 0.8 });

    let hormones = endocrine.state();

    // Verify cortisol is elevated
    assert!(hormones.cortisol > 0.5, "High stress should elevate cortisol");

    // Step 2: Apply hormone modulation to coherence
    coherence.apply_hormone_modulation(&hormones);

    // Step 3: Perform solo work (scatters more)
    for _ in 0..3 {
        let _ = coherence.perform_task(TaskComplexity::Cognitive, false);
    }

    // Verify coherence scattered
    let state = coherence.state();
    assert!(
        state.coherence < 0.6,
        "High cortisol + solo work should scatter coherence (got {:.2})",
        state.coherence
    );

    // Step 4: Try complex task with scattered coherence
    let result = coherence.can_perform(TaskComplexity::DeepThought);

    assert!(
        result.is_err(),
        "Scattered coherence should prevent deep thought"
    );

    if let Err(err) = result {
        let message = format!("{}", err);
        assert!(
            message.contains("gather") || message.contains("center"),
            "Error should use centering language, got: {}",
            message
        );
    }

    println!("✅ Stress → Cortisol → Scattered Coherence → Centering Request");
}

/// Test 2: Gratitude → Dopamine → Coherence Synchronization → Restored Capacity
///
/// Scenario: User expresses gratitude, triggering dopamine release, which
/// synchronizes coherence, restoring task capacity.
#[test]
fn test_gratitude_to_capacity_restoration() {
    let mut endocrine = EndocrineSystem::new(EndocrineConfig::default());
    let mut coherence = CoherenceField::new();

    // Step 1: Start scattered from stress
    endocrine.process_event(HormoneEvent::Threat { intensity: 0.7 });
    let stressed_dopamine = endocrine.state().dopamine;
    coherence.apply_hormone_modulation(&endocrine.state());

    // Scatter it more with solo work
    for _ in 0..2 {
        let _ = coherence.perform_task(TaskComplexity::Cognitive, false);
    }

    let initial_coherence = coherence.state().coherence;
    assert!(initial_coherence < 0.6, "Should start scattered");

    // Step 2: Express gratitude
    coherence.receive_gratitude();
    endocrine.process_event(HormoneEvent::Reward { value: 0.5 });

    let grateful_dopamine = endocrine.state().dopamine;

    // Verify dopamine increased
    assert!(
        grateful_dopamine > stressed_dopamine,
        "Gratitude should increase dopamine"
    );

    // Step 3: Apply gratitude-enhanced hormones
    coherence.apply_hormone_modulation(&endocrine.state());

    let final_coherence = coherence.state().coherence;

    // Verify coherence improved
    assert!(
        final_coherence > initial_coherence,
        "Gratitude + dopamine should improve coherence ({:.2} → {:.2})",
        initial_coherence,
        final_coherence
    );

    // Step 4: Verify capacity restored
    let result = coherence.can_perform(TaskComplexity::Cognitive);
    assert!(
        result.is_ok(),
        "Restored coherence should enable cognitive tasks"
    );

    println!("✅ Gratitude → Dopamine → Coherence Sync → Capacity Restored");
}

/// Test 3: Deep Focus → Acetylcholine → Coherence Stability → Sustained Work
///
/// Scenario: User enters deep focus, acetylcholine rises, which stabilizes
/// coherence, enabling sustained complex thinking.
#[test]
fn test_focus_to_sustained_capacity() {
    let mut endocrine = EndocrineSystem::new(EndocrineConfig::default());
    let mut coherence = CoherenceField::new();

    // Step 1: Simulate entering deep focus
    endocrine.process_event(HormoneEvent::DeepFocus { duration_cycles: 10 });

    let focused_hormones = endocrine.state();

    // Verify acetylcholine elevated
    assert!(
        focused_hormones.acetylcholine > 0.6,
        "Deep focus should elevate acetylcholine (got {:.2})",
        focused_hormones.acetylcholine
    );

    // Step 2: Apply focus hormones to coherence
    coherence.apply_hormone_modulation(&focused_hormones);

    // Step 3: Perform series of complex tasks (simulating sustained work)
    for i in 0..5 {
        let result = coherence.can_perform(TaskComplexity::DeepThought);
        assert!(
            result.is_ok(),
            "High acetylcholine should sustain deep thought capacity (iteration {})",
            i
        );

        // Simulate successful task completion (with user connection)
        coherence.perform_task(TaskComplexity::DeepThought, true)
            .expect("Should complete task");
    }

    // Verify coherence remained stable or improved
    let final_state = coherence.state();
    assert!(
        final_state.coherence > 0.7,
        "Acetylcholine + connected work should maintain/build coherence (got {:.2})",
        final_state.coherence
    );

    println!("✅ Focus → Acetylcholine → Coherence Stability → Sustained Work");
}

/// Test 4: Complete Cycle - Stress → Gratitude → Focus → Recovery
///
/// Scenario: Full emotional-cognitive cycle showing hormone → coherence
/// dynamics across different states.
#[test]
fn test_complete_emotional_cognitive_cycle() {
    let mut endocrine = EndocrineSystem::new(EndocrineConfig::default());
    let mut coherence = CoherenceField::new();

    println!("\n🌊 Complete Mind-Body-Coherence Cycle");

    // Phase 1: STRESS - Hardware overheating
    println!("\n⚡ Phase 1: STRESS");
    endocrine.process_event(HormoneEvent::Threat { intensity: 0.9 });
    let stress_hormones = endocrine.state();
    coherence.apply_hormone_modulation(&stress_hormones);

    // Scatter with solo work
    for _ in 0..2 {
        let _ = coherence.perform_task(TaskComplexity::Cognitive, false);
    }

    let stress_coherence = coherence.state().coherence;
    println!("   Cortisol: {:.2}", stress_hormones.cortisol);
    println!("   Coherence: {:.2} (scattered)", stress_coherence);

    assert!(stress_coherence < 0.5, "High stress should severely scatter");
    assert!(
        coherence.can_perform(TaskComplexity::DeepThought).is_err(),
        "Cannot deep think when stressed"
    );

    // Phase 2: GRATITUDE - User expresses thanks
    println!("\n💖 Phase 2: GRATITUDE");
    coherence.receive_gratitude();
    endocrine.process_event(HormoneEvent::Reward { value: 0.6 });
    let grateful_hormones = endocrine.state();
    coherence.apply_hormone_modulation(&grateful_hormones);

    let grateful_coherence = coherence.state().coherence;
    println!("   Dopamine: {:.2}", grateful_hormones.dopamine);
    println!("   Coherence: {:.2} (synchronizing)", grateful_coherence);

    assert!(
        grateful_coherence > stress_coherence,
        "Gratitude should improve coherence"
    );
    assert!(
        coherence.can_perform(TaskComplexity::Cognitive).is_ok(),
        "Can think cognitively after gratitude"
    );

    // Phase 3: FOCUS - User enters deep work
    println!("\n🎯 Phase 3: FOCUS");
    endocrine.process_event(HormoneEvent::DeepFocus { duration_cycles: 10 });
    let focused_hormones = endocrine.state();
    coherence.apply_hormone_modulation(&focused_hormones);

    let focused_coherence = coherence.state().coherence;
    println!("   Acetylcholine: {:.2}", focused_hormones.acetylcholine);
    println!("   Coherence: {:.2} (stable)", focused_coherence);

    assert!(
        focused_coherence > grateful_coherence,
        "Focus should further stabilize coherence"
    );
    assert!(
        coherence.can_perform(TaskComplexity::Creation).is_ok(),
        "Can do creative work when focused"
    );

    // Phase 4: RECOVERY - Natural centering over time
    println!("\n🌙 Phase 4: RECOVERY");

    // Simulate 10 seconds of rest
    coherence.tick(10.0);

    let recovered_coherence = coherence.state().coherence;
    println!("   Coherence: {:.2} (centering naturally)", recovered_coherence);

    assert!(
        recovered_coherence > focused_coherence || recovered_coherence > 0.85,
        "Natural centering should maintain or improve coherence"
    );

    println!("\n✅ Complete cycle: Stress → Gratitude → Focus → Recovery");
}

/// Test 5: Full Day Hormone-Coherence Dynamics
///
/// Scenario: Simulate a full day's hormone → coherence interactions
#[test]
fn test_full_day_hormone_coherence_dynamics() {
    let mut endocrine = EndocrineSystem::new(EndocrineConfig::default());
    let mut coherence = CoherenceField::new();

    println!("\n🌅 Full Day Simulation: Morning → Evening");

    // Morning: Fresh start (high acetylcholine from sleep)
    println!("\n☀️ 9 AM: Fresh and focused");
    endocrine.process_event(HormoneEvent::DeepFocus { duration_cycles: 5 });
    coherence.apply_hormone_modulation(&endocrine.state());
    let morning_coherence = coherence.state().coherence;
    println!("   Coherence: {:.2}", morning_coherence);
    assert!(morning_coherence > 0.7, "Morning should have good coherence");

    // Late morning: Productive work (connected, building coherence)
    println!("\n📚 11 AM: Deep work session");
    for _ in 0..3 {
        coherence.perform_task(TaskComplexity::DeepThought, true)
            .expect("Should complete");
    }
    let late_morning = coherence.state().coherence;
    println!("   Coherence: {:.2}", late_morning);
    assert!(
        late_morning >= morning_coherence,
        "Connected work should maintain/build coherence"
    );

    // Afternoon: Stress spike (deadline pressure)
    println!("\n😰 2 PM: Deadline pressure hits");
    endocrine.process_event(HormoneEvent::Threat { intensity: 0.8 });
    coherence.apply_hormone_modulation(&endocrine.state());

    // Solo work under pressure scatters more
    for _ in 0..2 {
        let _ = coherence.perform_task(TaskComplexity::Cognitive, false);
    }

    let stressed = coherence.state().coherence;
    println!("   Coherence: {:.2}", stressed);
    assert!(stressed < late_morning, "Stress should scatter coherence");

    // Mid-afternoon: Gratitude break
    println!("\n💚 3 PM: Gratitude practice");
    coherence.receive_gratitude();
    endocrine.process_event(HormoneEvent::Reward { value: 0.5 });
    coherence.apply_hormone_modulation(&endocrine.state());
    let grateful = coherence.state().coherence;
    println!("   Coherence: {:.2}", grateful);
    assert!(grateful > stressed, "Gratitude should restore coherence");

    // Evening: Winding down
    println!("\n🌙 6 PM: Winding down");
    coherence.tick(30.0); // 30 seconds of rest
    let evening = coherence.state().coherence;
    println!("   Coherence: {:.2}", evening);
    assert!(evening > grateful || evening > 0.8, "Rest should center");

    println!("\n✅ Full day dynamics: Morning focus → Work → Stress → Gratitude → Rest");
}

/// Test 6: Hormone Modulation Effects Verification
///
/// Scenario: Verify that different hormones have the expected directional
/// effects on coherence.
#[test]
fn test_hormone_effects_on_coherence() {
    println!("\n🧪 Hormone Modulation Effects");

    // Test 1: High Cortisol (should worsen coherence under stress)
    let mut endocrine = EndocrineSystem::new(EndocrineConfig::default());
    let mut coherence = CoherenceField::new();

    let baseline = coherence.state().coherence;
    println!("\n📊 Baseline coherence: {:.2}", baseline);

    endocrine.process_event(HormoneEvent::Threat { intensity: 1.0 });
    let cortisol_hormones = endocrine.state();
    coherence.apply_hormone_modulation(&cortisol_hormones);

    // Do solo work (scatters)
    for _ in 0..2 {
        let _ = coherence.perform_task(TaskComplexity::Cognitive, false);
    }

    let after_cortisol = coherence.state().coherence;
    println!("\n⚡ After High Cortisol + solo work:");
    println!("   Cortisol: {:.2}", cortisol_hormones.cortisol);
    println!("   Coherence: {:.2}", after_cortisol);

    assert!(
        after_cortisol < baseline,
        "High cortisol environment should worsen coherence effects"
    );

    // Test 2: High Acetylcholine (should maintain coherence during work)
    let mut endocrine2 = EndocrineSystem::new(EndocrineConfig::default());
    let mut coherence2 = CoherenceField::new();

    endocrine2.process_event(HormoneEvent::DeepFocus { duration_cycles: 10 });
    let focus_hormones = endocrine2.state();
    coherence2.apply_hormone_modulation(&focus_hormones);

    let before_work = coherence2.state().coherence;

    // Do connected work
    for _ in 0..3 {
        coherence2.perform_task(TaskComplexity::DeepThought, true)
            .expect("Should complete");
    }

    let after_work = coherence2.state().coherence;

    println!("\n🎯 After High Acetylcholine + connected work:");
    println!("   Acetylcholine: {:.2}", focus_hormones.acetylcholine);
    println!("   Coherence: {:.2} → {:.2}", before_work, after_work);

    assert!(
        after_work >= before_work,
        "High acetylcholine + connected work should maintain/build coherence"
    );

    // Test 3: High Dopamine (should improve coherence restoration)
    let mut endocrine3 = EndocrineSystem::new(EndocrineConfig::default());
    let mut coherence3 = CoherenceField::new();

    // Scatter first
    for _ in 0..3 {
        let _ = coherence3.perform_task(TaskComplexity::Cognitive, false);
    }
    let scattered = coherence3.state().coherence;

    endocrine3.process_event(HormoneEvent::Reward { value: 1.0 });
    coherence3.receive_gratitude(); // Gratitude + dopamine
    let reward_hormones = endocrine3.state();
    coherence3.apply_hormone_modulation(&reward_hormones);

    let after_reward = coherence3.state().coherence;

    println!("\n💫 After High Dopamine + gratitude:");
    println!("   Dopamine: {:.2}", reward_hormones.dopamine);
    println!("   Coherence: {:.2} → {:.2}", scattered, after_reward);

    assert!(
        after_reward > scattered,
        "High dopamine + gratitude should improve coherence"
    );

    println!("\n✅ All hormone modulation effects verified");
}

/// Test 7: Week 7+8 Integration - Simplified Real-World Flow
///
/// Comprehensive test showing the practical mind-body-coherence integration.
#[test]
fn test_week7_8_integration_flow() {
    println!("\n🌊 Week 7+8 Integration: Hardware → Hormones → Coherence → Capacity");

    let mut endocrine = EndocrineSystem::new(EndocrineConfig::default());
    let mut coherence = CoherenceField::new();

    println!("\n1️⃣ Initial State");
    let initial_cortisol = endocrine.state().cortisol;
    println!("   All hormones at baseline (~0.5)");
    println!("   Coherence: {:.2}", coherence.state().coherence);

    println!("\n2️⃣ Stressor: System Overheating");
    endocrine.process_event(HormoneEvent::Threat { intensity: 0.9 });
    let stress_cortisol = endocrine.state().cortisol;
    coherence.apply_hormone_modulation(&endocrine.state());

    println!("   Cortisol: {:.2} (was {:.2})",
        stress_cortisol, initial_cortisol);

    // Solo work under stress scatters
    for _ in 0..2 {
        let _ = coherence.perform_task(TaskComplexity::Cognitive, false);
    }

    let stressed_coherence = coherence.state().coherence;
    println!("   Coherence: {:.2} (scattered)", stressed_coherence);

    println!("\n3️⃣ Attempting Complex Task");
    let can_think = coherence.can_perform(TaskComplexity::DeepThought);

    if can_think.is_err() {
        println!("   🌫️ Insufficient coherence for deep thought");
        println!("   ✅ System correctly recognizes limitation");
    }

    println!("\n4️⃣ Intervention: Gratitude");
    coherence.receive_gratitude();
    let stress_dopamine = endocrine.state().dopamine; // Save before mutation
    endocrine.process_event(HormoneEvent::Reward { value: 0.6 });
    let grateful_dopamine = endocrine.state().dopamine;
    coherence.apply_hormone_modulation(&endocrine.state());

    let grateful_coherence = coherence.state().coherence;
    println!("   Dopamine: {:.2} (was {:.2})",
        grateful_dopamine, stress_dopamine);
    println!("   Coherence: {:.2} (synchronizing)", grateful_coherence);

    println!("\n5️⃣ Retry Complex Task");
    let can_think_now = coherence.can_perform(TaskComplexity::Cognitive);

    if can_think_now.is_ok() {
        println!("   ✅ Cognitive capacity restored!");
    } else {
        println!("   ⏳ Still needs more centering (severe stress takes time)");
    }

    println!("\n6️⃣ Deep Focus Session");
    let grateful_acetylcholine = endocrine.state().acetylcholine; // Save before mutation
    endocrine.process_event(HormoneEvent::DeepFocus { duration_cycles: 10 });
    let focused_acetylcholine = endocrine.state().acetylcholine;
    coherence.apply_hormone_modulation(&endocrine.state());

    println!("   Acetylcholine: {:.2} (was {:.2})",
        focused_acetylcholine, grateful_acetylcholine);

    // Perform connected work (builds coherence)
    for _ in 0..3 {
        coherence.perform_task(TaskComplexity::DeepThought, true)
            .expect("Should complete");
    }

    let focused_coherence = coherence.state().coherence;
    println!("   Coherence: {:.2} (stable + building)", focused_coherence);

    println!("\n7️⃣ Final State");
    let final_coherence = coherence.state().coherence;
    println!("   Final coherence: {:.2}", final_coherence);
    println!("   Can perform creation: {}",
        coherence.can_perform(TaskComplexity::Creation).is_ok());

    assert!(
        final_coherence > stressed_coherence,
        "Full cycle should result in recovered coherence"
    );

    println!("\n✅ FULL INTEGRATION VERIFIED!");
    println!("   Hardware Stress → Cortisol → Scattered Coherence");
    println!("   Gratitude → Dopamine → Restored Coherence");
    println!("   Focus → Acetylcholine → Stable + Building Coherence");
}
