// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for the brain/ and infrastructure/ modules.
//!
//! Tests cover:
//! - Actor system creation, message passing, and tick simulation
//! - Prefrontal cortex working memory, graduation, and executive decisions
//! - Social coherence agent observation, interaction recording, and cooperation
//! - LRU cache insertion, retrieval, eviction, and hit-rate statistics
//! - Pagination offset calculation and page slicing
//! - Metrics collector Phi tracking and Prometheus export

use symthaea::brain::actor_model::ActorSystemConfig;
use symthaea::brain::social_coherence::InteractionType;
use symthaea::brain::{
    ActorRole, ActorSystem, MessageType, PrefrontalConfig, PrefrontalCortex, SocialCoherence,
    SocialCoherenceConfig, WorkingMemoryItem,
};
use symthaea::infrastructure::{LruCache, MetricsCollector, Page, PageRequest};
use symthaea_core::hdc::ContinuousHV;

// ============================================================================
// Actor System Tests
// ============================================================================

#[test]
fn test_actor_system_creation_and_spawn() {
    let config = ActorSystemConfig {
        max_actors: 10,
        queue_capacity: 16,
        dimension: 64,
        ..Default::default()
    };
    let mut system = ActorSystem::new(config);

    let sensor = system.spawn("sensor-1", ActorRole::Sensor);
    assert!(sensor.is_some(), "Spawning within capacity should succeed");
    assert_eq!(system.actor_count(), 1);

    let processor = system.spawn("processor-1", ActorRole::Processor);
    assert!(processor.is_some());
    assert_eq!(system.actor_count(), 2);
}

#[test]
fn test_actor_system_max_actors_enforced() {
    let config = ActorSystemConfig {
        max_actors: 2,
        dimension: 32,
        ..Default::default()
    };
    let mut system = ActorSystem::new(config);

    assert!(system.spawn("a1", ActorRole::Sensor).is_some());
    assert!(system.spawn("a2", ActorRole::Processor).is_some());
    assert!(
        system.spawn("a3", ActorRole::Effector).is_none(),
        "Spawning beyond max_actors should return None"
    );
    assert_eq!(system.actor_count(), 2);
}

#[test]
fn test_actor_system_message_passing() {
    let config = ActorSystemConfig {
        dimension: 64,
        ..Default::default()
    };
    let mut system = ActorSystem::new(config);

    let sender = system.spawn("sender", ActorRole::Sensor).unwrap();
    let receiver = system.spawn("receiver", ActorRole::Processor).unwrap();

    system.connect(&sender, &receiver);

    let content = ContinuousHV::random(64, 0xDEAD_0001);
    let msg_id = system.send(&sender, &receiver, MessageType::Query, content);
    assert!(msg_id > 0, "Message ID should be positive");

    // Before tick, message is in the queue
    assert_eq!(system.stats().total_messages, 1);

    // After tick, message is delivered and processed
    system.tick();

    let recv = system.get_actor(&receiver).unwrap();
    assert!(
        recv.stats.messages_received > 0,
        "Receiver should have received at least one message"
    );
}

#[test]
fn test_actor_system_broadcast() {
    let config = ActorSystemConfig {
        dimension: 32,
        ..Default::default()
    };
    let mut system = ActorSystem::new(config);

    let coord = system.spawn("coord", ActorRole::Coordinator).unwrap();
    system.spawn("p1", ActorRole::Processor);
    system.spawn("p2", ActorRole::Processor);
    system.spawn("p3", ActorRole::Processor);

    let content = ContinuousHV::random(32, 0xBCA5_0001);
    system.broadcast(&coord, ActorRole::Processor, content);
    system.tick();

    // All 3 processors should have received broadcast messages
    let processors = system.actors_by_role(ActorRole::Processor);
    assert_eq!(processors.len(), 3);
    for proc_actor in &processors {
        assert!(
            proc_actor.stats.messages_received > 0,
            "Processor {} should have received the broadcast",
            proc_actor.id
        );
    }
}

#[test]
fn test_actor_system_tick_cycles() {
    let config = ActorSystemConfig {
        dimension: 32,
        ..Default::default()
    };
    let mut system = ActorSystem::new(config);

    let s1 = system.spawn("sensor", ActorRole::Sensor).unwrap();
    let p1 = system.spawn("proc", ActorRole::Processor).unwrap();
    system.connect(&s1, &p1);

    // Send an activation message and run several ticks
    system.send(
        &s1,
        &p1,
        MessageType::Activate,
        ContinuousHV::random(32, 0xFEED_0001),
    );

    system.run(10);

    assert_eq!(system.stats().total_ticks, 10);
    assert_eq!(system.current_tick(), 10);
    assert!(
        system.stats().total_messages >= 1,
        "At least the initial message should have been tracked"
    );
}

// ============================================================================
// Prefrontal Cortex Tests
// ============================================================================

#[test]
fn test_prefrontal_working_memory_and_graduation() {
    let config = PrefrontalConfig {
        working_memory_capacity: 3,
        dimension: 32,
        attention_decay: 0.05,
        ..Default::default()
    };
    let mut pfc = PrefrontalCortex::new(config);

    // Add three items at full activation
    for i in 0..3 {
        let item = WorkingMemoryItem::new(
            format!("item-{}", i),
            ContinuousHV::random(32, 0xCA00 + i as u64),
        );
        pfc.add_to_memory(item);
    }
    assert_eq!(pfc.memory_contents().len(), 3);

    // Manually lower the first item's activation so it will be evicted
    // when we add a fourth. We modify through add_to_memory (rehearsal) instead:
    // Add a fourth item -- the lowest activation item should be evicted
    let mut low_item = WorkingMemoryItem::new("low-act", ContinuousHV::random(32, 0xCA10));
    low_item.activation = 0.5; // Above graduation threshold (0.3)
    pfc.add_to_memory(low_item);

    // Now add a 5th item to force eviction of lowest
    let new_item = WorkingMemoryItem::new("new-5", ContinuousHV::random(32, 0xCA20));
    pfc.add_to_memory(new_item);

    // Check graduates
    let graduates = pfc.drain_graduates();
    // The evicted item should have been graduated if its activation >= 0.3
    assert!(
        !graduates.is_empty(),
        "Evicted item with activation >= 0.3 should graduate"
    );
    assert!(pfc.stats().graduations > 0);
}

#[test]
fn test_prefrontal_focus_and_rehearsal() {
    let config = PrefrontalConfig {
        dimension: 32,
        ..Default::default()
    };
    let mut pfc = PrefrontalCortex::new(config);

    let item = WorkingMemoryItem::new("focus-target", ContinuousHV::random(32, 0xF0C0));
    pfc.add_to_memory(item);

    assert!(pfc.set_focus("focus-target"));
    assert_eq!(pfc.current_focus(), Some("focus-target"));

    // Rehearsal should have boosted the item
    let mem_item = pfc.get_memory_item("focus-target").unwrap();
    assert!(
        mem_item.rehearsal_count >= 1,
        "Setting focus should rehearse the item"
    );
}

#[test]
fn test_prefrontal_decay_via_tick() {
    let config = PrefrontalConfig {
        working_memory_capacity: 5,
        dimension: 32,
        attention_decay: 0.2,
        ..Default::default()
    };
    let mut pfc = PrefrontalCortex::new(config);

    let item = WorkingMemoryItem::new("decay-me", ContinuousHV::random(32, 0xDEC0));
    pfc.add_to_memory(item);

    let initial = pfc.get_memory_item("decay-me").unwrap().activation;

    // Tick several times to decay
    for _ in 0..3 {
        pfc.tick();
    }

    // After 3 ticks at 0.2 decay, activation should have dropped
    let after = pfc.get_memory_item("decay-me");
    if let Some(item) = after {
        assert!(
            item.activation < initial,
            "Activation should decrease after ticks: before={}, after={}",
            initial,
            item.activation
        );
    }
    // If the item was evicted, that also confirms decay worked
}

// ============================================================================
// Social Coherence Tests
// ============================================================================

#[test]
fn test_social_coherence_observation_and_cooperation() {
    let config = SocialCoherenceConfig {
        dimension: 32,
        cooperation_threshold: 0.3,
        ..Default::default()
    };
    let mut sc = SocialCoherence::new(config);

    let behavior = ContinuousHV::random(32, 0xBEE1);
    let context = ContinuousHV::random(32, 0xBEE2);

    // Observe an agent
    sc.observe_agent("alice", &behavior, &context);
    let model = sc.get_mental_model("alice");
    assert!(model.is_some(), "Mental model should be created on observe");
    assert_eq!(model.unwrap().observation_count, 1);

    // Default cooperation with unknown agents (no interactions yet)
    assert!(
        sc.should_cooperate("alice"),
        "Should cooperate with unknown agent by default"
    );

    // Build positive relationship
    for i in 0..5 {
        sc.record_interaction(
            "alice",
            InteractionType::Cooperation,
            0.8,
            ContinuousHV::random(32, 0xBEE3 + i),
            "helped",
            "thanked",
        );
    }

    assert!(
        sc.should_cooperate("alice"),
        "Should cooperate after positive interactions"
    );

    let rel = sc.get_relationship("alice").unwrap();
    assert!(
        rel.trust > 0.5,
        "Trust should be above baseline after positive interactions: got {}",
        rel.trust
    );
    assert_eq!(rel.positive_interactions, 5);
}

#[test]
fn test_social_coherence_rival_detection() {
    let config = SocialCoherenceConfig {
        dimension: 32,
        cooperation_threshold: 0.3,
        ..Default::default()
    };
    let mut sc = SocialCoherence::new(config);

    // Build negative relationship
    for i in 0..10 {
        sc.record_interaction(
            "adversary",
            InteractionType::Conflict,
            -0.7,
            ContinuousHV::random(32, 0xAD00 + i),
            "proposed",
            "rejected",
        );
    }

    assert!(
        !sc.should_cooperate("adversary"),
        "Should not cooperate with untrusted agent"
    );

    let rel = sc.get_relationship("adversary").unwrap();
    assert!(
        rel.trust < 0.3,
        "Trust should be low after negative interactions: got {}",
        rel.trust
    );
}

#[test]
fn test_social_coherence_predict_response() {
    let config = SocialCoherenceConfig {
        dimension: 32,
        ..Default::default()
    };
    let mut sc = SocialCoherence::new(config);

    // Observe agent first to create a mental model
    let behavior = ContinuousHV::random(32, 0xD001);
    let context = ContinuousHV::random(32, 0xD002);
    sc.observe_agent("bob", &behavior, &context);

    // Record an interaction to establish a relationship
    sc.record_interaction(
        "bob",
        InteractionType::Communication,
        0.5,
        ContinuousHV::random(32, 0xD003),
        "asked",
        "answered",
    );

    // Now predict response
    let action = ContinuousHV::random(32, 0xD004);
    let prediction = sc.predict_response("bob", &action);
    assert!(
        prediction.is_some(),
        "Should produce a prediction for observed agent"
    );

    let result = prediction.unwrap();
    assert!(result.confidence.is_finite(), "Confidence should be finite");
    assert!(
        result.risk.is_finite() && result.risk >= 0.0,
        "Risk should be non-negative and finite"
    );

    // Unknown agent should return None
    assert!(sc.predict_response("unknown-agent", &action).is_none());
}

// ============================================================================
// LRU Cache Tests
// ============================================================================

#[test]
fn test_lru_cache_insert_get_eviction() {
    let mut cache: LruCache<String, i32> = LruCache::new(3);

    cache.insert("a".to_string(), 10);
    cache.insert("b".to_string(), 20);
    cache.insert("c".to_string(), 30);

    assert_eq!(cache.get(&"a".to_string()), Some(10));
    assert_eq!(cache.get(&"b".to_string()), Some(20));
    assert_eq!(cache.get(&"c".to_string()), Some(30));
    assert_eq!(cache.len(), 3);

    // Insert a fourth element -- should evict "a" (LRU after "b" and "c" were accessed by get())
    // Actually, we just accessed "a" above, so the LRU is now the one accessed earliest.
    // Access order after gets: a, b, c (all accessed in order). Since we get "a" first, then "b",
    // then "c", the access order is [a, b, c]. Inserting "d" should evict "a" (first in order).
    // Wait -- after get("a"), get("b"), get("c"), the access order is [a, b, c].
    // Actually let's re-read: access order is updated on get. After insert a,b,c: [a,b,c].
    // get(a) -> touch(a) -> [b,c,a]. get(b) -> touch(b) -> [c,a,b]. get(c) -> touch(c) -> [a,b,c].
    // So inserting "d" evicts "a" (first in access_order).

    cache.insert("d".to_string(), 40);
    assert_eq!(cache.len(), 3);
    assert_eq!(
        cache.get(&"a".to_string()),
        None,
        "a should have been evicted"
    );
    assert_eq!(cache.get(&"d".to_string()), Some(40));

    assert_eq!(cache.stats().evictions, 1);
}

#[test]
fn test_lru_cache_access_order_preserves_recently_used() {
    let mut cache: LruCache<String, i32> = LruCache::new(2);

    cache.insert("x".to_string(), 1);
    cache.insert("y".to_string(), 2);

    // Access "x" to make it most recently used
    cache.get(&"x".to_string());

    // Insert "z" -- should evict "y" (LRU)
    cache.insert("z".to_string(), 3);

    assert_eq!(
        cache.get(&"x".to_string()),
        Some(1),
        "x should survive (recently used)"
    );
    assert_eq!(
        cache.get(&"y".to_string()),
        None,
        "y should have been evicted"
    );
    assert_eq!(cache.get(&"z".to_string()), Some(3));
}

#[test]
fn test_lru_cache_stats_hit_rate() {
    let mut cache: LruCache<String, i32> = LruCache::new(10);

    cache.insert("hit".to_string(), 42);

    // 3 hits
    for _ in 0..3 {
        assert_eq!(cache.get(&"hit".to_string()), Some(42));
    }
    // 2 misses
    for _ in 0..2 {
        assert_eq!(cache.get(&"miss".to_string()), None);
    }

    let stats = cache.stats();
    assert_eq!(stats.hits, 3);
    assert_eq!(stats.misses, 2);

    // Hit rate should be 60%
    let hit_rate = stats.hit_rate();
    assert!(
        (hit_rate - 60.0).abs() < 0.01,
        "Hit rate should be 60%, got {}",
        hit_rate
    );
}

#[test]
fn test_lru_cache_remove_and_clear() {
    let mut cache: LruCache<String, i32> = LruCache::new(5);

    cache.insert("a".to_string(), 1);
    cache.insert("b".to_string(), 2);

    assert!(cache.contains(&"a".to_string()));
    let removed = cache.remove(&"a".to_string());
    assert_eq!(removed, Some(1));
    assert!(!cache.contains(&"a".to_string()));
    assert_eq!(cache.len(), 1);

    cache.clear();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

// ============================================================================
// Pagination Tests
// ============================================================================

#[test]
fn test_page_request_offset_calculation() {
    let req = PageRequest::new(0, 25);
    assert_eq!(req.offset(), 0, "First page offset should be 0");
    assert_eq!(req.page_size, 25);

    let req2 = PageRequest::new(3, 10);
    assert_eq!(req2.offset(), 30, "Page 3 at size 10 should offset 30");

    let req3 = PageRequest::new(5, 50);
    assert_eq!(req3.offset(), 250);
}

#[test]
fn test_page_from_slice() {
    let data: Vec<i32> = (0..100).collect();

    // First page
    let req0 = PageRequest::new(0, 20);
    let page0 = Page::from_slice(&data, &req0);
    assert_eq!(page0.items.len(), 20);
    assert_eq!(page0.items[0], 0);
    assert_eq!(page0.items[19], 19);
    assert_eq!(page0.total_items, 100);
    assert_eq!(page0.total_pages, 5);
    assert!(!page0.has_prev);
    assert!(page0.has_next);

    // Middle page
    let req2 = PageRequest::new(2, 20);
    let page2 = Page::from_slice(&data, &req2);
    assert_eq!(page2.items[0], 40);
    assert!(page2.has_prev);
    assert!(page2.has_next);

    // Last page
    let req4 = PageRequest::new(4, 20);
    let page4 = Page::from_slice(&data, &req4);
    assert_eq!(page4.items.len(), 20);
    assert_eq!(page4.items[0], 80);
    assert!(page4.has_prev);
    assert!(!page4.has_next);

    // Page map
    let mapped = page0.map(|x| x * 2);
    assert_eq!(mapped.items[0], 0);
    assert_eq!(mapped.items[1], 2);
    assert_eq!(mapped.total_items, 100);
}

#[test]
fn test_page_empty_dataset() {
    let data: Vec<i32> = vec![];
    let req = PageRequest::new(0, 10);
    let page = Page::from_slice(&data, &req);

    assert!(page.items.is_empty());
    assert_eq!(page.total_items, 0);
    assert!(!page.has_next);
    assert!(!page.has_prev);
}

// ============================================================================
// Metrics Collector Tests
// ============================================================================

#[test]
fn test_metrics_collector_phi_tracking() {
    let collector = MetricsCollector::new();

    // Initial export should have phi = 0
    let output = collector.export();
    assert!(
        output.contains("symthaea_phi"),
        "Export should contain phi metric"
    );

    // Set phi and verify
    collector.set_phi(0.73);
    let output = collector.export();
    assert!(
        output.contains("0.73"),
        "Export should contain the phi value 0.73"
    );

    // Set coherence and consciousness level
    collector.set_coherence(0.91);
    collector.set_consciousness_level(0.65);
    let output = collector.export();
    assert!(output.contains("0.91"));
    assert!(output.contains("0.65"));
}

#[test]
fn test_metrics_collector_request_counting() {
    let collector = MetricsCollector::new();

    for _ in 0..5 {
        collector.inc_requests();
    }

    let output = collector.export();
    assert!(
        output.contains("symthaea_requests_total"),
        "Should have requests metric"
    );
    assert!(output.contains("5"), "Should show 5 total requests");
}

#[test]
fn test_metrics_collector_prometheus_format() {
    let collector = MetricsCollector::new();
    collector.set_phi(0.42);

    let output = collector.export();

    // Verify Prometheus exposition format
    assert!(output.contains("# HELP symthaea_phi"));
    assert!(output.contains("# TYPE symthaea_phi gauge"));
    assert!(output.contains("symthaea_phi 0.42"));
}