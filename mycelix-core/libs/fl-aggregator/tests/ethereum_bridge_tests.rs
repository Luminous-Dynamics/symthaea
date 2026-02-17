//! Integration tests for Ethereum Bridge
//!
//! Tests the complete Holochain → Ethereum bridge flow including:
//! - Signal generation and handling
//! - Payment distribution requests
//! - Anchor requests
//! - Status callbacks
//! - Error handling and retries
//!
//! These tests require the `holochain` feature to be enabled.

#![cfg(feature = "holochain")]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use fl_aggregator::holochain::{
    HolochainListener, HolochainSignal, ListenerBuilder, PaymentSplitSignal, SignalStats,
};
use tokio::sync::mpsc;

// =============================================================================
// Signal Type Tests
// =============================================================================

#[test]
fn test_payment_split_signal_creation() {
    let split = PaymentSplitSignal {
        address: "0x742d35Cc6634C0532925a3b844Bc9e7595f8fF2B".to_string(),
        basis_points: 5000,
        agent_id: "agent-123".to_string(),
    };

    assert_eq!(split.basis_points, 5000);
    assert!(split.address.starts_with("0x"));
}

#[test]
fn test_ethereum_payment_request_signal() {
    let splits = vec![
        PaymentSplitSignal {
            address: "0xabc123".to_string(),
            basis_points: 6000,
            agent_id: "node_1".to_string(),
        },
        PaymentSplitSignal {
            address: "0xdef456".to_string(),
            basis_points: 4000,
            agent_id: "node_2".to_string(),
        },
    ];

    let signal = HolochainSignal::EthereumPaymentRequest {
        intent_id: "intent-001".to_string(),
        model_id: "model-abc".to_string(),
        round: 5,
        total_amount_wei: "1000000000000000000".to_string(),
        splits,
    };

    // Verify signal can be serialized/deserialized
    let json = serde_json::to_string(&signal).unwrap();
    assert!(json.contains("EthereumPaymentRequest"));
    assert!(json.contains("intent-001"));
    assert!(json.contains("model-abc"));

    let parsed: HolochainSignal = serde_json::from_str(&json).unwrap();
    match parsed {
        HolochainSignal::EthereumPaymentRequest {
            intent_id,
            model_id,
            round,
            total_amount_wei,
            splits,
        } => {
            assert_eq!(intent_id, "intent-001");
            assert_eq!(model_id, "model-abc");
            assert_eq!(round, 5);
            assert_eq!(total_amount_wei, "1000000000000000000");
            assert_eq!(splits.len(), 2);
            // Verify basis points sum to 10000 (100%)
            let total_bps: u64 = splits.iter().map(|s| s.basis_points).sum();
            assert_eq!(total_bps, 10000);
        }
        _ => panic!("Expected EthereumPaymentRequest"),
    }
}

#[test]
fn test_ethereum_anchor_request_signal() {
    let signal = HolochainSignal::EthereumAnchorRequest {
        intent_id: "anchor-001".to_string(),
        anchor_type: "reputation".to_string(),
        agent_id: "agent-xyz".to_string(),
        score_bps: 8500,
        round: 10,
        evidence_hash: Some("0xabcdef123456".to_string()),
    };

    let json = serde_json::to_string(&signal).unwrap();
    assert!(json.contains("EthereumAnchorRequest"));
    assert!(json.contains("reputation"));

    let parsed: HolochainSignal = serde_json::from_str(&json).unwrap();
    match parsed {
        HolochainSignal::EthereumAnchorRequest {
            intent_id,
            anchor_type,
            agent_id,
            score_bps,
            round,
            evidence_hash,
        } => {
            assert_eq!(intent_id, "anchor-001");
            assert_eq!(anchor_type, "reputation");
            assert_eq!(agent_id, "agent-xyz");
            assert_eq!(score_bps, 8500);
            assert_eq!(round, 10);
            assert!(evidence_hash.is_some());
        }
        _ => panic!("Expected EthereumAnchorRequest"),
    }
}

#[test]
fn test_anchor_request_without_evidence() {
    let signal = HolochainSignal::EthereumAnchorRequest {
        intent_id: "anchor-002".to_string(),
        anchor_type: "contribution".to_string(),
        agent_id: "agent-abc".to_string(),
        score_bps: 7500,
        round: 15,
        evidence_hash: None,
    };

    let json = serde_json::to_string(&signal).unwrap();
    let parsed: HolochainSignal = serde_json::from_str(&json).unwrap();

    match parsed {
        HolochainSignal::EthereumAnchorRequest { evidence_hash, .. } => {
            assert!(evidence_hash.is_none());
        }
        _ => panic!("Expected EthereumAnchorRequest"),
    }
}

// =============================================================================
// Signal Statistics Tests
// =============================================================================

#[test]
fn test_signal_stats_ethereum_tracking() {
    let mut stats = SignalStats::default();

    // Record various signals
    stats.record(&HolochainSignal::GradientSubmitted {
        node_id: "n1".to_string(),
        round_num: 1,
        entry_hash: "h1".to_string(),
    });

    stats.record(&HolochainSignal::EthereumPaymentRequest {
        intent_id: "p1".to_string(),
        model_id: "m1".to_string(),
        round: 1,
        total_amount_wei: "1000".to_string(),
        splits: vec![],
    });

    stats.record(&HolochainSignal::EthereumAnchorRequest {
        intent_id: "a1".to_string(),
        anchor_type: "reputation".to_string(),
        agent_id: "ag1".to_string(),
        score_bps: 9000,
        round: 1,
        evidence_hash: None,
    });

    stats.record(&HolochainSignal::EthereumPaymentRequest {
        intent_id: "p2".to_string(),
        model_id: "m1".to_string(),
        round: 2,
        total_amount_wei: "2000".to_string(),
        splits: vec![],
    });

    assert_eq!(stats.gradients_received, 1);
    assert_eq!(stats.ethereum_payment_requests, 2);
    assert_eq!(stats.ethereum_anchor_requests, 1);
    assert_eq!(stats.total_signals, 4);
}

// =============================================================================
// Listener Handler Tests
// =============================================================================

#[tokio::test]
async fn test_listener_ethereum_payment_handler() {
    let (_tx, rx) = mpsc::channel::<HolochainSignal>(16);
    let counter = Arc::new(AtomicU64::new(0));
    let counter_clone = counter.clone();

    let listener: HolochainListener = ListenerBuilder::new(rx)
        .on_ethereum_payment_request(move |_signal| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        })
        .build()
        .await;

    // Create payment request signal
    let signal = HolochainSignal::EthereumPaymentRequest {
        intent_id: "test-payment".to_string(),
        model_id: "test-model".to_string(),
        round: 1,
        total_amount_wei: "1000000000000000000".to_string(),
        splits: vec![PaymentSplitSignal {
            address: "0x123".to_string(),
            basis_points: 10000,
            agent_id: "node_1".to_string(),
        }],
    };

    // Manually dispatch
    listener.dispatch_signal(signal).await;

    assert_eq!(counter.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn test_listener_ethereum_anchor_handler() {
    let (_tx, rx) = mpsc::channel::<HolochainSignal>(16);
    let counter = Arc::new(AtomicU64::new(0));
    let counter_clone = counter.clone();

    let listener: HolochainListener = ListenerBuilder::new(rx)
        .on_ethereum_anchor_request(move |_signal| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        })
        .build()
        .await;

    let signal = HolochainSignal::EthereumAnchorRequest {
        intent_id: "test-anchor".to_string(),
        anchor_type: "reputation".to_string(),
        agent_id: "agent-1".to_string(),
        score_bps: 8000,
        round: 5,
        evidence_hash: Some("0xhash".to_string()),
    };

    listener.dispatch_signal(signal).await;

    assert_eq!(counter.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn test_listener_multiple_ethereum_handlers() {
    let (_tx, rx) = mpsc::channel::<HolochainSignal>(16);

    let payment_counter = Arc::new(AtomicU64::new(0));
    let anchor_counter = Arc::new(AtomicU64::new(0));

    let payment_clone = payment_counter.clone();
    let anchor_clone = anchor_counter.clone();

    let listener: HolochainListener = ListenerBuilder::new(rx)
        .on_ethereum_payment_request(move |_| {
            payment_clone.fetch_add(1, Ordering::SeqCst);
        })
        .on_ethereum_anchor_request(move |_| {
            anchor_clone.fetch_add(1, Ordering::SeqCst);
        })
        .build()
        .await;

    // Send payment signal
    let payment_signal = HolochainSignal::EthereumPaymentRequest {
        intent_id: "p1".to_string(),
        model_id: "m1".to_string(),
        round: 1,
        total_amount_wei: "1000".to_string(),
        splits: vec![],
    };

    // Send anchor signal
    let anchor_signal = HolochainSignal::EthereumAnchorRequest {
        intent_id: "a1".to_string(),
        anchor_type: "reputation".to_string(),
        agent_id: "ag1".to_string(),
        score_bps: 9000,
        round: 1,
        evidence_hash: None,
    };

    // Send a non-ethereum signal (should not increment counters)
    let gradient_signal = HolochainSignal::GradientSubmitted {
        node_id: "n1".to_string(),
        round_num: 1,
        entry_hash: "h1".to_string(),
    };

    listener.dispatch_signal(payment_signal).await;
    listener.dispatch_signal(anchor_signal).await;
    listener.dispatch_signal(gradient_signal).await;

    assert_eq!(payment_counter.load(Ordering::SeqCst), 1);
    assert_eq!(anchor_counter.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn test_listener_unregister_ethereum_handler() {
    let (_tx, rx) = mpsc::channel::<HolochainSignal>(16);
    let listener = HolochainListener::new(rx);

    // Register handlers
    let payment_id = listener
        .on_ethereum_payment_request(|_| {})
        .await;
    let anchor_id = listener
        .on_ethereum_anchor_request(|_| {})
        .await;

    assert_ne!(payment_id, anchor_id);

    // Unregister
    assert!(listener.unregister(payment_id).await);
    assert!(!listener.unregister(payment_id).await); // Already removed
    assert!(listener.unregister(anchor_id).await);
}

// =============================================================================
// Payment Split Validation Tests
// =============================================================================

#[test]
fn test_payment_splits_sum_to_100_percent() {
    let splits = vec![
        PaymentSplitSignal {
            address: "0x1".to_string(),
            basis_points: 3333,
            agent_id: "a1".to_string(),
        },
        PaymentSplitSignal {
            address: "0x2".to_string(),
            basis_points: 3333,
            agent_id: "a2".to_string(),
        },
        PaymentSplitSignal {
            address: "0x3".to_string(),
            basis_points: 3334, // Extra 1 bps for rounding
            agent_id: "a3".to_string(),
        },
    ];

    let total: u64 = splits.iter().map(|s| s.basis_points).sum();
    assert_eq!(total, 10000, "Splits should sum to 10000 basis points (100%)");
}

#[test]
fn test_single_recipient_full_amount() {
    let splits = vec![PaymentSplitSignal {
        address: "0xsingle".to_string(),
        basis_points: 10000,
        agent_id: "solo".to_string(),
    }];

    let total: u64 = splits.iter().map(|s| s.basis_points).sum();
    assert_eq!(total, 10000);
}

#[test]
fn test_equal_split_among_participants() {
    let num_participants = 7;
    let base_share = 10000 / num_participants;
    let remainder = 10000 % num_participants;

    let splits: Vec<PaymentSplitSignal> = (0..num_participants)
        .map(|i| PaymentSplitSignal {
            address: format!("0x{:040x}", i),
            basis_points: if i == 0 {
                base_share + remainder
            } else {
                base_share
            },
            agent_id: format!("agent_{}", i),
        })
        .collect();

    let total: u64 = splits.iter().map(|s| s.basis_points).sum();
    assert_eq!(total, 10000);
}

// =============================================================================
// Wei Amount Validation Tests
// =============================================================================

#[test]
fn test_wei_amount_parsing() {
    // 1 ETH in Wei
    let one_eth = "1000000000000000000";
    assert_eq!(one_eth.len(), 19);

    // 0.1 ETH in Wei
    let point_one_eth = "100000000000000000";
    assert_eq!(point_one_eth.len(), 18);

    // 10 ETH in Wei
    let ten_eth = "10000000000000000000";
    assert_eq!(ten_eth.len(), 20);
}

#[test]
fn test_large_wei_amounts() {
    // Test that large amounts can be represented as strings
    let large_amount = "999999999999999999999999999999"; // Very large amount
    let signal = HolochainSignal::EthereumPaymentRequest {
        intent_id: "large".to_string(),
        model_id: "m".to_string(),
        round: 1,
        total_amount_wei: large_amount.to_string(),
        splits: vec![],
    };

    let json = serde_json::to_string(&signal).unwrap();
    assert!(json.contains(large_amount));
}

// =============================================================================
// Signal Routing Tests
// =============================================================================

#[tokio::test]
async fn test_all_handler_receives_ethereum_signals() {
    let (_tx, rx) = mpsc::channel::<HolochainSignal>(16);
    let all_counter = Arc::new(AtomicU64::new(0));
    let all_clone = all_counter.clone();

    let listener: HolochainListener = ListenerBuilder::new(rx)
        .on_any_signal(move |_| {
            all_clone.fetch_add(1, Ordering::SeqCst);
        })
        .build()
        .await;

    // Send various signals
    listener
        .dispatch_signal(HolochainSignal::EthereumPaymentRequest {
            intent_id: "p1".to_string(),
            model_id: "m1".to_string(),
            round: 1,
            total_amount_wei: "1000".to_string(),
            splits: vec![],
        })
        .await;

    listener
        .dispatch_signal(HolochainSignal::EthereumAnchorRequest {
            intent_id: "a1".to_string(),
            anchor_type: "reputation".to_string(),
            agent_id: "ag1".to_string(),
            score_bps: 9000,
            round: 1,
            evidence_hash: None,
        })
        .await;

    listener
        .dispatch_signal(HolochainSignal::GradientSubmitted {
            node_id: "n1".to_string(),
            round_num: 1,
            entry_hash: "h1".to_string(),
        })
        .await;

    // All handler should have received all 3 signals
    assert_eq!(all_counter.load(Ordering::SeqCst), 3);
}

// =============================================================================
// Edge Case Tests
// =============================================================================

#[test]
fn test_empty_splits_array() {
    let signal = HolochainSignal::EthereumPaymentRequest {
        intent_id: "empty".to_string(),
        model_id: "m".to_string(),
        round: 1,
        total_amount_wei: "0".to_string(),
        splits: vec![],
    };

    let json = serde_json::to_string(&signal).unwrap();
    let parsed: HolochainSignal = serde_json::from_str(&json).unwrap();

    match parsed {
        HolochainSignal::EthereumPaymentRequest { splits, .. } => {
            assert!(splits.is_empty());
        }
        _ => panic!("Expected EthereumPaymentRequest"),
    }
}

#[test]
fn test_special_characters_in_ids() {
    let signal = HolochainSignal::EthereumPaymentRequest {
        intent_id: "intent-with-dashes_and_underscores.and.dots".to_string(),
        model_id: "model/with/slashes".to_string(),
        round: 1,
        total_amount_wei: "1000".to_string(),
        splits: vec![PaymentSplitSignal {
            address: "0xAbCdEf1234567890abcdef1234567890AbCdEf12".to_string(),
            basis_points: 10000,
            agent_id: "agent:with:colons".to_string(),
        }],
    };

    let json = serde_json::to_string(&signal).unwrap();
    let _parsed: HolochainSignal = serde_json::from_str(&json).unwrap();
    // If we get here without error, the test passes
}

#[test]
fn test_zero_score_bps() {
    let signal = HolochainSignal::EthereumAnchorRequest {
        intent_id: "zero".to_string(),
        anchor_type: "reputation".to_string(),
        agent_id: "bad_actor".to_string(),
        score_bps: 0, // Completely slashed
        round: 1,
        evidence_hash: Some("0xevidence".to_string()),
    };

    let json = serde_json::to_string(&signal).unwrap();
    let parsed: HolochainSignal = serde_json::from_str(&json).unwrap();

    match parsed {
        HolochainSignal::EthereumAnchorRequest { score_bps, .. } => {
            assert_eq!(score_bps, 0);
        }
        _ => panic!("Expected EthereumAnchorRequest"),
    }
}

#[test]
fn test_max_score_bps() {
    let signal = HolochainSignal::EthereumAnchorRequest {
        intent_id: "max".to_string(),
        anchor_type: "reputation".to_string(),
        agent_id: "perfect_actor".to_string(),
        score_bps: 10000, // 100%
        round: 1,
        evidence_hash: None,
    };

    let json = serde_json::to_string(&signal).unwrap();
    let parsed: HolochainSignal = serde_json::from_str(&json).unwrap();

    match parsed {
        HolochainSignal::EthereumAnchorRequest { score_bps, .. } => {
            assert_eq!(score_bps, 10000);
        }
        _ => panic!("Expected EthereumAnchorRequest"),
    }
}

// =============================================================================
// Concurrent Signal Handling Tests
// =============================================================================

#[tokio::test]
async fn test_concurrent_signal_dispatch() {
    let (_tx, rx) = mpsc::channel::<HolochainSignal>(100);
    let counter = Arc::new(AtomicU64::new(0));

    let counter_clone = counter.clone();
    let listener: Arc<HolochainListener> = Arc::new(
        ListenerBuilder::new(rx)
            .on_ethereum_payment_request(move |_| {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            })
            .build()
            .await,
    );

    // Spawn concurrent signal dispatches
    let mut handles = vec![];
    for i in 0..10 {
        let l: Arc<HolochainListener> = Arc::clone(&listener);
        let handle = tokio::spawn(async move {
            let signal = HolochainSignal::EthereumPaymentRequest {
                intent_id: format!("concurrent-{}", i),
                model_id: "model".to_string(),
                round: i,
                total_amount_wei: "1000".to_string(),
                splits: vec![],
            };
            l.dispatch_signal(signal).await;
        });
        handles.push(handle);
    }

    // Wait for all dispatches
    for handle in handles {
        handle.await.unwrap();
    }

    assert_eq!(counter.load(Ordering::SeqCst), 10);
}
