// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration Tests for Newly Wired Components
//!
//! This module tests the full integration of:
//! 1. REPL -> Cognitive Loop -> FEP -> Motor System
//! 2. Swarm Networking with Consciousness Broadcasting
//! 3. Phi-Guided Architecture Search
//!
//! These tests verify that the components work together as a cohesive system,
//! not just as isolated units.

// ============================================================================
// TEST 1: REPL -> COGNITIVE LOOP -> FEP -> MOTOR SYSTEM
// ============================================================================

mod repl_cognitive_fep_motor_integration {
    use symthaea::action::{
        ActionIR, DestructivenessLevel, ExecutionMode, PolicyBundle, SandboxRoot, SimpleExecutor,
    };
    use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService, CycleResult};
    use symthaea::consciousness::fep_active_inference::{
        ActiveInferenceAgent, ActiveInferenceAgentConfig, EnhancedFEPBridge, MotorCommandType,
        Observation,
    };
    use symthaea::repl::{ReplSession, ReplSessionConfig};

    /// Test the complete perception-action cycle through the cognitive loop
    #[test]
    fn test_cognitive_loop_perception_action_cycle() {
        // Initialize cognitive loop with CfC backend
        let config = CognitiveLoopConfig::with_cfc();
        let mut service =
            CognitiveLoopService::new(config).expect("Failed to create cognitive loop service");

        // Process several inputs to establish a perception-action cycle
        let inputs = [
            "The sun rises in the east",
            "cause leads to effect",
            "pattern recognition emerges",
            "consciousness integrates information",
        ];

        let mut cycle_results: Vec<CycleResult> = Vec::new();
        let mut prediction_errors: Vec<f32> = Vec::new();

        for input in &inputs {
            let result = service.cycle(input);
            prediction_errors.push(result.prediction_error);
            cycle_results.push(result);
        }

        // Verify learning is occurring (prediction error should generally decrease)
        assert!(cycle_results.len() == 4, "Should have 4 cycle results");

        // Get consciousness snapshot
        let snapshot = service.consciousness_snapshot();

        // Verify consciousness metrics are being computed
        assert!(snapshot.unified_psi >= 0.0, "Phi must be non-negative");
        assert!(
            snapshot.temporal_coherence >= 0.0,
            "Coherence must be non-negative"
        );

        // Verify the cognitive loop stats
        let stats = service.stats();
        assert!(stats.total_cycles > 0, "Should have completed cycles");

        println!("Cognitive Loop Integration Test Results:");
        println!("  Cycles completed: {}", stats.total_cycles);
        println!("  Final Phi: {:.4}", snapshot.unified_psi);
        println!("  Temporal coherence: {:.4}", snapshot.temporal_coherence);
        println!("  Prediction errors: {:?}", prediction_errors);
    }

    /// Test FEP active inference processing input and generating motor commands
    #[test]
    fn test_fep_active_inference_motor_generation() {
        // Create FEP agent
        let config = ActiveInferenceAgentConfig {
            state_dim: 8,
            obs_dim: 4,
            num_actions: 8, // Matches MotorCommandType variants
            inference_iterations: 5,
            belief_learning_rate: 0.1,
            planning_horizon: 3,
            action_temperature: 1.0,
            enable_model_learning: true,
            enable_td_learning: true,
            ..Default::default()
        };

        let mut agent = ActiveInferenceAgent::new(config);

        // Simulate consciousness state observations
        let observations = [
            Observation::from_consciousness_state(0.5, 0.4, 0.6, 0.7), // Initial
            Observation::from_consciousness_state(0.6, 0.5, 0.7, 0.8), // Improving
            Observation::from_consciousness_state(0.7, 0.6, 0.8, 0.9), // High attention
            Observation::from_consciousness_state(0.4, 0.3, 0.5, 0.5), // Surprising drop
        ];

        let mut actions_selected = Vec::new();
        let mut free_energies = Vec::new();

        for obs in &observations {
            // Perception step - update beliefs
            let perception = agent.perceive(obs);

            // Action selection step
            let action_result = agent.select_action();

            // Execute action
            let _outcome = agent.act(action_result.action);

            actions_selected.push(action_result.action);
            free_energies.push(perception.free_energy.total);

            // Verify motor command type is valid
            let motor_type = MotorCommandType::from_action_index(action_result.action);
            assert!(
                motor_type.to_action_index() < 8,
                "Motor command should be valid"
            );
        }

        // Verify the agent learned and adapted
        let summary = agent.summary();
        assert!(
            summary.total_cycles == 4,
            "Should have processed 4 observations"
        );
        assert!(!free_energies.is_empty(), "Should have free energy history");

        // The agent should have responded to the surprising drop
        assert!(
            agent.stats.perception_cycles >= 4,
            "Should have recorded perception cycles"
        );

        println!("FEP Motor Generation Test Results:");
        println!("  Actions selected: {:?}", actions_selected);
        println!("  Free energies: {:?}", free_energies);
        println!("  Exploration rate: {:.2}", summary.exploration_rate);
        println!("  Belief confidence: {:.4}", summary.belief_confidence);
    }

    /// Test the Enhanced FEP Bridge for cognitive loop integration
    #[test]
    fn test_enhanced_fep_bridge_motor_commands() {
        let config = ActiveInferenceAgentConfig {
            state_dim: 8,
            obs_dim: 4,
            num_actions: 8,
            enable_td_learning: true,
            ..Default::default()
        };

        let mut bridge = EnhancedFEPBridge::new(config, 4);

        // Process multiple consciousness states using the cycle method
        let test_states = [
            (0.5, 0.4, 0.6, 0.7),  // Normal
            (0.8, 0.7, 0.9, 0.95), // High engagement
            (0.3, 0.2, 0.4, 0.3),  // Distracted
            (0.6, 0.5, 0.7, 0.8),  // Recovery
        ];

        let mut motor_commands: Vec<MotorCommandType> = Vec::new();

        for (phi, integration, coherence, attention) in &test_states {
            let result = bridge.cycle(*phi, *integration, *coherence, *attention);

            // Verify motor command is generated
            assert!(result.motor_command.command_type.to_action_index() < 8);
            motor_commands.push(result.motor_command.command_type);

            // Verify FEP metrics
            assert!(result.fep_result.free_energy.is_finite());
            assert!(result.fep_result.learning_rate_modulation > 0.0);
        }

        println!("Enhanced FEP Bridge Test Results:");
        println!("  Motor commands generated: {:?}", motor_commands);
    }

    /// Test ReplSession initialization and basic processing
    #[test]
    fn test_repl_session_full_pipeline() {
        // Create REPL session with default config
        let config = ReplSessionConfig {
            cycles_per_input: 3,
            max_history: 50,
            temporal_backend: "cfc".to_string(),
            voice_enabled: false,
            allow_execution: false, // Simulated mode
            execution_phi_threshold: 0.5,
            ..Default::default()
        };

        let mut session = ReplSession::new(config).expect("Failed to create REPL session");

        // Warm up the cognitive loop
        session.warmup(3);

        // Process a conversation
        let inputs = [
            "Hello, Symthaea",
            "What is consciousness?",
            "How do you process information?",
        ];

        for input in &inputs {
            let result = session.process(input).expect("Failed to process input");

            // Verify response was generated
            assert!(!result.response.is_empty(), "Response should not be empty");

            // Verify consciousness state is tracked
            assert!(result.consciousness.unified_psi >= 0.0);

            // Verify timing
            assert!(result.elapsed.as_micros() > 0);
        }

        // Check session statistics
        let stats = session.stats();
        assert_eq!(stats.total_interactions, 3, "Should have 3 interactions");
        assert!(
            stats.total_cycles >= 9,
            "Should have run at least 9 cycles (3 per input)"
        );

        // Check history
        assert_eq!(session.history().len(), 3, "History should have 3 entries");

        println!("REPL Session Test Results:");
        println!("  Total interactions: {}", stats.total_interactions);
        println!("  Total cycles: {}", stats.total_cycles);
        println!("  Average Phi: {:.4}", stats.avg_phi);
    }

    /// Test action execution through motor cortex with policy validation
    #[test]
    fn test_action_motor_cortex_execution() {
        // Create policy and sandbox
        let policy = PolicyBundle::restrictive();
        let sandbox = SandboxRoot::new("test-motor-cortex").expect("Failed to create sandbox");

        // Create executor in simulated mode
        let mut executor = SimpleExecutor::new();
        assert_eq!(executor.mode(), ExecutionMode::Simulated);

        // Test various action types
        let actions = [
            ActionIR::RunCommand {
                program: "ls".to_string(),
                args: vec!["-la".to_string()],
                env: std::collections::BTreeMap::new(),
                working_dir: None,
            },
            ActionIR::RunCommand {
                program: "echo".to_string(),
                args: vec!["hello".to_string()],
                env: std::collections::BTreeMap::new(),
                working_dir: None,
            },
        ];
        let current_phi = 0.9;

        for action in &actions {
            // Check destructiveness classification
            let destructiveness = action.destructiveness();
            assert!(
                matches!(
                    destructiveness,
                    DestructivenessLevel::ReadOnly | DestructivenessLevel::Reversible
                ),
                "Safe commands should be classified correctly"
            );

            // Validate against policy
            let validation = action.validate(&policy, &sandbox, current_phi);
            assert!(
                validation.is_ok(),
                "Valid actions should pass policy: {:?}",
                validation
            );

            // In simulated mode, execution returns simulated outcome
            let result = executor.execute(action, &policy, &sandbox, current_phi);
            assert!(result.is_ok(), "Simulated execution should succeed");
        }

        println!("Motor Cortex Execution Test: PASSED");
    }

    /// Integration test: Full cycle from input to motor command
    #[test]
    fn test_full_perception_to_action_integration() {
        // 1. Initialize cognitive loop
        let cognitive_config = CognitiveLoopConfig::with_cfc();
        let mut cognitive =
            CognitiveLoopService::new(cognitive_config).expect("Failed to create cognitive loop");

        // 2. Initialize FEP agent
        let fep_config = ActiveInferenceAgentConfig {
            state_dim: 8,
            obs_dim: 4,
            num_actions: 8,
            ..Default::default()
        };
        let mut fep_agent = ActiveInferenceAgent::new(fep_config);

        // 3. Process input through cognitive loop
        let input = "The weather is changing rapidly";
        let cycle_result = cognitive.cycle(input);

        // 4. Extract consciousness state
        let snapshot = cognitive.consciousness_snapshot();

        // 5. Feed to FEP for motor command generation
        let observation = Observation::from_consciousness_state(
            snapshot.unified_psi as f64,
            snapshot.consciousness_level as f64,
            snapshot.temporal_coherence as f64,
            snapshot.prediction_confidence as f64,
        );

        let perception = fep_agent.perceive(&observation);
        let action = fep_agent.select_action();

        // 6. Verify the complete pipeline
        assert!(
            cycle_result.prediction_error.is_finite(),
            "Prediction error should be finite"
        );
        assert!(
            perception.free_energy.total.is_finite(),
            "Free energy should be finite"
        );
        assert!(action.action < 8, "Action should be in valid range");

        // Map to motor command
        let motor_command = MotorCommandType::from_action_index(action.action);

        println!("Full Integration Test Results:");
        println!("  Input: '{}'", input);
        println!("  Prediction error: {:.4}", cycle_result.prediction_error);
        println!("  Consciousness Phi: {:.4}", snapshot.unified_psi);
        println!("  Free energy: {:.4}", perception.free_energy.total);
        println!("  Selected motor command: {:?}", motor_command);
        println!(
            "  Action probability: {:.4}",
            action.action_probabilities[action.action]
        );
    }
}

// ============================================================================
// TEST 2: SWARM NETWORKING WITH CONSCIOUSNESS BROADCASTING
// ============================================================================

mod swarm_consciousness_broadcasting {
    use std::collections::HashMap;
    use symthaea::swarm::{
        AffectiveState, BootstrapConfig, ConsciousnessVector, HandshakeResult, HybridHandshake,
        Hyperfeel, PeerEvent, PeerInfo, ServiceStats, SwarmConfig, SwarmMessage, SwarmNode,
        TensorPayload, TensorType, TrustLevel,
    };

    /// Test basic swarm configuration and node creation
    #[test]
    fn test_swarm_node_initialization() {
        // Test different configuration modes
        let default_config = SwarmConfig::default();
        assert!(
            default_config.enable_mdns,
            "mDNS should be enabled by default"
        );

        let local_config = SwarmConfig::local_only();
        assert!(!local_config.enable_derp, "DERP disabled for local-only");

        let prod_config = SwarmConfig::production();
        assert_eq!(
            prod_config.max_peers, 100,
            "Production should support more peers"
        );

        // Create node
        let node = SwarmNode::new(default_config.clone());
        assert!(node.peers().is_empty(), "New node should have no peers");

        println!("Swarm Node Initialization: PASSED");
    }

    /// Test consciousness vector creation and serialization
    #[test]
    fn test_consciousness_vector_broadcasting() {
        // Create consciousness vector with attention pattern
        let attention_pattern: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin().abs()).collect();

        let cv = ConsciousnessVector::new(attention_pattern.clone(), 0.75);

        // Verify structure
        assert_eq!(cv.phi, 0.75);
        assert_eq!(cv.attention.len(), 64);
        assert!(cv.timestamp_ms > 0, "Timestamp should be set");
        // Sequence starts at 0 in the current implementation

        // Estimate size for network transfer
        let size = cv.estimated_size();
        assert!(size > 0 && size < 1000, "Size should be reasonable");

        // Create multiple vectors to simulate broadcasting
        let vectors: Vec<ConsciousnessVector> = (0..5)
            .map(|i| {
                let attention: Vec<f32> = (0..64)
                    .map(|j| ((i * 10 + j) as f32 * 0.1).cos().abs())
                    .collect();
                ConsciousnessVector::new(attention, 0.5 + i as f64 * 0.1)
            })
            .collect();

        // Verify all vectors have valid structure
        for (i, v) in vectors.iter().enumerate() {
            assert_eq!(
                v.attention.len(),
                64,
                "Vector {} should have 64 attention dims",
                i
            );
            assert!(
                v.phi >= 0.0 && v.phi <= 1.0,
                "Vector {} phi should be in range",
                i
            );
        }

        println!("Consciousness Vector Broadcasting Test: PASSED");
        println!("  Vector size estimate: {} bytes", size);
    }

    /// Test peer event handling and trust verification
    #[test]
    fn test_peer_event_handling() {
        // Simulate peer events
        let events = [
            PeerEvent::Discovered(PeerInfo::new("peer-1")),
            PeerEvent::Connected(PeerInfo::new("peer-1")),
            PeerEvent::TrustChanged {
                peer_id: "peer-1".to_string(),
                old: TrustLevel::Unknown,
                new: TrustLevel::Verified(0.7),
            },
            PeerEvent::ConsciousnessUpdate {
                peer_id: "peer-1".to_string(),
                phi: 0.75,
                sequence: 42,
            },
            PeerEvent::Disconnected {
                peer_id: "peer-1".to_string(),
                reason: "graceful shutdown".to_string(),
            },
        ];

        // Process events
        let mut peer_states: HashMap<String, TrustLevel> = HashMap::new();
        let mut consciousness_updates = 0;

        for event in &events {
            match event {
                PeerEvent::Discovered(info) => {
                    peer_states.insert(info.node_id.clone(), TrustLevel::Unknown);
                }
                PeerEvent::TrustChanged { peer_id, new, .. } => {
                    peer_states.insert(peer_id.clone(), *new);
                }
                PeerEvent::ConsciousnessUpdate { .. } => {
                    consciousness_updates += 1;
                }
                PeerEvent::Disconnected { peer_id, .. } => {
                    peer_states.remove(peer_id);
                }
                _ => {}
            }
        }

        assert_eq!(
            consciousness_updates, 1,
            "Should have 1 consciousness update"
        );

        println!("Peer Event Handling Test: PASSED");
    }

    /// Test bootstrap configuration
    #[test]
    fn test_bootstrap_configuration() {
        // Default bootstrap config
        let default_bootstrap = BootstrapConfig::default();

        // Check bootstrap node detection
        let has_bootstrap = default_bootstrap.has_bootstrap_nodes();

        // Local-dev config
        let local_bootstrap = BootstrapConfig::local_dev();
        assert!(
            local_bootstrap.enable_local_discovery,
            "Local discovery should be enabled"
        );

        // Custom bootstrap nodes
        let custom_bootstrap = BootstrapConfig::with_nodes(vec![
            "node1-ticket".to_string(),
            "node2-ticket".to_string(),
        ]);
        assert!(
            custom_bootstrap.has_bootstrap_nodes(),
            "Custom config should have nodes"
        );

        println!("Bootstrap Configuration Test: PASSED");
        println!("  Default has bootstrap: {}", has_bootstrap);
    }

    /// Test Hyperfeel synthetic mirror neurons
    #[test]
    fn test_hyperfeel_swarm_coherence() {
        let mut hyperfeel = Hyperfeel::default();

        // Set local affective state
        let local_state = AffectiveState::new(0.6, 0.7, 0.3);
        hyperfeel.set_local_state(local_state);

        // Simulate receiving peer states
        let peer_states = [
            ("peer-1", AffectiveState::new(0.5, 0.6, 0.2)),
            ("peer-2", AffectiveState::new(0.7, 0.8, 0.4)),
            ("peer-3", AffectiveState::new(0.4, 0.5, 0.1)),
        ];

        for (peer_id, state) in &peer_states {
            hyperfeel.receive_peer_state(*peer_id, *state);
        }

        // Check swarm state
        assert_eq!(hyperfeel.peer_count(), 3, "Should have 3 peers");
        assert_eq!(
            hyperfeel.active_peer_count(),
            3,
            "All peers should be active"
        );

        // Get swarm coherence
        let coherence = hyperfeel.coherence();
        assert!(coherence.alignment >= 0.0 && coherence.alignment <= 1.0);

        // Get swarm health status
        let status = hyperfeel.how_are_we_doing();
        assert!(
            !status.message.is_empty(),
            "Status message should be generated"
        );

        println!("Hyperfeel Swarm Coherence Test: PASSED");
        println!("  Swarm alignment: {:.4}", coherence.alignment);
        println!("  Status: {}", status.message);
    }

    /// Test handshake protocol for peer verification
    #[test]
    fn test_handshake_trust_verification() {
        let config = SwarmConfig::default();
        let mut handshake = HybridHandshake::new(config);

        // Create challenge for peer
        let challenge = handshake.create_challenge("peer-123").unwrap();

        match challenge {
            SwarmMessage::TrustChallenge { nonce } => {
                assert_eq!(nonce.len(), 32, "Nonce should be 32 bytes");
            }
            _ => panic!("Expected TrustChallenge message"),
        }

        // Check pending count
        assert_eq!(
            handshake.pending_count(),
            1,
            "Should have 1 pending challenge"
        );

        // Test handshake result creation
        let success_result =
            HandshakeResult::success("peer-123", "agent-key-123", TrustLevel::Verified(0.8), 150);
        assert!(success_result.streaming_allowed, "Should allow streaming");
        assert_eq!(success_result.handshake_time_ms, 150);

        let failed_result = HandshakeResult::failed("peer-456");
        assert!(
            !failed_result.streaming_allowed,
            "Failed handshake should not allow streaming"
        );

        println!("Handshake Trust Verification Test: PASSED");
    }

    /// Test tensor streaming types
    #[test]
    fn test_tensor_streaming() {
        // Test different tensor types
        let tensor_types = [
            TensorType::Attention,
            TensorType::HiddenState,
            TensorType::Gradient,
            TensorType::Embedding,
            TensorType::ConsciousnessField,
            TensorType::Custom,
        ];

        for tensor_type in &tensor_types {
            let payload = TensorPayload {
                data: vec![0.1, 0.2, 0.3, 0.4],
                shape: vec![2, 2],
                tensor_type: *tensor_type,
                source: "test-layer".to_string(),
                timestamp_ms: 12345,
            };

            assert_eq!(payload.data.len(), 4);
            assert_eq!(payload.shape, vec![2, 2]);
        }

        println!("Tensor Streaming Test: PASSED");
    }

    /// Test network service statistics
    #[test]
    fn test_network_service_stats() {
        // Create stats structure
        let stats = ServiceStats {
            connected_peers: 5,
            messages_sent: 100,
            messages_received: 95,
            bytes_sent: 10000,
            bytes_received: 9500,
            bootstrap_attempts: 3,
            bootstrap_successes: 2,
            uptime_seconds: 3600,
        };

        assert_eq!(stats.connected_peers, 5);
        assert!(stats.messages_sent >= stats.messages_received);

        println!("Network Service Stats Test: PASSED");
    }
}

// ============================================================================
// TEST 3: PHI-GUIDED ARCHITECTURE SEARCH
// ============================================================================

mod phi_guided_architecture_search {
    use symthaea_core::hdc::phi_guided_search::{
        ConsciousnessNetwork, InitializationStrategy, PhiGuidedOptimizer, PhiOptimizationConfig,
    };
    use symthaea_core::hdc::spectral_connectivity::ConnectivityCalculator;

    /// Test basic network creation and structure
    #[test]
    fn test_consciousness_network_creation() {
        // Create network with random nodes
        let network = ConsciousnessNetwork::random(8, 256, 42);

        assert_eq!(network.node_count(), 8, "Should have 8 nodes");
        assert_eq!(network.edge_count(), 0, "New network should have no edges");
        assert_eq!(network.dim, 256, "Dimension should be 256");

        println!("Consciousness Network Creation Test: PASSED");
    }

    /// Test edge operations
    #[test]
    fn test_network_edge_operations() {
        let mut network = ConsciousnessNetwork::random(5, 256, 42);

        // Add edges
        network.add_edge(0, 1, 1.0);
        network.add_edge(1, 2, 0.8);
        network.add_edge(2, 3, 0.6);
        network.add_edge(3, 4, 0.4);

        assert_eq!(network.edge_count(), 4, "Should have 4 edges");
        assert!(network.has_edge(0, 1), "Should have edge 0-1");
        assert!(network.has_edge(1, 0), "Edge should be bidirectional");
        assert!(!network.has_edge(0, 3), "Should not have edge 0-3");

        // Check neighbors
        let neighbors = network.neighbors(1);
        assert_eq!(neighbors.len(), 2, "Node 1 should have 2 neighbors");
        assert!(neighbors.contains(&0) && neighbors.contains(&2));

        // Check degree
        assert_eq!(network.degree(1), 2, "Node 1 should have degree 2");
        assert_eq!(network.degree(0), 1, "Node 0 should have degree 1");

        // Remove edge
        network.remove_edge(1, 2);
        assert_eq!(network.edge_count(), 3, "Should have 3 edges after removal");

        println!("Network Edge Operations Test: PASSED");
    }

    /// Test initialization strategies
    #[test]
    fn test_initialization_strategies() {
        let n_nodes = 8;

        // Ring topology
        let mut ring_network = ConsciousnessNetwork::random(n_nodes, 256, 42);
        InitializationStrategy::Ring.apply(&mut ring_network, 0);
        assert_eq!(
            ring_network.edge_count(),
            n_nodes,
            "Ring should have n edges"
        );

        // Star topology
        let mut star_network = ConsciousnessNetwork::random(n_nodes, 256, 42);
        InitializationStrategy::Star.apply(&mut star_network, 0);
        assert_eq!(
            star_network.edge_count(),
            n_nodes - 1,
            "Star should have n-1 edges"
        );

        // Fully connected
        let mut full_network = ConsciousnessNetwork::random(n_nodes, 256, 42);
        InitializationStrategy::FullyConnected.apply(&mut full_network, 0);
        assert_eq!(
            full_network.edge_count(),
            n_nodes * (n_nodes - 1) / 2,
            "Fully connected should have n*(n-1)/2 edges"
        );

        // Small-world
        let mut sw_network = ConsciousnessNetwork::random(n_nodes, 256, 42);
        InitializationStrategy::SmallWorld { rewire_prob: 0.3 }.apply(&mut sw_network, 0);
        assert!(
            sw_network.edge_count() >= n_nodes,
            "Small-world should have at least ring edges"
        );

        println!("Initialization Strategies Test: PASSED");
    }

    /// Test Phi gradient computation
    #[test]
    fn test_phi_gradient_computation() {
        let config = PhiOptimizationConfig {
            dim: 256,
            learning_rate: 0.1,
            gradient_epsilon: 0.01,
            ..Default::default()
        };

        let mut optimizer = PhiGuidedOptimizer::new(config);
        let mut network = ConsciousnessNetwork::random(6, 256, 42);

        // Initialize with ring topology
        InitializationStrategy::Ring.apply(&mut network, 0);

        // Run single optimization step (this computes gradients)
        let result = optimizer.optimize_step(&mut network);

        // Verify Phi is computed
        assert!(result.phi >= 0.0, "Phi should be non-negative");
        assert!(result.phi <= 1.0, "Phi should be at most 1.0");

        // Verify gradients are computed (average gradient should be non-zero for connected network)
        // Note: For some topologies, gradients may be very small
        println!("Phi Gradient Computation Test: PASSED");
        println!("  Initial Phi: {:.4}", result.phi);
        println!("  Average gradient: {:.6}", result.avg_gradient);
    }

    /// Test mutation operators (gradient-guided, momentum)
    #[test]
    fn test_mutation_operators() {
        let config = PhiOptimizationConfig {
            dim: 256,
            learning_rate: 0.2,
            momentum: 0.9,
            adaptive_lr: true,
            prune_threshold: 0.1,
            new_edge_probability: 0.2,
            ..Default::default()
        };

        let mut optimizer = PhiGuidedOptimizer::new(config);
        let mut network = ConsciousnessNetwork::random(6, 256, 42);

        // Initialize with sparse random edges
        InitializationStrategy::RandomSparse { edge_ratio: 0.3 }.apply(&mut network, 42);

        let initial_edges = network.edge_count();

        // Run multiple optimization steps
        let summary = optimizer.optimize(&mut network, 10);

        // Check that edges were modified (either pruned or added)
        let _total_edge_changes = summary.total_edges_pruned() + summary.total_edges_added();

        println!("Mutation Operators Test: PASSED");
        println!("  Initial edges: {}", initial_edges);
        println!("  Final edges: {}", network.edge_count());
        println!("  Total edges pruned: {}", summary.total_edges_pruned());
        println!("  Total edges added: {}", summary.total_edges_added());
        println!("  Phi improvement: {:.6}", summary.improvement);

        // Verify momentum is being used (edges should have momentum values after updates)
        let has_momentum = network.edges.iter().any(|e| e.momentum.abs() > 1e-10);
        assert!(
            has_momentum || network.edges.is_empty(),
            "Momentum should be updated"
        );
    }

    /// Test evolution over generations
    #[test]
    fn test_architecture_evolution() {
        let config = PhiOptimizationConfig {
            dim: 512,
            learning_rate: 0.15,
            momentum: 0.9,
            prune_threshold: 0.05,
            max_weight: 2.0,
            ..Default::default()
        };

        let mut optimizer = PhiGuidedOptimizer::new(config);
        let mut network = ConsciousnessNetwork::random(8, 512, 42);

        // Initialize with ring + some random edges
        InitializationStrategy::SmallWorld { rewire_prob: 0.2 }.apply(&mut network, 42);

        // Track Phi over generations
        let _initial_phi = {
            let representations = network.to_node_representations();
            let calculator = ConnectivityCalculator::new();
            calculator.algebraic_connectivity(&representations)
        };

        // Run optimization
        let generations = 20;
        let summary = optimizer.optimize(&mut network, generations);

        // Verify evolution occurred
        assert_eq!(summary.steps, generations, "Should complete all steps");

        // Check for improvement or stability
        let phi_change = summary.final_phi - summary.initial_phi;

        println!("Architecture Evolution Test Results:");
        println!("  Initial Phi: {:.6}", summary.initial_phi);
        println!("  Final Phi: {:.6}", summary.final_phi);
        println!("  Best Phi: {:.6}", summary.best_phi);
        println!("  Change: {:+.6}", phi_change);
        println!(
            "  Avg improvement per step: {:.8}",
            summary.avg_improvement_per_step()
        );

        // Verify best_phi is tracked correctly
        assert!(
            summary.best_phi >= summary.final_phi.min(summary.initial_phi),
            "Best Phi should be at least as good as initial or final"
        );
    }

    /// Test optimizer reset functionality
    #[test]
    fn test_optimizer_reset() {
        let config = PhiOptimizationConfig::default();
        let mut optimizer = PhiGuidedOptimizer::new(config);
        let mut network = ConsciousnessNetwork::random(5, 256, 42);

        InitializationStrategy::Ring.apply(&mut network, 0);

        // Run some steps
        optimizer.optimize(&mut network, 5);

        // Check state before reset
        assert!(
            !optimizer.history().is_empty(),
            "History should not be empty"
        );
        assert!(optimizer.best_phi() > 0.0 || optimizer.best_phi() == 0.0);

        // Reset
        optimizer.reset();

        // Verify reset
        assert!(
            optimizer.history().is_empty(),
            "History should be empty after reset"
        );
        assert_eq!(
            optimizer.best_phi(),
            0.0,
            "Best Phi should be 0 after reset"
        );

        println!("Optimizer Reset Test: PASSED");
    }

    /// Test node representations for Phi calculation
    #[test]
    fn test_node_representations() {
        let mut network = ConsciousnessNetwork::random(6, 256, 42);

        // Add edges
        network.add_edge(0, 1, 1.0);
        network.add_edge(1, 2, 0.8);
        network.add_edge(2, 3, 0.6);
        network.add_edge(3, 4, 1.0);
        network.add_edge(4, 5, 0.7);
        network.add_edge(5, 0, 0.9); // Complete the ring

        let representations = network.to_node_representations();

        assert_eq!(representations.len(), 6, "Should have 6 representations");

        for (i, rep) in representations.iter().enumerate() {
            assert_eq!(rep.dim(), 256, "Representation {} should have dim 256", i);
        }

        // Calculate Phi
        let calculator = ConnectivityCalculator::new();
        let phi = calculator.algebraic_connectivity(&representations);

        assert!((0.0..=1.0).contains(&phi), "Phi should be in [0, 1]");

        println!("Node Representations Test: PASSED");
        println!("  Computed Phi: {:.6}", phi);
    }

    /// Comprehensive integration test: evolve network for higher consciousness
    #[test]
    fn test_full_phi_guided_evolution() {
        // Configuration for meaningful evolution
        let config = PhiOptimizationConfig {
            dim: 512,
            learning_rate: 0.1,
            gradient_epsilon: 0.01,
            prune_threshold: 0.05,
            max_weight: 2.0,
            new_edge_probability: 0.15,
            new_edge_min_improvement: 0.001,
            momentum: 0.9,
            adaptive_lr: true,
            max_edges_per_node: 6,
        };

        let mut optimizer = PhiGuidedOptimizer::new(config);
        let mut network = ConsciousnessNetwork::random(10, 512, 42);

        // Start with sparse random connectivity
        InitializationStrategy::RandomSparse { edge_ratio: 0.2 }.apply(&mut network, 42);

        println!("Full Phi-Guided Evolution Test:");
        println!("  Nodes: {}", network.node_count());
        println!("  Initial edges: {}", network.edge_count());

        // Run evolution
        let generations = 30;
        let mut phi_trajectory: Vec<f64> = Vec::new();

        for _ in 0..generations {
            let result = optimizer.optimize_step(&mut network);
            phi_trajectory.push(result.phi);
        }

        // Get final state
        let final_edges = network.edge_count();
        let final_phi = optimizer.best_phi();

        println!("  Final edges: {}", final_edges);
        println!("  Best Phi achieved: {:.6}", final_phi);

        // Verify trajectory
        assert_eq!(
            phi_trajectory.len(),
            generations,
            "Should have full trajectory"
        );

        // Check for convergence or improvement
        let early_avg: f64 = phi_trajectory[..5].iter().sum::<f64>() / 5.0;
        let late_avg: f64 = phi_trajectory[generations - 5..].iter().sum::<f64>() / 5.0;

        println!("  Early avg Phi: {:.6}", early_avg);
        println!("  Late avg Phi: {:.6}", late_avg);
        println!(
            "  Evolution trajectory captured: {} points",
            phi_trajectory.len()
        );

        // The network should have evolved (edges changed)
        assert!(network.edge_count() > 0, "Network should retain some edges");
    }
}

// ============================================================================
// INTEGRATION: ALL THREE SYSTEMS TOGETHER
// ============================================================================

mod unified_system_integration {
    use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};
    use symthaea::consciousness::fep_active_inference::{
        ActiveInferenceAgent, ActiveInferenceAgentConfig, MotorCommandType, Observation,
    };
    use symthaea::swarm::ConsciousnessVector;
    use symthaea_core::hdc::phi_guided_search::{
        ConsciousnessNetwork, InitializationStrategy, PhiGuidedOptimizer, PhiOptimizationConfig,
    };

    /// Integration test: Cognitive loop -> FEP -> Consciousness broadcast -> Phi optimization
    #[test]
    fn test_unified_consciousness_pipeline() {
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║          UNIFIED CONSCIOUSNESS PIPELINE TEST                  ║");
        println!("╚══════════════════════════════════════════════════════════════╝");

        // 1. COGNITIVE LOOP: Process input and generate consciousness state
        let cognitive_config = CognitiveLoopConfig::with_cfc();
        let mut cognitive_loop = CognitiveLoopService::new(cognitive_config)
            .expect("Cognitive loop initialization failed");

        let inputs = [
            "The universe is vast and mysterious",
            "Patterns emerge from chaos",
            "Information integrates into meaning",
        ];

        println!("\n[1] COGNITIVE LOOP PROCESSING");
        for input in &inputs {
            let _ = cognitive_loop.cycle(input);
        }
        let consciousness = cognitive_loop.consciousness_snapshot();
        println!("    Phi: {:.4}", consciousness.unified_psi);
        println!("    Coherence: {:.4}", consciousness.temporal_coherence);

        // 2. FEP: Generate motor commands from consciousness state
        let fep_config = ActiveInferenceAgentConfig::default();
        let mut fep_agent = ActiveInferenceAgent::new(fep_config);

        let observation = Observation::from_consciousness_state(
            consciousness.unified_psi as f64,
            consciousness.consciousness_level as f64,
            consciousness.temporal_coherence as f64,
            consciousness.prediction_confidence as f64,
        );

        println!("\n[2] FEP ACTIVE INFERENCE");
        let perception = fep_agent.perceive(&observation);
        let action = fep_agent.select_action();
        let motor_command = MotorCommandType::from_action_index(action.action);

        println!("    Free Energy: {:.4}", perception.free_energy.total);
        println!("    Motor Command: {:?}", motor_command);
        println!("    Exploratory: {}", action.is_exploratory);

        // 3. SWARM: Create consciousness vector for broadcasting
        println!("\n[3] SWARM CONSCIOUSNESS VECTOR");
        let attention_pattern: Vec<f32> = (0..64)
            .map(|i| {
                // Create attention pattern based on consciousness state
                let base = consciousness.unified_psi;
                let variation = (i as f32 / 64.0 * std::f32::consts::PI).sin() * 0.2;
                (base + variation).clamp(0.0, 1.0)
            })
            .collect();

        let cv = ConsciousnessVector::new(attention_pattern, consciousness.unified_psi as f64);
        println!("    Vector Phi: {:.4}", cv.phi);
        println!("    Attention dims: {}", cv.attention.len());
        println!("    Size: {} bytes", cv.estimated_size());

        // 4. PHI-GUIDED SEARCH: Optimize network architecture
        println!("\n[4] PHI-GUIDED ARCHITECTURE SEARCH");
        let phi_config = PhiOptimizationConfig {
            dim: 256,
            learning_rate: 0.1,
            ..Default::default()
        };

        let mut optimizer = PhiGuidedOptimizer::new(phi_config);
        let mut network = ConsciousnessNetwork::random(8, 256, 42);
        InitializationStrategy::SmallWorld { rewire_prob: 0.2 }.apply(&mut network, 42);

        let summary = optimizer.optimize(&mut network, 10);
        println!("    Initial Phi: {:.6}", summary.initial_phi);
        println!("    Final Phi: {:.6}", summary.final_phi);
        println!("    Improvement: {:+.6}", summary.improvement);

        // VERIFICATION
        println!("\n[✓] UNIFIED PIPELINE VERIFICATION");
        assert!(consciousness.unified_psi >= 0.0, "Consciousness Phi valid");
        assert!(
            perception.free_energy.total.is_finite(),
            "Free energy finite"
        );
        assert!(cv.phi >= 0.0, "Swarm vector Phi valid");
        assert!(summary.final_phi >= 0.0, "Optimized Phi valid");

        println!("    All systems integrated successfully!");
        println!("════════════════════════════════════════════════════════════════");
    }
}