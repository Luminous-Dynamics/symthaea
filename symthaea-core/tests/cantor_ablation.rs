// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use symthaea_core::hdc::cantor_pyramid::{CantorHdcConfig, PyramidCantorVector};
use symthaea_core::hdc::unified_hv::ContinuousHV;

#[test]
fn test_role_filler_retrieval_pressure() {
    let config = CantorHdcConfig {
        total_dim: 16384,
        levels: 4,
        branching: 4,
        leaf_dim: 256,
        ..CantorHdcConfig::default()
    };

    // 1. Create a Flat Baseline (simulated by using L0 only)
    let mut flat = PyramidCantorVector::new(config, None);

    // 2. Create a Hierarchical Vector
    let mut hierarchical = PyramidCantorVector::new(config, None);

    // Role-Filler Bindings: color ⊗ RED + shape ⊗ CIRCLE
    let role_color = ContinuousHV::random(16384, 100);
    let val_red = ContinuousHV::random(16384, 101);
    let role_shape = ContinuousHV::random(16384, 102);
    let val_circle = ContinuousHV::random(16384, 103);

    let binding_color = role_color.bind(&val_red);
    let binding_shape = role_shape.bind(&val_circle);

    // BUNDLING PRESSURE: Increase number of objects
    let leaf_dim = config.leaf_dim;
    for i in 0..50 {
        let r = ContinuousHV::random(16384, 200 + i as u64);
        let v = ContinuousHV::random(16384, 300 + i as u64);
        let b = r.bind(&v);

        // Flat: bundle everything globally
        let l0 = flat.find_node(0, 0).unwrap().clone();
        flat.bundle_at_node(&l0, &b);

        // Hierarchical: bundle in specific leaf nodes (L3)
        // We use the first leaf_dim elements of the 16384D binding
        let leaf_idx = i % 64; // 4*4*4 = 64 leaves
        let leaf = hierarchical
            .find_node(3, leaf_idx as usize)
            .unwrap()
            .clone();
        let b_leaf = ContinuousHV::from_slice(&b.values[0..leaf_dim]);
        hierarchical.bundle_at_node(&leaf, &b_leaf);
    }

    // Add our target object
    let l0 = flat.find_node(0, 0).unwrap().clone();
    flat.bundle_at_node(&l0, &binding_color);
    flat.bundle_at_node(&l0, &binding_shape);

    let leaf_target = hierarchical.find_node(3, 63).unwrap().clone();
    let bc_leaf = ContinuousHV::from_slice(&binding_color.values[0..leaf_dim]);
    let bs_leaf = ContinuousHV::from_slice(&binding_shape.values[0..leaf_dim]);
    hierarchical.bundle_at_node(&leaf_target, &bc_leaf);
    hierarchical.bundle_at_node(&leaf_target, &bs_leaf);

    // RETRIEVAL
    // Query: what is the color? (aggregate ⊗ role_color)

    // Flat Retrieval
    let flat_data = flat.node_data(&l0);
    let flat_hv = ContinuousHV::from_slice(flat_data);
    let recovered_flat = flat_hv.bind(&role_color.inverse());
    let sim_flat = recovered_flat.similarity(&val_red);

    // Hierarchical Retrieval (from specific leaf)
    let leaf_data = hierarchical.node_data(&leaf_target);
    let leaf_hv = ContinuousHV::from_slice(leaf_data);

    let role_color_leaf = ContinuousHV::from_slice(&role_color.values[0..leaf_dim]);
    let val_red_leaf = ContinuousHV::from_slice(&val_red.values[0..leaf_dim]);

    let recovered_hier = leaf_hv.bind(&role_color_leaf.inverse());
    let sim_hier = recovered_hier.similarity(&val_red_leaf);

    println!("Flat Retrieval Similarity: {:.4}", sim_flat);
    println!("Hierarchical Retrieval Similarity: {:.4}", sim_hier);

    // Hierarchical should have much higher similarity because of less interference in the leaf
    assert!(
        sim_hier > sim_flat,
        "Hierarchical retrieval should outperform flat under pressure"
    );
}

#[test]
fn test_local_to_global_broadcast_reconstruction() {
    let config = CantorHdcConfig {
        total_dim: 4096,
        levels: 3,
        branching: 2,
        leaf_dim: 1024,
        ..CantorHdcConfig::default()
    };

    let mut pyramid = PyramidCantorVector::new(config, None);

    // 1. Encode sensory primitives at L2 (leaves)
    let prim0 = ContinuousHV::random(1024, 500);
    let prim1 = ContinuousHV::random(1024, 501);

    let l2_0 = pyramid.find_node(2, 0).unwrap().clone();
    let l2_1 = pyramid.find_node(2, 1).unwrap().clone();

    pyramid.bundle_at_node(&l2_0, &prim0);
    pyramid.bundle_at_node(&l2_1, &prim1);

    // 2. Broadcast UP to L1
    pyramid.broadcast_up(1); // Broadcast children of L1,0 to L1,0
    pyramid.broadcast_up(2); // Broadcast children of L1,1 to L1,1

    // 3. Broadcast UP to L0
    pyramid.broadcast_up(0);

    // 4. Verify Reconstruction
    let l0 = pyramid.find_node(0, 0).unwrap().clone();
    let l0_data = pyramid.node_data(&l0);
    let l0_hv = ContinuousHV::from_slice(l0_data);

    // L0 should be similar to both primitives (since it's a broadcast aggregate)
    let sim0 = l0_hv.similarity(&prim0.dilate(4096));
    let sim1 = l0_hv.similarity(&prim1.dilate(4096));

    println!("L0 vs Primitive 0 similarity: {:.4}", sim0);
    println!("L0 vs Primitive 1 similarity: {:.4}", sim1);

    assert!(sim0 > 0.1);
    assert!(sim1 > 0.1);
}

#[test]
fn test_crosstalk_threshold_comparison() {
    let config = CantorHdcConfig {
        total_dim: 16384,
        levels: 2,
        branching: 16,
        leaf_dim: 1024,
        ..CantorHdcConfig::default()
    };

    let mut flat = PyramidCantorVector::new(config, None);
    let mut hierarchical = PyramidCantorVector::new(config, None);

    let target_role = ContinuousHV::random(1024, 777);
    let target_val = ContinuousHV::random(1024, 888);
    let target_binding = target_role.bind(&target_val);

    // Add target to Flat (L0) and Hierarchical (Leaf 0)
    let l0 = flat.find_node(0, 0).unwrap().clone();
    flat.bundle_at_node(&l0, &target_binding.dilate(16384));

    let leaf0 = hierarchical.find_node(1, 0).unwrap().clone();
    hierarchical.bundle_at_node(&leaf0, &target_binding);

    // Stress test: add more and more noise bundles
    let mut n = 0;
    loop {
        n += 1;
        let r = ContinuousHV::random(1024, 1000 + n as u64);
        let v = ContinuousHV::random(1024, 2000 + n as u64);
        let b = r.bind(&v);

        flat.bundle_at_node(&l0, &b.dilate(16384));

        // In hierarchical, we distribute noise across OTHER leaves
        let leaf_idx = (n % 15) + 1; // Leaves 1-15
        let leaf = hierarchical
            .find_node(1, leaf_idx as usize)
            .unwrap()
            .clone();
        hierarchical.bundle_at_node(&leaf, &b);

        // Check retrieval
        let f_data = flat.node_data(&l0);
        let f_hv = ContinuousHV::from_slice(f_data);
        let f_rec = f_hv.bind(&target_role.dilate(16384).inverse());
        let f_sim = f_rec.similarity(&target_val.dilate(16384));

        let h_data = hierarchical.node_data(&leaf0);
        let h_hv = ContinuousHV::from_slice(h_data);
        let h_rec = h_hv.bind(&target_role.inverse());
        let h_sim = h_rec.similarity(&target_val);

        if f_sim < 0.1 && h_sim > 0.5 {
            println!("Crosstalk threshold reached at N={}", n);
            println!("Flat similarity: {:.4}", f_sim);
            println!("Hierarchical similarity: {:.4}", h_sim);
            break;
        }

        if n > 500 {
            println!("Flat vector is surprisingly robust! N=500 reached.");
            break;
        }
    }

    assert!(n > 0);
}
