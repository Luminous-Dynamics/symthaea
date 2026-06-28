use symthaea_quantum_comp::{
    BinaryHypervector, BindingProbeConfig, BindingProbeRunner, CorrelationBindingSketch,
    NoiseSweepConfig, NoiseSweepRunner, PhaseHypervector,
};

#[test]
fn binary_hdc_baseline_is_deterministic() {
    let a = BinaryHypervector::random(512, 123).unwrap();
    let b = BinaryHypervector::random(512, 123).unwrap();
    assert_eq!(a, b);
}

#[test]
fn phase_from_binary_roundtrips_under_binding() {
    let a = BinaryHypervector::random(512, 1).unwrap();
    let k = BinaryHypervector::random(512, 2).unwrap();
    let pa = PhaseHypervector::from_binary(&a);
    let pk = PhaseHypervector::from_binary(&k);
    let recovered = pa.bind_phase(&pk).unwrap().unbind_phase(&pk).unwrap();
    assert!(pa.circular_similarity(&recovered).unwrap() > 0.999);
}

#[test]
fn correlation_sketch_recovers_item_and_key() {
    let item = BinaryHypervector::random(512, 77).unwrap();
    let key = BinaryHypervector::random(512, 78).unwrap();
    let sketch = CorrelationBindingSketch::bind(&item, &key).unwrap();
    assert_eq!(item, sketch.recover_item(&key).unwrap());
    assert_eq!(key, sketch.recover_key(&item).unwrap());
}

#[test]
fn binding_probe_is_reproducible() {
    let config = BindingProbeConfig {
        dimension: 512,
        trials: 8,
        noise: 0.03,
        seed: 999,
        topology_threshold: 0.55,
    };
    let a = BindingProbeRunner::new(config).unwrap().run().unwrap();
    let b = BindingProbeRunner::new(config).unwrap().run().unwrap();
    assert_eq!(a, b);
    assert_eq!(
        a.manifest.reproducibility_fingerprint(),
        b.manifest.reproducibility_fingerprint()
    );
}

#[test]
fn noise_sweep_is_reproducible() {
    let base = BindingProbeConfig {
        dimension: 256,
        trials: 4,
        noise: 0.0,
        seed: 222,
        topology_threshold: 0.55,
    };
    let cfg = NoiseSweepConfig {
        base,
        steps: 4,
        max_noise: 0.2,
    };
    let a = NoiseSweepRunner::new(cfg).unwrap().run().unwrap();
    let b = NoiseSweepRunner::new(cfg).unwrap().run().unwrap();
    assert_eq!(a, b);
}
