use symthaea_quantum_comp::{
    EntanglementProxyConfig, EntanglementProxyRunner, NegativeControlConfig, NegativeControlRunner,
};

#[test]
fn alpha3_entanglement_proxy_is_reproducible() {
    let cfg = EntanglementProxyConfig {
        dimension: 128,
        trials: 4,
        decoherence: 0.03,
        seed: 1234,
    };
    let a = EntanglementProxyRunner::new(cfg).unwrap().run().unwrap();
    let b = EntanglementProxyRunner::new(cfg).unwrap().run().unwrap();
    assert_eq!(a, b);
    assert!(a.recovery_gap > 0.3);
}

#[test]
fn alpha3_negative_control_is_reproducible() {
    let cfg = NegativeControlConfig {
        dimension: 128,
        trials: 4,
        noise: 0.05,
        seed: 9876,
    };
    let a = NegativeControlRunner::new(cfg).unwrap().run().unwrap();
    let b = NegativeControlRunner::new(cfg).unwrap().run().unwrap();
    assert_eq!(a, b);
    assert!(a.control_gap > 0.3);
}
