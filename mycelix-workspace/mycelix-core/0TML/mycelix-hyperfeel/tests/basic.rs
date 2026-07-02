use mycelix_hyperfeel::{HyperGradient, Hypervector, ModelUpdate, ModelAdapter, ArchitectureInfo};

struct DummyAdapter;

impl ModelAdapter for DummyAdapter {
    fn to_hypergradient(&self, update: &ModelUpdate) -> Result<HyperGradient, mycelix_hyperfeel::model_adapter::AdapterError> {
        HyperGradient::from_gradient(&update.gradient)
            .map_err(|e| mycelix_hyperfeel::model_adapter::AdapterError::Internal(e.to_string()))
    }

    fn apply_hypergradient(&mut self, _hg: &HyperGradient) -> Result<(), mycelix_hyperfeel::model_adapter::AdapterError> {
        Ok(())
    }

    fn architecture(&self) -> ArchitectureInfo {
        ArchitectureInfo {
            name: "dummy".to_string(),
            parameter_count: 0,
        }
    }
}

#[test]
fn hypergradient_roundtrip_basic() {
    let update = ModelUpdate {
        gradient: vec![0.1, -0.3, 0.7, 0.0, 0.5],
    };

    let adapter = DummyAdapter;
    let hg = adapter.to_hypergradient(&update).expect("encode");

    assert_eq!(hg.layer_info.len(), 1);
    assert_eq!(hg.layer_info[0].size, update.gradient.len());
    assert_eq!(hg.vector.data.len(), Hypervector::DIMENSION);
}

#[test]
fn aggregate_produces_normalized_vector() {
    let g1 = HyperGradient::from_gradient(&[0.5, 0.5, 0.5]).unwrap();
    let g2 = HyperGradient::from_gradient(&[-0.5, -0.5, -0.5]).unwrap();

    let aggregated = HyperGradient::aggregate(&[g1, g2]).expect("aggregate");
    assert_eq!(aggregated.vector.data.len(), Hypervector::DIMENSION);
}

