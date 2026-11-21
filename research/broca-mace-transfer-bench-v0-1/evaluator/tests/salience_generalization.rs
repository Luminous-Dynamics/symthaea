use broca_mace_consolidation::salience_learning::{
    calibrate,evaluate,qualify_splits,train,ExampleV1,FeaturesV1,ModelConfigV1
};
use broca_mace_policy::SourceRootDigest;
#[derive(Clone,Copy)]
struct Rng(u64);
impl Rng {
    fn new(seed:u64)->Self { Self(seed.max(1)) }
    fn next(&mut self)->u64 {
        let mut x=self.0;x^=x>>12;x^=x<<25;x^=x>>27;self.0=x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self)->f64 {(self.next()>>11) as f64/9_007_199_254_740_992.0}
    fn bps(&mut self)->u16 {(self.unit()*10_001.0).floor().min(10_000.0) as u16}
}
fn sigmoid(x:f64)->f64 {1.0/(1.0+(-x.clamp(-20.0,20.0)).exp())}
fn make_split(
    target_model_sha256: &str,
    seed:u64,event:u32,cluster:u64,family:u32,count:u32,shift:f64
)->Vec<ExampleV1> {
    let mut rng=Rng::new(seed);let mut result=Vec::new();
    for c in 0..count {
        for j in 0..8_u32 {
            let features=FeaturesV1 {
                recency_bps:rng.bps(),
                frequency_bps:rng.bps(),
                task_similarity_bps:rng.bps(),
                exact_scope_match:rng.next()%5!=0,
                source_support_bps:rng.bps(),
                measurement_quality_bps:rng.bps(),
                failure_memory:rng.next()%4==0,
            };
            let z=-3.0
                +1.2*f64::from(features.recency_bps)/10_000.0
                +f64::from(features.frequency_bps)/10_000.0
                +2.0*f64::from(features.task_similarity_bps)/10_000.0
                +if features.exact_scope_match {0.8} else {0.0}
                +0.9*f64::from(features.source_support_bps)/10_000.0
                +1.1*f64::from(features.measurement_quality_bps)/10_000.0
                +if features.failure_memory {0.25} else {0.0}
                +shift;
            let id=event+c*8+j+1;
            let root=id+50_000;
            let mut root_bytes=[0_u8;32];
            root_bytes[..4].copy_from_slice(&root.to_be_bytes());
            result.push(ExampleV1 {
                event_id:id,
                cluster_id:cluster+u64::from(c)+1,
                task_family_id:family+c%8,
                source_root:SourceRootDigest::from_bytes(root_bytes),
                target_model_sha256: target_model_sha256.to_owned(),
                features,
                future_reused:rng.unit()<sigmoid(z),
            });
        }
    }
    result
}
#[test]
fn learned_salience_uses_disjoint_train_calibration_and_heldout_partitions() {
    let training=make_split("a".repeat(64).as_str(),0x51a1,1,1,100,64,0.0);
    let calibration=make_split("a".repeat(64).as_str(),0xc411,100_000,10_000,200,32,0.10);
    let heldout=make_split("a".repeat(64).as_str(),0x7e57,200_000,20_000,300,32,0.05);
    let manifest=qualify_splits(&training,&calibration,&heldout).unwrap();
    assert_eq!(manifest.training_clusters,64);
    assert_eq!(manifest.calibration_clusters,32);
    assert_eq!(manifest.heldout_clusters,32);
    let model=train(&training,ModelConfigV1::default()).unwrap();
    let calibrator=calibrate(&model,&calibration).unwrap();
    let metrics=evaluate(&model,&calibrator,&heldout).unwrap();
    assert_eq!(metrics.examples,heldout.len() as u32);
    assert_eq!(metrics.task_families,8);
    assert_eq!(metrics.source_roots,heldout.len() as u32);
    println!(
        "learned_salience heldout={} brier={} base_brier={} ece={} worst_family_ece={}",
        metrics.examples,metrics.brier_micros,metrics.base_rate_brier_micros,
        metrics.ece_micros,metrics.worst_family_ece_micros
    );
}
#[test]
fn task_family_overlap_fails_split_qualification() {
    let training=make_split("a".repeat(64).as_str(),0x51a1,1,1,100,64,0.0);
    let calibration=make_split("a".repeat(64).as_str(),0xc411,100_000,10_000,100,32,0.1);
    let heldout=make_split("a".repeat(64).as_str(),0x7e57,200_000,20_000,300,32,0.05);
    assert!(qualify_splits(&training,&calibration,&heldout).is_err());
}


#[test]
fn target_model_must_be_identical_across_all_memory_splits() {
    let training=make_split("a".repeat(64).as_str(),0x51a1,1,1,100,64,0.0);
    let calibration=make_split("a".repeat(64).as_str(),0xc411,100_000,10_000,200,32,0.10);
    let heldout=make_split("b".repeat(64).as_str(),0x7e57,200_000,20_000,300,32,0.05);
    assert!(qualify_splits(&training,&calibration,&heldout).is_err());
}


#[test]
fn model_conditioned_salience_beats_a_pooled_predictor_on_heterogeneous_targets() {
    let small_model = "a".repeat(64);
    let large_model = "b".repeat(64);
    let pooled_model = "c".repeat(64);

    let train_small =
        make_split(&small_model,0x51a1,1,1,100,64,-1.4);
    let calibration_small =
        make_split(&small_model,0xc411,100_000,10_000,200,32,-1.4);
    let heldout_small =
        make_split(&small_model,0x7e57,200_000,20_000,300,32,-1.4);

    let train_large =
        make_split(&large_model,0x61b2,300_000,30_000,500,64,1.4);
    let calibration_large =
        make_split(&large_model,0xd522,400_000,40_000,600,32,1.4);
    let heldout_large =
        make_split(&large_model,0x8f73,500_000,50_000,700,32,1.4);

    let small = train(&train_small,ModelConfigV1::default()).unwrap();
    let small_calibrator = calibrate(&small,&calibration_small).unwrap();
    let large = train(&train_large,ModelConfigV1::default()).unwrap();
    let large_calibrator = calibrate(&large,&calibration_large).unwrap();

    let small_metrics = evaluate(&small,&small_calibrator,&heldout_small).unwrap();
    let large_metrics = evaluate(&large,&large_calibrator,&heldout_large).unwrap();

    // Pooled baseline deliberately erases per-target identity and treats both
    // agents as one population, while retaining the same model budget.
    let mut pooled_training = train_small.clone();
    pooled_training.extend(train_large.clone());
    for row in &mut pooled_training {
        row.target_model_sha256 = pooled_model.clone();
    }
    let mut pooled_calibration = calibration_small.clone();
    pooled_calibration.extend(calibration_large.clone());
    for row in &mut pooled_calibration {
        row.target_model_sha256 = pooled_model.clone();
    }
    let mut pooled_heldout_small = heldout_small.clone();
    let mut pooled_heldout_large = heldout_large.clone();
    for row in pooled_heldout_small.iter_mut().chain(pooled_heldout_large.iter_mut()) {
        row.target_model_sha256 = pooled_model.clone();
    }

    let pooled = train(&pooled_training,ModelConfigV1::default()).unwrap();
    let pooled_calibrator = calibrate(&pooled,&pooled_calibration).unwrap();
    let pooled_small_metrics =
        evaluate(&pooled,&pooled_calibrator,&pooled_heldout_small).unwrap();
    let pooled_large_metrics =
        evaluate(&pooled,&pooled_calibrator,&pooled_heldout_large).unwrap();

    assert_ne!(small.target_model_sha256(),large.target_model_sha256());
    assert_ne!(
        small.training_base_rate_bps,
        large.training_base_rate_bps
    );

    let conditioned_error = small_metrics.brier_micros
        + large_metrics.brier_micros;
    let pooled_error = pooled_small_metrics.brier_micros
        + pooled_large_metrics.brier_micros;
    assert!(
        conditioned_error < pooled_error,
        "model-conditioned Brier={conditioned_error}, pooled Brier={pooled_error}"
    );
}
