use broca_mace_consolidation::salience_learning::{
    calibrate,evaluate,qualify_splits,train,FutureUseExampleV1,
    FutureUseFeaturesV1,ModelConfigV1
};
use broca_mace_policy::SourceRootDigest;

#[derive(Clone,Copy)]
struct Rng(u64);
impl Rng {
    fn new(seed:u64)->Self { Self(seed.max(1)) }
    fn next(&mut self)->u64 {
        let mut x=self.0;
        x^=x>>12;x^=x<<25;x^=x>>27;self.0=x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self)->f64 {
        (self.next()>>11) as f64/9_007_199_254_740_992.0
    }
    fn bps(&mut self)->u16 { (self.unit()*10_001.0).floor().min(10_000.0) as u16 }
}
fn sigmoid(x:f64)->f64 {1.0/(1.0+(-x.clamp(-20.0,20.0)).exp())}
/// Hidden future-reuse response and labels remain evaluator-owned.
fn make_split(seed:u64,event:u32,cluster:u64,family:u32,count:u32,shift:f64)
->Vec<FutureUseExampleV1> {
    let mut rng=Rng::new(seed);let mut rows=Vec::new();
    for c in 0..count {
        for j in 0..8_u32 {
            let f=FutureUseFeaturesV1 {
                recency_bps:rng.bps(),frequency_bps:rng.bps(),
                task_similarity_bps:rng.bps(),exact_scope_match:rng.next()%5!=0,
                source_support_bps:rng.bps(),measurement_quality_bps:rng.bps(),
                failure_memory:rng.next()%4==0,
            };
            let z=-3.0+1.2*f64::from(f.recency_bps)/10_000.0
                +f64::from(f.frequency_bps)/10_000.0
                +2.0*f64::from(f.task_similarity_bps)/10_000.0
                +0.8*f64::from(f.exact_scope_match)
                +0.9*f64::from(f.source_support_bps)/10_000.0
                +1.1*f64::from(f.measurement_quality_bps)/10_000.0
                +0.25*f64::from(f.failure_memory)+shift;
            let id=event+c*8+j+1;
            let root=id+50_000;
            let mut bytes=[0_u8;32];bytes[..4].copy_from_slice(&root.to_be_bytes());
            rows.push(FutureUseExampleV1 {
                event_id:id,cluster_id:cluster+u64::from(c)+1,
                task_family_id:family+c%8,
                source_root:SourceRootDigest::from_bytes(bytes),
                features:f,future_reused:rng.unit()<sigmoid(z),
            });
        }
    }
    rows
}
#[test]
fn learned_salience_runs_train_calibrate_and_heldout_without_split_leakage() {
    let training=make_split(0x51a1,1,1,100,64,0.0);
    let calibration=make_split(0xc411,100_000,10_000,200,32,0.10);
    let heldout=make_split(0x7e57,200_000,20_000,300,32,0.05);
    let split=qualify_splits(&training,&calibration,&heldout).unwrap();
    assert_eq!(split.training_clusters,64);
    assert_eq!(split.calibration_clusters,32);
    assert_eq!(split.heldout_clusters,32);
    let model=train(&training,ModelConfigV1::default()).unwrap();
    let calibrator=calibrate(&model,&calibration).unwrap();
    let metrics=evaluate(&model,&calibrator,&heldout).unwrap();
    assert_eq!(metrics.count,heldout.len() as u32);
    assert_eq!(metrics.task_families,8);
    assert_eq!(metrics.source_roots,heldout.len() as u32);
    println!(
        "learned_salience heldout={} brier={} base_rate_brier={} ece={} worst_family_ece={}",
        metrics.count,metrics.brier_micros,metrics.base_rate_brier_micros,
        metrics.ece_micros,metrics.worst_family_ece_micros
    );
}
#[test]
fn task_family_overlap_blocks_learned_salience_evaluation() {
    let training=make_split(0x51a1,1,1,100,64,0.0);
    let calibration=make_split(0xc411,100_000,10_000,100,32,0.1);
    let heldout=make_split(0x7e57,200_000,20_000,300,32,0.05);
    assert!(qualify_splits(&training,&calibration,&heldout).is_err());
}
