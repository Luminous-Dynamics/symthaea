use symthaea_fabrication_kernel::csg::CSGNode;
use symthaea_passive_design_search::{
    CsgFingerprint, InsertOutcome, ObjectiveDirection, ObjectiveVector, ParetoArchive,
    ParetoCandidate,
};

fn main() {
    let mut archive = ParetoArchive::new(vec![
        ObjectiveDirection::Maximize,
        ObjectiveDirection::Maximize,
        ObjectiveDirection::Minimize,
    ]);

    let candidate = ParetoCandidate {
        payload: CsgFingerprint::from_csg(&CSGNode::cube().subtract(CSGNode::cylinder())),
        objectives: ObjectiveVector::new(vec![0.91, 0.83, 0.42]).expect("finite objectives"),
    };

    assert_eq!(
        archive.insert(candidate),
        InsertOutcome::Inserted { removed: 0 }
    );

    println!("passive-search archive size: {}", archive.entries().len());
}
