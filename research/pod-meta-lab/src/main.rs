use pod_meta_lab::{
    choose_next, CandidateOperation, Obligation, ObligationState, OperationKind,
    SchedulingClass, Selection, SelectionContext,
};
use std::collections::BTreeSet;

fn main() {
    let obligations = [
        Obligation::new("missing-premise", ObligationState::Open, 7),
        Obligation::new("derivation", ObligationState::CandidateEvidenceAvailable, 5),
    ];
    let operations = [
        CandidateOperation::new(
            "retrieve-premise",
            OperationKind::Retrieve,
            ["missing-premise"],
            SchedulingClass::Competitive,
            2,
            "query-evidence-state-v1",
        ),
        CandidateOperation::new(
            "verify-existing",
            OperationKind::Verify,
            ["derivation"],
            SchedulingClass::Competitive,
            1,
            "query-evidence-state-v1",
        ),
    ];
    let attempts = BTreeSet::new();
    let decision = choose_next(&SelectionContext {
        obligations: &obligations,
        operations: &operations,
        attempted_without_progress: &attempts,
        remaining_work_units: 10,
        mandatory_reserve_units: 2,
    });

    println!("POAR V1 research-only demo");
    println!("selection: {:?}", decision.selection);
    println!("rejected candidates: {}", decision.rejected.len());
    println!("competitive budget: {}", decision.competitive_work_limit);

    match decision.selection {
        Selection::Run { .. } => {
            println!("Selection does not imply operation success or proof verification.");
        }
        Selection::Stop(reason) => println!("stop reason: {reason:?}"),
    }
}
