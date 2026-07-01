use symthaea_quantum_comp::{ReplayPlan, ReplayScope};

fn main() {
    let plan = ReplayPlan::for_scope(ReplayScope::Smoke);
    println!("{}", plan.to_markdown());
}
