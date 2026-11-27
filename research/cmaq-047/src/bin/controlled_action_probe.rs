#![forbid(unsafe_code)]

//! Deterministic protocol probe for controlled null-safe diagnostic action selection.
//!
//! Runs four bounded two-dimensional alternatives. The action catalog contains
//! four direction-specific conditional-mean boxes. The monitor must choose an
//! action satisfying the current supporting-halfspace condition before each
//! result is supplied. This demonstrates action adaptation and one-use
//! reconciliation; it is not a statistical power or physical-safety benchmark.

use std::env;
use std::fmt::Write as _;

use cmaq_046_convex_target_approachability::BoxTarget;
use cmaq_047_controlled_blackwell_actions::{
    ActionCatalog, CertifiedDiagnosticAction, ControlledBlackwellMonitor,
};

const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const SHA_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
const SHA_D: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
const SHA_E: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
const GLOBAL_TARGET: [[f64; 2]; 2] = [[-0.05, -0.05], [0.05, 0.05]];
const SHIFT: f64 = 0.35;

const DIRECTIONS: [(&str, [f64; 2]); 4] = [
    ("positive_x", [1.0, 0.0]),
    ("negative_x", [-1.0, 0.0]),
    ("positive_y", [0.0, 1.0]),
    ("negative_y", [0.0, -1.0]),
];

#[derive(Clone, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn sign(&mut self) -> f64 {
        if self.next_u64() & 1 == 0 { -1.0 } else { 1.0 }
    }
}

fn action(
    action_id: &str,
    lower: [f64; 2],
    upper: [f64; 2],
    signature: [f64; 2],
    contract_sha: &str,
) -> Result<CertifiedDiagnosticAction, String> {
    Ok(CertifiedDiagnosticAction {
        action_id: action_id.into(),
        action_scope_id: format!("diagnostic-sensor/{action_id}"),
        null_mean_box: BoxTarget::new(lower.to_vec(), upper.to_vec())
            .map_err(|error| format!("action box invalid: {error:?}"))?,
        diagnostic_signature: signature.to_vec(),
        action_contract_sha256: contract_sha.into(),
        authority_receipt_sha256: SHA_D.into(),
        action_scope_sha256: SHA_E.into(),
    })
}

fn catalog() -> Result<ActionCatalog, String> {
    let actions = vec![
        action(
            "sensor-x-negative",
            [-0.04, -0.05],
            [0.2, 0.05],
            [-1.0, 0.0],
            SHA_D,
        )?,
        action(
            "sensor-x-positive",
            [-0.2, -0.05],
            [0.04, 0.05],
            [1.0, 0.0],
            SHA_E,
        )?,
        action(
            "sensor-y-negative",
            [-0.05, -0.04],
            [0.05, 0.2],
            [0.0, -1.0],
            SHA_D,
        )?,
        action(
            "sensor-y-positive",
            [-0.05, -0.2],
            [0.05, 0.04],
            [0.0, 1.0],
            SHA_E,
        )?,
    ];
    ActionCatalog::new(
        SHA_A.into(),
        SHA_B.into(),
        SHA_C.into(),
        actions,
        2,
    )
    .map_err(|error| format!("catalog invalid: {error:?}"))
}

fn monitor() -> Result<ControlledBlackwellMonitor, String> {
    let target = BoxTarget::new(
        GLOBAL_TARGET[0].to_vec(),
        GLOBAL_TARGET[1].to_vec(),
    )
    .map_err(|error| format!("global target invalid: {error:?}"))?;
    ControlledBlackwellMonitor::new(target, 1.0, 1, 20, SHA_A.into(), catalog()?)
        .map_err(|error| format!("monitor construction failed: {error:?}"))
}

fn null_mean_for(action_id: &str) -> [f64; 2] {
    match action_id {
        "sensor-x-positive" => [-0.08, 0.0],
        "sensor-x-negative" => [0.08, 0.0],
        "sensor-y-positive" => [0.0, -0.08],
        "sensor-y-negative" => [0.0, 0.08],
        _ => [0.0, 0.0],
    }
}

fn observation(
    action_id: &str,
    alternative_direction: Option<[f64; 2]>,
    rng: &mut SplitMix64,
) -> [f64; 2] {
    let mean = alternative_direction.map_or_else(
        || null_mean_for(action_id),
        |direction| [SHIFT * direction[0], SHIFT * direction[1]],
    );
    [
        mean[0] + 0.01 * rng.sign(),
        mean[1] + 0.01 * rng.sign(),
    ]
}

fn run_direction(
    scenario: &str,
    direction_name: &str,
    direction: [f64; 2],
    horizon: usize,
    seed: u64,
    csv: &mut String,
) -> Result<(), String> {
    let mut rng = SplitMix64::new(seed);
    let mut monitor = monitor()?;
    for time in 1..=horizon {
        let plan = monitor
            .propose_action()
            .map_err(|error| format!("action proposal failed: {error:?}"))?;
        let result = observation(
            &plan.action_id,
            (scenario == "directional_alternative").then_some(direction),
            &mut rng,
        );
        let update = monitor
            .reconcile_action_result(&plan, &plan.action_id, &result)
            .map_err(|error| format!("action reconciliation failed: {error:?}"))?;
        writeln!(
            csv,
            "{scenario},{direction_name},{time},{},{:.9},{:.9},{:.9},{:.9},{}",
            update.action_plan.action_id,
            update.action_plan.normal[0],
            update.action_plan.normal[1],
            update.action_plan.null_safety_slack,
            update.action_plan.diagnostic_alignment,
            update.evidence.log_e_value,
        )
        .map_err(|error| error.to_string())?;
        if monitor.is_quarantined() {
            break;
        }
    }
    Ok(())
}

fn parse_horizon() -> Result<usize, String> {
    let mut values = env::args().skip(1);
    let horizon = values
        .next()
        .unwrap_or_else(|| "32".into())
        .parse::<usize>()
        .map_err(|_| "invalid horizon".to_string())?;
    if horizon == 0 || values.next().is_some() {
        return Err("usage: controlled_action_probe [horizon]".into());
    }
    Ok(horizon)
}

fn main() -> Result<(), String> {
    let horizon = parse_horizon()?;
    let mut csv = String::from(
        "scenario,direction,time,action_id,normal_x,normal_y,null_safety_slack,diagnostic_alignment,log_e_value\n",
    );
    for (index, (name, direction)) in DIRECTIONS.into_iter().enumerate() {
        run_direction(
            "conditional_action_null",
            name,
            direction,
            horizon,
            20261127 + index as u64,
            &mut csv,
        )?;
        run_direction(
            "directional_alternative",
            name,
            direction,
            horizon,
            20262127 + index as u64,
            &mut csv,
        )?;
    }
    print!("{csv}");
    Ok(())
}
