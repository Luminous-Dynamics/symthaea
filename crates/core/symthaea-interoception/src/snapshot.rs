use serde::{Deserialize, Serialize};

use crate::{
    assess_allostasis, assess_homeostasis, AllostaticConfig, AllostaticReport,
    HomeostaticReport, NativeInteroceptiveModel, NativeInteroceptiveState,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InteroceptiveSnapshot {
    pub cycle: u64,
    pub state: NativeInteroceptiveState,
    pub homeostasis: HomeostaticReport,
    pub allostasis: AllostaticReport,
}

impl InteroceptiveSnapshot {
    pub fn capture(model: &NativeInteroceptiveModel, allostatic_config: AllostaticConfig) -> Self {
        let state = model.state().clone();
        Self {
            cycle: model.cycle(),
            homeostasis: assess_homeostasis(&state),
            allostasis: assess_allostasis(&state, allostatic_config),
            state,
        }
    }
}
