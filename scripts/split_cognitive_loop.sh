#!/usr/bin/env bash
# Split cognitive_loop.rs (7,703 lines) into a directory module
# Run from the symthaea project root: bash scripts/split_cognitive_loop.sh
#
# This script:
# 1. Creates src/cognitive_loop/ directory
# 2. Extracts each section into a submodule file
# 3. Creates mod.rs with re-exports
# 4. Removes original cognitive_loop.rs
# 5. Runs cargo check to verify

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$PROJECT_ROOT/src/cognitive_loop.rs"
DIR="$PROJECT_ROOT/src/cognitive_loop"

if [ ! -f "$SRC" ]; then
    echo "ERROR: $SRC not found"
    exit 1
fi

echo "==> Splitting cognitive_loop.rs into directory module..."

# Create directory
mkdir -p "$DIR"

# Backup original
cp "$SRC" "$SRC.bak"

# ─── Common header for all submodules ─────────────────────────────────────

COMMON_IMPORTS='use serde::{Serialize, Deserialize};'

# ─── 1. config.rs (lines 122-430) ────────────────────────────────────────

cat > "$DIR/config.rs" << 'ENDOFFILE'
//! Configuration types for the cognitive loop

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

use symthaea_core::hdc::predictive_encoder::PredictiveEncoderConfig;
use crate::hdc_ltc_bridge::{HdcLtcBridgeConfig};

ENDOFFILE

# Extract TemporalBackend through end of CognitiveLoopConfig impl + CycleResult + Experience + AdaptiveBehavior + ActionHint
sed -n '122,430p' "$SRC" >> "$DIR/config.rs"

# ─── 2. flow.rs (lines 430-707) ──────────────────────────────────────────

cat > "$DIR/flow.rs" << 'ENDOFFILE'
//! Flow state detection and temporal encoding

use serde::{Serialize, Deserialize};
use std::time::Instant;

use crate::dynamics::temporal_signatures::ConsciousnessPattern;

ENDOFFILE

sed -n '430,707p' "$SRC" >> "$DIR/flow.rs"

# ─── 3. learning.rs (lines 708-999) ──────────────────────────────────────

cat > "$DIR/learning.rs" << 'ENDOFFILE'
//! Closed learning loop - strategy-based behavioral adaptation

use serde::{Serialize, Deserialize};
use rand::Rng;

ENDOFFILE

sed -n '708,999p' "$SRC" >> "$DIR/learning.rs"

# ─── 4. bridges.rs (lines 1000-1459) ─────────────────────────────────────

cat > "$DIR/bridges.rs" << 'ENDOFFILE'
//! Memory, goal, and world model bridges

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

use symthaea_core::hdc::predictive_encoder::PredictiveHdcEncoder;
use crate::memory::semantic_memory::SemanticMemory;
use super::config::CycleResult;

ENDOFFILE

sed -n '1000,1459p' "$SRC" >> "$DIR/bridges.rs"

# ─── 5. emotion.rs (lines 1460-1617) ─────────────────────────────────────

cat > "$DIR/emotion.rs" << 'ENDOFFILE'
//! Emotion contagion - emotional content influences consciousness state

use serde::{Serialize, Deserialize};

use crate::dynamics::temporal_signatures::ConsciousnessPattern;

ENDOFFILE

sed -n '1460,1617p' "$SRC" >> "$DIR/emotion.rs"

# ─── 6. curiosity.rs (lines 1618-1765) ───────────────────────────────────

cat > "$DIR/curiosity.rs" << 'ENDOFFILE'
//! Curiosity drive - novelty-seeking mechanism

use serde::{Serialize, Deserialize};

ENDOFFILE

sed -n '1618,1765p' "$SRC" >> "$DIR/curiosity.rs"

# ─── 7. reflection.rs (lines 1766-2274) ──────────────────────────────────

cat > "$DIR/reflection.rs" << 'ENDOFFILE'
//! Self-reflection and meta-cognitive assessment

use serde::{Serialize, Deserialize};

use super::flow::FlowState;
use super::config::ActionHint;

ENDOFFILE

sed -n '1766,2274p' "$SRC" >> "$DIR/reflection.rs"

# ─── 8. thalamic.rs (lines 2275-2436) ────────────────────────────────────

cat > "$DIR/thalamic.rs" << 'ENDOFFILE'
//! Thalamic router - 3-path routing based on novelty/urgency

use serde::{Serialize, Deserialize};

ENDOFFILE

sed -n '2275,2436p' "$SRC" >> "$DIR/thalamic.rs"

# ─── 9. active_inference.rs (lines 2437-2593) ────────────────────────────

cat > "$DIR/active_inference.rs" << 'ENDOFFILE'
//! Active inference bridge - prediction-outcome coupling

use serde::{Serialize, Deserialize};

ENDOFFILE

sed -n '2437,2593p' "$SRC" >> "$DIR/active_inference.rs"

# ─── 10. snapshot.rs (lines 2594-3013) ───────────────────────────────────

cat > "$DIR/snapshot.rs" << 'ENDOFFILE'
//! Unified consciousness snapshot - aggregates all cognitive metrics

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

use crate::dynamics::temporal_signatures::{ConsciousnessPattern, TemporalStateSummary};
use crate::dynamics::cfc_coherence::CoherenceSummary;
use crate::voice::voice_feedback::{VoiceOutputMetrics, VoiceQualitySummary};
use super::config::{ActionHint, AdaptiveBehavior};
use super::flow::FlowState;
use super::emotion::EmotionContagion;
use super::curiosity::CuriosityDrive;
use super::reflection::SelfReflection;
use super::thalamic::ThalamicRouter;
use super::active_inference::ActiveInferenceBridge;
use super::learning::ClosedLearningLoop;

ENDOFFILE

sed -n '2594,3013p' "$SRC" >> "$DIR/snapshot.rs"

# ─── 11. stats.rs (lines 3014-3563) ──────────────────────────────────────

cat > "$DIR/stats.rs" << 'ENDOFFILE'
//! Loop statistics and metrics

use serde::{Serialize, Deserialize};

use super::config::ActionHint;

ENDOFFILE

sed -n '3014,3563p' "$SRC" >> "$DIR/stats.rs"

# ─── 12. service.rs (lines 3564-6017) ────────────────────────────────────

cat > "$DIR/service.rs" << 'ENDOFFILE'
//! CognitiveLoopService - the main orchestration service

use anyhow::Result;
use rand::Rng;
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, VecDeque};
use std::sync::mpsc;
use std::time::{Duration, Instant};
use ndarray::Array1;
use symthaea_core::genesis::ShakeRng;

use symthaea_core::hdc::predictive_encoder::{PredictiveHdcEncoder, PredictiveEncoderConfig};
use crate::cfc::CfCNetwork;
use crate::dynamics::cfc_coherence::{CfCCoherenceBridge, CoherenceConfig, CoherenceSummary};
use crate::dynamics::temporal_signatures::{
    TemporalSignatureEncoder, SignatureConfig, ConsciousnessPattern, TemporalStateSummary
};
use crate::voice::voice_feedback::{VoiceFeedbackBridge, VoiceFeedbackConfig, VoiceOutputMetrics, VoiceQualitySummary};
use crate::consciousness::consciousness_unification::{
    ConsciousnessUnificationEngine, UnifiedEmotionalState, UnifiedEmotion,
    EmotionalPattern,
};
use crate::consciousness::fep_active_inference::{
    ActiveInferenceAgent, ActiveInferenceAgentConfig, Observation,
    EnhancedFEPBridge, MotorCommandType,
};
use crate::memory::coherence_tracker::ConversationCoherenceTracker;
use crate::memory::semantic_memory::SemanticMemory;
use crate::hdc_ltc_bridge::{HdcLtcBridge, HdcLtcBridgeConfig};
use crate::consciousness::stability_regime::{StabilityRegimeProcessor, RegimeTransition};
use crate::consciousness::primitive_discovery::{PrimitiveDiscoveryService, DiscoveryServiceConfig};
use symthaea_core::hdc::phi_topology_validation::real_hv_to_hv16;
use crate::causal::{CausalLoopEnhancer, CausalEnhancerConfig, CausalGraph, DiscoveredRelationship};

#[cfg(feature = "neural-bridge")]
use crate::perception::NeuralBridge;

use super::*;

ENDOFFILE

sed -n '3564,6017p' "$SRC" >> "$DIR/service.rs"

# ─── 13. builder.rs (lines 6018-6216) ────────────────────────────────────

cat > "$DIR/builder.rs" << 'ENDOFFILE'
//! CognitiveLoopBuilder - builder pattern for configuring the cognitive loop

use anyhow::Result;

use super::config::CognitiveLoopConfig;
use super::service::CognitiveLoopService;

ENDOFFILE

sed -n '6018,6216p' "$SRC" >> "$DIR/builder.rs"

# ─── 14. executor.rs (lines 6217-6340) ───────────────────────────────────

cat > "$DIR/executor.rs" << 'ENDOFFILE'
//! CognitiveLoopExecutor - command execution with Phi-gating

use super::*;

ENDOFFILE

sed -n '6217,6340p' "$SRC" >> "$DIR/executor.rs"

# ─── 15. mod.rs ──────────────────────────────────────────────────────────

# Extract the doc comment (lines 1-76) and create mod.rs
cat > "$DIR/mod.rs" << 'ENDOFMOD'
ENDOFMOD

sed -n '1,76p' "$SRC" >> "$DIR/mod.rs"

cat >> "$DIR/mod.rs" << 'ENDOFMOD'

mod config;
mod flow;
mod learning;
mod bridges;
mod emotion;
mod curiosity;
mod reflection;
mod thalamic;
mod active_inference;
mod snapshot;
mod stats;
pub mod service;
mod builder;
mod executor;

// Re-export all public types
pub use config::*;
pub use flow::*;
pub use learning::*;
pub use bridges::*;
pub use emotion::*;
pub use curiosity::*;
pub use reflection::*;
pub use thalamic::*;
pub use active_inference::*;
pub use snapshot::*;
pub use stats::*;
pub use service::CognitiveLoopService;
pub use builder::CognitiveLoopBuilder;
pub use executor::CognitiveLoopExecutor;

ENDOFMOD

# Extract the test module (lines 6341-end)
sed -n '6341,$p' "$SRC" >> "$DIR/mod.rs"

# ─── Remove original and verify ──────────────────────────────────────────

echo "==> Files created:"
ls -la "$DIR/"
echo ""
echo "==> Running cargo check..."

cd "$PROJECT_ROOT"
if cargo check 2>&1; then
    echo "==> SUCCESS: Compilation passed!"
    echo "==> Removing backup..."
    rm "$SRC.bak"
    echo "==> Done! cognitive_loop.rs split into $(ls "$DIR"/*.rs | wc -l) submodules."
else
    echo ""
    echo "==> COMPILATION FAILED. Restoring original..."
    rm -rf "$DIR"
    mv "$SRC.bak" "$SRC"
    echo "==> Original restored. Fix the errors and re-run."
    exit 1
fi
