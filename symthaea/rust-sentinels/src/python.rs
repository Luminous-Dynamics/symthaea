// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Python bindings for the Sentinels library
//!
//! Exposes all six consciousness detection Sentinels to Python via PyO3.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::sentinels::{
    // Extended Sentinels
    AttentionSentinel,
    // Trilogy Sentinels
    EmotionSentinel,
    EngagementSentinel,
    FlowSentinel,
    MeditationSentinel,
    SleepSentinel,
};
use crate::{AnalysisMode, analyze_consciousness, analyze_extended};

// ============================================================================
// Python-compatible wrapper structs
// ============================================================================

/// Python-compatible EmotionSentinel wrapper
#[pyclass(name = "EmotionSentinel")]
pub struct PyEmotionSentinel {
    inner: EmotionSentinel,
}

#[pymethods]
impl PyEmotionSentinel {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: EmotionSentinel::default(),
        }
    }

    /// Calibrate baseline from resting state data (two-channel for asymmetry)
    pub fn calibrate(&mut self, left_data: Vec<f32>, right_data: Vec<f32>, sample_rate: f32) {
        self.inner.calibrate(&left_data, &right_data, sample_rate);
    }

    /// Analyze emotional state from EEG data (single channel)
    /// Returns dict with valence, arousal, confidence, quadrant
    pub fn analyze(
        &self,
        py: Python<'_>,
        data: Vec<f32>,
        sample_rate: f32,
    ) -> PyResult<Py<PyDict>> {
        let score = self.inner.analyze(&data, sample_rate);
        let dict = PyDict::new_bound(py);
        dict.set_item("valence", score.valence)?;
        dict.set_item("arousal", score.arousal)?;
        dict.set_item("confidence", score.confidence)?;
        dict.set_item("quadrant", format!("{:?}", score.quadrant()))?;
        Ok(dict.unbind())
    }
}

/// Python-compatible SleepSentinel wrapper
#[pyclass(name = "SleepSentinel")]
pub struct PySleepSentinel {
    inner: SleepSentinel,
}

#[pymethods]
impl PySleepSentinel {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: SleepSentinel::default(),
        }
    }

    /// Analyze sleep stage from EEG data
    /// Returns dict with stage, confidence, delta_power, sleep_quality
    pub fn analyze(
        &self,
        py: Python<'_>,
        data: Vec<f32>,
        sample_rate: f32,
    ) -> PyResult<Py<PyDict>> {
        let score = self.inner.analyze(&data, sample_rate);
        let dict = PyDict::new_bound(py);
        dict.set_item("stage", score.stage.name())?;
        dict.set_item("confidence", score.confidence)?;
        dict.set_item("delta_power", score.delta_power)?;
        dict.set_item("sleep_quality", score.sleep_quality)?;
        dict.set_item("is_deep_sleep", score.is_deep_sleep())?;
        dict.set_item("is_awake", score.is_awake())?;
        Ok(dict.unbind())
    }
}

/// Python-compatible MeditationSentinel wrapper
#[pyclass(name = "MeditationSentinel")]
pub struct PyMeditationSentinel {
    inner: MeditationSentinel,
}

#[pymethods]
impl PyMeditationSentinel {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: MeditationSentinel::default(),
        }
    }

    /// Calibrate baseline from resting state
    pub fn calibrate(&mut self, data: Vec<f32>, sample_rate: f32) {
        self.inner.calibrate(&data, sample_rate);
    }

    /// Analyze meditation state from EEG data
    /// Returns dict with depth, stability, alpha_power, theta_power, meditation_index, state
    pub fn analyze(
        &self,
        py: Python<'_>,
        data: Vec<f32>,
        sample_rate: f32,
    ) -> PyResult<Py<PyDict>> {
        let score = self.inner.analyze(&data, sample_rate);
        let dict = PyDict::new_bound(py);
        dict.set_item("depth", score.depth)?;
        dict.set_item("stability", score.stability)?;
        dict.set_item("alpha_power", score.alpha_power)?;
        dict.set_item("theta_power", score.theta_power)?;
        dict.set_item("meditation_index", score.meditation_index)?;
        dict.set_item("state", format!("{:?}", score.state()))?;
        Ok(dict.unbind())
    }
}

/// Python-compatible AttentionSentinel wrapper
#[pyclass(name = "AttentionSentinel")]
pub struct PyAttentionSentinel {
    inner: AttentionSentinel,
}

#[pymethods]
impl PyAttentionSentinel {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: AttentionSentinel::default(),
        }
    }

    /// Calibrate baseline from resting state
    pub fn calibrate(&mut self, data: Vec<f32>, sample_rate: f32) {
        self.inner.calibrate(&data, sample_rate);
    }

    /// Analyze attention state from EEG data
    /// Returns dict with sustained, selective, alertness, cognitive_load, attention_index, state, confidence
    pub fn analyze(
        &self,
        py: Python<'_>,
        data: Vec<f32>,
        sample_rate: f32,
    ) -> PyResult<Py<PyDict>> {
        let score = self.inner.analyze(&data, sample_rate);
        let dict = PyDict::new_bound(py);
        dict.set_item("sustained", score.sustained)?;
        dict.set_item("selective", score.selective)?;
        dict.set_item("alertness", score.alertness)?;
        dict.set_item("cognitive_load", score.cognitive_load)?;
        dict.set_item("attention_index", score.attention_index)?;
        dict.set_item("state", format!("{:?}", score.state))?;
        dict.set_item("confidence", score.confidence)?;
        dict.set_item("description", score.description())?;
        Ok(dict.unbind())
    }
}

/// Python-compatible FlowSentinel wrapper
#[pyclass(name = "FlowSentinel")]
pub struct PyFlowSentinel {
    inner: FlowSentinel,
}

#[pymethods]
impl PyFlowSentinel {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: FlowSentinel::default(),
        }
    }

    /// Calibrate baseline from resting state
    pub fn calibrate(&mut self, data: Vec<f32>, sample_rate: f32) {
        self.inner.calibrate(&data, sample_rate);
    }

    /// Analyze flow state from EEG data
    /// Returns dict with immersion, effortlessness, time_dilation, performance, flow_index, state, confidence
    pub fn analyze(
        &self,
        py: Python<'_>,
        data: Vec<f32>,
        sample_rate: f32,
    ) -> PyResult<Py<PyDict>> {
        let score = self.inner.analyze(&data, sample_rate);
        let dict = PyDict::new_bound(py);
        dict.set_item("immersion", score.immersion)?;
        dict.set_item("effortlessness", score.effortlessness)?;
        dict.set_item("time_dilation", score.time_dilation)?;
        dict.set_item("performance", score.performance)?;
        dict.set_item("flow_index", score.flow_index)?;
        dict.set_item("state", format!("{:?}", score.state))?;
        dict.set_item("confidence", score.confidence)?;
        dict.set_item("in_flow", score.in_flow())?;
        dict.set_item("description", score.description())?;
        Ok(dict.unbind())
    }
}

/// Python-compatible EngagementSentinel wrapper
#[pyclass(name = "EngagementSentinel")]
pub struct PyEngagementSentinel {
    inner: EngagementSentinel,
}

#[pymethods]
impl PyEngagementSentinel {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: EngagementSentinel::default(),
        }
    }

    /// Calibrate baseline from resting/neutral state
    pub fn calibrate(&mut self, data: Vec<f32>, sample_rate: f32) {
        self.inner.calibrate(&data, sample_rate);
    }

    /// Analyze engagement state from EEG data
    /// Returns dict with cognitive, emotional, interest, fatigue, workload, engagement_index, level, confidence
    pub fn analyze(
        &self,
        py: Python<'_>,
        data: Vec<f32>,
        sample_rate: f32,
    ) -> PyResult<Py<PyDict>> {
        let score = self.inner.analyze(&data, sample_rate);
        let dict = PyDict::new_bound(py);
        dict.set_item("cognitive", score.cognitive)?;
        dict.set_item("emotional", score.emotional)?;
        dict.set_item("interest", score.interest)?;
        dict.set_item("fatigue", score.fatigue)?;
        dict.set_item("workload", score.workload)?;
        dict.set_item("engagement_index", score.engagement_index)?;
        dict.set_item("level", format!("{:?}", score.level))?;
        dict.set_item("confidence", score.confidence)?;
        dict.set_item("is_engaged", score.is_engaged())?;
        dict.set_item("needs_break", score.needs_break())?;
        dict.set_item("description", score.description())?;
        Ok(dict.unbind())
    }
}

// ============================================================================
// High-level API functions
// ============================================================================

/// Analyze consciousness state with automatic mode selection
#[pyfunction]
#[pyo3(name = "analyze", signature = (data, sample_rate, mode = "auto"))]
pub fn py_analyze_consciousness(
    py: Python<'_>,
    data: Vec<f32>,
    sample_rate: f32,
    mode: &str,
) -> PyResult<Py<PyDict>> {
    let analysis_mode = match mode.to_lowercase().as_str() {
        "auto" => AnalysisMode::Auto,
        "trilogy" => AnalysisMode::Trilogy,
        "emotion" => AnalysisMode::Emotion,
        "sleep" => AnalysisMode::Sleep,
        "meditation" => AnalysisMode::Meditation,
        "attention" => AnalysisMode::Attention,
        "flow" => AnalysisMode::Flow,
        "engagement" => AnalysisMode::Engagement,
        "extended" => AnalysisMode::Extended,
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown mode: {}. Valid modes: auto, trilogy, emotion, sleep, meditation, attention, flow, engagement, extended",
                mode
            )));
        }
    };

    match analyze_consciousness(&data, sample_rate, analysis_mode) {
        Ok(poc) => {
            let dict = PyDict::new_bound(py);
            dict.set_item("timestamp", poc.timestamp)?;
            dict.set_item("state", format!("{:?}", poc.state))?;
            dict.set_item("consciousness_level", poc.consciousness_level)?;
            dict.set_item("wellbeing_score", poc.wellbeing_score)?;

            if let Some(emotion) = poc.emotion {
                let emotion_dict = PyDict::new_bound(py);
                emotion_dict.set_item("valence", emotion.valence)?;
                emotion_dict.set_item("arousal", emotion.arousal)?;
                emotion_dict.set_item("confidence", emotion.confidence)?;
                dict.set_item("emotion", emotion_dict)?;
            }

            if let Some(sleep) = poc.sleep {
                let sleep_dict = PyDict::new_bound(py);
                sleep_dict.set_item("stage", sleep.stage.name())?;
                sleep_dict.set_item("confidence", sleep.confidence)?;
                sleep_dict.set_item("delta_power", sleep.delta_power)?;
                sleep_dict.set_item("sleep_quality", sleep.sleep_quality)?;
                dict.set_item("sleep", sleep_dict)?;
            }

            if let Some(meditation) = poc.meditation {
                let meditation_dict = PyDict::new_bound(py);
                meditation_dict.set_item("depth", meditation.depth)?;
                meditation_dict.set_item("stability", meditation.stability)?;
                meditation_dict.set_item("alpha_power", meditation.alpha_power)?;
                meditation_dict.set_item("theta_power", meditation.theta_power)?;
                meditation_dict.set_item("meditation_index", meditation.meditation_index)?;
                dict.set_item("meditation", meditation_dict)?;
            }

            Ok(dict.unbind())
        }
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "{}",
            e
        ))),
    }
}

/// Analyze with Extended Proofs (all six Sentinels)
#[pyfunction]
#[pyo3(name = "analyze_extended", signature = (data, sample_rate, mode = "extended"))]
pub fn py_analyze_extended(
    py: Python<'_>,
    data: Vec<f32>,
    sample_rate: f32,
    mode: &str,
) -> PyResult<Py<PyDict>> {
    let analysis_mode = match mode.to_lowercase().as_str() {
        "attention" => AnalysisMode::Attention,
        "flow" => AnalysisMode::Flow,
        "engagement" => AnalysisMode::Engagement,
        "extended" | _ => AnalysisMode::Extended,
    };

    match analyze_extended(&data, sample_rate, analysis_mode) {
        Ok(extended) => {
            let dict = PyDict::new_bound(py);
            dict.set_item("performance_potential", extended.performance_potential)?;
            dict.set_item("learning_readiness", extended.learning_readiness)?;

            // Attention
            if let Some(attention) = extended.attention {
                let att_dict = PyDict::new_bound(py);
                att_dict.set_item("sustained", attention.sustained)?;
                att_dict.set_item("selective", attention.selective)?;
                att_dict.set_item("alertness", attention.alertness)?;
                att_dict.set_item("cognitive_load", attention.cognitive_load)?;
                att_dict.set_item("attention_index", attention.attention_index)?;
                att_dict.set_item("state", format!("{:?}", attention.state))?;
                att_dict.set_item("confidence", attention.confidence)?;
                dict.set_item("attention", att_dict)?;
            }

            // Flow
            if let Some(flow) = extended.flow {
                let flow_dict = PyDict::new_bound(py);
                flow_dict.set_item("immersion", flow.immersion)?;
                flow_dict.set_item("effortlessness", flow.effortlessness)?;
                flow_dict.set_item("time_dilation", flow.time_dilation)?;
                flow_dict.set_item("performance", flow.performance)?;
                flow_dict.set_item("flow_index", flow.flow_index)?;
                flow_dict.set_item("state", format!("{:?}", flow.state))?;
                flow_dict.set_item("confidence", flow.confidence)?;
                dict.set_item("flow", flow_dict)?;
            }

            // Engagement
            if let Some(engagement) = extended.engagement {
                let eng_dict = PyDict::new_bound(py);
                eng_dict.set_item("cognitive", engagement.cognitive)?;
                eng_dict.set_item("emotional", engagement.emotional)?;
                eng_dict.set_item("interest", engagement.interest)?;
                eng_dict.set_item("fatigue", engagement.fatigue)?;
                eng_dict.set_item("workload", engagement.workload)?;
                eng_dict.set_item("engagement_index", engagement.engagement_index)?;
                eng_dict.set_item("level", format!("{:?}", engagement.level))?;
                eng_dict.set_item("confidence", engagement.confidence)?;
                dict.set_item("engagement", eng_dict)?;
            }

            // Include trilogy if present
            if let Some(emotion) = extended.emotion {
                let emotion_dict = PyDict::new_bound(py);
                emotion_dict.set_item("valence", emotion.valence)?;
                emotion_dict.set_item("arousal", emotion.arousal)?;
                emotion_dict.set_item("confidence", emotion.confidence)?;
                dict.set_item("emotion", emotion_dict)?;
            }

            if let Some(sleep) = extended.sleep {
                let sleep_dict = PyDict::new_bound(py);
                sleep_dict.set_item("stage", sleep.stage.name())?;
                sleep_dict.set_item("confidence", sleep.confidence)?;
                dict.set_item("sleep", sleep_dict)?;
            }

            if let Some(meditation) = extended.meditation {
                let meditation_dict = PyDict::new_bound(py);
                meditation_dict.set_item("depth", meditation.depth)?;
                meditation_dict.set_item("stability", meditation.stability)?;
                dict.set_item("meditation", meditation_dict)?;
            }

            Ok(dict.unbind())
        }
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "{}",
            e
        ))),
    }
}

// ============================================================================
// Module registration
// ============================================================================

/// Python module for consciousness state detection from biosignals
#[pymodule]
fn sentinels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Trilogy Sentinels
    m.add_class::<PyEmotionSentinel>()?;
    m.add_class::<PySleepSentinel>()?;
    m.add_class::<PyMeditationSentinel>()?;

    // Extended Sentinels
    m.add_class::<PyAttentionSentinel>()?;
    m.add_class::<PyFlowSentinel>()?;
    m.add_class::<PyEngagementSentinel>()?;

    // High-level functions
    m.add_function(wrap_pyfunction!(py_analyze_consciousness, m)?)?;
    m.add_function(wrap_pyfunction!(py_analyze_extended, m)?)?;

    // Version
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}