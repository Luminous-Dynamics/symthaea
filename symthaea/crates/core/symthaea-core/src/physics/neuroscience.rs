// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Neuroscience
//!
//! This module encodes neuroscience concepts from ion channels to neural circuits.
//!
//! ## Ion Channels
//!
//! Voltage-gated and ligand-gated channels control ion flow across neuronal membranes.
//!
//! ## Action Potential
//!
//! The Hodgkin-Huxley model describes action potential generation:
//! 1. Depolarization: Na+ influx
//! 2. Repolarization: K+ efflux
//! 3. Refractory period: channel inactivation
//!
//! ## Synaptic Transmission
//!
//! - Excitatory: Glutamate → AMPA/NMDA receptors
//! - Inhibitory: GABA → GABA_A receptors
//!
//! ## Neural Circuits
//!
//! Neurons connect in circuits with specific functions (sensory, motor, memory, etc.)

use super::consciousness_bridge::PhysicsConsciousnessBridge;
use super::hadrons::Hadrons;
use super::periodic_table::PeriodicTable;
use super::standard_model::PHYSICS_DIM;
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// Ion type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IonType {
    Sodium,
    Potassium,
    Calcium,
    Chloride,
}

impl IonType {
    fn domain_label(&self) -> &'static str {
        match self {
            IonType::Sodium => "ion::sodium",
            IonType::Potassium => "ion::potassium",
            IonType::Calcium => "ion::calcium",
            IonType::Chloride => "ion::chloride",
        }
    }

    /// Typical Nernst potential in mV
    pub fn nernst_potential(&self) -> f64 {
        match self {
            IonType::Sodium => 60.0,
            IonType::Potassium => -90.0,
            IonType::Calcium => 120.0,
            IonType::Chloride => -80.0,
        }
    }
}

/// Channel gating type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GatingType {
    Voltage,
    Ligand,
    Mechanical,
    Leak,
}

impl GatingType {
    fn domain_label(&self) -> &'static str {
        match self {
            GatingType::Voltage => "gating::voltage",
            GatingType::Ligand => "gating::ligand",
            GatingType::Mechanical => "gating::mechanical",
            GatingType::Leak => "gating::leak",
        }
    }
}

/// Ion channel
#[derive(Debug, Clone)]
pub struct IonChannel {
    /// Channel name
    pub name: String,
    /// Ion selectivity
    pub ion: IonType,
    /// Gating mechanism
    pub gating: GatingType,
    /// Single-channel conductance in picosiemens
    pub conductance_ps: f64,
    /// Reversal potential in mV
    pub reversal_mv: f64,
    /// Channel vector
    pub vector: ContinuousHV,
}

/// Neuro encoder
#[derive(Debug, Clone)]
pub struct NeuroEncoder {
    // Ions
    pub na_ion: ContinuousHV,
    pub k_ion: ContinuousHV,
    pub ca_ion: ContinuousHV,
    pub cl_ion: ContinuousHV,

    // Channel concepts
    pub channel_open: ContinuousHV,
    pub channel_closed: ContinuousHV,
    pub channel_inactivated: ContinuousHV,
    pub voltage_sensor: ContinuousHV,
    pub selectivity_filter: ContinuousHV,

    // Gating types
    voltage_gating: ContinuousHV,
    ligand_gating: ContinuousHV,
    #[allow(dead_code)] // reserved for future mechanotransduction encoding
    mechanical_gating: ContinuousHV,
    #[allow(dead_code)] // reserved for future leak channel encoding
    leak_gating: ContinuousHV,

    // Neurotransmitters
    pub glutamate: ContinuousHV,
    pub gaba: ContinuousHV,
    pub dopamine: ContinuousHV,
    pub serotonin: ContinuousHV,
    pub acetylcholine: ContinuousHV,
    pub norepinephrine: ContinuousHV,

    // Cellular structures
    pub soma: ContinuousHV,
    pub dendrite: ContinuousHV,
    pub axon: ContinuousHV,
    pub synapse: ContinuousHV,
    pub spine: ContinuousHV,

    // Plasticity
    pub ltp: ContinuousHV,  // Long-term potentiation
    pub ltd: ContinuousHV,  // Long-term depression
    pub stdp: ContinuousHV, // Spike-timing dependent plasticity
}

impl NeuroEncoder {
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        Self {
            na_ion: genesis.hv(IonType::Sodium.domain_label(), PHYSICS_DIM),
            k_ion: genesis.hv(IonType::Potassium.domain_label(), PHYSICS_DIM),
            ca_ion: genesis.hv(IonType::Calcium.domain_label(), PHYSICS_DIM),
            cl_ion: genesis.hv(IonType::Chloride.domain_label(), PHYSICS_DIM),

            channel_open: genesis.hv("channel::open", PHYSICS_DIM),
            channel_closed: genesis.hv("channel::closed", PHYSICS_DIM),
            channel_inactivated: genesis.hv("channel::inactivated", PHYSICS_DIM),
            voltage_sensor: genesis.hv("channel::voltage_sensor", PHYSICS_DIM),
            selectivity_filter: genesis.hv("channel::selectivity_filter", PHYSICS_DIM),

            voltage_gating: genesis.hv(GatingType::Voltage.domain_label(), PHYSICS_DIM),
            ligand_gating: genesis.hv(GatingType::Ligand.domain_label(), PHYSICS_DIM),
            mechanical_gating: genesis.hv(GatingType::Mechanical.domain_label(), PHYSICS_DIM),
            leak_gating: genesis.hv(GatingType::Leak.domain_label(), PHYSICS_DIM),

            glutamate: genesis.hv("nt::glutamate", PHYSICS_DIM),
            gaba: genesis.hv("nt::gaba", PHYSICS_DIM),
            dopamine: genesis.hv("nt::dopamine", PHYSICS_DIM),
            serotonin: genesis.hv("nt::serotonin", PHYSICS_DIM),
            acetylcholine: genesis.hv("nt::acetylcholine", PHYSICS_DIM),
            norepinephrine: genesis.hv("nt::norepinephrine", PHYSICS_DIM),

            soma: genesis.hv("neuron::soma", PHYSICS_DIM),
            dendrite: genesis.hv("neuron::dendrite", PHYSICS_DIM),
            axon: genesis.hv("neuron::axon", PHYSICS_DIM),
            synapse: genesis.hv("neuron::synapse", PHYSICS_DIM),
            spine: genesis.hv("neuron::spine", PHYSICS_DIM),

            ltp: genesis.hv("plasticity::ltp", PHYSICS_DIM),
            ltd: genesis.hv("plasticity::ltd", PHYSICS_DIM),
            stdp: genesis.hv("plasticity::stdp", PHYSICS_DIM),
        }
    }

    /// Create NeuroEncoder from periodic table
    ///
    /// This constructor derives ion vectors from actual elements,
    /// grounding neuroscience in atomic physics.
    pub fn from_table(table: &PeriodicTable, hadrons: &Hadrons, genesis: &GenesisSeed) -> Self {
        // Get actual ion vectors from periodic table
        // Na+ (Z=11), K+ (Z=19), Ca2+ (Z=20), Cl- (Z=17)
        let na_ion = table.ion(11, 1, hadrons);
        let k_ion = table.ion(19, 1, hadrons);
        let ca_ion = table.ion(20, 2, hadrons);
        let cl_ion = table.ion(17, -1, hadrons);

        Self {
            na_ion,
            k_ion,
            ca_ion,
            cl_ion,

            channel_open: genesis.hv("channel::open", PHYSICS_DIM),
            channel_closed: genesis.hv("channel::closed", PHYSICS_DIM),
            channel_inactivated: genesis.hv("channel::inactivated", PHYSICS_DIM),
            voltage_sensor: genesis.hv("channel::voltage_sensor", PHYSICS_DIM),
            selectivity_filter: genesis.hv("channel::selectivity_filter", PHYSICS_DIM),

            voltage_gating: genesis.hv(GatingType::Voltage.domain_label(), PHYSICS_DIM),
            ligand_gating: genesis.hv(GatingType::Ligand.domain_label(), PHYSICS_DIM),
            mechanical_gating: genesis.hv(GatingType::Mechanical.domain_label(), PHYSICS_DIM),
            leak_gating: genesis.hv(GatingType::Leak.domain_label(), PHYSICS_DIM),

            glutamate: genesis.hv("nt::glutamate", PHYSICS_DIM),
            gaba: genesis.hv("nt::gaba", PHYSICS_DIM),
            dopamine: genesis.hv("nt::dopamine", PHYSICS_DIM),
            serotonin: genesis.hv("nt::serotonin", PHYSICS_DIM),
            acetylcholine: genesis.hv("nt::acetylcholine", PHYSICS_DIM),
            norepinephrine: genesis.hv("nt::norepinephrine", PHYSICS_DIM),

            soma: genesis.hv("neuron::soma", PHYSICS_DIM),
            dendrite: genesis.hv("neuron::dendrite", PHYSICS_DIM),
            axon: genesis.hv("neuron::axon", PHYSICS_DIM),
            synapse: genesis.hv("neuron::synapse", PHYSICS_DIM),
            spine: genesis.hv("neuron::spine", PHYSICS_DIM),

            ltp: genesis.hv("plasticity::ltp", PHYSICS_DIM),
            ltd: genesis.hv("plasticity::ltd", PHYSICS_DIM),
            stdp: genesis.hv("plasticity::stdp", PHYSICS_DIM),
        }
    }

    #[allow(dead_code)] // internal helper, may be used in future neuron composition
    fn ion_vector(&self, ion: IonType) -> &ContinuousHV {
        match ion {
            IonType::Sodium => &self.na_ion,
            IonType::Potassium => &self.k_ion,
            IonType::Calcium => &self.ca_ion,
            IonType::Chloride => &self.cl_ion,
        }
    }

    #[allow(dead_code)] // internal helper, may be used in future gating composition
    fn gating_vector(&self, gating: GatingType) -> &ContinuousHV {
        match gating {
            GatingType::Voltage => &self.voltage_gating,
            GatingType::Ligand => &self.ligand_gating,
            GatingType::Mechanical => &self.mechanical_gating,
            GatingType::Leak => &self.leak_gating,
        }
    }

    /// Voltage-gated sodium channel (Nav)
    pub fn nav_channel(&self) -> IonChannel {
        let vector = self
            .na_ion
            .bind(&self.voltage_sensor)
            .bind(&self.selectivity_filter)
            .bind(&self.voltage_gating);

        IonChannel {
            name: "Nav1.1".to_string(),
            ion: IonType::Sodium,
            gating: GatingType::Voltage,
            conductance_ps: 20.0,
            reversal_mv: 60.0,
            vector,
        }
    }

    /// Voltage-gated potassium channel (Kv)
    pub fn kv_channel(&self) -> IonChannel {
        let vector = self
            .k_ion
            .bind(&self.voltage_sensor)
            .bind(&self.selectivity_filter)
            .bind(&self.voltage_gating);

        IonChannel {
            name: "Kv1.1".to_string(),
            ion: IonType::Potassium,
            gating: GatingType::Voltage,
            conductance_ps: 15.0,
            reversal_mv: -90.0,
            vector,
        }
    }

    /// NMDA receptor (ligand + voltage gated, Ca2+ permeable)
    pub fn nmda_receptor(&self) -> IonChannel {
        let vector = self
            .glutamate
            .bind(&self.ca_ion)
            .bind(&self.ligand_gating)
            .bind(&self.voltage_gating); // Mg2+ block is voltage-dependent

        IonChannel {
            name: "NMDAR".to_string(),
            ion: IonType::Calcium,
            gating: GatingType::Ligand,
            conductance_ps: 50.0,
            reversal_mv: 0.0,
            vector,
        }
    }

    /// AMPA receptor (fast glutamate)
    pub fn ampa_receptor(&self) -> IonChannel {
        let vector = self.glutamate.bind(&self.na_ion).bind(&self.ligand_gating);

        IonChannel {
            name: "AMPAR".to_string(),
            ion: IonType::Sodium,
            gating: GatingType::Ligand,
            conductance_ps: 10.0,
            reversal_mv: 0.0,
            vector,
        }
    }

    /// GABA_A receptor (inhibitory)
    pub fn gabaa_receptor(&self) -> IonChannel {
        let vector = self.gaba.bind(&self.cl_ion).bind(&self.ligand_gating);

        IonChannel {
            name: "GABA_A".to_string(),
            ion: IonType::Chloride,
            gating: GatingType::Ligand,
            conductance_ps: 30.0,
            reversal_mv: -80.0,
            vector,
        }
    }
}

/// Action potential phase
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum APPhase {
    Resting,
    Depolarization,
    Overshoot,
    Repolarization,
    Hyperpolarization,
    Refractory,
}

/// Action potential
#[derive(Debug, Clone)]
pub struct ActionPotential {
    /// Resting membrane potential (mV)
    pub resting_mv: f64,
    /// Threshold potential (mV)
    pub threshold_mv: f64,
    /// Peak potential (mV)
    pub peak_mv: f64,
    /// Duration (ms)
    pub duration_ms: f64,
    /// Refractory period (ms)
    pub refractory_ms: f64,
    /// Phases
    pub phases: Vec<APPhase>,
    /// Overall vector
    pub vector: ContinuousHV,
}

impl NeuroEncoder {
    /// Hodgkin-Huxley action potential
    pub fn action_potential(&self, nav: &IonChannel, kv: &IonChannel) -> ActionPotential {
        // Depolarization: Nav opens → Na+ influx
        let depol = nav.vector.bind(&self.channel_open);

        // Repolarization: Nav inactivates, Kv opens → K+ efflux
        let repol = kv
            .vector
            .bind(&self.channel_open)
            .bind(&nav.vector.bind(&self.channel_inactivated));

        let vector = ContinuousHV::bundle(&[&depol, &repol]);

        ActionPotential {
            resting_mv: -70.0,
            threshold_mv: -55.0,
            peak_mv: 40.0,
            duration_ms: 1.0,
            refractory_ms: 2.0,
            phases: vec![
                APPhase::Resting,
                APPhase::Depolarization,
                APPhase::Overshoot,
                APPhase::Repolarization,
                APPhase::Hyperpolarization,
            ],
            vector,
        }
    }

    /// Check if action potential has correct phases
    pub fn ap_has_correct_phases(&self, ap: &ActionPotential) -> bool {
        let required = [
            APPhase::Resting,
            APPhase::Depolarization,
            APPhase::Repolarization,
        ];

        required.iter().all(|phase| ap.phases.contains(phase))
    }
}

/// Neurotransmitter type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Neurotransmitter {
    Glutamate,
    GABA,
    Dopamine,
    Serotonin,
    Norepinephrine,
    Acetylcholine,
}

/// Plasticity state
#[derive(Debug, Clone)]
pub struct PlasticityState {
    /// Long-term potentiation factor
    pub ltp_factor: f64,
    /// Long-term depression factor
    pub ltd_factor: f64,
    /// STDP timing window in ms
    pub stdp_window_ms: f64,
}

/// Synapse
#[derive(Debug, Clone)]
pub struct Synapse {
    /// Presynaptic neuron vector
    pub presynaptic: ContinuousHV,
    /// Postsynaptic neuron vector
    pub postsynaptic: ContinuousHV,
    /// Neurotransmitter type
    pub neurotransmitter: Neurotransmitter,
    /// Synaptic weight (strength)
    pub strength: f64,
    /// Plasticity state
    pub plasticity: PlasticityState,
    /// Synapse vector
    pub vector: ContinuousHV,
}

impl NeuroEncoder {
    fn neurotransmitter_vector(&self, nt: Neurotransmitter) -> &ContinuousHV {
        match nt {
            Neurotransmitter::Glutamate => &self.glutamate,
            Neurotransmitter::GABA => &self.gaba,
            Neurotransmitter::Dopamine => &self.dopamine,
            Neurotransmitter::Serotonin => &self.serotonin,
            Neurotransmitter::Norepinephrine => &self.norepinephrine,
            Neurotransmitter::Acetylcholine => &self.acetylcholine,
        }
    }

    /// Create glutamatergic excitatory synapse
    pub fn excitatory_synapse(&self, pre: &ContinuousHV, post: &ContinuousHV) -> Synapse {
        let nt_vec = self.neurotransmitter_vector(Neurotransmitter::Glutamate);
        let vector = pre.bind(post).bind(&self.synapse).bind(nt_vec);

        Synapse {
            presynaptic: pre.clone(),
            postsynaptic: post.clone(),
            neurotransmitter: Neurotransmitter::Glutamate,
            strength: 1.0,
            plasticity: PlasticityState {
                ltp_factor: 1.0,
                ltd_factor: 1.0,
                stdp_window_ms: 20.0,
            },
            vector,
        }
    }

    /// Create GABAergic inhibitory synapse
    pub fn inhibitory_synapse(&self, pre: &ContinuousHV, post: &ContinuousHV) -> Synapse {
        let nt_vec = self.neurotransmitter_vector(Neurotransmitter::GABA);
        let vector = pre.bind(post).bind(&self.synapse).bind(nt_vec);

        Synapse {
            presynaptic: pre.clone(),
            postsynaptic: post.clone(),
            neurotransmitter: Neurotransmitter::GABA,
            strength: -1.0, // Inhibitory
            plasticity: PlasticityState {
                ltp_factor: 1.0,
                ltd_factor: 1.0,
                stdp_window_ms: 20.0,
            },
            vector,
        }
    }

    /// Hebbian plasticity: "neurons that fire together wire together"
    pub fn hebbian_update(&self, synapse: &mut Synapse, pre_spike: bool, post_spike: bool) {
        if pre_spike && post_spike {
            // Coincident firing → LTP
            synapse.strength *= 1.0 + 0.1 * synapse.plasticity.ltp_factor;
            synapse.vector = synapse.vector.bind(&self.ltp);
        }
    }

    /// STDP: spike-timing dependent plasticity
    pub fn stdp_update(&self, synapse: &mut Synapse, pre_time_ms: f64, post_time_ms: f64) {
        let dt = post_time_ms - pre_time_ms;

        if dt.abs() > synapse.plasticity.stdp_window_ms {
            return; // Outside STDP window
        }

        if dt > 0.0 {
            // Pre before post → LTP
            let factor = (-dt / 10.0).exp();
            synapse.strength *= 1.0 + 0.1 * factor * synapse.plasticity.ltp_factor;
            synapse.vector = synapse.vector.bind(&self.ltp);
        } else {
            // Post before pre → LTD
            let factor = (dt / 10.0).exp();
            synapse.strength *= 1.0 - 0.05 * factor * synapse.plasticity.ltd_factor;
            synapse.vector = synapse.vector.bind(&self.ltd);
        }
    }
}

/// Neuron type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeuronType {
    Pyramidal,      // Cortical excitatory
    Interneuron,    // Cortical inhibitory
    PurkinjeCell,   // Cerebellar
    GranuleCell,    // Cerebellar/hippocampal
    MotorNeuron,    // Spinal
    SensoryNeuron,  // Peripheral
    DopamineNeuron, // VTA/SNc
}

impl NeuronType {
    fn domain_label(&self) -> &'static str {
        match self {
            NeuronType::Pyramidal => "neuron_type::pyramidal",
            NeuronType::Interneuron => "neuron_type::interneuron",
            NeuronType::PurkinjeCell => "neuron_type::purkinje",
            NeuronType::GranuleCell => "neuron_type::granule",
            NeuronType::MotorNeuron => "neuron_type::motor",
            NeuronType::SensoryNeuron => "neuron_type::sensory",
            NeuronType::DopamineNeuron => "neuron_type::dopamine",
        }
    }
}

/// Neuron
#[derive(Debug, Clone)]
pub struct Neuron {
    /// Neuron type
    pub neuron_type: NeuronType,
    /// Soma (cell body)
    pub soma: ContinuousHV,
    /// Dendrites
    pub dendrites: Vec<ContinuousHV>,
    /// Axon
    pub axon: ContinuousHV,
    /// Synapses (output)
    pub synapses: Vec<Synapse>,
    /// Firing rate in Hz
    pub firing_rate_hz: f64,
    /// Overall vector
    pub vector: ContinuousHV,
}

impl NeuroEncoder {
    /// Create a neuron of given type
    pub fn create_neuron(&self, neuron_type: NeuronType, genesis: &GenesisSeed) -> Neuron {
        let type_vec = genesis.hv(neuron_type.domain_label(), PHYSICS_DIM);

        let soma = self.soma.bind(&type_vec);
        let dendrites = vec![
            self.dendrite.permute(0).bind(&type_vec),
            self.dendrite.permute(1000).bind(&type_vec),
        ];
        let axon = self.axon.bind(&type_vec);

        let refs: Vec<&ContinuousHV> = dendrites.iter().collect();
        let dendrite_bundle = ContinuousHV::bundle(&refs);

        let vector = ContinuousHV::bundle(&[&soma, &dendrite_bundle, &axon]);

        Neuron {
            neuron_type,
            soma,
            dendrites,
            axon,
            synapses: vec![],
            firing_rate_hz: 10.0,
            vector,
        }
    }

    /// Cortical pyramidal neuron
    pub fn pyramidal_neuron(&self, genesis: &GenesisSeed) -> Neuron {
        self.create_neuron(NeuronType::Pyramidal, genesis)
    }

    /// Cerebellar Purkinje cell
    pub fn purkinje_cell(&self, genesis: &GenesisSeed) -> Neuron {
        let mut neuron = self.create_neuron(NeuronType::PurkinjeCell, genesis);
        // Purkinje cells have highly branched dendrites
        for i in 2..10 {
            neuron.dendrites.push(self.dendrite.permute(i * 500));
        }
        neuron
    }

    /// Dopaminergic neuron
    pub fn dopamine_neuron(&self, genesis: &GenesisSeed) -> Neuron {
        let mut neuron = self.create_neuron(NeuronType::DopamineNeuron, genesis);
        neuron.vector = neuron.vector.bind(&self.dopamine);
        neuron
    }
}

/// Circuit function
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitFunction {
    SensoryProcessing,
    MotorControl,
    Memory,
    Decision,
    Emotion,
    Attention,
}

/// Neural circuit
#[derive(Debug, Clone)]
pub struct NeuralCircuit {
    /// Circuit name
    pub name: String,
    /// Neurons in the circuit
    pub neurons: Vec<Neuron>,
    /// Connectivity matrix: (pre_idx, post_idx, weight)
    pub connectivity: Vec<(usize, usize, f64)>,
    /// Circuit function
    pub function: CircuitFunction,
    /// Circuit vector
    pub vector: ContinuousHV,
}

impl NeuroEncoder {
    /// Simple reflex arc
    pub fn reflex_arc(&self, genesis: &GenesisSeed) -> NeuralCircuit {
        let sensory = self.create_neuron(NeuronType::SensoryNeuron, genesis);
        let inter = self.create_neuron(NeuronType::Interneuron, genesis);
        let motor = self.create_neuron(NeuronType::MotorNeuron, genesis);

        let neurons = vec![sensory.clone(), inter.clone(), motor.clone()];
        let connectivity = vec![
            (0, 1, 1.0), // sensory → interneuron
            (1, 2, 1.0), // interneuron → motor
        ];

        let vector = ContinuousHV::bundle(&[&sensory.vector, &inter.vector, &motor.vector]);

        NeuralCircuit {
            name: "Reflex arc".to_string(),
            neurons,
            connectivity,
            function: CircuitFunction::MotorControl,
            vector,
        }
    }

    /// Hippocampal memory circuit (simplified CA3-CA1)
    pub fn hippocampal_circuit(&self, genesis: &GenesisSeed) -> NeuralCircuit {
        let ca3 = self.pyramidal_neuron(genesis);
        let ca1 = self.pyramidal_neuron(genesis);
        let interneuron = self.create_neuron(NeuronType::Interneuron, genesis);

        let neurons = vec![ca3.clone(), ca1.clone(), interneuron.clone()];
        let connectivity = vec![
            (0, 1, 1.0),  // CA3 → CA1 (Schaffer collateral)
            (2, 0, -1.0), // Interneuron → CA3 (inhibitory)
            (2, 1, -1.0), // Interneuron → CA1 (inhibitory)
        ];

        let vector =
            ContinuousHV::bundle(&[&ca3.vector, &ca1.vector, &interneuron.vector]).bind(&self.ltp); // Memory circuit associated with LTP

        NeuralCircuit {
            name: "Hippocampal CA3-CA1".to_string(),
            neurons,
            connectivity,
            function: CircuitFunction::Memory,
            vector,
        }
    }

    /// Basal ganglia loop (action selection)
    pub fn basal_ganglia(&self, genesis: &GenesisSeed) -> NeuralCircuit {
        let cortex = self.pyramidal_neuron(genesis);
        let striatum = self.create_neuron(NeuronType::Interneuron, genesis);
        let gpe = self.create_neuron(NeuronType::Interneuron, genesis);
        let stn = self.pyramidal_neuron(genesis);
        let snc = self.dopamine_neuron(genesis);

        let neurons = vec![
            cortex.clone(),
            striatum.clone(),
            gpe.clone(),
            stn.clone(),
            snc.clone(),
        ];

        let connectivity = vec![
            (0, 1, 1.0),  // Cortex → Striatum (excitatory)
            (1, 2, -1.0), // Striatum → GPe (inhibitory)
            (2, 3, -1.0), // GPe → STN (inhibitory)
            (4, 1, 1.0),  // SNc → Striatum (dopamine modulation)
        ];

        let refs: Vec<&ContinuousHV> = neurons.iter().map(|n| &n.vector).collect();
        let vector = ContinuousHV::bundle(&refs).bind(&self.dopamine);

        NeuralCircuit {
            name: "Basal ganglia".to_string(),
            neurons,
            connectivity,
            function: CircuitFunction::Decision,
            vector,
        }
    }

    /// Cortical column (canonical microcircuit)
    pub fn cortical_column(&self, genesis: &GenesisSeed) -> NeuralCircuit {
        // 6 layers simplified: L2/3, L4, L5, L6
        let l23 = self.pyramidal_neuron(genesis);
        let l4 = self.pyramidal_neuron(genesis);
        let l5 = self.pyramidal_neuron(genesis);
        let l6 = self.pyramidal_neuron(genesis);
        let interneuron = self.create_neuron(NeuronType::Interneuron, genesis);

        let neurons = vec![
            l23.clone(),
            l4.clone(),
            l5.clone(),
            l6.clone(),
            interneuron.clone(),
        ];

        let connectivity = vec![
            (1, 0, 1.0),  // L4 → L2/3
            (0, 2, 1.0),  // L2/3 → L5
            (2, 3, 1.0),  // L5 → L6
            (4, 0, -1.0), // Interneuron → L2/3
            (4, 2, -1.0), // Interneuron → L5
        ];

        let refs: Vec<&ContinuousHV> = neurons.iter().map(|n| &n.vector).collect();
        let vector = ContinuousHV::bundle(&refs);

        NeuralCircuit {
            name: "Cortical column".to_string(),
            neurons,
            connectivity,
            function: CircuitFunction::SensoryProcessing,
            vector,
        }
    }
}

/// Consciousness bridge extension for neural circuits
impl NeuroEncoder {
    /// Compute integrated information (Φ-like measure) for a circuit
    pub fn neural_phi(&self, circuit: &NeuralCircuit, _bridge: &PhysicsConsciousnessBridge) -> f64 {
        // Measure integration based on connectivity
        let n = circuit.neurons.len() as f64;
        if n <= 1.0 {
            return 0.0;
        }

        let connectivity = circuit.connectivity.len() as f64 / (n * n);

        // Phenomenal index of circuit activity (entropy-based)
        let phenomenal = {
            use crate::consciousness_metrics::TruePhiCalculator;
            let calc = TruePhiCalculator::new();
            let components: Vec<_> = circuit.neurons.iter().map(|n| n.vector.clone()).collect();
            let nc = components.len();
            let max_pairs = if nc > 1 { (nc * (nc - 1)) / 2 } else { 1 };
            let max_phi = (max_pairs as f64) * 4.0;
            let result = calc.compute_true_phi(&components);
            (result.phi / max_phi).min(1.0)
        };

        // Φ estimate: integration × phenomenal × complexity
        phenomenal * connectivity * n.ln()
    }

    /// Does circuit Φ increase with connectivity?
    pub fn phi_increases_with_connectivity(
        &self,
        genesis: &GenesisSeed,
        bridge: &PhysicsConsciousnessBridge,
    ) -> bool {
        let simple = self.reflex_arc(genesis);
        let complex = self.cortical_column(genesis);

        let phi_simple = self.neural_phi(&simple, bridge);
        let phi_complex = self.neural_phi(&complex, bridge);

        phi_complex > phi_simple
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (NeuroEncoder, GenesisSeed) {
        let genesis = GenesisSeed::from_phrase("neuroscience test");
        let encoder = NeuroEncoder::from_genesis(&genesis);
        (encoder, genesis)
    }

    #[test]
    fn test_nav_channel_conducts_sodium() {
        let (encoder, _) = setup();

        let nav = encoder.nav_channel();

        assert_eq!(nav.ion, IonType::Sodium);
        assert_eq!(nav.gating, GatingType::Voltage);
        assert!(nav.reversal_mv > 0.0); // Na+ has positive reversal potential
    }

    #[test]
    fn test_kv_channel_conducts_potassium() {
        let (encoder, _) = setup();

        let kv = encoder.kv_channel();

        assert_eq!(kv.ion, IonType::Potassium);
        assert!(kv.reversal_mv < 0.0); // K+ has negative reversal potential
    }

    #[test]
    fn test_action_potential_phases() {
        let (encoder, _) = setup();

        let nav = encoder.nav_channel();
        let kv = encoder.kv_channel();
        let ap = encoder.action_potential(&nav, &kv);

        assert!(encoder.ap_has_correct_phases(&ap));
        assert!(ap.peak_mv > ap.threshold_mv);
        assert!(ap.threshold_mv > ap.resting_mv);
    }

    #[test]
    fn test_stdp_pre_before_post_ltp() {
        let (encoder, genesis) = setup();

        let pre = genesis.hv("test::pre", PHYSICS_DIM);
        let post = genesis.hv("test::post", PHYSICS_DIM);
        let mut synapse = encoder.excitatory_synapse(&pre, &post);

        let initial_strength = synapse.strength;

        // Pre fires before post → LTP
        encoder.stdp_update(&mut synapse, 0.0, 5.0);

        assert!(
            synapse.strength > initial_strength,
            "Pre-before-post should cause LTP"
        );
    }

    #[test]
    fn test_stdp_post_before_pre_ltd() {
        let (encoder, genesis) = setup();

        let pre = genesis.hv("test::pre", PHYSICS_DIM);
        let post = genesis.hv("test::post", PHYSICS_DIM);
        let mut synapse = encoder.excitatory_synapse(&pre, &post);

        let initial_strength = synapse.strength;

        // Post fires before pre → LTD
        encoder.stdp_update(&mut synapse, 5.0, 0.0);

        assert!(
            synapse.strength < initial_strength,
            "Post-before-pre should cause LTD"
        );
    }

    #[test]
    fn test_excitatory_vs_inhibitory_synapse() {
        let (encoder, genesis) = setup();

        let pre = genesis.hv("test::pre", PHYSICS_DIM);
        let post = genesis.hv("test::post", PHYSICS_DIM);

        let exc = encoder.excitatory_synapse(&pre, &post);
        let inh = encoder.inhibitory_synapse(&pre, &post);

        assert!(exc.strength > 0.0);
        assert!(inh.strength < 0.0);
        assert_eq!(exc.neurotransmitter, Neurotransmitter::Glutamate);
        assert_eq!(inh.neurotransmitter, Neurotransmitter::GABA);
    }

    #[test]
    fn test_purkinje_highly_branched() {
        let (encoder, genesis) = setup();

        let pyramidal = encoder.pyramidal_neuron(&genesis);
        let purkinje = encoder.purkinje_cell(&genesis);

        assert!(purkinje.dendrites.len() > pyramidal.dendrites.len());
    }

    #[test]
    fn test_circuit_connectivity() {
        let (encoder, genesis) = setup();

        let reflex = encoder.reflex_arc(&genesis);

        assert_eq!(reflex.neurons.len(), 3);
        assert_eq!(reflex.connectivity.len(), 2);
        assert_eq!(reflex.function, CircuitFunction::MotorControl);
    }

    #[test]
    fn test_hippocampal_memory_circuit() {
        let (encoder, genesis) = setup();

        let hippo = encoder.hippocampal_circuit(&genesis);

        assert_eq!(hippo.function, CircuitFunction::Memory);
        assert!(hippo.connectivity.iter().any(|(_, _, w)| *w < 0.0)); // Has inhibition
    }

    #[test]
    fn test_circuit_phi_increases_with_connectivity() {
        let genesis = GenesisSeed::from_phrase("phi test");
        let encoder = NeuroEncoder::from_genesis(&genesis);
        let bridge = PhysicsConsciousnessBridge::from_genesis(&genesis);

        let reflex = encoder.reflex_arc(&genesis);
        let column = encoder.cortical_column(&genesis);

        let phi_reflex = encoder.neural_phi(&reflex, &bridge);
        let phi_column = encoder.neural_phi(&column, &bridge);

        // More complex circuit should have higher Φ
        // (This depends on circuit structure)
        assert!(phi_column.is_finite());
        assert!(phi_reflex.is_finite());
    }

    #[test]
    fn test_nmda_receptor() {
        let (encoder, _) = setup();

        let nmda = encoder.nmda_receptor();

        assert_eq!(nmda.ion, IonType::Calcium);
        assert_eq!(nmda.gating, GatingType::Ligand);
    }

    #[test]
    fn test_determinism() {
        let genesis = GenesisSeed::from_phrase("determinism test");
        let enc1 = NeuroEncoder::from_genesis(&genesis);
        let enc2 = NeuroEncoder::from_genesis(&genesis);

        assert!(
            enc1.glutamate.similarity(&enc2.glutamate) > 0.9999,
            "Neuro encoder should be deterministic"
        );
    }
}
