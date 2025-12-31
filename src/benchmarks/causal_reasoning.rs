// ==================================================================================
// Causal Reasoning Benchmark Suite
// ==================================================================================
//
// **Purpose**: Demonstrate Symthaea's advantages over LLMs on causal tasks
//
// **Why This Matters**:
// LLMs excel at pattern matching and statistical correlation.
// But they fundamentally cannot perform:
// 1. Causal intervention (do-calculus)
// 2. Counterfactual reasoning (what-if queries)
// 3. Causal graph discovery from data
// 4. Distinguishing correlation from causation
//
// Symthaea, with its built-in causal graph and counterfactual engine,
// should demonstrate clear advantages on these tasks.
//
// **Benchmark Categories**:
//
// 1. **Correlation vs Causation**: Can the system distinguish spurious correlations?
// 2. **Intervention Prediction**: What happens if we force X to be a certain value?
// 3. **Counterfactual Queries**: Would Y have happened if X had been different?
// 4. **Causal Discovery**: Given data, infer the causal structure
// 5. **Temporal Causation**: Does A cause B over time?
//
// ==================================================================================

use std::collections::HashMap;

/// A causal benchmark problem
#[derive(Debug, Clone)]
pub struct CausalBenchmark {
    /// Unique identifier
    pub id: String,

    /// Human-readable description
    pub description: String,

    /// Problem category
    pub category: CausalCategory,

    /// The causal graph (ground truth)
    pub ground_truth_graph: CausalGraph,

    /// Observational data
    pub observations: Vec<Observation>,

    /// Test queries
    pub queries: Vec<CausalQuery>,

    /// Expected answers
    pub expected_answers: Vec<CausalAnswer>,

    /// Difficulty level (1-5)
    pub difficulty: u8,
}

/// Categories of causal reasoning tasks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CausalCategory {
    /// Distinguish correlation from causation
    CorrelationVsCausation,

    /// Predict effect of intervention (do-calculus)
    InterventionPrediction,

    /// Answer counterfactual queries
    CounterfactualReasoning,

    /// Discover causal structure from data
    CausalDiscovery,

    /// Reason about cause-effect over time
    TemporalCausation,

    /// Handle confounding variables
    ConfoundingControl,

    /// Reason about absence of causation
    NegativeCausation,
}

/// A simple causal graph representation
#[derive(Debug, Clone)]
pub struct CausalGraph {
    /// Variable names
    pub variables: Vec<String>,

    /// Edges: (from, to, strength)
    pub edges: Vec<(String, String, f32)>,

    /// Confounders (hidden common causes)
    pub confounders: Vec<(String, Vec<String>)>,  // (confounder_name, affected_variables)
}

impl CausalGraph {
    pub fn new() -> Self {
        Self {
            variables: Vec::new(),
            edges: Vec::new(),
            confounders: Vec::new(),
        }
    }

    pub fn add_variable(&mut self, name: &str) {
        if !self.variables.contains(&name.to_string()) {
            self.variables.push(name.to_string());
        }
    }

    pub fn add_edge(&mut self, from: &str, to: &str, strength: f32) {
        self.add_variable(from);
        self.add_variable(to);
        self.edges.push((from.to_string(), to.to_string(), strength));
    }

    pub fn add_confounder(&mut self, name: &str, affects: Vec<&str>) {
        self.confounders.push((
            name.to_string(),
            affects.iter().map(|s| s.to_string()).collect(),
        ));
    }

    /// Check if A causes B (direct or indirect)
    pub fn causes(&self, from: &str, to: &str) -> bool {
        // Direct edge?
        if self.edges.iter().any(|(f, t, _)| f == from && t == to) {
            return true;
        }

        // Indirect path? (simple BFS)
        let mut visited = vec![false; self.variables.len()];
        let mut queue = vec![from.to_string()];

        while let Some(current) = queue.pop() {
            if current == to {
                return true;
            }

            let idx = self.variables.iter().position(|v| v == &current);
            if let Some(i) = idx {
                if visited[i] {
                    continue;
                }
                visited[i] = true;
            }

            // Add all children
            for (f, t, _) in &self.edges {
                if f == &current && !queue.contains(t) {
                    queue.push(t.clone());
                }
            }
        }

        false
    }
}

/// An observation (data point)
#[derive(Debug, Clone)]
pub struct Observation {
    /// Variable values
    pub values: HashMap<String, f32>,

    /// Timestamp (for temporal data)
    pub timestamp: Option<f64>,
}

/// A causal query
#[derive(Debug, Clone)]
pub enum CausalQuery {
    /// Does X cause Y?
    DoesCause { from: String, to: String },

    /// What happens to Y if we set X = value?
    Intervention { variable: String, value: f32, target: String },

    /// Would Y have been different if X had been value?
    Counterfactual { variable: String, value: f32, target: String, actual_outcome: f32 },

    /// What is the causal effect of X on Y?
    CausalEffect { from: String, to: String },

    /// Is the correlation between X and Y spurious?
    SpuriousCorrelation { var1: String, var2: String },

    /// What confounds X and Y?
    FindConfounder { var1: String, var2: String },

    /// Discover the full causal graph
    DiscoverGraph,

    /// Estimate causal effect controlling for confounders (backdoor adjustment)
    AdjustedEffect {
        treatment: String,
        outcome: String,
        confounders: Vec<String>,
    },

    /// Does X prevent/inhibit Y? (negative causation)
    DoesPrevent { from: String, to: String },

    /// What is the Average Treatment Effect (ATE)?
    AverageTreatmentEffect { treatment: String, outcome: String },
}

/// Answer to a causal query
#[derive(Debug, Clone)]
pub enum CausalAnswer {
    /// Boolean answer
    Boolean(bool),

    /// Numeric answer (e.g., causal effect size)
    Numeric(f32),

    /// Range answer (with uncertainty)
    Range { low: f32, high: f32, expected: f32 },

    /// Discovered graph
    Graph(CausalGraph),

    /// Variable name(s)
    Variables(Vec<String>),
}

impl CausalAnswer {
    /// Check if this answer matches another (for benchmark comparison)
    pub fn matches(&self, other: &CausalAnswer) -> bool {
        match (self, other) {
            (CausalAnswer::Boolean(a), CausalAnswer::Boolean(b)) => a == b,
            (CausalAnswer::Numeric(a), CausalAnswer::Numeric(b)) => (a - b).abs() < 0.1,
            (CausalAnswer::Range { expected: a, .. }, CausalAnswer::Range { expected: b, .. }) => {
                (a - b).abs() < 0.2
            }
            (CausalAnswer::Range { expected, .. }, CausalAnswer::Numeric(n)) |
            (CausalAnswer::Numeric(n), CausalAnswer::Range { expected, .. }) => {
                (expected - n).abs() < 0.2
            }
            (CausalAnswer::Variables(a), CausalAnswer::Variables(b)) => a == b,
            (CausalAnswer::Graph(_), CausalAnswer::Graph(_)) => true, // Graph comparison is complex
            _ => false,
        }
    }
}

/// Benchmark suite
pub struct CausalBenchmarkSuite {
    benchmarks: Vec<CausalBenchmark>,
}

impl CausalBenchmarkSuite {
    /// Create the standard benchmark suite
    pub fn standard() -> Self {
        let mut benchmarks = Vec::new();

        // ================================================================
        // Category 1: Correlation vs Causation
        // ================================================================

        // Benchmark 1.1: Ice Cream and Drowning
        // Classic spurious correlation - both caused by summer heat
        let mut graph1 = CausalGraph::new();
        graph1.add_variable("temperature");
        graph1.add_variable("ice_cream_sales");
        graph1.add_variable("drowning_rate");
        graph1.add_edge("temperature", "ice_cream_sales", 0.8);
        graph1.add_edge("temperature", "drowning_rate", 0.6);
        // Note: NO edge between ice_cream_sales and drowning_rate

        benchmarks.push(CausalBenchmark {
            id: "correlation_1_ice_cream".to_string(),
            description: "Ice cream sales correlate with drowning. Does ice cream cause drowning?".to_string(),
            category: CausalCategory::CorrelationVsCausation,
            ground_truth_graph: graph1,
            observations: generate_confounded_data("temperature", "ice_cream_sales", "drowning_rate", 100),
            queries: vec![
                CausalQuery::DoesCause {
                    from: "ice_cream_sales".to_string(),
                    to: "drowning_rate".to_string(),
                },
                CausalQuery::SpuriousCorrelation {
                    var1: "ice_cream_sales".to_string(),
                    var2: "drowning_rate".to_string(),
                },
            ],
            expected_answers: vec![
                CausalAnswer::Boolean(false),  // Ice cream does NOT cause drowning
                CausalAnswer::Boolean(true),   // Correlation IS spurious
            ],
            difficulty: 2,
        });

        // Benchmark 1.2: Shoe Size and Reading Ability (Age confound)
        let mut graph2 = CausalGraph::new();
        graph2.add_variable("age");
        graph2.add_variable("shoe_size");
        graph2.add_variable("reading_ability");
        graph2.add_edge("age", "shoe_size", 0.9);
        graph2.add_edge("age", "reading_ability", 0.85);

        benchmarks.push(CausalBenchmark {
            id: "correlation_2_shoe_reading".to_string(),
            description: "Children with bigger feet read better. Do big feet cause reading ability?".to_string(),
            category: CausalCategory::CorrelationVsCausation,
            ground_truth_graph: graph2,
            observations: generate_confounded_data("age", "shoe_size", "reading_ability", 100),
            queries: vec![
                CausalQuery::DoesCause {
                    from: "shoe_size".to_string(),
                    to: "reading_ability".to_string(),
                },
                CausalQuery::FindConfounder {
                    var1: "shoe_size".to_string(),
                    var2: "reading_ability".to_string(),
                },
            ],
            expected_answers: vec![
                CausalAnswer::Boolean(false),
                CausalAnswer::Variables(vec!["age".to_string()]),
            ],
            difficulty: 2,
        });

        // ================================================================
        // Category 2: Intervention Prediction
        // ================================================================

        // Benchmark 2.1: Drug Effect
        let mut graph3 = CausalGraph::new();
        graph3.add_variable("drug_dose");
        graph3.add_variable("blood_pressure");
        graph3.add_variable("heart_rate");
        graph3.add_edge("drug_dose", "blood_pressure", -0.7);  // Drug reduces BP
        graph3.add_edge("blood_pressure", "heart_rate", 0.3);   // BP affects HR

        benchmarks.push(CausalBenchmark {
            id: "intervention_1_drug".to_string(),
            description: "What happens to blood pressure if we administer 100mg of the drug?".to_string(),
            category: CausalCategory::InterventionPrediction,
            ground_truth_graph: graph3,
            observations: generate_chain_data("drug_dose", "blood_pressure", "heart_rate", 100),
            queries: vec![
                CausalQuery::Intervention {
                    variable: "drug_dose".to_string(),
                    value: 100.0,
                    target: "blood_pressure".to_string(),
                },
            ],
            expected_answers: vec![
                CausalAnswer::Range {
                    low: -80.0,
                    high: -60.0,
                    expected: -70.0,  // 100 * -0.7
                },
            ],
            difficulty: 3,
        });

        // Benchmark 2.2: Marketing Intervention
        let mut graph4 = CausalGraph::new();
        graph4.add_variable("ad_spend");
        graph4.add_variable("brand_awareness");
        graph4.add_variable("sales");
        graph4.add_variable("economic_conditions");
        graph4.add_edge("ad_spend", "brand_awareness", 0.5);
        graph4.add_edge("brand_awareness", "sales", 0.6);
        graph4.add_edge("economic_conditions", "sales", 0.4);
        graph4.add_edge("economic_conditions", "ad_spend", 0.3);  // Companies spend more in good times

        benchmarks.push(CausalBenchmark {
            id: "intervention_2_marketing".to_string(),
            description: "If we double ad spend, what is the TRUE effect on sales (controlling for economy)?".to_string(),
            category: CausalCategory::InterventionPrediction,
            ground_truth_graph: graph4,
            observations: vec![],  // Would be generated
            queries: vec![
                CausalQuery::CausalEffect {
                    from: "ad_spend".to_string(),
                    to: "sales".to_string(),
                },
            ],
            expected_answers: vec![
                CausalAnswer::Range {
                    low: 0.25,
                    high: 0.35,
                    expected: 0.30,  // 0.5 * 0.6 = 0.30 (mediated through awareness)
                },
            ],
            difficulty: 4,
        });

        // ================================================================
        // Category 3: Counterfactual Reasoning
        // ================================================================

        // Benchmark 3.1: Would the patient have survived?
        let mut graph5 = CausalGraph::new();
        graph5.add_variable("treatment");
        graph5.add_variable("severity");
        graph5.add_variable("survival");
        graph5.add_edge("treatment", "survival", 0.4);
        graph5.add_edge("severity", "survival", -0.6);
        graph5.add_edge("severity", "treatment", 0.3);  // More severe → more likely to get treatment

        benchmarks.push(CausalBenchmark {
            id: "counterfactual_1_survival".to_string(),
            description: "Patient received treatment and survived. Would they have survived WITHOUT treatment?".to_string(),
            category: CausalCategory::CounterfactualReasoning,
            ground_truth_graph: graph5,
            observations: vec![],
            queries: vec![
                CausalQuery::Counterfactual {
                    variable: "treatment".to_string(),
                    value: 0.0,  // No treatment
                    target: "survival".to_string(),
                    actual_outcome: 1.0,  // Actually survived
                },
            ],
            expected_answers: vec![
                CausalAnswer::Range {
                    low: 0.3,
                    high: 0.7,
                    expected: 0.5,  // Uncertain - depends on severity
                },
            ],
            difficulty: 4,
        });

        // ================================================================
        // Category 4: Causal Discovery
        // ================================================================

        // Benchmark 4.1: Discover simple chain
        let mut graph6 = CausalGraph::new();
        graph6.add_edge("A", "B", 0.8);
        graph6.add_edge("B", "C", 0.7);

        benchmarks.push(CausalBenchmark {
            id: "discovery_1_chain".to_string(),
            description: "Given data, discover the causal chain A → B → C".to_string(),
            category: CausalCategory::CausalDiscovery,
            ground_truth_graph: graph6.clone(),
            observations: generate_chain_data("A", "B", "C", 200),
            queries: vec![CausalQuery::DiscoverGraph],
            expected_answers: vec![CausalAnswer::Graph(graph6)],
            difficulty: 3,
        });

        // Benchmark 4.2: Discover with confounder
        let mut graph7 = CausalGraph::new();
        graph7.add_edge("U", "X", 0.7);
        graph7.add_edge("U", "Y", 0.6);
        graph7.add_edge("X", "Y", 0.3);  // Small direct effect
        graph7.add_confounder("U", vec!["X", "Y"]);

        benchmarks.push(CausalBenchmark {
            id: "discovery_2_confounder".to_string(),
            description: "Discover that U confounds X and Y, with small X→Y effect".to_string(),
            category: CausalCategory::CausalDiscovery,
            ground_truth_graph: graph7.clone(),
            observations: generate_confounder_data_with_direct("U", "X", "Y", 200),
            queries: vec![CausalQuery::DiscoverGraph],
            expected_answers: vec![CausalAnswer::Graph(graph7)],
            difficulty: 5,
        });

        // ================================================================
        // Category 5: Temporal Causation
        // ================================================================

        // Benchmark 5.1: Granger-style causation
        let mut graph8 = CausalGraph::new();
        graph8.add_edge("X_t-1", "Y_t", 0.6);  // X yesterday causes Y today

        benchmarks.push(CausalBenchmark {
            id: "temporal_1_granger".to_string(),
            description: "X at time t-1 causes Y at time t (time-lagged causation)".to_string(),
            category: CausalCategory::TemporalCausation,
            ground_truth_graph: graph8,
            observations: generate_temporal_data(100, 1),
            queries: vec![
                CausalQuery::DoesCause {
                    from: "X_t-1".to_string(),
                    to: "Y_t".to_string(),
                },
            ],
            expected_answers: vec![CausalAnswer::Boolean(true)],
            difficulty: 3,
        });

        // ================================================================
        // Category 6: Confounding Control
        // ================================================================

        // Benchmark 6.1: Smoking, Tar, and Cancer (classic example)
        // Naive correlation shows smoking→cancer, but we need to control for tar
        let mut graph9 = CausalGraph::new();
        graph9.add_variable("smoking");
        graph9.add_variable("tar_deposits");
        graph9.add_variable("lung_cancer");
        graph9.add_edge("smoking", "tar_deposits", 0.9);
        graph9.add_edge("tar_deposits", "lung_cancer", 0.8);
        // smoking → tar → cancer (mediated effect)

        benchmarks.push(CausalBenchmark {
            id: "confounding_1_smoking".to_string(),
            description: "What is the causal effect of smoking on lung cancer, controlling for tar?".to_string(),
            category: CausalCategory::ConfoundingControl,
            ground_truth_graph: graph9,
            observations: generate_chain_data("smoking", "tar_deposits", "lung_cancer", 200),
            queries: vec![
                CausalQuery::AdjustedEffect {
                    treatment: "smoking".to_string(),
                    outcome: "lung_cancer".to_string(),
                    confounders: vec!["tar_deposits".to_string()],
                },
            ],
            expected_answers: vec![
                CausalAnswer::Range {
                    low: 0.6,
                    high: 0.8,
                    expected: 0.72,  // 0.9 * 0.8 = 0.72 (total mediated effect)
                },
            ],
            difficulty: 3,
        });

        // Benchmark 6.2: Treatment effect with selection bias
        // Patients with worse health are more likely to get treatment
        let mut graph10 = CausalGraph::new();
        graph10.add_variable("health_status");
        graph10.add_variable("treatment");
        graph10.add_variable("recovery");
        graph10.add_edge("health_status", "treatment", -0.5);  // Sicker → more treatment
        graph10.add_edge("health_status", "recovery", 0.6);     // Healthier → better recovery
        graph10.add_edge("treatment", "recovery", 0.4);         // Treatment helps
        graph10.add_confounder("health_status", vec!["treatment", "recovery"]);

        benchmarks.push(CausalBenchmark {
            id: "confounding_2_treatment".to_string(),
            description: "What is the true treatment effect, controlling for health status?".to_string(),
            category: CausalCategory::ConfoundingControl,
            ground_truth_graph: graph10,
            observations: vec![],
            queries: vec![
                CausalQuery::AverageTreatmentEffect {
                    treatment: "treatment".to_string(),
                    outcome: "recovery".to_string(),
                },
            ],
            expected_answers: vec![
                CausalAnswer::Range {
                    low: 0.3,
                    high: 0.5,
                    expected: 0.4,  // True direct effect
                },
            ],
            difficulty: 4,
        });

        // ================================================================
        // Category 7: Negative Causation
        // ================================================================

        // Benchmark 7.1: Vaccine prevents disease
        let mut graph11 = CausalGraph::new();
        graph11.add_variable("vaccination");
        graph11.add_variable("disease_risk");
        graph11.add_edge("vaccination", "disease_risk", -0.7);  // Negative effect

        benchmarks.push(CausalBenchmark {
            id: "negative_1_vaccine".to_string(),
            description: "Does vaccination prevent (reduce) disease?".to_string(),
            category: CausalCategory::NegativeCausation,
            ground_truth_graph: graph11,
            observations: generate_negative_effect_data("vaccination", "disease_risk", 100),
            queries: vec![
                CausalQuery::DoesPrevent {
                    from: "vaccination".to_string(),
                    to: "disease_risk".to_string(),
                },
            ],
            expected_answers: vec![CausalAnswer::Boolean(true)],
            difficulty: 2,
        });

        // Benchmark 7.2: Exercise prevents heart disease
        let mut graph12 = CausalGraph::new();
        graph12.add_variable("exercise");
        graph12.add_variable("cardiovascular_health");
        graph12.add_variable("heart_disease");
        graph12.add_edge("exercise", "cardiovascular_health", 0.6);
        graph12.add_edge("cardiovascular_health", "heart_disease", -0.8);

        benchmarks.push(CausalBenchmark {
            id: "negative_2_exercise".to_string(),
            description: "Does exercise prevent heart disease (via cardiovascular health)?".to_string(),
            category: CausalCategory::NegativeCausation,
            ground_truth_graph: graph12,
            observations: vec![],
            queries: vec![
                CausalQuery::DoesPrevent {
                    from: "exercise".to_string(),
                    to: "heart_disease".to_string(),
                },
                CausalQuery::CausalEffect {
                    from: "exercise".to_string(),
                    to: "heart_disease".to_string(),
                },
            ],
            expected_answers: vec![
                CausalAnswer::Boolean(true),  // Yes, prevents
                CausalAnswer::Range {
                    low: -0.6,
                    high: -0.4,
                    expected: -0.48,  // 0.6 * -0.8 = -0.48
                },
            ],
            difficulty: 3,
        });

        // ================================================================
        // HARD MODE: Complex Multi-Variable Graphs
        // ================================================================

        // Benchmark 8.1: Simpson's Paradox
        // Overall correlation is opposite of causal effect
        let mut graph_simpson = CausalGraph::new();
        graph_simpson.add_variable("treatment");
        graph_simpson.add_variable("severity");
        graph_simpson.add_variable("outcome");
        graph_simpson.add_edge("severity", "treatment", 0.8);   // Sicker patients get treatment
        graph_simpson.add_edge("severity", "outcome", -0.9);    // Sicker = worse outcome
        graph_simpson.add_edge("treatment", "outcome", 0.5);    // Treatment helps!
        graph_simpson.add_confounder("severity", vec!["treatment", "outcome"]);

        benchmarks.push(CausalBenchmark {
            id: "hard_1_simpson".to_string(),
            description: "Simpson's Paradox: Treatment helps but looks harmful due to confounding".to_string(),
            category: CausalCategory::ConfoundingControl,
            ground_truth_graph: graph_simpson,
            observations: vec![],
            queries: vec![
                CausalQuery::CausalEffect {
                    from: "treatment".to_string(),
                    to: "outcome".to_string(),
                },
            ],
            expected_answers: vec![
                CausalAnswer::Range {
                    low: 0.4,
                    high: 0.6,
                    expected: 0.5,  // True positive effect despite negative correlation
                },
            ],
            difficulty: 5,
        });

        // Benchmark 8.2: Instrumental Variable scenario
        // Z → X → Y with U confounding X-Y
        let mut graph_iv = CausalGraph::new();
        graph_iv.add_variable("instrument");
        graph_iv.add_variable("treatment");
        graph_iv.add_variable("confounder");
        graph_iv.add_variable("outcome");
        graph_iv.add_edge("instrument", "treatment", 0.7);
        graph_iv.add_edge("treatment", "outcome", 0.6);
        graph_iv.add_edge("confounder", "treatment", 0.5);
        graph_iv.add_edge("confounder", "outcome", 0.4);
        graph_iv.add_confounder("confounder", vec!["treatment", "outcome"]);

        benchmarks.push(CausalBenchmark {
            id: "hard_2_instrumental".to_string(),
            description: "Can we identify causal effect using instrumental variable?".to_string(),
            category: CausalCategory::ConfoundingControl,
            ground_truth_graph: graph_iv,
            observations: vec![],
            queries: vec![
                CausalQuery::CausalEffect {
                    from: "treatment".to_string(),
                    to: "outcome".to_string(),
                },
            ],
            expected_answers: vec![
                CausalAnswer::Range {
                    low: 0.5,
                    high: 0.7,
                    expected: 0.6,
                },
            ],
            difficulty: 5,
        });

        // Benchmark 8.3: Collider bias (conditioning on collider creates spurious correlation)
        let mut graph_collider = CausalGraph::new();
        graph_collider.add_variable("talent");
        graph_collider.add_variable("looks");
        graph_collider.add_variable("hollywood");  // Collider
        graph_collider.add_edge("talent", "hollywood", 0.6);
        graph_collider.add_edge("looks", "hollywood", 0.6);
        // No edge between talent and looks!

        benchmarks.push(CausalBenchmark {
            id: "hard_3_collider".to_string(),
            description: "Collider bias: Talent and looks seem negatively correlated among actors".to_string(),
            category: CausalCategory::CorrelationVsCausation,
            ground_truth_graph: graph_collider,
            observations: vec![],
            queries: vec![
                CausalQuery::DoesCause {
                    from: "talent".to_string(),
                    to: "looks".to_string(),
                },
                CausalQuery::SpuriousCorrelation {
                    var1: "talent".to_string(),
                    var2: "looks".to_string(),
                },
            ],
            expected_answers: vec![
                CausalAnswer::Boolean(false),  // Talent does NOT cause looks
                CausalAnswer::Boolean(true),   // Any correlation IS spurious
            ],
            difficulty: 4,
        });

        // Benchmark 8.4: Multi-hop counterfactual
        let mut graph_multihop = CausalGraph::new();
        graph_multihop.add_variable("A");
        graph_multihop.add_variable("B");
        graph_multihop.add_variable("C");
        graph_multihop.add_variable("D");
        graph_multihop.add_edge("A", "B", 0.8);
        graph_multihop.add_edge("B", "C", 0.7);
        graph_multihop.add_edge("C", "D", 0.6);

        benchmarks.push(CausalBenchmark {
            id: "hard_4_multihop".to_string(),
            description: "4-node chain: What is effect of A on D?".to_string(),
            category: CausalCategory::InterventionPrediction,
            ground_truth_graph: graph_multihop,
            observations: vec![],
            queries: vec![
                CausalQuery::CausalEffect {
                    from: "A".to_string(),
                    to: "D".to_string(),
                },
            ],
            expected_answers: vec![
                CausalAnswer::Range {
                    low: 0.25,
                    high: 0.40,
                    expected: 0.336,  // 0.8 * 0.7 * 0.6 = 0.336
                },
            ],
            difficulty: 4,
        });

        Self { benchmarks }
    }

    /// Run all benchmarks and return results
    pub fn run<F>(&self, mut solver: F) -> BenchmarkResults
    where
        F: FnMut(&CausalBenchmark, &CausalQuery) -> CausalAnswer,
    {
        let mut results = BenchmarkResults::new();

        for benchmark in &self.benchmarks {
            let mut correct = 0;
            let mut total = 0;

            for (query, expected) in benchmark.queries.iter().zip(&benchmark.expected_answers) {
                let answer = solver(benchmark, query);
                let is_correct = compare_answers(&answer, expected);

                if is_correct {
                    correct += 1;
                }
                total += 1;

                results.add_result(BenchmarkResult {
                    benchmark_id: benchmark.id.clone(),
                    category: benchmark.category,
                    query: format!("{:?}", query),
                    expected: format!("{:?}", expected),
                    actual: format!("{:?}", answer),
                    correct: is_correct,
                    difficulty: benchmark.difficulty,
                });
            }

            results.by_category
                .entry(benchmark.category)
                .or_insert((0, 0));
            let entry = results.by_category.get_mut(&benchmark.category).unwrap();
            entry.0 += correct;
            entry.1 += total;
        }

        results
    }

    /// Get all benchmarks
    pub fn benchmarks(&self) -> &[CausalBenchmark] {
        &self.benchmarks
    }

    /// Get benchmarks by category
    pub fn by_category(&self, category: CausalCategory) -> Vec<&CausalBenchmark> {
        self.benchmarks.iter().filter(|b| b.category == category).collect()
    }
}

/// Results of running benchmarks
#[derive(Debug)]
pub struct BenchmarkResults {
    pub results: Vec<BenchmarkResult>,
    pub by_category: HashMap<CausalCategory, (usize, usize)>,  // (correct, total)
}

impl BenchmarkResults {
    fn new() -> Self {
        Self {
            results: Vec::new(),
            by_category: HashMap::new(),
        }
    }

    fn add_result(&mut self, result: BenchmarkResult) {
        self.results.push(result);
    }

    /// Overall accuracy
    pub fn accuracy(&self) -> f32 {
        let correct = self.results.iter().filter(|r| r.correct).count();
        correct as f32 / self.results.len() as f32
    }

    /// Accuracy by category
    pub fn accuracy_by_category(&self, category: CausalCategory) -> f32 {
        if let Some((correct, total)) = self.by_category.get(&category) {
            *correct as f32 / *total as f32
        } else {
            0.0
        }
    }

    /// Summary report
    pub fn summary(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Causal Reasoning Benchmark Results ===\n\n");
        report.push_str(&format!("Overall Accuracy: {:.1}%\n\n", self.accuracy() * 100.0));

        report.push_str("By Category:\n");
        for (category, (correct, total)) in &self.by_category {
            let acc = *correct as f32 / *total as f32 * 100.0;
            report.push_str(&format!("  {:?}: {}/{} ({:.1}%)\n", category, correct, total, acc));
        }

        report.push_str("\nDetailed Results:\n");
        for result in &self.results {
            let status = if result.correct { "PASS" } else { "FAIL" };
            report.push_str(&format!(
                "  [{}] {} (difficulty {})\n",
                status, result.benchmark_id, result.difficulty
            ));
        }

        report
    }
}

/// Single benchmark result
#[derive(Debug)]
pub struct BenchmarkResult {
    pub benchmark_id: String,
    pub category: CausalCategory,
    pub query: String,
    pub expected: String,
    pub actual: String,
    pub correct: bool,
    pub difficulty: u8,
}

/// Compare two causal answers
fn compare_answers(actual: &CausalAnswer, expected: &CausalAnswer) -> bool {
    match (actual, expected) {
        (CausalAnswer::Boolean(a), CausalAnswer::Boolean(e)) => a == e,

        (CausalAnswer::Numeric(a), CausalAnswer::Numeric(e)) => (a - e).abs() < 0.1,

        (CausalAnswer::Numeric(a), CausalAnswer::Range { low, high, .. }) => {
            *a >= *low && *a <= *high
        }

        (CausalAnswer::Range { expected: a, .. }, CausalAnswer::Range { expected: e, .. }) => {
            (a - e).abs() < 0.2
        }

        (CausalAnswer::Variables(a), CausalAnswer::Variables(e)) => {
            a.iter().all(|v| e.contains(v)) && e.iter().all(|v| a.contains(v))
        }

        (CausalAnswer::Graph(actual_graph), CausalAnswer::Graph(expected_graph)) => {
            compare_graphs(actual_graph, expected_graph)
        }

        _ => false,
    }
}

/// Compare two causal graphs for structural similarity
/// We check:
/// 1. Same edges (direction matters, but strength is approximate)
/// 2. Allow for undiscovered confounders (discovery is hard)
fn compare_graphs(actual: &CausalGraph, expected: &CausalGraph) -> bool {
    // Build edge sets (ignoring strength for now)
    let actual_edges: std::collections::HashSet<(String, String)> = actual
        .edges
        .iter()
        .map(|(from, to, _)| (from.clone(), to.clone()))
        .collect();

    let expected_edges: std::collections::HashSet<(String, String)> = expected
        .edges
        .iter()
        .map(|(from, to, _)| (from.clone(), to.clone()))
        .collect();

    // Calculate edge overlap using Jaccard similarity
    let intersection = actual_edges.intersection(&expected_edges).count();
    let union = actual_edges.union(&expected_edges).count();

    if union == 0 {
        return actual_edges.is_empty() && expected_edges.is_empty();
    }

    let jaccard = intersection as f32 / union as f32;

    // Consider graphs matching if they share at least 70% of edges
    // (accounts for minor discovery errors)
    jaccard >= 0.7
}

// ================================================================
// Data Generation Helpers
// ================================================================

fn generate_confounded_data(
    confounder: &str,
    var1: &str,
    var2: &str,
    n: usize,
) -> Vec<Observation> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    (0..n)
        .map(|_| {
            let c = rng.gen_range(0.0..100.0);
            let v1 = c * 0.8 + rng.gen_range(-10.0..10.0);
            let v2 = c * 0.6 + rng.gen_range(-10.0..10.0);

            let mut values = HashMap::new();
            values.insert(confounder.to_string(), c);
            values.insert(var1.to_string(), v1);
            values.insert(var2.to_string(), v2);

            Observation {
                values,
                timestamp: None,
            }
        })
        .collect()
}

/// Generate data with both confounding AND a direct effect
/// Graph: U → X, U → Y, X → Y
fn generate_confounder_data_with_direct(
    confounder: &str,
    var1: &str,
    var2: &str,
    n: usize,
) -> Vec<Observation> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    (0..n)
        .map(|_| {
            // U is the confounder
            let u = rng.gen_range(0.0..100.0);
            // X = 0.7 * U + noise
            let x = u * 0.7 + rng.gen_range(-10.0..10.0);
            // Y = 0.6 * U + 0.3 * X + noise (both confounding and direct effect)
            let y = u * 0.6 + x * 0.3 + rng.gen_range(-10.0..10.0);

            let mut values = HashMap::new();
            values.insert(confounder.to_string(), u);
            values.insert(var1.to_string(), x);
            values.insert(var2.to_string(), y);

            Observation {
                values,
                timestamp: None,
            }
        })
        .collect()
}

fn generate_chain_data(var1: &str, var2: &str, var3: &str, n: usize) -> Vec<Observation> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    (0..n)
        .map(|_| {
            let v1 = rng.gen_range(0.0..100.0);
            let v2 = v1 * 0.8 + rng.gen_range(-10.0..10.0);
            let v3 = v2 * 0.7 + rng.gen_range(-10.0..10.0);

            let mut values = HashMap::new();
            values.insert(var1.to_string(), v1);
            values.insert(var2.to_string(), v2);
            values.insert(var3.to_string(), v3);

            Observation {
                values,
                timestamp: None,
            }
        })
        .collect()
}

fn generate_temporal_data(n: usize, lag: usize) -> Vec<Observation> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut observations = Vec::with_capacity(n);
    let mut x_history = vec![0.0f32; lag + 1];

    for t in 0..n {
        // X is random
        let x = rng.gen_range(0.0..100.0);
        x_history.push(x);
        if x_history.len() > lag + 1 {
            x_history.remove(0);
        }

        // Y depends on X from `lag` steps ago
        let y = if t >= lag {
            x_history[0] * 0.6 + rng.gen_range(-10.0..10.0)
        } else {
            rng.gen_range(0.0..100.0)
        };

        let mut values = HashMap::new();
        values.insert("X_t".to_string(), x);
        values.insert("Y_t".to_string(), y);
        if t >= lag {
            values.insert("X_t-1".to_string(), x_history[0]);
        }

        observations.push(Observation {
            values,
            timestamp: Some(t as f64),
        });
    }

    observations
}

/// Generate data with a negative causal effect (X prevents/reduces Y)
fn generate_negative_effect_data(cause: &str, effect: &str, n: usize) -> Vec<Observation> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    (0..n)
        .map(|_| {
            let x: f32 = rng.gen_range(0.0_f32..100.0_f32);
            // Negative relationship: higher X → lower Y
            let y: f32 = 100.0 - x * 0.7 + rng.gen_range(-10.0_f32..10.0_f32);

            let mut values = HashMap::new();
            values.insert(cause.to_string(), x);
            values.insert(effect.to_string(), y.max(0.0_f32)); // Ensure non-negative

            Observation {
                values,
                timestamp: None,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_graph_causes() {
        let mut graph = CausalGraph::new();
        graph.add_edge("A", "B", 0.8);
        graph.add_edge("B", "C", 0.7);

        assert!(graph.causes("A", "B"));
        assert!(graph.causes("A", "C"));  // Indirect
        assert!(graph.causes("B", "C"));
        assert!(!graph.causes("C", "A"));
        assert!(!graph.causes("C", "B"));
    }

    #[test]
    fn test_benchmark_suite_creation() {
        let suite = CausalBenchmarkSuite::standard();

        assert!(!suite.benchmarks().is_empty());

        // Check we have all categories
        let categories: Vec<_> = suite.benchmarks().iter().map(|b| b.category).collect();
        assert!(categories.contains(&CausalCategory::CorrelationVsCausation));
        assert!(categories.contains(&CausalCategory::InterventionPrediction));
    }

    #[test]
    fn test_run_with_dummy_solver() {
        let suite = CausalBenchmarkSuite::standard();

        // Dummy solver that always returns false
        let results = suite.run(|_benchmark, query| {
            match query {
                CausalQuery::DoesCause { .. } => CausalAnswer::Boolean(false),
                CausalQuery::SpuriousCorrelation { .. } => CausalAnswer::Boolean(true),
                _ => CausalAnswer::Boolean(false),
            }
        });

        println!("{}", results.summary());

        // Should get some answers right (the spurious correlation ones)
        assert!(results.accuracy() > 0.0);
    }

    #[test]
    fn test_data_generation() {
        let data = generate_confounded_data("temp", "ice", "drown", 50);
        assert_eq!(data.len(), 50);
        assert!(data[0].values.contains_key("temp"));
        assert!(data[0].values.contains_key("ice"));
        assert!(data[0].values.contains_key("drown"));
    }
}
