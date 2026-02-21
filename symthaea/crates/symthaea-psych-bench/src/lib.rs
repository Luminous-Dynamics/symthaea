//! # Symthaea Psychological Benchmark Suite
//!
//! Standardized evaluation of Symthaea's neuroscience-inspired subsystems
//! against established psychological benchmarks and published human baselines.
//!
//! ## Benchmarks (35 across 9 domains)
//!
//! - **WorM** — Working memory tasks (N-back, change detection, serial recall,
//!   spatial updating, binding, digit span)
//! - **CogBench** — Cognitive psychology experiments via FEP active inference
//!   (probabilistic reasoning, exploration, bandits, instrumental learning,
//!   two-step task, temporal discounting, BART, reversal learning)
//! - **Executive** — Executive function (WCST rule-shifting, Iowa Gambling Task
//!   loss aversion, Raven's Progressive Matrices, Stroop interference, Flanker,
//!   Tower of London planning)
//! - **Metacognition** — Metacognitive calibration (confidence-accuracy covariance)
//! - **Butlin** — Consciousness indicators from 6 theories (RPT, GWT, HOT, PP, AST, IIT)
//! - **ToMBench** — Theory of Mind tasks (false belief, faux-pas, persuasion,
//!   strange story, hinting)
//! - **MemoryAgentBench** — Full memory pipeline competencies (retrieval,
//!   test-time learning, long-range, conflict resolution)
//! - **Affect** — Affective processing (valence classification, mood-congruent recall)
//! - **Creativity** — Creative cognition (Remote Associates Test, Alternate Uses Task)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea_psych_bench::harness::{PsychBenchmark, BenchmarkConfig};
//! use symthaea_psych_bench::benchmarks::worm::NBackBenchmark;
//!
//! let config = BenchmarkConfig::default();
//! let bench = NBackBenchmark;
//! let result = bench.run(&config);
//! println!("{}", result.summary());
//! ```

pub mod adapter;
pub mod benchmarks;
pub mod harness;
pub mod wm;
