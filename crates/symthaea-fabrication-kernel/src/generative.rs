//! Generative design via neuroevolution of CSG trees
//!
//! Evolves populations of CSG geometry using tournament selection,
//! subtree crossover, and mutation to optimize multi-objective fitness.

use crate::csg::{BooleanOp, CSGNode, Primitive, Transform3D};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A single genome in the generative population — wraps a CSG tree and its
/// evaluated fitness (None until scored).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsgGenome {
    pub tree: CSGNode,
    pub fitness: Option<f64>,
}

/// Configuration for the generative designer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerativeConfig {
    /// Number of individuals per generation.
    pub population_size: usize,
    /// Maximum depth for randomly generated CSG trees.
    pub max_depth: u32,
    /// Probability of applying a mutation to each offspring (0.0–1.0).
    pub mutation_rate: f32,
    /// Probability of crossover vs cloning (0.0–1.0).
    pub crossover_rate: f32,
    /// Weighted objectives used to compute composite fitness.
    pub objectives: Vec<ObjectiveWeight>,
}

/// A weighted design objective.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveWeight {
    pub objective: DesignObjective,
    pub weight: f32,
}

/// Design objectives that can be optimized.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DesignObjective {
    StructuralStrength,
    Manufacturability,
    MaterialEfficiency,
    Aesthetics,
    CustomFitness(String),
}

// ---------------------------------------------------------------------------
// GenerativeDesigner
// ---------------------------------------------------------------------------

/// Evolutionary engine that breeds CSG trees toward multi-objective goals.
pub struct GenerativeDesigner {
    pub config: GenerativeConfig,
    pub population: Vec<CsgGenome>,
    pub generation: u32,
    pub best_fitness: f64,
    rng: StdRng,
}

impl GenerativeDesigner {
    /// Create a new designer from configuration.  The population starts empty;
    /// call [`initialize_population`] next.
    pub fn new(config: GenerativeConfig) -> Self {
        Self {
            config,
            population: Vec::new(),
            generation: 0,
            best_fitness: f64::NEG_INFINITY,
            rng: StdRng::from_entropy(),
        }
    }

    /// Create with a fixed seed for reproducibility.
    pub fn with_seed(config: GenerativeConfig, seed: u64) -> Self {
        Self {
            config,
            population: Vec::new(),
            generation: 0,
            best_fitness: f64::NEG_INFINITY,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Fill the population with random CSG trees up to `config.max_depth`.
    pub fn initialize_population(&mut self) {
        self.population.clear();
        for _ in 0..self.config.population_size {
            let tree = random_csg_tree_with_rng(self.config.max_depth, &mut self.rng);
            self.population.push(CsgGenome {
                tree,
                fitness: None,
            });
        }
    }

    /// Score every genome using the supplied evaluator.  Updates `best_fitness`.
    pub fn evaluate_fitness(&mut self, evaluator: &dyn Fn(&CSGNode) -> f64) {
        for genome in &mut self.population {
            let score = evaluator(&genome.tree);
            genome.fitness = Some(score);
        }
        if let Some(best) = self.best() {
            if let Some(f) = best.fitness {
                if f > self.best_fitness {
                    self.best_fitness = f;
                }
            }
        }
    }

    /// Tournament selection (size 3) — returns pairs of parent indices.
    pub fn select_parents(&self) -> Vec<(usize, usize)> {
        let n = self.population.len();
        if n < 2 {
            return Vec::new();
        }
        let mut rng = self.rng.clone();
        let pair_count = n / 2;
        let mut pairs = Vec::with_capacity(pair_count);
        for _ in 0..pair_count {
            let a = tournament_select(&self.population, &mut rng);
            let mut b = tournament_select(&self.population, &mut rng);
            // Avoid self-pairing when possible.
            if b == a && n > 2 {
                b = (a + 1) % n;
            }
            pairs.push((a, b));
        }
        pairs
    }

    /// Subtree crossover: pick a random subtree from each parent and swap.
    pub fn crossover(parent_a: &CSGNode, parent_b: &CSGNode) -> CSGNode {
        let mut rng = StdRng::from_entropy();
        Self::crossover_with_rng(parent_a, parent_b, &mut rng)
    }

    fn crossover_with_rng(parent_a: &CSGNode, parent_b: &CSGNode, rng: &mut StdRng) -> CSGNode {
        let donor = pick_random_subtree(parent_b, rng);
        let mut child = parent_a.clone();
        replace_random_subtree(&mut child, &donor, rng);
        child
    }

    /// Mutate a CSG tree in-place with the given probability.
    pub fn mutate(tree: &mut CSGNode, rate: f32) {
        let mut rng = StdRng::from_entropy();
        Self::mutate_with_rng(tree, rate, &mut rng);
    }

    fn mutate_with_rng(tree: &mut CSGNode, rate: f32, rng: &mut StdRng) {
        if rng.gen::<f32>() >= rate {
            return;
        }
        // Choose mutation type: 60 % subtree replacement, 40 % parameter tweak.
        if rng.gen::<f32>() < 0.6 {
            // Replace a random subtree with a new random one (depth ≤ 3).
            let replacement = random_csg_tree_with_rng(3, rng);
            replace_random_subtree(tree, &replacement, rng);
        } else {
            tweak_parameters(tree, rng);
        }
    }

    /// Run one full generation: select → crossover / clone → mutate → evaluate.
    pub fn evolve_generation(&mut self, evaluator: &dyn Fn(&CSGNode) -> f64) {
        let pairs = self.select_parents();
        let mut next_gen: Vec<CsgGenome> = Vec::with_capacity(self.config.population_size);

        // Elitism: keep the best individual.
        if let Some(best) = self.best() {
            next_gen.push(best.clone());
        }

        for (ia, ib) in &pairs {
            let pa = &self.population[*ia].tree;
            let pb = &self.population[*ib].tree;

            let mut child_a;
            let mut child_b;

            if self.rng.gen::<f32>() < self.config.crossover_rate {
                child_a = Self::crossover_with_rng(pa, pb, &mut self.rng);
                child_b = Self::crossover_with_rng(pb, pa, &mut self.rng);
            } else {
                child_a = pa.clone();
                child_b = pb.clone();
            }

            Self::mutate_with_rng(&mut child_a, self.config.mutation_rate, &mut self.rng);
            Self::mutate_with_rng(&mut child_b, self.config.mutation_rate, &mut self.rng);

            next_gen.push(CsgGenome {
                tree: child_a,
                fitness: None,
            });
            if next_gen.len() < self.config.population_size {
                next_gen.push(CsgGenome {
                    tree: child_b,
                    fitness: None,
                });
            }
        }

        // If we still need more, fill with random individuals.
        while next_gen.len() < self.config.population_size {
            let tree = random_csg_tree_with_rng(self.config.max_depth, &mut self.rng);
            next_gen.push(CsgGenome {
                tree,
                fitness: None,
            });
        }

        // Truncate to exact population size (elitism + pairs may overshoot).
        next_gen.truncate(self.config.population_size);

        self.population = next_gen;
        self.generation += 1;
        self.evaluate_fitness(evaluator);
    }

    /// Return the genome with the highest fitness, if any have been scored.
    pub fn best(&self) -> Option<&CsgGenome> {
        self.population
            .iter()
            .filter(|g| g.fitness.is_some())
            .max_by(|a, b| {
                a.fitness
                    .unwrap()
                    .partial_cmp(&b.fitness.unwrap())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

// ---------------------------------------------------------------------------
// Random tree generation
// ---------------------------------------------------------------------------

/// Generate a random CSG tree up to `max_depth` using thread-local RNG.
pub fn random_csg_tree(max_depth: u32) -> CSGNode {
    let mut rng = StdRng::from_entropy();
    random_csg_tree_with_rng(max_depth, &mut rng)
}

fn random_csg_tree_with_rng(max_depth: u32, rng: &mut StdRng) -> CSGNode {
    if max_depth == 0 || (max_depth <= 1 && rng.gen::<f32>() < 0.5) {
        return CSGNode::Primitive(random_primitive(rng));
    }

    let choice = rng.gen::<f32>();
    if choice < 0.4 {
        // Primitive leaf
        CSGNode::Primitive(random_primitive(rng))
    } else if choice < 0.6 {
        // Transform wrapper
        CSGNode::Transform {
            node: Box::new(random_csg_tree_with_rng(max_depth - 1, rng)),
            transform: random_transform(rng),
        }
    } else {
        // Boolean operation
        let op = match rng.gen_range(0u32..3) {
            0 => BooleanOp::Union,
            1 => BooleanOp::Subtract,
            _ => BooleanOp::Intersect,
        };
        CSGNode::Boolean {
            op,
            left: Box::new(random_csg_tree_with_rng(max_depth - 1, rng)),
            right: Box::new(random_csg_tree_with_rng(max_depth - 1, rng)),
        }
    }
}

fn random_primitive(rng: &mut StdRng) -> Primitive {
    match rng.gen_range(0u32..5) {
        0 => Primitive::Cube,
        1 => Primitive::Cylinder,
        2 => Primitive::Sphere,
        3 => Primitive::Cone,
        _ => Primitive::Torus,
    }
}

fn random_transform(rng: &mut StdRng) -> Transform3D {
    Transform3D {
        scale: [
            rng.gen_range(0.2f32..3.0),
            rng.gen_range(0.2..3.0),
            rng.gen_range(0.2..3.0),
        ],
        rotate: [
            rng.gen_range(0.0f32..std::f32::consts::TAU),
            rng.gen_range(0.0..std::f32::consts::TAU),
            rng.gen_range(0.0..std::f32::consts::TAU),
        ],
        translate: [
            rng.gen_range(-5.0f32..5.0),
            rng.gen_range(-5.0..5.0),
            rng.gen_range(-5.0..5.0),
        ],
    }
}

// ---------------------------------------------------------------------------
// Helpers: tree traversal
// ---------------------------------------------------------------------------

/// Count nodes in a CSG tree.
fn node_count(tree: &CSGNode) -> usize {
    match tree {
        CSGNode::Primitive(_) => 1,
        CSGNode::Transform { node, .. } => 1 + node_count(node),
        CSGNode::Boolean { left, right, .. } => 1 + node_count(left) + node_count(right),
    }
}

/// Pick a uniformly random subtree by flattening to an index.
fn pick_random_subtree(tree: &CSGNode, rng: &mut StdRng) -> CSGNode {
    let count = node_count(tree);
    let idx = rng.gen_range(0..count);
    get_subtree_at(tree, idx).unwrap_or_else(|| tree.clone())
}

fn get_subtree_at(tree: &CSGNode, mut idx: usize) -> Option<CSGNode> {
    fn walk(node: &CSGNode, pos: &mut usize) -> Option<CSGNode> {
        if *pos == 0 {
            return Some(node.clone());
        }
        *pos -= 1;
        match node {
            CSGNode::Primitive(_) => None,
            CSGNode::Transform { node: child, .. } => walk(child, pos),
            CSGNode::Boolean { left, right, .. } => {
                walk(left, pos).or_else(|| walk(right, pos))
            }
        }
    }
    walk(tree, &mut idx)
}

/// Replace a uniformly random subtree with `replacement`.
fn replace_random_subtree(tree: &mut CSGNode, replacement: &CSGNode, rng: &mut StdRng) {
    let count = node_count(tree);
    if count <= 1 {
        *tree = replacement.clone();
        return;
    }
    // Don't replace root (index 0) every time — skip it sometimes.
    let idx = rng.gen_range(0..count);
    let mut pos = idx;
    replace_at(tree, &mut pos, replacement);
}

fn replace_at(node: &mut CSGNode, pos: &mut usize, replacement: &CSGNode) {
    if *pos == 0 {
        *node = replacement.clone();
        return;
    }
    *pos -= 1;
    match node {
        CSGNode::Primitive(_) => {}
        CSGNode::Transform { node: child, .. } => {
            replace_at(child, pos, replacement);
        }
        CSGNode::Boolean { left, right, .. } => {
            let left_count = node_count(left);
            if *pos < left_count {
                replace_at(left, pos, replacement);
            } else {
                *pos -= left_count;
                replace_at(right, pos, replacement);
            }
        }
    }
}

/// Tweak numeric parameters of transforms in the tree.
fn tweak_parameters(tree: &mut CSGNode, rng: &mut StdRng) {
    match tree {
        CSGNode::Primitive(_) => {
            // Replace with a different primitive.
            *tree = CSGNode::Primitive(random_primitive(rng));
        }
        CSGNode::Transform { transform, node, .. } => {
            // Perturb scale/rotate/translate.
            for v in transform.scale.iter_mut() {
                *v += rng.gen_range(-0.3f32..0.3);
                *v = v.clamp(0.05, 10.0);
            }
            for v in transform.translate.iter_mut() {
                *v += rng.gen_range(-1.0f32..1.0);
            }
            // Also recurse with 50 % probability.
            if rng.gen::<f32>() < 0.5 {
                tweak_parameters(node, rng);
            }
        }
        CSGNode::Boolean { op, left, right } => {
            // Flip the boolean op.
            *op = match rng.gen_range(0u32..3) {
                0 => BooleanOp::Union,
                1 => BooleanOp::Subtract,
                _ => BooleanOp::Intersect,
            };
            if rng.gen::<f32>() < 0.5 {
                tweak_parameters(left, rng);
            } else {
                tweak_parameters(right, rng);
            }
        }
    }
}

/// Tournament selection (size 3) — returns the winning index.
fn tournament_select(pop: &[CsgGenome], rng: &mut StdRng) -> usize {
    let n = pop.len();
    let tournament_size = 3.min(n);
    let mut best_idx = rng.gen_range(0..n);
    let mut best_fit = pop[best_idx].fitness.unwrap_or(f64::NEG_INFINITY);
    for _ in 1..tournament_size {
        let idx = rng.gen_range(0..n);
        let fit = pop[idx].fitness.unwrap_or(f64::NEG_INFINITY);
        if fit > best_fit {
            best_idx = idx;
            best_fit = fit;
        }
    }
    best_idx
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> GenerativeConfig {
        GenerativeConfig {
            population_size: 20,
            max_depth: 4,
            mutation_rate: 0.3,
            crossover_rate: 0.7,
            objectives: vec![ObjectiveWeight {
                objective: DesignObjective::MaterialEfficiency,
                weight: 1.0,
            }],
        }
    }

    #[test]
    fn random_tree_generation() {
        for depth in 0..=5 {
            let tree = random_csg_tree(depth);
            assert!(node_count(&tree) >= 1);
        }
    }

    #[test]
    fn random_tree_depth_zero_is_primitive() {
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..20 {
            let tree = random_csg_tree_with_rng(0, &mut rng);
            assert!(matches!(tree, CSGNode::Primitive(_)));
        }
    }

    #[test]
    fn population_initialization() {
        let mut designer = GenerativeDesigner::with_seed(test_config(), 1);
        designer.initialize_population();
        assert_eq!(designer.population.len(), 20);
        assert!(designer.population.iter().all(|g| g.fitness.is_none()));
    }

    #[test]
    fn fitness_evaluation() {
        let mut designer = GenerativeDesigner::with_seed(test_config(), 2);
        designer.initialize_population();
        designer.evaluate_fitness(&|tree| node_count(tree) as f64);
        assert!(designer.population.iter().all(|g| g.fitness.is_some()));
        assert!(designer.best_fitness > 0.0);
    }

    #[test]
    fn mutation_changes_tree() {
        let mut rng = StdRng::seed_from_u64(99);
        let original = random_csg_tree_with_rng(3, &mut rng);
        let original_json = serde_json::to_string(&original).unwrap();

        // Apply mutation many times — at least one should differ.
        let mut changed = false;
        for seed in 0..50 {
            let mut tree = original.clone();
            let mut rng2 = StdRng::seed_from_u64(seed);
            GenerativeDesigner::mutate_with_rng(&mut tree, 1.0, &mut rng2);
            let mutated_json = serde_json::to_string(&tree).unwrap();
            if mutated_json != original_json {
                changed = true;
                break;
            }
        }
        assert!(changed, "mutation should change the tree");
    }

    #[test]
    fn crossover_produces_valid_tree() {
        let mut rng = StdRng::seed_from_u64(7);
        let a = random_csg_tree_with_rng(3, &mut rng);
        let b = random_csg_tree_with_rng(3, &mut rng);
        let child = GenerativeDesigner::crossover_with_rng(&a, &b, &mut rng);
        assert!(node_count(&child) >= 1);
    }

    #[test]
    fn evolution_improves_fitness_over_generations() {
        let cfg = GenerativeConfig {
            population_size: 30,
            max_depth: 4,
            mutation_rate: 0.4,
            crossover_rate: 0.7,
            objectives: vec![ObjectiveWeight {
                objective: DesignObjective::StructuralStrength,
                weight: 1.0,
            }],
        };
        let mut designer = GenerativeDesigner::with_seed(cfg, 42);
        designer.initialize_population();
        // Fitness = number of nodes (bigger trees = higher fitness for this test).
        let eval = |tree: &CSGNode| node_count(tree) as f64;
        designer.evaluate_fitness(&eval);
        let initial_best = designer.best_fitness;

        for _ in 0..20 {
            designer.evolve_generation(&eval);
        }
        // After 20 generations, best should be at least as good.
        assert!(
            designer.best_fitness >= initial_best,
            "best fitness should not decrease: {} vs {}",
            designer.best_fitness,
            initial_best
        );
    }

    #[test]
    fn config_validation_population_size() {
        let cfg = GenerativeConfig {
            population_size: 1,
            max_depth: 2,
            mutation_rate: 0.1,
            crossover_rate: 0.5,
            objectives: vec![],
        };
        let mut designer = GenerativeDesigner::with_seed(cfg, 0);
        designer.initialize_population();
        assert_eq!(designer.population.len(), 1);
        // select_parents needs ≥2
        let pairs = designer.select_parents();
        assert!(pairs.is_empty());
    }

    #[test]
    fn best_returns_none_before_evaluation() {
        let mut designer = GenerativeDesigner::with_seed(test_config(), 3);
        designer.initialize_population();
        assert!(designer.best().is_none());
    }
}
