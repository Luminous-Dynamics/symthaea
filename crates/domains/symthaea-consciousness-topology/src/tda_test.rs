use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Simplex {
    pub vertices: Vec<usize>,
}
impl Simplex {
    pub fn new(mut vertices: Vec<usize>) -> Self {
        vertices.sort();
        Self { vertices }
    }
    pub fn dimension(&self) -> usize {
        self.vertices.len().saturating_sub(1)
    }
    pub fn faces(&self) -> Vec<Simplex> {
        if self.vertices.len() <= 1 {
            return vec![];
        }
        (0..self.vertices.len())
            .map(|i| {
                let mut face_vertices = self.vertices.clone();
                face_vertices.remove(i);
                Simplex::new(face_vertices)
            })
            .collect()
    }
}

pub struct SimplicialComplex {
    pub filtration_values: HashMap<Simplex, f64>,
}
impl SimplicialComplex {
    pub fn new() -> Self {
        Self {
            filtration_values: HashMap::new(),
        }
    }
    pub fn add_simplex(&mut self, simplex: Simplex, filtration: f64) {
        self.filtration_values.insert(simplex, filtration);
    }
}

#[derive(Debug)]
pub struct PersistencePair {
    pub dimension: usize,
    pub birth: f64,
    pub death: Option<f64>,
}

fn symmetric_difference(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut result = Vec::new();
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        if a[i] < b[j] {
            result.push(a[i]);
            i += 1;
        } else if a[i] > b[j] {
            result.push(b[j]);
            j += 1;
        } else {
            i += 1;
            j += 1;
        }
    }
    while i < a.len() {
        result.push(a[i]);
        i += 1;
    }
    while j < b.len() {
        result.push(b[j]);
        j += 1;
    }
    result
}

pub fn compute_persistent_homology(complex: &SimplicialComplex) -> Vec<PersistencePair> {
    let mut simplices: Vec<(Simplex, f64)> = complex
        .filtration_values
        .iter()
        .map(|(s, f)| (s.clone(), *f))
        .collect();

    simplices.sort_by(|(s1, f1), (s2, f2)| {
        f1.partial_cmp(f2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| s1.dimension().cmp(&s2.dimension()))
    });

    let mut simplex_to_idx: HashMap<Simplex, usize> = HashMap::new();
    for (i, (s, _)) in simplices.iter().enumerate() {
        simplex_to_idx.insert(s.clone(), i);
    }

    let n = simplices.len();
    let mut r: Vec<Vec<usize>> = vec![Vec::new(); n];

    for (j, (s, _)) in simplices.iter().enumerate() {
        let mut boundary = Vec::new();
        for face in s.faces() {
            if let Some(&i) = simplex_to_idx.get(&face) {
                boundary.push(i);
            }
        }
        boundary.sort_unstable();
        r[j] = boundary;
    }

    let mut low_to_col: HashMap<usize, usize> = HashMap::new();
    let mut pairs = Vec::new();

    for j in 0..n {
        while let Some(&i) = r[j].last() {
            if let Some(&j_prime) = low_to_col.get(&i) {
                r[j] = symmetric_difference(&r[j], &r[j_prime]);
            } else {
                low_to_col.insert(i, j);
                break;
            }
        }
    }

    let mut deaths: HashSet<usize> = HashSet::new();

    for (&i, &j) in &low_to_col {
        deaths.insert(i);
        deaths.insert(j);

        let (s_birth, birth_val) = &simplices[i];
        let (s_death, death_val) = &simplices[j];

        let dim = s_birth.dimension();
        pairs.push(PersistencePair {
            dimension: dim,
            birth: *birth_val,
            death: Some(*death_val),
        });
    }

    for i in 0..n {
        if !deaths.contains(&i) {
            let (s_birth, birth_val) = &simplices[i];
            pairs.push(PersistencePair {
                dimension: s_birth.dimension(),
                birth: *birth_val,
                death: None,
            });
        }
    }

    pairs
}

fn main() {
    let mut complex = SimplicialComplex::new();
    complex.add_simplex(Simplex::new(vec![0]), 0.0);
    complex.add_simplex(Simplex::new(vec![1]), 0.0);
    complex.add_simplex(Simplex::new(vec![2]), 0.0);
    complex.add_simplex(Simplex::new(vec![0, 1]), 1.0);
    complex.add_simplex(Simplex::new(vec![1, 2]), 1.0);
    complex.add_simplex(Simplex::new(vec![0, 2]), 2.0); // Creates a loop
    complex.add_simplex(Simplex::new(vec![0, 1, 2]), 3.0); // Fills the loop

    let pairs = compute_persistent_homology(&complex);
    for p in pairs {
        println!("{:?}", p);
    }
}
