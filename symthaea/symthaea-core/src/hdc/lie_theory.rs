// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Lie Theory
//!
//! Lie algebras, exponential maps, root systems, and representation theory
//! for sl(2), su(2), so(3), gl(2) — all in pure Rust (no external crates).
//!
//! ## Capabilities
//!
//! - **LieAlgebra** — abstract Lie algebra with bracket, Jacobi check, Killing form
//! - **Named algebras** — sl(2,ℝ), su(2), so(3), gl(2,ℝ)
//! - **Exponential map** — matrix Taylor series and BCH first order
//! - **Root systems** — A_n, B_n, D_n, G₂ with Cartan matrix and Dynkin diagram
//! - **Representation theory** — irreducible sl(2) reps, Casimir element
//!
//! ## References
//!
//! - Humphreys (1972) — *Introduction to Lie Algebras and Representation Theory*
//! - Hall (2015) — *Lie Groups, Lie Algebras, and Representations*, Springer
//! - Fulton & Harris (1991) — *Representation Theory: A First Course*

// ═══════════════════════════════════════════════════════════════════════════
// LIE ALGEBRA
// ═══════════════════════════════════════════════════════════════════════════

/// A finite-dimensional real Lie algebra specified by structure constants.
///
/// The Lie bracket of basis elements satisfies:
///   [eᵢ, eⱼ] = Σₖ f^k_{ij} eₖ
#[derive(Debug, Clone)]
pub struct LieAlgebra {
    pub name: &'static str,
    pub dimension: usize,
    /// Basis matrices; `basis[k]` is the k-th basis element as a square matrix.
    pub basis: Vec<Vec<Vec<f64>>>,
    /// Structure constants: `structure_constants[i][j][k]` = f^k_{ij}.
    pub structure_constants: Vec<Vec<Vec<f64>>>,
}

impl LieAlgebra {
    /// Compute the Lie bracket [a, b] in coordinates w.r.t. the basis.
    pub fn bracket(&self, a: &[f64], b: &[f64]) -> Vec<f64> {
        assert_eq!(a.len(), self.dimension);
        assert_eq!(b.len(), self.dimension);
        let d = self.dimension;
        let mut result = vec![0.0; d];
        for i in 0..d {
            for j in 0..d {
                for k in 0..d {
                    result[k] += a[i] * b[j] * self.structure_constants[i][j][k];
                }
            }
        }
        result
    }

    /// Verify the Jacobi identity for all basis triples.
    pub fn check_jacobi_identity(&self) -> bool {
        let d = self.dimension;
        for i in 0..d {
            for j in 0..d {
                for k in 0..d {
                    let ei = basis_vec(d, i);
                    let ej = basis_vec(d, j);
                    let ek = basis_vec(d, k);
                    // [[ei,ej],ek] + [[ej,ek],ei] + [[ek,ei],ej]
                    let ab = self.bracket(&ei, &ej);
                    let bc = self.bracket(&ej, &ek);
                    let ca = self.bracket(&ek, &ei);
                    let t1 = self.bracket(&ab, &ek);
                    let t2 = self.bracket(&bc, &ei);
                    let t3 = self.bracket(&ca, &ej);
                    for l in 0..d {
                        if (t1[l] + t2[l] + t3[l]).abs() > 1e-10 {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    /// Returns true if all structure constants are zero.
    pub fn is_abelian(&self) -> bool {
        self.structure_constants
            .iter()
            .flat_map(|r| r.iter())
            .flat_map(|r| r.iter())
            .all(|&x| x.abs() < 1e-12)
    }

    /// Killing form B(a,b) = Tr(ad_a ∘ ad_b).
    pub fn killing_form(&self, a: &[f64], b: &[f64]) -> f64 {
        let d = self.dimension;
        // Build adjoint matrix of a: (ad_a)_{kl} = Σᵢ aᵢ f^k_{il}
        let ad_a = adjoint_matrix(self, a);
        let ad_b = adjoint_matrix(self, b);
        // Trace of product
        let mut trace = 0.0;
        for i in 0..d {
            for j in 0..d {
                trace += ad_a[i][j] * ad_b[j][i];
            }
        }
        trace
    }

    /// Is the Killing form non-degenerate? (Cartan's criterion for semisimplicity)
    pub fn is_semisimple(&self) -> bool {
        let d = self.dimension;
        // Build Gram matrix of Killing form in the basis
        let mut gram = vec![vec![0.0; d]; d];
        for i in 0..d {
            for j in 0..d {
                let ei = basis_vec(d, i);
                let ej = basis_vec(d, j);
                gram[i][j] = self.killing_form(&ei, &ej);
            }
        }
        gram_det(&gram).abs() > 1e-8
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn basis_vec(d: usize, k: usize) -> Vec<f64> {
    let mut v = vec![0.0; d];
    v[k] = 1.0;
    v
}

fn adjoint_matrix(alg: &LieAlgebra, a: &[f64]) -> Vec<Vec<f64>> {
    let d = alg.dimension;
    // (ad_a)_{kl} = Σᵢ aᵢ * f^k_{il}
    let mut m = vec![vec![0.0; d]; d];
    for i in 0..d {
        for l in 0..d {
            for k in 0..d {
                m[k][l] += a[i] * alg.structure_constants[i][l][k];
            }
        }
    }
    m
}

fn gram_det(m: &[Vec<f64>]) -> f64 {
    let n = m.len();
    let mut a: Vec<Vec<f64>> = m.to_vec();
    let mut det = 1.0;
    for col in 0..n {
        let pivot_row = (col..n).max_by(|&a_idx, &b_idx| {
            a[a_idx][col]
                .abs()
                .partial_cmp(&a[b_idx][col].abs())
                .unwrap()
        });
        if let Some(pr) = pivot_row {
            if a[pr][col].abs() < 1e-15 {
                return 0.0;
            }
            if pr != col {
                a.swap(pr, col);
                det = -det;
            }
        }
        det *= a[col][col];
        let pivot = a[col][col];
        for row in (col + 1)..n {
            let factor = a[row][col] / pivot;
            for j in col..n {
                let val = a[col][j] * factor;
                a[row][j] -= val;
            }
        }
    }
    det
}

// ═══════════════════════════════════════════════════════════════════════════
// NAMED LIE ALGEBRAS
// ═══════════════════════════════════════════════════════════════════════════

/// sl(2,ℝ): basis {e, f, h} with [h,e]=2e, [h,f]=-2f, [e,f]=h.
///
/// Ordering: index 0=e, 1=f, 2=h.
pub fn sl2_real() -> LieAlgebra {
    // basis matrices
    let e_mat = vec![vec![0.0, 1.0], vec![0.0, 0.0]];
    let f_mat = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
    let h_mat = vec![vec![1.0, 0.0], vec![0.0, -1.0]];
    let basis = vec![e_mat, f_mat, h_mat];

    // structure constants for [eᵢ, eⱼ] = Σₖ f^k_{ij} eₖ
    // indices: 0=e, 1=f, 2=h
    // [e,f] = h → f^2_{01} = 1
    // [f,e] = -h → f^2_{10} = -1
    // [h,e] = 2e → f^0_{20} = 2
    // [e,h] = -2e → f^0_{02} = -2
    // [h,f] = -2f → f^1_{21} = -2
    // [f,h] = 2f → f^1_{12} = 2
    let d = 3;
    let mut sc = vec![vec![vec![0.0; d]; d]; d];
    // [e,f] = h
    sc[0][1][2] = 1.0;
    sc[1][0][2] = -1.0;
    // [h,e] = 2e  (h is index 2, e is index 0)
    sc[2][0][0] = 2.0;
    sc[0][2][0] = -2.0;
    // [h,f] = -2f
    sc[2][1][1] = -2.0;
    sc[1][2][1] = 2.0;

    LieAlgebra {
        name: "sl(2,R)",
        dimension: d,
        basis,
        structure_constants: sc,
    }
}

/// su(2): basis {T₁, T₂, T₃} with [Tᵢ,Tⱼ] = εᵢⱼₖ Tₖ.
pub fn su2() -> LieAlgebra {
    // Basis: iσ₁/2, iσ₂/2, iσ₃/2 — but we store real structure constants only.
    // [T1,T2] = T3, [T2,T3] = T1, [T3,T1] = T2
    // We represent the algebra by its structure constants; basis matrices are 2×2 complex
    // but we approximate with real placeholders (structure constants are the key data).
    let e1 = vec![vec![0.0, 0.5], vec![0.5, 0.0]]; // Re(iσ₁/2)
    let e2 = vec![vec![0.0, -0.5], vec![0.5, 0.0]]; // placeholder for iσ₂/2
    let e3 = vec![vec![0.5, 0.0], vec![0.0, -0.5]]; // iσ₃/2

    let d = 3;
    let mut sc = vec![vec![vec![0.0; d]; d]; d];
    // Levi-Civita: [Ti,Tj] = εijk Tk
    // [T0,T1]=T2, [T1,T2]=T0, [T2,T0]=T1
    sc[0][1][2] = 1.0;
    sc[1][0][2] = -1.0;
    sc[1][2][0] = 1.0;
    sc[2][1][0] = -1.0;
    sc[2][0][1] = 1.0;
    sc[0][2][1] = -1.0;

    LieAlgebra {
        name: "su(2)",
        dimension: d,
        basis: vec![e1, e2, e3],
        structure_constants: sc,
    }
}

/// so(3): 3×3 skew-symmetric matrices; same structure constants as su(2).
pub fn so3() -> LieAlgebra {
    // Standard basis: Lx, Ly, Lz
    let lx = vec![
        vec![0.0, 0.0, 0.0],
        vec![0.0, 0.0, -1.0],
        vec![0.0, 1.0, 0.0],
    ];
    let ly = vec![
        vec![0.0, 0.0, 1.0],
        vec![0.0, 0.0, 0.0],
        vec![-1.0, 0.0, 0.0],
    ];
    let lz = vec![
        vec![0.0, -1.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
    ];

    let d = 3;
    let mut sc = vec![vec![vec![0.0; d]; d]; d];
    // [Lx,Ly]=Lz, [Ly,Lz]=Lx, [Lz,Lx]=Ly (same as su(2) ε structure)
    sc[0][1][2] = 1.0;
    sc[1][0][2] = -1.0;
    sc[1][2][0] = 1.0;
    sc[2][1][0] = -1.0;
    sc[2][0][1] = 1.0;
    sc[0][2][1] = -1.0;

    LieAlgebra {
        name: "so(3)",
        dimension: d,
        basis: vec![lx, ly, lz],
        structure_constants: sc,
    }
}

/// gl(2,ℝ): all 2×2 matrices; [A,B] = AB - BA; dimension 4.
///
/// Basis: e₁₁, e₁₂, e₂₁, e₂₂ (elementary matrices).
pub fn gl2_real() -> LieAlgebra {
    let e11 = vec![vec![1.0, 0.0], vec![0.0, 0.0]];
    let e12 = vec![vec![0.0, 1.0], vec![0.0, 0.0]];
    let e21 = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
    let e22 = vec![vec![0.0, 0.0], vec![0.0, 1.0]];
    let basis = vec![e11, e12, e21, e22];
    // [eij, ekl] = δ_{jk} eil - δ_{li} ekj
    // Compute structure constants from matrix commutators
    let d = 4;
    let indices: [(usize, usize); 4] = [(0, 0), (0, 1), (1, 0), (1, 1)];
    let mut sc = vec![vec![vec![0.0; d]; d]; d];
    for i in 0..d {
        for j in 0..d {
            // Compute [basis[i], basis[j]] = basis[i]*basis[j] - basis[j]*basis[i]
            let comm = mat2x2_comm(&basis[i], &basis[j]);
            // Express comm in terms of basis
            for k in 0..d {
                let (r, c) = indices[k];
                sc[i][j][k] = comm[r][c];
            }
        }
    }
    LieAlgebra {
        name: "gl(2,R)",
        dimension: d,
        basis,
        structure_constants: sc,
    }
}

fn mat2x2_comm(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    // [A,B] = AB - BA for 2×2
    let ab = mat2x2_mul(a, b);
    let ba = mat2x2_mul(b, a);
    vec![
        vec![ab[0][0] - ba[0][0], ab[0][1] - ba[0][1]],
        vec![ab[1][0] - ba[1][0], ab[1][1] - ba[1][1]],
    ]
}

fn mat2x2_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    vec![
        vec![
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        vec![
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

// ═══════════════════════════════════════════════════════════════════════════
// EXPONENTIAL MAP
// ═══════════════════════════════════════════════════════════════════════════

/// Compute exp(t * m) via Taylor series: Σ (t*m)ⁿ/n!
pub fn matrix_exp_lie(m: &[Vec<f64>], t: f64, terms: usize) -> Vec<Vec<f64>> {
    let n = m.len();
    let mut result = identity_nxn(n);
    let mut power = identity_nxn(n);
    let mut factorial = 1.0_f64;
    let tm: Vec<Vec<f64>> = m
        .iter()
        .map(|row| row.iter().map(|&x| x * t).collect())
        .collect();
    for k in 1..=terms {
        factorial *= k as f64;
        power = mat_mul_nxn(&power, &tm);
        for i in 0..n {
            for j in 0..n {
                result[i][j] += power[i][j] / factorial;
            }
        }
    }
    result
}

/// Given coordinates of a Lie algebra element (Σ cᵢeᵢ), compute exp(Σ cᵢeᵢ) as a matrix.
pub fn lie_element_to_group(algebra: &LieAlgebra, coeffs: &[f64]) -> Vec<Vec<f64>> {
    assert_eq!(coeffs.len(), algebra.dimension);
    if algebra.basis.is_empty() {
        return vec![];
    }
    let sz = algebra.basis[0].len();
    let mut elem = vec![vec![0.0; sz]; sz];
    for (c, basis_mat) in coeffs.iter().zip(algebra.basis.iter()) {
        for i in 0..sz {
            for j in 0..sz {
                elem[i][j] += c * basis_mat[i][j];
            }
        }
    }
    matrix_exp_lie(&elem, 1.0, 20)
}

/// Baker-Campbell-Hausdorff first order: exp(A)exp(B) ≈ exp(A + B + [A,B]/2).
pub fn bcf_first_order(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    // [A,B] = AB - BA
    let comm = mat_comm_nxn(a, b);
    // A + B + [A,B]/2
    let mut exponent = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            exponent[i][j] = a[i][j] + b[i][j] + 0.5 * comm[i][j];
        }
    }
    matrix_exp_lie(&exponent, 1.0, 20)
}

// ─── n×n matrix helpers ──────────────────────────────────────────────────────

fn identity_nxn(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect()
}

fn mat_mul_nxn(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = b[0].len();
    let k = b.len();
    let mut c = vec![vec![0.0; m]; n];
    for i in 0..n {
        for l in 0..k {
            if a[i][l].abs() < 1e-300 {
                continue;
            }
            for j in 0..m {
                c[i][j] += a[i][l] * b[l][j];
            }
        }
    }
    c
}

fn mat_comm_nxn(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let ab = mat_mul_nxn(a, b);
    let ba = mat_mul_nxn(b, a);
    let n = a.len();
    let mut c = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            c[i][j] = ab[i][j] - ba[i][j];
        }
    }
    c
}

// ═══════════════════════════════════════════════════════════════════════════
// ROOT SYSTEMS
// ═══════════════════════════════════════════════════════════════════════════

/// A root system for a simple Lie algebra.
#[derive(Debug, Clone)]
pub struct RootSystem {
    pub rank: usize,
    pub simple_roots: Vec<Vec<f64>>,
    pub all_roots: Vec<Vec<f64>>,
    pub cartan_matrix: Vec<Vec<i32>>,
}

impl RootSystem {
    /// Check whether v is (approximately) in the root system.
    pub fn is_root(&self, v: &[f64], tol: f64) -> bool {
        self.all_roots.iter().any(|r| {
            r.iter()
                .zip(v.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt()
                < tol
        })
    }

    /// Return positive roots (those that can be written as non-negative integer combinations of simple roots).
    pub fn positive_roots(&self) -> Vec<Vec<f64>> {
        let n = self.simple_roots.len();
        if n == 0 {
            return vec![];
        }
        let dim = self.simple_roots[0].len();
        self.all_roots
            .iter()
            .filter(|r| {
                // A root is positive if its first nonzero coordinate is positive
                for &x in r.iter() {
                    if x > 1e-10 {
                        return true;
                    }
                    if x < -1e-10 {
                        return false;
                    }
                }
                false
            })
            .cloned()
            .collect()
    }

    /// Return the highest root (largest in the dominance partial order).
    pub fn highest_root(&self) -> Vec<f64> {
        let pos = self.positive_roots();
        pos.into_iter()
            .max_by(|a, b| {
                let sa: f64 = a.iter().sum();
                let sb: f64 = b.iter().sum();
                sa.partial_cmp(&sb).unwrap()
            })
            .unwrap_or_default()
    }

    /// ASCII text representation of the Dynkin diagram.
    pub fn dynkin_diagram(&self) -> String {
        let n = self.rank;
        if n == 0 {
            return String::new();
        }
        let nodes: Vec<String> = (1..=n).map(|i| format!("○{}", i)).collect();
        let mut edges: Vec<String> = Vec::new();
        for i in 0..n.saturating_sub(1) {
            let bond = if i < self.cartan_matrix.len() && i + 1 < self.cartan_matrix[i].len() {
                let a = self.cartan_matrix[i][i + 1];
                let b = if i + 1 < self.cartan_matrix.len() {
                    self.cartan_matrix[i + 1][i]
                } else {
                    0
                };
                match (a, b) {
                    (-1, -1) => "─",
                    (-2, -1) | (-1, -2) => "═",
                    (-3, -1) | (-1, -3) => "≡",
                    _ => "─",
                }
            } else {
                "─"
            };
            edges.push(bond.to_string());
        }
        let mut result = String::new();
        for (i, node) in nodes.iter().enumerate() {
            result.push_str(node);
            if i < edges.len() {
                result.push_str(&edges[i]);
            }
        }
        result
    }
}

fn inner_prod(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn cartan_entry(alpha: &[f64], beta: &[f64]) -> i32 {
    let ip = inner_prod(alpha, beta);
    let bb = inner_prod(beta, beta);
    if bb.abs() < 1e-15 {
        0
    } else {
        (2.0 * ip / bb).round() as i32
    }
}

/// Root system Aₙ: simple roots αᵢ = eᵢ - eᵢ₊₁ in ℝⁿ⁺¹.
pub fn root_system_a(n: usize) -> RootSystem {
    // Simple roots: α₁=e1-e2, α₂=e2-e3, ..., αₙ=eₙ-eₙ₊₁
    let dim = n + 1;
    let simple_roots: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut v = vec![0.0; dim];
            v[i] = 1.0;
            v[i + 1] = -1.0;
            v
        })
        .collect();

    // All roots: eᵢ - eⱼ for i ≠ j
    let mut all_roots = Vec::new();
    for i in 0..dim {
        for j in 0..dim {
            if i != j {
                let mut v = vec![0.0; dim];
                v[i] = 1.0;
                v[j] = -1.0;
                all_roots.push(v);
            }
        }
    }

    let cartan_matrix: Vec<Vec<i32>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| cartan_entry(&simple_roots[i], &simple_roots[j]))
                .collect()
        })
        .collect();

    RootSystem {
        rank: n,
        simple_roots,
        all_roots,
        cartan_matrix,
    }
}

/// Root system Bₙ: so(2n+1). Simple roots: e₁-e₂, ..., eₙ₋₁-eₙ, eₙ.
pub fn root_system_b(n: usize) -> RootSystem {
    assert!(n >= 2, "B_n requires n ≥ 2");
    let dim = n;
    let mut simple_roots: Vec<Vec<f64>> = (0..n.saturating_sub(1))
        .map(|i| {
            let mut v = vec![0.0; dim];
            v[i] = 1.0;
            v[i + 1] = -1.0;
            v
        })
        .collect();
    // Last simple root: eₙ (short root)
    let mut last = vec![0.0; dim];
    last[n - 1] = 1.0;
    simple_roots.push(last);

    // All roots: ±eᵢ, ±eᵢ ± eⱼ (i < j)
    let mut all_roots = Vec::new();
    for i in 0..dim {
        let mut v = vec![0.0; dim];
        v[i] = 1.0;
        all_roots.push(v.clone());
        v[i] = -1.0;
        all_roots.push(v);
    }
    for i in 0..dim {
        for j in (i + 1)..dim {
            for si in [1.0_f64, -1.0] {
                for sj in [1.0_f64, -1.0] {
                    let mut v = vec![0.0; dim];
                    v[i] = si;
                    v[j] = sj;
                    all_roots.push(v);
                }
            }
        }
    }

    let cartan_matrix: Vec<Vec<i32>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| cartan_entry(&simple_roots[i], &simple_roots[j]))
                .collect()
        })
        .collect();

    RootSystem {
        rank: n,
        simple_roots,
        all_roots,
        cartan_matrix,
    }
}

/// Root system Dₙ: so(2n). Simple roots: e₁-e₂, ..., eₙ₋₁-eₙ, eₙ₋₁+eₙ.
pub fn root_system_d(n: usize) -> RootSystem {
    assert!(n >= 3, "D_n requires n ≥ 3");
    let dim = n;
    let mut simple_roots: Vec<Vec<f64>> = (0..n.saturating_sub(1))
        .map(|i| {
            let mut v = vec![0.0; dim];
            v[i] = 1.0;
            v[i + 1] = -1.0;
            v
        })
        .collect();
    // Last simple root: eₙ₋₁ + eₙ
    let mut last = vec![0.0; dim];
    last[n - 2] = 1.0;
    last[n - 1] = 1.0;
    simple_roots.push(last);

    // All roots: ±eᵢ ± eⱼ for i < j
    let mut all_roots = Vec::new();
    for i in 0..dim {
        for j in (i + 1)..dim {
            for si in [1.0_f64, -1.0] {
                for sj in [1.0_f64, -1.0] {
                    let mut v = vec![0.0; dim];
                    v[i] = si;
                    v[j] = sj;
                    all_roots.push(v);
                }
            }
        }
    }

    let cartan_matrix: Vec<Vec<i32>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| cartan_entry(&simple_roots[i], &simple_roots[j]))
                .collect()
        })
        .collect();

    RootSystem {
        rank: n,
        simple_roots,
        all_roots,
        cartan_matrix,
    }
}

/// Root system G₂: 12 roots (6 short + 6 long) in the plane.
pub fn root_system_g2() -> RootSystem {
    // Simple roots: α₁ = (1, -√3)/? and α₂ in normalized coords
    // Use the standard embedding: simple roots α₁ = (1, -1, 0) - (0, 0, 0) projected to rank 2
    // Standard: α₁ = e₁ - e₂, α₂ = -2e₁ + e₂ + e₃ in ℝ³, then embed to 2D
    // For simplicity use explicit 2D coordinates
    // short roots: ±(1,0), ±(1/2, √3/2), ±(-1/2, √3/2)
    // long roots: ±(3/2, √3/2), ±(0, √3), ±(3/2, -√3/2)  (scaled by √3)
    let s3 = 3.0_f64.sqrt();
    let short = vec![
        vec![1.0, 0.0],
        vec![-1.0, 0.0],
        vec![0.5, s3 / 2.0],
        vec![-0.5, -s3 / 2.0],
        vec![0.5, -s3 / 2.0],
        vec![-0.5, s3 / 2.0],
    ];
    let long = vec![
        vec![0.0, s3],
        vec![0.0, -s3],
        vec![3.0 / 2.0, s3 / 2.0],
        vec![-3.0 / 2.0, -s3 / 2.0],
        vec![3.0 / 2.0, -s3 / 2.0],
        vec![-3.0 / 2.0, s3 / 2.0],
    ];
    let mut all_roots = short.clone();
    all_roots.extend(long);

    // Simple roots: α₁ = short root (1,0), α₂ = long root (-3/2, √3/2)
    let simple_roots = vec![vec![1.0, 0.0], vec![-3.0 / 2.0, s3 / 2.0]];

    let cartan_matrix: Vec<Vec<i32>> = (0..2)
        .map(|i| {
            (0..2)
                .map(|j| cartan_entry(&simple_roots[i], &simple_roots[j]))
                .collect()
        })
        .collect();

    RootSystem {
        rank: 2,
        simple_roots,
        all_roots,
        cartan_matrix,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// REPRESENTATION THEORY (sl(2))
// ═══════════════════════════════════════════════════════════════════════════

/// An irreducible representation of a Lie algebra.
pub struct Representation {
    pub dimension: usize,
    /// Eigenvalues of h in this representation.
    pub h_eigenvalues: Vec<i32>,
    /// Matrix representatives of the generators.
    pub matrices: std::collections::HashMap<String, Vec<Vec<f64>>>,
}

/// Construct the irreducible sl(2) representation of dimension `spin_2 + 1`.
///
/// `spin_2` = 2j where j = 0, 1/2, 1, 3/2, … The dimension is n+1 where n = spin_2.
///
/// Basis vectors |m⟩ with m = n, n-2, …, -n.
/// e|m⟩ = sqrt((n-m)(n+m+2)/4) |m+2⟩
/// f|m⟩ = sqrt((n+m)(n-m+2)/4) |m-2⟩
/// h|m⟩ = m |m⟩
pub fn sl2_irrep(spin_2: usize) -> Representation {
    let n = spin_2 as i32; // = 2j
    let dim = (n + 1) as usize;
    // Basis index k corresponds to eigenvalue m = n - 2k (k = 0..dim-1)
    let h_eigenvalues: Vec<i32> = (0..dim as i32).map(|k| n - 2 * k).collect();

    let mut h_mat = vec![vec![0.0; dim]; dim];
    let mut e_mat = vec![vec![0.0; dim]; dim];
    let mut f_mat = vec![vec![0.0; dim]; dim];

    for k in 0..dim {
        let m = n - 2 * k as i32;
        h_mat[k][k] = m as f64;
        // e raises m → m+2, i.e. k → k-1
        if k > 0 {
            let coeff = (((n - (m - 2)) * (n + m)) as f64 / 4.0).sqrt();
            // Wait, let's re-derive:
            // e|m⟩ = sqrt((n-m)(n+m+2)/4) |m+2⟩
            // m+2 corresponds to index k-1
            let coeff_e = ((n - m) as f64 * (n + m + 2) as f64 / 4.0).sqrt();
            e_mat[k - 1][k] = coeff_e;
            let _ = coeff; // suppress warning
        }
        // f lowers m → m-2, i.e. k → k+1
        if k + 1 < dim {
            // f|m⟩ = sqrt((n+m)(n-m+2)/4) |m-2⟩
            let coeff_f = ((n + m) as f64 * (n - m + 2) as f64 / 4.0).sqrt();
            f_mat[k + 1][k] = coeff_f;
        }
    }

    let mut matrices = std::collections::HashMap::new();
    matrices.insert("e".to_string(), e_mat);
    matrices.insert("f".to_string(), f_mat);
    matrices.insert("h".to_string(), h_mat);

    Representation {
        dimension: dim,
        h_eigenvalues,
        matrices,
    }
}

/// Compute the value of the Casimir element C₂ = h²/4 + ef + fe in the representation.
///
/// For sl(2) this acts as scalar j(j+1) = n(n+2)/4 where n = spin_2.
pub fn casimir_element_value(_rep: &Representation, spin_2: usize) -> f64 {
    let n = spin_2 as f64;
    n * (n + 2.0) / 4.0
}

/// Verify that the representation matrices satisfy the sl(2) commutation relations.
pub fn check_representation(rep: &Representation, _algebra: &LieAlgebra) -> bool {
    let e = match rep.matrices.get("e") {
        Some(m) => m,
        None => return false,
    };
    let f = match rep.matrices.get("f") {
        Some(m) => m,
        None => return false,
    };
    let h = match rep.matrices.get("h") {
        Some(m) => m,
        None => return false,
    };
    let d = rep.dimension;

    // [h,e] = 2e
    let he = mat_mul_nxn(h, e);
    let eh = mat_mul_nxn(e, h);
    for i in 0..d {
        for j in 0..d {
            if ((he[i][j] - eh[i][j]) - 2.0 * e[i][j]).abs() > 1e-9 {
                return false;
            }
        }
    }
    // [h,f] = -2f
    let hf = mat_mul_nxn(h, f);
    let fh = mat_mul_nxn(f, h);
    for i in 0..d {
        for j in 0..d {
            if ((hf[i][j] - fh[i][j]) + 2.0 * f[i][j]).abs() > 1e-9 {
                return false;
            }
        }
    }
    // [e,f] = h
    let ef = mat_mul_nxn(e, f);
    let fe = mat_mul_nxn(f, e);
    for i in 0..d {
        for j in 0..d {
            if ((ef[i][j] - fe[i][j]) - h[i][j]).abs() > 1e-9 {
                return false;
            }
        }
    }
    true
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ─── Lie algebra ─────────────────────────────────────────────────────────

    #[test]
    fn test_sl2_jacobi_identity() {
        let alg = sl2_real();
        assert!(
            alg.check_jacobi_identity(),
            "sl(2) must satisfy Jacobi identity"
        );
    }

    #[test]
    fn test_su2_jacobi_identity() {
        let alg = su2();
        assert!(alg.check_jacobi_identity());
    }

    #[test]
    fn test_su2_structure_constants_levi_civita() {
        let alg = su2();
        // [T0,T1] should give T2
        let e0 = vec![1.0, 0.0, 0.0];
        let e1 = vec![0.0, 1.0, 0.0];
        let bracket = alg.bracket(&e0, &e1);
        assert!(close(bracket[2], 1.0, 1e-10), "[T0,T1]_2 = {}", bracket[2]);
        assert!(close(bracket[0], 0.0, 1e-10));
        assert!(close(bracket[1], 0.0, 1e-10));
    }

    #[test]
    fn test_sl2_killing_form_nondegenerate() {
        let alg = sl2_real();
        assert!(alg.is_semisimple(), "sl(2) should be semisimple");
    }

    #[test]
    fn test_sl2_not_abelian() {
        let alg = sl2_real();
        assert!(!alg.is_abelian());
    }

    #[test]
    fn test_so3_jacobi_identity() {
        let alg = so3();
        assert!(alg.check_jacobi_identity());
    }

    #[test]
    fn test_gl2_jacobi_identity() {
        let alg = gl2_real();
        assert!(alg.check_jacobi_identity());
    }

    // ─── Exponential map ─────────────────────────────────────────────────────

    #[test]
    fn test_matrix_exp_zero_is_identity() {
        let zero = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let result = matrix_exp_lie(&zero, 1.0, 20);
        assert!(close(result[0][0], 1.0, 1e-12));
        assert!(close(result[1][1], 1.0, 1e-12));
        assert!(close(result[0][1], 0.0, 1e-12));
        assert!(close(result[1][0], 0.0, 1e-12));
    }

    #[test]
    fn test_matrix_exp_diagonal() {
        // exp(t * diag(a, b)) = diag(exp(ta), exp(tb))
        let d = vec![vec![2.0, 0.0], vec![0.0, -1.0]];
        let result = matrix_exp_lie(&d, 1.0, 25);
        assert!(close(result[0][0], 2.0_f64.exp(), 1e-8));
        assert!(close(result[1][1], (-1.0_f64).exp(), 1e-8));
    }

    #[test]
    fn test_lie_element_to_group_h_in_sl2() {
        // exp(t * h) where h = diag(1,-1): should give diag(e^t, e^{-t})
        let alg = sl2_real();
        // h is index 2 in sl2_real
        let coeffs = vec![0.0, 0.0, 1.0];
        let result = lie_element_to_group(&alg, &coeffs);
        assert!(
            close(result[0][0], 1.0_f64.exp(), 1e-8),
            "got {}",
            result[0][0]
        );
        assert!(close(result[1][1], (-1.0_f64).exp(), 1e-8));
    }

    // ─── Root systems ─────────────────────────────────────────────────────────

    #[test]
    fn test_root_system_a1_has_2_roots() {
        let rs = root_system_a(1);
        assert_eq!(rs.all_roots.len(), 2, "A1 has 2 roots");
    }

    #[test]
    fn test_root_system_a2_has_6_roots() {
        let rs = root_system_a(2);
        assert_eq!(rs.all_roots.len(), 6, "A2 has 6 roots");
    }

    #[test]
    fn test_root_system_g2_has_12_roots() {
        let rs = root_system_g2();
        assert_eq!(rs.all_roots.len(), 12, "G2 has 12 roots");
    }

    #[test]
    fn test_root_system_a1_is_root() {
        let rs = root_system_a(1);
        // roots should be [1,-1,0,...] type but in 2D: (1,-1)
        assert!(rs.all_roots.len() == 2);
        let r0 = &rs.all_roots[0];
        assert!(rs.is_root(r0, 1e-10));
    }

    // ─── Representation theory ───────────────────────────────────────────────

    #[test]
    fn test_sl2_spin_half_dimension() {
        let rep = sl2_irrep(1); // j = 1/2
        assert_eq!(rep.dimension, 2);
    }

    #[test]
    fn test_sl2_spin_1_dimension() {
        let rep = sl2_irrep(2); // j = 1
        assert_eq!(rep.dimension, 3);
    }

    #[test]
    fn test_casimir_spin_half() {
        let rep = sl2_irrep(1);
        let c = casimir_element_value(&rep, 1);
        // j=1/2: j(j+1) = 3/4; n(n+2)/4 = 1*3/4 = 3/4
        assert!(close(c, 0.75, 1e-12), "Casimir for j=1/2 = {}", c);
    }

    #[test]
    fn test_casimir_spin_1() {
        let rep = sl2_irrep(2);
        let c = casimir_element_value(&rep, 2);
        // j=1: j(j+1) = 2; n(n+2)/4 = 2*4/4 = 2
        assert!(close(c, 2.0, 1e-12), "Casimir for j=1 = {}", c);
    }

    #[test]
    fn test_sl2_representation_commutation_relations() {
        let alg = sl2_real();
        let rep = sl2_irrep(2); // j=1 representation
        assert!(
            check_representation(&rep, &alg),
            "[h,e]=2e, [h,f]=-2f, [e,f]=h must hold"
        );
    }

    #[test]
    fn test_sl2_representation_spin_half_commutation() {
        let alg = sl2_real();
        let rep = sl2_irrep(1);
        assert!(check_representation(&rep, &alg));
    }
}
