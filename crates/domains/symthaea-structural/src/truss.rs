// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! 2D pin-jointed truss analysis by the method of joints.
//!
//! Solves the axial force in every member of a statically determinate, planar,
//! pin-jointed truss by assembling the joint-equilibrium equations
//! (ΣFx = 0, ΣFy = 0 at each node) and solving the resulting linear system.
//! Member forces are signed: **positive = tension, negative = compression**.
//!
//! Determinacy: a planar truss is statically determinate when
//! `members + reactions == 2 · nodes`. Otherwise this returns an error rather
//! than a meaningless answer.

/// Support condition at a node, in terms of which reaction components exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Support {
    /// No reactions.
    Free,
    /// Pinned: reacts in both x and y.
    Pin,
    /// Roller on a horizontal surface: vertical reaction only.
    RollerVertical,
    /// Roller on a vertical surface: horizontal reaction only.
    RollerHorizontal,
}

impl Support {
    fn reaction_count(self) -> usize {
        match self {
            Support::Free => 0,
            Support::Pin => 2,
            Support::RollerVertical | Support::RollerHorizontal => 1,
        }
    }
}

/// A joint (node) at a position, with a support condition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Node {
    pub x: f64,
    pub y: f64,
    pub support: Support,
}

impl Node {
    pub fn free(x: f64, y: f64) -> Node {
        Node {
            x,
            y,
            support: Support::Free,
        }
    }
    pub fn pin(x: f64, y: f64) -> Node {
        Node {
            x,
            y,
            support: Support::Pin,
        }
    }
    pub fn roller_vertical(x: f64, y: f64) -> Node {
        Node {
            x,
            y,
            support: Support::RollerVertical,
        }
    }
}

/// A member connecting two node indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Member {
    pub i: usize,
    pub j: usize,
}

/// An external point load applied at a node (N).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Load {
    pub node: usize,
    pub fx: f64,
    pub fy: f64,
}

/// A planar pin-jointed truss.
#[derive(Debug, Clone, PartialEq)]
pub struct Truss {
    pub nodes: Vec<Node>,
    pub members: Vec<Member>,
    pub loads: Vec<Load>,
}

/// A reaction force resolved at a supported node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reaction {
    pub node: usize,
    pub fx: f64,
    pub fy: f64,
}

/// The solved internal state of a truss.
#[derive(Debug, Clone, PartialEq)]
pub struct TrussSolution {
    /// Axial force per member (same order as `Truss::members`). +tension, −compression.
    pub member_forces: Vec<f64>,
    /// Support reactions, one entry per supported node.
    pub reactions: Vec<Reaction>,
}

/// Errors from truss analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrussError {
    /// `members + reactions != 2·nodes` — not statically determinate.
    NotDeterminate { equations: usize, unknowns: usize },
    /// A member has zero length (coincident nodes).
    ZeroLengthMember(usize),
    /// The equilibrium system is singular (mechanism / unstable geometry).
    Singular,
}

impl Truss {
    /// Solve for member forces and support reactions.
    pub fn solve(&self) -> Result<TrussSolution, TrussError> {
        let n = self.nodes.len();
        let m = self.members.len();
        let reaction_total: usize = self
            .nodes
            .iter()
            .map(|nd| nd.support.reaction_count())
            .sum();
        let equations = 2 * n;
        let unknowns = m + reaction_total;
        if equations != unknowns {
            return Err(TrussError::NotDeterminate {
                equations,
                unknowns,
            });
        }

        // Reaction unknown layout: columns m..unknowns, one per reaction
        // component, in node order. Record (node, (dir_x, dir_y)) per column.
        let mut reaction_cols: Vec<(usize, f64, f64)> = Vec::new();
        for (idx, nd) in self.nodes.iter().enumerate() {
            match nd.support {
                Support::Free => {}
                Support::Pin => {
                    reaction_cols.push((idx, 1.0, 0.0));
                    reaction_cols.push((idx, 0.0, 1.0));
                }
                Support::RollerVertical => reaction_cols.push((idx, 0.0, 1.0)),
                Support::RollerHorizontal => reaction_cols.push((idx, 1.0, 0.0)),
            }
        }

        // Assemble A (equations × unknowns) and b.
        let mut a = vec![vec![0.0f64; unknowns]; equations];
        let mut b = vec![0.0f64; equations];

        // Member columns 0..m: a tensile force pulls each end joint toward the
        // other end.
        for (col, mem) in self.members.iter().enumerate() {
            let ni = self.nodes[mem.i];
            let nj = self.nodes[mem.j];
            let dx = nj.x - ni.x;
            let dy = nj.y - ni.y;
            let len = (dx * dx + dy * dy).sqrt();
            if len < 1e-12 {
                return Err(TrussError::ZeroLengthMember(col));
            }
            let (ux, uy) = (dx / len, dy / len);
            // Node i: +u toward j.
            a[2 * mem.i][col] += ux;
            a[2 * mem.i + 1][col] += uy;
            // Node j: −u toward i.
            a[2 * mem.j][col] -= ux;
            a[2 * mem.j + 1][col] -= uy;
        }

        // Reaction columns.
        for (k, (node, rx, ry)) in reaction_cols.iter().enumerate() {
            let col = m + k;
            a[2 * node][col] += rx;
            a[2 * node + 1][col] += ry;
        }

        // RHS: member + reaction terms balance the applied loads → = −external.
        for load in &self.loads {
            b[2 * load.node] -= load.fx;
            b[2 * load.node + 1] -= load.fy;
        }

        let x = solve_linear_system(a, b).ok_or(TrussError::Singular)?;

        let member_forces = x[..m].to_vec();
        // Fold reaction components back to per-node (fx, fy).
        let mut reactions: Vec<Reaction> = Vec::new();
        for (k, (node, rx, ry)) in reaction_cols.iter().enumerate() {
            let val = x[m + k];
            if let Some(r) = reactions.iter_mut().find(|r| r.node == *node) {
                r.fx += rx * val;
                r.fy += ry * val;
            } else {
                reactions.push(Reaction {
                    node: *node,
                    fx: rx * val,
                    fy: ry * val,
                });
            }
        }

        Ok(TrussSolution {
            member_forces,
            reactions,
        })
    }
}

/// Solve a square linear system `A x = b` by Gaussian elimination with partial
/// pivoting. Returns `None` if the matrix is singular.
fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    if a.len() != n || a.iter().any(|row| row.len() != n) {
        return None;
    }
    for col in 0..n {
        // Partial pivot.
        let mut pivot = col;
        let mut best = a[col][col].abs();
        for r in (col + 1)..n {
            if a[r][col].abs() > best {
                best = a[r][col].abs();
                pivot = r;
            }
        }
        if best < 1e-12 {
            return None; // singular
        }
        a.swap(col, pivot);
        b.swap(col, pivot);

        // Eliminate below.
        for r in (col + 1)..n {
            let factor = a[r][col] / a[col][col];
            if factor != 0.0 {
                for c in col..n {
                    a[r][c] -= factor * a[col][c];
                }
                b[r] -= factor * b[col];
            }
        }
    }

    // Back-substitution.
    let mut x = vec![0.0f64; n];
    for row in (0..n).rev() {
        let mut sum = b[row];
        for c in (row + 1)..n {
            sum -= a[row][c] * x[c];
        }
        x[row] = sum / a[row][row];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Symmetric triangular truss: A(0,0) pin, B(4,0) roller, C(2,3) apex with
    /// a 10 N downward load. By symmetry reactions are 5 N each; hand solution:
    /// bottom chord AB = +3.333 (tension), the two inclined members = −6.009
    /// (compression).
    fn triangle() -> Truss {
        Truss {
            nodes: vec![
                Node::pin(0.0, 0.0),
                Node::roller_vertical(4.0, 0.0),
                Node::free(2.0, 3.0),
            ],
            members: vec![
                Member { i: 0, j: 1 }, // AB
                Member { i: 0, j: 2 }, // AC
                Member { i: 1, j: 2 }, // BC
            ],
            loads: vec![Load {
                node: 2,
                fx: 0.0,
                fy: -10.0,
            }],
        }
    }

    #[test]
    fn triangular_truss_member_forces() {
        let sol = triangle().solve().unwrap();
        assert!(
            (sol.member_forces[0] - 3.3333).abs() < 1e-3,
            "AB={}",
            sol.member_forces[0]
        );
        assert!(
            (sol.member_forces[1] + 6.0093).abs() < 1e-3,
            "AC={}",
            sol.member_forces[1]
        );
        assert!(
            (sol.member_forces[2] + 6.0093).abs() < 1e-3,
            "BC={}",
            sol.member_forces[2]
        );
    }

    #[test]
    fn triangular_truss_reactions_balance_load() {
        let sol = triangle().solve().unwrap();
        let total_ry: f64 = sol.reactions.iter().map(|r| r.fy).sum();
        let total_rx: f64 = sol.reactions.iter().map(|r| r.fx).sum();
        assert!((total_ry - 10.0).abs() < 1e-6, "ΣRy={total_ry}"); // balances 10 N down
        assert!(total_rx.abs() < 1e-6, "ΣRx={total_rx}");
        // Symmetry: both vertical reactions ≈ 5 N.
        for r in &sol.reactions {
            assert!((r.fy - 5.0).abs() < 1e-6);
        }
    }

    #[test]
    fn tension_compression_signs() {
        let sol = triangle().solve().unwrap();
        assert!(sol.member_forces[0] > 0.0, "bottom chord should be tension");
        assert!(
            sol.member_forces[1] < 0.0,
            "inclined members should be compression"
        );
    }

    #[test]
    fn indeterminate_truss_is_rejected() {
        // Add an extra member to the determinate triangle → over-constrained.
        let mut t = triangle();
        t.members.push(Member { i: 0, j: 1 });
        assert!(matches!(t.solve(), Err(TrussError::NotDeterminate { .. })));
    }

    #[test]
    fn linear_solver_basic() {
        // 2x + y = 5 ; x + 3y = 10  → x=1, y=3.
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let b = vec![5.0, 10.0];
        let x = solve_linear_system(a, b).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-12);
        assert!((x[1] - 3.0).abs() < 1e-12);
    }
}
