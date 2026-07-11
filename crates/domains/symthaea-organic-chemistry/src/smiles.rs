// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! SMILES parser and molecular-graph model.
//!
//! Supports the common organic subset of SMILES (Daylight/OpenSMILES):
//! - organic-subset atoms `B C N O P S F Cl Br I` (implicit hydrogens derived)
//! - bracket atoms `[...]` with explicit element, hydrogen count, and charge
//! - bonds `-` `=` `#` `:` (single/double/triple/aromatic), implicit single
//! - branches `( ... )`
//! - ring-closure digits `1`–`9` and `%nn`
//! - lowercase aromatic atoms `c n o s p`
//!
//! Out of v0.1 scope: isotopes, stereochemistry (`/ \ @`), disconnected
//! structures (`.`), and radicals. These parse-or-error explicitly rather than
//! silently mis-modelling.

use crate::element;

/// Bond multiplicity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BondOrder {
    Single,
    Double,
    Triple,
    Aromatic,
}

impl BondOrder {
    /// Numeric valence contribution. Aromatic bonds contribute 1.5 (Kekulé
    /// average), used for implicit-H derivation on aromatic atoms.
    pub fn valence_contribution(self) -> f64 {
        match self {
            BondOrder::Single => 1.0,
            BondOrder::Double => 2.0,
            BondOrder::Triple => 3.0,
            BondOrder::Aromatic => 1.5,
        }
    }
}

/// A heavy atom (never hydrogen) in the molecular graph.
#[derive(Debug, Clone, PartialEq)]
pub struct Atom {
    /// Element symbol (proper casing), e.g. `"C"`, `"Cl"`.
    pub element: &'static str,
    /// Aromatic flag (written lowercase or `:`-bonded).
    pub aromatic: bool,
    /// Formal charge.
    pub charge: i8,
    /// Attached hydrogens (implicit-derived or bracket-explicit).
    pub hydrogens: u8,
}

/// An edge between two heavy atoms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bond {
    pub a: usize,
    pub b: usize,
    pub order: BondOrder,
}

/// A parsed molecule: heavy-atom graph plus per-atom hydrogen counts.
#[derive(Debug, Clone, PartialEq)]
pub struct Molecule {
    pub atoms: Vec<Atom>,
    pub bonds: Vec<Bond>,
}

/// A SMILES parse error with a human-readable reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError(pub String);

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SMILES parse error: {}", self.0)
    }
}

impl std::error::Error for ParseError {}

impl Molecule {
    /// Parse a SMILES string into a molecular graph.
    pub fn from_smiles(smiles: &str) -> Result<Molecule, ParseError> {
        Parser::new(smiles).parse()
    }

    /// Neighbours of atom `i` as `(other_index, bond_order)`.
    pub fn neighbors(&self, i: usize) -> Vec<(usize, BondOrder)> {
        let mut out = Vec::new();
        for b in &self.bonds {
            if b.a == i {
                out.push((b.b, b.order));
            } else if b.b == i {
                out.push((b.a, b.order));
            }
        }
        out
    }

    /// Total atom count (heavy atoms + all hydrogens).
    pub fn total_atom_count(&self) -> usize {
        self.atoms.len()
            + self
                .atoms
                .iter()
                .map(|a| a.hydrogens as usize)
                .sum::<usize>()
    }

    /// Element → count list (insertion order), hydrogens folded in.
    fn element_counts(&self) -> Vec<(&'static str, usize)> {
        let mut counts: Vec<(&'static str, usize)> = Vec::new();
        let mut bump = |sym: &'static str, n: usize| {
            if n == 0 {
                return;
            }
            if let Some(entry) = counts.iter_mut().find(|(s, _)| *s == sym) {
                entry.1 += n;
            } else {
                counts.push((sym, n));
            }
        };
        for a in &self.atoms {
            bump(a.element, 1);
        }
        let h: usize = self.atoms.iter().map(|a| a.hydrogens as usize).sum();
        bump("H", h);
        counts
    }

    /// Molecular formula in Hill notation (C first, H second, then alphabetical).
    pub fn molecular_formula(&self) -> String {
        let counts = self.element_counts();
        let has_carbon = counts.iter().any(|(s, _)| *s == "C");

        let mut ordered: Vec<(&'static str, usize)> = Vec::new();
        if has_carbon {
            if let Some(c) = counts.iter().find(|(s, _)| *s == "C") {
                ordered.push(*c);
            }
            if let Some(h) = counts.iter().find(|(s, _)| *s == "H") {
                ordered.push(*h);
            }
            let mut rest: Vec<_> = counts
                .iter()
                .filter(|(s, _)| *s != "C" && *s != "H")
                .copied()
                .collect();
            rest.sort_by(|a, b| a.0.cmp(b.0));
            ordered.extend(rest);
        } else {
            ordered = counts.clone();
            ordered.sort_by(|a, b| a.0.cmp(b.0));
        }

        let mut s = String::new();
        for (sym, n) in ordered {
            s.push_str(sym);
            if n > 1 {
                s.push_str(&n.to_string());
            }
        }
        s
    }

    /// Molecular weight (g/mol) from standard atomic weights.
    pub fn molecular_weight(&self) -> f64 {
        self.element_counts()
            .iter()
            .map(|(sym, n)| element::lookup(sym).map(|e| e.weight).unwrap_or(0.0) * *n as f64)
            .sum()
    }
}

/// SMILES parser over a char buffer.
struct Parser {
    chars: Vec<char>,
    pos: usize,
    atoms: Vec<Atom>,
    bonds: Vec<Bond>,
    /// Indices of atoms that came from bracket syntax; their H count is
    /// authoritative and must not be overwritten by implicit-H derivation.
    bracketed: Vec<usize>,
    /// Ring-closure bookkeeping: digit → (atom index, pending bond order).
    ring_bonds: Vec<Option<(usize, Option<BondOrder>)>>,
}

impl Parser {
    fn new(smiles: &str) -> Self {
        Parser {
            chars: smiles.chars().collect(),
            pos: 0,
            atoms: Vec::new(),
            bonds: Vec::new(),
            bracketed: Vec::new(),
            ring_bonds: vec![None; 100],
        }
    }

    fn parse(mut self) -> Result<Molecule, ParseError> {
        if self.chars.is_empty() {
            return Err(ParseError("empty input".into()));
        }
        let mut branch_stack: Vec<usize> = Vec::new();
        let mut prev: Option<usize> = None;
        let mut pending_bond: Option<BondOrder> = None;

        while self.pos < self.chars.len() {
            let c = self.chars[self.pos];
            match c {
                '(' => {
                    let p = prev.ok_or_else(|| ParseError("'(' before any atom".into()))?;
                    branch_stack.push(p);
                    self.pos += 1;
                }
                ')' => {
                    prev = Some(
                        branch_stack
                            .pop()
                            .ok_or_else(|| ParseError("unbalanced ')'".into()))?,
                    );
                    self.pos += 1;
                }
                '-' => {
                    pending_bond = Some(BondOrder::Single);
                    self.pos += 1;
                }
                '=' => {
                    pending_bond = Some(BondOrder::Double);
                    self.pos += 1;
                }
                '#' => {
                    pending_bond = Some(BondOrder::Triple);
                    self.pos += 1;
                }
                ':' => {
                    pending_bond = Some(BondOrder::Aromatic);
                    self.pos += 1;
                }
                '.' | '/' | '\\' | '@' => {
                    return Err(ParseError(format!(
                        "unsupported SMILES feature '{c}' (v0.1 scope: no stereo/disconnected/isotope)"
                    )));
                }
                d if d.is_ascii_digit() || d == '%' => {
                    let ring_num = self.read_ring_number()?;
                    let cur =
                        prev.ok_or_else(|| ParseError("ring closure before any atom".into()))?;
                    self.close_ring(ring_num, cur, pending_bond)?;
                    pending_bond = None;
                }
                '[' => {
                    let atom = self.read_bracket_atom()?;
                    let idx = self.push_atom(atom);
                    self.bracketed.push(idx);
                    self.link(prev, idx, &mut pending_bond)?;
                    prev = Some(idx);
                }
                _ => {
                    let atom = self.read_organic_atom()?;
                    let idx = self.push_atom(atom);
                    self.link(prev, idx, &mut pending_bond)?;
                    prev = Some(idx);
                }
            }
        }

        if !branch_stack.is_empty() {
            return Err(ParseError("unbalanced '(' — missing ')'".into()));
        }
        if let Some(i) = self.ring_bonds.iter().position(|r| r.is_some()) {
            return Err(ParseError(format!("unclosed ring bond {i}")));
        }

        self.assign_implicit_hydrogens();
        Ok(Molecule {
            atoms: self.atoms,
            bonds: self.bonds,
        })
    }

    fn push_atom(&mut self, atom: Atom) -> usize {
        self.atoms.push(atom);
        self.atoms.len() - 1
    }

    /// Add a bond from `prev` (if any) to `cur`, honoring a pending explicit
    /// bond order (default single, or aromatic if both atoms are aromatic).
    fn link(
        &mut self,
        prev: Option<usize>,
        cur: usize,
        pending: &mut Option<BondOrder>,
    ) -> Result<(), ParseError> {
        if let Some(p) = prev {
            let order = pending.take().unwrap_or_else(|| {
                if self.atoms[p].aromatic && self.atoms[cur].aromatic {
                    BondOrder::Aromatic
                } else {
                    BondOrder::Single
                }
            });
            self.bonds.push(Bond {
                a: p,
                b: cur,
                order,
            });
        } else if pending.is_some() {
            return Err(ParseError("bond symbol before any atom".into()));
        }
        Ok(())
    }

    fn read_ring_number(&mut self) -> Result<usize, ParseError> {
        let c = self.chars[self.pos];
        if c == '%' {
            let d1 = self.chars.get(self.pos + 1).copied();
            let d2 = self.chars.get(self.pos + 2).copied();
            match (d1, d2) {
                (Some(a), Some(b)) if a.is_ascii_digit() && b.is_ascii_digit() => {
                    self.pos += 3;
                    Ok((a.to_digit(10).unwrap() * 10 + b.to_digit(10).unwrap()) as usize)
                }
                _ => Err(ParseError("'%' ring closure needs two digits".into())),
            }
        } else {
            self.pos += 1;
            Ok(c.to_digit(10).unwrap() as usize)
        }
    }

    fn close_ring(
        &mut self,
        ring_num: usize,
        cur: usize,
        pending: Option<BondOrder>,
    ) -> Result<(), ParseError> {
        if ring_num >= self.ring_bonds.len() {
            return Err(ParseError(format!("ring number {ring_num} out of range")));
        }
        match self.ring_bonds[ring_num].take() {
            None => {
                self.ring_bonds[ring_num] = Some((cur, pending));
                Ok(())
            }
            Some((open_atom, open_order)) => {
                let order = pending.or(open_order).unwrap_or_else(|| {
                    if self.atoms[open_atom].aromatic && self.atoms[cur].aromatic {
                        BondOrder::Aromatic
                    } else {
                        BondOrder::Single
                    }
                });
                self.bonds.push(Bond {
                    a: open_atom,
                    b: cur,
                    order,
                });
                Ok(())
            }
        }
    }

    /// Read an organic-subset atom (possibly two-letter `Cl`/`Br`, possibly
    /// lowercase aromatic).
    fn read_organic_atom(&mut self) -> Result<Atom, ParseError> {
        let c = self.chars[self.pos];
        let two: String = self.chars[self.pos..].iter().take(2).collect();
        if two == "Cl" || two == "Br" {
            self.pos += 2;
            return Ok(Atom {
                element: element::lookup(&two).unwrap().symbol,
                aromatic: false,
                charge: 0,
                hydrogens: 0,
            });
        }
        let (sym, aromatic) = match c {
            'B' | 'C' | 'N' | 'O' | 'P' | 'S' | 'F' | 'I' => (c.to_string(), false),
            'b' | 'c' | 'n' | 'o' | 'p' | 's' => (c.to_ascii_uppercase().to_string(), true),
            other => {
                return Err(ParseError(format!(
                    "unexpected character '{other}' (not an organic-subset atom; bracket it as [..])"
                )));
            }
        };
        let el =
            element::lookup(&sym).ok_or_else(|| ParseError(format!("unknown element '{sym}'")))?;
        self.pos += 1;
        Ok(Atom {
            element: el.symbol,
            aromatic,
            charge: 0,
            hydrogens: 0,
        })
    }

    /// Read a bracket atom `[<element><Hn><charge>]` with explicit hydrogens.
    fn read_bracket_atom(&mut self) -> Result<Atom, ParseError> {
        self.pos += 1; // consume '['
        if self.chars.get(self.pos).is_some_and(|c| c.is_ascii_digit()) {
            return Err(ParseError("isotopes unsupported in v0.1".into()));
        }
        let first = *self
            .chars
            .get(self.pos)
            .ok_or_else(|| ParseError("unterminated '['".into()))?;
        let aromatic = first.is_ascii_lowercase();
        let mut sym = String::new();
        sym.push(first.to_ascii_uppercase());
        self.pos += 1;
        if let Some(&second) = self.chars.get(self.pos) {
            if second.is_ascii_lowercase() && !aromatic {
                let candidate = format!("{sym}{second}");
                if element::lookup(&candidate).is_some() {
                    sym = candidate;
                    self.pos += 1;
                }
            }
        }
        let el = element::lookup(&sym)
            .ok_or_else(|| ParseError(format!("unknown bracket element '{sym}'")))?;

        let mut hydrogens: u8 = 0;
        if self.chars.get(self.pos) == Some(&'H') {
            self.pos += 1;
            hydrogens = 1;
            if let Some(&d) = self.chars.get(self.pos) {
                if d.is_ascii_digit() {
                    hydrogens = d.to_digit(10).unwrap() as u8;
                    self.pos += 1;
                }
            }
        }

        let mut charge: i8 = 0;
        while let Some(&c) = self.chars.get(self.pos) {
            match c {
                '+' => {
                    charge += 1;
                    self.pos += 1;
                }
                '-' => {
                    charge -= 1;
                    self.pos += 1;
                }
                d if d.is_ascii_digit() && charge != 0 => {
                    let mag = d.to_digit(10).unwrap() as i8;
                    charge = charge.signum() * mag;
                    self.pos += 1;
                }
                _ => break,
            }
        }

        if self.chars.get(self.pos) != Some(&']') {
            return Err(ParseError(
                "unterminated bracket atom (expected ']')".into(),
            ));
        }
        self.pos += 1;

        Ok(Atom {
            element: el.symbol,
            aromatic,
            charge,
            hydrogens,
        })
    }

    /// Derive implicit hydrogens for non-bracket, neutral organic-subset atoms.
    /// Bracket atoms carry authoritative explicit counts and are skipped.
    fn assign_implicit_hydrogens(&mut self) {
        let mut bond_sum = vec![0.0f64; self.atoms.len()];
        for b in &self.bonds {
            let c = b.order.valence_contribution();
            bond_sum[b.a] += c;
            bond_sum[b.b] += c;
        }
        let is_bracketed: Vec<bool> = (0..self.atoms.len())
            .map(|i| self.bracketed.contains(&i))
            .collect();
        for (i, atom) in self.atoms.iter_mut().enumerate() {
            if is_bracketed[i] {
                continue;
            }
            if atom.charge != 0 || !element::is_organic_subset(atom.element) {
                continue;
            }
            let valence = element::lookup(atom.element).unwrap().normal_valence as f64;
            let implicit = (valence - bond_sum[i]).round();
            if implicit > 0.0 {
                atom.hydrogens = implicit as u8;
            }
        }
    }
}
