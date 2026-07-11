// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Ranked-ballot voting rules. A **ballot** is a complete ranking of the
//! candidates, best first (`vec!["alice", "bob", "carol"]`). Ties are broken
//! deterministically in favour of the alphabetically-first candidate, so every
//! rule returns a definite winner where one is defined.

use std::collections::{BTreeSet, HashMap};

type Ballot = Vec<String>;

/// The sorted set of candidates appearing on any ballot.
pub fn candidates(ballots: &[Ballot]) -> Vec<String> {
    let mut set: BTreeSet<String> = BTreeSet::new();
    for b in ballots {
        for c in b {
            set.insert(c.clone());
        }
    }
    set.into_iter().collect()
}

/// First-preference tally.
pub fn first_choice_tally(ballots: &[Ballot]) -> HashMap<String, usize> {
    let mut t = HashMap::new();
    for b in ballots {
        if let Some(first) = b.first() {
            *t.entry(first.clone()).or_insert(0) += 1;
        }
    }
    t
}

/// Pick the candidate with the greatest score, breaking ties alphabetically.
fn arg_max(cands: &[String], score: impl Fn(&str) -> f64) -> Option<String> {
    let mut best: Option<(String, f64)> = None;
    for c in cands {
        let s = score(c);
        match &best {
            Some((_, bs)) if *bs >= s => {}
            _ => best = Some((c.clone(), s)),
        }
    }
    best.map(|(c, _)| c)
}

/// Plurality (first-past-the-post): most first-preferences wins.
pub fn plurality_winner(ballots: &[Ballot]) -> Option<String> {
    if ballots.is_empty() {
        return None;
    }
    let tally = first_choice_tally(ballots);
    arg_max(&candidates(ballots), |c| {
        tally.get(c).copied().unwrap_or(0) as f64
    })
}

/// Borda count: on a ballot ranking `m` candidates, rank `i` (0-based) scores
/// `m − 1 − i` points. Highest total wins. Assumes complete rankings.
pub fn borda_winner(ballots: &[Ballot]) -> Option<String> {
    if ballots.is_empty() {
        return None;
    }
    let cands = candidates(ballots);
    let m = cands.len();
    let mut score: HashMap<String, usize> = HashMap::new();
    for b in ballots {
        for (i, c) in b.iter().enumerate() {
            *score.entry(c.clone()).or_insert(0) += m.saturating_sub(1 + i);
        }
    }
    arg_max(&cands, |c| score.get(c).copied().unwrap_or(0) as f64)
}

/// How many ballots rank `x` above `y` (both must appear on the ballot).
pub fn pairwise(ballots: &[Ballot], x: &str, y: &str) -> usize {
    ballots
        .iter()
        .filter(|b| {
            let px = b.iter().position(|c| c == x);
            let py = b.iter().position(|c| c == y);
            matches!((px, py), (Some(a), Some(b)) if a < b)
        })
        .count()
}

/// The Condorcet winner — the candidate who beats every other in a pairwise
/// majority — if one exists (it need not).
pub fn condorcet_winner(ballots: &[Ballot]) -> Option<String> {
    let cands = candidates(ballots);
    'outer: for x in &cands {
        for y in &cands {
            if x == y {
                continue;
            }
            if pairwise(ballots, x, y) <= pairwise(ballots, y, x) {
                continue 'outer;
            }
        }
        return Some(x.clone());
    }
    None
}

/// Instant-runoff (ranked-choice): repeatedly eliminate the candidate with the
/// fewest first-preferences among those still active until one holds a
/// majority. Ties for elimination broken alphabetically.
pub fn instant_runoff_winner(ballots: &[Ballot]) -> Option<String> {
    if ballots.is_empty() {
        return None;
    }
    let mut active: BTreeSet<String> = candidates(ballots).into_iter().collect();
    loop {
        // Tally each ballot's top *active* candidate.
        let mut tally: HashMap<String, usize> = HashMap::new();
        for b in ballots {
            if let Some(top) = b.iter().find(|c| active.contains(*c)) {
                *tally.entry(top.clone()).or_insert(0) += 1;
            }
        }
        let total: usize = tally.values().sum();
        if total == 0 {
            return active.into_iter().next();
        }
        // Majority check.
        for (c, &v) in &tally {
            if v * 2 > total {
                return Some(c.clone());
            }
        }
        if active.len() <= 1 {
            return active.into_iter().next();
        }
        // Eliminate the alphabetically-first candidate with the fewest votes.
        let min = active
            .iter()
            .map(|c| tally.get(c).copied().unwrap_or(0))
            .min()
            .unwrap();
        let victim = active
            .iter()
            .find(|c| tally.get(*c).copied().unwrap_or(0) == min)
            .cloned()
            .unwrap();
        active.remove(&victim);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(order: &[&str]) -> Ballot {
        order.iter().map(|s| s.to_string()).collect()
    }

    /// Classic profile where plurality and the other rules disagree.
    /// 4× a>b>c, 3× b>c>a, 2× c>b>a  (9 voters). `b` is the compromise:
    /// plurality picks `a`, but Condorcet/Borda/IRV all pick `b`.
    fn split_profile() -> Vec<Ballot> {
        let mut v = Vec::new();
        for _ in 0..4 {
            v.push(b(&["a", "b", "c"]));
        }
        for _ in 0..3 {
            v.push(b(&["b", "c", "a"]));
        }
        for _ in 0..2 {
            v.push(b(&["c", "b", "a"]));
        }
        v
    }

    #[test]
    fn plurality_picks_most_firsts() {
        // a=4 firsts, b=3, c=2 → a.
        assert_eq!(plurality_winner(&split_profile()), Some("a".to_string()));
    }

    #[test]
    fn condorcet_beats_plurality_winner() {
        // b beats a (5–4) and b beats c (7–2) → b is the Condorcet winner,
        // even though a wins plurality.
        assert_eq!(condorcet_winner(&split_profile()), Some("b".to_string()));
    }

    #[test]
    fn borda_also_favours_the_compromise() {
        // Borda (2/1/0): a=8, b=12, c=7 → b.
        assert_eq!(borda_winner(&split_profile()), Some("b".to_string()));
    }

    #[test]
    fn irv_eliminates_then_finds_majority() {
        // c (2, unique lowest) is eliminated; its votes flow to b →
        // b has 5 of 9 → majority → b.
        assert_eq!(
            instant_runoff_winner(&split_profile()),
            Some("b".to_string())
        );
    }

    #[test]
    fn condorcet_cycle_has_no_winner() {
        // Condorcet paradox: a>b>c, b>c>a, c>a>b — a cycle, no winner.
        let v = vec![
            b(&["a", "b", "c"]),
            b(&["b", "c", "a"]),
            b(&["c", "a", "b"]),
        ];
        assert_eq!(condorcet_winner(&v), None);
    }
}
