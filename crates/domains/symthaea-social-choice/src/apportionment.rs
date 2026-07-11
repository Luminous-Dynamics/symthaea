// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Seat apportionment: turning vote (or population) counts into a whole number
//! of seats. Two families:
//! - **Highest averages** (D'Hondt, Sainte-Laguë) — assign seats one at a time
//!   to the party with the largest votes ÷ divisor.
//! - **Largest remainder** (Hamilton) — give each party its integer quota, then
//!   hand out leftover seats by fractional remainder.
//!
//! Every function returns allocations in the input order and totalling exactly
//! `seats`.

/// Generic highest-averages method: `divisor(s)` is the divisor applied to a
/// party currently holding `s` seats. Ties go to the earlier party.
fn highest_averages(
    votes: &[(String, u64)],
    seats: u64,
    divisor: impl Fn(u64) -> f64,
) -> Vec<(String, u64)> {
    let mut alloc = vec![0u64; votes.len()];
    for _ in 0..seats {
        let mut best = 0usize;
        let mut best_q = f64::NEG_INFINITY;
        for (i, (_, v)) in votes.iter().enumerate() {
            let q = *v as f64 / divisor(alloc[i]);
            if q > best_q {
                best_q = q;
                best = i;
            }
        }
        alloc[best] += 1;
    }
    votes
        .iter()
        .zip(alloc)
        .map(|((name, _), s)| (name.clone(), s))
        .collect()
}

/// D'Hondt / Jefferson (divisors 1, 2, 3, …). Mildly favours larger parties.
pub fn dhondt(votes: &[(String, u64)], seats: u64) -> Vec<(String, u64)> {
    highest_averages(votes, seats, |s| (s + 1) as f64)
}

/// Sainte-Laguë / Webster (divisors 1, 3, 5, …). More proportional to small
/// parties than D'Hondt.
pub fn sainte_lague(votes: &[(String, u64)], seats: u64) -> Vec<(String, u64)> {
    highest_averages(votes, seats, |s| (2 * s + 1) as f64)
}

/// Hamilton / largest-remainder method. Each party first gets ⌊quota⌋ seats,
/// then the remaining seats go to the largest fractional remainders (ties to
/// the earlier party).
pub fn hamilton(votes: &[(String, u64)], seats: u64) -> Vec<(String, u64)> {
    let total: u64 = votes.iter().map(|(_, v)| v).sum();
    if total == 0 || seats == 0 {
        return votes.iter().map(|(n, _)| (n.clone(), 0)).collect();
    }
    let mut alloc: Vec<u64> = Vec::with_capacity(votes.len());
    let mut remainders: Vec<(f64, usize)> = Vec::with_capacity(votes.len());
    for (i, (_, v)) in votes.iter().enumerate() {
        let exact = *v as f64 * seats as f64 / total as f64;
        let base = exact.floor();
        alloc.push(base as u64);
        remainders.push((exact - base, i));
    }
    let assigned: u64 = alloc.iter().sum();
    let mut left = seats - assigned;
    // Largest remainder first; ties broken by earlier index.
    remainders.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
    for &(_, i) in &remainders {
        if left == 0 {
            break;
        }
        alloc[i] += 1;
        left -= 1;
    }
    votes
        .iter()
        .zip(alloc)
        .map(|((name, _), s)| (name.clone(), s))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> Vec<(String, u64)> {
        vec![
            ("A".into(), 100),
            ("B".into(), 80),
            ("C".into(), 30),
            ("D".into(), 20),
        ]
    }

    fn seats_of(r: &[(String, u64)]) -> Vec<u64> {
        r.iter().map(|(_, s)| *s).collect()
    }

    #[test]
    fn dhondt_classic() {
        // Textbook D'Hondt for 100/80/30/20 over 8 seats → 4/3/1/0.
        let r = dhondt(&profile(), 8);
        assert_eq!(seats_of(&r), vec![4, 3, 1, 0]);
        assert_eq!(seats_of(&r).iter().sum::<u64>(), 8);
    }

    #[test]
    fn sainte_lague_helps_small_party() {
        // Same votes/seats: Sainte-Laguë gives D a seat that D'Hondt denied.
        let r = sainte_lague(&profile(), 8);
        assert_eq!(seats_of(&r), vec![3, 3, 1, 1]);
    }

    #[test]
    fn hamilton_largest_remainder() {
        // base 3/2/1/0 (sum 6), remainders favour B then D → 3/3/1/1.
        let r = hamilton(&profile(), 8);
        assert_eq!(seats_of(&r), vec![3, 3, 1, 1]);
        assert_eq!(seats_of(&r).iter().sum::<u64>(), 8);
    }

    #[test]
    fn all_methods_conserve_seats() {
        for seats in [1, 5, 10, 13] {
            for r in [
                dhondt(&profile(), seats),
                sainte_lague(&profile(), seats),
                hamilton(&profile(), seats),
            ] {
                assert_eq!(seats_of(&r).iter().sum::<u64>(), seats);
            }
        }
    }
}
