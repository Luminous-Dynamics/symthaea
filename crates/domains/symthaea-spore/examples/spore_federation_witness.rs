//! SPORE-FED-001B — deterministic semantic federation witness.
//!
//! This example intentionally models *semantics*, not a second transport stack.
//! It is an executable oracle for the showcase event vocabulary.
//!
//! Claim ceiling: deterministic software/network semantics only.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum Lane {
    Observation,
    Analysis,
    Recommendation,
    Authorization,
    Execution,
    Capability,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum Disposition {
    Applied,
    Queued,
    Rejected(&'static str),
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Event {
    seq: u64,
    time: u64,
    actor: &'static str,
    subject: &'static str,
    source_generation: u64,
    lane: Lane,
    payload: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Node {
    id: &'static str,
    identity_generation: u64,
    capability_generation: u64,
    capabilities: BTreeSet<&'static str>,
    authority_epoch: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Link {
    open: bool,
    delay: u64,
    expiry: u64,
    queue_limit: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Delivery {
    deliver_at: u64,
    target: &'static str,
    event: Event,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Witness {
    clock: u64,
    nodes: BTreeMap<&'static str, Node>,
    link: Link,
    queue: VecDeque<Delivery>,
    ledger: Vec<(Event, Disposition)>,
    next_seq: u64,
}

impl Witness {
    fn new() -> Self {
        let mut nodes = BTreeMap::new();
        for id in ["EARTH-A", "EARTH-B", "MARS"] {
            nodes.insert(id, Node {
                id,
                identity_generation: 1,
                capability_generation: 1,
                capabilities: BTreeSet::new(),
                authority_epoch: 1,
            });
        }
        Self {
            clock: 0,
            nodes,
            link: Link { open: true, delay: 0, expiry: 100, queue_limit: 8 },
            queue: VecDeque::new(),
            ledger: Vec::new(),
            next_seq: 1,
        }
    }

    fn emit(
        &mut self,
        actor: &'static str,
        subject: &'static str,
        source_generation: u64,
        lane: Lane,
        payload: &'static str,
    ) -> Event {
        let e = Event {
            seq: self.next_seq,
            time: self.clock,
            actor,
            subject,
            source_generation,
            lane,
            payload,
        };
        self.next_seq += 1;
        e
    }

    fn share_capability(&mut self, from: &'static str, to: &'static str, capability: &'static str) {
        let source_generation = self.nodes[from].capability_generation;
        let e = self.emit(from, to, source_generation, Lane::Capability, capability);
        self.deliver_or_queue(to, e);
    }

    fn deliver_or_queue(&mut self, target: &'static str, e: Event) {
        if !self.link.open {
            if self.queue.len() >= self.link.queue_limit {
                self.ledger.push((e, Disposition::Rejected("bounded-queue")));
            } else {
                self.queue.push_back(Delivery {
                    deliver_at: self.clock + self.link.delay,
                    target,
                    event: e,
                });
                self.ledger.push((self.queue.back().unwrap().event.clone(), Disposition::Queued));
            }
            return;
        }
        self.apply(target, e);
    }

    fn apply(&mut self, target: &'static str, e: Event) {
        let disposition = match e.lane {
            Lane::Capability => {
                if e.source_generation != self.nodes[e.actor].capability_generation {
                    Disposition::Rejected("stale-capability-generation")
                } else {
                    self.nodes.get_mut(target).unwrap().capabilities.insert(e.payload);
                    Disposition::Applied
                }
            }
            Lane::Authorization => Disposition::Rejected("authorization-boundary"),
            _ => Disposition::Applied,
        };
        self.ledger.push((e, disposition));
    }

    fn partition(&mut self, delay: u64) {
        self.link.open = false;
        self.link.delay = delay;
        self.ledger.push((
            self.emit("EARTH-A", "MARS", 0, Lane::Observation, "partition"),
            Disposition::Applied,
        ));
    }

    fn heal(&mut self) {
        self.link.open = true;
        while let Some(d) = self.queue.pop_front() {
            if d.event.time + self.link.expiry < self.clock {
                self.ledger.push((d.event, Disposition::Rejected("expired")));
            } else {
                self.apply(d.target, d.event);
            }
        }
    }

    fn advance(&mut self, ticks: u64) {
        self.clock += ticks;
    }

    fn analysis_cannot_authorize(&mut self) {
        let e = self.emit(
            "EARTH-A", "MARS", self.nodes["EARTH-A"].capability_generation,
            Lane::Authorization, "analysis-result",
        );
        self.apply("MARS", e);
    }
}

fn run() -> Witness {
    let mut w = Witness::new();

    // S0–S2: germination, federation, capability bootstrap.
    w.nodes.get_mut("EARTH-A").unwrap().capabilities.insert("local-analysis");
    w.share_capability("EARTH-A", "EARTH-B", "local-analysis");

    // S3: an authorization-looking payload must not cross the authority boundary.
    w.analysis_cannot_authorize();

    // S4–S5: partition + deterministic delayed queue.
    w.partition(20);
    w.share_capability("EARTH-A", "MARS", "local-analysis");
    w.advance(5);

    // Capability generation changes while the link is partitioned.
    w.nodes.get_mut("EARTH-A").unwrap().capability_generation += 1;
    w.share_capability("EARTH-A", "MARS", "new-capability");

    // S6: heal makes queued events visible; old-generation event is rejected.
    w.heal();

    w
}

fn main() {
    let w = run();
    let applied = w.ledger.iter().filter(|(_, d)| *d == Disposition::Applied).count();
    let queued = w.ledger.iter().filter(|(_, d)| *d == Disposition::Queued).count();
    let rejected = w.ledger.iter().filter(|(_, d)| matches!(d, Disposition::Rejected(_))).count();

    println!("SPORE-FED-001B");
    println!("events={}", w.ledger.len());
    println!("applied={applied}");
    println!("queued={queued}");
    println!("rejected={rejected}");
    println!("identity_earth_a={}", w.nodes["EARTH-A"].identity_generation);
    println!("identity_mars={}", w.nodes["MARS"].identity_generation);
    println!("mars_capabilities={:?}", w.nodes["MARS"].capabilities);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_does_not_copy_identity() {
        let w = run();
        assert_ne!(w.nodes["EARTH-A"].id, w.nodes["EARTH-B"].id);
        assert_eq!(w.nodes["EARTH-A"].identity_generation, 1);
        assert_eq!(w.nodes["EARTH-B"].identity_generation, 1);
    }

    #[test]
    fn analysis_cannot_become_authorization() {
        let w = run();
        assert!(w.ledger.iter().any(|(_, d)| *d == Disposition::Rejected("authorization-boundary")));
    }

    #[test]
    fn generation_change_rejects_stale_capability() {
        let w = run();
        assert!(w.ledger.iter().any(|(e, d)| {
            e.lane == Lane::Capability && *d == Disposition::Rejected("stale-capability-generation")
        }));
    }

    #[test]
    fn partition_is_bounded_and_visible() {
        let w = run();
        assert!(w.ledger.iter().any(|(_, d)| *d == Disposition::Queued));
    }

    #[test]
    fn replay_is_deterministic() {
        assert_eq!(run(), run());
    }

    #[test]
    fn authority_is_not_derived_from_capabilities() {
        let w = run();
        assert_eq!(w.nodes["EARTH-A"].authority_epoch, 1);
        assert_eq!(w.nodes["EARTH-B"].authority_epoch, 1);
        assert_eq!(w.nodes["MARS"].authority_epoch, 1);
    }

    #[test]
    fn no_physical_claim_is_encoded() {
        let w = run();
        assert!(w.ledger.iter().all(|(e, _)| e.payload != "physical-link-proven"));
    }
}
