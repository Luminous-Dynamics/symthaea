use super::{
    AgentInstanceId, AgentNode, CommunicationGraphProfile, GeneratorProvenance, MessageChannel,
    MessageClass, PrincipalId, TopologyFamily, ValidationErrors, GRAPH_SCHEMA_VERSION,
};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

const RNG_ALGORITHM: &str = "splitmix64-v1";

#[derive(Clone, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
        value ^ (value >> 31)
    }

    fn chance_per_mille(&mut self, probability: u64) -> bool {
        self.next_u64() % 1000 < probability
    }
}

fn node_id(index: usize) -> AgentInstanceId {
    AgentInstanceId(format!("AGENT-{index:03}"))
}

fn add_channel(
    channels: &mut Vec<MessageChannel>,
    prefix: &str,
    sender: usize,
    recipient: usize,
    active_from_round: u64,
    active_until_round: Option<u64>,
) {
    let classes = BTreeSet::from([
        MessageClass::Claim,
        MessageClass::Challenge,
        MessageClass::Control,
        MessageClass::Correction,
        MessageClass::Coordination,
        MessageClass::Evidence,
    ]);
    channels.push(MessageChannel {
        channel_id: format!("{prefix}-AGENT-{sender:03}-AGENT-{recipient:03}"),
        sender: node_id(sender),
        recipient: node_id(recipient),
        allowed_message_classes: classes,
        active_from_round,
        active_until_round,
        per_round_capacity: 8,
        max_payload_bytes: 65_536,
    });
}

fn add_reciprocal(
    channels: &mut Vec<MessageChannel>,
    prefix: &str,
    left: usize,
    right: usize,
    active_from: u64,
    active_until: Option<u64>,
) {
    add_channel(channels, prefix, left, right, active_from, active_until);
    add_channel(channels, prefix, right, left, active_from, active_until);
}

/// Deterministically construct one explicit CommunicationGraph topology.
///
/// This is an offline fixture generator, not a social-network inference model.
/// The seed and RNG algorithm are bound in GeneratorProvenance on every profile.
pub fn generate_topology(
    family: TopologyFamily,
    node_count: usize,
    seed: u64,
) -> Result<CommunicationGraphProfile, ValidationErrors> {
    if node_count < 3 {
        return Err(ValidationErrors {
            findings: vec!["topology_generator_requires_at_least_three_nodes".to_string()],
        });
    }
    if family == TopologyFamily::StochasticBlock && node_count < 6 {
        return Err(ValidationErrors {
            findings: vec!["stochastic_block_requires_at_least_six_nodes".to_string()],
        });
    }

    let mut channels = Vec::new();
    let mut rng = SplitMix64::new(seed);
    let mut parameters = BTreeMap::from([
        ("node_count".to_string(), node_count as u64),
        ("seed".to_string(), seed),
    ]);

    match family {
        TopologyFamily::DirectedChain => {
            for sender in 0..(node_count - 1) {
                add_channel(&mut channels, "CHAIN", sender, sender + 1, 1, None);
            }
        }
        TopologyFamily::ReciprocalStar => {
            for leaf in 1..node_count {
                add_reciprocal(&mut channels, "STAR", 0, leaf, 1, None);
            }
        }
        TopologyFamily::BalancedTree => {
            for child in 1..node_count {
                let parent = (child - 1) / 2;
                add_reciprocal(&mut channels, "TREE", parent, child, 1, None);
            }
        }
        TopologyFamily::CompleteDirected => {
            for sender in 0..node_count {
                for recipient in 0..node_count {
                    if sender != recipient {
                        add_channel(&mut channels, "COMPLETE", sender, recipient, 1, None);
                    }
                }
            }
        }
        TopologyFamily::SparseRandomDirected => {
            parameters.insert("extra_arc_probability_per_mille".to_string(), 140);
            // A directed ring supplies the explicit strong-connectivity target.
            for sender in 0..node_count {
                add_channel(
                    &mut channels,
                    "SPARSE-RING",
                    sender,
                    (sender + 1) % node_count,
                    1,
                    None,
                );
            }
            for sender in 0..node_count {
                for recipient in 0..node_count {
                    if sender != recipient
                        && recipient != (sender + 1) % node_count
                        && rng.chance_per_mille(140)
                    {
                        add_channel(&mut channels, "SPARSE-RANDOM", sender, recipient, 1, None);
                    }
                }
            }
        }
        TopologyFamily::StochasticBlock => {
            parameters.insert("community_count".to_string(), 2);
            parameters.insert("intra_block_extra_probability_per_mille".to_string(), 220);
            let split = node_count / 2;
            // A directed ring in each block and two cross-block bridges make
            // the directed block graph strongly connected by construction.
            for (start, end) in [(0, split), (split, node_count)] {
                for sender in start..end {
                    let recipient = if sender + 1 == end { start } else { sender + 1 };
                    add_reciprocal(&mut channels, "BLOCK-RING", sender, recipient, 1, None);
                }
            }
            add_reciprocal(&mut channels, "BLOCK-BRIDGE", split - 1, split, 1, None);

            for (start, end) in [(0, split), (split, node_count)] {
                for sender in start..end {
                    for recipient in (sender + 1)..end {
                        if rng.chance_per_mille(220) {
                            add_reciprocal(&mut channels, "BLOCK-RANDOM", sender, recipient, 1, None);
                        }
                    }
                }
            }
        }
        TopologyFamily::CorePeriphery => {
            let core_size = (node_count / 3).max(2);
            parameters.insert("core_size".to_string(), core_size as u64);
            parameters.insert("periphery_directed_ring".to_string(), 1);
            for sender in 0..core_size {
                for recipient in 0..core_size {
                    if sender != recipient {
                        add_channel(&mut channels, "CORE", sender, recipient, 1, None);
                    }
                }
            }
            for peripheral in core_size..node_count {
                let hub = peripheral % core_size;
                add_reciprocal(&mut channels, "PERIPHERY-LINK", hub, peripheral, 1, None);
                let next_peripheral = if peripheral + 1 == node_count {
                    core_size
                } else {
                    peripheral + 1
                };
                add_channel(&mut channels, "PERIPHERY-RING", peripheral, next_peripheral, 1, None);
            }
        }
        TopologyFamily::TemporalChurn => {
            parameters.insert("initial_channel_expiry_round".to_string(), 5);
            parameters.insert("restoration_active_from_round".to_string(), 8);
            for sender in 0..node_count {
                let recipient = (sender + 1) % node_count;
                add_reciprocal(&mut channels, "CHURN-INITIAL", sender, recipient, 1, Some(5));
                add_reciprocal(&mut channels, "CHURN-RESTORED", sender, recipient, 8, None);
            }
        }
    }

    let nodes = (0..node_count)
        .map(|index| AgentNode {
            agent_instance_id: node_id(index),
            principal_ref: Some(PrincipalId(format!("PRINCIPAL-{index:03}"))),
        })
        .collect::<Vec<_>>();

    let profile = CommunicationGraphProfile {
        schema_version: GRAPH_SCHEMA_VERSION.to_string(),
        profile_id: format!("{:?}-N{}-SEED{}", family, node_count, seed),
        profile_version: "0.1.0".to_string(),
        generator: GeneratorProvenance {
            generator_id: "communication-graph-generator".to_string(),
            generator_version: "0.1.0".to_string(),
            rng_algorithm_id: RNG_ALGORITHM.to_string(),
            rng_seed: seed,
            topology_family: family,
            parameters,
        },
        nodes,
        channels,
    };
    profile.validate()?;
    Ok(profile)
}

fn strongly_connected_at(profile: &CommunicationGraphProfile, round: u64) -> bool {
    let nodes = profile.nodes.iter()
        .map(|node| node.agent_instance_id.clone())
        .collect::<BTreeSet<_>>();
    if nodes.is_empty() {
        return false;
    }
    for source in &nodes {
        let mut visited = BTreeSet::from([source.clone()]);
        let mut queue = VecDeque::from([source.clone()]);
        while let Some(current) = queue.pop_front() {
            for channel in profile.channels.iter()
                .filter(|channel| channel.sender == current && channel.active_at(round))
            {
                if visited.insert(channel.recipient.clone()) {
                    queue.push_back(channel.recipient.clone());
                }
            }
        }
        if visited.len() != nodes.len() {
            return false;
        }
    }
    true
}

fn reciprocal_at(profile: &CommunicationGraphProfile, round: u64) -> bool {
    let arcs = profile.channels.iter()
        .filter(|channel| channel.active_at(round))
        .map(|channel| (channel.sender.clone(), channel.recipient.clone()))
        .collect::<BTreeSet<_>>();
    arcs.iter().all(|(sender, recipient)| arcs.contains(&(recipient.clone(), sender.clone())))
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TopologyFixtureManifest {
    pub schema_version: String,
    pub fixture_count: usize,
    pub fixtures: Vec<TopologyFixture>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TopologyFixture {
    pub fixture_id: String,
    pub family: TopologyFamily,
    pub node_count: usize,
    pub seed: u64,
    pub minimum_channel_count: usize,
    pub expected_strongly_connected_round_1: bool,
    pub expected_strongly_connected_round_6: bool,
    pub expected_strongly_connected_round_8: bool,
    pub expected_reciprocal_round_1: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TopologyFixtureReport {
    pub fixture_count: usize,
    pub family_count: usize,
    pub minimum_channels_checked: usize,
}

/// Generate and check every descriptor in a topology fixture manifest.
pub fn qualify_topology_fixture_manifest(
    source: &str,
) -> Result<TopologyFixtureReport, ValidationErrors> {
    let manifest: TopologyFixtureManifest = serde_json::from_str(source)
        .map_err(|error| ValidationErrors {
            findings: vec![format!("topology_fixture_manifest_parse_failed:{error}")],
        })?;
    let mut findings = Vec::new();
    if manifest.schema_version != "communication-topology-fixtures-v0.1" {
        findings.push("topology_fixture_schema_version_mismatch".to_string());
    }
    if manifest.fixture_count != manifest.fixtures.len() || manifest.fixtures.len() < 30 {
        findings.push("topology_fixture_count_mismatch_or_below_30".to_string());
    }

    let mut ids = BTreeSet::new();
    let mut families = BTreeSet::new();
    let mut minimum_channels_checked = 0usize;
    for fixture in &manifest.fixtures {
        if fixture.fixture_id.trim().is_empty() || !ids.insert(fixture.fixture_id.clone()) {
            findings.push(format!("duplicate_or_empty_topology_fixture_id:{}", fixture.fixture_id));
        }
        families.insert(fixture.family.clone());

        let profile = match generate_topology(fixture.family.clone(), fixture.node_count, fixture.seed) {
            Ok(value) => value,
            Err(error) => {
                findings.extend(error.findings.into_iter().map(|finding| {
                    format!("{}:{finding}", fixture.fixture_id)
                }));
                continue;
            }
        };
        let replay = match generate_topology(fixture.family.clone(), fixture.node_count, fixture.seed) {
            Ok(value) => value,
            Err(error) => {
                findings.extend(error.findings);
                continue;
            }
        };
        if profile.sha256().ok() != replay.sha256().ok() {
            findings.push(format!("{}:seed_replay_digest_mismatch", fixture.fixture_id));
        }
        if profile.channels.len() < fixture.minimum_channel_count {
            findings.push(format!("{}:channel_count_below_minimum", fixture.fixture_id));
        } else {
            minimum_channels_checked += 1;
        }
        if strongly_connected_at(&profile, 1) != fixture.expected_strongly_connected_round_1 {
            findings.push(format!("{}:round_1_connectivity_mismatch", fixture.fixture_id));
        }
        if strongly_connected_at(&profile, 6) != fixture.expected_strongly_connected_round_6 {
            findings.push(format!("{}:round_6_connectivity_mismatch", fixture.fixture_id));
        }
        if strongly_connected_at(&profile, 8) != fixture.expected_strongly_connected_round_8 {
            findings.push(format!("{}:round_8_connectivity_mismatch", fixture.fixture_id));
        }
        if reciprocal_at(&profile, 1) != fixture.expected_reciprocal_round_1 {
            findings.push(format!("{}:round_1_reciprocity_mismatch", fixture.fixture_id));
        }
    }

    if families.len() != 8 {
        findings.push("topology_fixture_manifest_must_cover_eight_families".to_string());
    }
    if findings.is_empty() {
        Ok(TopologyFixtureReport {
            fixture_count: manifest.fixtures.len(),
            family_count: families.len(),
            minimum_channels_checked,
        })
    } else {
        Err(ValidationErrors::new(findings))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURES: &str = include_str!("../TOPOLOGY_FIXTURES_V0_1.json");

    #[test]
    fn thirty_two_topology_fixtures_replay_deterministically() {
        let report = qualify_topology_fixture_manifest(FIXTURES).unwrap();
        assert_eq!(report.fixture_count, 32);
        assert_eq!(report.family_count, 8);
        assert_eq!(report.minimum_channels_checked, 32);
    }

    #[test]
    fn each_topology_family_has_its_declared_semantics() {
        let cases = [
            (TopologyFamily::DirectedChain, false, false),
            (TopologyFamily::ReciprocalStar, true, true),
            (TopologyFamily::BalancedTree, true, true),
            (TopologyFamily::CompleteDirected, true, true),
            (TopologyFamily::SparseRandomDirected, true, false),
            (TopologyFamily::StochasticBlock, true, true),
            (TopologyFamily::CorePeriphery, true, false),
            (TopologyFamily::TemporalChurn, true, true),
        ];
        for (family, connected, reciprocal) in cases {
            let profile = generate_topology(family, 8, 42).unwrap();
            assert_eq!(strongly_connected_at(&profile, 1), connected);
            assert_eq!(reciprocal_at(&profile, 1), reciprocal);
        }
    }

    #[test]
    fn temporal_churn_expires_then_restores_channels() {
        let profile = generate_topology(TopologyFamily::TemporalChurn, 8, 1).unwrap();
        assert!(strongly_connected_at(&profile, 1));
        assert!(!strongly_connected_at(&profile, 6));
        assert!(strongly_connected_at(&profile, 8));
    }

    #[test]
    fn seeded_sparse_and_block_profiles_are_reproducible() {
        for family in [TopologyFamily::SparseRandomDirected, TopologyFamily::StochasticBlock] {
            let first = generate_topology(family.clone(), 12, 1234).unwrap();
            let second = generate_topology(family, 12, 1234).unwrap();
            assert_eq!(first.sha256().unwrap(), second.sha256().unwrap());
        }
    }
}
