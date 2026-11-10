pub mod generator;

//! Typed communication graphs for multi-agent experiments.
//!
//! Delegation graphs and communication graphs are different domain objects.
//! Delegation carries scoped mandate semantics; MessageChannel describes a
//! possible information-delivery path between AgentInstanceId values.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

pub const GRAPH_SCHEMA_VERSION: &str = "communication-graph-profile-v0.1";
pub const EVENT_SCHEMA_VERSION: &str = "communication-event-log-v0.1";
const CENTRALITY_EPSILON: f64 = 1.0e-12;

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct AgentInstanceId(pub String);

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct PrincipalId(pub String);

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageClass {
    Claim,
    Challenge,
    Control,
    Correction,
    Coordination,
    Evidence,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AgentNode {
    pub agent_instance_id: AgentInstanceId,
    pub principal_ref: Option<PrincipalId>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MessageChannel {
    pub channel_id: String,
    pub sender: AgentInstanceId,
    pub recipient: AgentInstanceId,
    pub allowed_message_classes: BTreeSet<MessageClass>,
    pub active_from_round: u64,
    /// Exclusive upper bound. The channel is inactive when round >= value.
    pub active_until_round: Option<u64>,
    /// All attempts, including rejected attempts, consume per-round capacity.
    pub per_round_capacity: u32,
    pub max_payload_bytes: u64,
}

impl MessageChannel {
    pub fn active_at(&self, round: u64) -> bool {
        round >= self.active_from_round
            && self.active_until_round.is_none_or(|until| round < until)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TopologyFamily {
    DirectedChain,
    ReciprocalStar,
    BalancedTree,
    CompleteDirected,
    SparseRandomDirected,
    StochasticBlock,
    CorePeriphery,
    TemporalChurn,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GeneratorProvenance {
    pub generator_id: String,
    pub generator_version: String,
    pub rng_algorithm_id: String,
    pub rng_seed: u64,
    pub topology_family: TopologyFamily,
    pub parameters: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CommunicationGraphProfile {
    pub schema_version: String,
    pub profile_id: String,
    pub profile_version: String,
    pub generator: GeneratorProvenance,
    pub nodes: Vec<AgentNode>,
    pub channels: Vec<MessageChannel>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryOutcome {
    Delivered,
    RejectedInactive,
    RejectedMessageClass,
    RejectedPayloadLimit,
    RejectedCapacity,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EventLogCompleteness {
    Complete,
    Incomplete,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeliveryEvent {
    pub sequence: u64,
    pub event_id: String,
    pub round: u64,
    pub message_id: String,
    pub channel_id: String,
    pub sender: AgentInstanceId,
    pub recipient: AgentInstanceId,
    pub message_class: MessageClass,
    pub payload_sha256: String,
    pub payload_len_bytes: u64,
    pub outcome: DeliveryOutcome,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CommunicationEventLog {
    pub schema_version: String,
    pub graph_sha256: String,
    pub completeness: EventLogCompleteness,
    pub events: Vec<DeliveryEvent>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidationErrors {
    pub findings: Vec<String>,
}

impl ValidationErrors {
    fn new(mut findings: Vec<String>) -> Self {
        findings.sort();
        findings.dedup();
        Self { findings }
    }
}

impl std::fmt::Display for ValidationErrors {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.findings.join("; "))
    }
}

impl std::error::Error for ValidationErrors {}

#[derive(Clone, Debug, PartialEq)]
pub struct CentralityProfile {
    pub metric_id: &'static str,
    pub evaluation_round: u64,
    pub graph_sha256: String,
    pub scores: BTreeMap<AgentInstanceId, f64>,
    pub minimum_tie_set: Vec<AgentInstanceId>,
    pub maximum_tie_set: Vec<AgentInstanceId>,
    pub selected_minimum_node: AgentInstanceId,
    pub selected_maximum_node: AgentInstanceId,
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn digest(bytes: &[u8]) -> String {
    let result = Sha256::digest(bytes);
    result.iter().map(|byte| format!("{byte:02x}")).collect()
}

impl CommunicationGraphProfile {
    pub fn validate(&self) -> Result<(), ValidationErrors> {
        let mut findings = Vec::new();
        if self.schema_version != GRAPH_SCHEMA_VERSION {
            findings.push("schema_version_mismatch".to_string());
        }
        if self.profile_id.trim().is_empty() || self.profile_version.trim().is_empty() {
            findings.push("profile_identity_missing".to_string());
        }
        if self.generator.generator_id != "communication-graph-generator"
            || self.generator.generator_version != "0.1.0"
            || self.generator.rng_algorithm_id != "splitmix64-v1"
        {
            findings.push("generator_provenance_identity_mismatch".to_string());
        }
        if self.nodes.len() < 2 {
            findings.push("minimum_two_agent_nodes_required".to_string());
        }

        let mut node_ids = BTreeSet::new();
        for node in &self.nodes {
            if node.agent_instance_id.0.trim().is_empty() {
                findings.push("empty_agent_instance_id".to_string());
            }
            if !node_ids.insert(node.agent_instance_id.clone()) {
                findings.push(format!("duplicate_agent_instance_id:{}", node.agent_instance_id.0));
            }
            if node.principal_ref.as_ref().is_some_and(|principal| principal.0.trim().is_empty()) {
                findings.push(format!("empty_principal_ref:{}", node.agent_instance_id.0));
            }
        }

        let mut channel_ids = BTreeSet::new();
        for channel in &self.channels {
            if channel.channel_id.trim().is_empty() || !channel_ids.insert(channel.channel_id.clone()) {
                findings.push(format!("empty_or_duplicate_channel_id:{}", channel.channel_id));
            }
            if channel.sender == channel.recipient {
                findings.push(format!("self_loop_forbidden:{}", channel.channel_id));
            }
            if !node_ids.contains(&channel.sender) {
                findings.push(format!("unknown_channel_sender:{}", channel.channel_id));
            }
            if !node_ids.contains(&channel.recipient) {
                findings.push(format!("unknown_channel_recipient:{}", channel.channel_id));
            }
            if channel.allowed_message_classes.is_empty() {
                findings.push(format!("channel_has_no_allowed_message_classes:{}", channel.channel_id));
            }
            if channel.per_round_capacity == 0 || channel.max_payload_bytes == 0 {
                findings.push(format!("channel_capacity_or_payload_limit_zero:{}", channel.channel_id));
            }
            if channel.active_until_round.is_some_and(|until| until <= channel.active_from_round) {
                findings.push(format!("invalid_channel_activation_interval:{}", channel.channel_id));
            }
        }

        if findings.is_empty() { Ok(()) } else { Err(ValidationErrors::new(findings)) }
    }

    /// Canonical graph bytes sort nodes and channels by stable identity. Struct
    /// field order and BTreeSet enum order are fixed by this v0.1 schema.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ValidationErrors> {
        self.validate()?;
        let mut canonical = self.clone();
        canonical.nodes.sort_by(|left, right| left.agent_instance_id.cmp(&right.agent_instance_id));
        canonical.channels.sort_by(|left, right| left.channel_id.cmp(&right.channel_id));
        serde_json::to_vec(&canonical)
            .map_err(|error| ValidationErrors::new(vec![format!("canonical_serialize_failed:{error}")]))
    }

    pub fn sha256(&self) -> Result<String, ValidationErrors> {
        Ok(digest(&self.canonical_bytes()?))
    }

    /// Potential reachability under active channel and message-class policy.
    /// This does not assert that a message was attempted or delivered.
    pub fn potentially_reachable(
        &self,
        from: &AgentInstanceId,
        to: &AgentInstanceId,
        round: u64,
        message_class: &MessageClass,
    ) -> Result<bool, ValidationErrors> {
        self.validate()?;
        let nodes = self.nodes.iter()
            .map(|node| node.agent_instance_id.clone())
            .collect::<BTreeSet<_>>();
        if !nodes.contains(from) || !nodes.contains(to) {
            return Err(ValidationErrors::new(vec!["reachability_endpoint_unknown".to_string()]));
        }
        if from == to {
            return Ok(true);
        }

        let mut adjacency = BTreeMap::<AgentInstanceId, BTreeSet<AgentInstanceId>>::new();
        for node in &self.nodes {
            adjacency.insert(node.agent_instance_id.clone(), BTreeSet::new());
        }
        for channel in &self.channels {
            if channel.active_at(round) && channel.allowed_message_classes.contains(message_class) {
                adjacency.get_mut(&channel.sender).expect("validated sender").insert(channel.recipient.clone());
            }
        }

        let mut visited = BTreeSet::from([from.clone()]);
        let mut queue = VecDeque::from([from.clone()]);
        while let Some(current) = queue.pop_front() {
            for neighbor in adjacency.get(&current).into_iter().flatten() {
                if neighbor == to {
                    return Ok(true);
                }
                if visited.insert(neighbor.clone()) {
                    queue.push_back(neighbor.clone());
                }
            }
        }
        Ok(false)
    }
}

impl CommunicationEventLog {
    /// Validate every attempted message against channel policy, temporal state,
    /// capacity and the exact ordered event trace.
    pub fn validate_against(
        &self,
        graph: &CommunicationGraphProfile,
    ) -> Result<(), ValidationErrors> {
        let mut findings = Vec::new();
        if self.schema_version != EVENT_SCHEMA_VERSION {
            findings.push("event_schema_version_mismatch".to_string());
        }
        if self.completeness != EventLogCompleteness::Complete {
            findings.push("event_log_incomplete".to_string());
        }

        let graph_digest = match graph.sha256() {
            Ok(value) => Some(value),
            Err(error) => {
                findings.extend(error.findings);
                None
            }
        };
        if graph_digest.as_deref() != Some(self.graph_sha256.as_str()) {
            findings.push("event_log_graph_digest_mismatch".to_string());
        }

        let channels = graph.channels.iter()
            .map(|channel| (channel.channel_id.as_str(), channel))
            .collect::<BTreeMap<_, _>>();
        let nodes = graph.nodes.iter()
            .map(|node| node.agent_instance_id.clone())
            .collect::<BTreeSet<_>>();

        let mut event_ids = BTreeSet::new();
        let mut message_recipient_pairs = BTreeSet::new();
        let mut attempt_counts = BTreeMap::<(String, u64), u32>::new();
        let mut previous_round = 0u64;

        for (index, event) in self.events.iter().enumerate() {
            let context = format!("events[{index}]");
            if event.sequence != index as u64 + 1 {
                findings.push(format!("{context}:event_sequence_not_contiguous"));
            }
            if event.round < previous_round {
                findings.push(format!("{context}:event_round_order_regression"));
            }
            previous_round = event.round;

            if event.event_id.trim().is_empty() || !event_ids.insert(event.event_id.clone()) {
                findings.push(format!("{context}:empty_or_duplicate_event_id"));
            }
            if event.message_id.trim().is_empty()
                || !message_recipient_pairs.insert((event.message_id.clone(), event.recipient.clone()))
            {
                findings.push(format!("{context}:empty_or_duplicate_message_recipient"));
            }
            if !nodes.contains(&event.sender) || !nodes.contains(&event.recipient) {
                findings.push(format!("{context}:event_endpoint_unknown"));
            }
            if event.sender == event.recipient {
                findings.push(format!("{context}:self_delivery_forbidden"));
            }
            if !is_sha256(&event.payload_sha256) {
                findings.push(format!("{context}:invalid_payload_sha256"));
            }

            let Some(channel) = channels.get(event.channel_id.as_str()).copied() else {
                findings.push(format!("{context}:unknown_channel"));
                continue;
            };
            if event.sender != channel.sender || event.recipient != channel.recipient {
                findings.push(format!("{context}:event_endpoints_do_not_match_channel"));
            }

            let count = attempt_counts.entry((event.channel_id.clone(), event.round)).or_default();
            *count = count.saturating_add(1);
            let derived = if !channel.active_at(event.round) {
                DeliveryOutcome::RejectedInactive
            } else if !channel.allowed_message_classes.contains(&event.message_class) {
                DeliveryOutcome::RejectedMessageClass
            } else if event.payload_len_bytes > channel.max_payload_bytes {
                DeliveryOutcome::RejectedPayloadLimit
            } else if *count > channel.per_round_capacity {
                DeliveryOutcome::RejectedCapacity
            } else {
                DeliveryOutcome::Delivered
            };
            if event.outcome != derived {
                findings.push(format!("{context}:delivery_outcome_mismatch"));
            }
        }

        if findings.is_empty() { Ok(()) } else { Err(ValidationErrors::new(findings)) }
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ValidationErrors> {
        if self.schema_version != EVENT_SCHEMA_VERSION {
            return Err(ValidationErrors::new(vec!["event_schema_version_mismatch".to_string()]));
        }
        serde_json::to_vec(self)
            .map_err(|error| ValidationErrors::new(vec![format!("event_log_serialize_failed:{error}")]))
    }

    pub fn sha256(&self) -> Result<String, ValidationErrors> {
        Ok(digest(&self.canonical_bytes()?))
    }
}

/// Directed, unweighted, hop-count Brandes betweenness on a reciprocal,
/// strongly connected CommunicationGraph with at least three nodes.
///
/// Multiple channels between one ordered pair collapse to one simple directed
/// arc for centrality only. Scores normalize by (n-1)(n-2).
pub fn normalized_betweenness_v1(
    graph: &CommunicationGraphProfile,
    evaluation_round: u64,
) -> Result<CentralityProfile, ValidationErrors> {
    graph.validate()?;
    let count = graph.nodes.len();
    if count < 3 {
        return Err(ValidationErrors::new(vec!["centrality_requires_at_least_three_nodes".to_string()]));
    }

    let node_ids = graph.nodes.iter()
        .map(|node| node.agent_instance_id.clone())
        .collect::<BTreeSet<_>>();
    let mut adjacency = BTreeMap::<AgentInstanceId, BTreeSet<AgentInstanceId>>::new();
    for node in &node_ids {
        adjacency.insert(node.clone(), BTreeSet::new());
    }
    for channel in &graph.channels {
        if channel.active_at(evaluation_round) {
            adjacency.get_mut(&channel.sender).expect("validated sender").insert(channel.recipient.clone());
        }
    }

    for (sender, recipients) in &adjacency {
        for recipient in recipients {
            if !adjacency.get(recipient).is_some_and(|reverse| reverse.contains(sender)) {
                return Err(ValidationErrors::new(vec![
                    format!("centrality_requires_reciprocal_arcs:{}->{}", sender.0, recipient.0)
                ]));
            }
        }
    }

    for source in &node_ids {
        let mut visited = BTreeSet::from([source.clone()]);
        let mut queue = VecDeque::from([source.clone()]);
        while let Some(current) = queue.pop_front() {
            for neighbor in adjacency.get(&current).into_iter().flatten() {
                if visited.insert(neighbor.clone()) {
                    queue.push_back(neighbor.clone());
                }
            }
        }
        if visited.len() != count {
            return Err(ValidationErrors::new(vec![
                format!("centrality_requires_strong_connectivity_from:{}", source.0)
            ]));
        }
    }

    let mut scores = node_ids.iter()
        .cloned()
        .map(|node| (node, 0.0f64))
        .collect::<BTreeMap<_, _>>();

    for source in &node_ids {
        let mut stack = Vec::<AgentInstanceId>::new();
        let mut predecessors = node_ids.iter()
            .cloned()
            .map(|node| (node, Vec::<AgentInstanceId>::new()))
            .collect::<BTreeMap<_, _>>();
        let mut sigma = node_ids.iter()
            .cloned()
            .map(|node| (node, 0u128))
            .collect::<BTreeMap<_, _>>();
        let mut distance = node_ids.iter()
            .cloned()
            .map(|node| (node, None::<u64>))
            .collect::<BTreeMap<_, _>>();

        sigma.insert(source.clone(), 1);
        distance.insert(source.clone(), Some(0));
        let mut queue = VecDeque::from([source.clone()]);

        while let Some(vertex) = queue.pop_front() {
            stack.push(vertex.clone());
            let vertex_distance = distance[&vertex].expect("queued nodes have distance");
            for neighbor in adjacency.get(&vertex).into_iter().flatten() {
                if distance[neighbor].is_none() {
                    distance.insert(neighbor.clone(), Some(vertex_distance + 1));
                    queue.push_back(neighbor.clone());
                }
                if distance[neighbor] == Some(vertex_distance + 1) {
                    let updated = sigma[neighbor].checked_add(sigma[&vertex]).ok_or_else(|| {
                        ValidationErrors::new(vec!["centrality_shortest_path_count_overflow".to_string()])
                    })?;
                    sigma.insert(neighbor.clone(), updated);
                    predecessors.get_mut(neighbor).expect("known node").push(vertex.clone());
                }
            }
        }

        let mut dependency = node_ids.iter()
            .cloned()
            .map(|node| (node, 0.0f64))
            .collect::<BTreeMap<_, _>>();
        while let Some(vertex) = stack.pop() {
            let vertex_sigma = sigma[&vertex] as f64;
            for predecessor in &predecessors[&vertex] {
                let predecessor_sigma = sigma[predecessor] as f64;
                let contribution = (predecessor_sigma / vertex_sigma) * (1.0 + dependency[&vertex]);
                *dependency.get_mut(predecessor).expect("known node") += contribution;
            }
            if &vertex != source {
                *scores.get_mut(&vertex).expect("known node") += dependency[&vertex];
            }
        }
    }

    let scale = ((count - 1) * (count - 2)) as f64;
    for score in scores.values_mut() {
        *score /= scale;
    }

    let minimum = scores.values().copied().fold(f64::INFINITY, f64::min);
    let maximum = scores.values().copied().fold(f64::NEG_INFINITY, f64::max);
    let minimum_tie_set = scores.iter()
        .filter(|(_, score)| (**score - minimum).abs() <= CENTRALITY_EPSILON)
        .map(|(node, _)| node.clone())
        .collect::<Vec<_>>();
    let maximum_tie_set = scores.iter()
        .filter(|(_, score)| (**score - maximum).abs() <= CENTRALITY_EPSILON)
        .map(|(node, _)| node.clone())
        .collect::<Vec<_>>();
    let selected_minimum_node = minimum_tie_set.first().cloned().expect("nonempty graph");
    let selected_maximum_node = maximum_tie_set.first().cloned().expect("nonempty graph");
    if selected_minimum_node == selected_maximum_node {
        return Err(ValidationErrors::new(vec![
            "centrality_has_no_distinguishable_minimum_maximum_contrast".to_string()
        ]));
    }

    Ok(CentralityProfile {
        metric_id: "normalized_betweenness_v1",
        evaluation_round,
        graph_sha256: graph.sha256()?,
        scores,
        minimum_tie_set,
        maximum_tie_set,
        selected_minimum_node,
        selected_maximum_node,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(value: &str) -> AgentInstanceId { AgentInstanceId(value.to_string()) }
    fn principal(value: &str) -> PrincipalId { PrincipalId(value.to_string()) }

    fn channel(name: &str, sender: &str, recipient: &str) -> MessageChannel {
        MessageChannel {
            channel_id: name.to_string(),
            sender: id(sender),
            recipient: id(recipient),
            allowed_message_classes: BTreeSet::from([
                MessageClass::Claim,
                MessageClass::Challenge,
                MessageClass::Correction,
                MessageClass::Evidence,
            ]),
            active_from_round: 1,
            active_until_round: None,
            per_round_capacity: 2,
            max_payload_bytes: 1024,
        }
    }

    fn star() -> CommunicationGraphProfile {
        CommunicationGraphProfile {
            schema_version: GRAPH_SCHEMA_VERSION.to_string(),
            profile_id: "test-star".to_string(),
            profile_version: "0.1.0".to_string(),
            generator: GeneratorProvenance {
                generator_id: "communication-graph-generator".to_string(),
                generator_version: "0.1.0".to_string(),
                rng_algorithm_id: "splitmix64-v1".to_string(),
                rng_seed: 0,
                topology_family: TopologyFamily::ReciprocalStar,
                parameters: BTreeMap::from([("node_count".to_string(), 4)]),
            },
            nodes: vec![
                AgentNode { agent_instance_id: id("AGENT-A"), principal_ref: Some(principal("PRINCIPAL-X")) },
                AgentNode { agent_instance_id: id("AGENT-B"), principal_ref: Some(principal("PRINCIPAL-X")) },
                AgentNode { agent_instance_id: id("AGENT-C"), principal_ref: Some(principal("PRINCIPAL-C")) },
                AgentNode { agent_instance_id: id("AGENT-D"), principal_ref: Some(principal("PRINCIPAL-D")) },
            ],
            channels: vec![
                channel("A-B", "AGENT-A", "AGENT-B"),
                channel("B-A", "AGENT-B", "AGENT-A"),
                channel("A-C", "AGENT-A", "AGENT-C"),
                channel("C-A", "AGENT-C", "AGENT-A"),
                channel("A-D", "AGENT-A", "AGENT-D"),
                channel("D-A", "AGENT-D", "AGENT-A"),
            ],
        }
    }

    fn complete_graph() -> CommunicationGraphProfile {
        let mut graph = star();
        graph.profile_id = "test-complete".to_string();
        graph.channels.clear();
        let nodes = graph.nodes.iter().map(|node| node.agent_instance_id.clone()).collect::<Vec<_>>();
        for sender in &nodes {
            for recipient in &nodes {
                if sender != recipient {
                    graph.channels.push(channel(&format!("{}-{}", sender.0, recipient.0), &sender.0, &recipient.0));
                }
            }
        }
        graph
    }

    fn event(sequence: u64, event_id: &str, round: u64, message_id: &str,
        channel_id: &str, sender: &str, recipient: &str, outcome: DeliveryOutcome) -> DeliveryEvent {
        DeliveryEvent {
            sequence,
            event_id: event_id.to_string(),
            round,
            message_id: message_id.to_string(),
            channel_id: channel_id.to_string(),
            sender: id(sender),
            recipient: id(recipient),
            message_class: MessageClass::Evidence,
            payload_sha256: "a".repeat(64),
            payload_len_bytes: 20,
            outcome,
        }
    }

    #[test]
    fn sha256_matches_standard_known_answer() {
        assert_eq!(
            digest(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn principal_multiplicity_does_not_merge_agent_instances() {
        let graph = star();
        graph.validate().unwrap();
        assert_ne!(graph.nodes[0].agent_instance_id, graph.nodes[1].agent_instance_id);
        assert_eq!(graph.nodes[0].principal_ref, graph.nodes[1].principal_ref);
    }

    #[test]
    fn reciprocal_channels_are_explicit_and_not_inferred() {
        let mut graph = star();
        graph.channels.retain(|channel| channel.channel_id != "B-A");
        assert!(!graph.potentially_reachable(&id("AGENT-B"), &id("AGENT-A"), 1, &MessageClass::Evidence).unwrap());
    }

    #[test]
    fn potential_reachability_is_not_a_delivery_event() {
        let graph = star();
        assert!(graph.potentially_reachable(&id("AGENT-A"), &id("AGENT-C"), 1, &MessageClass::Evidence).unwrap());
        let log = CommunicationEventLog {
            schema_version: EVENT_SCHEMA_VERSION.to_string(),
            graph_sha256: graph.sha256().unwrap(),
            completeness: EventLogCompleteness::Complete,
            events: vec![],
        };
        log.validate_against(&graph).unwrap();
        assert!(log.events.is_empty());
    }

    #[test]
    fn graph_digest_is_invariant_to_node_and_channel_vector_order() {
        let graph = star();
        let mut reordered = graph.clone();
        reordered.nodes.reverse();
        reordered.channels.reverse();
        assert_eq!(graph.sha256().unwrap(), reordered.sha256().unwrap());
    }

    #[test]
    fn graph_digest_changes_when_channel_policy_changes() {
        let graph = star();
        let mut modified = graph.clone();
        modified.channels[0].per_round_capacity += 1;
        assert_ne!(graph.sha256().unwrap(), modified.sha256().unwrap());
    }

    #[test]
    fn duplicate_agent_instance_ids_are_rejected() {
        let mut graph = star();
        graph.nodes.push(graph.nodes[0].clone());
        assert!(graph.validate().unwrap_err().findings.iter().any(|item| item.contains("duplicate_agent_instance_id")));
    }

    #[test]
    fn self_loop_is_rejected() {
        let mut graph = star();
        graph.channels.push(channel("A-A", "AGENT-A", "AGENT-A"));
        assert!(graph.validate().unwrap_err().findings.iter().any(|item| item.contains("self_loop_forbidden")));
    }

    #[test]
    fn expired_channel_is_not_potentially_reachable() {
        let mut graph = star();
        graph.channels.iter_mut().find(|channel| channel.channel_id == "A-B").unwrap().active_until_round = Some(3);
        assert!(graph.potentially_reachable(&id("AGENT-A"), &id("AGENT-B"), 2, &MessageClass::Evidence).unwrap());
        assert!(!graph.potentially_reachable(&id("AGENT-A"), &id("AGENT-B"), 3, &MessageClass::Evidence).unwrap());
    }

    #[test]
    fn message_class_policy_restricts_reachability() {
        let mut graph = star();
        graph.channels.iter_mut().find(|channel| channel.channel_id == "A-B").unwrap()
            .allowed_message_classes = BTreeSet::from([MessageClass::Claim]);
        assert!(!graph.potentially_reachable(&id("AGENT-A"), &id("AGENT-B"), 1, &MessageClass::Evidence).unwrap());
    }

    #[test]
    fn delivery_event_must_bind_exact_channel_endpoints() {
        let graph = star();
        let log = CommunicationEventLog {
            schema_version: EVENT_SCHEMA_VERSION.to_string(),
            graph_sha256: graph.sha256().unwrap(),
            completeness: EventLogCompleteness::Complete,
            events: vec![event(1, "EV-1", 1, "M-1", "A-B", "AGENT-B", "AGENT-A", DeliveryOutcome::Delivered)],
        };
        assert!(log.validate_against(&graph).unwrap_err().findings.iter().any(|item| item.contains("event_endpoints_do_not_match_channel")));
    }

    #[test]
    fn delivered_attempts_consume_per_round_capacity() {
        let graph = star();
        let log = CommunicationEventLog {
            schema_version: EVENT_SCHEMA_VERSION.to_string(),
            graph_sha256: graph.sha256().unwrap(),
            completeness: EventLogCompleteness::Complete,
            events: vec![
                event(1, "EV-1", 1, "M-1", "A-B", "AGENT-A", "AGENT-B", DeliveryOutcome::Delivered),
                event(2, "EV-2", 1, "M-2", "A-B", "AGENT-A", "AGENT-B", DeliveryOutcome::Delivered),
                event(3, "EV-3", 1, "M-3", "A-B", "AGENT-A", "AGENT-B", DeliveryOutcome::RejectedCapacity),
            ],
        };
        log.validate_against(&graph).unwrap();
    }

    #[test]
    fn inactive_channel_attempt_must_be_explicitly_rejected() {
        let mut graph = star();
        graph.channels.iter_mut().find(|channel| channel.channel_id == "A-B").unwrap().active_until_round = Some(2);
        let log = CommunicationEventLog {
            schema_version: EVENT_SCHEMA_VERSION.to_string(),
            graph_sha256: graph.sha256().unwrap(),
            completeness: EventLogCompleteness::Complete,
            events: vec![event(1, "EV-1", 2, "M-1", "A-B", "AGENT-A", "AGENT-B", DeliveryOutcome::RejectedInactive)],
        };
        log.validate_against(&graph).unwrap();
    }

    #[test]
    fn sequence_gaps_invalidate_event_log() {
        let graph = star();
        let log = CommunicationEventLog {
            schema_version: EVENT_SCHEMA_VERSION.to_string(),
            graph_sha256: graph.sha256().unwrap(),
            completeness: EventLogCompleteness::Complete,
            events: vec![event(2, "EV-2", 1, "M-2", "A-B", "AGENT-A", "AGENT-B", DeliveryOutcome::Delivered)],
        };
        assert!(log.validate_against(&graph).unwrap_err().findings.iter().any(|item| item.contains("event_sequence_not_contiguous")));
    }

    #[test]
    fn round_regression_invalidates_event_log() {
        let graph = star();
        let log = CommunicationEventLog {
            schema_version: EVENT_SCHEMA_VERSION.to_string(),
            graph_sha256: graph.sha256().unwrap(),
            completeness: EventLogCompleteness::Complete,
            events: vec![
                event(1, "EV-1", 2, "M-1", "A-B", "AGENT-A", "AGENT-B", DeliveryOutcome::Delivered),
                event(2, "EV-2", 1, "M-2", "B-A", "AGENT-B", "AGENT-A", DeliveryOutcome::Delivered),
            ],
        };
        assert!(log.validate_against(&graph).unwrap_err().findings.iter().any(|item| item.contains("event_round_order_regression")));
    }

    #[test]
    fn incomplete_trace_is_not_a_pass() {
        let graph = star();
        let log = CommunicationEventLog {
            schema_version: EVENT_SCHEMA_VERSION.to_string(),
            graph_sha256: graph.sha256().unwrap(),
            completeness: EventLogCompleteness::Incomplete,
            events: vec![],
        };
        assert!(log.validate_against(&graph).unwrap_err().findings.contains(&"event_log_incomplete".to_string()));
    }

    #[test]
    fn payload_limit_requires_explicit_rejection() {
        let graph = star();
        let mut attempted = event(1, "EV-1", 1, "M-1", "A-B", "AGENT-A", "AGENT-B", DeliveryOutcome::RejectedPayloadLimit);
        attempted.payload_len_bytes = 1025;
        let log = CommunicationEventLog {
            schema_version: EVENT_SCHEMA_VERSION.to_string(),
            graph_sha256: graph.sha256().unwrap(),
            completeness: EventLogCompleteness::Complete,
            events: vec![attempted],
        };
        log.validate_against(&graph).unwrap();
    }

    #[test]
    fn centrality_known_answer_for_reciprocal_star() {
        let profile = normalized_betweenness_v1(&star(), 1).unwrap();
        assert_eq!(profile.selected_maximum_node, id("AGENT-A"));
        assert!((profile.scores[&id("AGENT-A")] - 1.0).abs() < CENTRALITY_EPSILON);
        for leaf in ["AGENT-B", "AGENT-C", "AGENT-D"] {
            assert!(profile.scores[&id(leaf)].abs() < CENTRALITY_EPSILON);
        }
    }

    #[test]
    fn complete_graph_has_no_distinguishable_centrality_contrast() {
        let error = normalized_betweenness_v1(&complete_graph(), 1).unwrap_err();
        assert!(error.findings.contains(&"centrality_has_no_distinguishable_minimum_maximum_contrast".to_string()));
    }

    #[test]
    fn directed_graph_without_reciprocal_arcs_is_unsupported_for_v1() {
        let mut graph = star();
        graph.channels.retain(|channel| channel.channel_id != "B-A");
        let error = normalized_betweenness_v1(&graph, 1).unwrap_err();
        assert!(error.findings.iter().any(|item| item.contains("centrality_requires_reciprocal_arcs")));
    }

    #[test]
    fn temporal_centrality_uses_the_requested_round_snapshot() {
        let mut graph = star();
        graph.channels.iter_mut().for_each(|channel| {
            channel.active_until_round = Some(5);
        });
        assert!(normalized_betweenness_v1(&graph, 1).is_ok());
        assert!(normalized_betweenness_v1(&graph, 5).is_err());
    }

    #[test]
    fn fewer_than_three_nodes_is_unsupported_for_centrality() {
        let mut graph = star();
        graph.nodes.truncate(2);
        graph.channels.retain(|channel| {
            channel.sender.0 != "AGENT-C" && channel.recipient.0 != "AGENT-C"
                && channel.sender.0 != "AGENT-D" && channel.recipient.0 != "AGENT-D"
        });
        assert!(normalized_betweenness_v1(&graph, 1).unwrap_err().findings.contains(&"centrality_requires_at_least_three_nodes".to_string()));
    }
}
