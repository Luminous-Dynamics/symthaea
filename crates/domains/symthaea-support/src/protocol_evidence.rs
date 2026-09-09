// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Privacy-aware, provider-neutral network protocol evidence -> `SystemObservationV1`.
//!
//! This layer consumes already-parsed packet/flow/session evidence. It deliberately
//! does **not** capture packets, infer canonical graph identity from IP addresses,
//! create topology relations, diagnose root cause, or grant network authority.
//!
//! Core non-equivalences:
//!
//! ```text
//! packet observed != endpoint identity established
//! packet exchange != durable dependency
//! parser classification != protocol truth
//! flow-derived anomaly != direct packet fact
//! protocol evidence != network continuity qualification
//! ```
//!
//! A future PCAP/live-capture parser should target this boundary rather than making
//! raw packet parsing own support-state, topology, or execution semantics.

use crate::scrubber::PrivacyScrubber;
use crate::system_state::{
    EntityId, ObservationClockV1, ObservationId, ObservationProvenanceV1,
    ObservationSourceKindV1, StateValueV1, SystemObservationV1, SystemStateGraphError,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProtocolEvidenceBasisV1 {
    /// Classification is supported by one parsed packet/frame/message.
    DirectPacket,
    /// Classification requires comparison across multiple packets in a flow.
    FlowDerived,
    /// Classification requires protocol/session state across one or more flows.
    SessionDerived,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProtocolSubjectRoleV1 {
    Source,
    Destination,
    Peer,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransportProtocolV1 {
    Tcp,
    Udp,
    Icmp,
    Icmpv6,
    Other(u8),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TcpProtocolEventV1 {
    Syn,
    SynAck,
    Ack,
    Fin,
    Rst,
    ZeroWindow,
    RetransmissionCandidate,
    RepeatedSynCandidate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DnsProtocolEventV1 {
    Query {
        qtype: Option<u16>,
        name: Option<String>,
    },
    Response {
        rcode: u16,
        answer_count: Option<u16>,
        name: Option<String>,
    },
    /// Absence-of-response claim derived over a bounded observation window.
    TimeoutCandidate {
        attempts: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TlsProtocolEventV1 {
    ClientHello {
        protocol_version: Option<String>,
        server_name: Option<String>,
    },
    ServerHello {
        protocol_version: Option<String>,
        alpn: Option<String>,
    },
    CertificateObserved {
        sha256_fingerprint: Option<String>,
        not_after_unix_ms: Option<u64>,
    },
    Alert {
        level: Option<u8>,
        description: u8,
    },
    /// Session-level conclusion; not equivalent to observing one handshake packet.
    HandshakeComplete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuicProtocolEventV1 {
    Initial,
    ZeroRtt,
    Handshake,
    Retry,
    OneRtt,
    VersionNegotiation,
    ConnectionClose {
        error_code: Option<u64>,
    },
    /// Requires connection/session context to distinguish from an ordinary short header.
    StatelessResetCandidate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IcmpProtocolEventV1 {
    EchoRequest,
    EchoReply,
    DestinationUnreachable { code: u8 },
    TimeExceeded { code: u8 },
    PacketTooBig { mtu: Option<u32> },
    FragmentationNeeded { mtu: Option<u32> },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProtocolEventV1 {
    Tcp(TcpProtocolEventV1),
    Dns(DnsProtocolEventV1),
    Tls(TlsProtocolEventV1),
    Quic(QuicProtocolEventV1),
    Icmp(IcmpProtocolEventV1),
    Marker {
        protocol: String,
        marker: String,
        code: Option<u64>,
    },
}

impl ProtocolEventV1 {
    fn family_name(&self) -> &str {
        match self {
            Self::Tcp(_) => "tcp",
            Self::Dns(_) => "dns",
            Self::Tls(_) => "tls",
            Self::Quic(_) => "quic",
            Self::Icmp(_) => "icmp",
            Self::Marker { protocol, .. } => protocol,
        }
    }

    fn requires_derived_basis(&self) -> bool {
        matches!(
            self,
            Self::Tcp(TcpProtocolEventV1::RetransmissionCandidate)
                | Self::Tcp(TcpProtocolEventV1::RepeatedSynCandidate)
                | Self::Dns(DnsProtocolEventV1::TimeoutCandidate { .. })
                | Self::Tls(TlsProtocolEventV1::HandshakeComplete)
                | Self::Quic(QuicProtocolEventV1::StatelessResetCandidate)
        )
    }

    fn validate(&self) -> Result<(), ProtocolEvidenceErrorV1> {
        match self {
            Self::Dns(DnsProtocolEventV1::Query { name, .. })
            | Self::Dns(DnsProtocolEventV1::Response { name, .. }) => {
                validate_optional_nonempty(name.as_deref(), "DNS name")?;
            }
            Self::Dns(DnsProtocolEventV1::TimeoutCandidate { attempts }) => {
                if *attempts == 0 {
                    return Err(ProtocolEvidenceErrorV1::InvalidField(
                        "DNS timeout attempts must be non-zero".into(),
                    ));
                }
            }
            Self::Tls(TlsProtocolEventV1::ClientHello {
                protocol_version,
                server_name,
            }) => {
                validate_optional_nonempty(protocol_version.as_deref(), "TLS protocol version")?;
                validate_optional_nonempty(server_name.as_deref(), "TLS server name")?;
            }
            Self::Tls(TlsProtocolEventV1::ServerHello {
                protocol_version,
                alpn,
            }) => {
                validate_optional_nonempty(protocol_version.as_deref(), "TLS protocol version")?;
                validate_optional_nonempty(alpn.as_deref(), "TLS ALPN")?;
            }
            Self::Tls(TlsProtocolEventV1::CertificateObserved {
                sha256_fingerprint,
                ..
            }) => {
                if let Some(fingerprint) = sha256_fingerprint {
                    validate_hex_digest(fingerprint, "TLS certificate SHA-256 fingerprint")?;
                }
            }
            Self::Marker {
                protocol, marker, ..
            } => {
                require_nonempty(protocol, "protocol marker family")?;
                require_nonempty(marker, "protocol marker")?;
            }
            _ => {}
        }
        Ok(())
    }
}

/// Opaque identity/provenance for parsed protocol evidence.
///
/// `record_digest` must bind the raw packet or canonical derived evidence record.
/// The support layer therefore does not need to hash private endpoint strings into
/// observation IDs merely to distinguish otherwise-similar packets.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtocolEvidenceRecordV1 {
    pub record_id: String,
    /// SHA-256/BLAKE3-sized digest (32 bytes encoded as 64 hex characters).
    pub record_digest: String,
    pub basis: ProtocolEvidenceBasisV1,
    pub transport: Option<TransportProtocolV1>,
    pub source_port: Option<u16>,
    pub destination_port: Option<u16>,
    pub packet_index: Option<u64>,
    /// Opaque parser-defined flow identity. Never interpreted as a graph entity.
    pub flow_id: Option<String>,
    pub event_time_unix_ms: Option<u64>,
    pub clock_uncertainty_ms: Option<u64>,
    pub event: ProtocolEventV1,
}

impl ProtocolEvidenceRecordV1 {
    pub fn validate(&self) -> Result<(), ProtocolEvidenceErrorV1> {
        require_nonempty(&self.record_id, "protocol record id")?;
        validate_hex_digest(&self.record_digest, "protocol record digest")?;
        validate_optional_nonempty(self.flow_id.as_deref(), "protocol flow id")?;
        self.event.validate()?;

        if self.event.requires_derived_basis()
            && self.basis == ProtocolEvidenceBasisV1::DirectPacket
        {
            return Err(ProtocolEvidenceErrorV1::DerivedClaimFromDirectPacket);
        }

        match &self.event {
            ProtocolEventV1::Tcp(_) if self.transport != Some(TransportProtocolV1::Tcp) => {
                return Err(ProtocolEvidenceErrorV1::TransportMismatch(
                    "TCP event requires TCP transport".into(),
                ));
            }
            ProtocolEventV1::Quic(_) if self.transport != Some(TransportProtocolV1::Udp) => {
                return Err(ProtocolEvidenceErrorV1::TransportMismatch(
                    "QUIC event requires UDP transport".into(),
                ));
            }
            ProtocolEventV1::Dns(_)
                if !matches!(
                    self.transport,
                    Some(TransportProtocolV1::Tcp) | Some(TransportProtocolV1::Udp)
                ) =>
            {
                return Err(ProtocolEvidenceErrorV1::TransportMismatch(
                    "DNS event requires TCP or UDP transport".into(),
                ));
            }
            ProtocolEventV1::Icmp(_)
                if !matches!(
                    self.transport,
                    Some(TransportProtocolV1::Icmp) | Some(TransportProtocolV1::Icmpv6)
                ) =>
            {
                return Err(ProtocolEvidenceErrorV1::TransportMismatch(
                    "ICMP event requires ICMP or ICMPv6 transport".into(),
                ));
            }
            _ => {}
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtocolObservationRetentionPolicyV1 {
    /// Canonical graph IDs are not raw addresses; retaining the resolved peer is useful
    /// for audit while still not creating a topology relation.
    pub retain_peer_entity: bool,
    pub retain_flow_id: bool,
    pub retain_dns_names: bool,
    pub retain_tls_server_name: bool,
    pub retain_tls_certificate_fingerprint: bool,
}

impl Default for ProtocolObservationRetentionPolicyV1 {
    fn default() -> Self {
        Self {
            retain_peer_entity: true,
            retain_flow_id: false,
            retain_dns_names: false,
            retain_tls_server_name: false,
            retain_tls_certificate_fingerprint: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProtocolObservationAdapterConfigV1 {
    pub source_id: String,
    pub collector: String,
    pub collector_version: Option<String>,
    /// Parser/profile identity, e.g. `pcap-parser-v1` or `ebpf-flow-v2`.
    pub parser_profile: String,
    pub max_age_ms: Option<u64>,
    /// Confidence in parsing/normalization mechanics, not protocol truth.
    pub normalization_confidence: f32,
}

pub struct ProtocolObservationAdapterV1 {
    config: ProtocolObservationAdapterConfigV1,
    policy: ProtocolObservationRetentionPolicyV1,
    scrubber: PrivacyScrubber,
}

impl ProtocolObservationAdapterV1 {
    pub fn new(
        config: ProtocolObservationAdapterConfigV1,
        policy: ProtocolObservationRetentionPolicyV1,
    ) -> Result<Self, ProtocolEvidenceErrorV1> {
        require_nonempty(&config.source_id, "protocol source id")?;
        require_nonempty(&config.collector, "protocol collector")?;
        require_nonempty(&config.parser_profile, "protocol parser profile")?;
        validate_optional_nonempty(config.collector_version.as_deref(), "collector version")?;
        if !config.normalization_confidence.is_finite()
            || !(0.0..=1.0).contains(&config.normalization_confidence)
        {
            return Err(ProtocolEvidenceErrorV1::InvalidConfidence(
                config.normalization_confidence,
            ));
        }
        Ok(Self {
            config,
            policy,
            scrubber: PrivacyScrubber::new(),
        })
    }

    /// Normalize parsed protocol evidence for an already-resolved canonical subject.
    ///
    /// `peer` is optional canonical context only. Neither ports nor parser flow IDs are
    /// used to infer graph entities or create `ConnectsTo`/dependency relations.
    pub fn normalize(
        &self,
        subject: EntityId,
        peer: Option<EntityId>,
        subject_role: ProtocolSubjectRoleV1,
        record: &ProtocolEvidenceRecordV1,
        observed_at_unix_ms: u64,
        ingested_at_unix_ms: Option<u64>,
    ) -> Result<SystemObservationV1, ProtocolEvidenceErrorV1> {
        require_nonempty(&subject.0, "protocol subject")?;
        if let Some(peer) = &peer {
            require_nonempty(&peer.0, "protocol peer")?;
            if peer == &subject {
                return Err(ProtocolEvidenceErrorV1::SubjectEqualsPeer);
            }
        }
        record.validate()?;

        let mut facts = BTreeMap::new();
        facts.insert(
            "network.protocol.family".into(),
            StateValueV1::Text(record.event.family_name().to_ascii_lowercase()),
        );
        facts.insert(
            "network.evidence.basis".into(),
            StateValueV1::Text(basis_name(record.basis).into()),
        );
        facts.insert(
            "network.subject.role".into(),
            StateValueV1::Text(subject_role_name(subject_role).into()),
        );
        if let Some(transport) = record.transport {
            facts.insert(
                "network.transport".into(),
                StateValueV1::Text(transport_name(transport)),
            );
        }
        if let Some(port) = record.source_port {
            facts.insert("network.source.port".into(), StateValueV1::U64(port.into()));
        }
        if let Some(port) = record.destination_port {
            facts.insert(
                "network.destination.port".into(),
                StateValueV1::U64(port.into()),
            );
        }
        if let Some(packet_index) = record.packet_index {
            facts.insert(
                "network.capture.packet_index".into(),
                StateValueV1::U64(packet_index),
            );
        }
        if self.policy.retain_flow_id {
            if let Some(flow_id) = &record.flow_id {
                facts.insert(
                    "network.flow.id".into(),
                    StateValueV1::Text(self.scrubber.scrub(flow_id).scrubbed_text),
                );
            }
        }
        if self.policy.retain_peer_entity {
            if let Some(peer) = &peer {
                facts.insert(
                    "network.peer.entity".into(),
                    StateValueV1::Text(peer.0.clone()),
                );
            }
        }

        self.event_facts(&record.event, &mut facts);

        let clock = ObservationClockV1 {
            event_time_unix_ms: record.event_time_unix_ms,
            observed_at_unix_ms,
            ingested_at_unix_ms,
            max_age_ms: self.config.max_age_ms,
            clock_uncertainty_ms: record.clock_uncertainty_ms,
        };
        let provenance = ObservationProvenanceV1 {
            source_id: self.config.source_id.clone(),
            source_kind: ObservationSourceKindV1::PacketCapture,
            collector: self.config.collector.clone(),
            collector_version: self.config.collector_version.clone(),
            schema_version: Some(format!(
                "protocol-evidence-v1:{}",
                self.config.parser_profile
            )),
            artifact_digest: Some(record.record_digest.clone()),
        };
        let id = ObservationId(self.observation_id(
            &subject,
            peer.as_ref(),
            subject_role,
            record,
            &clock,
            &provenance,
            &facts,
        )?);
        let observation = SystemObservationV1 {
            id,
            subject,
            provenance,
            clock,
            confidence: self.config.normalization_confidence,
            facts,
        };
        observation.validate()?;
        Ok(observation)
    }

    fn event_facts(&self, event: &ProtocolEventV1, facts: &mut BTreeMap<String, StateValueV1>) {
        match event {
            ProtocolEventV1::Tcp(event) => {
                facts.insert(
                    "network.protocol.tcp.event".into(),
                    StateValueV1::Text(tcp_event_name(event).into()),
                );
            }
            ProtocolEventV1::Dns(event) => match event {
                DnsProtocolEventV1::Query { qtype, name } => {
                    facts.insert(
                        "network.protocol.dns.event".into(),
                        StateValueV1::Text("query".into()),
                    );
                    if let Some(qtype) = qtype {
                        facts.insert(
                            "network.protocol.dns.qtype".into(),
                            StateValueV1::U64((*qtype).into()),
                        );
                    }
                    self.maybe_retain_dns_name(name.as_deref(), facts);
                }
                DnsProtocolEventV1::Response {
                    rcode,
                    answer_count,
                    name,
                } => {
                    facts.insert(
                        "network.protocol.dns.event".into(),
                        StateValueV1::Text("response".into()),
                    );
                    facts.insert(
                        "network.protocol.dns.rcode".into(),
                        StateValueV1::U64((*rcode).into()),
                    );
                    if let Some(answer_count) = answer_count {
                        facts.insert(
                            "network.protocol.dns.answer_count".into(),
                            StateValueV1::U64((*answer_count).into()),
                        );
                    }
                    self.maybe_retain_dns_name(name.as_deref(), facts);
                }
                DnsProtocolEventV1::TimeoutCandidate { attempts } => {
                    facts.insert(
                        "network.protocol.dns.event".into(),
                        StateValueV1::Text("timeout-candidate".into()),
                    );
                    facts.insert(
                        "network.protocol.dns.attempts".into(),
                        StateValueV1::U64((*attempts).into()),
                    );
                }
            },
            ProtocolEventV1::Tls(event) => match event {
                TlsProtocolEventV1::ClientHello {
                    protocol_version,
                    server_name,
                } => {
                    facts.insert(
                        "network.protocol.tls.event".into(),
                        StateValueV1::Text("client-hello".into()),
                    );
                    maybe_insert_scrubbed(
                        facts,
                        "network.protocol.tls.version",
                        protocol_version.as_deref(),
                        &self.scrubber,
                    );
                    if self.policy.retain_tls_server_name {
                        maybe_insert_scrubbed(
                            facts,
                            "network.protocol.tls.server_name",
                            server_name.as_deref(),
                            &self.scrubber,
                        );
                    }
                }
                TlsProtocolEventV1::ServerHello {
                    protocol_version,
                    alpn,
                } => {
                    facts.insert(
                        "network.protocol.tls.event".into(),
                        StateValueV1::Text("server-hello".into()),
                    );
                    maybe_insert_scrubbed(
                        facts,
                        "network.protocol.tls.version",
                        protocol_version.as_deref(),
                        &self.scrubber,
                    );
                    maybe_insert_scrubbed(
                        facts,
                        "network.protocol.tls.alpn",
                        alpn.as_deref(),
                        &self.scrubber,
                    );
                }
                TlsProtocolEventV1::CertificateObserved {
                    sha256_fingerprint,
                    not_after_unix_ms,
                } => {
                    facts.insert(
                        "network.protocol.tls.event".into(),
                        StateValueV1::Text("certificate-observed".into()),
                    );
                    if self.policy.retain_tls_certificate_fingerprint {
                        if let Some(fingerprint) = sha256_fingerprint {
                            facts.insert(
                                "network.protocol.tls.certificate.sha256".into(),
                                StateValueV1::Text(fingerprint.to_ascii_lowercase()),
                            );
                        }
                    }
                    if let Some(not_after) = not_after_unix_ms {
                        facts.insert(
                            "network.protocol.tls.certificate.not_after_unix_ms".into(),
                            StateValueV1::U64(*not_after),
                        );
                    }
                }
                TlsProtocolEventV1::Alert { level, description } => {
                    facts.insert(
                        "network.protocol.tls.event".into(),
                        StateValueV1::Text("alert".into()),
                    );
                    if let Some(level) = level {
                        facts.insert(
                            "network.protocol.tls.alert.level".into(),
                            StateValueV1::U64((*level).into()),
                        );
                    }
                    facts.insert(
                        "network.protocol.tls.alert.description".into(),
                        StateValueV1::U64((*description).into()),
                    );
                }
                TlsProtocolEventV1::HandshakeComplete => {
                    facts.insert(
                        "network.protocol.tls.event".into(),
                        StateValueV1::Text("handshake-complete".into()),
                    );
                }
            },
            ProtocolEventV1::Quic(event) => {
                facts.insert(
                    "network.protocol.quic.event".into(),
                    StateValueV1::Text(quic_event_name(event).into()),
                );
                if let QuicProtocolEventV1::ConnectionClose { error_code } = event {
                    if let Some(error_code) = error_code {
                        facts.insert(
                            "network.protocol.quic.error_code".into(),
                            StateValueV1::U64(*error_code),
                        );
                    }
                }
            }
            ProtocolEventV1::Icmp(event) => {
                facts.insert(
                    "network.protocol.icmp.event".into(),
                    StateValueV1::Text(icmp_event_name(event).into()),
                );
                match event {
                    IcmpProtocolEventV1::DestinationUnreachable { code }
                    | IcmpProtocolEventV1::TimeExceeded { code } => {
                        facts.insert(
                            "network.protocol.icmp.code".into(),
                            StateValueV1::U64((*code).into()),
                        );
                    }
                    IcmpProtocolEventV1::PacketTooBig { mtu }
                    | IcmpProtocolEventV1::FragmentationNeeded { mtu } => {
                        if let Some(mtu) = mtu {
                            facts.insert(
                                "network.protocol.icmp.mtu".into(),
                                StateValueV1::U64((*mtu).into()),
                            );
                        }
                    }
                    _ => {}
                }
            }
            ProtocolEventV1::Marker { marker, code, .. } => {
                facts.insert(
                    "network.protocol.marker".into(),
                    StateValueV1::Text(self.scrubber.scrub(marker).scrubbed_text),
                );
                if let Some(code) = code {
                    facts.insert("network.protocol.marker.code".into(), StateValueV1::U64(*code));
                }
            }
        }
    }

    fn maybe_retain_dns_name(
        &self,
        name: Option<&str>,
        facts: &mut BTreeMap<String, StateValueV1>,
    ) {
        if self.policy.retain_dns_names {
            maybe_insert_scrubbed(
                facts,
                "network.protocol.dns.name",
                name,
                &self.scrubber,
            );
        }
    }

    fn observation_id(
        &self,
        subject: &EntityId,
        peer: Option<&EntityId>,
        subject_role: ProtocolSubjectRoleV1,
        record: &ProtocolEvidenceRecordV1,
        clock: &ObservationClockV1,
        provenance: &ObservationProvenanceV1,
        facts: &BTreeMap<String, StateValueV1>,
    ) -> Result<String, ProtocolEvidenceErrorV1> {
        let normalized = serde_json::to_vec(&(
            "symthaea-protocol-observation-v1",
            subject,
            peer,
            subject_role,
            &record.record_id,
            &record.record_digest,
            record.basis,
            record.transport,
            record.source_port,
            record.destination_port,
            record.packet_index,
            &record.flow_id,
            clock,
            provenance,
            facts,
        ))
        .map_err(|err| ProtocolEvidenceErrorV1::Serialization(err.to_string()))?;
        Ok(format!("protocol:{}", blake3::hash(&normalized).to_hex()))
    }
}

fn maybe_insert_scrubbed(
    facts: &mut BTreeMap<String, StateValueV1>,
    key: &str,
    value: Option<&str>,
    scrubber: &PrivacyScrubber,
) {
    if let Some(value) = value {
        facts.insert(
            key.into(),
            StateValueV1::Text(scrubber.scrub(value).scrubbed_text),
        );
    }
}

fn basis_name(value: ProtocolEvidenceBasisV1) -> &'static str {
    match value {
        ProtocolEvidenceBasisV1::DirectPacket => "direct-packet",
        ProtocolEvidenceBasisV1::FlowDerived => "flow-derived",
        ProtocolEvidenceBasisV1::SessionDerived => "session-derived",
    }
}

fn subject_role_name(value: ProtocolSubjectRoleV1) -> &'static str {
    match value {
        ProtocolSubjectRoleV1::Source => "source",
        ProtocolSubjectRoleV1::Destination => "destination",
        ProtocolSubjectRoleV1::Peer => "peer",
        ProtocolSubjectRoleV1::Unknown => "unknown",
    }
}

fn transport_name(value: TransportProtocolV1) -> String {
    match value {
        TransportProtocolV1::Tcp => "tcp".into(),
        TransportProtocolV1::Udp => "udp".into(),
        TransportProtocolV1::Icmp => "icmp".into(),
        TransportProtocolV1::Icmpv6 => "icmpv6".into(),
        TransportProtocolV1::Other(number) => format!("ip-protocol-{number}"),
    }
}

fn tcp_event_name(value: &TcpProtocolEventV1) -> &'static str {
    match value {
        TcpProtocolEventV1::Syn => "syn",
        TcpProtocolEventV1::SynAck => "syn-ack",
        TcpProtocolEventV1::Ack => "ack",
        TcpProtocolEventV1::Fin => "fin",
        TcpProtocolEventV1::Rst => "rst",
        TcpProtocolEventV1::ZeroWindow => "zero-window",
        TcpProtocolEventV1::RetransmissionCandidate => "retransmission-candidate",
        TcpProtocolEventV1::RepeatedSynCandidate => "repeated-syn-candidate",
    }
}

fn quic_event_name(value: &QuicProtocolEventV1) -> &'static str {
    match value {
        QuicProtocolEventV1::Initial => "initial",
        QuicProtocolEventV1::ZeroRtt => "0-rtt",
        QuicProtocolEventV1::Handshake => "handshake",
        QuicProtocolEventV1::Retry => "retry",
        QuicProtocolEventV1::OneRtt => "1-rtt",
        QuicProtocolEventV1::VersionNegotiation => "version-negotiation",
        QuicProtocolEventV1::ConnectionClose { .. } => "connection-close",
        QuicProtocolEventV1::StatelessResetCandidate => "stateless-reset-candidate",
    }
}

fn icmp_event_name(value: &IcmpProtocolEventV1) -> &'static str {
    match value {
        IcmpProtocolEventV1::EchoRequest => "echo-request",
        IcmpProtocolEventV1::EchoReply => "echo-reply",
        IcmpProtocolEventV1::DestinationUnreachable { .. } => "destination-unreachable",
        IcmpProtocolEventV1::TimeExceeded { .. } => "time-exceeded",
        IcmpProtocolEventV1::PacketTooBig { .. } => "packet-too-big",
        IcmpProtocolEventV1::FragmentationNeeded { .. } => "fragmentation-needed",
    }
}

fn require_nonempty(value: &str, field: &'static str) -> Result<(), ProtocolEvidenceErrorV1> {
    if value.trim().is_empty() {
        Err(ProtocolEvidenceErrorV1::EmptyField(field))
    } else {
        Ok(())
    }
}

fn validate_optional_nonempty(
    value: Option<&str>,
    field: &'static str,
) -> Result<(), ProtocolEvidenceErrorV1> {
    if value.is_some_and(|value| value.trim().is_empty()) {
        Err(ProtocolEvidenceErrorV1::EmptyField(field))
    } else {
        Ok(())
    }
}

fn validate_hex_digest(value: &str, field: &'static str) -> Result<(), ProtocolEvidenceErrorV1> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(ProtocolEvidenceErrorV1::InvalidField(format!(
            "{field} must be exactly 64 hexadecimal characters"
        )));
    }
    Ok(())
}

#[derive(Debug)]
pub enum ProtocolEvidenceErrorV1 {
    EmptyField(&'static str),
    InvalidField(String),
    InvalidConfidence(f32),
    DerivedClaimFromDirectPacket,
    TransportMismatch(String),
    SubjectEqualsPeer,
    Serialization(String),
    State(SystemStateGraphError),
}

impl fmt::Display for ProtocolEvidenceErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "empty protocol evidence field {field}"),
            Self::InvalidField(message) => write!(f, "invalid protocol evidence: {message}"),
            Self::InvalidConfidence(value) => {
                write!(f, "invalid protocol normalization confidence {value}")
            }
            Self::DerivedClaimFromDirectPacket => write!(
                f,
                "derived protocol claim cannot be represented as direct-packet evidence"
            ),
            Self::TransportMismatch(message) => write!(f, "protocol transport mismatch: {message}"),
            Self::SubjectEqualsPeer => write!(f, "protocol subject and peer must be distinct"),
            Self::Serialization(message) => {
                write!(f, "protocol evidence identity serialization failed: {message}")
            }
            Self::State(err) => write!(f, "invalid protocol observation: {err}"),
        }
    }
}

impl Error for ProtocolEvidenceErrorV1 {}

impl From<SystemStateGraphError> for ProtocolEvidenceErrorV1 {
    fn from(value: SystemStateGraphError) -> Self {
        Self::State(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system_state::SystemStateGraphV1;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn adapter(policy: ProtocolObservationRetentionPolicyV1) -> ProtocolObservationAdapterV1 {
        ProtocolObservationAdapterV1::new(
            ProtocolObservationAdapterConfigV1 {
                source_id: "pcap:capture-1".into(),
                collector: "protocol-fixture-parser".into(),
                collector_version: Some("1".into()),
                parser_profile: "fixture-v1".into(),
                max_age_ms: Some(30_000),
                normalization_confidence: 1.0,
            },
            policy,
        )
        .unwrap()
    }

    fn tcp_syn() -> ProtocolEvidenceRecordV1 {
        ProtocolEvidenceRecordV1 {
            record_id: "packet-7".into(),
            record_digest: digest('a'),
            basis: ProtocolEvidenceBasisV1::DirectPacket,
            transport: Some(TransportProtocolV1::Tcp),
            source_port: Some(50_000),
            destination_port: Some(443),
            packet_index: Some(7),
            flow_id: Some("10.0.0.7:50000->10.0.0.9:443".into()),
            event_time_unix_ms: Some(1_700_000_000_000),
            clock_uncertainty_ms: Some(2),
            event: ProtocolEventV1::Tcp(TcpProtocolEventV1::Syn),
        }
    }

    #[test]
    fn direct_packet_normalizes_without_creating_topology() {
        let observation = adapter(Default::default())
            .normalize(
                EntityId("host:client".into()),
                Some(EntityId("service:https".into())),
                ProtocolSubjectRoleV1::Source,
                &tcp_syn(),
                1_700_000_000_010,
                None,
            )
            .unwrap();

        assert_eq!(
            observation.facts["network.protocol.tcp.event"],
            StateValueV1::Text("syn".into())
        );
        assert_eq!(observation.provenance.source_kind, ObservationSourceKindV1::PacketCapture);

        let mut graph = SystemStateGraphV1::new();
        graph.record_observation(observation).unwrap();
        assert_eq!(graph.relations().count(), 0);
    }

    #[test]
    fn default_policy_omits_flow_id_and_sensitive_names() {
        let mut record = tcp_syn();
        record.record_id = "dns-1".into();
        record.record_digest = digest('b');
        record.transport = Some(TransportProtocolV1::Udp);
        record.source_port = Some(53_000);
        record.destination_port = Some(53);
        record.event = ProtocolEventV1::Dns(DnsProtocolEventV1::Query {
            qtype: Some(1),
            name: Some("secret.internal.example".into()),
        });

        let observation = adapter(Default::default())
            .normalize(
                EntityId("host:client".into()),
                Some(EntityId("service:dns".into())),
                ProtocolSubjectRoleV1::Source,
                &record,
                100,
                None,
            )
            .unwrap();
        assert!(!observation.facts.contains_key("network.flow.id"));
        assert!(!observation.facts.contains_key("network.protocol.dns.name"));
    }

    #[test]
    fn derived_anomalies_cannot_masquerade_as_single_packet_facts() {
        let mut record = tcp_syn();
        record.event = ProtocolEventV1::Tcp(TcpProtocolEventV1::RetransmissionCandidate);
        assert!(matches!(
            record.validate(),
            Err(ProtocolEvidenceErrorV1::DerivedClaimFromDirectPacket)
        ));
    }

    #[test]
    fn dns_timeout_requires_derived_basis_and_nonzero_attempts() {
        let mut record = tcp_syn();
        record.transport = Some(TransportProtocolV1::Udp);
        record.event = ProtocolEventV1::Dns(DnsProtocolEventV1::TimeoutCandidate { attempts: 2 });
        assert!(record.validate().is_err());
        record.basis = ProtocolEvidenceBasisV1::FlowDerived;
        assert!(record.validate().is_ok());

        record.event = ProtocolEventV1::Dns(DnsProtocolEventV1::TimeoutCandidate { attempts: 0 });
        assert!(record.validate().is_err());
    }

    #[test]
    fn exact_replay_is_idempotent_but_distinct_raw_digest_is_distinct_evidence() {
        let adapter = adapter(Default::default());
        let first = adapter
            .normalize(
                EntityId("host:a".into()),
                Some(EntityId("service:b".into())),
                ProtocolSubjectRoleV1::Source,
                &tcp_syn(),
                100,
                Some(110),
            )
            .unwrap();
        let replay = adapter
            .normalize(
                EntityId("host:a".into()),
                Some(EntityId("service:b".into())),
                ProtocolSubjectRoleV1::Source,
                &tcp_syn(),
                100,
                Some(110),
            )
            .unwrap();
        assert_eq!(first.id, replay.id);

        let mut second_record = tcp_syn();
        second_record.record_digest = digest('c');
        let second = adapter
            .normalize(
                EntityId("host:a".into()),
                Some(EntityId("service:b".into())),
                ProtocolSubjectRoleV1::Source,
                &second_record,
                100,
                Some(110),
            )
            .unwrap();
        assert_ne!(first.id, second.id);
    }

    #[test]
    fn malformed_record_digest_is_rejected() {
        let mut record = tcp_syn();
        record.record_digest = "not-a-digest".into();
        assert!(record.validate().is_err());
    }

    #[test]
    fn quic_requires_udp_and_stateless_reset_requires_session_context() {
        let mut record = tcp_syn();
        record.event = ProtocolEventV1::Quic(QuicProtocolEventV1::Initial);
        assert!(record.validate().is_err());
        record.transport = Some(TransportProtocolV1::Udp);
        assert!(record.validate().is_ok());

        record.event = ProtocolEventV1::Quic(QuicProtocolEventV1::StatelessResetCandidate);
        assert!(record.validate().is_err());
        record.basis = ProtocolEvidenceBasisV1::SessionDerived;
        assert!(record.validate().is_ok());
    }

    #[test]
    fn tls_names_are_opt_in_but_certificate_fingerprint_is_retained_by_default() {
        let mut hello = tcp_syn();
        hello.record_digest = digest('d');
        hello.event = ProtocolEventV1::Tls(TlsProtocolEventV1::ClientHello {
            protocol_version: Some("TLS1.3".into()),
            server_name: Some("private.example".into()),
        });
        let default = adapter(Default::default())
            .normalize(
                EntityId("host:a".into()),
                None,
                ProtocolSubjectRoleV1::Source,
                &hello,
                100,
                None,
            )
            .unwrap();
        assert!(!default
            .facts
            .contains_key("network.protocol.tls.server_name"));

        let mut certificate = hello;
        certificate.record_digest = digest('e');
        certificate.event = ProtocolEventV1::Tls(TlsProtocolEventV1::CertificateObserved {
            sha256_fingerprint: Some(digest('f')),
            not_after_unix_ms: Some(2_000_000_000_000),
        });
        let observation = adapter(Default::default())
            .normalize(
                EntityId("host:a".into()),
                None,
                ProtocolSubjectRoleV1::Source,
                &certificate,
                100,
                None,
            )
            .unwrap();
        assert!(observation
            .facts
            .contains_key("network.protocol.tls.certificate.sha256"));
    }

    #[test]
    fn canonical_peer_is_context_not_identity_inference() {
        let observation = adapter(Default::default())
            .normalize(
                EntityId("host:a".into()),
                Some(EntityId("service:b".into())),
                ProtocolSubjectRoleV1::Source,
                &tcp_syn(),
                100,
                None,
            )
            .unwrap();
        assert_eq!(observation.subject, EntityId("host:a".into()));
        assert_eq!(
            observation.facts["network.peer.entity"],
            StateValueV1::Text("service:b".into())
        );
    }
}