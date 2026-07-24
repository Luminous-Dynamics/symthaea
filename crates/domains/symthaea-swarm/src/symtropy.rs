//! Bounded Symtropy-facing adapter over the authenticated direct transport.
//!
//! The adapter deliberately contains no Bevy or Lightyear types. A game plugin
//! can move bytes between its Link buffers and this queue while retaining clear
//! lane, delivery, idempotency, peer-binding, and backpressure semantics.

use crate::direct::{
    AuthenticatedDirectMessage, DirectDelivery, DirectEvent, DirectLane, DirectSendReceipt,
    DirectTransport, DirectTransportError, MAX_DIRECT_DATAGRAM_PAYLOAD_BYTES,
    MAX_DIRECT_RELIABLE_PAYLOAD_BYTES, ReliableDeliveryReceipt,
};
use iroh::EndpointId;
use std::collections::VecDeque;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymtropyPacketClass {
    SessionControl,
    PlayerInput,
    StateDelta,
    Checkpoint,
    Telemetry,
    RoboticsCommand,
    AssetTransfer,
}

impl SymtropyPacketClass {
    pub const fn lane(self) -> DirectLane {
        match self {
            Self::SessionControl => DirectLane::CONTROL,
            Self::PlayerInput => DirectLane::PLAYER_INPUT,
            Self::StateDelta | Self::Checkpoint => DirectLane::STATE_SNAPSHOT,
            Self::Telemetry => DirectLane::TELEMETRY,
            Self::RoboticsCommand => DirectLane::ROBOTICS,
            Self::AssetTransfer => DirectLane::ASSET_TRANSFER,
        }
    }

    pub const fn delivery(self) -> DirectDelivery {
        match self {
            Self::PlayerInput | Self::StateDelta | Self::Telemetry => DirectDelivery::Datagram,
            Self::SessionControl
            | Self::Checkpoint
            | Self::RoboticsCommand
            | Self::AssetTransfer => DirectDelivery::Reliable,
        }
    }

    pub const fn requires_idempotency(self) -> bool {
        matches!(
            self,
            Self::SessionControl | Self::Checkpoint | Self::RoboticsCommand | Self::AssetTransfer
        )
    }

    fn from_wire(lane: DirectLane, delivery: DirectDelivery) -> Option<Self> {
        match (lane, delivery) {
            (DirectLane::CONTROL, DirectDelivery::Reliable) => Some(Self::SessionControl),
            (DirectLane::PLAYER_INPUT, DirectDelivery::Datagram) => Some(Self::PlayerInput),
            (DirectLane::STATE_SNAPSHOT, DirectDelivery::Datagram) => Some(Self::StateDelta),
            (DirectLane::STATE_SNAPSHOT, DirectDelivery::Reliable) => Some(Self::Checkpoint),
            (DirectLane::TELEMETRY, DirectDelivery::Datagram) => Some(Self::Telemetry),
            (DirectLane::ROBOTICS, DirectDelivery::Reliable) => Some(Self::RoboticsCommand),
            (DirectLane::ASSET_TRANSFER, DirectDelivery::Reliable) => Some(Self::AssetTransfer),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymtropyPacket {
    pub class: SymtropyPacketClass,
    pub operation_id: Option<Uuid>,
    pub payload: Vec<u8>,
}

impl SymtropyPacket {
    pub fn new(
        class: SymtropyPacketClass,
        operation_id: Option<Uuid>,
        payload: Vec<u8>,
    ) -> Result<Self, SymtropyAdapterError> {
        let packet = Self {
            class,
            operation_id,
            payload,
        };
        packet.validate()?;
        Ok(packet)
    }

    pub fn validate(&self) -> Result<(), SymtropyAdapterError> {
        if self.operation_id == Some(Uuid::nil()) {
            return Err(SymtropyAdapterError::NilOperationId);
        }
        if self.class.requires_idempotency() && self.operation_id.is_none() {
            return Err(SymtropyAdapterError::OperationIdRequired { class: self.class });
        }
        if !self.class.requires_idempotency() && self.operation_id.is_some() {
            return Err(SymtropyAdapterError::UnexpectedOperationId { class: self.class });
        }
        let maximum = match self.class.delivery() {
            DirectDelivery::Reliable => MAX_DIRECT_RELIABLE_PAYLOAD_BYTES,
            DirectDelivery::Datagram => MAX_DIRECT_DATAGRAM_PAYLOAD_BYTES,
        };
        if self.payload.len() > maximum {
            return Err(SymtropyAdapterError::PacketTooLarge {
                class: self.class,
                size: self.payload.len(),
                maximum,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SymtropyAdapterConfig {
    pub maximum_outbound_packets: usize,
    pub maximum_outbound_bytes: usize,
    pub maximum_inbound_packets: usize,
    pub maximum_inbound_bytes: usize,
}

impl Default for SymtropyAdapterConfig {
    fn default() -> Self {
        Self {
            maximum_outbound_packets: 8_192,
            maximum_outbound_bytes: 32 * 1024 * 1024,
            maximum_inbound_packets: 8_192,
            maximum_inbound_bytes: 32 * 1024 * 1024,
        }
    }
}

impl SymtropyAdapterConfig {
    fn validate(self) -> Result<Self, SymtropyAdapterError> {
        if self.maximum_outbound_packets == 0
            || self.maximum_outbound_bytes == 0
            || self.maximum_inbound_packets == 0
            || self.maximum_inbound_bytes == 0
        {
            return Err(SymtropyAdapterError::ZeroCapacity);
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SymtropyAdapterMetrics {
    pub outbound_queued: usize,
    pub outbound_bytes: usize,
    pub inbound_queued: usize,
    pub inbound_bytes: usize,
    pub reliable_sent: u64,
    pub reliable_remote_duplicates: u64,
    pub datagrams_sent: u64,
    pub datagrams_dropped_on_ingress: u64,
    pub foreign_peer_events: u64,
    pub invalid_lane_events: u64,
    pub send_failures: u64,
}

#[derive(Debug, Clone)]
pub enum SymtropyFlushReceipt {
    Reliable(ReliableDeliveryReceipt),
    Datagram(DirectSendReceipt),
}

/// One-peer bounded queue bridge suitable for one Lightyear connection entity.
#[derive(Debug)]
pub struct SymtropyDirectAdapter {
    transport: DirectTransport,
    peer: EndpointId,
    config: SymtropyAdapterConfig,
    outbound: VecDeque<SymtropyPacket>,
    inbound: VecDeque<SymtropyPacket>,
    metrics: SymtropyAdapterMetrics,
}

impl SymtropyDirectAdapter {
    pub fn new(
        transport: DirectTransport,
        peer: EndpointId,
        config: SymtropyAdapterConfig,
    ) -> Result<Self, SymtropyAdapterError> {
        if peer == transport.node_id() {
            return Err(SymtropyAdapterError::SelfPeer);
        }
        Ok(Self {
            transport,
            peer,
            config: config.validate()?,
            outbound: VecDeque::new(),
            inbound: VecDeque::new(),
            metrics: SymtropyAdapterMetrics::default(),
        })
    }

    pub fn peer(&self) -> EndpointId {
        self.peer
    }

    pub fn metrics(&self) -> SymtropyAdapterMetrics {
        self.metrics
    }

    pub fn enqueue(&mut self, packet: SymtropyPacket) -> Result<(), SymtropyAdapterError> {
        packet.validate()?;
        let next_packets = self.metrics.outbound_queued.saturating_add(1);
        let next_bytes = self
            .metrics
            .outbound_bytes
            .saturating_add(packet.payload.len());
        if next_packets > self.config.maximum_outbound_packets
            || next_bytes > self.config.maximum_outbound_bytes
        {
            return Err(SymtropyAdapterError::OutboundQueueFull {
                queued_packets: self.metrics.outbound_queued,
                queued_bytes: self.metrics.outbound_bytes,
            });
        }
        self.metrics.outbound_queued = next_packets;
        self.metrics.outbound_bytes = next_bytes;
        self.outbound.push_back(packet);
        Ok(())
    }

    pub fn pop_inbound(&mut self) -> Option<SymtropyPacket> {
        let packet = self.inbound.pop_front()?;
        self.metrics.inbound_queued = self.metrics.inbound_queued.saturating_sub(1);
        self.metrics.inbound_bytes = self
            .metrics
            .inbound_bytes
            .saturating_sub(packet.payload.len());
        Some(packet)
    }

    /// Ingest one direct event. Reliable overflow is an error; datagram overflow
    /// is explicitly counted and dropped because retransmission is unavailable.
    pub fn ingest_event(&mut self, event: DirectEvent) -> Result<bool, SymtropyAdapterError> {
        let message = match event {
            DirectEvent::Reliable(message) | DirectEvent::Datagram(message) => message,
            _ => return Ok(false),
        };
        if message.author != self.peer {
            self.metrics.foreign_peer_events = self.metrics.foreign_peer_events.saturating_add(1);
            return Err(SymtropyAdapterError::ForeignPeer {
                expected: self.peer,
                received: message.author,
            });
        }
        self.ingest_message(message)
    }

    pub async fn flush_one(
        &mut self,
    ) -> Result<Option<SymtropyFlushReceipt>, SymtropyAdapterError> {
        let Some(packet) = self.outbound.pop_front() else {
            return Ok(None);
        };
        self.metrics.outbound_queued = self.metrics.outbound_queued.saturating_sub(1);
        self.metrics.outbound_bytes = self
            .metrics
            .outbound_bytes
            .saturating_sub(packet.payload.len());

        let result = match packet.class.delivery() {
            DirectDelivery::Reliable => {
                let operation_id =
                    packet
                        .operation_id
                        .ok_or(SymtropyAdapterError::OperationIdRequired {
                            class: packet.class,
                        })?;
                self.transport
                    .send_reliable_idempotent(
                        self.peer,
                        packet.class.lane(),
                        operation_id,
                        packet.payload.clone(),
                    )
                    .await
                    .map(SymtropyFlushReceipt::Reliable)
            }
            DirectDelivery::Datagram => self
                .transport
                .send_datagram(self.peer, packet.class.lane(), packet.payload.clone())
                .await
                .map(SymtropyFlushReceipt::Datagram),
        };

        match result {
            Ok(receipt) => {
                match &receipt {
                    SymtropyFlushReceipt::Reliable(receipt) => {
                        self.metrics.reliable_sent = self.metrics.reliable_sent.saturating_add(1);
                        if receipt.remote_duplicate {
                            self.metrics.reliable_remote_duplicates =
                                self.metrics.reliable_remote_duplicates.saturating_add(1);
                        }
                    }
                    SymtropyFlushReceipt::Datagram(_) => {
                        self.metrics.datagrams_sent = self.metrics.datagrams_sent.saturating_add(1);
                    }
                }
                Ok(Some(receipt))
            }
            Err(error) => {
                self.metrics.send_failures = self.metrics.send_failures.saturating_add(1);
                self.metrics.outbound_queued = self.metrics.outbound_queued.saturating_add(1);
                self.metrics.outbound_bytes = self
                    .metrics
                    .outbound_bytes
                    .saturating_add(packet.payload.len());
                self.outbound.push_front(packet);
                Err(SymtropyAdapterError::Transport(error))
            }
        }
    }

    fn ingest_message(
        &mut self,
        message: AuthenticatedDirectMessage,
    ) -> Result<bool, SymtropyAdapterError> {
        let Some(class) = SymtropyPacketClass::from_wire(message.lane, message.delivery) else {
            self.metrics.invalid_lane_events = self.metrics.invalid_lane_events.saturating_add(1);
            return Err(SymtropyAdapterError::InvalidLaneDelivery {
                lane: message.lane,
                delivery: message.delivery,
            });
        };
        let delivery = message.delivery;
        let packet = SymtropyPacket::new(class, message.operation_id, message.payload)?;
        let next_packets = self.metrics.inbound_queued.saturating_add(1);
        let next_bytes = self
            .metrics
            .inbound_bytes
            .saturating_add(packet.payload.len());
        if next_packets > self.config.maximum_inbound_packets
            || next_bytes > self.config.maximum_inbound_bytes
        {
            if delivery == DirectDelivery::Datagram {
                self.metrics.datagrams_dropped_on_ingress =
                    self.metrics.datagrams_dropped_on_ingress.saturating_add(1);
                return Ok(false);
            }
            return Err(SymtropyAdapterError::InboundReliableQueueFull {
                queued_packets: self.metrics.inbound_queued,
                queued_bytes: self.metrics.inbound_bytes,
                packet,
            });
        }
        self.metrics.inbound_queued = next_packets;
        self.metrics.inbound_bytes = next_bytes;
        self.inbound.push_back(packet);
        Ok(true)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SymtropyAdapterError {
    #[error("Symtropy adapter queue capacities must be greater than zero")]
    ZeroCapacity,
    #[error("Symtropy direct adapter cannot target its own endpoint")]
    SelfPeer,
    #[error("operation ID must not be nil")]
    NilOperationId,
    #[error("{class:?} packets require a stable operation ID")]
    OperationIdRequired { class: SymtropyPacketClass },
    #[error("{class:?} packets must not carry an operation ID")]
    UnexpectedOperationId { class: SymtropyPacketClass },
    #[error("{class:?} packet is {size} bytes; maximum is {maximum}")]
    PacketTooLarge {
        class: SymtropyPacketClass,
        size: usize,
        maximum: usize,
    },
    #[error("outbound queue is full at {queued_packets} packets/{queued_bytes} bytes")]
    OutboundQueueFull {
        queued_packets: usize,
        queued_bytes: usize,
    },
    #[error("reliable inbound queue is full at {queued_packets} packets/{queued_bytes} bytes")]
    InboundReliableQueueFull {
        queued_packets: usize,
        queued_bytes: usize,
        /// The already transport-acknowledged packet. The caller must retain it,
        /// apply it immediately, or fail the connection rather than discard it.
        packet: SymtropyPacket,
    },
    #[error("received event from {received}; adapter is bound to {expected}")]
    ForeignPeer {
        expected: EndpointId,
        received: EndpointId,
    },
    #[error("unsupported direct lane/delivery pair: {lane:?}/{delivery:?}")]
    InvalidLaneDelivery {
        lane: DirectLane,
        delivery: DirectDelivery,
    },
    #[error("direct transport failed: {0}")]
    Transport(DirectTransportError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packet_classes_bind_lane_delivery_and_idempotency() {
        assert_eq!(
            SymtropyPacketClass::PlayerInput.delivery(),
            DirectDelivery::Datagram
        );
        assert!(SymtropyPacketClass::Checkpoint.requires_idempotency());
        assert!(matches!(
            SymtropyPacket::new(SymtropyPacketClass::Checkpoint, None, vec![1]),
            Err(SymtropyAdapterError::OperationIdRequired { .. })
        ));
        assert!(
            SymtropyPacket::new(
                SymtropyPacketClass::Checkpoint,
                Some(Uuid::from_u128(1)),
                vec![1],
            )
            .is_ok()
        );
        assert!(matches!(
            SymtropyPacket::new(
                SymtropyPacketClass::PlayerInput,
                None,
                vec![0; MAX_DIRECT_DATAGRAM_PAYLOAD_BYTES + 1],
            ),
            Err(SymtropyAdapterError::PacketTooLarge { .. })
        ));
    }
}
