// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic lockstep protocol (same-arch milestone).
//!
//! This module provides a minimal lockstep tick pipeline:
//! - each peer broadcasts a `TickPacket` containing its input commands + state hash
//! - a deterministic leader is selected per epoch (round-robin over sorted peer IDs)
//! - if a non-leader detects a hash mismatch vs the epoch leader, it requests a resync
//! - the leader responds with a bitwise `WorldSnapshot` (ordered by `NetId`)
//! - the requester applies the snapshot and continues
//!
//! Note: this is the **same-arch** milestone: it detects divergence and provides
//! resync, but it does not attempt to force cross-arch bitwise determinism.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use nalgebra::{SMatrix, SVector};
use serde::{Deserialize, Serialize};
use symtropy_math::{Bivector, Point, Rotor};
use symtropy_physics::body::{BodyType, NetId};
use symtropy_physics::integrator;
use symtropy_physics::world::PhysicsWorld;

use crate::peer::PeerId;
use crate::transport::{Channel, Transport, TransportEvent};

pub const LOCKSTEP_PROTOCOL_VERSION: u32 = 1;

/// Fixed-size bit-vector with serde support for const-generic `D`.
///
/// `serde` does not implement `Serialize/Deserialize` for `[T; D]` when `D` is a
/// const-generic parameter (it only has impls for concrete lengths). We keep a
/// fixed-size in-memory representation while serializing as a `Seq` of length
/// `D`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitsVec<const D: usize>(pub [u64; D]);

impl<const D: usize> Serialize for BitsVec<D> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeSeq;
        let mut seq = serializer.serialize_seq(Some(D))?;
        for v in &self.0 {
            seq.serialize_element(v)?;
        }
        seq.end()
    }
}

impl<'de, const D: usize> Deserialize<'de> for BitsVec<D> {
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: serde::Deserializer<'de>,
    {
        struct V<const D: usize>;

        impl<'de, const D: usize> serde::de::Visitor<'de> for V<D> {
            type Value = BitsVec<D>;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "an array of {D} u64 elements")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let mut out = [0u64; D];
                for i in 0..D {
                    out[i] = seq
                        .next_element::<u64>()?
                        .ok_or_else(|| serde::de::Error::invalid_length(i, &self))?;
                }
                if seq.next_element::<u64>()?.is_some() {
                    return Err(serde::de::Error::invalid_length(D + 1, &self));
                }
                Ok(BitsVec(out))
            }
        }

        deserializer.deserialize_seq(V::<D>)
    }
}

/// Fixed-size bit-matrix (row-major) with serde support for const-generic `D`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitsMat<const D: usize>(pub [[u64; D]; D]);

impl<const D: usize> Serialize for BitsMat<D> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeSeq;
        let len = D * D;
        let mut seq = serializer.serialize_seq(Some(len))?;
        for r in 0..D {
            for c in 0..D {
                seq.serialize_element(&self.0[r][c])?;
            }
        }
        seq.end()
    }
}

impl<'de, const D: usize> Deserialize<'de> for BitsMat<D> {
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: serde::Deserializer<'de>,
    {
        struct V<const D: usize>;

        impl<'de, const D: usize> serde::de::Visitor<'de> for V<D> {
            type Value = BitsMat<D>;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "a flat array of {} u64 elements", D * D)
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let mut out = [[0u64; D]; D];
                let len = D * D;
                for i in 0..len {
                    let v = seq
                        .next_element::<u64>()?
                        .ok_or_else(|| serde::de::Error::invalid_length(i, &self))?;
                    out[i / D][i % D] = v;
                }
                if seq.next_element::<u64>()?.is_some() {
                    return Err(serde::de::Error::invalid_length(len + 1, &self));
                }
                Ok(BitsMat(out))
            }
        }

        deserializer.deserialize_seq(V::<D>)
    }
}

#[derive(Debug, Clone)]
pub struct LockstepConfig {
    /// Number of ticks per epoch (leader rotates once per epoch).
    pub epoch_len_ticks: u64,
}

impl Default for LockstepConfig {
    fn default() -> Self {
        Self {
            epoch_len_ticks: 256,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickPacket<const D: usize> {
    pub protocol_version: u32,
    pub epoch: u64,
    pub tick: u64,
    pub leader: u64,
    pub sender: u64,
    /// Hash of the sender's world state at the start of this tick.
    pub state_hash: u64,
    /// Input commands for this tick (must be sorted deterministically).
    pub commands: Vec<Command<D>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Command<const D: usize> {
    ApplyForce {
        net_id: u64,
        force_bits: BitsVec<D>,
    },
    ApplyImpulse {
        net_id: u64,
        impulse_bits: BitsVec<D>,
    },
    SetLinearVelocity {
        net_id: u64,
        velocity_bits: BitsVec<D>,
    },
    SetAngularVelocity {
        net_id: u64,
        matrix_bits: BitsMat<D>,
    },
    Wake {
        net_id: u64,
    },
}

impl<const D: usize> Command<D> {
    fn kind_id(&self) -> u8 {
        match self {
            Self::ApplyForce { .. } => 0,
            Self::ApplyImpulse { .. } => 1,
            Self::SetLinearVelocity { .. } => 2,
            Self::SetAngularVelocity { .. } => 3,
            Self::Wake { .. } => 4,
        }
    }

    fn net_id(&self) -> u64 {
        match self {
            Self::ApplyForce { net_id, .. } => *net_id,
            Self::ApplyImpulse { net_id, .. } => *net_id,
            Self::SetLinearVelocity { net_id, .. } => *net_id,
            Self::SetAngularVelocity { net_id, .. } => *net_id,
            Self::Wake { net_id } => *net_id,
        }
    }
}

pub fn sort_commands_deterministic<const D: usize>(cmds: &mut [Command<D>]) {
    cmds.sort_by(|a, b| {
        let ka = (a.net_id(), a.kind_id());
        let kb = (b.net_id(), b.kind_id());
        ka.cmp(&kb)
    });
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResyncRequest {
    pub protocol_version: u32,
    pub epoch: u64,
    pub tick: u64,
    pub leader: u64,
    pub requester: u64,
    pub requester_state_hash: u64,
    pub leader_state_hash: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResyncResponse<const D: usize> {
    pub protocol_version: u32,
    pub epoch: u64,
    pub tick: u64,
    pub leader: u64,
    pub snapshot: WorldSnapshot<D>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LockstepMessage<const D: usize> {
    TickPacket(TickPacket<D>),
    ResyncRequest(ResyncRequest),
    ResyncResponse(ResyncResponse<D>),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorldSnapshot<const D: usize> {
    /// Bodies, ordered by `net_id` ascending.
    pub bodies: Vec<BodySnapshot<D>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BodySnapshot<const D: usize> {
    pub net_id: u64,
    pub body_type: u8,
    pub translation: BitsVec<D>,
    pub rotation: BitsMat<D>,
    pub linear_velocity: BitsVec<D>,
    pub angular_velocity: BitsMat<D>,
    pub sleeping: bool,
    pub sleep_counter: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SnapshotError {
    MissingNetId { handle: usize },
    UnknownBody { net_id: u64 },
    BodyTypeMismatch { net_id: u64 },
}

pub fn capture_world_snapshot<const D: usize>(
    world: &PhysicsWorld<D>,
) -> Result<WorldSnapshot<D>, SnapshotError> {
    let mut bodies = Vec::with_capacity(world.bodies.len());
    for body in &world.bodies {
        let Some(net_id) = world.net_id_for_handle(body.handle) else {
            return Err(SnapshotError::MissingNetId {
                handle: body.handle.0,
            });
        };

        let rot = body.transform.rotation.to_matrix();
        let ang = body.angular_velocity.to_matrix();

        bodies.push(BodySnapshot {
            net_id: net_id.0,
            body_type: match body.body_type {
                BodyType::Dynamic => 0,
                BodyType::Static => 1,
                BodyType::Kinematic => 2,
            },
            translation: BitsVec(std::array::from_fn(|i| {
                body.transform.translation.0[i].to_bits()
            })),
            rotation: BitsMat(std::array::from_fn(|r| {
                std::array::from_fn(|c| rot[(r, c)].to_bits())
            })),
            linear_velocity: BitsVec(std::array::from_fn(|i| body.linear_velocity[i].to_bits())),
            angular_velocity: BitsMat(std::array::from_fn(|r| {
                std::array::from_fn(|c| ang[(r, c)].to_bits())
            })),
            sleeping: body.sleeping,
            sleep_counter: body.sleep_counter,
        });
    }

    bodies.sort_by_key(|b| b.net_id);
    Ok(WorldSnapshot { bodies })
}

pub fn apply_world_snapshot<const D: usize>(
    world: &mut PhysicsWorld<D>,
    snapshot: &WorldSnapshot<D>,
) -> Result<(), SnapshotError> {
    for b in &snapshot.bodies {
        let Some(handle) = world.handle_for_net_id(NetId(b.net_id)) else {
            return Err(SnapshotError::UnknownBody { net_id: b.net_id });
        };
        let Some(body) = world.body_mut(handle) else {
            return Err(SnapshotError::UnknownBody { net_id: b.net_id });
        };

        let expected_type = match b.body_type {
            0 => BodyType::Dynamic,
            1 => BodyType::Static,
            2 => BodyType::Kinematic,
            _ => return Err(SnapshotError::BodyTypeMismatch { net_id: b.net_id }),
        };
        if body.body_type != expected_type {
            return Err(SnapshotError::BodyTypeMismatch { net_id: b.net_id });
        }

        let t =
            SVector::<f64, D>::from(std::array::from_fn(|i| f64::from_bits(b.translation.0[i])));
        body.transform.translation = Point(t);

        let rot = SMatrix::<f64, D, D>::from_fn(|r, c| f64::from_bits(b.rotation.0[r][c]));
        body.transform.rotation = Rotor::from_matrix(rot);

        let v = SVector::<f64, D>::from(std::array::from_fn(|i| {
            f64::from_bits(b.linear_velocity.0[i])
        }));
        body.linear_velocity = v;

        let ang = SMatrix::<f64, D, D>::from_fn(|r, c| f64::from_bits(b.angular_velocity.0[r][c]));
        body.angular_velocity = Bivector::from_matrix(&ang);

        body.sleeping = b.sleeping;
        body.sleep_counter = b.sleep_counter;
        body.clear_accumulators();
    }
    Ok(())
}

pub fn snapshot_hash_fnv1a64<const D: usize>(snapshot: &WorldSnapshot<D>) -> u64 {
    let mut h = Fnv1a64::new();
    h.update_u32(snapshot.bodies.len() as u32);
    for b in &snapshot.bodies {
        h.update_u64(b.net_id);
        h.update_u8(b.body_type);
        for v in b.translation.0 {
            h.update_u64(v);
        }
        for row in b.rotation.0 {
            for v in row {
                h.update_u64(v);
            }
        }
        for v in b.linear_velocity.0 {
            h.update_u64(v);
        }
        for row in b.angular_velocity.0 {
            for v in row {
                h.update_u64(v);
            }
        }
        h.update_u8(if b.sleeping { 1 } else { 0 });
        h.update_u32(b.sleep_counter);
    }
    h.finish()
}

pub fn leader_for_epoch_round_robin(epoch: u64, members: &[PeerId]) -> Option<PeerId> {
    if members.is_empty() {
        return None;
    }
    Some(members[(epoch as usize) % members.len()])
}

#[derive(Debug)]
pub enum LockstepProgress {
    /// Waiting for peer tick packets for the current tick.
    WaitingForPeers,
    /// Waiting for resync snapshot.
    WaitingForResync,
    /// Stepped simulation and advanced to next tick.
    Stepped,
}

/// Lockstep session for deterministic tick coordination.
pub struct LockstepSession<T: Transport, const D: usize> {
    pub transport: T,
    pub config: LockstepConfig,
    pub tick: u64,
    peers: BTreeSet<PeerId>,
    sent_tick_packet_for_tick: Option<u64>,
    packets_by_tick: BTreeMap<u64, BTreeMap<PeerId, TickPacket<D>>>,
    resync_in_progress: bool,
}

impl<T: Transport, const D: usize> LockstepSession<T, D> {
    pub fn new(transport: T, config: LockstepConfig) -> Self {
        let local = transport.local_peer_id();
        let mut peers = BTreeSet::new();
        peers.insert(local);
        Self {
            transport,
            config,
            tick: 0,
            peers,
            sent_tick_packet_for_tick: None,
            packets_by_tick: BTreeMap::new(),
            resync_in_progress: false,
        }
    }

    pub fn join(&mut self, room_id: &str) -> Result<(), String> {
        self.transport.connect(room_id)
    }

    pub fn local_peer_id(&self) -> PeerId {
        self.transport.local_peer_id()
    }

    fn epoch(&self) -> u64 {
        if self.config.epoch_len_ticks == 0 {
            return 0;
        }
        self.tick / self.config.epoch_len_ticks
    }

    fn members_sorted(&self) -> Vec<PeerId> {
        self.peers.iter().copied().collect()
    }

    fn leader_for_current_epoch(&self) -> PeerId {
        leader_for_epoch_round_robin(self.epoch(), &self.members_sorted())
            .unwrap_or(self.local_peer_id())
    }

    fn handle_transport_events(&mut self, world: &mut PhysicsWorld<D>) -> Result<(), String> {
        let local = self.local_peer_id();
        let events = self.transport.poll();
        for ev in events {
            match ev {
                TransportEvent::PeerConnected(peer) => {
                    self.peers.insert(peer);
                }
                TransportEvent::PeerDisconnected(peer) => {
                    self.peers.remove(&peer);
                    for packets in self.packets_by_tick.values_mut() {
                        packets.remove(&peer);
                    }
                }
                TransportEvent::Message(msg) => {
                    // Learn peers opportunistically.
                    self.peers.insert(msg.from);
                    if msg.channel != Channel::Reliable {
                        continue;
                    }
                    let Ok(m) = rmp_serde::from_slice::<LockstepMessage<D>>(&msg.data) else {
                        continue;
                    };
                    match m {
                        LockstepMessage::TickPacket(p) => {
                            self.packets_by_tick
                                .entry(p.tick)
                                .or_default()
                                .insert(PeerId(p.sender), p);
                        }
                        LockstepMessage::ResyncRequest(req) => {
                            let leader = self.leader_for_current_epoch();
                            if leader != local {
                                continue;
                            }
                            if req.tick != self.tick || req.epoch != self.epoch() {
                                continue;
                            }
                            // Respond with a bitwise snapshot of our current tick boundary state.
                            let snapshot = capture_world_snapshot(world)
                                .map_err(|e| format!("capture snapshot failed: {e:?}"))?;
                            let resp = ResyncResponse {
                                protocol_version: LOCKSTEP_PROTOCOL_VERSION,
                                epoch: req.epoch,
                                tick: req.tick,
                                leader: leader.0,
                                snapshot,
                            };
                            let msg = LockstepMessage::ResyncResponse(resp);
                            if let Ok(data) = rmp_serde::to_vec(&msg) {
                                let _ = self.transport.send(
                                    PeerId(req.requester),
                                    Channel::Reliable,
                                    &data,
                                );
                            }
                        }
                        LockstepMessage::ResyncResponse(resp) => {
                            if resp.tick != self.tick || resp.epoch != self.epoch() {
                                continue;
                            }
                            apply_world_snapshot(world, &resp.snapshot)
                                .map_err(|e| format!("apply snapshot failed: {e:?}"))?;

                            // Update our local packet hash to the post-resync state.
                            if let Ok(snapshot) = capture_world_snapshot(world) {
                                let hash = snapshot_hash_fnv1a64(&snapshot);
                                if let Some(p) = self
                                    .packets_by_tick
                                    .entry(self.tick)
                                    .or_default()
                                    .get_mut(&local)
                                {
                                    p.state_hash = hash;
                                }
                            }

                            self.resync_in_progress = false;
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn build_local_tick_packet(
        &self,
        world: &PhysicsWorld<D>,
        mut commands: Vec<Command<D>>,
    ) -> Result<TickPacket<D>, SnapshotError> {
        sort_commands_deterministic(&mut commands);
        let snapshot = capture_world_snapshot(world)?;
        let state_hash = snapshot_hash_fnv1a64(&snapshot);

        let leader = self.leader_for_current_epoch();
        Ok(TickPacket {
            protocol_version: LOCKSTEP_PROTOCOL_VERSION,
            epoch: self.epoch(),
            tick: self.tick,
            leader: leader.0,
            sender: self.local_peer_id().0,
            state_hash,
            commands,
        })
    }

    fn broadcast_tick_packet(&mut self, packet: &TickPacket<D>) {
        let msg = LockstepMessage::TickPacket(packet.clone());
        if let Ok(data) = rmp_serde::to_vec(&msg) {
            let _ = self.transport.broadcast(Channel::Reliable, &data);
        }
    }

    fn request_resync(&mut self, leader: PeerId, leader_hash: u64, requester_hash: u64) {
        let req = ResyncRequest {
            protocol_version: LOCKSTEP_PROTOCOL_VERSION,
            epoch: self.epoch(),
            tick: self.tick,
            leader: leader.0,
            requester: self.local_peer_id().0,
            requester_state_hash: requester_hash,
            leader_state_hash: leader_hash,
        };
        let msg: LockstepMessage<D> = LockstepMessage::ResyncRequest(req);
        if let Ok(data) = rmp_serde::to_vec(&msg) {
            let _ = self.transport.send(leader, Channel::Reliable, &data);
        }
        self.resync_in_progress = true;
    }

    fn apply_commands(world: &mut PhysicsWorld<D>, commands: &[Command<D>]) -> Result<(), String> {
        for cmd in commands {
            match cmd {
                Command::ApplyForce { net_id, force_bits } => {
                    let Some(handle) = world.handle_for_net_id(NetId(*net_id)) else {
                        return Err(format!("unknown net_id {net_id}"));
                    };
                    let force = SVector::<f64, D>::from(std::array::from_fn(|i| {
                        f64::from_bits(force_bits.0[i])
                    }));
                    let Some(body) = world.body_mut(handle) else {
                        return Err(format!("missing body for net_id {net_id}"));
                    };
                    body.apply_force(force);
                }
                Command::ApplyImpulse {
                    net_id,
                    impulse_bits,
                } => {
                    let Some(handle) = world.handle_for_net_id(NetId(*net_id)) else {
                        return Err(format!("unknown net_id {net_id}"));
                    };
                    let impulse = SVector::<f64, D>::from(std::array::from_fn(|i| {
                        f64::from_bits(impulse_bits.0[i])
                    }));
                    let Some(body) = world.body_mut(handle) else {
                        return Err(format!("missing body for net_id {net_id}"));
                    };
                    integrator::apply_impulse(body, &impulse);
                }
                Command::SetLinearVelocity {
                    net_id,
                    velocity_bits,
                } => {
                    let Some(handle) = world.handle_for_net_id(NetId(*net_id)) else {
                        return Err(format!("unknown net_id {net_id}"));
                    };
                    let vel = SVector::<f64, D>::from(std::array::from_fn(|i| {
                        f64::from_bits(velocity_bits.0[i])
                    }));
                    let Some(body) = world.body_mut(handle) else {
                        return Err(format!("missing body for net_id {net_id}"));
                    };
                    body.linear_velocity = vel;
                }
                Command::SetAngularVelocity {
                    net_id,
                    matrix_bits,
                } => {
                    let Some(handle) = world.handle_for_net_id(NetId(*net_id)) else {
                        return Err(format!("unknown net_id {net_id}"));
                    };
                    let mat =
                        SMatrix::<f64, D, D>::from_fn(|r, c| f64::from_bits(matrix_bits.0[r][c]));
                    let bv = Bivector::from_matrix(&mat);
                    let Some(body) = world.body_mut(handle) else {
                        return Err(format!("missing body for net_id {net_id}"));
                    };
                    body.angular_velocity = bv;
                }
                Command::Wake { net_id } => {
                    let Some(handle) = world.handle_for_net_id(NetId(*net_id)) else {
                        return Err(format!("unknown net_id {net_id}"));
                    };
                    let Some(body) = world.body_mut(handle) else {
                        return Err(format!("missing body for net_id {net_id}"));
                    };
                    body.wake();
                }
            }
        }
        Ok(())
    }

    fn merged_commands_for_current_tick(&self) -> Vec<Command<D>> {
        let packets = self.packets_by_tick.get(&self.tick);
        let mut packets: Vec<_> = packets
            .into_iter()
            .flat_map(|m| m.values().cloned())
            .collect();
        packets.sort_by_key(|p| p.sender);
        let mut commands = Vec::new();
        for p in packets {
            commands.extend(p.commands);
        }
        commands
    }

    /// Pump networking + determinism checks. Returns whether we stepped this tick.
    pub fn pump(
        &mut self,
        world: &mut PhysicsWorld<D>,
        dt: f64,
        local_commands: Vec<Command<D>>,
    ) -> Result<LockstepProgress, String> {
        let local = self.local_peer_id();
        self.handle_transport_events(world)?;

        // Don't advance if we know there are connected peers but we haven't learned their IDs yet.
        let expected_members = (self.transport.peer_count() as usize) + 1;
        if self.peers.len() < expected_members {
            // Still broadcast our tick packet (so others can learn us), but don't step.
            if self.sent_tick_packet_for_tick != Some(self.tick) {
                let packet = self
                    .build_local_tick_packet(world, local_commands)
                    .map_err(|e| format!("build tick packet failed: {e:?}"))?;
                self.broadcast_tick_packet(&packet);
                self.packets_by_tick
                    .entry(self.tick)
                    .or_default()
                    .insert(local, packet);
                self.sent_tick_packet_for_tick = Some(self.tick);
            }
            return Ok(LockstepProgress::WaitingForPeers);
        }

        if self.sent_tick_packet_for_tick != Some(self.tick) {
            let packet = self
                .build_local_tick_packet(world, local_commands)
                .map_err(|e| format!("build tick packet failed: {e:?}"))?;
            self.broadcast_tick_packet(&packet);
            self.packets_by_tick
                .entry(self.tick)
                .or_default()
                .insert(local, packet);
            self.sent_tick_packet_for_tick = Some(self.tick);
        }

        let leader = self.leader_for_current_epoch();
        if !self.resync_in_progress && leader != local {
            let Some(packets) = self.packets_by_tick.get(&self.tick) else {
                return Ok(LockstepProgress::WaitingForPeers);
            };
            let Some(leader_packet) = packets.get(&leader) else {
                return Ok(LockstepProgress::WaitingForPeers);
            };
            let Some(local_packet) = packets.get(&local) else {
                return Ok(LockstepProgress::WaitingForPeers);
            };
            if local_packet.state_hash != leader_packet.state_hash {
                self.request_resync(leader, leader_packet.state_hash, local_packet.state_hash);
                return Ok(LockstepProgress::WaitingForResync);
            }
        }

        if self.resync_in_progress {
            return Ok(LockstepProgress::WaitingForResync);
        }

        // Lockstep barrier: step only when we have packets from all known peers.
        let packets = self.packets_by_tick.entry(self.tick).or_default();
        let have_all = self.peers.iter().all(|p| packets.contains_key(p));
        if !have_all {
            return Ok(LockstepProgress::WaitingForPeers);
        }

        let commands = self.merged_commands_for_current_tick();
        Self::apply_commands(world, &commands)?;
        world.step(dt);

        self.packets_by_tick.remove(&self.tick);
        self.tick += 1;
        self.sent_tick_packet_for_tick = None;

        Ok(LockstepProgress::Stepped)
    }
}

struct Fnv1a64 {
    state: u64,
}

impl Fnv1a64 {
    fn new() -> Self {
        Self {
            state: 14695981039346656037,
        }
    }

    fn update(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.state ^= *b as u64;
            self.state = self.state.wrapping_mul(1099511628211);
        }
    }

    fn update_u8(&mut self, v: u8) {
        self.update(&[v]);
    }

    fn update_u32(&mut self, v: u32) {
        self.update(&v.to_le_bytes());
    }

    fn update_u64(&mut self, v: u64) {
        self.update(&v.to_le_bytes());
    }

    fn finish(self) -> u64 {
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loopback::loopback_pair;
    use symtropy_physics::RigidBody;

    fn build_world(order: &[u64]) -> PhysicsWorld<3> {
        let gravity = SVector::from([0.0, -10.0, 0.0]);
        let mut world = PhysicsWorld::<3>::new(gravity);
        let bodies: Vec<_> = order
            .iter()
            .map(|id| {
                let net_id = NetId(*id);
                let x = *id as f64;
                let body = RigidBody::<3>::dynamic_sphere(
                    symtropy_physics::BodyHandle(999),
                    Point::new([x, 1.0, 0.0]),
                    0.5,
                    1.0,
                );
                (net_id, body)
            })
            .collect();
        world.add_bodies_deterministic(bodies).unwrap();
        world
    }

    fn force_cmd(net_id: u64, fx: f64) -> Command<3> {
        Command::ApplyForce {
            net_id,
            force_bits: BitsVec([fx.to_bits(), 0.0f64.to_bits(), 0.0f64.to_bits()]),
        }
    }

    #[test]
    fn leader_round_robin_rotates() {
        let members = vec![PeerId(0), PeerId(1), PeerId(7)];
        assert_eq!(leader_for_epoch_round_robin(0, &members), Some(PeerId(0)));
        assert_eq!(leader_for_epoch_round_robin(1, &members), Some(PeerId(1)));
        assert_eq!(leader_for_epoch_round_robin(2, &members), Some(PeerId(7)));
        assert_eq!(leader_for_epoch_round_robin(3, &members), Some(PeerId(0)));
    }

    #[test]
    fn lockstep_two_peers_stay_in_sync_and_resync_on_divergence() {
        let (a_transport, b_transport) = loopback_pair();

        // Keep epoch long enough that peer 0 remains leader for the whole test.
        let config = LockstepConfig { epoch_len_ticks: 8 };
        let mut a = LockstepSession::<_, 3>::new(a_transport, config.clone());
        let mut b = LockstepSession::<_, 3>::new(b_transport, config);

        a.join("room").unwrap();
        b.join("room").unwrap();

        let mut wa = build_world(&[2, 3, 4]);
        let mut wb = build_world(&[4, 2, 3]); // different insertion order on purpose

        let dt = 1.0 / 64.0;

        // Run a few clean ticks.
        for t in 0..6u64 {
            // Deterministic per-peer inputs.
            let ca = vec![force_cmd(2, (t as f64) / 128.0)];
            let cb = vec![force_cmd(3, -(t as f64) / 128.0)];

            // Pump until both step.
            let mut stepped_a = false;
            let mut stepped_b = false;
            for _ in 0..8 {
                if !stepped_a {
                    stepped_a = matches!(
                        a.pump(&mut wa, dt, ca.clone()).unwrap(),
                        LockstepProgress::Stepped
                    );
                }
                if !stepped_b {
                    stepped_b = matches!(
                        b.pump(&mut wb, dt, cb.clone()).unwrap(),
                        LockstepProgress::Stepped
                    );
                }
                if stepped_a && stepped_b {
                    break;
                }
            }
            assert!(stepped_a && stepped_b, "failed to step at t={t}");
        }

        let ha = snapshot_hash_fnv1a64(&capture_world_snapshot(&wa).unwrap());
        let hb = snapshot_hash_fnv1a64(&capture_world_snapshot(&wb).unwrap());
        assert_eq!(ha, hb);

        // Introduce divergence on B: mutate a body's velocity without broadcasting it.
        let handle = wb.handle_for_net_id(NetId(2)).unwrap();
        wb.body_mut(handle).unwrap().linear_velocity[0] += 1.0;

        // Next tick should detect mismatch and resync back to leader.
        let ca = vec![force_cmd(2, 0.0)];
        let cb = vec![force_cmd(3, 0.0)];

        let mut stepped_a = false;
        let mut stepped_b = false;
        for _ in 0..16 {
            if !stepped_a {
                stepped_a = matches!(
                    a.pump(&mut wa, dt, ca.clone()).unwrap(),
                    LockstepProgress::Stepped
                );
            }
            if !stepped_b {
                stepped_b = matches!(
                    b.pump(&mut wb, dt, cb.clone()).unwrap(),
                    LockstepProgress::Stepped
                );
            }
            if stepped_a && stepped_b {
                break;
            }
        }
        assert!(stepped_a && stepped_b, "failed to step after resync");

        let ha2 = snapshot_hash_fnv1a64(&capture_world_snapshot(&wa).unwrap());
        let hb2 = snapshot_hash_fnv1a64(&capture_world_snapshot(&wb).unwrap());
        assert_eq!(ha2, hb2);
    }
}
