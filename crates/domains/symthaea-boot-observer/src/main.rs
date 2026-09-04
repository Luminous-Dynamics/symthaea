// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Non-authoritative systemd boot observer.
//!
//! This process subscribes to structured systemd job signals, queries unit
//! state, and emits the intentionally lossy Symthaea boot protocol. It never
//! parses journal text and failure of this process must never become a dependency
//! of the machine's boot path.

#![forbid(unsafe_code)]

use std::error::Error;
use std::fs;
use std::io;
use std::os::unix::net::UnixDatagram;
use std::path::{Path, PathBuf};

use symthaea_boot_observer::{
    ObserverConfig, WatchedUnit, domain_state_from_active_state, health_at_boot_ready,
};
use symthaea_boot_protocol::state::BootStateReducer;
use symthaea_boot_protocol::wire::{
    MAX_WIRE_BYTES, ObservationId, WireMessage, validate_datagram_size,
};
use symthaea_boot_protocol::{
    BootDomain, BootEvent, BootHealth, BootPhase, BootSnapshot, Criticality, DomainSnapshot,
    DomainState,
};
use zbus::blocking::{Connection, Proxy};
use zbus::zvariant::OwnedObjectPath;

const SYSTEMD_DESTINATION: &str = "org.freedesktop.systemd1";
const SYSTEMD_MANAGER_PATH: &str = "/org/freedesktop/systemd1";
const SYSTEMD_MANAGER_INTERFACE: &str = "org.freedesktop.systemd1.Manager";
const SYSTEMD_UNIT_INTERFACE: &str = "org.freedesktop.systemd1.Unit";

fn main() {
    if let Err(error) = run() {
        eprintln!("symthaea-boot-observer: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    let config = load_config(args.config.as_deref())?;

    if args.print_config {
        println!("{}", serde_json::to_string_pretty(&config)?);
        return Ok(());
    }

    config.validate()?;

    let connection = Connection::system()?;
    let manager = Proxy::new(
        &connection,
        SYSTEMD_DESTINATION,
        SYSTEMD_MANAGER_PATH,
        SYSTEMD_MANAGER_INTERFACE,
    )?;

    // Subscribe before the initial scan. Signals that arrive during reconstruction
    // remain queued; each queued signal is later treated only as a trigger to
    // re-read current structured state.
    let mut job_removed = manager.receive_signal("JobRemoved")?;
    let _: () = manager.call("Subscribe", &())?;

    let observation = derive_observation_id();
    let initial = build_initial_snapshot(&connection, &manager, &config)?;
    let mut emitter = EventEmitter::new(&config, observation, initial)?;
    emitter.publish_snapshot()?;

    while let Some(message) = job_removed.next() {
        let body = message.body();
        let decoded: Result<(u32, OwnedObjectPath, String, String), _> = body.deserialize();
        let (_job_id, _job_path, unit, _result) = match decoded {
            Ok(decoded) => decoded,
            Err(error) => {
                eprintln!("symthaea-boot-observer: invalid JobRemoved body: {error}");
                continue;
            }
        };

        let Some(watched) = config.find(&unit) else {
            continue;
        };

        // Never trust a queued textual job result as current health. Re-read the
        // unit's structured ActiveState so an already-recovered unit cannot be
        // rendered as failed because an older D-Bus signal was still queued.
        match query_unit_state(&connection, &manager, &unit) {
            Ok(Some(state)) => {
                if let Err(error) = emitter.apply_current_state(watched, &state) {
                    eprintln!("symthaea-boot-observer: failed to publish state for {unit}: {error}");
                }
            }
            Ok(None) => {}
            Err(error) => {
                eprintln!("symthaea-boot-observer: state query failed for {unit}: {error}")
            }
        }
    }

    Ok(())
}

#[derive(Debug, Clone)]
struct UnitState {
    active_state: String,
    transition_elapsed_ms: Option<u64>,
}

/// Convert a raw structured systemd state into the protocol domain state for a
/// particular watched unit.
///
/// A non-critical systemd failure is represented as `Degraded`, not `Failed`,
/// regardless of whether it was discovered during initial reconstruction or by
/// a later signal. This keeps semantic truth independent of observer start time.
fn semantic_domain_state(watched: &WatchedUnit, raw: DomainState) -> DomainState {
    if raw == DomainState::Failed && watched.criticality != Criticality::Critical {
        DomainState::Degraded
    } else {
        raw
    }
}

/// Resolve aggregate health from the currently represented domain facts.
///
/// This is intentionally current-state based at the `BootReady` boundary. A
/// historical failure that has fully recovered may stop claiming Failed or
/// Degraded, but recovery alone does not prove Normal: absent current negative
/// evidence resolves to Unknown unless Normal was independently established.
fn aggregate_health_from_domains(
    domains: &[DomainSnapshot],
    previously_established: BootHealth,
) -> BootHealth {
    let mut aggregate = BootHealth::Unknown;
    for domain in domains {
        let observed = match domain.state {
            DomainState::Failed => Some(BootHealth::Failed),
            DomainState::Degraded => Some(BootHealth::Degraded),
            DomainState::Delayed => Some(BootHealth::Delayed),
            DomainState::Pending | DomainState::Starting | DomainState::Ready => None,
        };
        if let Some(observed) = observed {
            if observed.severity() > aggregate.severity() {
                aggregate = observed;
            }
        }
    }

    if aggregate == BootHealth::Unknown && previously_established == BootHealth::Normal {
        BootHealth::Normal
    } else {
        aggregate
    }
}

fn build_initial_snapshot(
    connection: &Connection,
    manager: &Proxy<'_>,
    config: &ObserverConfig,
) -> Result<BootSnapshot, Box<dyn Error>> {
    let elapsed_ms = boot_elapsed_ms();
    let mut snapshot = BootSnapshot::new(
        1,
        std::time::Duration::from_millis(elapsed_ms),
        BootPhase::Kernel,
    );
    let mut highest_phase = BootPhase::Kernel;
    let mut boot_ready = false;

    for watched in &config.watched_units {
        let Some(unit) = query_unit_state(connection, manager, &watched.unit)? else {
            continue;
        };
        let Some(raw_state) = domain_state_from_active_state(&unit.active_state) else {
            continue;
        };
        let state = semantic_domain_state(watched, raw_state);

        if state != DomainState::Pending {
            upsert_domain(
                &mut snapshot,
                watched.domain,
                state,
                unit.transition_elapsed_ms.map(|ms| ms.min(elapsed_ms)),
            );
        }

        if matches!(state, DomainState::Ready | DomainState::Starting) {
            if let Some(phase) = watched.phase {
                if phase_rank(phase) > phase_rank(highest_phase) {
                    highest_phase = phase;
                }
            }
        }

        if watched.boot_ready && state == DomainState::Ready {
            boot_ready = true;
        }
    }

    let aggregate_health = aggregate_health_from_domains(&snapshot.domains, BootHealth::Unknown);
    snapshot.phase = if boot_ready {
        BootPhase::Ready
    } else {
        highest_phase
    };
    snapshot.health = if boot_ready {
        health_at_boot_ready(aggregate_health)
    } else {
        aggregate_health
    };
    snapshot.validate()?;
    Ok(snapshot)
}

fn upsert_domain(
    snapshot: &mut BootSnapshot,
    domain: BootDomain,
    state: DomainState,
    elapsed_ms: Option<u64>,
) {
    if let Some(existing) = snapshot.domains.iter_mut().find(|item| item.domain == domain) {
        if domain_state_rank(state) >= domain_state_rank(existing.state) {
            existing.state = state;
            existing.elapsed_ms = elapsed_ms.or(existing.elapsed_ms);
        }
        return;
    }
    snapshot.domains.push(DomainSnapshot {
        domain,
        state,
        elapsed_ms,
    });
}

fn query_unit_state(
    connection: &Connection,
    manager: &Proxy<'_>,
    unit_name: &str,
) -> zbus::Result<Option<UnitState>> {
    let unit_path: OwnedObjectPath = match manager.call("GetUnit", &unit_name) {
        Ok(path) => path,
        Err(error) => {
            let text = error.to_string();
            if text.contains("NoSuchUnit") || text.contains("not loaded") {
                return Ok(None);
            }
            return Err(error);
        }
    };

    let unit = Proxy::new(
        connection,
        SYSTEMD_DESTINATION,
        unit_path.as_str(),
        SYSTEMD_UNIT_INTERFACE,
    )?;
    let active_state: String = unit.get_property("ActiveState")?;
    let transition_elapsed_us: u64 = unit
        .get_property("StateChangeTimestampMonotonic")
        .unwrap_or(0);

    Ok(Some(UnitState {
        active_state,
        transition_elapsed_ms: (transition_elapsed_us > 0).then_some(transition_elapsed_us / 1000),
    }))
}

struct EventEmitter {
    destination: PathBuf,
    state_path: PathBuf,
    socket: UnixDatagram,
    observation: ObservationId,
    reducer: BootStateReducer,
    next_sequence: u64,
    boot_ready_emitted: bool,
}

impl EventEmitter {
    fn new(
        config: &ObserverConfig,
        observation: ObservationId,
        initial: BootSnapshot,
    ) -> Result<Self, Box<dyn Error>> {
        if let Some(parent) = config.state_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let next_sequence = initial.sequence.saturating_add(1);
        let boot_ready_emitted = initial.phase == BootPhase::Ready;
        let mut reducer = BootStateReducer::default();
        reducer.try_replace(initial)?;

        Ok(Self {
            destination: config.output_socket.clone(),
            state_path: config.state_path.clone(),
            socket: UnixDatagram::unbound()?,
            observation,
            reducer,
            next_sequence,
            boot_ready_emitted,
        })
    }

    fn publish_snapshot(&self) -> Result<(), Box<dyn Error>> {
        let wire = WireMessage::snapshot(self.observation, self.reducer.snapshot());
        self.persist_wire_snapshot(&wire)?;
        self.send_wire(&wire);
        Ok(())
    }

    fn apply_current_state(
        &mut self,
        watched: &WatchedUnit,
        unit: &UnitState,
    ) -> Result<(), Box<dyn Error>> {
        let Some(raw_state) = domain_state_from_active_state(&unit.active_state) else {
            return Ok(());
        };
        let state = semantic_domain_state(watched, raw_state);

        let previous = self
            .reducer
            .snapshot()
            .domains
            .into_iter()
            .find(|domain| domain.domain == watched.domain)
            .map(|domain| domain.state)
            .unwrap_or(DomainState::Pending);

        if state == previous {
            if watched.boot_ready && state == DomainState::Ready && !self.boot_ready_emitted {
                self.boot_ready()?;
            }
            return Ok(());
        }

        // One structured systemd observation can imply several protocol facts
        // (domain state, semantic phase, and BootReady). Build all of them first
        // and commit them as one durable observer transaction so persistence
        // failure cannot leave memory, sequence, or readiness half-advanced.
        let elapsed_ms = boot_elapsed_ms().max(self.reducer.snapshot().elapsed_ms);
        let mut next_sequence = self.next_sequence;
        let mut events = Vec::with_capacity(3);

        let domain_event = match state {
            DomainState::Starting => BootEvent::DomainStarting {
                sequence: next_sequence,
                elapsed_ms,
                domain: watched.domain,
            },
            DomainState::Ready if matches!(previous, DomainState::Failed | DomainState::Degraded) => {
                BootEvent::DomainRecovered {
                    sequence: next_sequence,
                    elapsed_ms,
                    domain: watched.domain,
                }
            }
            DomainState::Ready => BootEvent::DomainReady {
                sequence: next_sequence,
                elapsed_ms,
                domain: watched.domain,
            },
            DomainState::Delayed => BootEvent::DomainDelayed {
                sequence: next_sequence,
                elapsed_ms,
                domain: watched.domain,
            },
            DomainState::Degraded => BootEvent::DomainDegraded {
                sequence: next_sequence,
                elapsed_ms,
                domain: watched.domain,
                criticality: watched.criticality,
                detail: None,
            },
            DomainState::Failed => BootEvent::DomainFailed {
                sequence: next_sequence,
                elapsed_ms,
                domain: watched.domain,
                criticality: watched.criticality,
                detail: None,
            },
            DomainState::Pending => return Ok(()),
        };
        events.push(domain_event);
        next_sequence = next_sequence.saturating_add(1);

        let mut preview = self.reducer.clone();
        for event in &events {
            if !preview.try_apply(event)? {
                return Err(io::Error::other("observer transaction produced a stale event").into());
            }
        }

        if let Some(phase) = watched.phase {
            let current_phase = preview.snapshot().phase;
            if phase_rank(phase) > phase_rank(current_phase) {
                let event = BootEvent::PhaseEntered {
                    sequence: next_sequence,
                    elapsed_ms,
                    phase,
                };
                if !preview.try_apply(&event)? {
                    return Err(io::Error::other("observer transaction produced a stale phase event").into());
                }
                events.push(event);
                next_sequence = next_sequence.saturating_add(1);
            }
        }

        let emits_boot_ready = watched.boot_ready
            && state == DomainState::Ready
            && !self.boot_ready_emitted;
        if emits_boot_ready {
            let current = preview.snapshot();
            let health = health_at_boot_ready(aggregate_health_from_domains(
                &current.domains,
                current.health,
            ));
            let event = BootEvent::BootReady {
                sequence: next_sequence,
                elapsed_ms,
                health,
            };
            if !preview.try_apply(&event)? {
                return Err(io::Error::other("observer transaction produced a stale Ready event").into());
            }
            events.push(event);
        }

        self.publish_transaction(events, preview)?;
        if emits_boot_ready {
            self.boot_ready_emitted = true;
        }
        Ok(())
    }

    fn boot_ready(&mut self) -> Result<(), Box<dyn Error>> {
        if self.boot_ready_emitted {
            return Ok(());
        }
        let current = self.reducer.snapshot();
        let health = health_at_boot_ready(aggregate_health_from_domains(
            &current.domains,
            current.health,
        ));
        let event = BootEvent::BootReady {
            sequence: self.next_sequence,
            elapsed_ms: boot_elapsed_ms().max(current.elapsed_ms),
            health,
        };
        let mut preview = self.reducer.clone();
        if !preview.try_apply(&event)? {
            return Err(io::Error::other("observer transaction produced a stale Ready event").into());
        }
        self.publish_transaction(vec![event], preview)?;
        self.boot_ready_emitted = true;
        Ok(())
    }

    /// Persist the final snapshot for one normalized observation before making
    /// any in-memory state visible. Datagram events are best-effort presentation
    /// delivery after the durable side channel and reducer commit agree.
    fn publish_transaction(
        &mut self,
        events: Vec<BootEvent>,
        candidate: BootStateReducer,
    ) -> Result<(), Box<dyn Error>> {
        let Some(last_sequence) = events.last().map(BootEvent::sequence) else {
            return Ok(());
        };
        for event in &events {
            event.validate()?;
        }

        let snapshot_wire = WireMessage::snapshot(self.observation, candidate.snapshot());
        self.persist_wire_snapshot(&snapshot_wire)?;

        // From here onward no fallible truth-state operation remains. Commit the
        // exact candidate whose snapshot was persisted, then advance sequence.
        self.reducer = candidate;
        self.next_sequence = last_sequence.saturating_add(1);

        for event in events {
            self.send_wire(&WireMessage::event(self.observation, event));
        }
        Ok(())
    }

    fn persist_wire_snapshot(&self, wire: &WireMessage) -> Result<(), Box<dyn Error>> {
        wire.validate()?;
        let encoded = serde_json::to_vec(wire)?;
        validate_datagram_size(&encoded)?;
        let temporary = self.state_path.with_extension("json.tmp");
        fs::write(&temporary, encoded)?;
        fs::rename(&temporary, &self.state_path)?;
        Ok(())
    }

    fn send_wire(&self, wire: &WireMessage) {
        let Ok(encoded) = serde_json::to_vec(wire) else {
            return;
        };
        if validate_datagram_size(&encoded).is_err() || encoded.len() > MAX_WIRE_BYTES {
            return;
        }
        match self.socket.send_to(&encoded, &self.destination) {
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => eprintln!(
                "symthaea-boot-observer: presentation socket {} unavailable: {error}",
                self.destination.display()
            ),
        }
    }
}

#[derive(Debug)]
struct Args {
    config: Option<PathBuf>,
    print_config: bool,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut config = None;
        let mut print_config = false;
        let mut args = std::env::args().skip(1);
        while let Some(argument) = args.next() {
            match argument.as_str() {
                "--config" => {
                    let Some(path) = args.next() else {
                        return Err("--config requires a path".into());
                    };
                    config = Some(PathBuf::from(path));
                }
                "--print-config" => print_config = true,
                "--help" | "-h" => {
                    println!("Usage: symthaea-boot-observer [--config PATH] [--print-config]");
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument: {other}").into()),
            }
        }
        Ok(Self { config, print_config })
    }
}

fn load_config(path: Option<&Path>) -> Result<ObserverConfig, Box<dyn Error>> {
    let config = match path {
        Some(path) => serde_json::from_slice(&fs::read(path)?)?,
        None => ObserverConfig::builtin(),
    };
    config.validate()?;
    Ok(config)
}

fn derive_observation_id() -> ObservationId {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-boot-observation-v1\0");
    if let Ok(boot_id) = fs::read("/proc/sys/kernel/random/boot_id") {
        hasher.update(&boot_id);
    }
    // `/proc/sys/kernel/random/uuid` yields a fresh kernel-generated nonce on
    // each read. If unavailable, PID + monotonic uptime still distinguish the
    // normal restart case; this ID is lineage separation, not authentication.
    if let Ok(observer_nonce) = fs::read("/proc/sys/kernel/random/uuid") {
        hasher.update(&observer_nonce);
    }
    hasher.update(&std::process::id().to_le_bytes());
    hasher.update(&boot_elapsed_ms().to_le_bytes());
    let digest = hasher.finalize();
    let mut bytes = [0_u8; 16];
    bytes.copy_from_slice(&digest.as_bytes()[..16]);
    ObservationId::from_bytes(bytes)
}

fn boot_elapsed_ms() -> u64 {
    let Ok(contents) = fs::read_to_string("/proc/uptime") else {
        return 0;
    };
    let Some(seconds) = contents.split_whitespace().next() else {
        return 0;
    };
    let Ok(seconds) = seconds.parse::<f64>() else {
        return 0;
    };
    if !seconds.is_finite() || seconds.is_sign_negative() {
        return 0;
    }
    let millis = seconds * 1000.0;
    if millis >= u64::MAX as f64 {
        u64::MAX
    } else {
        millis as u64
    }
}

const fn phase_rank(phase: BootPhase) -> u8 {
    match phase {
        BootPhase::Kernel => 0,
        BootPhase::Initrd => 1,
        BootPhase::Storage => 2,
        BootPhase::Filesystems => 3,
        BootPhase::Security => 4,
        BootPhase::Network => 5,
        BootPhase::Services => 6,
        BootPhase::Graphics => 7,
        BootPhase::Session => 8,
        BootPhase::Ready => 9,
    }
}

const fn domain_state_rank(state: DomainState) -> u8 {
    match state {
        DomainState::Pending => 0,
        DomainState::Starting => 1,
        DomainState::Ready => 2,
        DomainState::Delayed => 3,
        DomainState::Degraded => 4,
        DomainState::Failed => 5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    fn temp_directory(suffix: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after Unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "symthaea-boot-observer-{}-{nonce}-{suffix}",
            std::process::id()
        ))
    }

    fn test_emitter(directory: &Path) -> EventEmitter {
        fs::create_dir_all(directory).unwrap();
        let mut config = ObserverConfig::builtin();
        config.state_path = directory.join("state.json");
        config.output_socket = directory.join("missing.sock");
        let initial = BootSnapshot::new(1, Duration::from_millis(1), BootPhase::Kernel);
        EventEmitter::new(
            &config,
            ObservationId::from_bytes([0xA5; 16]),
            initial,
        )
        .unwrap()
    }

    #[test]
    fn duplicate_unit_domains_collapse_in_snapshot() {
        let mut snapshot = BootSnapshot::new(
            1,
            std::time::Duration::from_millis(100),
            BootPhase::Graphics,
        );
        upsert_domain(
            &mut snapshot,
            BootDomain::Graphics,
            DomainState::Ready,
            Some(80),
        );
        upsert_domain(
            &mut snapshot,
            BootDomain::Graphics,
            DomainState::Failed,
            Some(90),
        );
        assert_eq!(snapshot.domains.len(), 1);
        assert_eq!(snapshot.domains[0].state, DomainState::Failed);
        snapshot.validate().unwrap();
    }

    #[test]
    fn lower_severity_duplicate_does_not_hide_failure() {
        let mut snapshot = BootSnapshot::new(
            1,
            std::time::Duration::from_millis(100),
            BootPhase::Graphics,
        );
        upsert_domain(
            &mut snapshot,
            BootDomain::Graphics,
            DomainState::Failed,
            Some(70),
        );
        upsert_domain(
            &mut snapshot,
            BootDomain::Graphics,
            DomainState::Ready,
            Some(90),
        );
        assert_eq!(snapshot.domains[0].state, DomainState::Failed);
    }

    #[test]
    fn noncritical_failure_has_same_semantics_before_or_after_observer_start() {
        let watched = WatchedUnit::new(
            "network.target",
            BootDomain::Network,
            Some(BootPhase::Network),
            Criticality::NonCritical,
            false,
        );
        assert_eq!(
            semantic_domain_state(&watched, DomainState::Failed),
            DomainState::Degraded
        );

        let critical = WatchedUnit::new(
            "local-fs.target",
            BootDomain::Filesystems,
            Some(BootPhase::Filesystems),
            Criticality::Critical,
            false,
        );
        assert_eq!(
            semantic_domain_state(&critical, DomainState::Failed),
            DomainState::Failed
        );
    }

    #[test]
    fn ready_health_uses_current_domains_and_does_not_invent_normal() {
        let degraded = vec![DomainSnapshot {
            domain: BootDomain::Network,
            state: DomainState::Degraded,
            elapsed_ms: Some(10),
        }];
        assert_eq!(
            aggregate_health_from_domains(&degraded, BootHealth::Failed),
            BootHealth::Degraded
        );

        let recovered = vec![DomainSnapshot {
            domain: BootDomain::Network,
            state: DomainState::Ready,
            elapsed_ms: Some(20),
        }];
        assert_eq!(
            aggregate_health_from_domains(&recovered, BootHealth::Failed),
            BootHealth::Unknown
        );
        assert_eq!(
            aggregate_health_from_domains(&recovered, BootHealth::Normal),
            BootHealth::Normal
        );
    }

    #[test]
    fn persistence_failure_rolls_back_whole_observation_transaction() {
        let directory = temp_directory("transaction");
        let mut emitter = test_emitter(&directory);
        let watched = WatchedUnit::new(
            "local-fs-pre.target",
            BootDomain::Storage,
            Some(BootPhase::Storage),
            Criticality::Critical,
            false,
        );
        let unit = UnitState {
            active_state: "active".to_string(),
            transition_elapsed_ms: Some(2),
        };
        let initial_sequence = emitter.next_sequence;
        let blocker = emitter.state_path.with_extension("json.tmp");
        fs::create_dir(&blocker).unwrap();

        assert!(emitter.apply_current_state(&watched, &unit).is_err());
        let failed_snapshot = emitter.reducer.snapshot();
        assert_eq!(failed_snapshot.sequence, 1);
        assert_eq!(failed_snapshot.phase, BootPhase::Kernel);
        assert!(failed_snapshot.domains.is_empty());
        assert_eq!(emitter.next_sequence, initial_sequence);

        fs::remove_dir(&blocker).unwrap();
        emitter.apply_current_state(&watched, &unit).unwrap();
        let committed = emitter.reducer.snapshot();
        assert_eq!(committed.sequence, initial_sequence + 1);
        assert_eq!(committed.phase, BootPhase::Storage);
        assert_eq!(committed.domains.len(), 1);
        assert_eq!(committed.domains[0].state, DomainState::Ready);
        assert_eq!(emitter.next_sequence, initial_sequence + 2);

        let persisted: WireMessage =
            serde_json::from_slice(&fs::read(&emitter.state_path).unwrap()).unwrap();
        match persisted {
            WireMessage::Snapshot { snapshot, .. } => assert_eq!(snapshot, committed),
            WireMessage::Event { .. } => panic!("observer state path must contain a snapshot"),
        }

        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn ready_publication_failure_remains_retryable() {
        let directory = temp_directory("ready-transaction");
        let mut emitter = test_emitter(&directory);
        let watched = WatchedUnit::new(
            "graphical.target",
            BootDomain::Session,
            Some(BootPhase::Ready),
            Criticality::Critical,
            true,
        );
        let unit = UnitState {
            active_state: "active".to_string(),
            transition_elapsed_ms: Some(2),
        };
        let initial_sequence = emitter.next_sequence;
        let blocker = emitter.state_path.with_extension("json.tmp");
        fs::create_dir(&blocker).unwrap();

        assert!(emitter.apply_current_state(&watched, &unit).is_err());
        assert!(!emitter.boot_ready_emitted);
        assert_eq!(emitter.reducer.snapshot().phase, BootPhase::Kernel);
        assert_eq!(emitter.next_sequence, initial_sequence);

        fs::remove_dir(&blocker).unwrap();
        emitter.apply_current_state(&watched, &unit).unwrap();
        let committed = emitter.reducer.snapshot();
        assert!(emitter.boot_ready_emitted);
        assert_eq!(committed.phase, BootPhase::Ready);
        assert_eq!(committed.health, BootHealth::Unknown);
        assert_eq!(committed.sequence, initial_sequence + 2);
        assert_eq!(emitter.next_sequence, initial_sequence + 3);

        let persisted: WireMessage =
            serde_json::from_slice(&fs::read(&emitter.state_path).unwrap()).unwrap();
        match persisted {
            WireMessage::Snapshot { snapshot, .. } => assert_eq!(snapshot, committed),
            WireMessage::Event { .. } => panic!("observer state path must contain a snapshot"),
        }

        let _ = fs::remove_dir_all(directory);
    }
}
