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
    BootEvent, BootHealth, BootPhase, BootSnapshot, Criticality, DomainSnapshot, DomainState,
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

        // Job completion is only a trigger to re-read authoritative current
        // state. A queued failure signal must not degrade a unit that has already
        // recovered by the time this observer handles it.
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

fn build_initial_snapshot(
    connection: &Connection,
    manager: &Proxy<'_>,
    config: &ObserverConfig,
) -> Result<BootSnapshot, Box<dyn Error>> {
    let elapsed_ms = boot_elapsed_ms();
    let mut snapshot = BootSnapshot::new(1, std::time::Duration::from_millis(elapsed_ms), BootPhase::Kernel);
    let mut highest_phase = BootPhase::Kernel;
    let mut boot_ready = false;
    let mut aggregate_health = BootHealth::Unknown;

    for watched in &config.watched_units {
        let Some(unit) = query_unit_state(connection, manager, &watched.unit)? else {
            continue;
        };
        let Some(state) = domain_state_from_active_state(&unit.active_state) else {
            continue;
        };

        if state != DomainState::Pending {
            snapshot.domains.push(DomainSnapshot {
                domain: watched.domain,
                state,
                elapsed_ms: unit.transition_elapsed_ms.map(|ms| ms.min(elapsed_ms)),
            });
        }

        if matches!(state, DomainState::Ready | DomainState::Starting) {
            if let Some(phase) = watched.phase {
                if phase_rank(phase) > phase_rank(highest_phase) {
                    highest_phase = phase;
                }
            }
        }

        if state == DomainState::Failed {
            aggregate_health = match watched.criticality {
                Criticality::Critical => BootHealth::Failed,
                Criticality::Informational | Criticality::NonCritical => {
                    if aggregate_health.severity() < BootHealth::Degraded.severity() {
                        BootHealth::Degraded
                    } else {
                        aggregate_health
                    }
                }
            };
        }

        if watched.boot_ready && state == DomainState::Ready {
            boot_ready = true;
        }
    }

    snapshot.phase = if boot_ready { BootPhase::Ready } else { highest_phase };
    snapshot.health = if boot_ready {
        health_at_boot_ready(aggregate_health)
    } else {
        aggregate_health
    };
    snapshot.validate()?;
    Ok(snapshot)
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
    let transition_elapsed_us: u64 = unit.get_property("StateChangeTimestampMonotonic").unwrap_or(0);

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
        let Some(state) = domain_state_from_active_state(&unit.active_state) else {
            return Ok(());
        };

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

        let elapsed_ms = boot_elapsed_ms().max(self.reducer.snapshot().elapsed_ms);
        let sequence = self.sequence();
        let event = match state {
            DomainState::Starting => BootEvent::DomainStarting {
                sequence,
                elapsed_ms,
                domain: watched.domain,
            },
            DomainState::Ready if matches!(previous, DomainState::Failed | DomainState::Degraded) => {
                BootEvent::DomainRecovered {
                    sequence,
                    elapsed_ms,
                    domain: watched.domain,
                }
            }
            DomainState::Ready => BootEvent::DomainReady {
                sequence,
                elapsed_ms,
                domain: watched.domain,
            },
            DomainState::Failed => match watched.criticality {
                Criticality::Critical => BootEvent::DomainFailed {
                    sequence,
                    elapsed_ms,
                    domain: watched.domain,
                    criticality: watched.criticality,
                    detail: None,
                },
                Criticality::Informational | Criticality::NonCritical => BootEvent::DomainDegraded {
                    sequence,
                    elapsed_ms,
                    domain: watched.domain,
                    criticality: watched.criticality,
                    detail: None,
                },
            },
            DomainState::Pending | DomainState::Delayed | DomainState::Degraded => return Ok(()),
        };
        self.publish_event(event)?;

        if let Some(phase) = watched.phase {
            let current = self.reducer.snapshot().phase;
            if phase_rank(phase) > phase_rank(current) {
                let sequence = self.sequence();
                self.publish_event(BootEvent::PhaseEntered {
                    sequence,
                    elapsed_ms: boot_elapsed_ms().max(self.reducer.snapshot().elapsed_ms),
                    phase,
                })?;
            }
        }

        if watched.boot_ready && state == DomainState::Ready {
            self.boot_ready()?;
        }
        Ok(())
    }

    fn boot_ready(&mut self) -> Result<(), Box<dyn Error>> {
        if self.boot_ready_emitted {
            return Ok(());
        }
        self.boot_ready_emitted = true;
        let health = health_at_boot_ready(self.reducer.snapshot().health);
        let sequence = self.sequence();
        self.publish_event(BootEvent::BootReady {
            sequence,
            elapsed_ms: boot_elapsed_ms().max(self.reducer.snapshot().elapsed_ms),
            health,
        })
    }

    fn sequence(&mut self) -> u64 {
        let sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.saturating_add(1);
        sequence
    }

    fn publish_event(&mut self, event: BootEvent) -> Result<(), Box<dyn Error>> {
        event.validate()?;
        if !self.reducer.try_apply(&event)? {
            return Ok(());
        }
        let wire = WireMessage::event(self.observation, event);
        self.persist_wire_snapshot(&WireMessage::snapshot(
            self.observation,
            self.reducer.snapshot(),
        ))?;
        self.send_wire(&wire);
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
    if millis >= u64::MAX as f64 { u64::MAX } else { millis as u64 }
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
