// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Non-authoritative systemd boot observer.
//!
//! This process subscribes to structured systemd job signals, queries unit
//! state/properties, and emits the intentionally lossy Symthaea boot protocol.
//! It never parses journal text and failure of this process must never become a
//! machine boot-health fact.

#![forbid(unsafe_code)]

use std::error::Error;
use std::fs;
use std::io;
use std::os::unix::net::UnixDatagram;
use std::path::{Path, PathBuf};

use symthaea_boot_observer::{
    JobOutcome, ObserverConfig, WatchedUnit, classify_job_result, domain_state_from_active_state,
    health_at_boot_ready,
};
use symthaea_boot_protocol::state::BootStateReducer;
use symthaea_boot_protocol::wire::{MAX_WIRE_BYTES, WireMessage};
use symthaea_boot_protocol::{
    BootEvent, BootPhase, BoundedDetail, Criticality, DomainState,
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

    let connection = Connection::system()?;
    let manager = Proxy::new(
        &connection,
        SYSTEMD_DESTINATION,
        SYSTEMD_MANAGER_PATH,
        SYSTEMD_MANAGER_INTERFACE,
    )?;

    // Install the bus match rule before asking systemd to emit lifecycle
    // signals, then query current state. Signals arriving during the initial
    // scan remain queued by the zbus iterator.
    let mut job_removed = manager.receive_signal("JobRemoved")?;
    let _: () = manager.call("Subscribe", &())?;

    let mut emitter = EventEmitter::new(&config)?;

    // The observer may start after storage/filesystems are already ready. Use
    // systemd's monotonic transition timestamps and sort the initial snapshot so
    // we reconstruct the observed history instead of pretending every domain
    // changed at observer startup.
    let mut initial = Vec::new();
    for watched in &config.watched_units {
        match query_unit_observation(&connection, &manager, &watched.unit) {
            Ok(Some(observation)) => initial.push((watched, observation)),
            Ok(None) => {}
            Err(error) => eprintln!(
                "symthaea-boot-observer: initial state unavailable for {}: {error}",
                watched.unit
            ),
        }
    }
    initial.sort_by_key(|(_, observation)| observation.elapsed_ms);
    for (watched, observation) in initial {
        apply_unit_state(&mut emitter, watched, &observation)?;
    }

    while let Some(message) = job_removed.next() {
        let body = message.body();
        let decoded: Result<(u32, OwnedObjectPath, String, String), _> = body.deserialize();
        let (_job_id, _job_path, unit, result) = match decoded {
            Ok(decoded) => decoded,
            Err(error) => {
                eprintln!("symthaea-boot-observer: invalid JobRemoved body: {error}");
                continue;
            }
        };

        let Some(watched) = config.find(&unit) else {
            continue;
        };

        match classify_job_result(&result) {
            JobOutcome::QueryCurrentState => {
                match query_unit_observation(&connection, &manager, &unit) {
                    Ok(Some(observation)) => {
                        if let Err(error) = apply_unit_state(&mut emitter, watched, &observation) {
                            eprintln!(
                                "symthaea-boot-observer: failed to publish state for {unit}: {error}"
                            );
                        }
                    }
                    Ok(None) => {}
                    Err(error) => eprintln!(
                        "symthaea-boot-observer: state query failed for {unit}: {error}"
                    ),
                }
            }
            JobOutcome::Failed => {
                if let Err(error) = emitter.unit_failed(watched, &result, boot_elapsed_ms()) {
                    eprintln!(
                        "symthaea-boot-observer: failed to publish failure for {unit}: {error}"
                    );
                }
            }
            JobOutcome::Ignore => {}
        }
    }

    Ok(())
}

#[derive(Debug)]
struct UnitObservation {
    active_state: String,
    elapsed_ms: u64,
}

fn apply_unit_state(
    emitter: &mut EventEmitter,
    watched: &WatchedUnit,
    observation: &UnitObservation,
) -> Result<(), Box<dyn Error>> {
    let Some(state) = domain_state_from_active_state(&observation.active_state) else {
        eprintln!(
            "symthaea-boot-observer: ignoring unknown ActiveState {:?} for {}",
            observation.active_state, watched.unit
        );
        return Ok(());
    };

    match state {
        DomainState::Starting => {
            emitter.enter_phase(watched.phase, observation.elapsed_ms)?;
            emitter.domain_starting(watched, observation.elapsed_ms)?;
        }
        DomainState::Ready => {
            emitter.enter_phase(watched.phase, observation.elapsed_ms)?;
            emitter.domain_ready(watched, observation.elapsed_ms)?;
            if watched.boot_ready {
                emitter.boot_ready(observation.elapsed_ms)?;
            }
        }
        DomainState::Failed => {
            emitter.unit_failed(watched, "failed", observation.elapsed_ms)?;
        }
        // An inactive/deactivating unit is not evidence that boot is unhealthy.
        DomainState::Pending => {}
        DomainState::Delayed | DomainState::Degraded => {
            // These normalized states are produced by later policy layers rather
            // than inferred directly from systemd's ActiveState vocabulary.
        }
    }

    Ok(())
}

fn query_unit_observation(
    connection: &Connection,
    manager: &Proxy<'_>,
    unit_name: &str,
) -> zbus::Result<Option<UnitObservation>> {
    let unit_path: OwnedObjectPath = match manager.call("GetUnit", &unit_name) {
        Ok(path) => path,
        Err(error) => {
            // A unit may legitimately be absent/not loaded on a given host. The
            // observer treats that as unknown presentation state, never failure.
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

    // Unit timestamps are monotonic microseconds since boot. They let a late
    // observer reconstruct transitions already completed before it started.
    let state_change_us: u64 = unit.get_property("StateChangeTimestampMonotonic").unwrap_or(0);
    let active_enter_us: u64 = unit.get_property("ActiveEnterTimestampMonotonic").unwrap_or(0);
    let timestamp_us = if matches!(active_state.as_str(), "active" | "reloading" | "refreshing") {
        active_enter_us.max(state_change_us)
    } else {
        state_change_us
    };
    let elapsed_ms = if timestamp_us == 0 {
        boot_elapsed_ms()
    } else {
        timestamp_us / 1000
    };

    Ok(Some(UnitObservation {
        active_state,
        elapsed_ms,
    }))
}

struct EventEmitter {
    destination: PathBuf,
    state_path: PathBuf,
    socket: UnixDatagram,
    reducer: BootStateReducer,
    next_sequence: u64,
    boot_ready_emitted: bool,
}

impl EventEmitter {
    fn new(config: &ObserverConfig) -> io::Result<Self> {
        Ok(Self {
            destination: config.output_socket.clone(),
            state_path: config.state_path.clone(),
            socket: UnixDatagram::unbound()?,
            reducer: BootStateReducer::default(),
            next_sequence: 1,
            boot_ready_emitted: false,
        })
    }

    fn enter_phase(
        &mut self,
        phase: Option<BootPhase>,
        elapsed_ms: u64,
    ) -> Result<(), Box<dyn Error>> {
        let Some(phase) = phase else {
            return Ok(());
        };
        let current = self.reducer.snapshot().phase;
        if phase_rank(phase) <= phase_rank(current) {
            return Ok(());
        }
        let sequence = self.sequence();
        self.publish(BootEvent::PhaseEntered {
            sequence,
            elapsed_ms,
            phase,
        })
    }

    fn domain_starting(
        &mut self,
        watched: &WatchedUnit,
        elapsed_ms: u64,
    ) -> Result<(), Box<dyn Error>> {
        let sequence = self.sequence();
        self.publish(BootEvent::DomainStarting {
            sequence,
            elapsed_ms,
            domain: watched.domain,
        })
    }

    fn domain_ready(
        &mut self,
        watched: &WatchedUnit,
        elapsed_ms: u64,
    ) -> Result<(), Box<dyn Error>> {
        let before = self
            .reducer
            .snapshot()
            .domains
            .into_iter()
            .find(|domain| domain.domain == watched.domain)
            .map(|domain| domain.state);
        if before == Some(DomainState::Ready) {
            return Ok(());
        }

        let sequence = self.sequence();
        self.publish(BootEvent::DomainReady {
            sequence,
            elapsed_ms,
            domain: watched.domain,
        })
    }

    fn unit_failed(
        &mut self,
        watched: &WatchedUnit,
        result: &str,
        elapsed_ms: u64,
    ) -> Result<(), Box<dyn Error>> {
        let detail = BoundedDetail::new(format!("{}: {result}", watched.unit)).ok();
        let sequence = self.sequence();

        let event = match watched.criticality {
            Criticality::Critical => BootEvent::DomainFailed {
                sequence,
                elapsed_ms,
                domain: watched.domain,
                criticality: watched.criticality,
                detail,
            },
            Criticality::Informational | Criticality::NonCritical => BootEvent::DomainDegraded {
                sequence,
                elapsed_ms,
                domain: watched.domain,
                criticality: watched.criticality,
                detail,
            },
        };
        self.publish(event)
    }

    fn boot_ready(&mut self, elapsed_ms: u64) -> Result<(), Box<dyn Error>> {
        if self.boot_ready_emitted {
            return Ok(());
        }

        let health = health_at_boot_ready(self.reducer.snapshot().health);
        let sequence = self.sequence();
        self.publish(BootEvent::BootReady {
            sequence,
            elapsed_ms,
            health,
        })?;
        self.boot_ready_emitted = true;
        Ok(())
    }

    fn sequence(&mut self) -> u64 {
        let sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.saturating_add(1);
        sequence
    }

    fn publish(&mut self, event: BootEvent) -> Result<(), Box<dyn Error>> {
        event.validate()?;
        if !self.reducer.apply(&event) {
            return Ok(());
        }

        // Snapshot persistence improves late-consumer recovery, but a state-file
        // permission/disk problem must never terminate live observation.
        if let Err(error) = self.persist_snapshot() {
            eprintln!(
                "symthaea-boot-observer: snapshot persistence unavailable at {}: {error}",
                self.state_path.display()
            );
        }

        let wire = WireMessage::event(event);
        wire.validate()?;
        let encoded = serde_json::to_vec(&wire)?;
        if encoded.len() > MAX_WIRE_BYTES {
            return Err(format!(
                "wire message exceeds {MAX_WIRE_BYTES} bytes: {}",
                encoded.len()
            )
            .into());
        }

        match self.socket.send_to(&encoded, &self.destination) {
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                // The renderer may not have bound its socket yet. Current state
                // can still be recovered from the snapshot when available.
            }
            Err(error) => {
                // Presentation delivery failure is diagnostic only. Never make it
                // a boot-health fact and never terminate the observer loop.
                eprintln!(
                    "symthaea-boot-observer: presentation socket {} unavailable: {error}",
                    self.destination.display()
                );
            }
        }
        Ok(())
    }

    fn persist_snapshot(&self) -> Result<(), Box<dyn Error>> {
        let snapshot = self.reducer.snapshot();
        snapshot.validate()?;
        let encoded = serde_json::to_vec(&snapshot)?;
        if encoded.len() > MAX_WIRE_BYTES {
            return Err(format!(
                "boot snapshot exceeds {MAX_WIRE_BYTES} bytes: {}",
                encoded.len()
            )
            .into());
        }

        if let Some(parent) = self.state_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let temporary = self.state_path.with_extension("json.tmp");
        fs::write(&temporary, encoded)?;
        fs::rename(&temporary, &self.state_path)?;
        Ok(())
    }
}

fn phase_rank(phase: BootPhase) -> u8 {
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
                    println!(
                        "Usage: symthaea-boot-observer [--config PATH] [--print-config]\n\n\
                         Observes structured systemd boot state and emits normalized Spore boot events."
                    );
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument: {other}").into()),
            }
        }

        Ok(Self {
            config,
            print_config,
        })
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

/// `/proc/uptime` is the kernel's monotonic boot age and avoids treating the
/// observer's own process lifetime as system boot progress.
fn boot_elapsed_ms() -> u64 {
    let Ok(contents) = fs::read_to_string("/proc/uptime") else {
        return 0;
    };
    parse_uptime_ms(&contents).unwrap_or(0)
}

fn parse_uptime_ms(contents: &str) -> Option<u64> {
    let seconds = contents.split_whitespace().next()?.parse::<f64>().ok()?;
    if !seconds.is_finite() || seconds.is_sign_negative() {
        return None;
    }
    let millis = seconds * 1000.0;
    Some(if millis >= u64::MAX as f64 {
        u64::MAX
    } else {
        millis as u64
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uptime_parser_is_bounded() {
        assert_eq!(parse_uptime_ms("12.345 90.0"), Some(12_345));
        assert_eq!(parse_uptime_ms("bad"), None);
        assert_eq!(parse_uptime_ms("-1.0 0"), None);
    }

    #[test]
    fn boot_phases_are_strictly_ranked() {
        assert!(phase_rank(BootPhase::Ready) > phase_rank(BootPhase::Graphics));
        assert!(phase_rank(BootPhase::Network) > phase_rank(BootPhase::Filesystems));
    }
}
