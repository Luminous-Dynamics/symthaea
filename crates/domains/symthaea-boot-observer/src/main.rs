// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Non-authoritative systemd boot observer.
//!
//! This process subscribes to structured systemd job signals, queries unit
//! `ActiveState`, and emits the intentionally lossy Symthaea boot protocol. It
//! never parses journal text and failure of this process must never become a
//! dependency of the machine's boot path.

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
    BootEvent, BootHealth, BootPhase, BoundedDetail, Criticality, DomainState,
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

    // Install the bus match rule before asking systemd to emit lifecycle
    // signals, then query current state. Signals that arrive while the initial
    // scan runs remain queued by the zbus iterator.
    let mut job_removed = manager.receive_signal("JobRemoved")?;
    let _: () = manager.call("Subscribe", &())?;

    let mut emitter = EventEmitter::new(&config)?;

    for watched in &config.watched_units {
        match query_active_state(&connection, &manager, &watched.unit) {
            Ok(Some(active_state)) => {
                apply_unit_state(&mut emitter, watched, &active_state)?;
            }
            Ok(None) => {}
            Err(error) => {
                eprintln!(
                    "symthaea-boot-observer: initial state unavailable for {}: {error}",
                    watched.unit
                );
            }
        }
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
                match query_active_state(&connection, &manager, &unit) {
                    Ok(Some(active_state)) => {
                        if let Err(error) = apply_unit_state(&mut emitter, watched, &active_state) {
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
                if let Err(error) = emitter.unit_failed(watched, &result) {
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

fn apply_unit_state(
    emitter: &mut EventEmitter,
    watched: &WatchedUnit,
    active_state: &str,
) -> Result<(), Box<dyn Error>> {
    let Some(state) = domain_state_from_active_state(active_state) else {
        eprintln!(
            "symthaea-boot-observer: ignoring unknown ActiveState {:?} for {}",
            active_state, watched.unit
        );
        return Ok(());
    };

    match state {
        DomainState::Starting => {
            emitter.enter_phase(watched.phase)?;
            emitter.domain_starting(watched)?;
        }
        DomainState::Ready => {
            emitter.enter_phase(watched.phase)?;
            emitter.domain_ready(watched)?;
            if watched.boot_ready {
                emitter.boot_ready()?;
            }
        }
        DomainState::Failed => emitter.unit_failed(watched, "failed")?,
        // An inactive/deactivating unit is not evidence that boot is unhealthy.
        DomainState::Pending => {}
        DomainState::Delayed | DomainState::Degraded => {
            // These normalized states are produced by later policy layers rather
            // than inferred directly from systemd's ActiveState vocabulary.
        }
    }

    Ok(())
}

fn query_active_state(
    connection: &Connection,
    manager: &Proxy<'_>,
    unit_name: &str,
) -> zbus::Result<Option<String>> {
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
    Ok(Some(active_state))
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
        if let Some(parent) = config.state_path.parent() {
            fs::create_dir_all(parent)?;
        }

        Ok(Self {
            destination: config.output_socket.clone(),
            state_path: config.state_path.clone(),
            socket: UnixDatagram::unbound()?,
            reducer: BootStateReducer::default(),
            next_sequence: 1,
            boot_ready_emitted: false,
        })
    }

    fn enter_phase(&mut self, phase: Option<BootPhase>) -> Result<(), Box<dyn Error>> {
        let Some(phase) = phase else {
            return Ok(());
        };
        if self.reducer.snapshot().phase == phase {
            return Ok(());
        }
        let sequence = self.sequence();
        self.publish(BootEvent::PhaseEntered {
            sequence,
            elapsed_ms: boot_elapsed_ms(),
            phase,
        })
    }

    fn domain_starting(&mut self, watched: &WatchedUnit) -> Result<(), Box<dyn Error>> {
        let sequence = self.sequence();
        self.publish(BootEvent::DomainStarting {
            sequence,
            elapsed_ms: boot_elapsed_ms(),
            domain: watched.domain,
        })
    }

    fn domain_ready(&mut self, watched: &WatchedUnit) -> Result<(), Box<dyn Error>> {
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
            elapsed_ms: boot_elapsed_ms(),
            domain: watched.domain,
        })
    }

    fn unit_failed(
        &mut self,
        watched: &WatchedUnit,
        result: &str,
    ) -> Result<(), Box<dyn Error>> {
        let detail = BoundedDetail::new(format!("{}: {result}", watched.unit)).ok();
        let sequence = self.sequence();
        let elapsed_ms = boot_elapsed_ms();

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

    fn boot_ready(&mut self) -> Result<(), Box<dyn Error>> {
        if self.boot_ready_emitted {
            return Ok(());
        }
        self.boot_ready_emitted = true;

        let health = health_at_boot_ready(self.reducer.snapshot().health);
        let sequence = self.sequence();
        self.publish(BootEvent::BootReady {
            sequence,
            elapsed_ms: boot_elapsed_ms(),
            health,
        })
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

        self.persist_snapshot()?;

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
                // remains available through the snapshot for late consumers.
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

        let temporary = self.state_path.with_extension("json.tmp");
        fs::write(&temporary, encoded)?;
        fs::rename(&temporary, &self.state_path)?;
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boot_elapsed_parser_fallback_is_non_panicking() {
        let _ = boot_elapsed_ms();
    }

    #[test]
    fn default_args_are_empty() {
        // Parsing process-global argv is intentionally not unit-tested here; the
        // behavior is kept tiny and deterministic. This test protects the simple
        // data shape from growing hidden execution capability.
        let args = Args {
            config: None,
            print_config: false,
        };
        assert!(args.config.is_none());
        assert!(!args.print_config);
    }
}
