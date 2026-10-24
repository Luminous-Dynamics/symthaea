// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::path::PathBuf;
use symthaea_boot_ecology::{PreviousTermination, StorageState};
use symthaea_boot_state::{BootStateStore, PrepareInput};

const DEFAULT_STATE_DIR: &str = "/var/lib/spore-boot";
const DEFAULT_RUNTIME_DIR: &str = "/run/spore-boot";

fn main() {
    if let Err(error) = run() {
        eprintln!("spore-boot-state: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() || matches!(args[0].as_str(), "-h" | "--help") {
        print_usage();
        return Ok(());
    }

    let command = args.remove(0);
    match command.as_str() {
        "prepare" => run_prepare(args),
        "bless" => run_bless(args),
        "shutdown" => run_shutdown(args),
        "show" => run_show(args),
        other => Err(format!("unknown command {other:?}")),
    }
}

fn run_prepare(args: Vec<String>) -> Result<(), String> {
    let parsed = ParsedArgs::parse(args)?;
    let generation = parsed
        .required("generation")?
        .to_string();
    let mut input = PrepareInput::minimal(generation);
    input.hardware_fingerprint = parsed.get("hardware-fingerprint").map(str::to_string);
    input.storage_state = parse_storage_state(parsed.get("storage-state").unwrap_or("unknown"))?;
    input.oom_events = parsed.parse_or("oom-events", 0u32)?;
    input.thermal_events = parsed.parse_or("thermal-events", 0u32)?;
    input.mesh_enabled = parsed.flag("mesh-enabled");
    input.mesh_peers_last_seen = parsed.parse_or("mesh-peers", 0u32)?;

    let store = parsed.store();
    let result = store.prepare(&input)?;
    let genome = symthaea_boot_ecology::BootEcologyComposer::compose(&result.receipt, &result.lineage);

    println!("receipt={}", store.receipt_path().display());
    println!("lineage={}", store.runtime_dir().join(symthaea_boot_state::LINEAGE_FILE).display());
    println!("family={:?}", genome.family);
    println!("cue={:?}", genome.cue);
    println!("seed={}", genome.seed_hex());
    println!(
        "previous_interrupted={}",
        result.inferred_interrupted_previous_boot
    );
    Ok(())
}

fn run_bless(args: Vec<String>) -> Result<(), String> {
    let parsed = ParsedArgs::parse(args)?;
    let generation = parsed.required("generation")?;
    let store = parsed.store();
    let lineage = store.bless(generation)?;
    println!("last_known_good={}", lineage.last_known_good_generation.as_deref().unwrap_or(""));
    println!("successful_boots={}", lineage.successful_boots);
    Ok(())
}

fn run_shutdown(args: Vec<String>) -> Result<(), String> {
    let parsed = ParsedArgs::parse(args)?;
    let kind = parsed.required("kind")?;
    let termination = match kind {
        "poweroff" => PreviousTermination::CleanPoweroff,
        "reboot" => PreviousTermination::CleanReboot,
        "suspend" => PreviousTermination::Suspend,
        "hibernate" => PreviousTermination::Hibernate,
        other => return Err(format!("unsupported shutdown kind {other:?}")),
    };
    let uptime_secs = parsed.parse_or("uptime-secs", 0u64)?;
    let generation = parsed.get("generation").map(str::to_string);
    let hardware = parsed.get("hardware-fingerprint").map(str::to_string);
    parsed
        .store()
        .mark_shutdown(termination, uptime_secs, generation, hardware)?;
    Ok(())
}

fn run_show(args: Vec<String>) -> Result<(), String> {
    let parsed = ParsedArgs::parse(args)?;
    let store = parsed.store();
    let state = store.load_state()?;
    let lineage = store.load_lineage()?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "state": state,
            "lineage": lineage,
            "receipt": store.receipt_path(),
        }))
        .map_err(|e| e.to_string())?
    );
    Ok(())
}

fn parse_storage_state(value: &str) -> Result<StorageState, String> {
    if value == "clean" {
        return Ok(StorageState::Clean);
    }
    if value == "journal-replayed" {
        return Ok(StorageState::JournalReplayed);
    }
    if value == "degraded" {
        return Ok(StorageState::Degraded);
    }
    if value == "unknown" {
        return Ok(StorageState::Unknown);
    }
    if let Some(count) = value.strip_prefix("repaired:") {
        return count
            .parse::<u32>()
            .map(|repairs| StorageState::Repaired { repairs })
            .map_err(|e| format!("invalid repaired count {count:?}: {e}"));
    }
    Err(format!("invalid storage state {value:?}"))
}

#[derive(Debug, Default)]
struct ParsedArgs {
    values: std::collections::BTreeMap<String, String>,
    flags: std::collections::BTreeSet<String>,
}

impl ParsedArgs {
    fn parse(args: Vec<String>) -> Result<Self, String> {
        let mut parsed = Self::default();
        let mut i = 0;
        while i < args.len() {
            let arg = &args[i];
            if !arg.starts_with("--") {
                return Err(format!("unexpected positional argument {arg:?}"));
            }
            let key = arg.trim_start_matches("--").to_string();
            if matches!(key.as_str(), "mesh-enabled") {
                parsed.flags.insert(key);
                i += 1;
                continue;
            }
            i += 1;
            if i >= args.len() {
                return Err(format!("missing value for --{key}"));
            }
            parsed.values.insert(key, args[i].clone());
            i += 1;
        }
        Ok(parsed)
    }

    fn required(&self, key: &str) -> Result<&str, String> {
        self.get(key)
            .ok_or_else(|| format!("--{key} is required"))
    }

    fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(String::as_str)
    }

    fn flag(&self, key: &str) -> bool {
        self.flags.contains(key)
    }

    fn parse_or<T>(&self, key: &str, default: T) -> Result<T, String>
    where
        T: std::str::FromStr,
        T::Err: std::fmt::Display,
    {
        match self.get(key) {
            Some(value) => value
                .parse::<T>()
                .map_err(|e| format!("invalid --{key} value {value:?}: {e}")),
            None => Ok(default),
        }
    }

    fn store(&self) -> BootStateStore {
        let persistent = self
            .get("state-dir")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_STATE_DIR));
        let runtime = self
            .get("runtime-dir")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_RUNTIME_DIR));
        BootStateStore::new(persistent, runtime)
    }
}

fn print_usage() {
    eprintln!(
        "spore-boot-state COMMAND [OPTIONS]\n\n\
         Commands:\n\
           prepare  --generation GEN [--hardware-fingerprint DIGEST] [--storage-state STATE]\n\
                    [--oom-events N] [--thermal-events N] [--mesh-enabled] [--mesh-peers N]\n\
           bless    --generation GEN\n\
           shutdown --kind poweroff|reboot|suspend|hibernate [--uptime-secs N] [--generation GEN]\n\
           show\n\n\
         Common options:\n\
           --state-dir PATH    Persistent state directory (default: /var/lib/spore-boot)\n\
           --runtime-dir PATH  Runtime receipt directory (default: /run/spore-boot)\n\n\
         Storage states: clean, journal-replayed, degraded, unknown, repaired:N"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_storage_states() {
        assert!(matches!(parse_storage_state("clean").unwrap(), StorageState::Clean));
        assert!(matches!(
            parse_storage_state("repaired:3").unwrap(),
            StorageState::Repaired { repairs: 3 }
        ));
        assert!(parse_storage_state("mystical").is_err());
    }

    #[test]
    fn parser_keeps_mesh_as_flag() {
        let args = ParsedArgs::parse(vec![
            "--generation".into(),
            "gen-1".into(),
            "--mesh-enabled".into(),
            "--mesh-peers".into(),
            "7".into(),
        ])
        .unwrap();
        assert!(args.flag("mesh-enabled"));
        assert_eq!(args.required("generation").unwrap(), "gen-1");
        assert_eq!(args.parse_or("mesh-peers", 0u32).unwrap(), 7);
    }
}
