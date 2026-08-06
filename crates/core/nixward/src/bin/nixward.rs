// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! nixward: Conscious NixOS management CLI.
//!
//! Entry point that dispatches to subcommands or processes
//! natural language input through the cognitive core.

use clap::Parser;
use nixward::action::config_writer::ConfigWriter;
use nixward::action::executor::NixOSCommand;
use nixward::action::flake_ops::FlakeOps;
use nixward::action::gc_manager::GcManager;
use nixward::action::generation_manager::GenerationManager;
use nixward::action::service_manager::ServiceManager;
use nixward::cli::commands::{
    Cli, Command, ConfigCommand, ObserveDomain, OutputFormat, RebuildMode,
};
use nixward::cli::completions;
use nixward::cli::interactive;

fn main() {
    let cli = Cli::parse();

    // Natural language input takes priority
    if cli.has_natural_input() {
        interactive::process_oneshot(&cli.natural_input(), cli.dry_run, cli.format, cli.phi);
        return;
    }

    // No subcommand and no input → interactive REPL
    let Some(command) = cli.command else {
        let mut repl = interactive::ConsciousRepl::new(cli.dry_run, cli.format);
        if let Some(phi) = cli.phi {
            repl = repl.with_phi(phi);
        }
        if let Err(e) = repl.run() {
            eprintln!("REPL error: {}", e);
            std::process::exit(1);
        }
        return;
    };

    // Dispatch subcommands
    match command {
        Command::Search {
            query,
            options,
            limit,
        } => {
            cmd_search(&query, options, limit, cli.format);
        }

        Command::Rebuild {
            mode,
            flake,
            extra_args,
        } => {
            let cmd = match mode {
                RebuildMode::Switch => NixOSCommand::RebuildSwitch { flake, extra_args },
                RebuildMode::Test => NixOSCommand::RebuildTest { flake, extra_args },
                RebuildMode::Boot => NixOSCommand::RebuildBoot { flake, extra_args },
            };
            cmd_execute(cmd, cli.dry_run, cli.phi);
        }

        Command::Rollback { generation } => {
            let cmd = if let Some(g) = generation {
                GenerationManager::switch_to(g)
            } else {
                GenerationManager::rollback()
            };
            cmd_execute(cmd, cli.dry_run, cli.phi);
        }

        Command::Observe { domain } => {
            cmd_observe(domain, cli.format);
        }

        Command::Doctor => {
            cmd_doctor(cli.format);
        }

        Command::Generations {
            diff,
            from,
            to,
            delete_older_than,
        } => {
            if let Some(days) = delete_older_than {
                let cmd = GenerationManager::delete_older_than(days);
                cmd_execute(cmd, cli.dry_run, cli.phi);
            } else if diff {
                cmd_generations_diff(from, to);
            } else {
                cmd_generations_list(cli.format);
            }
        }

        Command::Flake { op } => {
            use nixward::cli::commands::FlakeCommand;
            match op {
                FlakeCommand::Check => {
                    let cmd = FlakeOps::check();
                    cmd_execute(cmd, cli.dry_run, cli.phi);
                }
                FlakeCommand::Update { inputs } => {
                    let cmd = if inputs.is_empty() {
                        FlakeOps::update_all(std::path::Path::new("."))
                    } else {
                        let refs: Vec<&str> = inputs.iter().map(|s| s.as_str()).collect();
                        FlakeOps::update_inputs(&refs)
                    };
                    cmd_execute(cmd, cli.dry_run, cli.phi);
                }
                FlakeCommand::Show => {
                    let cmd = FlakeOps::show();
                    cmd_execute(cmd, cli.dry_run, cli.phi);
                }
                FlakeCommand::Info => {
                    cmd_flake_info();
                }
            }
        }

        Command::Gc {
            analyze,
            older_than,
            aggressive,
        } => {
            if analyze {
                cmd_gc_analyze(cli.format);
            } else if aggressive {
                let cmd = GcManager::collect_aggressive();
                cmd_execute(cmd, cli.dry_run, cli.phi);
            } else if let Some(days) = older_than {
                let cmd = GcManager::collect_older_than(days);
                cmd_execute(cmd, cli.dry_run, cli.phi);
            } else {
                let cmd = GcManager::collect();
                cmd_execute(cmd, cli.dry_run, cli.phi);
            }
        }

        Command::Service { op } => {
            use nixward::cli::commands::ServiceCommand;
            match op {
                ServiceCommand::Status { name } => {
                    cmd_service_status(&name, cli.format);
                }
                ServiceCommand::Start { name } => {
                    cmd_execute(ServiceManager::start(&name), cli.dry_run, cli.phi);
                }
                ServiceCommand::Stop { name } => {
                    cmd_execute(ServiceManager::stop(&name), cli.dry_run, cli.phi);
                }
                ServiceCommand::Restart { name } => {
                    cmd_execute(ServiceManager::restart(&name), cli.dry_run, cli.phi);
                }
                ServiceCommand::Failed => {
                    cmd_service_failed(cli.format);
                }
            }
        }

        Command::Repl => {
            let mut repl = interactive::ConsciousRepl::new(cli.dry_run, cli.format);
            if let Some(phi) = cli.phi {
                repl = repl.with_phi(phi);
            }
            if let Err(e) = repl.run() {
                eprintln!("REPL error: {}", e);
                std::process::exit(1);
            }
        }

        Command::Completions { shell } => {
            completions::generate_completions(shell);
        }

        Command::Health { json } => {
            cmd_health(if json { OutputFormat::Json } else { cli.format });
        }

        Command::Predict { horizons } => {
            cmd_predict(&horizons, cli.format);
        }

        Command::Watch { timeout, interval } => {
            cmd_watch(timeout, interval, cli.format);
        }

        Command::Scrub { file } => {
            cmd_scrub(file.as_deref());
        }

        Command::Knowledge { query, limit } => {
            cmd_knowledge(&query, limit, cli.format);
        }

        Command::Config { op } => {
            cmd_config(op);
        }
    }
}

// ---- Command implementations ----

fn cmd_search(query: &str, options: bool, limit: usize, format: OutputFormat) {
    if options {
        // HDC semantic search over NixOS options
        use nixward::encoding::{NixCodebook, search_options};

        let mut codebook = NixCodebook::new();

        // Common NixOS option paths for semantic search
        let known_paths = [
            "services.nginx.enable",
            "services.nginx.package",
            "services.nginx.virtualHosts",
            "services.postgresql.enable",
            "services.postgresql.package",
            "services.openssh.enable",
            "services.openssh.settings.PermitRootLogin",
            "services.pipewire.enable",
            "services.pipewire.alsa.enable",
            "services.docker.enable",
            "services.podman.enable",
            "services.xserver.enable",
            "services.xserver.displayManager.gdm.enable",
            "services.xserver.desktopManager.gnome.enable",
            "networking.firewall.enable",
            "networking.firewall.allowedTCPPorts",
            "networking.networkmanager.enable",
            "networking.wireguard.enable",
            "boot.loader.grub.enable",
            "boot.loader.systemd-boot.enable",
            "hardware.opengl.enable",
            "hardware.nvidia.modesetting.enable",
            "hardware.pulseaudio.enable",
            "hardware.bluetooth.enable",
            "security.sudo.enable",
            "security.polkit.enable",
            "users.users",
            "users.defaultUserShell",
            "environment.systemPackages",
            "nixpkgs.config.allowUnfree",
            "programs.zsh.enable",
            "programs.fish.enable",
            "programs.steam.enable",
            "nix.gc.automatic",
            "nix.settings.experimental-features",
        ];
        let path_refs: Vec<&str> = known_paths.to_vec();

        let results = search_options(query, &mut codebook, &path_refs, limit);

        match format {
            OutputFormat::Json => {
                let json_results: Vec<serde_json::Value> = results
                    .iter()
                    .map(|r| {
                        serde_json::json!({
                            "path": r.path,
                            "similarity": r.similarity,
                            "reason": r.match_reason,
                        })
                    })
                    .collect();
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json_results).unwrap_or_default()
                );
            }
            OutputFormat::Minimal => {
                for r in &results {
                    println!("{}", r.path);
                }
            }
            OutputFormat::Human => {
                println!("  HDC Semantic Search: \"{}\"", query);
                println!();
                for (i, r) in results.iter().enumerate() {
                    println!("  {}. {} (similarity: {:.3})", i + 1, r.path, r.similarity);
                    println!("     {}", r.match_reason);
                }
                if results.is_empty() {
                    println!("  No matching options found.");
                }
            }
        }
    } else {
        // Package search via nix search
        let cmd = NixOSCommand::Search {
            query: query.to_string(),
            json: matches!(format, OutputFormat::Json),
        };
        let (bin, args) = cmd.to_command();
        println!("  {} {}", bin, args.join(" "));
    }
}

/// Default confidence level applied when the caller passes no `--phi` override.
///
/// This is NOT a real consciousness measurement — nothing in this crate computes an
/// actual Φ. It is a confirmation-level constant: `NixOSCommand::safety_level()`
/// classifies each command into a tier (ReadOnly/UserModify/SystemModify/
/// SystemCritical/Destructive) and each tier has a required threshold
/// (`SafetyLevel::required_phi()`, currently 0.2/0.3/0.4/0.4/0.6). Setting the
/// default to 0.35 means read-only and user-scoped commands run with no extra
/// ceremony, while system-wide or destructive commands (`nixos-rebuild switch`,
/// `nix-collect-garbage`, etc.) are blocked unless the operator explicitly passes
/// `--phi` at or above the required threshold. The previous default of 0.7 cleared
/// every tier including Destructive (0.6), so the gate never actually blocked
/// anything run via this CLI's default invocation — see
/// SYMTHAEA_NIXOS_MANAGEMENT_IMPROVEMENT_PLAN_2026-07-26.md Phase 1.
const DEFAULT_CLI_CONFIRMATION_LEVEL: f32 = 0.35;

fn cmd_execute(cmd: NixOSCommand, dry_run: bool, phi_override: Option<f64>) {
    let phi = phi_override
        .map(|p| p as f32)
        .unwrap_or(DEFAULT_CLI_CONFIRMATION_LEVEL);
    let safety = cmd.safety_level();
    let required = safety.required_phi();

    if phi < required {
        eprintln!(
            "  Blocked: {:?} requires Phi >= {:.2}, current Phi = {:.2}",
            safety, required, phi
        );
        eprintln!(
            "  Use --phi={:.1} to override, or confirm explicitly.",
            required
        );
        return;
    }

    let (bin, args) = cmd.to_command();
    if dry_run {
        println!("  [DRY-RUN] Would execute: {} {}", bin, args.join(" "));
    } else {
        println!("  Executing: {} {}", bin, args.join(" "));
        let status = std::process::Command::new(&bin).args(&args).status();
        match status {
            Ok(s) if s.success() => println!("  Done."),
            Ok(s) => {
                eprintln!("  Command failed with exit code: {:?}", s.code());
                std::process::exit(1);
            }
            Err(e) => {
                eprintln!("  Failed to execute: {}", e);
                std::process::exit(1);
            }
        }
    }
}

/// Preview or apply a `configuration.nix` edit. Unlike `cmd_execute`, this
/// never runs a shell command -- it produces a `ConfigWriter`/`ConfigPatch`
/// diff, always prints it, and only writes when `apply` is true. Not
/// Phi-gated: `ConfigWriter::apply_patch` already validates syntax
/// (`nix-instantiate --parse`) and takes a git backup before any write, and
/// `apply` itself is the confirmation.
fn cmd_config(op: ConfigCommand) {
    let (package_op, option_op, staging, apply) = match &op {
        ConfigCommand::AddPackage {
            package,
            staging,
            apply,
        } => (Some((package.clone(), true)), None, staging.clone(), *apply),
        ConfigCommand::RemovePackage {
            package,
            staging,
            apply,
        } => (
            Some((package.clone(), false)),
            None,
            staging.clone(),
            *apply,
        ),
        ConfigCommand::SetOption {
            option_path,
            value,
            staging,
            apply,
        } => (
            None,
            Some((option_path.clone(), value.clone())),
            staging.clone(),
            *apply,
        ),
    };

    let mut writer = ConfigWriter::new().with_dry_run(!apply);
    if let Some(dir) = &staging {
        writer = writer.with_config_root(dir.as_str());
    }

    let patch = if let Some((package, add)) = package_op {
        if add {
            writer.add_system_package(&package)
        } else {
            writer.remove_system_package(&package)
        }
    } else if let Some((option_path, value)) = option_op {
        writer.set_option(&option_path, &value)
    } else {
        unreachable!("ConfigCommand always yields exactly one operation")
    };

    let patch = match patch {
        Ok(p) => p,
        Err(e) => {
            eprintln!("  Failed to build patch: {e}");
            std::process::exit(1);
        }
    };

    if patch.is_noop() {
        println!("  No change: already in the desired state.");
        return;
    }

    println!("{}", patch.diff());

    match writer.apply_patch(&patch) {
        Ok(result) if apply => {
            println!(
                "  Applied. {}",
                result
                    .backup_path
                    .map(|p| format!("Backup: {}", p.display()))
                    .unwrap_or_else(|| "(no git backup -- git_backup disabled)".to_string())
            );
        }
        Ok(_) => {
            println!("  [PREVIEW] Not written -- pass --apply to write this change.");
        }
        Err(e) => {
            eprintln!("  Failed to apply patch: {e}");
            std::process::exit(1);
        }
    }
}

fn cmd_observe(domain: Option<ObserveDomain>, format: OutputFormat) {
    match domain {
        Some(ObserveDomain::Services) => {
            match nixward::observe::systemd::SystemdObserver::list_units() {
                Ok(units) => match format {
                    OutputFormat::Json => {
                        let json: Vec<serde_json::Value> = units
                            .iter()
                            .map(|u| {
                                serde_json::json!({
                                    "name": u.name,
                                    "active_state": u.active_state,
                                    "sub_state": u.sub_state,
                                    "description": u.description,
                                })
                            })
                            .collect();
                        println!(
                            "{}",
                            serde_json::to_string_pretty(&json).unwrap_or_default()
                        );
                    }
                    OutputFormat::Minimal => {
                        for unit in &units {
                            println!("{}\t{}\t{}", unit.name, unit.active_state, unit.sub_state);
                        }
                    }
                    _ => {
                        for unit in &units {
                            println!(
                                "  {} {} {} {}",
                                unit.name, unit.active_state, unit.sub_state, unit.description
                            );
                        }
                        println!("  {} units total", units.len());
                    }
                },
                Err(e) => eprintln!("  Failed to list services: {}", e),
            }
        }
        Some(ObserveDomain::Store) => match nixward::observe::store::StoreObserver::store_info() {
            Ok(info) => match format {
                OutputFormat::Json => {
                    let json = serde_json::json!({
                        "store_path": info.store_path,
                        "path_count": info.path_count,
                        "total_size_bytes": info.total_size_bytes,
                        "deriver_count": info.deriver_count,
                    });
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&json).unwrap_or_default()
                    );
                }
                OutputFormat::Minimal => {
                    println!("{}\t{}", info.path_count, info.total_size_bytes);
                }
                _ => {
                    println!("  Store paths: {}", info.path_count);
                    println!("  Total size: {} bytes", info.total_size_bytes);
                }
            },
            Err(e) => eprintln!("  Failed to query store: {}", e),
        },
        Some(ObserveDomain::Generations) => {
            cmd_generations_list(format);
        }
        Some(ObserveDomain::Hardware) => {
            match nixward::observe::hardware::HardwareObserver::probe() {
                Ok(info) => match format {
                    OutputFormat::Json => {
                        let gpus: Vec<serde_json::Value> = info
                            .gpus
                            .iter()
                            .map(|g| {
                                serde_json::json!({
                                    "name": g.name,
                                    "driver": g.driver,
                                })
                            })
                            .collect();
                        let json = serde_json::json!({
                            "cpu_model": info.cpu_model,
                            "cpu_cores": info.cpu_cores,
                            "memory_total_mb": info.memory_total_mb,
                            "gpus": gpus,
                        });
                        println!(
                            "{}",
                            serde_json::to_string_pretty(&json).unwrap_or_default()
                        );
                    }
                    _ => {
                        println!("  CPU: {} ({} cores)", info.cpu_model, info.cpu_cores);
                        println!("  Memory: {} MiB", info.memory_total_mb);
                        for gpu in &info.gpus {
                            let driver = gpu.driver.as_deref().unwrap_or("unknown");
                            println!("  GPU: {} ({})", gpu.name, driver);
                        }
                    }
                },
                Err(e) => eprintln!("  Failed to probe hardware: {}", e),
            }
        }
        _ => {
            match nixward::observe::SystemObserver::snapshot() {
                Ok(snap) => match format {
                    OutputFormat::Json => {
                        let services: Vec<serde_json::Value> = snap.services.iter().map(|(n, s)| {
                            serde_json::json!({"name": n, "state": format!("{:?}", s)})
                        }).collect();
                        let json = serde_json::json!({
                            "services": services,
                            "service_count": snap.services.len(),
                            "package_count": snap.packages.len(),
                            "generation": snap.generation,
                            "store_size_bytes": snap.store_size_bytes,
                            "store_path_count": snap.store_path_count,
                        });
                        println!(
                            "{}",
                            serde_json::to_string_pretty(&json).unwrap_or_default()
                        );
                    }
                    OutputFormat::Minimal => {
                        println!(
                            "services={} r#gen={} store={}",
                            snap.services.len(),
                            snap.generation.map_or("?".into(), |g| g.to_string()),
                            snap.store_size_bytes.unwrap_or(0),
                        );
                    }
                    _ => {
                        println!("  Taking system snapshot...");
                        println!("  Services: {}", snap.services.len());
                        println!("  Packages: {}", snap.packages.len());
                        if let Some(g) = snap.generation {
                            println!("  Generation: {}", g);
                        }
                        if let Some(size) = snap.store_size_bytes {
                            println!("  Store size: {} bytes", size);
                        }
                    }
                },
                Err(e) => eprintln!("  Failed to snapshot: {}", e),
            }
        }
    }
}

fn cmd_doctor(format: OutputFormat) {
    // 1. Take system snapshot + hardware probe
    let snapshot = match nixward::observe::SystemObserver::snapshot() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("  Failed to snapshot system: {}", e);
            return;
        }
    };
    let hw = nixward::observe::hardware::HardwareObserver::probe().ok();

    // 2. Run unified support assessment
    let mut codebook = nixward::encoding::NixCodebook::new();
    let mut assessor = nixward::support::SupportAssessor::new(&mut codebook);

    // Load persisted predictive history for trend-aware predictions (BC)
    let snapshot_path = nixward::ipc::default_snapshot_path();
    let pred_path = snapshot_path.with_file_name("predictive_history.json");
    let mut monitor = if let Ok(json) = std::fs::read_to_string(&pred_path) {
        if let Ok(saved) =
            serde_json::from_str::<nixward::support::predictive::SavedPredictiveState>(&json)
        {
            let sample_count = saved.samples.len();
            eprintln!(
                "nixward doctor: loaded {} historical samples for trend analysis",
                sample_count
            );
            nixward::support::PredictiveMonitor::load(
                saved,
                nixward::support::predictive::AlertThresholds::default(),
            )
        } else {
            nixward::support::PredictiveMonitor::with_defaults()
        }
    } else {
        nixward::support::PredictiveMonitor::with_defaults()
    };
    let telemetry = nixward::support::SystemTelemetry {
        disk_used_pct: hw.as_ref().map_or(0.0, |h| {
            h.disks.first().map_or(0.0, |d| {
                if d.total_bytes > 0 {
                    d.used_bytes as f64 / d.total_bytes as f64 * 100.0
                } else {
                    0.0
                }
            })
        }),
        memory_used_pct: hw.as_ref().map_or(0.0, |h| {
            if h.memory_total_mb > 0 {
                let used = h.memory_total_mb.saturating_sub(h.memory_available_mb);
                used as f64 / h.memory_total_mb as f64 * 100.0
            } else {
                0.0
            }
        }),
        store_path_count: snapshot.store_path_count.unwrap_or(0) as u64,
        failed_unit_count: snapshot
            .services
            .iter()
            .filter(|(_, s)| *s == nixward::encoding::ServiceState::Failed)
            .count() as u32,
        load_average_1m: hw.as_ref().map_or(0.0, |h| h.load_average[0]),
        swap_used_pct: hw.as_ref().map_or(0.0, |h| {
            if h.swap_total_mb > 0 {
                h.swap_used_mb as f64 / h.swap_total_mb as f64 * 100.0
            } else {
                0.0
            }
        }),
    };
    monitor.ingest(telemetry);

    let assessment = assessor.assess(&snapshot, hw.as_ref(), Some(&mut monitor), &mut codebook);

    // 3. Doctor-specific checks: journal anomalies
    let journal_anomalies: Vec<serde_json::Value> =
        match nixward::observe::journal::JournalObserver::recent_entries(50) {
            Ok(entries) => {
                let mut detector = nixward::mind::JournalAnomalyDetector::new();
                detector
                    .process_entries(&entries)
                    .iter()
                    .map(|a| {
                        serde_json::json!({
                            "reason": a.reason,
                            "score": a.anomaly_score,
                            "unit": a.entry.unit,
                        })
                    })
                    .collect()
            }
            Err(_) => Vec::new(),
        };

    // 4. Module structure analysis
    let module_info = {
        let config_path = std::path::Path::new("/etc/nixos/configuration.nix");
        if config_path.exists() {
            let mut mp = nixward::parser::module_parser::ModuleParser::new();
            mp.parse_file(config_path).ok()
        } else {
            None
        }
    };

    // 5. Generation count
    let gen_count = GenerationManager::list().map(|g| g.len()).unwrap_or(0);

    match format {
        OutputFormat::Json => {
            let checks_json: Vec<serde_json::Value> = assessment
                .health_checks
                .iter()
                .map(|c| {
                    serde_json::json!({
                        "name": c.name,
                        "status": format!("{:?}", c.status),
                        "message": c.message,
                        "category": c.category.to_string(),
                        "recommendations": c.recommendations,
                    })
                })
                .collect();
            let recs_json: Vec<serde_json::Value> = assessment
                .recommendations
                .iter()
                .map(|r| {
                    serde_json::json!({
                        "urgency": format!("{:?}", r.urgency),
                        "trigger": r.trigger,
                        "category": r.category,
                        "actions": r.actions,
                        "knowledge_articles": r.knowledge_article_ids,
                        "prediction_context": r.prediction_context,
                    })
                })
                .collect();
            let json = serde_json::json!({
                "overall_status": format!("{:?}", assessment.overall_status),
                "health_checks": checks_json,
                "recommendations": recs_json,
                "active_alerts": assessment.active_alerts.len(),
                "knowledge_matches": assessment.knowledge_matches_found,
                "generation_count": gen_count,
                "journal_anomalies": journal_anomalies,
                "module_info": module_info.as_ref().map(|m| serde_json::json!({
                    "is_nixos_module": m.is_nixos_module,
                    "imports": m.imports,
                    "option_declarations": m.option_decls.len(),
                    "config_settings": m.config_settings.len(),
                })),
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&json).unwrap_or_default()
            );
        }
        _ => {
            println!("  Running system diagnostics...");
            println!();

            // Display health checks
            println!("  Overall Health: {}", assessment.overall_status);
            println!();
            for check in &assessment.health_checks {
                println!("  [{:?}] {}: {}", check.status, check.name, check.message);
            }

            // Display unified recommendations
            if !assessment.recommendations.is_empty() {
                println!();
                println!("  Recommendations ({}):", assessment.recommendations.len());
                for (i, rec) in assessment.recommendations.iter().enumerate() {
                    println!("    {}. [{}] {}", i + 1, rec.urgency, rec.trigger);
                    for action in &rec.actions {
                        println!("       -> {}", action);
                    }
                    if !rec.knowledge_article_ids.is_empty() {
                        println!("       KB: {}", rec.knowledge_article_ids.join(", "));
                    }
                    if let Some(ref ctx) = rec.prediction_context {
                        println!("       Prediction: {}", ctx);
                    }
                }
            }

            // Doctor-specific: generation count
            println!();
            println!("  Generations: {} total", gen_count);
            if gen_count > 20 {
                println!("    Consider cleaning up old generations");
            }

            // Doctor-specific: journal anomalies
            if journal_anomalies.is_empty() {
                println!("  Journal: No anomalies in recent entries");
            } else {
                println!("  Journal: {} anomalies detected", journal_anomalies.len());
                for a in journal_anomalies.iter().take(3) {
                    println!(
                        "    - {} (score: {:.2})",
                        a["reason"].as_str().unwrap_or("?"),
                        a["score"].as_f64().unwrap_or(0.0)
                    );
                }
            }

            // Doctor-specific: module structure
            if let Some(ref mi) = module_info {
                println!(
                    "  Config: {} imports, {} option decls, {} settings",
                    mi.imports.len(),
                    mi.option_decls.len(),
                    mi.config_settings.len()
                );
            }

            println!();
        }
    }
}

fn cmd_generations_list(format: OutputFormat) {
    match GenerationManager::list() {
        Ok(gens) => match format {
            OutputFormat::Json => {
                let json: Vec<serde_json::Value> = gens
                    .iter()
                    .map(|g| {
                        serde_json::json!({
                            "number": g.number,
                            "date": g.date,
                            "nixos_version": g.nixos_version,
                            "kernel_version": g.kernel_version,
                            "current": g.current,
                        })
                    })
                    .collect();
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json).unwrap_or_default()
                );
            }
            OutputFormat::Minimal => {
                for g in &gens {
                    let cur = if g.current { "*" } else { "" };
                    println!(
                        "{}{}\t{}\t{}",
                        g.number, cur, g.nixos_version, g.kernel_version
                    );
                }
            }
            _ => {
                for g in &gens {
                    let current = if g.current { " (current)" } else { "" };
                    println!(
                        "  {}  {}  {}  {}{}",
                        g.number, g.date, g.nixos_version, g.kernel_version, current
                    );
                }
                println!("  {} generations total", gens.len());
            }
        },
        Err(e) => eprintln!("  Failed to list generations: {}", e),
    }
}

fn cmd_generations_diff(from: Option<u32>, to: Option<u32>) {
    let (from, to) = match (from, to) {
        (Some(f), Some(t)) => (f, t),
        _ => match GenerationManager::list() {
            Ok(gens) if gens.len() >= 2 => (gens[1].number, gens[0].number),
            _ => {
                eprintln!("  Need at least 2 generations for diff. Use --from and --to.");
                return;
            }
        },
    };

    match GenerationManager::diff(from, to) {
        Ok(diff) => {
            println!("  Generation {} -> {}:", diff.from, diff.to);
            if !diff.added.is_empty() {
                println!("  Added:");
                for pkg in &diff.added {
                    println!("    + {}", pkg);
                }
            }
            if !diff.removed.is_empty() {
                println!("  Removed:");
                for pkg in &diff.removed {
                    println!("    - {}", pkg);
                }
            }
            if !diff.changed.is_empty() {
                println!("  Changed:");
                for (name, old, new) in &diff.changed {
                    println!("    ~ {} {} -> {}", name, old, new);
                }
            }
            if diff.added.is_empty() && diff.removed.is_empty() && diff.changed.is_empty() {
                println!("  No differences found.");
            }
        }
        Err(e) => eprintln!("  Failed to diff generations: {}", e),
    }
}

fn cmd_gc_analyze(format: OutputFormat) {
    match GcManager::analyze() {
        Ok(analysis) => {
            let rec = GcManager::recommend(&analysis);
            match format {
                OutputFormat::Json => {
                    let json = serde_json::json!({
                        "total_store": analysis.total_store_human(),
                        "reclaimable": analysis.reclaimable_human(),
                        "reclaimable_percent": analysis.reclaimable_percent(),
                        "dead_paths": analysis.dead_path_count,
                        "live_roots": analysis.live_root_count,
                        "generations": analysis.total_generations,
                        "recommended": rec.recommended,
                        "reason": rec.reason,
                    });
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&json).unwrap_or_default()
                    );
                }
                _ => {
                    println!("  Store Analysis:");
                    println!("    Total:       {}", analysis.total_store_human());
                    println!(
                        "    Reclaimable: {} ({:.0}%)",
                        analysis.reclaimable_human(),
                        analysis.reclaimable_percent()
                    );
                    println!("    Dead paths:  {}", analysis.dead_path_count);
                    println!("    Live roots:  {}", analysis.live_root_count);
                    println!("    Generations: {}", analysis.total_generations);
                    println!();
                    if rec.recommended {
                        println!("  Recommendation: {}", rec.reason);
                    } else {
                        println!("  {}", rec.reason);
                    }
                }
            }
        }
        Err(e) => eprintln!("  Failed to analyze store: {}", e),
    }
}

fn cmd_flake_info() {
    let cwd = std::env::current_dir().unwrap_or_default();
    if !FlakeOps::is_flake(&cwd) {
        eprintln!("  Not a flake directory (no flake.nix found).");
        return;
    }

    // Parse local flake.nix with tree-sitter for structural info
    let mut flake_parser = nixward::parser::flake_parser::FlakeParser::new();
    if let Ok(project) = flake_parser.parse_dir(&cwd) {
        let info = &project.flake_info;
        if let Some(ref desc) = info.description {
            println!("  Description: {}", desc);
        }
        if !info.inputs.is_empty() {
            println!("  Declared inputs ({}):", info.inputs.len());
            for input in &info.inputs {
                let extra = match (&input.url, &input.follows) {
                    (Some(url), _) => format!(" ({})", url),
                    (_, Some(f)) => format!(" (follows {})", f),
                    _ => String::new(),
                };
                println!("    {}{}", input.name, extra);
            }
        }
        if !info.output_attrs.is_empty() {
            println!("  Output types: {}", info.output_attrs.join(", "));
        }

        // Show lock info if available
        if let Some(ref lock) = project.lock_info {
            println!(
                "  Locked inputs ({}, lock v{}):",
                lock.inputs.len(),
                lock.version
            );
            for input in &lock.inputs {
                let rev_short = input
                    .rev
                    .as_deref()
                    .map(|r| if r.len() > 8 { &r[..8] } else { r })
                    .unwrap_or("?");
                let source = match (&input.owner, &input.repo) {
                    (Some(o), Some(r)) => format!("{}:{}/{}", input.source_type, o, r),
                    _ => input.source_type.clone(),
                };
                println!("    {} ({} @ {})", input.name, source, rev_short);
            }
        }
    }

    // Also try nix flake metadata for resolved URLs
    match FlakeOps::metadata(&cwd) {
        Ok(meta) => {
            if !meta.url.is_empty() {
                println!("  Resolved URL: {}", meta.url);
            }
            if let Some(modified) = &meta.last_modified {
                println!("  Last modified: {}", modified);
            }
        }
        Err(_) => {
            // nix flake metadata not available — tree-sitter parse above is sufficient
        }
    }
}

fn cmd_service_status(name: &str, format: OutputFormat) {
    match ServiceManager::status(name) {
        Ok(status) => match format {
            OutputFormat::Json => {
                let json = serde_json::json!({
                    "name": status.name,
                    "active": status.active,
                    "enabled": status.enabled,
                    "active_state": status.active_state,
                    "sub_state": status.sub_state,
                });
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json).unwrap_or_default()
                );
            }
            _ => {
                let indicator = if status.active { "active" } else { "inactive" };
                let enabled = if status.enabled {
                    "enabled"
                } else {
                    "disabled"
                };
                println!("  {} ({}, {})", status.name, indicator, enabled);
                println!("    State: {} ({})", status.active_state, status.sub_state);
            }
        },
        Err(e) => eprintln!("  Failed to get status for {}: {}", name, e),
    }
}

fn cmd_service_failed(format: OutputFormat) {
    match nixward::observe::systemd::SystemdObserver::failed_units() {
        Ok(units) => match format {
            OutputFormat::Json => {
                let json: Vec<serde_json::Value> = units
                    .iter()
                    .map(|u| {
                        serde_json::json!({
                            "name": u.name,
                            "description": u.description,
                            "sub_state": u.sub_state,
                        })
                    })
                    .collect();
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json).unwrap_or_default()
                );
            }
            OutputFormat::Minimal => {
                for unit in &units {
                    println!("{}", unit.name);
                }
            }
            _ => {
                if units.is_empty() {
                    println!("  No failed services.");
                } else {
                    println!("  Failed services:");
                    for unit in &units {
                        println!("    {} ({})", unit.name, unit.description);
                    }
                }
            }
        },
        Err(e) => eprintln!("  Failed to list failed services: {}", e),
    }
}

fn cmd_health(format: OutputFormat) {
    use nixward::support::health_check::HealthAssessor;

    let snapshot = match nixward::observe::SystemObserver::snapshot() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("  Failed to snapshot system: {}", e);
            return;
        }
    };

    let hw = nixward::observe::hardware::HardwareObserver::probe().ok();
    let assessor = HealthAssessor::default();
    let (overall, checks) = assessor.assess_all(&snapshot, hw.as_ref());

    match format {
        OutputFormat::Json => {
            let json_checks: Vec<serde_json::Value> = checks
                .iter()
                .map(|c| {
                    serde_json::json!({
                        "name": c.name,
                        "status": format!("{:?}", c.status),
                        "message": c.message,
                        "recommendations": c.recommendations,
                    })
                })
                .collect();
            let json = serde_json::json!({
                "overall": format!("{:?}", overall),
                "checks": json_checks,
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&json).unwrap_or_default()
            );
        }
        _ => {
            println!("  System Health: {:?}", overall);
            println!();
            for check in &checks {
                println!("  [{:?}] {}: {}", check.status, check.name, check.message);
                for rec in &check.recommendations {
                    println!("    -> {}", rec);
                }
            }
        }
    }
}

fn cmd_predict(horizons_str: &str, format: OutputFormat) {
    use nixward::encoding::ServiceState;
    use nixward::support::predictive::{
        AlertThresholds, PredictiveMonitor, SavedPredictiveState, SystemTelemetry,
    };

    let snapshot = match nixward::observe::SystemObserver::snapshot() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("  Failed to snapshot system: {}", e);
            return;
        }
    };

    let hw = nixward::observe::hardware::HardwareObserver::probe().ok();

    let telemetry = SystemTelemetry {
        disk_used_pct: hw.as_ref().map_or(0.0, |h| {
            h.disks.first().map_or(0.0, |d| {
                if d.total_bytes > 0 {
                    d.used_bytes as f64 / d.total_bytes as f64 * 100.0
                } else {
                    0.0
                }
            })
        }),
        memory_used_pct: hw.as_ref().map_or(0.0, |h| {
            if h.memory_total_mb > 0 {
                let used = h.memory_total_mb.saturating_sub(h.memory_available_mb);
                used as f64 / h.memory_total_mb as f64 * 100.0
            } else {
                0.0
            }
        }),
        store_path_count: snapshot.store_path_count.unwrap_or(0) as u64,
        failed_unit_count: snapshot
            .services
            .iter()
            .filter(|(_, s)| *s == ServiceState::Failed)
            .count() as u32,
        load_average_1m: hw.as_ref().map_or(0.0, |h| h.load_average[0]),
        swap_used_pct: hw.as_ref().map_or(0.0, |h| {
            if h.swap_total_mb > 0 {
                h.swap_used_mb as f64 / h.swap_total_mb as f64 * 100.0
            } else {
                0.0
            }
        }),
    };

    // Try to load persisted history from daemon for richer predictions
    let pred_path = nixward::ipc::default_snapshot_path().with_file_name("predictive_history.json");
    let mut monitor = if let Ok(json) = std::fs::read_to_string(&pred_path) {
        if let Ok(saved) = serde_json::from_str::<SavedPredictiveState>(&json) {
            let count = saved.samples.len();
            let m = PredictiveMonitor::load(saved, AlertThresholds::default());
            if matches!(format, OutputFormat::Human) {
                println!("  Loaded {} historical samples from daemon.", count);
            }
            m
        } else {
            PredictiveMonitor::with_defaults()
        }
    } else {
        PredictiveMonitor::with_defaults()
    };
    monitor.ingest(telemetry);

    let horizons: Vec<f64> = horizons_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    let mut all_predictions = Vec::new();
    for hours in &horizons {
        all_predictions.extend(monitor.predict(*hours));
    }

    match format {
        OutputFormat::Json => {
            let json_preds: Vec<serde_json::Value> = all_predictions
                .iter()
                .map(|p| {
                    serde_json::json!({
                        "metric": p.metric,
                        "current": p.current_value,
                        "predicted": p.predicted_value,
                        "hours_ahead": p.hours_ahead,
                        "crosses_threshold": p.crosses_threshold,
                        "threshold": p.threshold,
                        "confidence": p.confidence,
                        "action": p.recommended_action,
                    })
                })
                .collect();
            println!(
                "{}",
                serde_json::to_string_pretty(&json_preds).unwrap_or_default()
            );
        }
        _ => {
            println!("  Predictive Analysis (single-sample baseline)");
            println!();
            for p in &all_predictions {
                let alert = if p.crosses_threshold { " ALERT" } else { "" };
                println!(
                    "  [+{}h] {}: {:.1} -> {:.1} (threshold: {:.1}, conf: {:.2}){}",
                    p.hours_ahead,
                    p.metric,
                    p.current_value,
                    p.predicted_value,
                    p.threshold,
                    p.confidence,
                    alert
                );
                if let Some(ref action) = p.recommended_action {
                    println!("    -> {}", action);
                }
            }
            println!();
            if monitor.sample_count() > 1 {
                println!(
                    "  ({} total samples — using daemon history)",
                    monitor.sample_count()
                );
            } else {
                println!("  Note: predictions improve with continuous monitoring via the daemon.");
            }
        }
    }
}

fn cmd_watch(timeout: u64, interval: u64, format: OutputFormat) {
    use nixward::encoding::NixCodebook;
    use nixward::support::watchdog::{Watchdog, WatchdogConfig};
    use std::time::Duration;

    let config = WatchdogConfig {
        timeout: Duration::from_secs(timeout),
        check_interval: Duration::from_secs(interval),
        ..WatchdogConfig::default()
    };

    let mut codebook = NixCodebook::new();

    // Capture baseline before monitoring
    let (baseline_hv, _baseline_snap) = match Watchdog::capture_baseline(&mut codebook) {
        Some(b) => b,
        None => {
            eprintln!("  Failed to capture baseline snapshot.");
            return;
        }
    };

    let current_gen = nixward::action::generation_manager::GenerationManager::current_generation()
        .unwrap_or(0) as u64;

    match format {
        OutputFormat::Json => {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "status": "monitoring",
                    "timeout_secs": timeout,
                    "interval_secs": interval,
                    "baseline_gen": current_gen,
                }))
                .unwrap_or_default()
            );
        }
        _ => {
            println!("  Watchdog monitoring started.");
            println!(
                "  Timeout: {}s, interval: {}s, r#gen: {}",
                timeout, interval, current_gen
            );
            println!("  Watching for system degradation...");
            println!();
        }
    }

    let watchdog = Watchdog::new(config);
    let verdict = watchdog.monitor(&mut codebook, &baseline_hv, current_gen);

    // Persist verdict for daemon/TUI to pick up
    let verdict_str = match &verdict {
        nixward::support::watchdog::WatchdogVerdict::Stabilized { .. } => "stabilized",
        nixward::support::watchdog::WatchdogVerdict::Degraded { .. } => "degraded",
        nixward::support::watchdog::WatchdogVerdict::Reverted { .. } => "reverted",
        nixward::support::watchdog::WatchdogVerdict::Error { .. } => "error",
    };
    let wd_path = nixward::ipc::default_snapshot_path().with_file_name("watchdog_verdict.txt");
    let _ = std::fs::write(&wd_path, verdict_str);

    match format {
        OutputFormat::Json => {
            let json = match &verdict {
                nixward::support::watchdog::WatchdogVerdict::Stabilized {
                    duration,
                    checks_performed,
                    ..
                } => serde_json::json!({
                    "verdict": "stabilized",
                    "duration_secs": duration.as_secs(),
                    "checks": checks_performed,
                }),
                nixward::support::watchdog::WatchdogVerdict::Degraded {
                    reason,
                    surprise,
                    checks_performed,
                    ..
                } => serde_json::json!({
                    "verdict": "degraded",
                    "reason": reason,
                    "surprise": surprise,
                    "checks": checks_performed,
                }),
                nixward::support::watchdog::WatchdogVerdict::Reverted {
                    reason, pre_gen, ..
                } => serde_json::json!({
                    "verdict": "reverted",
                    "reason": reason,
                    "pre_gen": pre_gen,
                }),
                nixward::support::watchdog::WatchdogVerdict::Error { message } => {
                    serde_json::json!({
                        "verdict": "error",
                        "message": message,
                    })
                }
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&json).unwrap_or_default()
            );
        }
        _ => match verdict {
            nixward::support::watchdog::WatchdogVerdict::Stabilized {
                duration,
                checks_performed,
                ..
            } => {
                println!(
                    "  STABILIZED after {}s ({} checks). Safe to promote.",
                    duration.as_secs(),
                    checks_performed
                );
            }
            nixward::support::watchdog::WatchdogVerdict::Degraded {
                reason,
                surprise,
                checks_performed,
                ..
            } => {
                eprintln!(
                    "  DEGRADED: {} (surprise: {:.3}, {} checks)",
                    reason, surprise, checks_performed
                );
            }
            nixward::support::watchdog::WatchdogVerdict::Reverted {
                reason, pre_gen, ..
            } => {
                eprintln!("  REVERTED to r#gen {}: {}", pre_gen, reason);
            }
            nixward::support::watchdog::WatchdogVerdict::Error { message } => {
                eprintln!("  ERROR: {}", message);
            }
        },
    }
}

fn cmd_scrub(file: Option<&str>) {
    use nixward::support::scrubber::Scrubber;

    let input = match file {
        Some(path) => match std::fs::read_to_string(path) {
            Ok(content) => content,
            Err(e) => {
                eprintln!("  Failed to read {}: {}", path, e);
                return;
            }
        },
        None => {
            use std::io::Read;
            let mut buf = String::new();
            if let Err(e) = std::io::stdin().read_to_string(&mut buf) {
                eprintln!("  Failed to read stdin: {}", e);
                return;
            }
            buf
        }
    };

    let scrubber = Scrubber::new();
    let result = scrubber.scrub(&input);
    print!("{}", result.scrubbed_text);

    eprintln!("  --- {} redactions applied ---", result.redaction_count);
}

fn cmd_knowledge(query: &str, limit: usize, format: OutputFormat) {
    use nixward::encoding::NixCodebook;
    use nixward::ipc::default_snapshot_path;
    use nixward::support::knowledge::KnowledgeBase;

    let mut codebook = NixCodebook::new();
    let mut kb = KnowledgeBase::new(&mut codebook);

    // Load dynamic articles from daemon's learned knowledge if available
    let kb_path = default_snapshot_path().with_file_name("knowledge_learned.json");
    if let Ok(json) = std::fs::read_to_string(&kb_path) {
        let before = kb.dynamic_len();
        kb.load_dynamic(&json, &mut codebook);
        let loaded = kb.dynamic_len() - before;
        if loaded > 0 {
            eprintln!("  Loaded {} learned articles from daemon", loaded);
        }
    }

    let static_count = kb.static_len();
    let dynamic_count = kb.dynamic_len();
    let results = kb.search_all(query, &mut codebook, limit);

    match format {
        OutputFormat::Json => {
            let json_results: Vec<serde_json::Value> = results
                .iter()
                .map(|r| {
                    serde_json::json!({
                        "id": r.id(),
                        "title": r.title(),
                        "category": format!("{:?}", r.category()),
                        "similarity": r.similarity(),
                        "solution": r.solution(),
                        "commands": r.commands(),
                        "learned": r.is_dynamic(),
                    })
                })
                .collect();
            println!(
                "{}",
                serde_json::to_string_pretty(&json_results).unwrap_or_default()
            );
        }
        _ => {
            println!("  Knowledge Base: \"{}\"", query);
            if dynamic_count > 0 {
                println!(
                    "  ({} static + {} learned articles)",
                    static_count, dynamic_count
                );
            }
            println!();
            if results.is_empty() {
                println!("  No matching articles found.");
            } else {
                for (i, r) in results.iter().enumerate() {
                    let tag = if r.is_dynamic() { " [learned]" } else { "" };
                    println!(
                        "  {}. {}{} (similarity: {:.3})",
                        i + 1,
                        r.title(),
                        tag,
                        r.similarity()
                    );
                    println!("     Category: {:?}", r.category());
                    println!("     Solution: {}", r.solution());
                    let cmds = r.commands();
                    if !cmds.is_empty() {
                        println!("     Commands:");
                        for cmd in cmds {
                            println!("       $ {}", cmd);
                        }
                    }
                    println!();
                }
            }
        }
    }
}

#[cfg(test)]
mod safety_default_tests {
    use super::DEFAULT_CLI_CONFIRMATION_LEVEL;
    use nixward::action::executor::SafetyLevel;

    /// The un-flagged CLI default must clear ReadOnly/UserModify (so ordinary
    /// commands stay ergonomic) but must NOT clear SystemModify/SystemCritical/
    /// Destructive — those require the operator to explicitly pass `--phi` at or
    /// above the tier's threshold. This is the invariant that makes `nixward
    /// rebuild switch` (and other system-wide commands) safe-by-default instead
    /// of a silent rubber stamp.
    #[test]
    fn default_confirmation_level_blocks_system_and_destructive_tiers() {
        assert!(
            DEFAULT_CLI_CONFIRMATION_LEVEL >= SafetyLevel::ReadOnly.required_phi(),
            "default must allow ReadOnly commands without extra flags"
        );
        assert!(
            DEFAULT_CLI_CONFIRMATION_LEVEL >= SafetyLevel::UserModify.required_phi(),
            "default must allow UserModify commands without extra flags"
        );
        assert!(
            DEFAULT_CLI_CONFIRMATION_LEVEL < SafetyLevel::SystemModify.required_phi(),
            "default must block SystemModify without an explicit --phi override"
        );
        assert!(
            DEFAULT_CLI_CONFIRMATION_LEVEL < SafetyLevel::SystemCritical.required_phi(),
            "default must block SystemCritical (e.g. nixos-rebuild switch) without an explicit --phi override"
        );
        assert!(
            DEFAULT_CLI_CONFIRMATION_LEVEL < SafetyLevel::Destructive.required_phi(),
            "default must block Destructive (e.g. nix-collect-garbage) without an explicit --phi override"
        );
    }
}
