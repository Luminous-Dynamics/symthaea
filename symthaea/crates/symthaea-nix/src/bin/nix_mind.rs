//! nix-mind: Conscious NixOS management CLI.
//!
//! Entry point that dispatches to subcommands or processes
//! natural language input through the cognitive core.

use clap::Parser;
use symthaea_nix::cli::commands::{Cli, Command, OutputFormat, RebuildMode, ObserveDomain};
use symthaea_nix::cli::interactive;
use symthaea_nix::cli::completions;
use symthaea_nix::action::executor::NixOSCommand;
use symthaea_nix::action::service_manager::ServiceManager;
use symthaea_nix::action::generation_manager::GenerationManager;
use symthaea_nix::action::gc_manager::GcManager;
use symthaea_nix::action::flake_ops::FlakeOps;

fn main() {
    let cli = Cli::parse();

    // Natural language input takes priority
    if cli.has_natural_input() {
        interactive::process_oneshot(
            &cli.natural_input(),
            cli.dry_run,
            cli.format,
            cli.phi,
        );
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
        Command::Search { query, options, limit } => {
            cmd_search(&query, options, limit, cli.format);
        }

        Command::Rebuild { mode, flake, extra_args } => {
            let cmd = match mode {
                RebuildMode::Switch => NixOSCommand::RebuildSwitch { flake, extra_args },
                RebuildMode::Test => NixOSCommand::RebuildTest { flake, extra_args },
                RebuildMode::Boot => NixOSCommand::RebuildBoot { flake, extra_args },
            };
            cmd_execute(cmd, cli.dry_run, cli.phi);
        }

        Command::Rollback { generation } => {
            let cmd = if let Some(gen) = generation {
                GenerationManager::switch_to(gen)
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

        Command::Generations { diff, from, to, delete_older_than } => {
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
            use symthaea_nix::cli::commands::FlakeCommand;
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

        Command::Gc { analyze, older_than, aggressive } => {
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
            use symthaea_nix::cli::commands::ServiceCommand;
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
    }
}

// ---- Command implementations ----

fn cmd_search(query: &str, options: bool, limit: usize, format: OutputFormat) {
    if options {
        // HDC semantic search over NixOS options
        use symthaea_nix::encoding::{NixCodebook, search_options};

        let mut codebook = NixCodebook::new();

        // Common NixOS option paths for semantic search
        let known_paths = [
            "services.nginx.enable", "services.nginx.package", "services.nginx.virtualHosts",
            "services.postgresql.enable", "services.postgresql.package",
            "services.openssh.enable", "services.openssh.settings.PermitRootLogin",
            "services.pipewire.enable", "services.pipewire.alsa.enable",
            "services.docker.enable", "services.podman.enable",
            "services.xserver.enable", "services.xserver.displayManager.gdm.enable",
            "services.xserver.desktopManager.gnome.enable",
            "networking.firewall.enable", "networking.firewall.allowedTCPPorts",
            "networking.networkmanager.enable", "networking.wireguard.enable",
            "boot.loader.grub.enable", "boot.loader.systemd-boot.enable",
            "hardware.opengl.enable", "hardware.nvidia.modesetting.enable",
            "hardware.pulseaudio.enable", "hardware.bluetooth.enable",
            "security.sudo.enable", "security.polkit.enable",
            "users.users", "users.defaultUserShell",
            "environment.systemPackages", "nixpkgs.config.allowUnfree",
            "programs.zsh.enable", "programs.fish.enable", "programs.steam.enable",
            "nix.gc.automatic", "nix.settings.experimental-features",
        ];
        let path_refs: Vec<&str> = known_paths.iter().copied().collect();

        let results = search_options(query, &mut codebook, &path_refs, limit);

        match format {
            OutputFormat::Json => {
                let json_results: Vec<serde_json::Value> = results.iter().map(|r| {
                    serde_json::json!({
                        "path": r.path,
                        "similarity": r.similarity,
                        "reason": r.match_reason,
                    })
                }).collect();
                println!("{}", serde_json::to_string_pretty(&json_results).unwrap_or_default());
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

fn cmd_execute(cmd: NixOSCommand, dry_run: bool, phi_override: Option<f64>) {
    let phi = phi_override.unwrap_or(0.7) as f32;
    let safety = cmd.safety_level();
    let required = safety.required_phi();

    if phi < required {
        eprintln!(
            "  Blocked: {:?} requires Phi >= {:.2}, current Phi = {:.2}",
            safety, required, phi
        );
        eprintln!("  Use --phi={:.1} to override, or confirm explicitly.", required);
        return;
    }

    let (bin, args) = cmd.to_command();
    if dry_run {
        println!("  [DRY-RUN] Would execute: {} {}", bin, args.join(" "));
    } else {
        println!("  Executing: {} {}", bin, args.join(" "));
        let status = std::process::Command::new(&bin)
            .args(&args)
            .status();
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

fn cmd_observe(domain: Option<ObserveDomain>, format: OutputFormat) {
    match domain {
        Some(ObserveDomain::Services) => {
            match symthaea_nix::observe::systemd::SystemdObserver::list_units() {
                Ok(units) => {
                    for unit in &units {
                        println!("  {} {} {} {}", unit.name, unit.active_state, unit.sub_state, unit.description);
                    }
                    println!("  {} units total", units.len());
                }
                Err(e) => eprintln!("  Failed to list services: {}", e),
            }
        }
        Some(ObserveDomain::Store) => {
            match symthaea_nix::observe::store::StoreObserver::store_info() {
                Ok(info) => {
                    println!("  Store paths: {}", info.path_count);
                    println!("  Total size: {} bytes", info.total_size_bytes);
                }
                Err(e) => eprintln!("  Failed to query store: {}", e),
            }
        }
        Some(ObserveDomain::Generations) => {
            cmd_generations_list(format);
        }
        Some(ObserveDomain::Hardware) => {
            match symthaea_nix::observe::hardware::HardwareObserver::probe() {
                Ok(info) => {
                    println!("  CPU: {} ({} cores)", info.cpu_model, info.cpu_cores);
                    println!("  Memory: {} MiB", info.memory_total_mb);
                    for gpu in &info.gpus {
                        let driver = gpu.driver.as_deref().unwrap_or("unknown");
                        println!("  GPU: {} ({})", gpu.name, driver);
                    }
                }
                Err(e) => eprintln!("  Failed to probe hardware: {}", e),
            }
        }
        _ => {
            println!("  Taking system snapshot...");
            match symthaea_nix::observe::SystemObserver::snapshot() {
                Ok(snap) => {
                    println!("  Services: {}", snap.services.len());
                    println!("  Packages: {}", snap.packages.len());
                    if let Some(gen) = snap.generation {
                        println!("  Generation: {}", gen);
                    }
                    if let Some(size) = snap.store_size_bytes {
                        println!("  Store size: {} bytes", size);
                    }
                }
                Err(e) => eprintln!("  Failed to snapshot: {}", e),
            }
        }
    }
}

fn cmd_doctor(_format: OutputFormat) {
    println!("  Running system diagnostics...");
    println!();

    // Check failed services
    match symthaea_nix::observe::systemd::SystemdObserver::list_units() {
        Ok(units) => {
            let failed: Vec<_> = units.iter().filter(|u| u.active_state == "failed").collect();
            if failed.is_empty() {
                println!("  Services: All OK");
            } else {
                println!("  Services: {} FAILED", failed.len());
                for unit in &failed {
                    println!("    - {} ({})", unit.name, unit.sub_state);
                }
            }
        }
        Err(_) => println!("  Services: Could not query systemd"),
    }

    // Check store health
    match GcManager::analyze() {
        Ok(analysis) => {
            let rec = GcManager::recommend(&analysis);
            println!("  Store: {} total, {} reclaimable", analysis.total_store_human(), analysis.reclaimable_human());
            if rec.recommended {
                println!("    Recommendation: {}", rec.reason);
            }
        }
        Err(_) => println!("  Store: Could not analyze"),
    }

    // Check generations
    match GenerationManager::list() {
        Ok(gens) => {
            println!("  Generations: {} total", gens.len());
            if gens.len() > 20 {
                println!("    Consider cleaning up old generations");
            }
        }
        Err(_) => println!("  Generations: Could not list"),
    }

    println!();
}

fn cmd_generations_list(_format: OutputFormat) {
    match GenerationManager::list() {
        Ok(gens) => {
            for gen in &gens {
                let current = if gen.current { " (current)" } else { "" };
                println!(
                    "  {}  {}  {}  {}{}",
                    gen.number, gen.date, gen.nixos_version, gen.kernel_version, current
                );
            }
            println!("  {} generations total", gens.len());
        }
        Err(e) => eprintln!("  Failed to list generations: {}", e),
    }
}

fn cmd_generations_diff(from: Option<u32>, to: Option<u32>) {
    let (from, to) = match (from, to) {
        (Some(f), Some(t)) => (f, t),
        _ => {
            match GenerationManager::list() {
                Ok(gens) if gens.len() >= 2 => {
                    (gens[1].number, gens[0].number)
                }
                _ => {
                    eprintln!("  Need at least 2 generations for diff. Use --from and --to.");
                    return;
                }
            }
        }
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
                    println!("{}", serde_json::to_string_pretty(&json).unwrap_or_default());
                }
                _ => {
                    println!("  Store Analysis:");
                    println!("    Total:       {}", analysis.total_store_human());
                    println!("    Reclaimable: {} ({:.0}%)", analysis.reclaimable_human(), analysis.reclaimable_percent());
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

    match FlakeOps::metadata(&cwd) {
        Ok(meta) => {
            println!("  Flake: {}", meta.url);
            if !meta.description.is_empty() {
                println!("  Description: {}", meta.description);
            }
            if let Some(modified) = &meta.last_modified {
                println!("  Last modified: {}", modified);
            }
            if !meta.inputs.is_empty() {
                println!("  Inputs:");
                for (name, url) in &meta.inputs {
                    println!("    {}: {}", name, url);
                }
            }
        }
        Err(e) => eprintln!("  Failed to get flake metadata: {}", e),
    }
}

fn cmd_service_status(name: &str, format: OutputFormat) {
    match ServiceManager::status(name) {
        Ok(status) => {
            match format {
                OutputFormat::Json => {
                    let json = serde_json::json!({
                        "name": status.name,
                        "active": status.active,
                        "enabled": status.enabled,
                        "active_state": status.active_state,
                        "sub_state": status.sub_state,
                    });
                    println!("{}", serde_json::to_string_pretty(&json).unwrap_or_default());
                }
                _ => {
                    let indicator = if status.active { "active" } else { "inactive" };
                    let enabled = if status.enabled { "enabled" } else { "disabled" };
                    println!("  {} ({}, {})", status.name, indicator, enabled);
                    println!("    State: {} ({})", status.active_state, status.sub_state);
                }
            }
        }
        Err(e) => eprintln!("  Failed to get status for {}: {}", name, e),
    }
}

fn cmd_service_failed(_format: OutputFormat) {
    match symthaea_nix::observe::systemd::SystemdObserver::failed_units() {
        Ok(units) => {
            if units.is_empty() {
                println!("  No failed services.");
            } else {
                println!("  Failed services:");
                for unit in &units {
                    println!("    {} ({})", unit.name, unit.description);
                }
            }
        }
        Err(e) => eprintln!("  Failed to list failed services: {}", e),
    }
}
