// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Interactive Story REPL for Symthaea Narrative Dynamics
//!
//! A command-line tool for composing stories through the HDC+CfC narrative
//! pipeline. Each scene produces a Ghost Signal (energy, surprise, valence,
//! tension, momentum) that captures the continuous dynamics of the story.
//!
//! Run: `cargo run --bin symthaea-story`

use std::io::{self, BufRead, Write};

use symthaea::dynamics::story_session::StorySession;
use symthaea::hdc::narrative_algebra::NarrativeMood;

fn main() {
    println!("=== Symthaea Story REPL ===");
    println!("Type /help for commands.\n");

    let mut session = StorySession::new();
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("story> ");
        stdout.flush().ok();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).is_err() || line.is_empty() {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if line.starts_with('/') {
            let parts: Vec<&str> = line.splitn(2, ' ').collect();
            let cmd = parts[0];
            let arg = parts.get(1).copied().unwrap_or("");

            match cmd {
                "/help" => print_help(),
                "/quit" | "/exit" => {
                    println!("Goodbye.");
                    break;
                }
                "/reset" => {
                    session.reset();
                    session = StorySession::new();
                    println!("Session reset.");
                }
                "/character" => cmd_character(&mut session, arg),
                "/theme" => {
                    if arg.is_empty() {
                        println!("Usage: /theme <text>");
                    } else {
                        session.add_theme(arg);
                        println!("Theme added: {}", arg);
                    }
                }
                "/scene" => cmd_scene(&mut session, arg),
                "/conflict" => {
                    if arg.is_empty() {
                        println!("Usage: /conflict <description>");
                    } else {
                        session.introduce_conflict(arg);
                        println!("Conflict introduced: {}", arg);
                    }
                }
                "/resolve" => {
                    if arg.is_empty() {
                        println!("Usage: /resolve <description>");
                    } else if session.resolve_conflict(arg) {
                        println!("Conflict resolved: {}", arg);
                    } else {
                        println!("No unresolved conflict matching: {}", arg);
                    }
                }
                "/scenes" => {
                    if let Ok(n) = arg.trim().parse::<usize>() {
                        session.set_expected_scenes(n);
                        println!(
                            "Expected scenes set to {} (enables position-aware tension shaping)",
                            n
                        );
                    } else {
                        println!("Usage: /scenes <number>  (e.g. /scenes 7)");
                    }
                }
                "/status" => cmd_status(&session),
                "/signal" => cmd_signal(&session),
                "/save" => cmd_save(&session, arg),
                "/load" => cmd_load(&mut session, arg),
                _ => println!("Unknown command: {}. Type /help for help.", cmd),
            }
        } else {
            println!("Type /help for commands, or use /scene to add a scene.");
        }
    }
}

fn print_help() {
    println!(
        r#"Commands:
  /character <name> <role>    Register a character (roles: protagonist, antagonist, other)
  /theme <text>               Add a thematic throughline
  /scene Title|Setting|Chars|Conflict|Mood
                              Add a scene (moods: peaceful, tense, mysterious,
                              melancholy, triumphant, hopeful)
  /scenes <number>             Set expected total scenes (enables position-aware shaping)
  /conflict <description>     Introduce a conflict
  /resolve <description>      Resolve a conflict
  /status                     Story state summary
  /signal                     ASCII tension trajectory
  /save <path>                Save session to file
  /load <path>                Load session from file
  /reset                      Start a new story
  /quit                       Exit"#
    );
}

fn cmd_character(session: &mut StorySession, arg: &str) {
    let parts: Vec<&str> = arg.splitn(2, ' ').collect();
    if parts.len() < 2 {
        println!("Usage: /character <name> <role>");
        println!("  Roles: protagonist, antagonist, other");
        return;
    }
    let name = parts[0];
    let role = parts[1].trim().to_lowercase();
    let prim = match role.as_str() {
        "protagonist" => session.algebra().primitives.protagonist.clone(),
        "antagonist" => session.algebra().primitives.antagonist.clone(),
        _ => session.algebra().primitives.theme.clone(), // generic "other"
    };
    session.register_character(name, &prim);
    println!("Character registered: {} ({})", name, role);
}

fn parse_mood(s: &str) -> NarrativeMood {
    match s.trim().to_lowercase().as_str() {
        "peaceful" => NarrativeMood::Peaceful,
        "tense" => NarrativeMood::Tense,
        "mysterious" => NarrativeMood::Mysterious,
        "melancholy" => NarrativeMood::Melancholy,
        "triumphant" => NarrativeMood::Triumphant,
        "hopeful" => NarrativeMood::Hopeful,
        _ => {
            println!("  (unknown mood '{}', defaulting to Mysterious)", s.trim());
            NarrativeMood::Mysterious
        }
    }
}

fn cmd_scene(session: &mut StorySession, arg: &str) {
    let fields: Vec<&str> = arg.split('|').collect();
    if fields.len() < 5 {
        println!("Usage: /scene Title|Setting|Characters|Conflict|Mood");
        println!("  Characters: comma-separated names");
        println!("  Mood: peaceful, tense, mysterious, melancholy, triumphant, hopeful");
        return;
    }

    let title = fields[0].trim();
    let setting = fields[1].trim();
    let chars_str = fields[2].trim();
    let conflict = fields[3].trim();
    let mood = parse_mood(fields[4]);

    let char_names: Vec<&str> = chars_str.split(',').map(|s| s.trim()).collect();

    let signal = session.add_scene(title, setting, &char_names, conflict, mood);

    println!("  Scene added: {}", title);
    println!(
        "  Ghost Signal: energy={:.3}  surprise={:.3}  valence={:.3}  tension={:.3}  momentum={:.3}",
        signal.energy, signal.surprise, signal.valence, signal.tension, signal.momentum,
    );
    println!("  Arc phase: {:?}", signal.arc_phase);
}

fn cmd_status(session: &StorySession) {
    let state = session.get_story_state();
    println!("  Scenes:      {}", state.total_scenes);
    println!("  Characters:  {}", state.character_count);
    println!(
        "  Conflicts:   {} ({} unresolved)",
        state.total_conflicts, state.unresolved_conflicts
    );
    println!("  Themes:      {:?}", state.themes);
    println!("  Arc phase:   {:?}", state.current_arc_phase);
    if let Some(ref sig) = state.latest_signal {
        println!(
            "  Latest:      E={:.3} S={:.3} V={:.3} T={:.3} M={:.3}",
            sig.energy, sig.surprise, sig.valence, sig.tension, sig.momentum,
        );
    }
}

fn cmd_signal(session: &StorySession) {
    let log = session.scene_log();
    if log.is_empty() {
        println!("  No scenes yet.");
        return;
    }
    println!("  Tension trajectory:");
    for record in log {
        let bars = (record.signal.tension * 40.0).round() as usize;
        let bar = "#".repeat(bars);
        println!(
            "  {:>3}. {:<20} |{:<40}| {:.3}",
            record.index + 1,
            &record.title[..record.title.len().min(20)],
            bar,
            record.signal.tension,
        );
    }
}

fn cmd_save(session: &StorySession, path: &str) {
    if path.is_empty() {
        println!("Usage: /save <path>");
        return;
    }
    match session.save() {
        Ok(json) => match std::fs::write(path, &json) {
            Ok(_) => println!("Saved to {}", path),
            Err(e) => println!("Write error: {}", e),
        },
        Err(e) => println!("Serialization error: {}", e),
    }
}

fn cmd_load(session: &mut StorySession, path: &str) {
    if path.is_empty() {
        println!("Usage: /load <path>");
        return;
    }
    match std::fs::read_to_string(path) {
        Ok(json) => match StorySession::load(&json) {
            Ok(restored) => {
                *session = restored;
                let state = session.get_story_state();
                println!(
                    "Loaded from {}: {} scenes, {} characters",
                    path, state.total_scenes, state.character_count
                );
            }
            Err(e) => println!("Parse error: {}", e),
        },
        Err(e) => println!("Read error: {}", e),
    }
}
