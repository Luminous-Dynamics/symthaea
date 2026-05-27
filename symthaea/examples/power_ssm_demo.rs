// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use anyhow::Result;
use std::time::Duration;

use symthaea::Symthaea;
use symthaea::databases::{DatabaseBackend, DatabaseConfig};

fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(default)
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_string(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}

#[tokio::main]
async fn main() -> Result<()> {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        tracing_subscriber::EnvFilter::new("symthaea::mind::sensor=debug,symthaea::ssm::power=info")
    });
    tracing_subscriber::fmt().with_env_filter(filter).init();

    let hdc_dim = env_usize("SYMTHAEA_HDC_DIM", 1024);
    let ltc_neurons = env_usize("SYMTHAEA_LTC_NEURONS", 64);
    let ticks = env_usize("SYMTHAEA_SSM_TICKS", 30);
    let hz = env_f32("SYMTHAEA_SSM_HZ", 10.0).max(0.1);
    let tick = Duration::from_secs_f32(1.0 / hz);

    let mut sym = if let Some(path) = env_string("SYMTHAEA_DB_PATH") {
        if let Some(parent) = std::path::Path::new(&path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let db_config = DatabaseConfig {
            backend: DatabaseBackend::Sqlite,
            path: Some(path),
        };
        Symthaea::with_database(hdc_dim, ltc_neurons, db_config).await?
    } else {
        Symthaea::new(hdc_dim, ltc_neurons).await?
    };
    #[cfg(feature = "ssm-power")]
    sym.attach_power_ssm_sensor()?;

    println!("Power SSM demo: {ticks} ticks at {hz:.2} Hz");
    for idx in 0..ticks {
        let start = std::time::Instant::now();
        sym.mind_mut().tick();
        if idx % 5 == 0 {
            println!(
                "tick {:03} | working_memory={}",
                idx,
                sym.mind().working_memory().len()
            );
        }
        let elapsed = start.elapsed();
        if tick > elapsed {
            std::thread::sleep(tick - elapsed);
        }
    }

    // Flush evicted working-memory items into the database via the processing pipeline.
    let _ = sym
        .process("Flush interoceptive memory evictions to persistent storage.")
        .await?;

    // Socratic recall query.
    let query = "Reflect on your recent thermodynamic load. Was your energy consumption stable?";
    let response = sym.process(query).await?;
    println!(
        "\nSocratic prompt: {query}\nResponse: {}\n",
        response.content
    );

    if let Some(db) = sym.database() {
        let records = db.list_all().await?;
        let thermo: Vec<_> = records
            .iter()
            .filter(|r| {
                r.topics
                    .iter()
                    .any(|t| t == symthaea::consciousness::interoception::InteroceptionTag::ThermodynamicLoad.as_topic())
            })
            .collect();
        println!(
            "Thermodynamic records in DB: {} (total records: {})",
            thermo.len(),
            records.len()
        );
        if let Some(latest) = thermo.last() {
            println!("Latest thermodynamic record: {}", latest.content);
        }
    }

    Ok(())
}
