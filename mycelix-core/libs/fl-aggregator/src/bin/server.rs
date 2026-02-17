//! FL Aggregator Server
//!
//! Run with: `cargo run --bin fl-server --features http-api`

use fl_aggregator::{
    aggregator::{AggregatorConfig, AsyncAggregator},
    byzantine::Defense,
    http::{create_router, AppState},
};
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info,tower_http=debug".into()),
        ))
        .with(tracing_subscriber::fmt::layer().json())
        .init();

    // Parse config from environment
    let expected_nodes: usize = std::env::var("FL_EXPECTED_NODES")
        .unwrap_or_else(|_| "10".into())
        .parse()
        .unwrap_or(10);

    let max_memory_mb: usize = std::env::var("FL_MAX_MEMORY_MB")
        .unwrap_or_else(|_| "2000".into())
        .parse()
        .unwrap_or(2000);

    let defense = match std::env::var("FL_DEFENSE")
        .unwrap_or_else(|_| "krum".into())
        .as_str()
    {
        "fedavg" => Defense::FedAvg,
        "krum" => {
            let f: usize = std::env::var("FL_BYZANTINE_F")
                .unwrap_or_else(|_| "1".into())
                .parse()
                .unwrap_or(1);
            Defense::Krum { f }
        }
        "multikrum" => {
            let f: usize = std::env::var("FL_BYZANTINE_F")
                .unwrap_or_else(|_| "1".into())
                .parse()
                .unwrap_or(1);
            let k: usize = std::env::var("FL_MULTIKRUM_K")
                .unwrap_or_else(|_| "3".into())
                .parse()
                .unwrap_or(3);
            Defense::MultiKrum { f, k }
        }
        "median" => Defense::Median,
        "trimmedmean" => {
            let beta: f32 = std::env::var("FL_TRIMMED_BETA")
                .unwrap_or_else(|_| "0.1".into())
                .parse()
                .unwrap_or(0.1);
            Defense::TrimmedMean { beta }
        }
        _ => Defense::Krum { f: 1 },
    };

    let config = AggregatorConfig::default()
        .with_expected_nodes(expected_nodes)
        .with_defense(defense.clone())
        .with_max_memory(max_memory_mb * 1_000_000);

    tracing::info!(
        expected_nodes = expected_nodes,
        max_memory_mb = max_memory_mb,
        defense = %defense,
        "Starting FL Aggregator server"
    );

    let state = Arc::new(AppState {
        aggregator: AsyncAggregator::new(config),
    });

    let app = create_router(state);

    let addr = std::env::var("FL_BIND_ADDR").unwrap_or_else(|_| "0.0.0.0:3000".into());
    let listener = TcpListener::bind(&addr).await?;

    tracing::info!("Listening on {}", addr);

    axum::serve(listener, app).await?;

    Ok(())
}
