/*!
Symthaea: Trace Demo

Runs SymthaeaHLB with a TraceObserver and writes a JSON trace
compatible with tools/symthaea-inspect.
*/

use anyhow::Result;
use symthaea::observability::{create_shared_observer, TraceObserver};
use symthaea::SymthaeaHLB;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging (info for symthaea by default)
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("symthaea=info".parse()?))
        .init();

    println!("\n🧠 Symthaea Trace Demo");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Use default trace path unless overridden via env
    let trace_path =
        std::env::var("SYMTHAEA_TRACE_PATH").unwrap_or_else(|_| "symthaea_trace.json".to_string());

    println!("📁 Writing trace to: {trace_path}");

    // Create shared observer
    let observer = create_shared_observer(TraceObserver::new(&trace_path)?);

    // Use canonical HDC/LTC sizes from the crate
    use symthaea::hdc::{HDC_DIMENSION, LTC_NEURONS};
    let mut symthaea = SymthaeaHLB::with_observer(HDC_DIMENSION, LTC_NEURONS, observer).await?;

    println!("✅ Symthaea initialized ({HDC_DIMENSION}D HDC, {LTC_NEURONS} LTC neurons)\n");

    // Simple scripted queries; in a real app you might use CLI args instead
    let queries = [
        "install nginx",
        "search for vim editor",
        "rebuild my nixos system",
    ];

    for q in &queries {
        println!("📝 Query: {q}");
        let response = symthaea.process(q).await?;
        println!("   → Response: {}", response.content);
        println!("     Confidence: {:.1}%", response.confidence * 100.0);
        println!("     Safe: {}\n", response.safe);
    }

    // Finalize trace (flush + summary)
    symthaea.finalize_trace()?;
    println!("💾 Trace finalized at: {trace_path}\n");

    println!("Tip: analyze with:");
    println!("  symthaea-inspect stats {trace_path}");
    println!("  symthaea-inspect export {trace_path} --metric phi --format csv\n");

    Ok(())
}

