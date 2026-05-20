use image::{ImageBuffer, Rgb};
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService, SwarmEvent};
use symthaea_swarm::SwarmStateMsg;

fn main() {
    println!("Initializing Mind's Eye...");
    let mut config = CognitiveLoopConfig::default();
    config.enable_vision_manifold = true;

    let mut node_a = CognitiveLoopService::new(config.clone()).unwrap();
    let id_a = node_a.node_id().unwrap();

    let mut node_b = CognitiveLoopService::new(config).unwrap();

    // 1. Shock Node A
    let shock_frame = vec![255u8; 64 * 64 * 3];
    node_a.inject_vision_frame(shock_frame);
    node_a.set_vision_free_energy_override(0.9);
    for _ in 0..5 {
        let _ = node_a.cycle("load");
    }

    // 2. Transmit SOS
    let sos_msg = SwarmStateMsg {
        node_id: id_a,
        local_phi: 0.88,
        consciousness_hv: node_a.consciousness_hv().unwrap(),
        intent_hv: node_a.last_intent_hv().unwrap(),
        timestamp: 0,
    };
    node_b
        .swarm_manager_mut()
        .inject_event(SwarmEvent::FullStateUpdate(sos_msg));
    for _ in 0..50 {
        let _ = node_b.cycle("process-sos");
    }

    // 3. Initialize Node B's subcortical canvas
    //    (The decoder needs a baseline to project physics against)
    let baseline_frame = vec![128u8; 64 * 64 * 3]; // Neutral gray
    node_b.inject_vision_frame(baseline_frame);
    let _ = node_b.cycle("sync-baseline");

    // 4. Generate Dream
    println!("Synthesizing 12-frame geodesic movie...");
    match node_b.collaborative_imagine_future(&id_a, 12) {
        Ok(movie) => {
            println!("Dream successful! Saving PNGs...");
            for (i, frame_data) in movie.frames.iter().enumerate() {
                // Convert raw RGB bytes to an image
                let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_raw(
                    movie.width as u32,
                    movie.height as u32,
                    frame_data.clone(),
                )
                .unwrap();
                let filename = format!("dream_frame_{:02}.png", i);
                img.save(&filename).unwrap();
                println!("Saved {}", filename);
            }
            println!("All frames saved. Download them to view the dream.");
        }
        Err(e) => println!("Dream failed: {:?}", e),
    }
}
