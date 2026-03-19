use leptos::prelude::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

use crate::components::glass_panel::GlassPanel;
use crate::state::AppState;
use crate::worker::EngineWorker;

/// The 12 cortical regions used in Symthaea's consciousness topology.
const REGION_NAMES: [&str; 12] = [
    "Prefrontal",
    "Motor",
    "Sensory",
    "Visual",
    "Auditory",
    "Language",
    "Memory",
    "Emotional",
    "Social",
    "Creative",
    "Executive",
    "Integration",
];

/// Region colors (solarpunk palette).
const REGION_COLORS: [&str; 12] = [
    "#7ec8a0", // Prefrontal  — leaf green
    "#d47070", // Motor       — warm red
    "#5b9bd5", // Sensory     — blue
    "#e8c547", // Visual      — solar gold
    "#d4a0c4", // Auditory    — soft pink
    "#5ab8a0", // Language    — teal
    "#c4956a", // Memory      — clay
    "#c76b5a", // Emotional   — autumn rust
    "#a0c4d4", // Social      — sky
    "#d4c470", // Creative    — amber
    "#7ea0c8", // Executive   — steel blue
    "#b0d890", // Integration — spring green
];

/// Node position in the force-directed layout (simple ring initialization).
#[derive(Clone, Copy)]
struct Node {
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    phi: f64,
}

/// Edge between two nodes.
#[derive(Clone, Copy)]
struct Edge {
    from: usize,
    to: usize,
    weight: f64,
}

/// Build the default connectivity: all-to-adjacent + some long-range links
/// resembling cortical connectivity patterns.
fn default_edges() -> Vec<Edge> {
    let mut edges = Vec::new();
    // Ring adjacency
    for i in 0..12 {
        edges.push(Edge {
            from: i,
            to: (i + 1) % 12,
            weight: 0.8,
        });
    }
    // Long-range connections (loosely inspired by neuroscience)
    let long_range = [
        (0, 6, 0.6),  // Prefrontal <-> Memory
        (0, 10, 0.7), // Prefrontal <-> Executive
        (0, 11, 0.5), // Prefrontal <-> Integration
        (2, 3, 0.7),  // Sensory <-> Visual
        (3, 4, 0.5),  // Visual <-> Auditory
        (5, 6, 0.6),  // Language <-> Memory
        (7, 8, 0.6),  // Emotional <-> Social
        (7, 11, 0.4), // Emotional <-> Integration
        (9, 5, 0.5),  // Creative <-> Language
        (1, 10, 0.5), // Motor <-> Executive
        (2, 7, 0.4),  // Sensory <-> Emotional
        (11, 6, 0.6), // Integration <-> Memory
    ];
    for (from, to, weight) in long_range {
        edges.push(Edge { from, to, weight });
    }
    edges
}

/// Initialize nodes in a ring layout.
fn init_nodes(cx: f64, cy: f64, radius: f64) -> Vec<Node> {
    (0..12)
        .map(|i| {
            let angle = (i as f64 / 12.0) * std::f64::consts::TAU - std::f64::consts::FRAC_PI_2;
            Node {
                x: cx + angle.cos() * radius,
                y: cy + angle.sin() * radius,
                vx: 0.0,
                vy: 0.0,
                phi: 0.0,
            }
        })
        .collect()
}

/// One step of a simple force-directed layout.
fn force_step(nodes: &mut [Node], edges: &[Edge], cx: f64, cy: f64) {
    let repulsion = 5000.0;
    let attraction = 0.005;
    let damping = 0.85;
    let center_pull = 0.002;
    let n = nodes.len();

    // Accumulate forces
    let mut fx = vec![0.0_f64; n];
    let mut fy = vec![0.0_f64; n];

    // Repulsion (all pairs)
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = nodes[i].x - nodes[j].x;
            let dy = nodes[i].y - nodes[j].y;
            let dist_sq = dx * dx + dy * dy + 1.0;
            let force = repulsion / dist_sq;
            let dist = dist_sq.sqrt();
            let fdx = force * dx / dist;
            let fdy = force * dy / dist;
            fx[i] += fdx;
            fy[i] += fdy;
            fx[j] -= fdx;
            fy[j] -= fdy;
        }
    }

    // Attraction (edges)
    for edge in edges {
        let dx = nodes[edge.to].x - nodes[edge.from].x;
        let dy = nodes[edge.to].y - nodes[edge.from].y;
        let dist = (dx * dx + dy * dy).sqrt().max(1.0);
        let force = attraction * dist * edge.weight;
        let fdx = force * dx / dist;
        let fdy = force * dy / dist;
        fx[edge.from] += fdx;
        fy[edge.from] += fdy;
        fx[edge.to] -= fdx;
        fy[edge.to] -= fdy;
    }

    // Center pull
    for i in 0..n {
        fx[i] += (cx - nodes[i].x) * center_pull;
        fy[i] += (cy - nodes[i].y) * center_pull;
    }

    // Apply forces
    for i in 0..n {
        nodes[i].vx = (nodes[i].vx + fx[i]) * damping;
        nodes[i].vy = (nodes[i].vy + fy[i]) * damping;
        nodes[i].x += nodes[i].vx;
        nodes[i].y += nodes[i].vy;
    }
}

/// Draw the topology to a canvas context.
fn draw_topology(
    ctx: &CanvasRenderingContext2d,
    nodes: &[Node],
    edges: &[Edge],
    width: f64,
    height: f64,
    time: f64,
) {
    // Clear
    ctx.set_fill_style_str("rgba(0, 0, 0, 0.2)");
    ctx.fill_rect(0.0, 0.0, width, height);

    // Draw edges
    for edge in edges {
        let from = &nodes[edge.from];
        let to = &nodes[edge.to];
        let alpha = 0.15 + edge.weight * 0.3;
        ctx.set_stroke_style_str(&format!("rgba(126, 200, 160, {:.2})", alpha));
        ctx.set_line_width(0.5 + edge.weight * 1.5);
        ctx.begin_path();
        ctx.move_to(from.x, from.y);
        ctx.line_to(to.x, to.y);
        ctx.stroke();
    }

    // Draw nodes
    for (i, node) in nodes.iter().enumerate() {
        let color = REGION_COLORS[i];
        let base_r = 8.0;
        // Pulse radius based on phi and time
        let pulse = 1.0 + node.phi.abs() * 0.5 * (time * 2.0 + i as f64 * 0.5).sin().abs();
        let r = base_r * pulse;

        // Glow
        ctx.set_shadow_color(color);
        ctx.set_shadow_blur(10.0 + node.phi.abs() * 20.0);

        // Fill circle
        ctx.set_fill_style_str(color);
        ctx.begin_path();
        let _ = ctx.arc(node.x, node.y, r, 0.0, std::f64::consts::TAU);
        ctx.fill();

        // Reset shadow
        ctx.set_shadow_blur(0.0);

        // Label
        ctx.set_fill_style_str("rgba(255, 255, 255, 0.7)");
        ctx.set_font("9px -apple-system, sans-serif");
        ctx.set_text_align("center");
        let _ = ctx.fill_text(REGION_NAMES[i], node.x, node.y + r + 14.0);
    }
}

/// Tab 2: Force-directed consciousness topology graph.
#[component]
pub fn TopologyPage() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let engine = use_context::<EngineWorker>().expect("EngineWorker");
    let canvas_ref = NodeRef::<leptos::html::Canvas>::new();

    let (betti_0, set_betti_0) = signal("--".to_string());
    let (betti_1, set_betti_1) = signal("--".to_string());
    let (betti_2, set_betti_2) = signal("--".to_string());
    let (euler, set_euler) = signal("--".to_string());
    let (topo_status, set_topo_status) = signal("Awaiting topology data from SporeEngine...".to_string());

    // Start the canvas animation loop on mount
    Effect::new(move |_| {
        let Some(canvas_el) = canvas_ref.get() else {
            return;
        };
        let canvas: HtmlCanvasElement = canvas_el.into();

        // Set actual pixel dimensions
        let display_width = canvas.client_width() as u32;
        let display_height = 420_u32;
        canvas.set_width(display_width.max(600));
        canvas.set_height(display_height);

        let ctx = canvas
            .get_context("2d")
            .unwrap()
            .unwrap()
            .dyn_into::<CanvasRenderingContext2d>()
            .unwrap();

        let width = canvas.width() as f64;
        let height = canvas.height() as f64;
        let cx = width / 2.0;
        let cy = height / 2.0;
        let radius = height * 0.35;

        let mut nodes = init_nodes(cx, cy, radius);
        let edges = default_edges();

        // Settle the layout with some initial force iterations
        for _ in 0..80 {
            force_step(&mut nodes, &edges, cx, cy);
        }

        // Animation state stored in a closure (single-threaded WASM)
        let nodes_rc = std::rc::Rc::new(std::cell::RefCell::new(nodes));
        let edges_rc = std::rc::Rc::new(edges);
        let frame_ref: std::rc::Rc<std::cell::RefCell<Option<Closure<dyn FnMut()>>>> =
            std::rc::Rc::new(std::cell::RefCell::new(None));
        let frame_ref_clone = frame_ref.clone();

        let closure = Closure::wrap(Box::new(move || {
            let time = web_sys::window()
                .and_then(|w| w.performance())
                .map(|p| p.now() / 1000.0)
                .unwrap_or(0.0);

            let mut nodes = nodes_rc.borrow_mut();
            force_step(&mut nodes, &edges_rc, cx, cy);
            draw_topology(&ctx, &nodes, &edges_rc, width, height, time);

            // Request next frame
            if let Some(window) = web_sys::window() {
                if let Some(ref cb) = *frame_ref_clone.borrow() {
                    let _ = window.request_animation_frame(cb.as_ref().unchecked_ref());
                }
            }
        }) as Box<dyn FnMut()>);

        // Kick off animation
        if let Some(window) = web_sys::window() {
            let _ = window.request_animation_frame(closure.as_ref().unchecked_ref());
        }

        *frame_ref.borrow_mut() = Some(closure);
    });

    // Fetch topology data from the engine when available
    let engine_topo = engine.clone();
    Effect::new(move |_| {
        if !state.worker_ready.get() {
            return;
        }
        let engine = engine_topo.clone();
        let set_b0 = set_betti_0;
        let set_b1 = set_betti_1;
        let set_b2 = set_betti_2;
        let set_eu = set_euler;
        let set_status = set_topo_status;

        wasm_bindgen_futures::spawn_local(async move {
            let promise = engine.send_simple("topologyAnalysis");
            match JsFuture::from(promise).await {
                Ok(result) => {
                    if let Ok(b0) = js_sys::Reflect::get(&result, &"betti_0".into()) {
                        if let Some(v) = b0.as_f64() {
                            set_b0.set(format!("{}", v as i32));
                        }
                    }
                    if let Ok(b1) = js_sys::Reflect::get(&result, &"betti_1".into()) {
                        if let Some(v) = b1.as_f64() {
                            set_b1.set(format!("{}", v as i32));
                        }
                    }
                    if let Ok(b2) = js_sys::Reflect::get(&result, &"betti_2".into()) {
                        if let Some(v) = b2.as_f64() {
                            set_b2.set(format!("{}", v as i32));
                        }
                    }
                    if let Ok(eu) = js_sys::Reflect::get(&result, &"euler_characteristic".into()) {
                        if let Some(v) = eu.as_f64() {
                            set_eu.set(format!("{}", v as i32));
                        }
                    }
                    set_status.set("Topology data loaded.".into());
                }
                Err(_) => {
                    set_status.set("Topology analysis not available in this build.".into());
                }
            }
        });
    });

    view! {
        <GlassPanel title="Consciousness Topology">
            <p style="font-size: 0.82rem; color: var(--fg-dim); line-height: 1.6; margin-bottom: 1rem;">
                "Force-directed graph of the 12 cortical regions, their causal connections, "
                "and real-time information flow. Each node pulses with its local Phi contribution."
            </p>
            <canvas
                node_ref=canvas_ref
                id="topo-canvas"
                width="1200"
                height="420"
            >
                "Canvas not supported"
            </canvas>
            <div class="topo-bottom">
                <GlassPanel title="Betti Numbers">
                    <div class="betti-grid">
                        <div class="betti-item">
                            <div class="betti-val">{move || betti_0.get()}</div>
                            <div class="betti-label">"B0 (components)"</div>
                        </div>
                        <div class="betti-item">
                            <div class="betti-val">{move || betti_1.get()}</div>
                            <div class="betti-label">"B1 (loops)"</div>
                        </div>
                        <div class="betti-item">
                            <div class="betti-val">{move || betti_2.get()}</div>
                            <div class="betti-label">"B2 (voids)"</div>
                        </div>
                        <div class="betti-item">
                            <div class="betti-val">{move || euler.get()}</div>
                            <div class="betti-label">"Euler char"</div>
                        </div>
                    </div>
                </GlassPanel>
                <GlassPanel title="Wave Packets">
                    <div class="wave-packet-list">
                        <div class="wave-packet-item">
                            {move || topo_status.get()}
                        </div>
                    </div>
                </GlassPanel>
            </div>
        </GlassPanel>
    }
}
