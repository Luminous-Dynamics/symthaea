// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use std::cell::RefCell;
use std::rc::Rc;
use leptos::prelude::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{HtmlCanvasElement, MouseEvent, WheelEvent, WebGl2RenderingContext};

use crate::data::geo;
use crate::data::types::*;
use crate::renderer::geometry;
use crate::renderer::math::Vec3;
use crate::renderer::picking;
use crate::renderer::{GlobeRenderer, MarkerInstance};
use crate::state::data_state::DataState;
use crate::state::globe_state::GlobeState;

/// Flat list of marker positions + metadata for picking.
struct PickableMarker {
    position: Vec3,
    info: HoverInfo,
    selected: SelectedItem,
}

#[component]
pub fn GlobeCanvas() -> impl IntoView {
    let canvas_ref = NodeRef::<leptos::html::Canvas>::new();
    let globe_state = expect_context::<GlobeState>();
    let data_state = expect_context::<DataState>();

    Effect::new(move |_| {
        let Some(canvas) = canvas_ref.get() else {
            log::warn!("Canvas ref not yet available");
            return;
        };
        let canvas_el: HtmlCanvasElement = canvas.into();

        let window = web_sys::window().unwrap();
        let w = window.inner_width().unwrap().as_f64().unwrap() as u32;
        let h = window.inner_height().unwrap().as_f64().unwrap() as u32;
        canvas_el.set_width(w);
        canvas_el.set_height(h);

        let gl: WebGl2RenderingContext = match canvas_el.get_context("webgl2") {
            Ok(Some(ctx)) => ctx.dyn_into().unwrap(),
            Ok(None) => { log::error!("WebGL2 not supported"); return; }
            Err(e) => { log::error!("WebGL2 context error: {:?}", e); return; }
        };

        log::info!("WebGL2 context created");

        let renderer = match GlobeRenderer::init(gl) {
            Ok(r) => { log::info!("GlobeRenderer initialized successfully"); r }
            Err(e) => { log::error!("GlobeRenderer init failed: {e}"); return; }
        };

        let renderer = Rc::new(RefCell::new(renderer));

        // Shared pickable markers list (rebuilt when layers change)
        let pickables: Rc<RefCell<Vec<PickableMarker>>> = Rc::new(RefCell::new(Vec::new()));

        // Track mouse-down position to distinguish click from drag
        let mouse_down_pos: Rc<RefCell<Option<(f32, f32)>>> = Rc::new(RefCell::new(None));

        // Clone handles for closures
        let r_md = renderer.clone();
        let mdp = mouse_down_pos.clone();

        // ── Mouse Down ──
        let mouse_down = Closure::wrap(Box::new(move |e: MouseEvent| {
            let x = e.client_x() as f32;
            let y = e.client_y() as f32;
            r_md.borrow_mut().camera.on_mouse_down(x, y);
            *mdp.borrow_mut() = Some((x, y));
        }) as Box<dyn FnMut(MouseEvent)>);
        canvas_el.add_event_listener_with_callback("mousedown", mouse_down.as_ref().unchecked_ref()).unwrap();
        mouse_down.forget();

        // ── Mouse Move (drag + hover picking) ──
        let r_mm = renderer.clone();
        let gs_mm = globe_state.clone();
        let pick_mm = pickables.clone();
        let canvas_mm = canvas_el.clone();

        let mouse_move = Closure::wrap(Box::new(move |e: MouseEvent| {
            let x = e.client_x() as f32;
            let y = e.client_y() as f32;

            r_mm.borrow_mut().camera.on_mouse_move(x, y);
            gs_mm.mouse_pos.set((x, y));

            // Hover picking (only when not dragging)
            if !r_mm.borrow().camera.is_dragging() {
                let cw = canvas_mm.client_width() as f32;
                let ch = canvas_mm.client_height() as f32;
                let r = r_mm.borrow();
                let proj = r.camera.projection_matrix(cw / ch);
                let view = r.camera.view_matrix(r.time_secs());

                if let Some((origin, dir)) = picking::screen_to_ray(x, y, cw, ch, &proj, &view) {
                    if let Some(hit) = picking::ray_sphere_intersect(origin, dir, Vec3::ZERO, 1.02) {
                        let picks = pick_mm.borrow();
                        let positions: Vec<Vec3> = picks.iter().map(|p| p.position).collect();
                        if let Some(idx) = picking::find_nearest_marker(hit, &positions, 0.08) {
                            gs_mm.hovered.set(Some(picks[idx].info.clone()));
                        } else {
                            gs_mm.hovered.set(None);
                        }
                    } else {
                        gs_mm.hovered.set(None);
                    }
                }
            }
        }) as Box<dyn FnMut(MouseEvent)>);
        canvas_el.add_event_listener_with_callback("mousemove", mouse_move.as_ref().unchecked_ref()).unwrap();
        mouse_move.forget();

        // ── Mouse Up (click detection) ──
        let r_mu = renderer.clone();
        let gs_mu = globe_state.clone();
        let pick_mu = pickables.clone();
        let mdp_mu = mouse_down_pos.clone();
        let canvas_mu = canvas_el.clone();

        let mouse_up = Closure::wrap(Box::new(move |e: MouseEvent| {
            r_mu.borrow_mut().camera.on_mouse_up();

            // Detect click (not drag): mouse moved less than 5px
            let x = e.client_x() as f32;
            let y = e.client_y() as f32;
            if let Some((dx, dy)) = mdp_mu.borrow().as_ref() {
                let dist = ((x - dx).powi(2) + (y - dy).powi(2)).sqrt();
                if dist < 5.0 {
                    // This is a click — select the hovered marker
                    let cw = canvas_mu.client_width() as f32;
                    let ch = canvas_mu.client_height() as f32;
                    let r = r_mu.borrow();
                    let proj = r.camera.projection_matrix(cw / ch);
                    let view = r.camera.view_matrix(r.time_secs());

                    if let Some((origin, dir)) = picking::screen_to_ray(x, y, cw, ch, &proj, &view) {
                        if let Some(hit) = picking::ray_sphere_intersect(origin, dir, Vec3::ZERO, 1.02) {
                            let picks = pick_mu.borrow();
                            let positions: Vec<Vec3> = picks.iter().map(|p| p.position).collect();
                            if let Some(idx) = picking::find_nearest_marker(hit, &positions, 0.08) {
                                gs_mu.selected.set(Some(picks[idx].selected.clone()));
                            } else {
                                gs_mu.selected.set(None);
                            }
                        } else {
                            gs_mu.selected.set(None);
                        }
                    }
                }
            }
            *mdp_mu.borrow_mut() = None;
        }) as Box<dyn FnMut(MouseEvent)>);
        canvas_el.add_event_listener_with_callback("mouseup", mouse_up.as_ref().unchecked_ref()).unwrap();
        mouse_up.forget();

        // ── Wheel ──
        let r_wh = renderer.clone();
        let wheel = Closure::wrap(Box::new(move |e: WheelEvent| {
            e.prevent_default();
            r_wh.borrow_mut().camera.on_wheel(e.delta_y() as f32);
        }) as Box<dyn FnMut(WheelEvent)>);
        let mut opts = web_sys::AddEventListenerOptions::new();
        opts.passive(false);
        canvas_el.add_event_listener_with_callback_and_add_event_listener_options(
            "wheel", wheel.as_ref().unchecked_ref(), &opts,
        ).unwrap();
        wheel.forget();

        // Build initial data
        update_renderer_data(&renderer, &pickables, &data_state, &globe_state);

        // ── Animation Loop ──
        let canvas_for_frame = canvas_el.clone();
        let r_frame = renderer.clone();
        let ds = data_state.clone();
        let gs2 = globe_state.clone();
        let picks_frame = pickables.clone();

        let f: Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>> = Rc::new(RefCell::new(None));
        let g = f.clone();
        let mut last_layer_hash = 0u64;
        let mut frame_count = 0u32;

        *g.borrow_mut() = Some(Closure::wrap(Box::new(move |time: f64| {
            let layer_hash = {
                let layers = gs2.active_layers.read();
                let mut h = 0u64;
                for l in layers.iter() { h ^= *l as u64 * 2654435761; }
                h ^= gs2.timeline_year.get() as u64 * 1000003; // timeline changes trigger rebuild
                h
            };
            if layer_hash != last_layer_hash {
                last_layer_hash = layer_hash;
                update_renderer_data(&r_frame, &picks_frame, &ds, &gs2);
            }

            let cw = canvas_for_frame.client_width() as u32;
            let ch = canvas_for_frame.client_height() as u32;
            if canvas_for_frame.width() != cw || canvas_for_frame.height() != ch {
                canvas_for_frame.set_width(cw);
                canvas_for_frame.set_height(ch);
            }

            {
                let mut renderer = r_frame.borrow_mut();
                renderer.show_core = gs2.show_core.get();

                // Handle planet fly-to navigation
                use crate::components::planet_nav::PlanetTarget;
                if let Some(planet) = gs2.focused_planet.get() {
                    if !renderer.camera.is_flying() {
                        let t = (time / 1000.0) as f32;
                        let (dist, speed, offset) = planet.orbit_params();
                        if dist > 0.01 {
                            // Compute current planet position
                            let angle = t * speed + offset;
                            let pos = crate::renderer::math::Vec3::new(
                                dist * angle.cos(),
                                0.0,
                                dist * angle.sin(),
                            );
                            renderer.camera.fly_to(pos, planet.camera_distance());
                        } else {
                            // Earth: fly home
                            renderer.camera.fly_home();
                        }
                    }
                }

                renderer.frame(time, cw, ch);
            }

            frame_count += 1;
            if frame_count == 1 {
                log::info!("First frame rendered ({}x{})", cw, ch);
            }

            let window = web_sys::window().unwrap();
            let _ = window.request_animation_frame(
                f.borrow().as_ref().unwrap().as_ref().unchecked_ref()
            );
        }) as Box<dyn FnMut(f64)>));

        let window = web_sys::window().unwrap();
        let _ = window.request_animation_frame(
            g.borrow().as_ref().unwrap().as_ref().unchecked_ref()
        );

        log::info!("Animation loop started");
    });

    view! {
        <canvas id="globe-canvas" node_ref=canvas_ref />
    }
}

fn update_renderer_data(
    renderer: &Rc<RefCell<GlobeRenderer>>,
    pickables: &Rc<RefCell<Vec<PickableMarker>>>,
    data_state: &DataState,
    globe_state: &GlobeState,
) {
    let layers = globe_state.active_layers.read();
    let mut markers = Vec::new();
    let mut picks = Vec::new();
    let mut arc_datas = Vec::new();

    if layers.contains(&Layer::Geothermal) {
        let nodes = data_state.geothermal_nodes.read();
        for node in nodes.iter() {
            let pos = geo::lat_lon_to_xyz(node.lat, node.lon, 1.005);
            let size = geo::marker_size_from_capacity(node.capacity_mw) * 2.5;
            markers.push(MarkerInstance {
                position: pos, color: Vec3::new(0.937, 0.267, 0.267),
                size, marker_type: 1.0,
            });
            picks.push(PickableMarker {
                position: pos,
                info: HoverInfo::GeothermalNode(node.clone()),
                selected: SelectedItem::GeothermalNode(node.clone()),
            });
        }
    }

    let timeline_year = globe_state.timeline_year.get();

    if layers.contains(&Layer::ResontiaVaults) {
        let vaults = data_state.resontia_vaults.read();
        // Vaults appear progressively: first at year 50, all 12 by year 150
        let max_vaults = match timeline_year {
            0..=49 => 0,
            50..=74 => 2,
            75..=99 => 4,
            100..=124 => 7,
            125..=149 => 10,
            _ => 12,
        };
        for (idx, vault) in vaults.iter().enumerate() {
            if idx >= max_vaults { break; }
            let pos = geo::lat_lon_to_xyz(vault.lat, vault.lon, 1.005);
            let color = if timeline_year >= 150 {
                Vec3::new(0.204, 0.827, 0.600) // operational (green)
            } else {
                Vec3::new(0.984, 0.749, 0.141) // under construction (amber)
            };
            markers.push(MarkerInstance {
                position: pos, color, size: 0.04, marker_type: 2.0,
            });
            picks.push(PickableMarker {
                position: pos,
                info: HoverInfo::ResontiaVault(vault.clone()),
                selected: SelectedItem::ResontiaVault(vault.clone()),
            });
        }
    }

    if layers.contains(&Layer::TerraLumina) {
        let sites = data_state.terra_lumina_sites.read();
        for site in sites.iter() {
            let pos = geo::lat_lon_to_xyz(site.lat, site.lon, 1.005);
            markers.push(MarkerInstance {
                position: pos, color: Vec3::new(0.655, 0.545, 0.984),
                size: 0.032, marker_type: 3.0,
            });
            picks.push(PickableMarker {
                position: pos,
                info: HoverInfo::TerraLuminaSite(site.clone()),
                selected: SelectedItem::TerraLuminaSite(site.clone()),
            });
        }
    }

    if layers.contains(&Layer::Energy) {
        let sites = data_state.sites.read();
        for site in sites.iter() {
            let pos = geo::lat_lon_to_xyz(site.lat, site.lon, 1.005);
            let c = site.energy_type.rgb();
            let size = geo::marker_size_from_capacity(site.capacity_mw);
            markers.push(MarkerInstance {
                position: pos, color: Vec3::new(c[0], c[1], c[2]),
                size, marker_type: 0.0,
            });
            picks.push(PickableMarker {
                position: pos,
                info: HoverInfo::Site(site.clone()),
                selected: SelectedItem::Site(site.clone()),
            });
        }
    }

    // Earth region markers (large, Phi-colored)
    if layers.contains(&Layer::Regions) {
        let regions = data_state.earth_regions.read();
        for region in regions.iter() {
            let pos = geo::lat_lon_to_xyz(region.lat, region.lon, 1.005);
            // Color by Phi: low=blue, mid=cyan, high=gold
            let phi = region.phi_mean as f32;
            let color = Vec3::new(
                phi * 2.0,              // red increases with Phi
                0.4 + phi * 0.8,        // green grows
                1.0 - phi * 0.5,        // blue fades with Phi
            );
            // Size by population (logarithmic)
            let size = (region.population_m as f32).ln() * 0.005 + 0.015;
            markers.push(MarkerInstance {
                position: pos, color, size,
                marker_type: 0.0, // use default energy style for now
            });
            picks.push(PickableMarker {
                position: pos,
                info: HoverInfo::EarthRegion(region.clone()),
                selected: SelectedItem::EarthRegion(region.clone()),
            });
        }
    }

    if layers.contains(&Layer::Maglev) {
        let corridors = data_state.maglev_corridors.read();
        // Corridors appear progressively: first at year 25, all by year 200
        let max_corridors = match timeline_year {
            0..=24 => 0,
            25..=49 => 3,
            50..=99 => 6,
            100..=149 => 10,
            150..=199 => 13,
            _ => 15,
        };
        for (i, corridor) in corridors.iter().enumerate() {
            if i >= max_corridors { break; }
            let from = geo::lat_lon_to_xyz(corridor.from_lat, corridor.from_lon, 1.0);
            let to = geo::lat_lon_to_xyz(corridor.to_lat, corridor.to_lon, 1.0);
            let peak = geo::arc_peak_height(corridor.distance_km);
            let verts = geometry::generate_arc_with_progress(from, to, peak, 32);
            let color = if corridor.submarine {
                Vec3::new(0.0, 0.4, 0.8)
            } else {
                Vec3::new(0.0, 1.0, 0.533)
            };
            arc_datas.push((verts, color, i as f32 * 0.7));
        }
    }

    // Supply chain trade routes (amber arcs)
    if layers.contains(&Layer::SupplyChain) {
        let routes = data_state.supply_routes.read();
        for (i, route) in routes.iter().enumerate() {
            let from = geo::lat_lon_to_xyz(route.from_lat, route.from_lon, 1.0);
            let to = geo::lat_lon_to_xyz(route.to_lat, route.to_lon, 1.0);
            let peak = 0.06 + route.capacity as f32 * 0.03;
            let verts = geometry::generate_arc_with_progress(from, to, peak, 32);
            // Color by transport mode
            let color = match route.mode.as_str() {
                "Maritime" => Vec3::new(0.96, 0.62, 0.04),   // amber
                "Air" => Vec3::new(0.8, 0.8, 0.95),          // light blue-white
                "Land" => Vec3::new(0.6, 0.4, 0.1),          // brown
                _ => Vec3::new(0.96, 0.62, 0.04),
            };
            arc_datas.push((verts, color, (i as f32 + 20.0) * 0.5));
        }
    }

    // Climate projects (green markers)
    if layers.contains(&Layer::Climate) {
        let projects = data_state.climate_projects.read();
        for project in projects.iter() {
            let pos = geo::lat_lon_to_xyz(project.lat, project.lon, 1.005);
            // Color by type
            let color = match project.project_type.as_str() {
                "reforestation" | "conservation" => Vec3::new(0.06, 0.73, 0.51), // emerald
                "solar" | "wind" => Vec3::new(0.98, 0.84, 0.0),                 // gold
                "carbon_capture" => Vec3::new(0.0, 0.87, 1.0),                  // cyan
                "ocean" => Vec3::new(0.0, 0.5, 0.8),                            // ocean blue
                _ => Vec3::new(0.06, 0.73, 0.51),
            };
            let size = (project.co2_offset_mt as f32).sqrt() * 0.004 + 0.008;
            markers.push(MarkerInstance {
                position: pos, color, size, marker_type: 0.0,
            });
        }
    }

    // Emergency shelters (red markers)
    if layers.contains(&Layer::Emergency) {
        let shelters = data_state.emergency_shelters.read();
        for shelter in shelters.iter() {
            let pos = geo::lat_lon_to_xyz(shelter.lat, shelter.lon, 1.005);
            markers.push(MarkerInstance {
                position: pos,
                color: Vec3::new(0.94, 0.27, 0.27), // red
                size: 0.018,
                marker_type: 0.0,
            });
        }
    }

    // Health facilities (pink markers)
    if layers.contains(&Layer::Health) {
        let facilities = data_state.health_facilities.read();
        for facility in facilities.iter() {
            let pos = geo::lat_lon_to_xyz(facility.lat, facility.lon, 1.005);
            let size = if facility.beds > 1000 { 0.015 } else { 0.010 };
            markers.push(MarkerInstance {
                position: pos,
                color: Vec3::new(0.93, 0.28, 0.60), // pink
                size,
                marker_type: 0.0,
            });
        }
    }

    // Robotics dispatch (purple markers)
    if layers.contains(&Layer::Robotics) {
        let dispatch = data_state.robotics_dispatch.read();
        for unit in dispatch.iter() {
            let pos = geo::lat_lon_to_xyz(unit.lat, unit.lon, 1.005);
            markers.push(MarkerInstance {
                position: pos,
                color: Vec3::new(0.55, 0.36, 0.96), // purple
                size: 0.020,
                marker_type: 0.0,
            });
        }
    }

    if layers.contains(&Layer::FossilDeposits) {
        let deposits = data_state.fossil_deposits.read();
        for deposit in deposits.iter() {
            let pos = geo::lat_lon_to_xyz(deposit.lat, deposit.lon, 1.005);
            let eroi = terra_atlas_core::economics::compute_eroi(deposit).unwrap_or(5.0);
            let c = terra_atlas_core::economics::eroi_color(eroi);
            let emissive = terra_atlas_core::geo::fossil_emissive_factor(&deposit.status);
            let scale = terra_atlas_core::geo::fossil_scale_factor(&deposit.status);
            let size = terra_atlas_core::geo::marker_size_from_reserves(deposit.proven_reserves_mboe) * scale;
            markers.push(MarkerInstance {
                position: pos,
                color: Vec3::new(c[0] * emissive, c[1] * emissive, c[2] * emissive),
                size,
                marker_type: 0.0,
            });
            picks.push(PickableMarker {
                position: pos,
                info: HoverInfo::FossilDeposit(deposit.clone()),
                selected: SelectedItem::FossilDeposit(deposit.clone()),
            });
        }
    }

    if layers.contains(&Layer::Nuclear) {
        let sites = data_state.nuclear_sites.read();
        let nc = Layer::Nuclear.rgb();
        for site in sites.iter() {
            let pos = geo::lat_lon_to_xyz(site.lat, site.lon, 1.005);
            let size = terra_atlas_core::geo::marker_size_from_capacity(site.capacity_mw);
            let brightness = if site.reactor_type.is_smr() { 1.4_f32 } else { 1.0 };
            markers.push(MarkerInstance {
                position: pos,
                color: Vec3::new(nc[0] * brightness, nc[1] * brightness, nc[2] * brightness),
                size,
                marker_type: 0.0,
            });
            picks.push(PickableMarker {
                position: pos,
                info: HoverInfo::NuclearSite(site.clone()),
                selected: SelectedItem::NuclearSite(site.clone()),
            });
        }
    }

    // Build Φ hotspots from Terra Lumina sites for earth shader warping
    {
        let tl = data_state.terra_lumina_sites.read();
        let mut hotspots: Vec<[f32; 4]> = tl.iter().take(12).map(|s| {
            let pos = geo::lat_lon_to_xyz(s.lat, s.lon, 1.0);
            [pos.x, pos.y, pos.z, s.score as f32 / 100.0]
        }).collect();
        // Pad to 12 if needed
        while hotspots.len() < 12 { hotspots.push([0.0, 0.0, 0.0, 0.0]); }
        renderer.borrow_mut().set_phi_hotspots(hotspots);
    }

    log::info!("Updated: {} markers, {} pickable, {} arcs", markers.len(), picks.len(), arc_datas.len());

    *pickables.borrow_mut() = picks;
    let mut r = renderer.borrow_mut();
    r.update_markers(&markers);
    r.update_arcs(arc_datas);
}
