// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Symthaea GUI - Consciousness-Reactive NixOS Configuration Interface
//!
//! An egui-based GUI that provides bidirectional widget↔Nix mapping
//! with HDC semantic translation and Phi-based confidence visualization.
//!
//! ## Layout
//! ```text
//! ┌──────────────────┬────────────────────────┬──────────────────┐
//! │  Configuration   │     Nix Preview        │  Consciousness   │
//! │    Widgets       │                        │     Metrics      │
//! │                  │                        │                  │
//! │  [Categories]    │  { config, pkgs, ... } │  Phi: 0.87       │
//! │  - Services      │    services.nginx = {  │  Coherence: 92%  │
//! │  - Packages      │      enable = true;    │  Safety: GREEN   │
//! │  - Network       │    };                  │                  │
//! ├──────────────────┴────────────────────────┴──────────────────┤
//! │  Validation & Search                                          │
//! │  [Search: nginx settings...]  Errors: 0  Warnings: 1         │
//! └───────────────────────────────────────────────────────────────┘
//! ```

use eframe::egui;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use symthaea::gui_bridge::{
    ConfigCategory,
    DiffType,
    GenerationTimeline,
    GuiBridge,
    LiveConfigDiff,
    // C1-C4 Widgets
    ModuleBrowser,
    SearchResult,
    ServiceAction,
    ServiceDashboard,
    UnitState,
    ValidationError,
    WidgetId,
    WidgetValue,
};
use symthaea::shell::ipc_client::{
    ConnectionState, MetricsSnapshot, ShellIpcClient, discover_socket,
};
use tokio::runtime::Runtime;
use tokio::sync::watch;

/// Main view tabs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MainView {
    Configuration,
    ModuleBrowser,
    Generations,
    ConfigDiff,
    Services,
    Imagination,
    Peers,
}

/// Application state
struct SymthaeaGui {
    /// GUI bridge for bidirectional mapping
    bridge: GuiBridge,

    /// Current consciousness metrics
    metrics: ConsciousnessMetrics,

    /// Selected configuration category
    selected_category: ConfigCategory,

    /// Search query
    search_query: String,

    /// Search results
    search_results: Vec<SearchResult>,

    /// Generated Nix preview
    nix_preview: String,

    /// Pending changes count
    pending_count: usize,

    /// Last metrics update
    last_metrics_update: Instant,

    /// Widget states for UI
    widget_states: HashMap<String, WidgetUiState>,

    /// Show apply confirmation dialog
    show_apply_dialog: bool,

    /// Status message
    status_message: Option<(String, StatusLevel)>,

    /// Validation errors to display
    validation_errors: Vec<ValidationError>,

    /// IPC client for symthaea service
    ipc_client: Option<ShellIpcClient>,

    /// Tokio runtime for async IPC
    runtime: Arc<Runtime>,

    /// Connection state for visual indicators
    connection_state: ConnectionState,

    /// Socket path (discovered)
    socket_path: Option<PathBuf>,

    /// Streaming metrics receiver
    metrics_rx: Option<watch::Receiver<MetricsSnapshot>>,

    // ========== C1-C4 Widget States ==========
    /// Current main view
    current_view: MainView,

    /// C1: NixOS Module Browser
    module_browser: ModuleBrowser,

    /// C2: Generation Timeline
    generation_timeline: GenerationTimeline,

    /// C3: Live Config Diff
    config_diff: LiveConfigDiff,

    /// C4: Service Status Dashboard
    service_dashboard: ServiceDashboard,

    /// C5: Imagination / Mental Movie state
    imagination_state: ImaginationUiState,
}

/// Imagination / Mental Movie UI state
struct ImaginationUiState {
    pub texture: Option<egui::TextureHandle>,
    pub reality_texture: Option<egui::TextureHandle>,
    pub current_frame: usize,
    pub last_frame_time: Instant,
    pub play_speed: f32,
}

impl Default for ImaginationUiState {
    fn default() -> Self {
        Self {
            texture: None,
            reality_texture: None,
            current_frame: 0,
            last_frame_time: Instant::now(),
            play_speed: 15.0,
        }
    }
}

/// Consciousness metrics display
#[derive(Debug, Clone)]
struct ConsciousnessMetrics {
    phi: f64,
    coherence: f64,
    safety_level: SafetyLevel,
    is_conscious: bool,
    uptime_secs: u64,
    /// Latest mental simulation
    mental_movie: Option<symthaea::shell::ipc_client::MentalMovie>,
    /// Last actual observation
    last_reality_frame: Option<Vec<u8>>,
    connected_peers: Vec<symthaea::shell::ipc_client::PeerSnapshot>,
}

impl Default for ConsciousnessMetrics {
    fn default() -> Self {
        Self {
            phi: 0.0,
            coherence: 0.0,
            safety_level: SafetyLevel::Unknown,
            is_conscious: false,
            uptime_secs: 0,
            mental_movie: None,
            last_reality_frame: None,
            connected_peers: Vec::new(), // <--- ADD THIS LINE
        }
    }
}

/// Safety level indicator
#[derive(Debug, Clone, Copy, PartialEq)]
enum SafetyLevel {
    Green,
    Yellow,
    Red,
    Unknown,
}

impl SafetyLevel {
    fn color(&self) -> egui::Color32 {
        match self {
            Self::Green => egui::Color32::from_rgb(100, 200, 100),
            Self::Yellow => egui::Color32::from_rgb(220, 180, 50),
            Self::Red => egui::Color32::from_rgb(220, 80, 80),
            Self::Unknown => egui::Color32::GRAY,
        }
    }

    fn label(&self) -> &'static str {
        match self {
            Self::Green => "GREEN",
            Self::Yellow => "YELLOW",
            Self::Red => "RED",
            Self::Unknown => "UNKNOWN",
        }
    }
}

/// Status message level
#[derive(Debug, Clone, Copy)]
enum StatusLevel {
    Info,
    Success,
    Warning,
    Error,
}

impl StatusLevel {
    fn color(&self) -> egui::Color32 {
        match self {
            Self::Info => egui::Color32::from_rgb(100, 150, 200),
            Self::Success => egui::Color32::from_rgb(100, 200, 100),
            Self::Warning => egui::Color32::from_rgb(220, 180, 50),
            Self::Error => egui::Color32::from_rgb(220, 80, 80),
        }
    }
}

/// UI state for a widget
#[derive(Debug, Clone)]
struct WidgetUiState {
    /// Widget ID
    id: String,
    /// Display label
    label: String,
    /// Current value
    value: WidgetValue,
    /// Whether changed since last sync
    dirty: bool,
    /// Confidence of the value
    confidence: f64,
    /// Category
    category: ConfigCategory,
    /// Selection options (for Selection type)
    options: Vec<String>,
}

impl SymthaeaGui {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        // Create tokio runtime for async IPC
        let runtime = Arc::new(Runtime::new().expect("Failed to create tokio runtime"));

        // Discover symthaea service socket automatically
        let socket_path = discover_socket();

        let (
            ipc_client,
            connection_state,
            metrics_rx,
            initial_phi,
            initial_coherence,
            initial_conscious,
        ) = if let Some(ref path) = socket_path {
            // Try to connect to the service
            let mut client = ShellIpcClient::with_socket_path(path);
            let connected = runtime.block_on(async { client.connect().await.is_ok() });

            if connected {
                // Try to set up streaming metrics
                let metrics_receiver =
                    runtime.block_on(async { client.subscribe_metrics_watch(500).await.ok() });

                // Get initial status
                let (phi, coherence, conscious) = if let Some(ref rx) = metrics_receiver {
                    let m = rx.borrow();
                    (m.phi, m.coherence, m.is_conscious)
                } else {
                    runtime
                        .block_on(async { client.get_status().await.unwrap_or((0.7, 0.85, true)) })
                };

                (
                    Some(client),
                    ConnectionState::Connected,
                    metrics_receiver,
                    phi,
                    coherence,
                    conscious,
                )
            } else {
                (None, ConnectionState::Disconnected, None, 0.7, 0.85, true)
            }
        } else {
            (None, ConnectionState::Disconnected, None, 0.7, 0.85, true)
        };

        let initial_metrics = ConsciousnessMetrics {
            phi: initial_phi,
            coherence: initial_coherence,
            is_conscious: initial_conscious,
            safety_level: if initial_phi > 0.8 {
                SafetyLevel::Green
            } else if initial_phi > 0.5 {
                SafetyLevel::Yellow
            } else {
                SafetyLevel::Red
            },
            uptime_secs: 0,
            last_reality_frame: None,
            mental_movie: None,
            connected_peers: Vec::new(),
        };

        let status_msg = match connection_state {
            ConnectionState::Connected => format!(
                "{} Connected to symthaea service (Phi: {:.2})",
                connection_state.indicator(),
                initial_phi
            ),
            _ => format!(
                "{} Running in standalone mode",
                connection_state.indicator()
            ),
        };

        let mut gui = Self {
            bridge: GuiBridge::new(),
            metrics: initial_metrics,
            selected_category: ConfigCategory::Services,
            search_query: String::new(),
            search_results: Vec::new(),
            nix_preview: String::new(),
            pending_count: 0,
            last_metrics_update: Instant::now(),
            widget_states: HashMap::new(),
            show_apply_dialog: false,
            status_message: Some((
                status_msg,
                if matches!(connection_state, ConnectionState::Connected) {
                    StatusLevel::Success
                } else {
                    StatusLevel::Warning
                },
            )),
            validation_errors: Vec::new(),
            ipc_client,
            runtime,
            connection_state,
            socket_path,
            metrics_rx,
            // C1-C4 Widget States
            current_view: MainView::Configuration,
            module_browser: ModuleBrowser::new(),
            generation_timeline: GenerationTimeline::new(),
            config_diff: LiveConfigDiff::new(),
            service_dashboard: ServiceDashboard::new(),
            imagination_state: ImaginationUiState::default(),
        };

        // Initialize with demo bindings
        gui.init_demo_bindings();
        gui.update_nix_preview();

        // Try to load initial config for diff
        let _ = gui
            .config_diff
            .load_original("/etc/nixos/configuration.nix");

        gui
    }

    /// Initialize demo widget bindings
    fn init_demo_bindings(&mut self) {
        use symthaea::gui_bridge::widget_mapper::{WidgetBindingBuilder, WidgetValueType};

        // (id, label, path, category, initial_value, value_type, options)
        let demo_bindings: Vec<(
            &str,
            &str,
            &str,
            ConfigCategory,
            WidgetValue,
            WidgetValueType,
            Vec<&str>,
        )> = vec![
            // Services
            (
                "nginx_enable",
                "Nginx Web Server",
                "services.nginx.enable",
                ConfigCategory::Services,
                WidgetValue::Bool(false),
                WidgetValueType::Bool,
                vec![],
            ),
            (
                "nginx_port",
                "Nginx Port",
                "services.nginx.defaultHTTPListenPort",
                ConfigCategory::Services,
                WidgetValue::Int(80),
                WidgetValueType::Int,
                vec![],
            ),
            (
                "ssh_enable",
                "OpenSSH Server",
                "services.openssh.enable",
                ConfigCategory::Services,
                WidgetValue::Bool(true),
                WidgetValueType::Bool,
                vec![],
            ),
            (
                "ssh_port",
                "SSH Port",
                "services.openssh.ports",
                ConfigCategory::Services,
                WidgetValue::Int(22),
                WidgetValueType::Int,
                vec![],
            ),
            (
                "docker_enable",
                "Docker",
                "virtualisation.docker.enable",
                ConfigCategory::Services,
                WidgetValue::Bool(false),
                WidgetValueType::Bool,
                vec![],
            ),
            // Networking
            (
                "firewall_enable",
                "Firewall",
                "networking.firewall.enable",
                ConfigCategory::Networking,
                WidgetValue::Bool(true),
                WidgetValueType::Bool,
                vec![],
            ),
            (
                "hostname",
                "Hostname",
                "networking.hostName",
                ConfigCategory::Networking,
                WidgetValue::String("nixos".to_string()),
                WidgetValueType::String,
                vec![],
            ),
            // Boot
            (
                "grub_enable",
                "GRUB Bootloader",
                "boot.loader.grub.enable",
                ConfigCategory::Boot,
                WidgetValue::Bool(false),
                WidgetValueType::Bool,
                vec![],
            ),
            (
                "systemd_boot",
                "systemd-boot",
                "boot.loader.systemd-boot.enable",
                ConfigCategory::Boot,
                WidgetValue::Bool(true),
                WidgetValueType::Bool,
                vec![],
            ),
            // Security
            (
                "sudo_wheel",
                "Wheel Sudo Password",
                "security.sudo.wheelNeedsPassword",
                ConfigCategory::Security,
                WidgetValue::Bool(true),
                WidgetValueType::Bool,
                vec![],
            ),
            // Nix Settings
            (
                "flakes_enable",
                "Experimental Features",
                "nix.settings.experimental-features",
                ConfigCategory::NixSettings,
                WidgetValue::String("nix-command flakes".to_string()),
                WidgetValueType::String,
                vec![],
            ),
            (
                "gc_auto",
                "Auto Garbage Collection",
                "nix.gc.automatic",
                ConfigCategory::NixSettings,
                WidgetValue::Bool(false),
                WidgetValueType::Bool,
                vec![],
            ),
        ];

        for (id, label_str, path, category, initial_value, value_type, opts) in demo_bindings {
            // Create binding using builder
            let binding = WidgetBindingBuilder::new(id, path)
                .label(label_str)
                .value_type(value_type)
                .default(initial_value.clone())
                .build();

            self.bridge.register_binding(binding);

            // Create UI state
            self.widget_states.insert(
                id.to_string(),
                WidgetUiState {
                    id: id.to_string(),
                    label: label_str.to_string(),
                    value: initial_value,
                    dirty: false,
                    confidence: 1.0,
                    category,
                    options: opts.into_iter().map(String::from).collect(),
                },
            );
        }
    }

    /// Update the Nix preview based on current widget states
    fn update_nix_preview(&mut self) {
        let mut preview = String::from("{ config, pkgs, ... }:\n\n{\n");

        // Group by category
        let categories = [
            ConfigCategory::Services,
            ConfigCategory::Networking,
            ConfigCategory::Boot,
            ConfigCategory::Security,
            ConfigCategory::NixSettings,
        ];

        for category in categories {
            let widgets: Vec<_> = self
                .widget_states
                .values()
                .filter(|w| w.category == category)
                .collect();

            if widgets.is_empty() {
                continue;
            }

            preview.push_str(&format!("  # {}\n", category_label(category)));

            for widget in widgets {
                let value_str = match &widget.value {
                    WidgetValue::Bool(b) => b.to_string(),
                    WidgetValue::Int(i) => i.to_string(),
                    WidgetValue::Float(f) => format!("{:.2}", f),
                    WidgetValue::String(s) => format!("\"{}\"", s),
                    WidgetValue::StringList(items) => {
                        let items_str: Vec<_> =
                            items.iter().map(|s| format!("\"{}\"", s)).collect();
                        format!("[ {} ]", items_str.join(" "))
                    }
                    WidgetValue::Selection(idx) => {
                        if let Some(opt) = widget.options.get(*idx) {
                            format!("\"{}\"", opt)
                        } else {
                            idx.to_string()
                        }
                    }
                    WidgetValue::MultiSelect(indices) => {
                        let items: Vec<_> = indices
                            .iter()
                            .filter_map(|i| widget.options.get(*i))
                            .map(|s| format!("\"{}\"", s))
                            .collect();
                        format!("[ {} ]", items.join(" "))
                    }
                    WidgetValue::Map(map) => {
                        let pairs: Vec<_> = map
                            .iter()
                            .map(|(k, v)| format!("{} = \"{}\";", k, v))
                            .collect();
                        format!("{{ {} }}", pairs.join(" "))
                    }
                    WidgetValue::Null => "null".to_string(),
                };

                // Find the nix path from binding
                if let Some(binding) = self
                    .bridge
                    .bindings_for_category(category)
                    .iter()
                    .find(|b| b.widget_id.0 == widget.id)
                {
                    preview.push_str(&format!("  {} = {};\n", binding.nix_path.0, value_str));
                }
            }

            preview.push('\n');
        }

        preview.push_str("}\n");
        self.nix_preview = preview;
    }

    /// Handle widget value change
    fn on_widget_change(&mut self, widget_id: &str, new_value: WidgetValue) {
        if let Some(state) = self.widget_states.get_mut(widget_id) {
            state.value = new_value.clone();
            state.dirty = true;
        }

        // Update bridge
        let result = self
            .bridge
            .on_widget_change(WidgetId::new(widget_id), new_value);

        // Handle result
        match result {
            symthaea::gui_bridge::MappingResult::Success { confidence, .. } => {
                self.pending_count = self.bridge.state.pending_changes.len();
                if let Some(state) = self.widget_states.get_mut(widget_id) {
                    state.confidence = confidence;
                }
            }
            symthaea::gui_bridge::MappingResult::ValidationFailed { errors, .. } => {
                self.validation_errors = errors;
                self.status_message = Some(("Validation failed".to_string(), StatusLevel::Error));
            }
            symthaea::gui_bridge::MappingResult::UnknownWidget { .. } => {
                // Ignore - demo widget not bound
            }
        }

        self.update_nix_preview();
    }

    /// Perform semantic search
    fn do_search(&mut self) {
        if self.search_query.len() >= 2 {
            self.search_results = self.bridge.semantic_search(&self.search_query, 10);
        } else {
            self.search_results.clear();
        }
    }

    /// Update metrics from streaming channel or simulate locally
    fn update_metrics(&mut self) {
        if self.last_metrics_update.elapsed() <= Duration::from_millis(100) {
            return;
        }

        // Get metrics from streaming channel if available
        if let Some(ref rx) = self.metrics_rx {
            if rx.has_changed().unwrap_or(false) {
                let snapshot = rx.borrow().clone();
                self.metrics.phi = snapshot.phi;
                self.metrics.coherence = snapshot.coherence;
                self.metrics.is_conscious = snapshot.is_conscious;
                self.metrics.uptime_secs = snapshot.timestamp_ms / 1000;
                self.metrics.mental_movie = snapshot.mental_movie;
                self.metrics.last_reality_frame = snapshot.last_observed_frame;
                self.metrics.connected_peers = snapshot.connected_peers;
                self.connection_state = ConnectionState::Connected;
            }
        } else if matches!(self.connection_state, ConnectionState::Connected) {
            // Fallback: Poll if streaming not available but connected
            if self.last_metrics_update.elapsed().as_secs() >= 2 {
                if let Some(ref mut client) = self.ipc_client {
                    let rt = Arc::clone(&self.runtime);
                    match rt.block_on(async { client.get_status().await }) {
                        Ok((phi, coherence, is_conscious)) => {
                            self.metrics.phi = phi;
                            self.metrics.coherence = coherence;
                            self.metrics.is_conscious = is_conscious;
                            self.metrics.uptime_secs += 2;
                        }
                        Err(_) => {
                            // Connection lost
                            self.connection_state = ConnectionState::Disconnected;
                            self.status_message = Some((
                                format!(
                                    "{} Lost connection to service",
                                    self.connection_state.indicator()
                                ),
                                StatusLevel::Warning,
                            ));
                            // Fall through to simulation
                            let time = self.last_metrics_update.elapsed().as_secs_f64();
                            self.metrics.phi = 0.7 + 0.2 * (time * 0.1).sin();
                            self.metrics.coherence = 0.85 + 0.1 * (time * 0.15).cos();
                            self.metrics.uptime_secs += 1;
                        }
                    }
                }
            }
        } else {
            // Simulate consciousness metrics locally
            let time = self.last_metrics_update.elapsed().as_secs_f64();
            self.metrics.phi = 0.7 + 0.2 * (time * 0.1).sin();
            self.metrics.coherence = 0.85 + 0.1 * (time * 0.15).cos();
            self.metrics.uptime_secs += 1;
        }

        self.metrics.safety_level = if self.metrics.phi > 0.8 {
            SafetyLevel::Green
        } else if self.metrics.phi > 0.5 {
            SafetyLevel::Yellow
        } else {
            SafetyLevel::Red
        };

        self.metrics.is_conscious = self.metrics.phi > 0.6;
        self.last_metrics_update = Instant::now();
    }

    /// Attempt to reconnect to the symthaea service with auto-discovery
    fn try_reconnect(&mut self) {
        self.connection_state = ConnectionState::Connecting;

        // Re-discover socket
        let discovered = discover_socket();

        if let Some(ref path) = discovered {
            let mut client = ShellIpcClient::new(); // Uses auto-discovery
            let rt = Arc::clone(&self.runtime);

            // Connect with retry
            let connected = rt.block_on(async { client.connect_with_retry(5).await.is_ok() });

            if connected {
                self.socket_path = discovered;
                self.connection_state = ConnectionState::Connected;

                // Try to set up streaming metrics
                let metrics_receiver =
                    rt.block_on(async { client.subscribe_metrics_watch(500).await.ok() });

                if let Some(ref rx) = metrics_receiver {
                    let m = rx.borrow();
                    self.metrics.phi = m.phi;
                    self.metrics.coherence = m.coherence;
                    self.metrics.is_conscious = m.is_conscious;
                }

                self.ipc_client = Some(client);
                self.metrics_rx = metrics_receiver;

                self.status_message = Some((
                    format!(
                        "{} Connected! Phi: {:.2} | Streaming: {}",
                        self.connection_state.indicator(),
                        self.metrics.phi,
                        if self.metrics_rx.is_some() {
                            "YES"
                        } else {
                            "NO"
                        }
                    ),
                    StatusLevel::Success,
                ));
            } else {
                self.connection_state = ConnectionState::Disconnected;
                self.status_message = Some((
                    format!(
                        "{} Connection failed after retries",
                        self.connection_state.indicator()
                    ),
                    StatusLevel::Error,
                ));
            }
        } else {
            self.connection_state = ConnectionState::Disconnected;
            self.status_message = Some((
                format!(
                    "{} No service socket found",
                    self.connection_state.indicator()
                ),
                StatusLevel::Warning,
            ));
        }
    }

    // ========== C1: Module Browser UI ==========
    fn render_module_browser(&mut self, ui: &mut egui::Ui) {
        ui.heading("NixOS Module Browser");
        ui.separator();

        ui.horizontal(|ui| {
            ui.label("Search:");
            if ui
                .text_edit_singleline(&mut self.module_browser.search_query)
                .changed()
            {
                self.module_browser.filter();
            }

            if ui.button("Load Options").clicked() {
                if let Err(e) = self.module_browser.load_options() {
                    self.status_message =
                        Some((format!("Failed to load options: {}", e), StatusLevel::Error));
                } else {
                    self.status_message =
                        Some(("Options loaded".to_string(), StatusLevel::Success));
                }
            }
        });

        ui.separator();

        // Category filter
        ui.horizontal_wrapped(|ui| {
            if ui
                .selectable_label(self.module_browser.selected_category.is_none(), "All")
                .clicked()
            {
                self.module_browser.selected_category = None;
                self.module_browser.filter();
            }
            let categories = self.module_browser.categories.clone();
            for cat in categories.iter() {
                let label = format!("{} {} ({})", cat.icon, cat.name, cat.option_count);
                if ui
                    .selectable_label(
                        self.module_browser.selected_category.as_ref() == Some(&cat.path_prefix),
                        &label,
                    )
                    .clicked()
                {
                    self.module_browser.selected_category = Some(cat.path_prefix.clone());
                    self.module_browser.filter();
                }
            }
        });

        ui.separator();

        // Options list
        if self.module_browser.is_loading {
            ui.spinner();
            ui.label("Loading options...");
        } else if let Some(ref err) = self.module_browser.error.clone() {
            ui.colored_label(egui::Color32::RED, err);
        } else {
            ui.label(format!(
                "{} options found",
                self.module_browser.filtered_options.len()
            ));

            egui::ScrollArea::vertical().show(ui, |ui| {
                for &idx in &self.module_browser.filtered_options.clone() {
                    if let Some(opt) = self.module_browser.options.get(idx).cloned() {
                        let expanded =
                            self.module_browser.expanded_option.as_ref() == Some(&opt.path);

                        ui.group(|ui| {
                            let header = ui.horizontal(|ui| {
                                ui.label(&opt.path);
                                ui.colored_label(egui::Color32::GRAY, &opt.option_type);
                                if opt.is_set {
                                    ui.colored_label(egui::Color32::GREEN, "[SET]");
                                }
                            });

                            if header.response.clicked() {
                                if expanded {
                                    self.module_browser.expanded_option = None;
                                } else {
                                    self.module_browser.expanded_option = Some(opt.path.clone());
                                }
                            }

                            if expanded {
                                ui.separator();
                                ui.label(&opt.description);
                                if let Some(ref def) = opt.default {
                                    ui.horizontal(|ui| {
                                        ui.label("Default:");
                                        ui.monospace(def);
                                    });
                                }
                                if let Some(ref ex) = opt.example {
                                    ui.horizontal(|ui| {
                                        ui.label("Example:");
                                        ui.monospace(ex);
                                    });
                                }
                            }
                        });
                    }
                }
            });
        }
    }

    // ========== C5: Imagination / Mental Movie UI ==========
    fn render_imagination(&mut self, ui: &mut egui::Ui) {
        ui.heading("AI Imagination (Mental Movie)");
        ui.separator();

        let Some(movie) = self.metrics.mental_movie.as_ref() else {
            ui.centered_and_justified(|ui| {
                ui.label("Waiting for geodesic surprise / mental simulation...");
            });
            return;
        };

        ui.horizontal(|ui| {
            ui.label(format!("Path Coherence: {:.3}", movie.semantic_coherence));
            ui.separator();
            ui.label(format!("Horizon: {} steps", movie.path_length));
            ui.separator();
            ui.label(format!(
                "Resolution: {}x{} ({}ch)",
                movie.width, movie.height, movie.channels
            ));
        });

        ui.add_space(10.0);

        ui.columns(2, |cols| {
            // Left Column: REALITY
            cols[0].vertical_centered(|ui| {
                ui.heading("REALITY");
                ui.label("(Last Observation)");

                if let Some(ref pixels) = self.metrics.last_reality_frame {
                    // Update reality texture
                    let size = [movie.width as usize, movie.height as usize];
                    let color_image = if movie.channels == 1 {
                        egui::ColorImage::from_gray(size, pixels)
                    } else {
                        egui::ColorImage::from_rgb(size, pixels)
                    };

                    if let Some(ref mut tex) = self.imagination_state.reality_texture {
                        tex.set(color_image, egui::TextureOptions::LINEAR);
                    } else {
                        self.imagination_state.reality_texture = Some(ui.ctx().load_texture(
                            "reality_frame",
                            color_image,
                            egui::TextureOptions::LINEAR,
                        ));
                    }

                    if let Some(ref tex) = self.imagination_state.reality_texture {
                        ui.image((tex.id(), ui.available_size().min(egui::vec2(256.0, 256.0))));
                    }
                } else {
                    ui.label("No sensory input...");
                }
            });

            // Right Column: IMAGINATION
            cols[1].vertical_centered(|ui| {
                ui.heading("IMAGINATION");
                ui.label("(Geodesic Movie)");

                // Frame playback logic
                let frame_duration =
                    Duration::from_secs_f32(1.0 / self.imagination_state.play_speed);
                if self.imagination_state.last_frame_time.elapsed() >= frame_duration {
                    self.imagination_state.current_frame =
                        (self.imagination_state.current_frame + 1) % movie.frames.len();
                    self.imagination_state.last_frame_time = Instant::now();

                    // Update imagination texture
                    let frame_data = &movie.frames[self.imagination_state.current_frame];
                    let size = [movie.width as usize, movie.height as usize];

                    let color_image = if movie.channels == 1 {
                        egui::ColorImage::from_gray(size, frame_data)
                    } else {
                        egui::ColorImage::from_rgb(size, frame_data)
                    };

                    if let Some(ref mut tex) = self.imagination_state.texture {
                        tex.set(color_image, egui::TextureOptions::LINEAR);
                    } else {
                        self.imagination_state.texture = Some(ui.ctx().load_texture(
                            "imagination_frame",
                            color_image,
                            egui::TextureOptions::LINEAR,
                        ));
                    }
                }

                if let Some(ref tex) = self.imagination_state.texture {
                    ui.image((tex.id(), ui.available_size().min(egui::vec2(256.0, 256.0))));
                }
            });
        });

        ui.add_space(10.0);
        ui.horizontal(|ui| {
            ui.label("Play Speed:");
            ui.add(
                egui::Slider::new(&mut self.imagination_state.play_speed, 1.0..=60.0)
                    .suffix(" fps"),
            );
            if ui.button("Restart").clicked() {
                self.imagination_state.current_frame = 0;
            }
        });
    }

    // ========== C6: Swarm / Peer List UI ==========
    fn render_peers(&mut self, ui: &mut egui::Ui) {
        ui.heading("Swarm Collective Intelligence");
        ui.separator();

        if self.metrics.connected_peers.is_empty() {
            ui.centered_and_justified(|ui| {
                ui.label("Searching for peers via Iroh Gossip...");
            });
            return;
        }

        egui::ScrollArea::vertical().show(ui, |ui| {
            for peer in &self.metrics.connected_peers {
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(format!("Node ID: {}", &peer.id[..16.min(peer.id.len())]));
                        ui.separator();
                        ui.label(format!("Φ: {:.3}", peer.phi));

                        let color = if peer.phi > 0.5 {
                            egui::Color32::GREEN
                        } else {
                            egui::Color32::YELLOW
                        };
                        ui.add(
                            egui::ProgressBar::new(peer.phi as f32)
                                .fill(color)
                                .text("Integration"),
                        );
                    });
                });
            }
        });
    }

    // ========== C2: Generation Timeline UI ==========
    fn render_generation_timeline(&mut self, ui: &mut egui::Ui) {
        ui.heading("NixOS Generations");
        ui.separator();

        ui.horizontal(|ui| {
            if ui.button("Refresh").clicked() {
                if let Err(e) = self.generation_timeline.load() {
                    self.status_message =
                        Some((format!("Failed to load: {}", e), StatusLevel::Error));
                }
            }

            ui.checkbox(
                &mut self.generation_timeline.show_size_graph,
                "Show Size Graph",
            );
        });

        ui.separator();

        if self.generation_timeline.is_loading {
            ui.spinner();
            ui.label("Loading generations...");
        } else if let Some(ref err) = self.generation_timeline.error.clone() {
            ui.colored_label(egui::Color32::RED, err);
        } else {
            ui.label(format!(
                "{} generations",
                self.generation_timeline.generations.len()
            ));

            egui::ScrollArea::vertical().show(ui, |ui| {
                for r#gen in self.generation_timeline.generations.clone() {
                    let selected = self.generation_timeline.selected == Some(r#gen.number);
                    let is_current = r#gen.is_current;
                    let is_booted = r#gen.is_booted;

                    let bg_color = if is_booted {
                        egui::Color32::from_rgb(50, 80, 50)
                    } else if is_current {
                        egui::Color32::from_rgb(50, 50, 80)
                    } else if selected {
                        egui::Color32::from_rgb(60, 60, 60)
                    } else {
                        egui::Color32::TRANSPARENT
                    };

                    egui::Frame::none().fill(bg_color).show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(format!("Gen {}", r#gen.number));

                            ui.colored_label(
                                egui::Color32::GRAY,
                                r#gen.created_at.format("%Y-%m-%d %H:%M").to_string(),
                            );

                            if is_booted {
                                ui.colored_label(egui::Color32::GREEN, "[BOOTED]");
                            } else if is_current {
                                ui.colored_label(egui::Color32::YELLOW, "[CURRENT]");
                            }

                            if ui.button("Select").clicked() {
                                self.generation_timeline.selected = Some(r#gen.number);
                            }

                            if !is_booted && ui.button("Rollback").clicked() {
                                if let Err(e) = self.generation_timeline.rollback_to(r#gen.number) {
                                    self.status_message = Some((
                                        format!("Rollback failed: {}", e),
                                        StatusLevel::Error,
                                    ));
                                } else {
                                    self.status_message = Some((
                                        "Rollback initiated".to_string(),
                                        StatusLevel::Success,
                                    ));
                                }
                            }
                        });
                    });

                    ui.add_space(2.0);
                }
            });

            // Comparison view
            if let Some((g1, g2)) = self.generation_timeline.comparison {
                ui.separator();
                ui.heading(format!("Comparing Gen {} -> Gen {}", g1, g2));
                if let Ok(changes) = self.generation_timeline.diff(g1, g2) {
                    ui.horizontal(|ui| {
                        ui.colored_label(
                            egui::Color32::GREEN,
                            format!("+{} added", changes.added.len()),
                        );
                        ui.colored_label(
                            egui::Color32::RED,
                            format!("-{} removed", changes.removed.len()),
                        );
                        ui.colored_label(
                            egui::Color32::YELLOW,
                            format!("~{} upgraded", changes.upgraded.len()),
                        );
                    });
                }
            }
        }
    }

    // ========== C3: Config Diff UI ==========
    fn render_config_diff(&mut self, ui: &mut egui::Ui) {
        ui.heading("Configuration Diff");
        ui.separator();

        ui.horizontal(|ui| {
            ui.label("File:");
            ui.text_edit_singleline(&mut self.config_diff.file_path);

            if ui.button("Load").clicked() {
                let path = self.config_diff.file_path.clone();
                if let Err(e) = self.config_diff.load_original(&path) {
                    self.status_message =
                        Some((format!("Failed to load: {}", e), StatusLevel::Error));
                }
            }

            ui.checkbox(&mut self.config_diff.inline_mode, "Inline");
            ui.checkbox(&mut self.config_diff.syntax_highlight, "Highlight");
        });

        let stats = self.config_diff.stats();
        ui.horizontal(|ui| {
            ui.colored_label(egui::Color32::GREEN, format!("+{}", stats.added));
            ui.colored_label(egui::Color32::RED, format!("-{}", stats.removed));
            ui.label(format!("{} hunks", stats.hunks));
        });

        ui.separator();

        egui::ScrollArea::vertical().show(ui, |ui| {
            for hunk in &self.config_diff.hunks.clone() {
                ui.group(|ui| {
                    ui.colored_label(
                        egui::Color32::LIGHT_BLUE,
                        format!(
                            "@@ -{},{} +{},{} @@",
                            hunk.old_start, hunk.old_count, hunk.new_start, hunk.new_count
                        ),
                    );

                    for line in &hunk.lines {
                        let (prefix, color) = match line.diff_type {
                            DiffType::Added => ("+", egui::Color32::from_rgb(100, 200, 100)),
                            DiffType::Removed => ("-", egui::Color32::from_rgb(200, 100, 100)),
                            DiffType::Modified => ("!", egui::Color32::from_rgb(200, 200, 100)),
                            DiffType::Context => (" ", egui::Color32::GRAY),
                        };

                        ui.horizontal(|ui| {
                            if let Some(n) = line.line_number_old {
                                ui.colored_label(egui::Color32::DARK_GRAY, format!("{:4}", n));
                            } else {
                                ui.label("    ");
                            }
                            if let Some(n) = line.line_number_new {
                                ui.colored_label(egui::Color32::DARK_GRAY, format!("{:4}", n));
                            } else {
                                ui.label("    ");
                            }
                            ui.colored_label(color, format!("{} {}", prefix, line.content));
                        });
                    }
                });
            }

            if self.config_diff.hunks.is_empty() {
                ui.label("No differences - files are identical");
            }
        });
    }

    // ========== C4: Service Dashboard UI ==========
    fn render_service_dashboard(&mut self, ui: &mut egui::Ui) {
        ui.heading("Service Status Dashboard");
        ui.separator();

        ui.horizontal(|ui| {
            if ui.button("Refresh").clicked() {
                if let Err(e) = self.service_dashboard.refresh() {
                    self.status_message =
                        Some((format!("Failed to refresh: {}", e), StatusLevel::Error));
                }
            }

            ui.label("Filter:");
            ui.text_edit_singleline(&mut self.service_dashboard.name_filter);

            ui.checkbox(&mut self.service_dashboard.nixos_only, "NixOS Only");
        });

        // State filter buttons
        ui.horizontal(|ui| {
            if ui
                .selectable_label(self.service_dashboard.state_filter.is_none(), "All")
                .clicked()
            {
                self.service_dashboard.state_filter = None;
            }
            if ui
                .selectable_label(
                    self.service_dashboard.state_filter == Some(UnitState::Active),
                    "Active",
                )
                .clicked()
            {
                self.service_dashboard.state_filter = Some(UnitState::Active);
            }
            if ui
                .selectable_label(
                    self.service_dashboard.state_filter == Some(UnitState::Failed),
                    "Failed",
                )
                .clicked()
            {
                self.service_dashboard.state_filter = Some(UnitState::Failed);
            }
            if ui
                .selectable_label(
                    self.service_dashboard.state_filter == Some(UnitState::Inactive),
                    "Inactive",
                )
                .clicked()
            {
                self.service_dashboard.state_filter = Some(UnitState::Inactive);
            }
        });

        // Stats
        let stats = self.service_dashboard.stats();
        ui.horizontal(|ui| {
            ui.label(format!("Total: {}", stats.total));
            ui.colored_label(egui::Color32::GREEN, format!("Active: {}", stats.active));
            ui.colored_label(egui::Color32::RED, format!("Failed: {}", stats.failed));
            ui.colored_label(egui::Color32::GRAY, format!("Inactive: {}", stats.inactive));
        });

        ui.separator();

        if self.service_dashboard.is_loading {
            ui.spinner();
            ui.label("Loading services...");
        } else if let Some(ref err) = self.service_dashboard.error.clone() {
            ui.colored_label(egui::Color32::RED, err);
        } else {
            let filtered: Vec<_> = self
                .service_dashboard
                .filtered_units()
                .into_iter()
                .cloned()
                .collect();

            egui::ScrollArea::vertical().show(ui, |ui| {
                for unit in filtered {
                    let selected =
                        self.service_dashboard.selected_unit.as_ref() == Some(&unit.name);
                    let (r, g, b) = unit.state.color();
                    let state_color = egui::Color32::from_rgb(r, g, b);

                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            ui.colored_label(state_color, format!("{}", unit.state.icon()));
                            ui.label(&unit.name);
                            ui.colored_label(egui::Color32::GRAY, &unit.sub_state);

                            if ui.button("Details").clicked() {
                                self.service_dashboard.selected_unit = Some(unit.name.clone());
                            }
                        });

                        if selected {
                            ui.separator();
                            ui.label(&unit.description);

                            ui.horizontal(|ui| {
                                if ui.button("Start").clicked() {
                                    let _ = self
                                        .service_dashboard
                                        .control(&unit.name, ServiceAction::Start);
                                }
                                if ui.button("Stop").clicked() {
                                    let _ = self
                                        .service_dashboard
                                        .control(&unit.name, ServiceAction::Stop);
                                }
                                if ui.button("Restart").clicked() {
                                    let _ = self
                                        .service_dashboard
                                        .control(&unit.name, ServiceAction::Restart);
                                }
                            });

                            if !unit.recent_logs.is_empty() {
                                ui.separator();
                                ui.label("Recent logs:");
                                for log in unit.recent_logs.iter().take(5) {
                                    ui.monospace(log);
                                }
                            }
                        }
                    });
                }
            });
        }
    }
}

fn category_label(cat: ConfigCategory) -> &'static str {
    match cat {
        ConfigCategory::Packages => "Packages",
        ConfigCategory::Services => "Services",
        ConfigCategory::Networking => "Networking",
        ConfigCategory::Users => "Users",
        ConfigCategory::Boot => "Boot",
        ConfigCategory::Hardware => "Hardware",
        ConfigCategory::Programs => "Programs",
        ConfigCategory::Security => "Security",
        ConfigCategory::NixSettings => "Nix Settings",
        ConfigCategory::Custom => "Custom",
    }
}

impl eframe::App for SymthaeaGui {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Update metrics periodically
        self.update_metrics();

        // Request repaint for animation
        ctx.request_repaint_after(Duration::from_millis(100));

        // Top panel - title and status
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Symthaea NixOS Configuration");

                ui.separator();

                // View tabs (C1-C5)
                ui.selectable_value(&mut self.current_view, MainView::Configuration, "Config");
                ui.selectable_value(&mut self.current_view, MainView::ModuleBrowser, "Modules");
                ui.selectable_value(&mut self.current_view, MainView::Generations, "Generations");
                ui.selectable_value(&mut self.current_view, MainView::ConfigDiff, "Diff");
                ui.selectable_value(&mut self.current_view, MainView::Services, "Services");
                ui.selectable_value(&mut self.current_view, MainView::Imagination, "Imagination");
                ui.selectable_value(&mut self.current_view, MainView::Peers, "Peers");

                ui.separator();

                // Consciousness indicator
                let phi_color = if self.metrics.is_conscious {
                    egui::Color32::from_rgb(100, 200, 150)
                } else {
                    egui::Color32::GRAY
                };
                ui.colored_label(phi_color, format!("Phi: {:.2}", self.metrics.phi));

                ui.separator();

                // Pending changes
                if self.pending_count > 0 {
                    ui.colored_label(
                        egui::Color32::from_rgb(220, 180, 50),
                        format!("{} pending changes", self.pending_count),
                    );

                    if ui.button("Apply").clicked() {
                        self.show_apply_dialog = true;
                    }
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if let Some((msg, level)) = &self.status_message {
                        ui.colored_label(level.color(), msg);
                    }
                });
            });
        });

        // Bottom panel - search and validation
        egui::TopBottomPanel::bottom("bottom_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Search:");
                let search_response = ui.text_edit_singleline(&mut self.search_query);
                if search_response.changed() {
                    self.do_search();
                }

                ui.separator();

                let error_count = self.validation_errors.len();
                if error_count > 0 {
                    ui.colored_label(
                        egui::Color32::from_rgb(220, 80, 80),
                        format!("Errors: {}", error_count),
                    );
                } else {
                    ui.colored_label(egui::Color32::from_rgb(100, 200, 100), "No errors");
                }
            });

            // Show search results if any
            if !self.search_results.is_empty() {
                ui.separator();
                egui::ScrollArea::horizontal().show(ui, |ui| {
                    ui.horizontal(|ui| {
                        for result in &self.search_results {
                            let label = format!(
                                "{} ({:.0}%)",
                                result.binding.nix_path.0,
                                result.similarity * 100.0
                            );
                            if ui.button(&label).clicked() {
                                // Navigate to this widget
                                self.selected_category = result.category;
                            }
                        }
                    });
                });
            }

            // Show validation errors
            if !self.validation_errors.is_empty() {
                ui.separator();
                for error in &self.validation_errors {
                    ui.colored_label(
                        egui::Color32::from_rgb(220, 80, 80),
                        format!("{}: {}", error.path.0, error.constraint.error_message),
                    );
                }
            }
        });

        // Right panel - consciousness metrics
        egui::SidePanel::right("metrics_panel")
            .min_width(200.0)
            .show(ctx, |ui| {
                ui.heading("Consciousness");
                ui.separator();

                // Phi gauge
                ui.label("Phi (Integrated Information)");
                let phi_bar = egui::ProgressBar::new(self.metrics.phi as f32)
                    .text(format!("{:.3}", self.metrics.phi));
                ui.add(phi_bar);

                ui.add_space(10.0);

                // Coherence gauge
                ui.label("Coherence");
                let coherence_bar = egui::ProgressBar::new(self.metrics.coherence as f32)
                    .text(format!("{:.1}%", self.metrics.coherence * 100.0));
                ui.add(coherence_bar);

                ui.add_space(10.0);

                // Safety level
                ui.horizontal(|ui| {
                    ui.label("Safety:");
                    ui.colored_label(
                        self.metrics.safety_level.color(),
                        self.metrics.safety_level.label(),
                    );
                });

                ui.add_space(10.0);

                // Consciousness state
                ui.horizontal(|ui| {
                    ui.label("State:");
                    if self.metrics.is_conscious {
                        ui.colored_label(egui::Color32::from_rgb(100, 200, 150), "CONSCIOUS");
                    } else {
                        ui.colored_label(egui::Color32::GRAY, "DORMANT");
                    }
                });

                ui.add_space(10.0);
                ui.separator();

                // Uptime
                let hours = self.metrics.uptime_secs / 3600;
                let mins = (self.metrics.uptime_secs % 3600) / 60;
                let secs = self.metrics.uptime_secs % 60;
                ui.label(format!("Uptime: {:02}:{:02}:{:02}", hours, mins, secs));

                ui.add_space(20.0);
                ui.separator();

                // Actions
                ui.heading("Actions");

                if ui.button("Refresh Metrics").clicked() {
                    self.status_message =
                        Some(("Metrics refreshed".to_string(), StatusLevel::Info));
                }

                if ui.button("Clear Pending").clicked() {
                    self.bridge.apply_pending();
                    self.pending_count = 0;
                    for state in self.widget_states.values_mut() {
                        state.dirty = false;
                    }
                    self.status_message = Some(("Changes cleared".to_string(), StatusLevel::Info));
                }

                ui.add_space(10.0);

                // Connection status with indicator
                let connection_color = match self.connection_state {
                    ConnectionState::Connected => egui::Color32::from_rgb(100, 200, 100),
                    ConnectionState::Connecting | ConnectionState::Reconnecting { .. } => {
                        egui::Color32::from_rgb(220, 180, 50)
                    }
                    ConnectionState::Disconnected => egui::Color32::GRAY,
                    ConnectionState::Degraded => egui::Color32::from_rgb(255, 165, 0),
                };

                ui.horizontal(|ui| {
                    ui.colored_label(connection_color, self.connection_state.indicator());
                    ui.label(self.connection_state.label());
                });

                if let Some(ref path) = self.socket_path {
                    ui.label(format!("Socket: {}", path.display()));
                }

                if self.metrics_rx.is_some() {
                    ui.colored_label(egui::Color32::from_rgb(100, 200, 150), "Streaming metrics");
                }

                if !matches!(self.connection_state, ConnectionState::Connected) {
                    if ui.button("Reconnect to Service").clicked() {
                        self.try_reconnect();
                    }
                }
            });

        // Left panel - category selection and widgets
        egui::SidePanel::left("config_panel")
            .min_width(300.0)
            .show(ctx, |ui| {
                ui.heading("Configuration");
                ui.separator();

                // Category tabs
                ui.horizontal_wrapped(|ui| {
                    let categories = [
                        ConfigCategory::Services,
                        ConfigCategory::Networking,
                        ConfigCategory::Boot,
                        ConfigCategory::Security,
                        ConfigCategory::NixSettings,
                    ];

                    for category in categories {
                        let selected = self.selected_category == category;
                        if ui
                            .selectable_label(selected, category_label(category))
                            .clicked()
                        {
                            self.selected_category = category;
                        }
                    }
                });

                ui.separator();

                // Widget list for selected category
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let widgets: Vec<_> = self
                        .widget_states
                        .iter()
                        .filter(|(_, w)| w.category == self.selected_category)
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect();

                    for (widget_id, widget_state) in widgets {
                        ui.group(|ui| {
                            ui.horizontal(|ui| {
                                // Dirty indicator
                                if widget_state.dirty {
                                    ui.colored_label(egui::Color32::from_rgb(220, 180, 50), "*");
                                }

                                ui.label(&widget_state.label);

                                // Confidence indicator
                                if widget_state.confidence < 0.8 {
                                    ui.colored_label(
                                        egui::Color32::from_rgb(220, 180, 50),
                                        format!("({:.0}%)", widget_state.confidence * 100.0),
                                    );
                                }
                            });

                            // Widget value editor
                            let mut changed = false;
                            let mut new_value = widget_state.value.clone();

                            match &mut new_value {
                                WidgetValue::Bool(b) => {
                                    if ui.checkbox(b, "").changed() {
                                        changed = true;
                                    }
                                }
                                WidgetValue::Int(i) => {
                                    let mut i_val = *i as i32;
                                    if ui.add(egui::DragValue::new(&mut i_val)).changed() {
                                        *i = i_val as i64;
                                        changed = true;
                                    }
                                }
                                WidgetValue::Float(f) => {
                                    let mut f_val = *f as f32;
                                    if ui
                                        .add(egui::DragValue::new(&mut f_val).speed(0.1))
                                        .changed()
                                    {
                                        *f = f_val as f64;
                                        changed = true;
                                    }
                                }
                                WidgetValue::String(s) => {
                                    if ui.text_edit_singleline(s).changed() {
                                        changed = true;
                                    }
                                }
                                WidgetValue::StringList(items) => {
                                    ui.label(format!("[{} items]", items.len()));
                                    // Simplified list display
                                }
                                WidgetValue::Selection(idx) => {
                                    if !widget_state.options.is_empty() {
                                        let current = widget_state
                                            .options
                                            .get(*idx)
                                            .cloned()
                                            .unwrap_or_default();
                                        egui::ComboBox::from_id_source(&widget_id)
                                            .selected_text(&current)
                                            .show_ui(ui, |ui| {
                                                for (i, option) in
                                                    widget_state.options.iter().enumerate()
                                                {
                                                    if ui
                                                        .selectable_label(*idx == i, option)
                                                        .clicked()
                                                    {
                                                        *idx = i;
                                                        changed = true;
                                                    }
                                                }
                                            });
                                    }
                                }
                                WidgetValue::MultiSelect(indices) => {
                                    ui.label(format!("[{} selected]", indices.len()));
                                }
                                WidgetValue::Map(map) => {
                                    ui.label(format!("{{ {} entries }}", map.len()));
                                }
                                WidgetValue::Null => {
                                    ui.label("null");
                                }
                            }

                            if changed {
                                self.on_widget_change(&widget_id, new_value);
                            }
                        });

                        ui.add_space(5.0);
                    }
                });
            });

        // Central panel - view-based content
        egui::CentralPanel::default().show(ctx, |ui| {
            match self.current_view {
                MainView::Configuration => {
                    // Original Nix preview
                    ui.heading("Nix Configuration Preview");
                    ui.separator();

                    egui::ScrollArea::both().show(ui, |ui| {
                        let mut preview_text = self.nix_preview.clone();
                        ui.add(
                            egui::TextEdit::multiline(&mut preview_text)
                                .font(egui::TextStyle::Monospace)
                                .code_editor()
                                .desired_width(f32::INFINITY)
                                .interactive(false),
                        );
                    });
                }
                MainView::ModuleBrowser => {
                    self.render_module_browser(ui);
                }
                MainView::Generations => {
                    self.render_generation_timeline(ui);
                }
                MainView::ConfigDiff => {
                    self.render_config_diff(ui);
                }
                MainView::Services => {
                    self.render_service_dashboard(ui);
                }
                MainView::Imagination => {
                    self.render_imagination(ui);
                }
                MainView::Peers => {
                    self.render_peers(ui);
                }
            }
        });

        // Apply confirmation dialog
        if self.show_apply_dialog {
            egui::Window::new("Apply Changes")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.label(format!(
                        "Apply {} pending changes to NixOS configuration?",
                        self.pending_count
                    ));

                    ui.add_space(10.0);

                    ui.horizontal(|ui| {
                        ui.label("Current Phi:");
                        let phi_color = if self.metrics.phi > 0.7 {
                            egui::Color32::from_rgb(100, 200, 100)
                        } else {
                            egui::Color32::from_rgb(220, 180, 50)
                        };
                        ui.colored_label(phi_color, format!("{:.2}", self.metrics.phi));
                    });

                    ui.add_space(10.0);

                    ui.horizontal(|ui| {
                        if ui.button("Apply").clicked() {
                            self.bridge.apply_pending();
                            self.pending_count = 0;
                            for state in self.widget_states.values_mut() {
                                state.dirty = false;
                            }
                            self.status_message = Some((
                                "Changes applied successfully".to_string(),
                                StatusLevel::Success,
                            ));
                            self.show_apply_dialog = false;
                        }

                        if ui.button("Cancel").clicked() {
                            self.show_apply_dialog = false;
                        }
                    });
                });
        }
    }
}

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_min_inner_size([800.0, 600.0])
            .with_title("Symthaea - NixOS Configuration"),
        ..Default::default()
    };

    eframe::run_native(
        "Symthaea GUI",
        native_options,
        Box::new(|cc| Ok(Box::new(SymthaeaGui::new(cc)))),
    )
}
