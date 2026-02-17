//! Proof Explorer TUI
//!
//! Interactive terminal UI for exploring and debugging zkSTARK proofs.
//!
//! ## Usage
//!
//! ```bash
//! # Launch explorer with a proof file
//! proof-explorer proof.bin
//!
//! # Launch explorer with GPU status
//! proof-explorer --gpu-status
//! ```
//!
//! ## Navigation
//!
//! - Arrow keys: Navigate
//! - Tab: Switch panels
//! - Enter: Select/Expand
//! - q: Quit
//! - h: Help

use std::io;
use std::path::PathBuf;
use std::time::Duration;

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Tabs, Gauge, Table, Row, Cell},
    Frame, Terminal,
};

use fl_aggregator::proofs::{
    ProofType, SecurityLevel, gpu,
    integration::ProofEnvelope,
};

/// Application state
struct App {
    /// Currently selected tab
    current_tab: usize,
    /// Tab names
    tabs: Vec<&'static str>,
    /// Loaded proof (if any)
    proof: Option<LoadedProof>,
    /// GPU status
    gpu_status: Option<GpuInfo>,
    /// Current list selection
    list_state: usize,
    /// Show help overlay
    show_help: bool,
    /// Status message
    status: String,
}

/// Loaded proof information
struct LoadedProof {
    path: PathBuf,
    envelope: ProofEnvelope,
    raw_size: usize,
}

/// GPU information
struct GpuInfo {
    available: bool,
    device_name: Option<String>,
    backend: Option<String>,
    compute_units: u32,
}

impl App {
    fn new() -> Self {
        let gpu_status = check_gpu();

        Self {
            current_tab: 0,
            tabs: vec!["Overview", "Structure", "Bytes", "GPU", "Help"],
            proof: None,
            gpu_status: Some(gpu_status),
            list_state: 0,
            show_help: false,
            status: "Ready. Press 'h' for help.".to_string(),
        }
    }

    fn load_proof(&mut self, path: PathBuf) -> io::Result<()> {
        let bytes = std::fs::read(&path)?;
        let raw_size = bytes.len();

        match ProofEnvelope::from_bytes(&bytes) {
            Ok(envelope) => {
                self.proof = Some(LoadedProof {
                    path,
                    envelope,
                    raw_size,
                });
                self.status = "Proof loaded successfully.".to_string();
            }
            Err(e) => {
                self.status = format!("Failed to load proof: {}", e);
            }
        }
        Ok(())
    }

    fn next_tab(&mut self) {
        self.current_tab = (self.current_tab + 1) % self.tabs.len();
    }

    fn previous_tab(&mut self) {
        if self.current_tab > 0 {
            self.current_tab -= 1;
        } else {
            self.current_tab = self.tabs.len() - 1;
        }
    }

    fn next_item(&mut self) {
        self.list_state = self.list_state.saturating_add(1);
    }

    fn previous_item(&mut self) {
        self.list_state = self.list_state.saturating_sub(1);
    }
}

fn check_gpu() -> GpuInfo {
    let status = gpu::check_gpu_availability();

    GpuInfo {
        available: status.available,
        device_name: status.device.as_ref().map(|d| d.name.clone()),
        backend: status.device.as_ref().map(|d| d.backend.clone()),
        compute_units: status.device.map(|d| d.compute_units).unwrap_or(0),
    }
}

fn main() -> Result<(), io::Error> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app and load proof if provided
    let mut app = App::new();
    if args.len() > 1 {
        let path = PathBuf::from(&args[1]);
        let _ = app.load_proof(path);
    }

    // Run main loop
    let res = run_app(&mut terminal, &mut app);

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        eprintln!("Error: {}", err);
    }

    Ok(())
}

fn run_app<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    app: &mut App,
) -> io::Result<()> {
    loop {
        terminal.draw(|f| ui(f, app))?;

        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') => return Ok(()),
                        KeyCode::Char('h') => app.show_help = !app.show_help,
                        KeyCode::Tab => app.next_tab(),
                        KeyCode::BackTab => app.previous_tab(),
                        KeyCode::Right => app.next_tab(),
                        KeyCode::Left => app.previous_tab(),
                        KeyCode::Down | KeyCode::Char('j') => app.next_item(),
                        KeyCode::Up | KeyCode::Char('k') => app.previous_item(),
                        _ => {}
                    }
                }
            }
        }
    }
}

fn ui(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Tabs
            Constraint::Min(0),     // Content
            Constraint::Length(3),  // Status bar
        ])
        .split(f.area());

    // Draw tabs
    draw_tabs(f, app, chunks[0]);

    // Draw content based on selected tab
    match app.current_tab {
        0 => draw_overview(f, app, chunks[1]),
        1 => draw_structure(f, app, chunks[1]),
        2 => draw_bytes(f, app, chunks[1]),
        3 => draw_gpu(f, app, chunks[1]),
        4 => draw_help_tab(f, chunks[1]),
        _ => {}
    }

    // Draw status bar
    draw_status(f, app, chunks[2]);

    // Draw help overlay if active
    if app.show_help {
        draw_help_overlay(f);
    }
}

fn draw_tabs(f: &mut Frame, app: &App, area: Rect) {
    let titles: Vec<Line> = app
        .tabs
        .iter()
        .map(|t| Line::from(Span::styled(*t, Style::default().fg(Color::White))))
        .collect();

    let tabs = Tabs::new(titles)
        .block(Block::default().borders(Borders::ALL).title("Proof Explorer"))
        .select(app.current_tab)
        .style(Style::default().fg(Color::White))
        .highlight_style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        );

    f.render_widget(tabs, area);
}

fn draw_overview(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    // Proof info
    let proof_info = if let Some(ref proof) = app.proof {
        let items = vec![
            format!("File: {}", proof.path.display()),
            format!("Size: {} bytes", proof.raw_size),
            format!("Type: {:?}", proof.envelope.proof_type),
            format!("Security: {:?}", proof.envelope.security_level),
            format!("Compressed: {}", proof.envelope.compressed),
            format!("Version: {}", proof.envelope.version),
            format!("Proof bytes: {}", proof.envelope.proof_bytes.len()),
            format!("Public inputs: {} bytes", proof.envelope.public_inputs.len()),
        ];
        items
    } else {
        vec![
            "No proof loaded.".to_string(),
            "".to_string(),
            "Usage: proof-explorer <file>".to_string(),
        ]
    };

    let list_items: Vec<ListItem> = proof_info
        .iter()
        .map(|s| ListItem::new(Line::from(s.as_str())))
        .collect();

    let list = List::new(list_items)
        .block(Block::default().borders(Borders::ALL).title("Proof Information"));

    f.render_widget(list, chunks[0]);

    // Security gauge
    let security_pct = if let Some(ref proof) = app.proof {
        match proof.envelope.security_level {
            SecurityLevel::Standard96 => 37,  // 96/256
            SecurityLevel::Standard128 => 50, // 128/256
            SecurityLevel::High256 => 100,
        }
    } else {
        0
    };

    let gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title("Security Level"))
        .gauge_style(Style::default().fg(Color::Green))
        .percent(security_pct as u16)
        .label(format!("{} bits", security_pct * 256 / 100));

    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(5), Constraint::Min(0)])
        .split(chunks[1]);

    f.render_widget(gauge, right_chunks[0]);

    // Proof type info
    let type_info = if let Some(ref proof) = app.proof {
        match proof.envelope.proof_type {
            ProofType::Range => vec![
                "Range Proof",
                "Proves value in [min, max]",
                "AIR: 3 columns, 64 rows",
            ],
            ProofType::GradientIntegrity => vec![
                "Gradient Integrity Proof",
                "Proves valid FL gradient",
                "AIR: 5 columns, N rows",
            ],
            ProofType::IdentityAssurance => vec![
                "Identity Assurance Proof",
                "Proves assurance level met",
                "AIR: 8 columns, 9 rows",
            ],
            ProofType::VoteEligibility => vec![
                "Vote Eligibility Proof",
                "Proves voter qualification",
                "AIR: 10 columns, 8 rows",
            ],
            ProofType::Membership => vec![
                "Membership Proof",
                "Proves Merkle tree inclusion",
                "AIR: 8 columns, depth rows",
            ],
        }
    } else {
        vec!["No proof loaded"]
    };

    let type_items: Vec<ListItem> = type_info
        .iter()
        .map(|s| ListItem::new(Line::from(*s)))
        .collect();

    let type_list = List::new(type_items)
        .block(Block::default().borders(Borders::ALL).title("Proof Type Details"));

    f.render_widget(type_list, right_chunks[1]);
}

fn draw_structure(f: &mut Frame, app: &App, area: Rect) {
    let structure_info = if let Some(ref proof) = app.proof {
        vec![
            ("Magic Bytes", format!("{:02X} {:02X} {:02X} {:02X}",
                0x50, 0x52, 0x4F, 0x46)), // "PROF"
            ("Version", format!("{}", proof.envelope.version)),
            ("Proof Type", format!("{:?}", proof.envelope.proof_type)),
            ("Security", format!("{:?}", proof.envelope.security_level)),
            ("Compressed", format!("{}", proof.envelope.compressed)),
            ("Proof Data", format!("{} bytes", proof.envelope.proof_bytes.len())),
            ("Public Inputs", format!("{} bytes", proof.envelope.public_inputs.len())),
            ("Checksum", format!("{:02X}{:02X}{:02X}{:02X}",
                proof.envelope.checksum[0],
                proof.envelope.checksum[1],
                proof.envelope.checksum[2],
                proof.envelope.checksum[3])),
        ]
    } else {
        vec![("Status", "No proof loaded".to_string())]
    };

    let rows: Vec<Row> = structure_info
        .iter()
        .map(|(field, value)| {
            Row::new(vec![
                Cell::from(Span::styled(*field, Style::default().fg(Color::Cyan))),
                Cell::from(value.as_str()),
            ])
        })
        .collect();

    let table = Table::new(
        rows,
        [Constraint::Length(15), Constraint::Min(40)],
    )
    .block(Block::default().borders(Borders::ALL).title("Proof Structure"))
    .header(Row::new(vec!["Field", "Value"]).style(Style::default().fg(Color::Yellow)));

    f.render_widget(table, area);
}

fn draw_bytes(f: &mut Frame, app: &App, area: Rect) {
    let hex_content = if let Some(ref proof) = app.proof {
        let bytes = &proof.envelope.proof_bytes;
        let start = app.list_state * 16;
        let end = (start + 256).min(bytes.len());

        let mut lines: Vec<Line> = Vec::new();

        for (i, chunk) in bytes[start..end].chunks(16).enumerate() {
            let offset = start + i * 16;
            let hex: String = chunk.iter().map(|b| format!("{:02X} ", b)).collect();
            let ascii: String = chunk
                .iter()
                .map(|&b| if b >= 32 && b < 127 { b as char } else { '.' })
                .collect();

            lines.push(Line::from(vec![
                Span::styled(format!("{:08X}  ", offset), Style::default().fg(Color::DarkGray)),
                Span::styled(format!("{:<48}", hex), Style::default().fg(Color::White)),
                Span::styled(ascii, Style::default().fg(Color::Green)),
            ]));
        }

        if lines.is_empty() {
            lines.push(Line::from("No proof data"));
        }

        lines
    } else {
        vec![Line::from("No proof loaded")]
    };

    let paragraph = Paragraph::new(hex_content)
        .block(Block::default().borders(Borders::ALL).title("Hex View (use j/k to scroll)"));

    f.render_widget(paragraph, area);
}

fn draw_gpu(f: &mut Frame, app: &App, area: Rect) {
    let gpu_info = if let Some(ref gpu) = app.gpu_status {
        vec![
            format!("Available: {}", if gpu.available { "Yes" } else { "No" }),
            format!("Device: {}", gpu.device_name.as_deref().unwrap_or("N/A")),
            format!("Backend: {}", gpu.backend.as_deref().unwrap_or("N/A")),
            format!("Compute Units: {}", gpu.compute_units),
            "".to_string(),
            "Expected Speedups:".to_string(),
            format!("  NTT: ~{}x", gpu::EXPECTED_NTT_SPEEDUP),
            format!("  Merkle: ~{}x", gpu::EXPECTED_MERKLE_SPEEDUP),
            format!("  Polynomial: ~{}x", gpu::EXPECTED_POLYNOMIAL_SPEEDUP),
            "".to_string(),
            format!("Roadmap Phase: {}", gpu::roadmap::CURRENT_PHASE),
        ]
    } else {
        vec!["GPU status unavailable".to_string()]
    };

    let items: Vec<ListItem> = gpu_info
        .iter()
        .map(|s| ListItem::new(Line::from(s.as_str())))
        .collect();

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title("GPU Acceleration Status"));

    f.render_widget(list, area);
}

fn draw_help_tab(f: &mut Frame, area: Rect) {
    let help_text = vec![
        "Proof Explorer - Interactive proof debugging TUI",
        "",
        "NAVIGATION:",
        "  Tab / Right   Next tab",
        "  Shift+Tab / Left   Previous tab",
        "  j / Down      Scroll down",
        "  k / Up        Scroll up",
        "",
        "ACTIONS:",
        "  h             Toggle help",
        "  q             Quit",
        "",
        "USAGE:",
        "  proof-explorer <proof-file>",
        "",
        "The explorer shows:",
        "  - Proof overview and metadata",
        "  - Internal structure details",
        "  - Raw hex bytes view",
        "  - GPU acceleration status",
    ];

    let items: Vec<ListItem> = help_text
        .iter()
        .map(|s| ListItem::new(Line::from(*s)))
        .collect();

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title("Help"));

    f.render_widget(list, area);
}

fn draw_status(f: &mut Frame, app: &App, area: Rect) {
    let status = Paragraph::new(Line::from(vec![
        Span::styled(" Status: ", Style::default().fg(Color::Cyan)),
        Span::raw(&app.status),
        Span::styled("  |  ", Style::default().fg(Color::DarkGray)),
        Span::styled("q", Style::default().fg(Color::Yellow)),
        Span::raw(":quit "),
        Span::styled("h", Style::default().fg(Color::Yellow)),
        Span::raw(":help "),
        Span::styled("Tab", Style::default().fg(Color::Yellow)),
        Span::raw(":switch"),
    ]))
    .block(Block::default().borders(Borders::ALL));

    f.render_widget(status, area);
}

fn draw_help_overlay(f: &mut Frame) {
    let area = centered_rect(60, 50, f.area());

    let help_text = vec![
        "",
        "  KEYBOARD SHORTCUTS",
        "",
        "  q        Quit application",
        "  h        Toggle this help",
        "  Tab      Next tab",
        "  Arrows   Navigate",
        "  j/k      Scroll list",
        "",
        "  Press any key to close",
        "",
    ];

    let items: Vec<ListItem> = help_text
        .iter()
        .map(|s| ListItem::new(Line::from(*s)))
        .collect();

    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Help")
                .style(Style::default().bg(Color::DarkGray)),
        )
        .style(Style::default().bg(Color::DarkGray));

    f.render_widget(ratatui::widgets::Clear, area);
    f.render_widget(list, area);
}

/// Helper function to create a centered rect
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}
