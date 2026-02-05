# Dashboard

Web-based monitoring dashboard for real-time consciousness visualization.

## Features

- Real-time Φ monitoring
- Consciousness graph visualization
- Brain subsystem activity
- Memory system status
- Performance metrics

## Running

```bash
# Start the dashboard server
cargo run --bin symthaea-dashboard

# Or with the API server
cargo run --features dashboard
```

## Architecture

```
dashboard/
├── index.html      # Main page
├── style.css       # Styling
├── app.js          # Frontend logic
└── api/            # Backend endpoints
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/phi` | Current Φ value |
| `GET /api/graph` | Consciousness graph state |
| `GET /api/brain` | Brain subsystem status |
| `WS /api/stream` | Real-time updates |

## Development

```bash
# Serve with hot reload
cd dashboard
python -m http.server 8080
```
