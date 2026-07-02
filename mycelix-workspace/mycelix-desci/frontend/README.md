# Mycelix-DeSci Frontend

Web interface for querying and managing DeSci claims in the Mycelix-DeSci network.

## Features

- Browse and search verifiable research claims
- Filter by epistemic tier (E0-E4)
- Category-based organization (genomics, longevity, climate, etc.)
- Modern, responsive UI built with Svelte

## Development

### Prerequisites

- Node.js 20+
- npm or yarn

### Installation

```bash
cd frontend
npm install
```

### Running

```bash
# Development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

The app will be available at `http://localhost:5173`

## Project Structure

```
frontend/
├── src/
│   ├── App.svelte       # Main application component
│   ├── main.ts          # Application entry point
│   └── app.css          # Global styles
├── public/              # Static assets
├── index.html           # HTML template
└── vite.config.ts       # Vite configuration
```

## Future Enhancements

- Integration with Mycelix-DeSci API
- Claim submission interface
- IP-NFT minting UI
- Federated learning dashboard
- Real-time updates via WebSocket

## License

MIT
