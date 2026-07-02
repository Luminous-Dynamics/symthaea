# Mycelix Knowledge - Browser Extension

Fact-check any text on the web using the Mycelix decentralized knowledge graph.

## Features

- **Right-click Fact-Check**: Select text → Right-click → "Fact Check with Mycelix"
- **Popup Interface**: Quick access to search and recent fact-checks
- **Inline Highlighting**: See credibility scores directly on the page
- **Decentralized**: Queries go through Holochain, not a central server

## Installation

### Chrome Web Store
[Install from Chrome Web Store](https://chrome.google.com/webstore/detail/mycelix-knowledge/...)

### Firefox Add-ons
[Install from Firefox Add-ons](https://addons.mozilla.org/firefox/addon/mycelix-knowledge/)

### Manual Installation (Development)

#### Chrome
1. Go to `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked"
4. Select the `browser-extension` folder

#### Firefox
1. Go to `about:debugging#/runtime/this-firefox`
2. Click "Load Temporary Add-on"
3. Select `manifest.json` from the `browser-extension` folder

## Usage

### Fact-Check Text
1. Select any text on a webpage
2. Right-click and choose "Fact Check with Mycelix"
3. View the verdict popup with credibility score

### Search Knowledge Graph
1. Click the extension icon in toolbar
2. Enter your search query
3. Browse matching claims

### View Claim Details
- Click on any fact-check result to see:
  - Full E-N-M classification
  - Supporting and contradicting claims
  - Source information
  - Credibility breakdown

## Configuration

Click the extension icon → Settings (gear icon):

| Setting | Description |
|---------|-------------|
| Holochain URL | WebSocket URL for conductor |
| App ID | Holochain app identifier |
| Auto-highlight | Highlight fact-checked text |
| Credibility threshold | Minimum score to show |
| Theme | Light/Dark/System |

## Building for Production

### Prerequisites
- Node.js 18+
- npm or yarn

### Build Commands

```bash
# Install dependencies
npm install

# Build for Chrome
npm run build:chrome

# Build for Firefox
npm run build:firefox

# Build both
npm run build
```

### Output
- `dist/chrome/` - Chrome Web Store package
- `dist/firefox/` - Firefox Add-ons package

## Store Submission

### Chrome Web Store

1. Create a developer account at [Chrome Web Store Developer Dashboard](https://chrome.google.com/webstore/devconsole/)
2. Pay one-time $5 registration fee
3. Create new item
4. Upload `dist/chrome.zip`
5. Fill in store listing (see `store/chrome/` for assets)
6. Submit for review

### Firefox Add-ons

1. Create account at [Firefox Add-on Developer Hub](https://addons.mozilla.org/developers/)
2. Submit new add-on
3. Upload `dist/firefox.zip`
4. Fill in listing details (see `store/firefox/` for assets)
5. Submit for review

## Privacy Policy

This extension:
- Does NOT collect personal data
- Does NOT track browsing history
- Sends selected text to Holochain network for fact-checking
- Stores preferences locally in browser storage

See [PRIVACY.md](PRIVACY.md) for full privacy policy.

## Permissions Explained

| Permission | Why Needed |
|------------|------------|
| `activeTab` | Access current page to read selected text |
| `contextMenus` | Add "Fact Check" to right-click menu |
| `storage` | Save user preferences locally |
| `<all_urls>` | Inject content script for highlighting |

## Development

### Project Structure

```
browser-extension/
├── manifest.json          # Chrome manifest (MV3)
├── manifest.firefox.json  # Firefox-specific manifest
├── background.js          # Service worker
├── content.js            # Page injection script
├── content.css           # Highlight styles
├── popup.html            # Popup UI
├── popup.js              # Popup logic
├── options.html          # Settings page
├── icons/                # Extension icons
└── store/                # Store listing assets
```

### Local Development

1. Make changes to source files
2. Reload extension in browser
3. Test on various websites

### Testing

```bash
npm test
```

## Troubleshooting

### "Cannot connect to Holochain"
- Ensure Holochain conductor is running
- Check WebSocket URL in settings
- Verify Mycelix Knowledge hApp is installed

### Fact-check not working
- Check browser console for errors
- Ensure text is selected
- Try refreshing the page

### Highlights not showing
- Enable "Auto-highlight" in settings
- Check if site has strict CSP (may block our styles)

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

MIT - See [LICENSE](../LICENSE) for details.

---

**Questions?** [Open an issue](https://github.com/Luminous-Dynamics/mycelix-knowledge/issues) or join our [Discord](https://discord.gg/mycelix).
