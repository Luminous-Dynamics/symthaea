# Mycelix Desktop Icons

This directory contains application icons for different platforms.

## Required Icons

For a production build, you'll need:

- `32x32.png` - Windows small icon
- `128x128.png` - macOS and Linux icon
- `128x128@2x.png` - macOS retina icon
- `icon.icns` - macOS bundle icon
- `icon.ico` - Windows bundle icon

## Temporary Development Icons

For development, Tauri will use default icons if these are missing. The app will still compile and run.

## Generating Icons

You can generate all required formats from a single 1024x1024 PNG using:

```bash
cargo tauri icon path/to/source-icon.png
```

This will automatically create all required sizes and formats.

## Icon Design Guidelines

The Mycelix icon should represent:
- 🍄 Mycelial network (interconnected mushrooms)
- 🌐 P2P connectivity
- 💜 Purple/cyan color scheme (matching app theme)
- Clean, modern design suitable for both light and dark themes

## TODO

Create proper Mycelix branding icons before v0.1.0 release.