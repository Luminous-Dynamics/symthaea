# Mycelix Praxis Assets

This directory contains visual assets for the project.

## Logo Files

- **`logo.svg`** - Main logo for light backgrounds
- **`logo-dark.svg`** - Logo variant for dark backgrounds

### Logo Design

The logo combines two key elements:
1. **Neural Network**: Represents federated learning and AI
   - Blue nodes: Input layer (learners)
   - Green nodes: Hidden layer (aggregation)
   - Orange nodes: Output layer (knowledge)
2. **Graduation Cap**: Represents education and learning

### Colors

- Primary Blue: `#3B82F6` (Tailwind blue-500)
- Success Green: `#10B981` (Tailwind emerald-500)
- Accent Orange: `#F59E0B` (Tailwind amber-500)
- Neutral Gray: `#6B7280` (Tailwind gray-500)

## Favicon Generation

To generate favicons from the SVG logo, use one of these methods:

### Method 1: Online Tool (Easiest)
1. Go to https://realfavicongenerator.net/
2. Upload `logo.svg`
3. Customize settings
4. Download generated files
5. Place in `apps/web/public/`

### Method 2: ImageMagick (Command Line)
```bash
# Install ImageMagick
# Ubuntu/Debian: sudo apt-get install imagemagick
# macOS: brew install imagemagick

# Generate favicons
convert -background none assets/logo.svg -resize 16x16 apps/web/public/favicon-16x16.png
convert -background none assets/logo.svg -resize 32x32 apps/web/public/favicon-32x32.png
convert -background none assets/logo.svg -resize 180x180 apps/web/public/apple-touch-icon.png
convert -background none assets/logo.svg -resize 192x192 apps/web/public/android-chrome-192x192.png
convert -background none assets/logo.svg -resize 512x512 apps/web/public/android-chrome-512x512.png
```

### Method 3: Inkscape (GUI)
1. Open `logo.svg` in Inkscape
2. Export PNG at desired sizes:
   - 16x16 (favicon-16x16.png)
   - 32x32 (favicon-32x32.png)
   - 180x180 (apple-touch-icon.png)
   - 192x192 & 512x512 (for Android)
3. Save to `apps/web/public/`

## Usage

### In React App
```tsx
import logo from '../../../assets/logo.svg';

<img src={logo} alt="Praxis Logo" width={48} height={48} />
```

### In README
```markdown
![Praxis Logo](assets/logo.svg)
```

### As Favicon
Add to `apps/web/index.html`:
```html
<link rel="icon" type="image/png" sizes="32x32" href="/favicon-32x32.png">
<link rel="icon" type="image/png" sizes="16x16" href="/favicon-16x16.png">
<link rel="apple-touch-icon" sizes="180x180" href="/apple-touch-icon.png">
```

## Screenshots

Screenshots will be stored in `docs/assets/screenshots/` and used in the README.

**Planned screenshots**:
- `home.png` - Home page with feature cards
- `courses.png` - Course discovery page
- `fl-rounds.png` - FL rounds page with timeline
- `credentials.png` - Credentials page with W3C VCs

## Social Preview

The social preview image (`docs/assets/social-preview.png`) is displayed when sharing links on social media (Twitter, LinkedIn, Discord, etc.).

**Dimensions**: 1200x630 pixels
**Format**: PNG
**Content**: Logo, tagline, key features

---

*Last updated: 2025-11-15*
