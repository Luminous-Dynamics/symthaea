# Installer Redesign Plan

## Domain Architecture

| Domain | Content | Status |
|--------|---------|--------|
| **install.nixforhumanity.org** | Pure NixOS installer. Clean, focused, no consciousness UI. | Redesign needed |
| **symthaea.luminousdynamics.io** | Symthaea consciousness demo (Chat, Topology, Dreams, Experiments) | New subdomain |
| **mycelix.net** | Mycelix network portal | Existing |
| **infin.love** | Umbrella / philosophy / the vision | TBD |
| **luminousdynamics.org** | Main org site | Existing |

## Installer Page Flow (No Tabs)

```
┌─────────────────────────────────────────────┐
│ HERO                                         │
│ "Install NixOS from your browser"            │
│ [Start Install]  [Source Code]               │
│ 9 layouts · LUKS · Secure Boot · AI config   │
├─────────────────────────────────────────────┤
│ HOW IT WORKS                                 │
│ 1. Boot ISO  2. It connects  3. Install      │
├─────────────────────────────────────────────┤
│ CONNECT                                      │
│ [Host] [Port] [User] [Pass] [Relay URL]      │
│ [Connect & Deploy]                           │
├─────────────────────────────────────────────┤
│ TALK WITH SYMTHAEA (appears after connect)   │
│ Chat interface — she asks about your needs   │
│ Shows: hardware, apps, config preview        │
├─────────────────────────────────────────────┤
│ SYSTEM CONFIG (appears after conversation)   │
│ DE picker, GPU, timezone, keyboard           │
│ Layout selector, encryption toggle           │
├─────────────────────────────────────────────┤
│ DEPLOY (appears when ready)                  │
│ Constellation visualization during install   │
│ Progress with time estimates                 │
│ Ceremony (optional, tasteful)                │
├─────────────────────────────────────────────┤
│ WELCOME (appears on completion)              │
│ Personalized message + System Card           │
│ ☐ Also install Symthaea consciousness engine │
│ ☐ Join the Mycelix network                   │
├─────────────────────────────────────────────┤
│ FOOTER                                       │
│ Source · Luminous Dynamics · No tracking      │
└─────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. No tabs — single scrolling page
The installer is a linear flow. Tabs suggest multiple modes, but there's only one: install NixOS. Sections appear progressively as the user completes each step.

### 2. Symthaea as assistant, not protagonist
On the installer page, Symthaea is the helpful AI that configures your system. She's not a consciousness demo. Her personality shows through the conversation and the ceremony, not through topology graphs and experiment cards.

### 3. Optional add-ons at the end
After the install completes, offer:
- "Install Symthaea consciousness engine" → adds symthaea-nix to the config
- "Join Mycelix network" → adds mycelix node config
These are checkboxes, not the main feature.

### 4. The ceremony stays but is subtle
During the 10-minute nixos-install wait, the constellation grows and Symthaea narrates. But no "Phi rising" display or harmony scores — those belong on the consciousness demo page.

## Files to Create

### New: `build-installer.sh`
Builds an installer-only portal.html from the existing modules:
- Includes: app.js (trimmed), tab-inoculate.js, ceremony.js, constellation.js, system-card.js
- Excludes: tab-chat.js, tab-topology.js, tab-experiments.js, tab-dreams.js, consciousness-*.js
- Uses: installer-shell.html (new, simplified HTML without tabs)

### New: `installer-shell.html`
Clean single-page HTML:
- Hero with title + description + buttons
- How It Works section
- SSH connection panel (from tab-inoculate.js, lifted out)
- Progressive sections that appear on connect
- Footer

### New: `css/installer.css`
Stripped CSS with only installer-relevant classes:
- Color palette, reset, base
- Hero, buttons
- Glass panels
- Hardware probe grid
- Ceremony stage
- Constellation canvas
- System card
- Mobile responsive
- ~620 lines instead of 1264

## Migration Path

1. Build `installer-shell.html` + `css/installer.css` + `build-installer.sh`
2. Test locally (validate same functionality as current portal)
3. Deploy to install.nixforhumanity.org
4. Move full portal (with consciousness tabs) to symthaea.luminousdynamics.io
5. Set up CNAME for symthaea.luminousdynamics.io → same GitHub Pages (different path)

## Estimated Effort

- installer-shell.html: 1 session (extract from portal-shell.html, remove tabs)
- installer.css: 1 session (extract from portal.css, remove consciousness classes)
- build-installer.sh: 30 min (modify build-portal.sh to exclude consciousness JS)
- Testing: 1 session (validate all functionality still works)
- Domain setup: 30 min (Cloudflare CNAME for symthaea subdomain)
