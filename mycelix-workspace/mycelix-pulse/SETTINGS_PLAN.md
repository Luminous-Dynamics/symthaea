# Mycelix Pulse — Comprehensive Settings Plan

## Current State

11 settings sections exist but most are shallow toggles. Key gap: 7 user preferences reset on page reload because they're only in-memory RwSignals, not persisted to localStorage.

### What's Persisted (8 items)
Theme, density, accent color, theme name, custom CSS, palette shortcut, welcome dismissed, inbox data, external accounts, offline queue

### What Resets on Reload (7 items)
Reading pane position, swipe actions (left/right), vacation responder, high contrast mode, sound enabled, keyboard shortcuts (all hardcoded)

---

## Settings Architecture

### Single `UserPreferences` struct — persisted as one JSON blob

```rust
struct UserPreferences {
    // Appearance
    theme: Theme,
    density: Density,
    accent_color: String,
    theme_name: String,
    custom_css: String,
    font_family: String,           // NEW
    font_scale: u32,               // NEW (percentage 75-150)
    high_contrast: bool,

    // Layout
    reading_pane: ReadingPanePosition,
    sidebar_collapsed: bool,       // NEW

    // Compose
    default_pqc: bool,             // NEW
    default_signature_id: Option<String>,  // NEW
    reply_style: ReplyStyle,       // NEW (InlineQuote, TopPost)
    
    // Privacy
    send_read_receipts: bool,      // NEW
    show_typing_indicators: bool,  // NEW
    share_availability: bool,      // NEW
    
    // Notifications
    sound_enabled: bool,
    desktop_notifications: bool,   // NEW
    notification_rules: Vec<NotificationRule>,  // NEW
    
    // Mobile
    swipe_left: SwipeAction,
    swipe_right: SwipeAction,
    
    // Keyboard
    palette_shortcut: String,
    custom_keybindings: HashMap<String, String>,  // NEW
    
    // Attention
    attention_budget_daily: u32,   // NEW (default 100)
    focus_time_enabled: bool,      // NEW
    
    // Data
    auto_archive_days: Option<u32>,  // NEW
    cache_size_mb: u32,            // NEW
    
    // Locale
    date_format: DateFormat,       // NEW
    language: Language,            // NEW
    
    // Vacation
    vacation: VacationResponder,
}
```

### Persistence: One localStorage key, loaded on init, saved via Effect

```
Key: "mycelix_pulse_preferences"
Value: JSON serialized UserPreferences
```

---

## Settings Page Structure (Redesigned)

### Section 1: Appearance
- Theme: Dark/Light toggle (existing)
- Named themes: 10 theme swatches (existing, move from panel to settings)
- Accent color: 10 color swatches (existing, move from panel)
- Density: Compact/Default/Comfortable (existing)
- Font family: System/Inter/JetBrains Mono/Serif
- Font size: Slider 75%-150%
- High contrast: Toggle
- Custom CSS: Textarea with Apply button (existing)

### Section 2: Layout
- Reading pane: Off/Right/Bottom (existing, needs persistence)
- Sidebar: Show/hide sections toggle
- Conversation grouping: Threaded/Flat default

### Section 3: Compose Defaults
- Default encryption: PQC/Standard toggle
- Default signature: Dropdown from signatures list
- Reply style: Inline quote / Top post
- Undo send delay: 3s/5s/10s/30s

### Section 4: Privacy & Security
- Read receipts: Send automatically / Ask each time / Never
- Typing indicators: Show/Hide
- Availability sharing: Everyone / Contacts only / Nobody
- Encryption indicator: Always show / Only for exceptions (existing behavior)

### Section 5: Notifications
- Desktop notifications: On/Off + permission request button
- Sound: On/Off
- Per-label rules: Table with label → action (Notify/Silent/Mute)
- Quiet hours: Start time / End time

### Section 6: Keyboard Shortcuts
- Full remappable shortcut table
- Two-column layout: Action | Current Key | Remap button
- Reset to defaults button

### Section 7: Attention Budget
- Daily budget: Slider 50-200 points
- Meeting costs: 1:1 (10), Group (20), Presentation (40)
- Auto-decline when depleted: On/Off
- Focus time schedule: Weekday time ranges

### Section 8: Data & Storage
- Auto-archive: After 30/60/90/Never days
- Cache size limit: 50/100/500 MB
- Storage used: Display current usage
- Clear cache button
- Reset demo data button (existing, in command palette)

### Section 9: Locale
- Date format: Auto (locale) / DD/MM/YYYY / MM/DD/YYYY / YYYY-MM-DD
- Language: English (+ future: Afrikaans, Zulu, French, Spanish)
- Time zone: Auto-detect / Manual override

### Section 10: Signatures (existing, enhanced)
- List of signatures with edit/delete
- Create new signature form
- Assign per-new/per-reply defaults

### Section 11: Filters & Rules (existing)
- List with enable/disable
- Create new filter form with conditions + actions

### Section 12: Labels (existing, enhanced)
- Color picker per label
- Create/rename/delete

### Section 13: External Accounts (moved from /accounts page)
- OAuth for Gmail/Outlook (existing)
- IMAP/SMTP for others (existing)

### Section 14: Vacation Responder (existing, needs persistence)
- Enable/disable
- Subject/body
- Date range
- Contacts only toggle

### Section 15: Backup & Restore (existing, wired)
- Download backup JSON
- Restore from JSON
- Export contacts vCard/CSV

---

## Build Order

### Phase 1: Persistence (fix what's broken)
1. Create `UserPreferences` struct
2. Load from localStorage on app init
3. Save via Effect on any change
4. Migrate existing individual keys to unified struct

### Phase 2: Core Settings (8 items from the user's request)
5. Notification preferences (sound, desktop, per-label rules)
6. Compose defaults (PQC default, default signature, reply style)
7. Privacy controls (read receipts, typing indicators, availability)
8. Keyboard shortcut remapping (full table)
9. Display preferences (conversation grouping, snippet length)
10. Account management (default send-from, sync frequency)
11. Attention budget config
12. Data management (auto-archive, cache, storage display)

### Phase 3: Locale + Font (3 items)
13. Font family picker
14. Date format selector
15. Language selector

---

## UX Notes

- Settings should be organized as a left-sidebar navigation within the settings page (not a single scroll)
- Each section should be independently saveable
- "Reset to defaults" button per section
- Search within settings (reuse command palette's search logic)
- Settings page should be the deepest, most polished page in the app — it's where power users live
