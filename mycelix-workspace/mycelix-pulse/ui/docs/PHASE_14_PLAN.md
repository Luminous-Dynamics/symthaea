# Phase 14: Productivity & Performance Enhancements

## Overview
Focus on features that maximize daily productivity and improve performance for power users managing hundreds of emails.

---

## Part 1: Smart Email Actions & Workflows ⚡

### Archive and Next Pattern
**Problem**: After reading an email, users manually archive and then click the next email (2 actions)
**Solution**: Smart actions that combine operations

**Features:**
- **Archive and Next** - Archive email and auto-select next unread
- **Delete and Next** - Delete email and auto-select next
- **Mark Read and Next** - Mark as read and move to next unread
- **Reply and Archive** - Send reply, archive original, select next
- **Star and Next** - Star email and select next
- **Snooze and Next** - Snooze email and select next

**Implementation:**
- Smart action buttons in EmailView toolbar
- Keyboard shortcuts: `e` (archive), `#` (delete), `]` (next), `[` (previous)
- Auto-selection logic that skips to next unread intelligently
- Toast notifications with undo support
- Preference to enable/disable auto-next behavior

### Auto Mark as Read
**Problem**: Users want emails marked read after viewing for X seconds
**Solution**: Configurable timer to auto-mark read

**Features:**
- **Configurable Timer** - 2s, 5s, 10s, Never
- **Visual Indicator** - Progress bar showing time until mark-read
- **Manual Override** - Click to mark read immediately
- **Settings Toggle** - Enable/disable in settings
- **Smart Detection** - Only counts when email is visible (not scrolled away)

**Implementation:**
- useAutoMarkRead hook with timer logic
- Progress bar component in EmailView
- Visibility detection with IntersectionObserver
- Settings integration with localStorage persistence

---

## Part 2: Enhanced Template System 📝

### Template Variables & Smart Insertion
**Problem**: Templates are static text, need personalization
**Solution**: Variable substitution system

**Variables:**
- `{name}` - Recipient's name (extracted from email)
- `{firstName}` - First name only
- `{email}` - Recipient's email address
- `{subject}` - Original email subject
- `{date}` - Current date (formatted)
- `{time}` - Current time
- `{myName}` - Your name
- `{myEmail}` - Your email
- `{company}` - Your company (from settings)

**Quick Template Insertion:**
- **Keyboard Shortcut** - Ctrl+Shift+T to open template picker
- **Fuzzy Search** - Type to filter templates instantly
- **Recently Used** - Show most recent templates at top
- **Template Preview** - Preview with variables filled in
- **One-Click Insert** - Click to insert at cursor position

**Template Management:**
- **Folders/Categories** - Organize templates (Work, Personal, Support)
- **Template Shortcuts** - Assign custom shortcuts (e.g., Ctrl+1 for template 1)
- **Usage Analytics** - Track most-used templates
- **Export/Import** - Share templates as JSON
- **Variables Helper** - Show available variables while editing

**Implementation:**
- Variable parser and replacement engine
- Enhanced TemplatePicker with search and recent
- Template categories in templateStore
- Settings UI for template management
- Keyboard shortcut system for templates

---

## Part 3: Bulk Operations Progress & Enhancements 📊

### Progress Indicators
**Problem**: Bulk operations on 100+ emails show no progress
**Solution**: Real-time progress tracking

**Features:**
- **Progress Modal** - Shows current operation status
- **Progress Bar** - Visual indicator (35/100 emails processed)
- **Speed Estimate** - "~15 seconds remaining"
- **Success/Failure Count** - "98 succeeded, 2 failed"
- **Cancellation** - Ability to cancel in-progress operation
- **Error Details** - See which emails failed and why

**Enhanced Bulk Operations:**
- **Select All Matching** - Select all emails matching current filter/search
- **Select Range** - Shift+Click to select range
- **Invert Selection** - Select all unselected emails
- **Smart Select** - "Select all unread", "Select all from sender"
- **Persistent Selection** - Selection survives folder changes (optional)

**Undo for Bulk Operations:**
- **Bulk Undo** - Undo operations on 100+ emails
- **Undo Queue** - Show what can be undone
- **Undo All** - Undo multiple operations at once

**Implementation:**
- BulkProgressModal component
- Enhanced bulk mutation with progress callbacks
- Selection enhancement in EmailList
- Extended actionHistoryStore for bulk operations

---

## Part 4: Advanced Keyboard Shortcuts & Customization ⌨️

### Additional Gmail-Style Shortcuts
- `e` - Archive email (like Gmail)
- `#` - Delete email
- `gi` - Go to Inbox
- `gs` - Go to Starred
- `gt` - Go to Sent
- `gl` - Go to Labels (show label picker)
- `]` - Next email (alternative to j)
- `[` - Previous email (alternative to k)
- `/` - Focus search (already implemented)
- `?` - Show keyboard shortcuts help (already implemented)
- `!` - Mark as spam
- `v` - Move to folder (show folder picker)
- `y` - Archive (Gmail alternative)

### Shortcut Customization
- **Customization UI** - Edit shortcuts in settings
- **Conflict Detection** - Warn about conflicting shortcuts
- **Reset to Default** - Restore default shortcuts
- **Export/Import** - Share shortcut configurations
- **Shortcut Cheat Sheet** - Printable reference

**Implementation:**
- Keyboard shortcut registry system
- Settings UI for shortcut customization
- Conflict detection algorithm
- Enhanced KeyboardShortcutsHelp component

---

## Part 5: Performance Optimizations 🚀

### Virtual Scrolling
**Problem**: EmailList with 1000+ emails is slow
**Solution**: Render only visible emails

**Implementation:**
- **react-window** - Windowing library for React
- **FixedSizeList** - For consistent email heights
- **VariableSizeList** - For dynamic heights with density
- **Scroll Restoration** - Remember scroll position
- **Keyboard Navigation** - Works with virtual scrolling

### Code Splitting
**Problem**: Large initial bundle size
**Solution**: Split code by route and feature

**Routes to Split:**
- DashboardPage (lazy load)
- SettingsPage (lazy load)
- LoginPage (lazy load)
- All modals (lazy load on open)

### Image Lazy Loading
**Problem**: Email images load immediately, slow rendering
**Solution**: Load images as they scroll into view

**Implementation:**
- IntersectionObserver for email avatars
- Lazy load email body images
- Placeholder while loading
- Error fallback images

### React Query Optimizations
- **Stale-While-Revalidate** - Show cached data while refetching
- **Prefetch Next Email** - Prefetch likely next email to view
- **Background Refetch** - Refetch on window focus
- **Optimistic Updates** - Instant UI feedback
- **Query Deduplication** - Prevent duplicate requests

**Implementation:**
- Enhanced React Query configuration
- Prefetch logic based on navigation patterns
- Optimistic update patterns for all mutations

---

## Part 6: Email Filters & Rules 🔧

### Client-Side Filters
**Problem**: Users want automatic organization
**Solution**: Gmail-style filters

**Filter Conditions:**
- From (contains, is, is not)
- To (contains, is, is not)
- Subject (contains, is, is not)
- Has attachment (yes, no)
- Size (greater than, less than)
- Date (before, after, between)
- Has label
- Is starred / Is unread

**Filter Actions:**
- Apply label
- Star email
- Mark as read
- Archive
- Delete
- Move to folder
- Skip inbox

**Filter Management:**
- **Filter List** - See all active filters
- **Filter Editor** - Visual filter builder
- **Filter Priority** - Order matters, drag to reorder
- **Enable/Disable** - Toggle filters on/off
- **Test Filter** - Preview which emails match
- **Import/Export** - Share filter configurations

**Implementation:**
- Filter engine that runs on email fetch
- filterStore with Zustand
- Filter builder UI component
- Settings integration

---

## Part 7: Notification Center 🔔

### Notification System
**Problem**: No centralized place to see notifications
**Solution**: Notification center like macOS/iOS

**Features:**
- **Notification Bell** - Icon in header with unread count
- **Notification Panel** - Slide-out panel from right
- **Notification Types**:
  - New email received
  - Snooze reminder
  - Filter action completed
  - Bulk operation completed
  - Draft auto-saved
  - Account sync status
- **Notification Actions** - Quick actions from notification
- **Mark All Read** - Clear all notifications
- **Notification History** - Last 50 notifications
- **Sound Preferences** - Enable/disable notification sounds

**Implementation:**
- notificationStore with Zustand
- NotificationPanel component
- Notification bell in header
- Integration with existing features

---

## Priority Order

1. **Part 1** - Smart Email Actions (Highest User Value)
2. **Part 2** - Enhanced Templates (High Productivity Gain)
3. **Part 3** - Bulk Operations Progress (Quality of Life)
4. **Part 4** - Additional Keyboard Shortcuts (Power Users)
5. **Part 5** - Performance Optimizations (Scalability)
6. **Part 6** - Email Filters (Advanced Organization)
7. **Part 7** - Notification Center (Nice to Have)

## Success Metrics

- **Task Completion Time**: 40% faster for common workflows
- **Keyboard Coverage**: 95% of actions accessible via keyboard
- **Performance**: Support 10,000+ email lists without lag
- **User Delight**: Professional features matching Gmail/Outlook
- **Productivity**: Power users can manage inbox 2x faster

---

## Estimated Impact

**Part 1 (Smart Actions)**: ⭐⭐⭐⭐⭐ (Game changer for daily use)
**Part 2 (Templates)**: ⭐⭐⭐⭐ (Huge time saver)
**Part 3 (Bulk Progress)**: ⭐⭐⭐⭐ (Professional polish)
**Part 4 (Shortcuts)**: ⭐⭐⭐⭐ (Power user essential)
**Part 5 (Performance)**: ⭐⭐⭐⭐⭐ (Enables scale)
**Part 6 (Filters)**: ⭐⭐⭐ (Advanced users)
**Part 7 (Notifications)**: ⭐⭐⭐ (Convenience)
