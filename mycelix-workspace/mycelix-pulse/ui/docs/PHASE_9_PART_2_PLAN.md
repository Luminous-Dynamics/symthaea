# Phase 9 Part 2: Smart Folders & Enhanced Navigation

## Overview
Build on the Labels system by adding smart virtual folders and enhanced keyboard navigation to make email organization even more powerful.

## Goals
- ✅ Implement 5 smart virtual folders
- ✅ Add keyboard shortcuts for labeling
- ✅ Enhance folder navigation with "Go to" shortcuts
- ✅ Add label filtering in EmailList
- ✅ Update keyboard shortcuts help

## Implementation Plan (1-2 hours)

### Priority 1: Smart Folders (45 min)

#### 1.1 Define Smart Folder System
Create virtual folders that dynamically filter emails:

1. **All Mail** (`__all_mail__`) - All emails except Trash/Spam
2. **Starred** (`__starred__`) - All starred emails
3. **Important** (`__important__`) - Emails with "Important" label
4. **Unread** (`__unread__`) - All unread emails
5. **Attachments** (`__attachments__`) - Emails with attachments

#### 1.2 Implementation Steps
- Add SMART_FOLDERS constants to DashboardPage
- Create smart folder objects with icons
- Add to folders array after regular folders
- Update FolderList to support smart folder icons
- Implement filtering logic in EmailList or create SmartFolderView components

#### 1.3 Technical Approach
```typescript
// In DashboardPage.tsx
const SMART_FOLDERS = {
  ALL_MAIL: '__all_mail__',
  STARRED: '__starred__',
  IMPORTANT: '__important__',
  UNREAD: '__unread__',
  ATTACHMENTS: '__attachments__',
} as const;

// Add smart folders to folder list
const smartFolders: Folder[] = [
  { id: SMART_FOLDERS.ALL_MAIL, name: 'All Mail', type: 'CUSTOM', icon: '📧', unreadCount: 0 },
  { id: SMART_FOLDERS.STARRED, name: 'Starred', type: 'CUSTOM', icon: '⭐', unreadCount: 0 },
  { id: SMART_FOLDERS.IMPORTANT, name: 'Important', type: 'CUSTOM', icon: '❗', unreadCount: 0 },
  { id: SMART_FOLDERS.UNREAD, name: 'Unread', type: 'CUSTOM', icon: '🔵', unreadCount: 0 },
  { id: SMART_FOLDERS.ATTACHMENTS, name: 'Attachments', type: 'CUSTOM', icon: '📎', unreadCount: 0 },
];
```

### Priority 2: Keyboard Shortcuts (30 min)

#### 2.1 Label Shortcuts
- `L` - Open label picker for selected email
- `Shift + L` - Bulk label selected emails
- `G then L` - Go to Labels (in settings)

#### 2.2 Navigation Shortcuts
- `G then A` - Go to All Mail
- `G then S` - Go to Starred
- `G then I` - Go to Important (Important label folder)
- `G then U` - Go to Unread

#### 2.3 Implementation
- Add to DashboardPage keyboard handler
- Use two-key sequence detection (G then X)
- Update KeyboardShortcutsHelp with new shortcuts

### Priority 3: Label Filtering (20 min)

#### 3.1 Filter by Label Click
- Make label chips in EmailList clickable
- Click label → filter to show only emails with that label
- Show "Filtered by: [Label]" indicator
- Add clear filter button

#### 3.2 Label Virtual Folders
- Create virtual folder for each label in sidebar
- Show under "Labels" section in folder list
- Click label folder → show all emails with that label

### Priority 4: Bulk Label Operations (15 min)

#### 4.1 Enhance BulkActionsToolbar
- Add "Label" button to bulk actions
- Opens LabelPicker in bulk mode
- Apply label to all selected emails

#### 4.2 Integration
- Pass bulk label handler to EmailList
- Update EmailList to support bulk labeling
- Show count of emails being labeled

## Updated File Structure

### Files to Modify:
1. `frontend/src/pages/DashboardPage.tsx`
   - Add smart folders
   - Add keyboard shortcuts
   - Implement filtering logic

2. `frontend/src/components/FolderList.tsx`
   - Support smart folder icons
   - Add "Labels" section with label folders
   - Highlight active folder

3. `frontend/src/components/EmailList.tsx`
   - Make labels clickable for filtering
   - Show filter indicator
   - Support smart folder filtering

4. `frontend/src/components/BulkActionsToolbar.tsx`
   - Add label button
   - Trigger bulk label picker

5. `frontend/src/components/KeyboardShortcutsHelp.tsx`
   - Add label shortcuts
   - Add navigation shortcuts (G+A, G+S, etc.)

### New Files:
- None needed! We'll use existing components

## Success Criteria
- ✅ All 5 smart folders work correctly
- ✅ Keyboard shortcuts navigate to smart folders
- ✅ L key opens label picker
- ✅ Shift+L bulk labels selected emails
- ✅ Labels are clickable in email list
- ✅ Bulk label button works in toolbar
- ✅ All shortcuts documented in help

## Technical Decisions

### Smart Folder Filtering
Option 1: Filter on frontend (Current approach)
- Pros: Fast, works with existing API
- Cons: Needs all emails loaded

Option 2: Backend filtering (Future enhancement)
- Pros: More efficient for large datasets
- Cons: Requires API changes

**Decision**: Use frontend filtering for now, can optimize later.

### Label Folders
Option 1: Show all labels as folders in sidebar
- Pros: Quick access to labeled emails
- Cons: Could clutter sidebar with many labels

Option 2: Separate "Labels" dropdown section
- Pros: Keeps sidebar clean
- Cons: Extra click to access label folders

**Decision**: Start without label folders, add if users request it.

### Two-Key Shortcuts (Gmail-style)
Implement "G then X" navigation:
- User presses G, system enters "navigation mode"
- Next key determines destination
- 500ms timeout to exit navigation mode
- Visual indicator when in navigation mode

## Timeline
- Smart Folders: 45 minutes
- Keyboard Shortcuts: 30 minutes
- Label Filtering: 20 minutes
- Bulk Operations: 15 minutes
- Testing & Polish: 20 minutes
- **Total: ~2 hours**

## Future Enhancements (Phase 9 Part 3+)
- Advanced search modal
- Saved searches
- Search history
- Filter chips bar
- Conversation threading
- Email importance detection (ML)
- Auto-labeling rules

Let's implement this now!
