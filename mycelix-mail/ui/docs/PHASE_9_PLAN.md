# Phase 9: Advanced Search & Organization System

## Overview
Transform Mycelix Mail into a powerful email organization tool with advanced search capabilities, labels/tags system, smart folders, and intelligent filtering. This phase focuses on helping users find and organize emails efficiently.

## Goals
- ✅ Implement comprehensive search with multiple filters
- ✅ Add labels/tags system for email categorization
- ✅ Create smart folders (All Mail, Starred, Important, Unread, etc.)
- ✅ Build advanced search UI with filter chips
- ✅ Add search history and suggestions
- ✅ Implement bulk labeling operations
- ✅ Create label management interface
- ✅ Add keyboard shortcuts for labeling

## Part 1: Labels & Tags System (1-2 hours)

### 1.1 Label Store (Zustand)
**File**: `frontend/src/store/labelStore.ts`

```typescript
interface Label {
  id: string;
  name: string;
  color: string; // hex color for visual distinction
  icon?: string; // optional emoji icon
  emailCount: number;
  createdAt: string;
  updatedAt: string;
}

interface EmailLabel {
  emailId: string;
  labelIds: string[];
}

interface LabelStore {
  labels: Label[];
  emailLabels: Map<string, string[]>; // emailId -> labelIds[]
  addLabel: (label: Omit<Label, 'id' | 'createdAt' | 'updatedAt'>) => void;
  updateLabel: (id: string, updates: Partial<Label>) => void;
  deleteLabel: (id: string) => void;
  addLabelToEmail: (emailId: string, labelId: string) => void;
  removeLabelFromEmail: (emailId: string, labelId: string) => void;
  getLabelsForEmail: (emailId: string) => Label[];
  getEmailsWithLabel: (labelId: string) => string[];
  bulkLabelEmails: (emailIds: string[], labelId: string) => void;
  bulkUnlabelEmails: (emailIds: string[], labelId: string) => void;
}
```

**Default Labels**:
- Work (🔧 #3B82F6 blue)
- Personal (👤 #10B981 green)
- Important (⭐ #F59E0B amber)
- Follow Up (📌 #EF4444 red)
- Later (📅 #8B5CF6 purple)
- Receipts (🧾 #6B7280 gray)

### 1.2 Label Manager Component
**File**: `frontend/src/components/LabelManager.tsx`

Features:
- Grid view of all labels with color coding
- Create new label with color picker
- Edit label (name, color, icon)
- Delete label (with confirmation)
- Email count per label
- Drag to reorder labels

### 1.3 Label Picker Component
**File**: `frontend/src/components/LabelPicker.tsx`

Features:
- Modal/dropdown for selecting labels
- Checkbox interface for multi-select
- Search/filter labels
- Quick create new label
- Show currently applied labels
- Keyboard navigation

### 1.4 Label Chips Component
**File**: `frontend/src/components/LabelChip.tsx`

Features:
- Colored pill/badge displaying label
- Click to filter by label
- Remove label action (X button)
- Tooltip showing label name
- Dark mode support

## Part 2: Smart Folders (1 hour)

### 2.1 Virtual Folders System
Extend existing folder system with smart folders:

**Smart Folders**:
1. **All Mail** - All emails across all folders (excluding Trash/Spam)
2. **Starred** - All starred emails
3. **Important** - Emails marked as important or with Important label
4. **Unread** - All unread emails
5. **Has Attachments** - Emails with one or more attachments
6. **Sent by Me** - All sent emails
7. **Drafts** - Draft emails

**Implementation**:
```typescript
// In DashboardPage.tsx
const SMART_FOLDERS = {
  ALL_MAIL: '__all_mail__',
  STARRED: '__starred__',
  IMPORTANT: '__important__',
  UNREAD: '__unread__',
  ATTACHMENTS: '__attachments__',
  SENT: '__sent__',
  DRAFTS: '__drafts__',
} as const;

// Add to folders array with special icons
const smartFolders: Folder[] = [
  { id: SMART_FOLDERS.ALL_MAIL, name: 'All Mail', type: 'CUSTOM', icon: '📧' },
  { id: SMART_FOLDERS.STARRED, name: 'Starred', type: 'CUSTOM', icon: '⭐' },
  { id: SMART_FOLDERS.IMPORTANT, name: 'Important', type: 'CUSTOM', icon: '❗' },
  { id: SMART_FOLDERS.UNREAD, name: 'Unread', type: 'CUSTOM', icon: '🔵' },
  { id: SMART_FOLDERS.ATTACHMENTS, name: 'Attachments', type: 'CUSTOM', icon: '📎' },
];
```

### 2.2 Smart Folder Views
Create filtering logic for each smart folder in EmailList component.

## Part 3: Advanced Search System (2 hours)

### 3.1 Search Store
**File**: `frontend/src/store/searchStore.ts`

```typescript
interface SearchFilter {
  query: string;
  from?: string;
  to?: string;
  subject?: string;
  hasAttachment?: boolean;
  isStarred?: boolean;
  isUnread?: boolean;
  labelIds?: string[];
  dateFrom?: Date;
  dateTo?: Date;
  folderIds?: string[];
}

interface SavedSearch {
  id: string;
  name: string;
  filters: SearchFilter;
  createdAt: string;
}

interface SearchStore {
  currentFilter: SearchFilter;
  searchHistory: string[]; // recent queries
  savedSearches: SavedSearch[];
  setFilter: (filter: Partial<SearchFilter>) => void;
  clearFilter: () => void;
  addToHistory: (query: string) => void;
  saveSearch: (name: string, filters: SearchFilter) => void;
  deleteSavedSearch: (id: string) => void;
}
```

### 3.2 Advanced Search Modal
**File**: `frontend/src/components/AdvancedSearch.tsx`

Features:
- Text search input (searches subject, body, from, to)
- Filter by sender (from)
- Filter by recipient (to)
- Filter by subject
- Date range picker (from/to dates)
- Has attachment toggle
- Is starred toggle
- Is unread toggle
- Label multi-select
- Folder multi-select
- Save search option
- Load saved search
- Clear all filters button
- Search results count preview

### 3.3 Search Filter Chips Bar
**File**: `frontend/src/components/SearchFilterChips.tsx`

Features:
- Display active filters as removable chips
- "From: john@example.com" chip
- "Has attachments" chip
- "Starred" chip
- Date range chip
- Label chips
- Click chip to edit filter
- Click X to remove filter
- Clear all button

### 3.4 Search Suggestions
Add autocomplete to search:
- Recent searches
- Saved searches
- Common senders
- Label names
- Folder names

## Part 4: Enhanced Email List Integration (1 hour)

### 4.1 Update EmailList Component
Add support for:
- Display labels on each email (as colored chips)
- Quick label menu on hover/right-click
- Bulk label operations
- Filter by label click
- Label badge colors

### 4.2 Update EmailView Component
Add:
- Label display in email header
- Quick add/remove label button
- Label dropdown menu
- Keyboard shortcut (L key) to open label picker

## Part 5: UI/UX Enhancements (1 hour)

### 5.1 Keyboard Shortcuts
- `L` - Open label picker for selected email
- `Shift + L` - Bulk label selected emails
- `/` - Focus search (already exists, enhance with advanced search)
- `Ctrl/Cmd + F` - Open advanced search modal
- `G then A` - Go to All Mail
- `G then S` - Go to Starred
- `G then I` - Go to Important

### 5.2 Visual Design
- Color-coded labels throughout
- Filter chips with material design
- Smooth transitions
- Empty states for each smart folder
- Loading states
- Icons for smart folders

### 5.3 Settings Integration
Add "Labels & Organization" tab to Settings:
- Label management
- Saved searches management
- Default label settings
- Auto-labeling rules (future)

## Part 6: Performance Optimizations

### 6.1 Efficient Filtering
- Index emails by labels in memory
- Cache search results
- Debounce search input
- Lazy load label data

### 6.2 Virtual Scrolling
For large email lists:
- Implement react-window or similar
- Render only visible emails
- Maintain smooth scrolling

## Technical Implementation Plan

### Priority 1 (Immediate Value):
1. Create labelStore with Zustand + persist
2. Add LabelPicker component
3. Integrate labels into EmailList (display)
4. Add LabelManager to Settings
5. Create smart folders (All Mail, Starred, Unread, etc.)

### Priority 2 (Search Enhancement):
1. Create searchStore
2. Build AdvancedSearch modal
3. Add SearchFilterChips bar
4. Implement filtering logic
5. Add search history

### Priority 3 (Polish):
1. Bulk label operations
2. Keyboard shortcuts for labels
3. Saved searches
4. Color picker for labels
5. Label autocomplete

### Priority 4 (Performance):
1. Virtual scrolling for long lists
2. Search result caching
3. Index optimization
4. Lazy loading

## Success Metrics
- Users can label any email with multiple labels
- Smart folders show correct filtered results
- Advanced search finds emails accurately
- Search is fast (< 200ms for typical queries)
- Labels are visually distinct and easy to identify
- Bulk operations work on 100+ emails efficiently
- All features work in dark mode
- Keyboard shortcuts reduce mouse usage

## Future Enhancements (Phase 10+)
- Auto-labeling rules (if from X, add label Y)
- ML-powered smart labels (auto-categorize emails)
- Label hierarchies (nested labels)
- Label sharing (for team accounts)
- Export/import labels
- Label statistics dashboard
- Regex search support
- Full-text search with highlighting

## Files to Create/Modify

### New Files:
- `frontend/src/store/labelStore.ts`
- `frontend/src/store/searchStore.ts`
- `frontend/src/components/LabelManager.tsx`
- `frontend/src/components/LabelPicker.tsx`
- `frontend/src/components/LabelChip.tsx`
- `frontend/src/components/AdvancedSearch.tsx`
- `frontend/src/components/SearchFilterChips.tsx`
- `frontend/src/components/SavedSearches.tsx`

### Modified Files:
- `frontend/src/components/EmailList.tsx` - Add label display
- `frontend/src/components/EmailView.tsx` - Add label picker
- `frontend/src/pages/DashboardPage.tsx` - Integrate smart folders
- `frontend/src/components/FolderList.tsx` - Add smart folder icons
- `frontend/src/pages/SettingsPage.tsx` - Add Labels tab
- `frontend/src/components/KeyboardShortcutsHelp.tsx` - Add label shortcuts
- `frontend/src/hooks/useKeyboardShortcuts.ts` - Add new shortcuts
- `README.md` - Document Phase 9 features

## Estimated Timeline
- **Part 1 (Labels)**: 1-2 hours
- **Part 2 (Smart Folders)**: 1 hour
- **Part 3 (Search)**: 2 hours
- **Part 4 (Integration)**: 1 hour
- **Part 5 (UX)**: 1 hour
- **Total**: 6-7 hours of focused development

## Design Principles
1. **Performance First** - Keep UI responsive even with thousands of emails
2. **Accessibility** - Full keyboard navigation, ARIA labels, screen reader support
3. **Visual Clarity** - Clear visual hierarchy, intuitive icons, consistent colors
4. **Progressive Disclosure** - Simple by default, powerful when needed
5. **Mobile-Friendly** - Responsive design, touch-friendly targets
6. **Data Persistence** - All labels and searches saved locally
7. **Dark Mode** - Fully integrated dark mode support
8. **Undo/Redo** - Support undoing label operations

This phase will transform Mycelix Mail from a good email client into a powerful organization tool that rivals Gmail, Outlook, and other commercial clients!
