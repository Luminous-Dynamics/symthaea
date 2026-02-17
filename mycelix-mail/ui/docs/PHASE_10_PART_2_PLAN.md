# Phase 10 Part 2: Undo/Redo, UI Polish & Advanced Features

## Overview
Complete the Phase 10 Professional Polish initiative with undo/redo system, advanced UI animations, and productivity enhancements.

## Goals
- ✅ Part 1: Conversation Threading, Avatars, Contacts, Quick Actions, Importance Detection (COMPLETED)
- ⏳ Part 2: Undo/Redo, UI Polish, Advanced Features (IN PROGRESS)

## Part 2A: Undo/Redo Action History System - 1 hour

### Why This Matters
Users make mistakes. An undo system gives confidence to use destructive actions without fear.

### Implementation

#### 1. Action History Store
**File**: `frontend/src/store/actionHistoryStore.ts`

```typescript
interface Action {
  id: string;
  type: 'delete' | 'archive' | 'label' | 'move' | 'mark_read' | 'star';
  timestamp: string;
  emailIds: string[];
  previousState: any; // Store previous state for rollback
  undo: () => Promise<void>;
  description: string; // "Deleted 3 emails", "Archived email from John"
}

interface ActionHistoryStore {
  history: Action[];
  currentIndex: number; // For redo support
  maxHistory: number; // Default 10

  addAction: (action: Omit<Action, 'id' | 'timestamp'>) => void;
  undo: () => Promise<void>;
  redo: () => Promise<void>;
  canUndo: boolean;
  canRedo: boolean;
  clearHistory: () => void;
}
```

#### 2. Toast with Undo Button
- Show toast notification after destructive action
- Include "UNDO" button in toast
- 5-second auto-dismiss (or until user clicks)
- Example: "Email deleted" [UNDO]

#### 3. Keyboard Shortcut
- `Ctrl/Cmd+Z` - Undo last action
- `Ctrl/Cmd+Shift+Z` or `Ctrl/Cmd+Y` - Redo action

#### 4. Integration Points
Wrap these operations with undo tracking:
- Delete email
- Archive email
- Apply/remove label
- Mark as read/unread
- Star/unstar
- Move to folder

#### 5. Implementation Strategy
```typescript
// Example: Delete with undo
const deleteWithUndo = async (emailIds: string[]) => {
  // Get current state
  const emails = emailIds.map(id => getEmailById(id));

  // Perform deletion
  await api.deleteEmails(emailIds);

  // Add to undo history
  addAction({
    type: 'delete',
    emailIds,
    previousState: emails,
    undo: async () => {
      await api.restoreEmails(emails);
    },
    description: `Deleted ${emailIds.length} email${emailIds.length > 1 ? 's' : ''}`
  });

  // Show toast with undo
  toast.success('Email deleted', {
    action: {
      label: 'Undo',
      onClick: () => undo()
    }
  });
};
```

## Part 2B: Advanced UI Polish & Animations - 1.5 hours

### Why This Matters
Smooth animations and polished UI make the app feel professional and enjoyable to use.

### Implementation

#### 1. Smooth Transitions & Animations
**File**: `frontend/src/styles/animations.css`

Add CSS animations for:
- **Email list items**: Fade-in on load, slide-out on delete
- **Modal entrance/exit**: Scale and fade
- **Button interactions**: Ripple effect
- **Toast notifications**: Slide-in from top-right
- **Sidebar navigation**: Smooth collapse/expand
- **Loading states**: Skeleton shimmer effect

```css
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

@keyframes slideIn {
  from { transform: translateY(-10px); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}

@keyframes slideOut {
  from { transform: translateX(0); opacity: 1; }
  to { transform: translateX(-100%); opacity: 0; }
}

@keyframes shimmer {
  0% { background-position: -1000px 0; }
  100% { background-position: 1000px 0; }
}

.email-item {
  animation: fadeIn 0.3s ease-out;
}

.modal-enter {
  animation: slideIn 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}

.skeleton-loading {
  animation: shimmer 2s infinite;
  background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
  background-size: 2000px 100%;
}
```

#### 2. Enhanced Skeleton Loading
Improve existing skeleton screens:
- **EmailList**: Show realistic email item skeletons with avatar placeholder
- **EmailView**: Content-aware skeleton (subject, body, attachments)
- **Sidebar**: Folder tree skeleton
- Add shimmer effect to all skeletons

#### 3. Micro-interactions
Add subtle feedback:
- **Star button**: Scale + rotate on click
- **Checkbox**: Checkmark animation
- **Label chips**: Bounce on add, fade on remove
- **Avatar**: Pulse on hover
- **Quick action buttons**: Scale + glow on hover

#### 4. Page Transitions
- Smooth fade between email list and email view
- Slide-in compose modal
- Scale-in settings modal

#### 5. Progress Indicators
- **Bulk operations**: Progress bar showing N/Total
- **Email sending**: Indeterminate progress
- **Attachment upload**: Determinate progress with percentage

## Part 2C: Preview Pane Toggle - 30 minutes

### Why This Matters
Different users prefer different layouts. Power users often prefer no preview pane for maximum email density.

### Implementation

#### 1. Preview Pane Modes
Three layout options:
1. **Vertical Split** (default): List on left | Preview on right
2. **Horizontal Split**: List on top | Preview on bottom
3. **No Preview**: List only, click opens full view in modal/overlay

#### 2. Settings Integration
**Add to Settings > General**:
```typescript
interface ViewSettings {
  previewPaneMode: 'vertical' | 'horizontal' | 'none';
  previewPaneSize: number; // Percentage (e.g., 40 for 40% width)
}
```

#### 3. Keyboard Shortcut
- `V` - Cycle through preview pane modes

#### 4. Resizable Splitter
- Allow drag-to-resize between list and preview
- Save preferred size to localStorage
- Minimum/maximum size constraints

## Part 2D: Advanced Bulk Operations - 30 minutes

### Implementation

#### 1. Enhanced Selection
**New features**:
- **Select by criteria**: "Select all unread", "Select all from sender"
- **Invert selection**: Select everything except current selection
- **Select thread**: Select entire conversation thread
- **Select with label**: Select all with specific label

#### 2. Selection Menu
Add dropdown in BulkActionsToolbar:
```
Select ▼
  ├─ All
  ├─ None
  ├─ Unread
  ├─ Starred
  ├─ With Attachments
  ├─ From Sender
  └─ Invert Selection
```

#### 3. Bulk Operations Progress
- Show progress modal for operations affecting >10 emails
- "Processing 45 of 100 emails..." with progress bar
- Cancel button for long-running operations
- Error handling: "45 succeeded, 5 failed" with retry option

#### 4. Bulk Export
- Export selected emails to JSON
- Export to EML format
- Download as ZIP archive

## Part 2E: Email Density Options - 15 minutes

### Implementation

#### 1. Density Levels
**Add to Settings > Appearance**:
- **Comfortable** (default): Current spacing (py-3)
- **Compact**: Tighter spacing (py-2), smaller font
- **Cozy**: Medium spacing (py-2.5)

#### 2. Visual Changes
```typescript
// Comfortable
<div className="py-3 px-4 text-sm">

// Compact
<div className="py-2 px-3 text-xs">

// Cozy
<div className="py-2.5 px-3.5 text-sm">
```

#### 3. More Emails Visible
- **Comfortable**: ~8 emails on 1080p screen
- **Compact**: ~12 emails on 1080p screen
- **Cozy**: ~10 emails on 1080p screen

## Part 2F: Enhanced Empty States - 15 minutes

### Implementation

#### 1. Better Illustrations
Replace emoji with SVG illustrations or icons:
- Empty inbox: Inbox with checkmark
- No search results: Magnifying glass with X
- No drafts: Pencil with empty paper
- No sent items: Paper airplane

#### 2. Contextual Actions
Add helpful actions to empty states:
- **Empty inbox**: "Compose your first email" button
- **No search results**: "Clear search" button
- **No drafts**: "Start composing" button

#### 3. Tips & Tricks
Show helpful tips in empty states:
- "Tip: Press 'c' to compose a new email"
- "Tip: Use labels to organize your emails"

## Priority Implementation Order

### High Priority (Must Have) - 2 hours
1. ✅ Undo/Redo System (1h) - Critical for user confidence
2. ✅ UI Animations (1h) - Professional polish

### Medium Priority (Should Have) - 1 hour
3. Preview Pane Toggle (30min) - Power user feature
4. Advanced Bulk Operations (30min) - Productivity boost

### Low Priority (Nice to Have) - 30 minutes
5. Email Density Options (15min)
6. Enhanced Empty States (15min)

## Technical Considerations

### Performance
- Animations should be GPU-accelerated (use transform, opacity)
- Debounce undo actions to prevent spam
- Limit undo history to 10 actions (configurable)
- Use CSS animations instead of JS when possible

### Accessibility
- Animations respect `prefers-reduced-motion`
- Undo announcements for screen readers
- Keyboard shortcuts don't conflict with browser defaults

### Storage
- Undo history cleared on page reload (not persisted)
- Critical actions (bulk delete >10) can be persisted
- Preview pane preferences saved to localStorage

## Success Criteria
- ✅ Undo works for all destructive actions
- ✅ Redo works after undo
- ✅ All animations smooth at 60fps
- ✅ Preview pane modes switch seamlessly
- ✅ Bulk operations show progress for large selections
- ✅ Density options provide noticeable difference
- ✅ Empty states are helpful and actionable

## Files to Create

### New Files:
1. `frontend/src/store/actionHistoryStore.ts` - Undo/redo state
2. `frontend/src/styles/animations.css` - Animation definitions
3. `frontend/src/components/ProgressModal.tsx` - Bulk operation progress
4. `frontend/src/hooks/usePreviewPane.ts` - Preview pane management
5. `frontend/src/hooks/useUndoRedo.ts` - Undo/redo hook

### Modified Files:
1. `frontend/src/components/EmailList.tsx` - Density, animations
2. `frontend/src/pages/DashboardPage.tsx` - Preview pane modes
3. `frontend/src/components/ToastContainer.tsx` - Undo button
4. `frontend/src/components/BulkActionsToolbar.tsx` - Advanced selection
5. `frontend/src/pages/SettingsPage.tsx` - Density, preview pane settings
6. `frontend/src/components/EmptyState.tsx` - Enhanced illustrations

## Estimated Timeline
- **High Priority**: 2 hours (Undo/Redo, Animations)
- **Medium Priority**: 1 hour (Preview Pane, Bulk Ops)
- **Low Priority**: 30 minutes (Density, Empty States)
- **Testing & Polish**: 30 minutes
- **Total**: ~4 hours

## Let's Start with Undo/Redo!
The undo/redo system is the highest-impact feature that will transform user confidence in destructive actions.
