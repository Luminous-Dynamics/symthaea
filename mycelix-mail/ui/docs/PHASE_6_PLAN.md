# Phase 6: Advanced UX, Performance & Code Quality

**Status**: In Progress
**Focus**: Power user features, performance optimizations, and code quality improvements
**Goal**: Achieve 98%+ production-ready status

---

## Overview

Phase 6 builds on the solid foundation from Phase 5, adding power user features, performance optimizations, and code quality improvements. The focus is on making the application feel polished, fast, and professional while maintaining excellent code quality.

---

## Power User Features (Priority P0)

### 1. Keyboard Shortcuts for Bulk Operations 🎹
**Status**: Not implemented
**Time**: 25 minutes
**Impact**: Very High - Power user productivity

**Shortcuts to add**:
- `Ctrl/Cmd + A`: Select all emails in current view
- `Shift + Delete`: Delete selected emails
- `Shift + U`: Mark selected as unread
- `Shift + R`: Mark selected as read
- `Shift + S`: Star selected emails
- `Shift + D`: Deselect all

**Implementation**:
```typescript
// DashboardPage.tsx
useEffect(() => {
  const handleKeyDown = (e: KeyboardEvent) => {
    // Don't trigger if typing in an input
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
      return;
    }

    const selectedIds = emailListRef.current?.getSelectedIds();

    // Select all
    if ((e.ctrlKey || e.metaKey) && e.key === 'a' && selectedFolderId) {
      e.preventDefault();
      emailListRef.current?.selectAll();
    }

    // Bulk operations (only if emails are selected)
    if (selectedIds && selectedIds.length > 0) {
      if (e.shiftKey && e.key === 'Delete') {
        e.preventDefault();
        bulkDeleteMutation.mutate(selectedIds);
      } else if (e.shiftKey && e.key.toLowerCase() === 'u') {
        e.preventDefault();
        bulkMarkReadMutation.mutate({ emailIds: selectedIds, isRead: false });
      } else if (e.shiftKey && e.key.toLowerCase() === 'r') {
        e.preventDefault();
        bulkMarkReadMutation.mutate({ emailIds: selectedIds, isRead: true });
      } else if (e.shiftKey && e.key.toLowerCase() === 's') {
        e.preventDefault();
        bulkStarMutation.mutate({ emailIds: selectedIds, isStarred: true });
      } else if (e.shiftKey && e.key.toLowerCase() === 'd') {
        e.preventDefault();
        emailListRef.current?.clearSelection();
      }
    }
  };

  window.addEventListener('keydown', handleKeyDown);
  return () => window.removeEventListener('keydown', handleKeyDown);
}, [selectedFolderId, bulkDeleteMutation, bulkMarkReadMutation, bulkStarMutation]);
```

**Files to Modify**:
- `frontend/src/pages/DashboardPage.tsx`
- `frontend/src/components/EmailList.tsx` (add selectAll to ref)
- `frontend/src/components/KeyboardShortcutsHelp.tsx` (document new shortcuts)

---

### 2. Email Draft Autosave 💾
**Status**: Not implemented
**Time**: 30 minutes
**Impact**: High - Prevents data loss

**Features**:
- Auto-save draft every 30 seconds
- Save to localStorage
- Restore draft on compose open
- Clear draft on send/discard

**Implementation**:
```typescript
// ComposeEmail.tsx
useEffect(() => {
  if (mode === 'new') {
    // Try to restore draft
    const savedDraft = localStorage.getItem('email-draft');
    if (savedDraft) {
      const draft = JSON.parse(savedDraft);
      setFormData(draft);
    }
  }
}, [mode]);

useEffect(() => {
  if (mode === 'new') {
    // Auto-save every 30 seconds
    const interval = setInterval(() => {
      localStorage.setItem('email-draft', JSON.stringify(formData));
    }, 30000);

    return () => clearInterval(interval);
  }
}, [formData, mode]);

const handleSend = async () => {
  // ... existing send logic
  // Clear draft after successful send
  localStorage.removeItem('email-draft');
};

const handleDiscard = () => {
  if (confirm('Discard this draft?')) {
    localStorage.removeItem('email-draft');
    onClose();
  }
};
```

**Files to Modify**:
- `frontend/src/components/ComposeEmail.tsx`

---

### 3. Folder Email Count Badges 🔢
**Status**: Not implemented
**Time**: 20 minutes
**Impact**: Medium - Better information at a glance

**Display**:
- Show unread count next to each folder
- Different styling for 0, 1-9, 10-99, 100+
- Update in real-time

**Implementation**:
```typescript
// FolderList.tsx
interface Folder {
  id: string;
  name: string;
  type: string;
  unreadCount?: number;
}

<div className="flex items-center justify-between">
  <span>{folder.name}</span>
  {folder.unreadCount > 0 && (
    <span className="inline-flex items-center justify-center px-2 py-0.5 text-xs font-semibold rounded-full bg-primary-600 text-white">
      {folder.unreadCount > 99 ? '99+' : folder.unreadCount}
    </span>
  )}
</div>
```

**Backend API Enhancement**:
```typescript
// backend/routes/folders.ts
router.get('/folders', async (req, res) => {
  const folders = await prisma.folder.findMany({
    where: { userId: req.user.id },
    include: {
      _count: {
        select: {
          emails: {
            where: { isRead: false }
          }
        }
      }
    }
  });

  const foldersWithCount = folders.map(f => ({
    ...f,
    unreadCount: f._count.emails
  }));

  res.json(foldersWithCount);
});
```

**Files to Modify**:
- `frontend/src/components/FolderList.tsx`
- `backend/src/routes/folders.ts`

---

## Performance Optimizations (Priority P1)

### 4. Query Stale Time Strategy 📊
**Status**: Using defaults
**Time**: 10 minutes
**Impact**: Medium - Reduces unnecessary API calls

**Implementation**:
```typescript
// App.tsx or main.tsx
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30000, // 30 seconds - data is fresh for 30s
      cacheTime: 5 * 60 * 1000, // 5 minutes - keep in cache
      retry: 1, // Retry failed requests once
      refetchOnWindowFocus: true, // Refresh on window focus
      refetchOnReconnect: true, // Refresh on reconnect
    },
    mutations: {
      retry: 0, // Don't retry mutations
    },
  },
});
```

**Files to Modify**:
- `frontend/src/App.tsx`

---

### 5. Memoize Callbacks 🧠
**Status**: Not implemented
**Time**: 20 minutes
**Impact**: Medium - Prevents unnecessary re-renders

**Implementation**:
```typescript
// DashboardPage.tsx
const handleBulkMarkRead = useCallback((emailIds: string[], isRead: boolean) => {
  bulkMarkReadMutation.mutate({ emailIds, isRead });
}, [bulkMarkReadMutation]);

const handleBulkStar = useCallback((emailIds: string[], isStarred: boolean) => {
  bulkStarMutation.mutate({ emailIds, isStarred });
}, [bulkStarMutation]);

const handleBulkDelete = useCallback((emailIds: string[]) => {
  bulkDeleteMutation.mutate(emailIds);
}, [bulkDeleteMutation]);

const handleReply = useCallback((email: Email) => {
  openCompose('reply', email);
}, []);

const handleReplyAll = useCallback((email: Email) => {
  openCompose('replyAll', email);
}, []);

const handleForward = useCallback((email: Email) => {
  openCompose('forward', email);
}, []);
```

**Files to Modify**:
- `frontend/src/pages/DashboardPage.tsx`

---

### 6. Debounced Folder Selection 🔄
**Status**: Not implemented
**Time**: 10 minutes
**Impact**: Low-Medium - Smoother navigation

**Prevents rapid folder switching from causing multiple API calls**:
```typescript
// DashboardPage.tsx
const [pendingFolderId, setPendingFolderId] = useState<string | null>(null);
const debouncedFolderId = useDebounce(pendingFolderId, 150);

useEffect(() => {
  setSelectedFolderId(debouncedFolderId);
}, [debouncedFolderId]);
```

**Files to Modify**:
- `frontend/src/pages/DashboardPage.tsx`

---

## Accessibility Improvements (Priority P1)

### 7. ARIA Labels & Roles 🦽
**Status**: Minimal
**Time**: 30 minutes
**Impact**: High - Accessibility compliance

**Add proper ARIA attributes**:
```typescript
// EmailList.tsx
<div role="list" aria-label="Email messages">
  {emails.map((email) => (
    <div
      key={email.id}
      role="listitem"
      aria-label={`Email from ${email.from.name || email.from.address}, subject: ${email.subject}`}
      aria-selected={selectedEmailId === email.id}
    >
      {/* email content */}
    </div>
  ))}
</div>

// FolderList.tsx
<nav role="navigation" aria-label="Folder navigation">
  {folders.map((folder) => (
    <button
      role="link"
      aria-current={selectedFolderId === folder.id ? 'page' : undefined}
      aria-label={`${folder.name} folder${folder.unreadCount ? `, ${folder.unreadCount} unread` : ''}`}
    >
      {folder.name}
    </button>
  ))}
</nav>

// ComposeEmail.tsx
<form aria-label="Compose email form">
  <label htmlFor="email-to" className="sr-only">To</label>
  <input id="email-to" aria-required="true" />

  <label htmlFor="email-subject" className="sr-only">Subject</label>
  <input id="email-subject" aria-required="true" />
</form>
```

**Files to Modify**:
- `frontend/src/components/EmailList.tsx`
- `frontend/src/components/FolderList.tsx`
- `frontend/src/components/ComposeEmail.tsx`
- `frontend/src/components/EmailView.tsx`

---

### 8. Focus Management 🎯
**Status**: Basic
**Time**: 20 minutes
**Impact**: Medium - Better keyboard navigation

**Improvements**:
- Focus first email when opening folder
- Focus compose form when opening
- Return focus after modal close
- Visible focus indicators

**Implementation**:
```typescript
// EmailList.tsx
const firstEmailRef = useRef<HTMLDivElement>(null);

useEffect(() => {
  if (emails.length > 0 && !selectedEmailId) {
    firstEmailRef.current?.focus();
  }
}, [emails, selectedEmailId]);

// Add visible focus styles
.focus-visible:focus {
  outline: 2px solid var(--primary-600);
  outline-offset: 2px;
}
```

**Files to Modify**:
- `frontend/src/components/EmailList.tsx`
- `frontend/src/components/ComposeEmail.tsx`
- `frontend/src/index.css`

---

## UX Enhancements (Priority P2)

### 9. Loading Skeleton for Email View 💀
**Status**: Email list only
**Time**: 15 minutes
**Impact**: Low-Medium - Better perceived performance

**Implementation**:
```typescript
// Skeleton.tsx
export function EmailViewSkeleton() {
  return (
    <div className="p-6 animate-pulse">
      <div className="h-10 bg-gray-200 dark:bg-gray-700 rounded mb-4 w-3/4" />
      <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded mb-2 w-1/2" />
      <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded mb-6 w-1/3" />
      <div className="space-y-3">
        <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded" />
        <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded" />
        <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-5/6" />
      </div>
    </div>
  );
}

// EmailView.tsx
if (isLoading) {
  return <EmailViewSkeleton />;
}
```

**Files to Modify**:
- `frontend/src/components/Skeleton.tsx`
- `frontend/src/components/EmailView.tsx`

---

### 10. Email Action Confirmations 💬
**Status**: Delete only
**Time**: 15 minutes
**Impact**: Low - Prevents mistakes

**Add confirmations for**:
- Mark all as read (if > 10 emails)
- Star all (if > 10 emails)
- Bulk delete (already implemented)

**Implementation**:
```typescript
const handleBulkMarkRead = () => {
  const count = selectedEmailIds.size;
  if (count > 10) {
    if (!confirm(`Mark ${count} emails as read?`)) return;
  }
  onBulkMarkRead?.(Array.from(selectedEmailIds), true);
  setSelectedEmailIds(new Set());
};
```

**Files to Modify**:
- `frontend/src/components/EmailList.tsx`

---

### 11. Responsive Email Count Display 📱
**Status**: Not implemented
**Time**: 10 minutes
**Impact**: Low - Better UX

**Show email count in header**:
```typescript
// EmailList.tsx
<div className="px-3 py-2 border-b border-gray-200 dark:border-gray-700">
  <div className="flex items-center justify-between">
    <span className="text-sm text-gray-600 dark:text-gray-400">
      {emails.length} email{emails.length !== 1 ? 's' : ''}
      {filterType !== 'all' && ` (filtered)`}
    </span>
  </div>
</div>
```

**Files to Modify**:
- `frontend/src/components/EmailList.tsx`

---

### 12. Improved Compose Email UX ✍️
**Status**: Basic
**Time**: 25 minutes
**Impact**: Medium - Better composing experience

**Enhancements**:
- Character counter for subject (recommended < 60 chars)
- Send button disabled if no recipient
- Tab navigation between fields
- Cmd/Ctrl+Enter to send

**Implementation**:
```typescript
// ComposeEmail.tsx
const isValid = formData.to.trim() !== '' && formData.subject.trim() !== '';

<div className="relative">
  <input
    value={formData.subject}
    onChange={(e) => setFormData({ ...formData, subject: e.target.value })}
    maxLength={100}
  />
  <span className={`text-xs ${formData.subject.length > 60 ? 'text-yellow-600' : 'text-gray-400'}`}>
    {formData.subject.length}/100 {formData.subject.length > 60 && '(recommend < 60)'}
  </span>
</div>

<button
  onClick={handleSend}
  disabled={!isValid || sendMutation.isPending}
  className="btn btn-primary"
>
  {sendMutation.isPending ? 'Sending...' : 'Send'}
</button>

// Keyboard shortcut
useEffect(() => {
  const handleKeyDown = (e: KeyboardEvent) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      if (isValid) handleSend();
    }
  };
  window.addEventListener('keydown', handleKeyDown);
  return () => window.removeEventListener('keydown', handleKeyDown);
}, [isValid]);
```

**Files to Modify**:
- `frontend/src/components/ComposeEmail.tsx`

---

## Code Quality (Priority P2)

### 13. Add PropTypes Comments 📝
**Status**: Minimal
**Time**: 30 minutes
**Impact**: Low - Better documentation

**Add JSDoc comments to all components**:
```typescript
/**
 * BulkActionsToolbar - Displays bulk operation controls when emails are selected
 *
 * @param selectedCount - Number of emails currently selected
 * @param isPending - Whether a bulk operation is in progress
 * @param onMarkRead - Callback to mark selected emails as read
 * @param onMarkUnread - Callback to mark selected emails as unread
 * @param onStar - Callback to star selected emails
 * @param onUnstar - Callback to unstar selected emails
 * @param onDelete - Callback to delete selected emails
 * @param onDeselectAll - Callback to clear selection
 */
export default function BulkActionsToolbar({ ... }: BulkActionsToolbarProps) {
  // ...
}
```

**Files to Modify**:
- All component files

---

### 14. Environment Variables 🔐
**Status**: Hardcoded
**Time**: 15 minutes
**Impact**: Medium - Better configuration

**Move to .env files**:
```bash
# .env
VITE_API_URL=http://localhost:3000
VITE_WS_URL=ws://localhost:3000
VITE_APP_NAME=Mycelix Mail
```

```typescript
// services/api.ts
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3000';
```

**Files to Create/Modify**:
- `.env.example`
- `frontend/src/services/api.ts`
- `frontend/src/hooks/useWebSocket.ts`

---

### 15. Error Logging Service 📋
**Status**: Console only
**Time**: 20 minutes
**Impact**: Low - Better debugging

**Centralized error logging**:
```typescript
// services/logger.ts
export const logger = {
  error: (message: string, error?: Error, context?: Record<string, any>) => {
    console.error(`[ERROR] ${message}`, { error, context });
    // Could send to Sentry, LogRocket, etc.
  },
  warn: (message: string, context?: Record<string, any>) => {
    console.warn(`[WARN] ${message}`, context);
  },
  info: (message: string, context?: Record<string, any>) => {
    console.info(`[INFO] ${message}`, context);
  },
};
```

**Files to Create**:
- `frontend/src/services/logger.ts`

**Files to Modify**:
- `frontend/src/components/ErrorBoundary.tsx`
- All mutation error handlers

---

## Implementation Order (This Session)

### Critical Path (Must Do - 90 min):
1. **Keyboard Shortcuts for Bulk Operations** (25 min) - Power user feature
2. **Query Stale Time Strategy** (10 min) - Performance
3. **Memoize Callbacks** (20 min) - Performance
4. **Email Draft Autosave** (30 min) - Data loss prevention
5. **ARIA Labels** (30 min) - Accessibility

### High Value (Should Do - 60 min):
6. **Folder Email Count Badges** (20 min) - Information density
7. **Improved Compose UX** (25 min) - Better experience
8. **Loading Skeleton for EmailView** (15 min) - Perceived performance

### Optional (Nice to Have - 45 min):
9. **Focus Management** (20 min) - Keyboard navigation
10. **Email Action Confirmations** (15 min) - Error prevention
11. **Environment Variables** (15 min) - Configuration

**Total Critical**: ~90 minutes
**Total with High Value**: ~150 minutes
**Total with Optional**: ~195 minutes

---

## Success Criteria

### Phase 6 Complete When:
- ✅ Keyboard shortcuts for all bulk operations
- ✅ Email drafts auto-save and restore
- ✅ Folder badges show unread counts
- ✅ Query client optimized with stale time
- ✅ Callbacks properly memoized
- ✅ ARIA labels throughout application
- ✅ Compose form has character counter and validation
- ✅ EmailView has loading skeleton
- ✅ Focus management implemented
- ✅ All confirmations in place

### Quality Checklist:
- [ ] Keyboard shortcuts documented in help modal
- [ ] No console errors or warnings
- [ ] All mutations have loading states
- [ ] Draft autosave tested
- [ ] Accessibility tested with screen reader
- [ ] Performance profiled (no slow renders)
- [ ] All callbacks memoized
- [ ] ARIA labels complete

---

## Application Maturity Target

**Current**: 95% production-ready
**After Critical Path**: 96% production-ready
**After High Value**: 97% production-ready
**After Optional**: 98% production-ready

---

## Future Phases Preview

**Phase 7**: Testing Infrastructure
- Unit tests with Vitest
- Component tests with Testing Library
- E2E tests with Playwright
- Test coverage reporting

**Phase 8**: Deployment & DevOps
- Docker containerization
- CI/CD with GitHub Actions
- Production environment setup
- Monitoring and analytics

**Phase 9**: Advanced Features
- Email threading/conversations
- Advanced search with filters
- Attachment previews
- Contact management
- Email templates
- Labels/tags

**Phase 10**: Final Polish & Launch
- Mobile responsiveness
- PWA capabilities
- Performance audit
- Security audit
- Launch checklist

---

## Notes

- Focus on power user features that make the app feel professional
- Performance optimizations should be measurable
- Accessibility is non-negotiable for production
- Every feature should enhance, not complicate
- Keep code quality high throughout

---

**End of Phase 6 Plan**
