# Phase 6 Wave 4: Testing, Error Handling & Production Polish

## Overview
Final production readiness wave focusing on comprehensive testing, user-friendly error handling, and visual polish.

## Goals
1. ✅ Implement comprehensive testing infrastructure
2. ✅ Create toast notification system for better UX
3. ✅ Add empty states with call-to-action
4. ✅ Enhance error boundary with better UI
5. ✅ Add Rust unit tests for critical paths
6. ✅ Improve type safety and code quality

## Priority Matrix

### Priority 1: Testing Infrastructure (Critical) - ~3 hours
**React Component Tests** (1.5h)
- Setup Vitest for React testing
- Test utilities and custom render function
- Unit tests for card components
  - CourseCard: rendering, onClick, keyboard nav
  - RoundCard: state colors, progress calculation
  - CredentialCard: score display, verification badge
- Integration tests for pages
  - CoursesPage: filtering, sorting, search
  - FLRoundsPage: tab navigation, filtering
  - CredentialsPage: stats calculation

**Rust Tests Enhancement** (1h)
- Additional edge case tests for aggregation
- Property-based tests with quickcheck
- Benchmark verification tests
- Error handling coverage

**Test Infrastructure** (0.5h)
- Test coverage reporting
- CI integration for tests
- Test documentation

### Priority 2: User Experience Enhancements (High Value) - ~2.5 hours
**Toast Notification System** (1h)
- Create Toast component with variants (success, error, info, warning)
- Toast context provider with queue management
- useToast hook for easy usage
- Auto-dismiss with configurable duration
- Accessibility (ARIA live regions)
- Animations (slide-in, fade-out)

**Empty States** (1h)
- EmptyState component with icon, title, description, CTA
- Empty states for all pages:
  - CoursesPage: No courses match filters
  - FLRoundsPage: No rounds in selected tab
  - CredentialsPage: No credentials yet
- Actionable prompts (clear filters, enroll in courses, etc.)

**Error Boundary Enhancement** (0.5h)
- Better visual design for error boundary
- Error details in development mode
- Retry button
- Report issue link

### Priority 3: Code Quality & Type Safety (Polish) - ~1.5 hours
**Type Safety Improvements** (0.5h)
- Add missing type annotations
- Stricter TypeScript config
- Type guards for runtime validation
- Better type inference

**Code Quality** (0.5h)
- Add JSDoc comments for complex functions
- Improve variable naming
- Extract magic numbers to constants
- Better error messages

**Development Experience** (0.5h)
- Better dev scripts in package.json
- Environment variable documentation
- Local development guide
- Troubleshooting section in README

## Success Metrics
- ✅ Test coverage >70% for critical components
- ✅ All Rust tests pass including new edge cases
- ✅ Toast system working with all notification types
- ✅ Empty states on all pages with CTAs
- ✅ Error boundary shows helpful recovery options
- ✅ Zero TypeScript errors in strict mode
- ✅ All builds pass (web + rust + tests)

## Implementation Plan

### Phase 1: Testing Infrastructure
1. Setup Vitest and React Testing Library
2. Create test utilities (custom render, mock providers)
3. Write component tests (CourseCard, RoundCard, CredentialCard)
4. Write page integration tests
5. Enhance Rust tests with edge cases
6. Configure coverage reporting

### Phase 2: Toast Notification System
1. Create Toast component with variants
2. Create ToastContext and provider
3. Create useToast hook
4. Integrate into App.tsx
5. Add toast notifications to error handlers
6. Add success notifications for actions

### Phase 3: Empty States & Error Boundary
1. Create EmptyState component
2. Add empty states to all pages
3. Enhance ErrorBoundary component
4. Add retry and reporting functionality
5. Test error scenarios

### Phase 4: Type Safety & Code Quality
1. Enable stricter TypeScript settings
2. Add type guards where needed
3. Add JSDoc comments
4. Extract constants
5. Improve error messages
6. Update documentation

## Detailed Task Breakdown

### React Testing Setup
**Files to create:**
- `apps/web/vitest.config.ts` - Vitest configuration
- `apps/web/src/test/setup.ts` - Test setup and globals
- `apps/web/src/test/utils.tsx` - Custom render, mock providers
- `apps/web/src/components/__tests__/CourseCard.test.tsx`
- `apps/web/src/components/__tests__/RoundCard.test.tsx`
- `apps/web/src/components/__tests__/CredentialCard.test.tsx`
- `apps/web/src/pages/__tests__/CoursesPage.test.tsx`

**Test patterns:**
```typescript
import { render, screen, fireEvent } from '@testing-library/react';
import { CourseCard } from '../CourseCard';

describe('CourseCard', () => {
  it('renders course information', () => {
    const course = mockCourses[0];
    render(<CourseCard course={course} />);

    expect(screen.getByText(course.title)).toBeInTheDocument();
    expect(screen.getByText(course.instructor)).toBeInTheDocument();
  });

  it('handles click events', () => {
    const handleClick = vi.fn();
    render(<CourseCard course={mockCourses[0]} onClick={handleClick} />);

    fireEvent.click(screen.getByRole('button'));
    expect(handleClick).toHaveBeenCalledWith(mockCourses[0]);
  });

  it('handles keyboard navigation', () => {
    const handleClick = vi.fn();
    render(<CourseCard course={mockCourses[0]} onClick={handleClick} />);

    const card = screen.getByRole('button');
    fireEvent.keyDown(card, { key: 'Enter' });
    expect(handleClick).toHaveBeenCalled();
  });
});
```

### Toast Notification System
**Files to create:**
- `apps/web/src/components/Toast.tsx` - Toast component
- `apps/web/src/contexts/ToastContext.tsx` - Context and provider
- `apps/web/src/hooks/useToast.ts` - Hook for using toasts
- `apps/web/src/styles/toast.css` - Toast animations

**Toast API:**
```typescript
const { toast } = useToast();

// Success
toast.success('Course enrolled successfully!');

// Error
toast.error('Failed to load courses. Please try again.');

// Info
toast.info('New FL round available');

// Warning
toast.warning('Connection lost. Attempting to reconnect...');

// Custom duration
toast.success('Saved!', { duration: 2000 });
```

### Empty States
**Files to create:**
- `apps/web/src/components/EmptyState.tsx` - Reusable empty state component

**Empty state patterns:**
```typescript
<EmptyState
  icon="📚"
  title="No courses found"
  description="Try adjusting your search or filters to find courses"
  action={{
    label: "Clear filters",
    onClick: handleClearFilters
  }}
/>
```

### Rust Testing Enhancements
**Additional tests:**
- Edge cases: empty vectors, single element
- Boundary conditions: trim_percent at 0.0 and 0.5
- Large datasets: performance with 1000+ updates
- NaN and infinity handling
- Concurrent access patterns

## Timeline
- **Phase 1** (Testing): 3 hours
- **Phase 2** (Toast System): 1 hour
- **Phase 3** (Empty States): 1 hour
- **Phase 4** (Type Safety): 1.5 hours
- **Total**: ~6.5 hours (executing Phases 1-3 now)

## Out of Scope (Future)
- E2E testing with Playwright
- Visual regression testing
- Performance profiling
- Internationalization (i18n)
- Advanced error tracking (Sentry)
- A/B testing infrastructure

## Notes
- Focus on high-value tests (critical user paths)
- Toast system should be accessible (ARIA live regions)
- Empty states should guide users to next actions
- Maintain build performance with lazy test imports
