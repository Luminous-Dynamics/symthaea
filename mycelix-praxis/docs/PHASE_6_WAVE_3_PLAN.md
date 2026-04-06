# Phase 6 Wave 3: Optimization, Accessibility & Developer Experience

## Overview
Focus on production-grade optimization, accessibility compliance, and enhanced developer experience.

## Goals
1. ✅ Optimize React component rendering with memoization
2. ✅ Implement comprehensive accessibility features (WCAG 2.1 AA)
3. ✅ Add API documentation with rustdoc
4. ✅ Enhance error handling and user feedback
5. ✅ Improve developer experience with better tooling

## Priority Matrix

### Priority 1: Must Have (Critical Path) - ~3 hours
- **React Performance Optimization** (1h)
  - Add React.memo to CourseCard, RoundCard, CredentialCard
  - useMemo for expensive filters/sorts in pages
  - useCallback for event handlers passed as props
  - Measure impact with React DevTools Profiler

- **Accessibility Compliance** (2h)
  - ARIA labels on all interactive elements
  - Keyboard navigation (Tab, Enter, Escape)
  - Focus management for modals and navigation
  - Semantic HTML improvements
  - Color contrast validation (WCAG AA)
  - Skip navigation link

### Priority 2: Should Have (High Value) - ~2.5 hours
- **API Documentation** (1h)
  - rustdoc documentation for all public APIs
  - Code examples in doc comments
  - Module-level documentation
  - Generate and verify HTML docs

- **Enhanced Error Handling** (1h)
  - User-friendly error messages
  - Error recovery suggestions
  - Retry mechanisms for failed operations
  - Toast notifications for errors/success
  - Better loading states with skeletons

- **Developer Experience** (0.5h)
  - Add helpful code comments
  - Type safety improvements
  - Better console logging in dev mode

### Priority 3: Nice to Have (Polish) - ~1.5 hours
- **Testing Infrastructure** (1h)
  - Unit tests for critical components
  - Integration tests for key workflows
  - Test coverage reporting

- **Visual Enhancements** (0.5h)
  - Empty states for lists
  - Better error boundary UI
  - Improved loading animations

## Success Metrics
- ✅ All interactive elements keyboard accessible
- ✅ WCAG 2.1 AA compliance verified
- ✅ 30%+ reduction in unnecessary re-renders
- ✅ 100% rustdoc coverage for public APIs
- ✅ User-friendly error messages throughout
- ✅ Test coverage >60% for critical paths

## Implementation Plan

### Phase 1: React Performance (Priority 1)
1. Analyze components with React DevTools Profiler
2. Add React.memo to card components
3. Add useMemo for filters/sorts
4. Add useCallback for event handlers
5. Verify performance improvements

### Phase 2: Accessibility (Priority 1)
1. Audit current accessibility state
2. Add ARIA labels to all buttons/links
3. Implement keyboard navigation
4. Add focus management
5. Validate color contrast
6. Add skip navigation link
7. Test with screen reader

### Phase 3: Documentation & DX (Priority 2)
1. Add rustdoc comments to all public types/functions
2. Generate and review HTML docs
3. Enhance error messages
4. Add toast notification system
5. Improve loading states

### Phase 4: Testing & Polish (Priority 3)
1. Add unit tests for components
2. Add integration tests
3. Create empty state components
4. Enhance error boundary UI

## Detailed Task Breakdown

### React Performance Optimization
**Files to modify:**
- `apps/web/src/components/CourseCard.tsx` - Add React.memo
- `apps/web/src/components/RoundCard.tsx` - Add React.memo
- `apps/web/src/components/CredentialCard.tsx` - Add React.memo
- `apps/web/src/pages/CoursesPage.tsx` - useMemo for filtering
- `apps/web/src/pages/FLRoundsPage.tsx` - useMemo for filtering
- `apps/web/src/pages/CredentialsPage.tsx` - useMemo for filtering

**Code patterns:**
```typescript
// Memoized component
export const CourseCard = React.memo(({ course, onClick }: Props) => {
  // Component implementation
});

// Memoized computation
const filteredCourses = useMemo(() =>
  courses.filter(c => c.category === selectedCategory),
  [courses, selectedCategory]
);

// Memoized callback
const handleClick = useCallback((id: string) => {
  console.log('Clicked:', id);
}, []);
```

### Accessibility Features
**Files to modify:**
- `apps/web/src/App.tsx` - Skip navigation link
- `apps/web/src/components/CourseCard.tsx` - ARIA labels
- `apps/web/src/components/RoundCard.tsx` - ARIA labels
- `apps/web/src/components/CredentialCard.tsx` - ARIA labels
- `apps/web/src/pages/CoursesPage.tsx` - Keyboard navigation
- `apps/web/src/pages/FLRoundsPage.tsx` - Keyboard navigation

**Accessibility checklist:**
- [ ] All images have alt text
- [ ] All buttons have aria-label or visible text
- [ ] All form inputs have labels
- [ ] Tab navigation works correctly
- [ ] Enter/Space activate buttons
- [ ] Escape closes modals
- [ ] Focus visible on all interactive elements
- [ ] Color contrast ≥4.5:1 for text
- [ ] Skip navigation link present
- [ ] ARIA landmarks used correctly

### API Documentation
**Files to document:**
- `crates/praxis-agg/src/lib.rs` - Module overview
- `crates/praxis-agg/src/aggregation.rs` - Public API
- `crates/praxis-agg/src/privacy.rs` - Public API
- `crates/praxis-agg/src/validation.rs` - Public API

**Documentation template:**
```rust
/// Aggregates model updates from multiple participants using the specified method.
///
/// This function implements privacy-preserving aggregation techniques for federated learning,
/// supporting multiple aggregation strategies and differential privacy mechanisms.
///
/// # Arguments
///
/// * `updates` - A vector of model updates from participants
/// * `method` - The aggregation method to use (e.g., mean, median, trimmed_mean)
/// * `privacy_params` - Optional differential privacy parameters
///
/// # Returns
///
/// Returns the aggregated model update with privacy guarantees applied.
///
/// # Examples
///
/// ```
/// use praxis_agg::{aggregate_updates, AggregationMethod};
///
/// let updates = vec![update1, update2, update3];
/// let result = aggregate_updates(updates, AggregationMethod::Mean, None)?;
/// ```
///
/// # Errors
///
/// Returns an error if:
/// - The updates vector is empty
/// - Privacy parameters are invalid
/// - Aggregation fails due to incompatible update shapes
pub fn aggregate_updates(...) -> Result<Update, AggregationError> {
    // Implementation
}
```

### Error Handling & UX
**Files to create:**
- `apps/web/src/components/Toast.tsx` - Toast notification system
- `apps/web/src/components/EmptyState.tsx` - Empty state component
- `apps/web/src/hooks/useToast.ts` - Toast hook

**Files to modify:**
- `apps/web/src/components/ErrorBoundary.tsx` - Better UI
- `apps/web/src/pages/CoursesPage.tsx` - Empty states
- `apps/web/src/pages/FLRoundsPage.tsx` - Empty states

## Timeline
- **Phase 1** (React Performance): 1 hour
- **Phase 2** (Accessibility): 2 hours
- **Phase 3** (Documentation & DX): 1.5 hours
- **Phase 4** (Testing & Polish): 1.5 hours
- **Total**: ~6 hours (executing Phases 1-3 now)

## Out of Scope (Future Waves)
- E2E testing with Playwright/Cypress
- Visual regression testing
- Internationalization (i18n)
- Advanced analytics
- Performance monitoring (Sentry, DataDog)
- CDN deployment optimization

## Notes
- Use React DevTools Profiler to measure performance improvements
- Test accessibility with screen readers (NVDA/JAWS/VoiceOver)
- Validate WCAG compliance with axe DevTools
- Generate rustdoc with `cargo doc --no-deps --open`
