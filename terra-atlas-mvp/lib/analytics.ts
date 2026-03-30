// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Analytics utility for tracking user interactions
// This is a placeholder that can be connected to any analytics service

type EventProperties = Record<string, string | number | boolean | undefined>

interface AnalyticsEvent {
  name: string
  properties?: EventProperties
  timestamp: string
}

// Store events in memory (in production, send to analytics service)
const eventQueue: AnalyticsEvent[] = []

/**
 * Track an analytics event
 */
export function trackEvent(name: string, properties?: EventProperties): void {
  const event: AnalyticsEvent = {
    name,
    properties,
    timestamp: new Date().toISOString(),
  }

  eventQueue.push(event)

  // Log in development
  if (process.env.NODE_ENV === 'development') {
    console.log('[Analytics]', name, properties)
  }

  // In production, you would send to your analytics service:
  // - Google Analytics 4
  // - Mixpanel
  // - Amplitude
  // - PostHog
  // - etc.
}

/**
 * Track a page view
 */
export function trackPageView(path: string, title?: string): void {
  trackEvent('page_view', { path, title })
}

/**
 * Track a form submission
 */
export function trackFormSubmission(formName: string, success: boolean): void {
  trackEvent('form_submission', { form_name: formName, success })
}

/**
 * Track a button click
 */
export function trackClick(elementName: string, context?: string): void {
  trackEvent('click', { element: elementName, context })
}

/**
 * Track investment-related events
 */
export function trackInvestmentEvent(
  action: 'view' | 'start' | 'complete' | 'cancel',
  projectId: string,
  amount?: number
): void {
  trackEvent('investment', {
    action,
    project_id: projectId,
    amount,
  })
}

/**
 * ✨ Tier 2 Feature Tracking - Investment Scorecard
 */
export function trackScorecardView(siteId: string, siteName: string, siteType?: string): void {
  trackEvent('scorecard_viewed', {
    site_id: siteId,
    site_name: siteName,
    site_type: siteType,
  })
}

export function trackScorecardMetric(metric: string, value?: number): void {
  trackEvent('scorecard_metric_viewed', { metric, value })
}

/**
 * ✨ Tier 2 Feature Tracking - Regional Comparison
 */
export function trackComparisonMode(opened: boolean): void {
  trackEvent(opened ? 'comparison_mode_opened' : 'comparison_mode_closed')
}

export function trackStateComparison(states: string[]): void {
  trackEvent('states_compared', {
    states: states.join(','),
    count: states.length,
  })
}

export function trackStateSelected(state: string, action: 'add' | 'remove'): void {
  trackEvent('state_selected', { state, action })
}

/**
 * ✨ Tier 2 Feature Tracking - Timeline Projections
 */
export function trackTimelineMode(opened: boolean): void {
  trackEvent(opened ? 'timeline_mode_opened' : 'timeline_mode_closed')
}

export function trackTimelineYear(year: number): void {
  trackEvent('timeline_year_changed', { year })
}

export function trackTimelineAnimation(action: 'play' | 'pause' | 'reset'): void {
  trackEvent('timeline_animation', { action })
}

/**
 * ✨ Performance Tracking
 */
export function trackPerformance(metric: string, value: number, unit: string = 'ms'): void {
  trackEvent('performance_metric', {
    metric,
    value,
    unit,
  })
}

/**
 * Get all queued events (for debugging)
 */
export function getEventQueue(): AnalyticsEvent[] {
  return [...eventQueue]
}

/**
 * Clear the event queue
 */
export function clearEventQueue(): void {
  eventQueue.length = 0
}

export default {
  trackEvent,
  trackPageView,
  trackFormSubmission,
  trackClick,
  trackInvestmentEvent,
  // ✨ Tier 2 Tracking
  trackScorecardView,
  trackScorecardMetric,
  trackComparisonMode,
  trackStateComparison,
  trackStateSelected,
  trackTimelineMode,
  trackTimelineYear,
  trackTimelineAnimation,
  trackPerformance,
  // Debug
  getEventQueue,
  clearEventQueue,
}
