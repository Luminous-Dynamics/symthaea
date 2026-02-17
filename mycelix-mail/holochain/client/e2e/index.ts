/**
 * E2E Test Suite for Mycelix Mail Holochain Client
 *
 * Comprehensive end-to-end testing covering:
 * - Messaging flows (compose, send, receive, reply, forward)
 * - Trust system (MATL attestations, network visualization)
 * - Contacts (CRUD, groups, vCard import/export)
 * - Onboarding (key generation, profile setup, backup)
 * - Accessibility (WCAG compliance, keyboard navigation)
 *
 * Usage:
 *   npx playwright test                    # Run all tests
 *   npx playwright test messaging.spec.ts  # Run specific test file
 *   npx playwright test --headed           # Run with visible browser
 *   npx playwright test --ui               # Run with Playwright UI
 *   npx playwright show-report             # View test report
 *
 * Configuration:
 *   See playwright.config.ts for browser and test settings
 *
 * Global Setup:
 *   Tests automatically start/stop Holochain conductor
 *   Set USE_EXISTING_CONDUCTOR=true to use an existing conductor
 */

export * from './playwright.config';
