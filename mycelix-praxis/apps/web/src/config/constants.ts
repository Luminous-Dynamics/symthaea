// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Application Constants
 *
 * Centralized configuration and constants for the Praxis application
 */

// ==================== API Configuration ====================
export const API_CONFIG = {
  BASE_URL: import.meta.env.VITE_API_URL || 'http://localhost:3000',
  TIMEOUT: 30000, // 30 seconds
  RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 2000, // 2 seconds
} as const;

// ==================== Toast Configuration ====================
export const TOAST_CONFIG = {
  DEFAULT_DURATION: 5000, // 5 seconds
  ERROR_DURATION: 7000, // 7 seconds
  SUCCESS_DURATION: 4000, // 4 seconds
  INFO_DURATION: 5000, // 5 seconds
  WARNING_DURATION: 6000, // 6 seconds
  MAX_TOASTS: 5,
} as const;

// ==================== Animation & Timing ====================
export const ANIMATION_TIMING = {
  LOADING_DELAY: {
    COURSES: 800, // ms
    FL_ROUNDS: 600, // ms
    CREDENTIALS: 500, // ms
  },
  DEBOUNCE_DELAY: 300, // ms for search inputs
  MODAL_TRANSITION: 200, // ms
} as const;

// ==================== Pagination & Limits ====================
export const PAGINATION = {
  COURSES_PER_PAGE: 12,
  ROUNDS_PER_PAGE: 10,
  CREDENTIALS_PER_PAGE: 9,
  MAX_SEARCH_RESULTS: 100,
} as const;

// ==================== Error Simulation ====================
export const ERROR_SIMULATION = {
  COURSES_ERROR_RATE: 0.05, // 5% chance
  FL_ROUNDS_ERROR_RATE: 0.05, // 5% chance
  CREDENTIALS_ERROR_RATE: 0.05, // 5% chance
} as const;

// ==================== Course Filters ====================
export const COURSE_FILTERS = {
  DIFFICULTIES: ['all', 'beginner', 'intermediate', 'advanced'] as const,
  SORT_OPTIONS: ['popular', 'recent', 'title'] as const,
} as const;

// ==================== FL Round States ====================
export const FL_ROUND_STATES = {
  ACTIVE: ['JOIN', 'ASSIGN', 'UPDATE', 'AGGREGATE'] as const,
  COMPLETED: 'COMPLETED',
  ALL_STATES: ['JOIN', 'ASSIGN', 'UPDATE', 'AGGREGATE', 'COMPLETED'] as const,
} as const;

// ==================== Privacy & Security ====================
export const PRIVACY_CONFIG = {
  MIN_EPSILON: 0.1,
  MAX_EPSILON: 10.0,
  MIN_DELTA: 1e-10,
  MAX_DELTA: 1e-3,
  DEFAULT_CLIP_NORM: 1.0,
} as const;

// ==================== UI Constants ====================
export const UI_CONSTANTS = {
  MAX_WIDTH: '1200px',
  CARD_MIN_WIDTH: {
    COURSE: '300px',
    FL_ROUND: '350px',
    CREDENTIAL: '320px',
  },
  GRID_GAP: '20px',
  BORDER_RADIUS: {
    SMALL: '6px',
    MEDIUM: '8px',
    LARGE: '12px',
    XLARGE: '16px',
  },
  Z_INDEX: {
    MODAL: 1000,
    TOAST: 2000,
    TOOLTIP: 3000,
  },
} as const;

// ==================== Error Messages ====================
export const ERROR_MESSAGES = {
  GENERIC: 'An unexpected error occurred. Please try again.',
  NETWORK: 'Network error. Please check your connection.',
  TIMEOUT: 'Request timed out. Please try again.',
  NOT_FOUND: 'The requested resource was not found.',
  UNAUTHORIZED: 'You are not authorized to perform this action.',
  FORBIDDEN: 'Access to this resource is forbidden.',
  SERVER_ERROR: 'Server error. Please try again later.',

  // Specific errors
  COURSES_LOAD_FAILED: 'Failed to load courses. Please try again.',
  FL_ROUNDS_LOAD_FAILED: 'Failed to load FL rounds. Please try again.',
  CREDENTIALS_LOAD_FAILED: 'Failed to load credentials. Please try again.',
  CREDENTIAL_VERIFY_FAILED: 'Failed to verify credential.',
  CREDENTIAL_COPY_FAILED: 'Failed to copy to clipboard',

  // Validation errors
  INVALID_INPUT: 'Invalid input. Please check your data.',
  REQUIRED_FIELD: 'This field is required.',
} as const;

// ==================== Success Messages ====================
export const SUCCESS_MESSAGES = {
  COURSES_LOADED: (count: number) => `Loaded ${count} courses successfully!`,
  FL_ROUNDS_LOADED: (count: number) => `Loaded ${count} FL rounds successfully!`,
  CREDENTIALS_LOADED: (count: number) => `Loaded ${count} credential${count !== 1 ? 's' : ''} successfully!`,

  CREDENTIAL_VERIFIED: 'Credential verified successfully!',
  CREDENTIAL_COPIED: 'Credential JSON copied to clipboard!',
  CREDENTIAL_DOWNLOADED: 'Credential downloaded successfully!',

  ROUND_JOINED: (roundId: string) => `Successfully joined round: ${roundId}!`,

  FILTERS_CLEARED: 'Filters cleared',
} as const;

// ==================== Info Messages ====================
export const INFO_MESSAGES = {
  VIEWING_COURSE: (title: string) => `Viewing: ${title}`,
  VIEWING_ROUND: (roundId: string) => `Viewing round: ${roundId}`,
  VIEWING_CREDENTIAL: 'Viewing credential details',

  SHOWING_ALL_COURSES: 'Showing all courses',
  SHOWING_ALL_ROUNDS: 'Showing all rounds',
  SHOWING_ACTIVE_ROUNDS: 'Showing active rounds',

  HOLOCHAIN_COMING_SOON: 'Holochain integration coming soon',
  ENROLLMENT_COMING_SOON: 'Enrollment coming soon!',
} as const;

// ==================== External Links ====================
export const EXTERNAL_LINKS = {
  GITHUB_REPO: 'https://github.com/Luminous-Dynamics/mycelix-praxis',
  GITHUB_ISSUES: 'https://github.com/Luminous-Dynamics/mycelix-praxis/issues',
  DOCUMENTATION: 'https://docs.mycelix-praxis.org',
} as const;

// ==================== Route Paths ====================
export const ROUTES = {
  HOME: '/',
  COURSES: '/courses',
  FL_ROUNDS: '/fl-rounds',
  CREDENTIALS: '/credentials',
} as const;

// ==================== Feature Flags ====================
export const FEATURE_FLAGS = {
  ENABLE_HOLOCHAIN: import.meta.env.VITE_ENABLE_HOLOCHAIN === 'true',
  ENABLE_MOCK_DATA: import.meta.env.VITE_ENABLE_MOCK_DATA !== 'false', // Default: true
  ENABLE_ERROR_SIMULATION: import.meta.env.VITE_ENABLE_ERROR_SIMULATION !== 'false', // Default: true
  ENABLE_ANALYTICS: import.meta.env.VITE_ENABLE_ANALYTICS === 'true',
} as const;

// ==================== Type Exports ====================
export type Difficulty = typeof COURSE_FILTERS.DIFFICULTIES[number];
export type SortOption = typeof COURSE_FILTERS.SORT_OPTIONS[number];
export type FlRoundState = typeof FL_ROUND_STATES.ALL_STATES[number];
export type RouteKey = keyof typeof ROUTES;
