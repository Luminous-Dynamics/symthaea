/**
 * Mycelix Mail - Vue Integration
 *
 * Complete Vue 3 support with composables for reactive state management.
 */

// Composables
export * from './composables';

// Re-export client for convenience
export { getMycelixClient, createMycelixClient } from '../bootstrap';
