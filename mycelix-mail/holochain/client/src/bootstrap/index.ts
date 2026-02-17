/**
 * Mycelix Mail - Bootstrap Module
 *
 * Unified client initialization and service container.
 */

export {
  MycelixClient,
  createMycelixClient,
  getMycelixClient,
  useMycelixClient,
} from './MycelixClient';

export type {
  MycelixClientConfig,
  ServiceContainer,
} from './MycelixClient';

// Re-export commonly used types
export type { SignalHub } from '../SignalHub';
