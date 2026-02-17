/**
 * Events Index
 *
 * Central export for the event system.
 */

export {
  TypedEventEmitter,
  getEventEmitter,
  resetEventEmitter,
} from './emitter';

export type { AppEvents, EventHandler } from './emitter';

export {
  registerCacheHandlers,
  registerLoggingHandlers,
  registerMilestoneHandlers,
  registerWebSocketHandlers,
  registerAllHandlers,
} from './handlers';
