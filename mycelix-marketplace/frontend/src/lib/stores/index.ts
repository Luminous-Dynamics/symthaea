/**
 * Central Store Exports
 *
 * Import all stores from this single location:
 * import { cartItems, auth, notifications } from '$lib/stores';
 */

// Cart stores
export {
  cartItems,
  itemCount,
  subtotal,
  tax,
  shipping,
  total,
  cartState,
} from './cart';

// Auth stores
export {
  auth,
  isAuthenticated,
  currentUser,
  userAgentId,
  userRoles,
  isArbitrator,
  isAdmin,
} from './auth';

// Holochain connection stores
export {
  holochain,
  isConnected,
  isConnecting,
  hasConnectionError,
  client,
  connectionError,
} from './holochain';

// Notifications stores
export {
  notifications,
  notificationCount,
  hasNotifications,
  notificationsByType,
} from './notifications';

// Favorites
export { favorites, favoritesSet, favoritesCount } from './favorites';

// Proof requests
export {
  proofRequests,
  markProofRequest,
  getProofKey,
  getProofStatus,
  markProofFulfilled,
  markProofDenied,
} from './proofs';

// Re-export types
export type { ConnectionStatus, HolochainState } from './holochain';
export type { NotificationType, Notification } from './notifications';
