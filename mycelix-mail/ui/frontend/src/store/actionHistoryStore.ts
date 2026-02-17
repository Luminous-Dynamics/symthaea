import { create } from 'zustand';

export type ActionType = 'delete' | 'archive' | 'label' | 'unlabel' | 'mark_read' | 'mark_unread' | 'star' | 'unstar' | 'move';

export interface Action {
  id: string;
  type: ActionType;
  timestamp: string;
  emailIds: string[];
  previousState: any; // Store previous state for rollback
  undo: () => Promise<void>;
  redo?: () => Promise<void>;
  description: string; // "Deleted 3 emails", "Archived email from John", etc.
}

interface ActionHistoryStore {
  history: Action[];
  currentIndex: number; // For undo/redo navigation
  maxHistory: number; // Maximum number of actions to keep

  // Add action to history
  addAction: (action: Omit<Action, 'id' | 'timestamp'>) => void;

  // Undo last action
  undo: () => Promise<void>;

  // Redo previously undone action
  redo: () => Promise<void>;

  // Can undo/redo?
  canUndo: () => boolean;
  canRedo: () => boolean;

  // Get last action description
  getLastActionDescription: () => string | null;

  // Clear all history
  clearHistory: () => void;

  // Remove specific action
  removeAction: (actionId: string) => void;
}

export const useActionHistoryStore = create<ActionHistoryStore>((set, get) => ({
  history: [],
  currentIndex: -1,
  maxHistory: 10,

  addAction: (actionData) => {
    const action: Action = {
      ...actionData,
      id: `action_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date().toISOString(),
    };

    set((state) => {
      // Remove any actions after current index (when adding new action after undo)
      const newHistory = state.history.slice(0, state.currentIndex + 1);

      // Add new action
      newHistory.push(action);

      // Keep only maxHistory actions
      if (newHistory.length > state.maxHistory) {
        newHistory.shift();
      }

      return {
        history: newHistory,
        currentIndex: newHistory.length - 1,
      };
    });
  },

  undo: async () => {
    const { history, currentIndex, canUndo } = get();

    if (!canUndo()) {
      console.warn('Nothing to undo');
      return;
    }

    const action = history[currentIndex];

    try {
      // Execute undo function
      await action.undo();

      // Move index back
      set((state) => ({
        currentIndex: state.currentIndex - 1,
      }));
    } catch (error) {
      console.error('Failed to undo action:', error);
      throw error;
    }
  },

  redo: async () => {
    const { history, currentIndex, canRedo } = get();

    if (!canRedo()) {
      console.warn('Nothing to redo');
      return;
    }

    // Get the action to redo (one after current index)
    const action = history[currentIndex + 1];

    try {
      // If action has a redo function, use it; otherwise, execute the original operation again
      if (action.redo) {
        await action.redo();
      } else {
        // For actions without explicit redo, we can't redo
        console.warn('Action does not support redo');
        return;
      }

      // Move index forward
      set((state) => ({
        currentIndex: state.currentIndex + 1,
      }));
    } catch (error) {
      console.error('Failed to redo action:', error);
      throw error;
    }
  },

  canUndo: () => {
    const { currentIndex } = get();
    return currentIndex >= 0;
  },

  canRedo: () => {
    const { history, currentIndex } = get();
    return currentIndex < history.length - 1;
  },

  getLastActionDescription: () => {
    const { history, currentIndex } = get();
    if (currentIndex >= 0 && currentIndex < history.length) {
      return history[currentIndex].description;
    }
    return null;
  },

  clearHistory: () => {
    set({
      history: [],
      currentIndex: -1,
    });
  },

  removeAction: (actionId: string) => {
    set((state) => {
      const actionIndex = state.history.findIndex((a) => a.id === actionId);
      if (actionIndex === -1) return state;

      const newHistory = state.history.filter((a) => a.id !== actionId);
      let newIndex = state.currentIndex;

      // Adjust current index if needed
      if (actionIndex <= state.currentIndex) {
        newIndex = Math.max(-1, state.currentIndex - 1);
      }

      return {
        history: newHistory,
        currentIndex: newIndex,
      };
    });
  },
}));

/**
 * Hook for using undo/redo with keyboard shortcuts
 */
export const useUndoRedoShortcuts = () => {
  const { undo, redo, canUndo, canRedo } = useActionHistoryStore();

  const handleKeyDown = (e: KeyboardEvent) => {
    const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
    const cmdOrCtrl = isMac ? e.metaKey : e.ctrlKey;

    // Cmd/Ctrl + Z - Undo
    if (cmdOrCtrl && e.key === 'z' && !e.shiftKey && canUndo()) {
      e.preventDefault();
      undo().catch(console.error);
    }

    // Cmd/Ctrl + Shift + Z or Cmd/Ctrl + Y - Redo
    if (cmdOrCtrl && ((e.key === 'z' && e.shiftKey) || e.key === 'y') && canRedo()) {
      e.preventDefault();
      redo().catch(console.error);
    }
  };

  return { handleKeyDown };
};
