// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Undo/Redo Hook
 *
 * Generic undo/redo functionality for any state type.
 * Useful for studio features like editing, effects, etc.
 */

import { useState, useCallback, useRef } from 'react';

export interface UndoRedoState<T> {
  current: T;
  canUndo: boolean;
  canRedo: boolean;
  historyLength: number;
  currentIndex: number;
}

export interface UndoRedoActions<T> {
  set: (value: T | ((prev: T) => T)) => void;
  undo: () => void;
  redo: () => void;
  reset: (initialValue?: T) => void;
  clear: () => void;
  checkpoint: () => void;
}

interface UndoRedoOptions {
  maxHistory?: number;
  debounceMs?: number;
}

/**
 * Hook for managing undo/redo state
 */
export function useUndoRedo<T>(
  initialValue: T,
  options: UndoRedoOptions = {}
): [UndoRedoState<T>, UndoRedoActions<T>] {
  const { maxHistory = 100, debounceMs = 0 } = options;

  // History stack
  const [history, setHistory] = useState<T[]>([initialValue]);
  const [currentIndex, setCurrentIndex] = useState(0);

  // Debounce timer
  const debounceTimer = useRef<NodeJS.Timeout | null>(null);
  const pendingValue = useRef<T | null>(null);

  // Get current value
  const current = history[currentIndex];
  const canUndo = currentIndex > 0;
  const canRedo = currentIndex < history.length - 1;

  // Set new value
  const set = useCallback(
    (value: T | ((prev: T) => T)) => {
      const newValue = typeof value === 'function' ? (value as (prev: T) => T)(history[currentIndex]) : value;

      // If debouncing, store pending value
      if (debounceMs > 0) {
        pendingValue.current = newValue;

        if (debounceTimer.current) {
          clearTimeout(debounceTimer.current);
        }

        debounceTimer.current = setTimeout(() => {
          if (pendingValue.current !== null) {
            commitValue(pendingValue.current);
            pendingValue.current = null;
          }
        }, debounceMs);

        // Update current value immediately for UI
        setHistory((prev) => {
          const newHistory = [...prev];
          newHistory[currentIndex] = newValue;
          return newHistory;
        });
      } else {
        commitValue(newValue);
      }
    },
    [currentIndex, history, debounceMs]
  );

  // Commit value to history
  const commitValue = useCallback(
    (newValue: T) => {
      setHistory((prev) => {
        // Remove any redo history
        const newHistory = prev.slice(0, currentIndex + 1);

        // Add new value
        newHistory.push(newValue);

        // Limit history size
        if (newHistory.length > maxHistory) {
          newHistory.shift();
          setCurrentIndex((i) => Math.max(0, i - 1));
        }

        return newHistory;
      });

      setCurrentIndex((prev) => Math.min(prev + 1, maxHistory - 1));
    },
    [currentIndex, maxHistory]
  );

  // Undo
  const undo = useCallback(() => {
    if (canUndo) {
      // Commit any pending changes first
      if (pendingValue.current !== null) {
        if (debounceTimer.current) {
          clearTimeout(debounceTimer.current);
        }
        commitValue(pendingValue.current);
        pendingValue.current = null;
      }

      setCurrentIndex((prev) => prev - 1);
    }
  }, [canUndo, commitValue]);

  // Redo
  const redo = useCallback(() => {
    if (canRedo) {
      setCurrentIndex((prev) => prev + 1);
    }
  }, [canRedo]);

  // Reset to initial or specified value
  const reset = useCallback((value?: T) => {
    const resetValue = value !== undefined ? value : initialValue;
    setHistory([resetValue]);
    setCurrentIndex(0);
    pendingValue.current = null;
    if (debounceTimer.current) {
      clearTimeout(debounceTimer.current);
    }
  }, [initialValue]);

  // Clear history but keep current
  const clear = useCallback(() => {
    setHistory([current]);
    setCurrentIndex(0);
  }, [current]);

  // Create a checkpoint (useful for saving specific state)
  const checkpoint = useCallback(() => {
    // Commit pending changes
    if (pendingValue.current !== null) {
      if (debounceTimer.current) {
        clearTimeout(debounceTimer.current);
      }
      commitValue(pendingValue.current);
      pendingValue.current = null;
    }
  }, [commitValue]);

  const state: UndoRedoState<T> = {
    current,
    canUndo,
    canRedo,
    historyLength: history.length,
    currentIndex,
  };

  const actions: UndoRedoActions<T> = {
    set,
    undo,
    redo,
    reset,
    clear,
    checkpoint,
  };

  return [state, actions];
}

/**
 * Hook for undo/redo with multiple named states
 */
export function useMultiUndoRedo<T extends Record<string, any>>(
  initialValues: T,
  options: UndoRedoOptions = {}
): [
  { [K in keyof T]: UndoRedoState<T[K]>['current'] },
  {
    set: <K extends keyof T>(key: K, value: T[K] | ((prev: T[K]) => T[K])) => void;
    undo: () => void;
    redo: () => void;
    reset: () => void;
    canUndo: boolean;
    canRedo: boolean;
  }
] {
  const [state, actions] = useUndoRedo<T>(initialValues, options);

  const currentValues = state.current;

  const multiActions = {
    set: <K extends keyof T>(key: K, value: T[K] | ((prev: T[K]) => T[K])) => {
      actions.set((prev) => ({
        ...prev,
        [key]: typeof value === 'function' ? (value as (p: T[K]) => T[K])(prev[key]) : value,
      }));
    },
    undo: actions.undo,
    redo: actions.redo,
    reset: () => actions.reset(initialValues),
    canUndo: state.canUndo,
    canRedo: state.canRedo,
  };

  return [currentValues, multiActions];
}

/**
 * Command-based undo/redo for complex operations
 */
export interface Command<T> {
  execute: (state: T) => T;
  undo: (state: T) => T;
  description?: string;
}

export function useCommandHistory<T>(initialState: T) {
  const [state, setState] = useState(initialState);
  const [history, setHistory] = useState<Command<T>[]>([]);
  const [currentIndex, setCurrentIndex] = useState(-1);

  const execute = useCallback(
    (command: Command<T>) => {
      setState((prev) => {
        const newState = command.execute(prev);

        // Add to history, removing any redo stack
        setHistory((hist) => [...hist.slice(0, currentIndex + 1), command]);
        setCurrentIndex((i) => i + 1);

        return newState;
      });
    },
    [currentIndex]
  );

  const undo = useCallback(() => {
    if (currentIndex >= 0) {
      const command = history[currentIndex];
      setState((prev) => command.undo(prev));
      setCurrentIndex((i) => i - 1);
    }
  }, [currentIndex, history]);

  const redo = useCallback(() => {
    if (currentIndex < history.length - 1) {
      const command = history[currentIndex + 1];
      setState((prev) => command.execute(prev));
      setCurrentIndex((i) => i + 1);
    }
  }, [currentIndex, history]);

  const reset = useCallback((newState?: T) => {
    setState(newState ?? initialState);
    setHistory([]);
    setCurrentIndex(-1);
  }, [initialState]);

  return {
    state,
    execute,
    undo,
    redo,
    reset,
    canUndo: currentIndex >= 0,
    canRedo: currentIndex < history.length - 1,
    history: history.map((cmd) => cmd.description || 'Action'),
    currentIndex,
  };
}

export default useUndoRedo;
