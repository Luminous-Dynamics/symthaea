// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * React Context for Knowledge Client
 */

import { createContext, useContext } from 'react';
import type { KnowledgeClient } from '../index';

export const KnowledgeContext = createContext<KnowledgeClient | null>(null);

/**
 * Hook to access the Knowledge client from context
 * @throws Error if used outside of KnowledgeProvider
 */
export function useKnowledgeClient(): KnowledgeClient {
  const client = useContext(KnowledgeContext);
  if (!client) {
    throw new Error(
      'useKnowledgeClient must be used within a KnowledgeProvider. ' +
      'Wrap your app with <KnowledgeProvider client={knowledgeClient}>...</KnowledgeProvider>'
    );
  }
  return client;
}

/**
 * Hook to optionally access the Knowledge client (returns null if not available)
 */
export function useOptionalKnowledgeClient(): KnowledgeClient | null {
  return useContext(KnowledgeContext);
}
