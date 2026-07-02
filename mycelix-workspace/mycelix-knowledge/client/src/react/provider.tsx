// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * React Provider for Knowledge Client
 *
 * @example
 * ```tsx
 * import { KnowledgeProvider } from '@mycelix/knowledge-sdk/react';
 * import { KnowledgeClient } from '@mycelix/knowledge-sdk';
 * import { AppWebsocket } from '@holochain/client';
 *
 * async function App() {
 *   const appClient = await AppWebsocket.connect('ws://localhost:8888');
 *   const knowledgeClient = new KnowledgeClient(appClient);
 *
 *   return (
 *     <KnowledgeProvider client={knowledgeClient}>
 *       <YourApp />
 *     </KnowledgeProvider>
 *   );
 * }
 * ```
 */

import React, { ReactNode, useMemo, useEffect, useState } from 'react';
import { KnowledgeContext } from './context';
import type { KnowledgeClient } from '../index';
import type { AppWebsocket } from '@holochain/client';

interface KnowledgeProviderProps {
  /** Pre-configured Knowledge client */
  client?: KnowledgeClient;
  /** Or provide connection URL to auto-connect */
  connectionUrl?: string;
  /** Role name for the Knowledge hApp */
  roleName?: string;
  /** Children to render */
  children: ReactNode;
  /** Called when client is ready */
  onReady?: (client: KnowledgeClient) => void;
  /** Called on connection error */
  onError?: (error: Error) => void;
}

/**
 * Provider component for Knowledge SDK
 */
export function KnowledgeProvider({
  client: providedClient,
  connectionUrl,
  roleName = 'knowledge',
  children,
  onReady,
  onError,
}: KnowledgeProviderProps): JSX.Element {
  const [client, setClient] = useState<KnowledgeClient | null>(providedClient || null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(!providedClient && !!connectionUrl);

  useEffect(() => {
    if (providedClient) {
      setClient(providedClient);
      onReady?.(providedClient);
      return;
    }

    if (!connectionUrl) return;

    let cancelled = false;

    async function connect() {
      try {
        setLoading(true);

        // Dynamic import to avoid bundling issues
        const { AppWebsocket } = await import('@holochain/client');
        const { KnowledgeClient } = await import('../index');

        const appClient = await AppWebsocket.connect(connectionUrl!);

        if (cancelled) {
          await appClient.client.close();
          return;
        }

        const newClient = new KnowledgeClient(appClient, roleName);
        setClient(newClient);
        onReady?.(newClient);
      } catch (err) {
        if (cancelled) return;
        const error = err instanceof Error ? err : new Error(String(err));
        setError(error);
        onError?.(error);
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    connect();

    return () => {
      cancelled = true;
    };
  }, [providedClient, connectionUrl, roleName, onReady, onError]);

  const value = useMemo(() => client, [client]);

  if (loading) {
    return (
      <div className="knowledge-provider-loading">
        Connecting to Knowledge...
      </div>
    );
  }

  if (error) {
    return (
      <div className="knowledge-provider-error">
        Failed to connect to Knowledge: {error.message}
      </div>
    );
  }

  return (
    <KnowledgeContext.Provider value={value}>
      {children}
    </KnowledgeContext.Provider>
  );
}

/**
 * HOC to inject Knowledge client as prop
 */
export function withKnowledge<P extends { knowledgeClient?: KnowledgeClient }>(
  Component: React.ComponentType<P>
): React.FC<Omit<P, 'knowledgeClient'>> {
  return function WrappedComponent(props: Omit<P, 'knowledgeClient'>) {
    const client = React.useContext(KnowledgeContext);
    return <Component {...(props as P)} knowledgeClient={client || undefined} />;
  };
}
