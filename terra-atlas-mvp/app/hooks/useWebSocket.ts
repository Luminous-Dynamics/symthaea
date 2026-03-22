import { useEffect, useRef, useState, useCallback } from 'react';

interface WebSocketMessage {
  type: string;
  data?: any;
  timestamp?: string;
}

interface WebSocketOptions {
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (error: Event) => void;
  onMessage?: (message: WebSocketMessage) => void;
}

export enum ConnectionState {
  Connecting = 'CONNECTING',
  Connected = 'CONNECTED',
  Disconnected = 'DISCONNECTED',
  Reconnecting = 'RECONNECTING',
  Error = 'ERROR'
}

export const useWebSocket = (url: string | null, options: WebSocketOptions = {}) => {
  const [connectionState, setConnectionState] = useState<ConnectionState>(ConnectionState.Disconnected);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [subscriptions, setSubscriptions] = useState<Set<string>>(new Set());
  
  const ws = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef(0);
  
  const {
    reconnectInterval = 3000,
    maxReconnectAttempts = 10,
    onOpen,
    onClose,
    onError,
    onMessage,
  } = options;

  const connect = useCallback(() => {
    if (!url || ws.current?.readyState === WebSocket.OPEN) return;
    
    setConnectionState(ConnectionState.Connecting);
    
    try {
      ws.current = new WebSocket(url);
      
      ws.current.onopen = () => {
        console.log('WebSocket connected');
        setConnectionState(ConnectionState.Connected);
        reconnectAttemptsRef.current = 0;
        
        // Resubscribe to previous channels
        subscriptions.forEach(channel => {
          sendMessage({
            type: 'subscribe',
            channel
          });
        });
        
        onOpen?.();
      };
      
      ws.current.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          setLastMessage(message);
          onMessage?.(message);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };
      
      ws.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionState(ConnectionState.Error);
        onError?.(error);
      };
      
      ws.current.onclose = () => {
        console.log('WebSocket disconnected');
        setConnectionState(ConnectionState.Disconnected);
        onClose?.();
        
        // Attempt to reconnect
        if (reconnectAttemptsRef.current < maxReconnectAttempts) {
          setConnectionState(ConnectionState.Reconnecting);
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectAttemptsRef.current++;
            connect();
          }, reconnectInterval);
        }
      };
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      setConnectionState(ConnectionState.Error);
    }
  }, [url, reconnectInterval, maxReconnectAttempts, onOpen, onClose, onError, onMessage, subscriptions]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    if (ws.current) {
      ws.current.close();
      ws.current = null;
    }
    
    setConnectionState(ConnectionState.Disconnected);
  }, []);

  const sendMessage = useCallback((message: WebSocketMessage | object) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(message));
      return true;
    }
    console.warn('WebSocket is not connected');
    return false;
  }, []);

  const subscribe = useCallback((channel: string) => {
    setSubscriptions(prev => new Set([...prev, channel]));
    
    if (connectionState === ConnectionState.Connected) {
      sendMessage({
        type: 'subscribe',
        channel
      });
    }
  }, [connectionState, sendMessage]);

  const unsubscribe = useCallback((channel: string) => {
    setSubscriptions(prev => {
      const newSet = new Set(prev);
      newSet.delete(channel);
      return newSet;
    });
    
    if (connectionState === ConnectionState.Connected) {
      sendMessage({
        type: 'unsubscribe',
        channel
      });
    }
  }, [connectionState, sendMessage]);

  // Auto-connect when URL is provided
  useEffect(() => {
    if (url) {
      connect();
    }
    
    return () => {
      disconnect();
    };
  }, [url]); // Only reconnect when URL changes

  return {
    connectionState,
    lastMessage,
    sendMessage,
    subscribe,
    unsubscribe,
    connect,
    disconnect,
    isConnected: connectionState === ConnectionState.Connected,
  };
};

// Hook for managing project-specific WebSocket subscriptions
export const useProjectWebSocket = (projectId: string | null) => {
  const [updates, setUpdates] = useState<any[]>([]);
  const [activities, setActivities] = useState<any[]>([]);
  
  const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8002/ws';
  const clientId = typeof window !== 'undefined' 
    ? `client-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    : null;
  
  const { 
    isConnected, 
    subscribe, 
    unsubscribe, 
    sendMessage,
    connectionState 
  } = useWebSocket(
    clientId ? `${wsUrl}/${clientId}` : null,
    {
      onMessage: (message) => {
        if (message.type === 'project_update') {
          setUpdates(prev => [...prev, message.data]);
        } else if (message.type === 'user_activity') {
          setActivities(prev => [...prev, message.data]);
        }
      }
    }
  );
  
  useEffect(() => {
    if (isConnected && projectId) {
      subscribe(`project:${projectId}`);
      
      return () => {
        unsubscribe(`project:${projectId}`);
      };
    }
  }, [isConnected, projectId, subscribe, unsubscribe]);
  
  const sendProjectUpdate = useCallback((updateType: string, data: any) => {
    if (!projectId) return;
    
    sendMessage({
      type: 'project_update',
      project_id: projectId,
      update_type: updateType,
      data
    });
  }, [projectId, sendMessage]);
  
  const sendUserActivity = useCallback((action: string, data: any = {}) => {
    sendMessage({
      type: 'user_activity',
      action,
      project_id: projectId,
      data
    });
  }, [projectId, sendMessage]);
  
  return {
    isConnected,
    connectionState,
    updates,
    activities,
    sendProjectUpdate,
    sendUserActivity,
  };
};

// Hook for price updates
export const usePriceWebSocket = () => {
  const [prices, setPrices] = useState<Record<string, any>>({});
  
  const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8002/ws';
  const clientId = typeof window !== 'undefined' 
    ? `price-client-${Date.now()}`
    : null;
  
  const { isConnected, subscribe } = useWebSocket(
    clientId ? `${wsUrl}/${clientId}` : null,
    {
      onOpen: () => {
        console.log('Price WebSocket connected');
      },
      onMessage: (message) => {
        if (message.type === 'price_update' && message.data) {
          const { project_id, ...priceData } = message.data;
          setPrices(prev => ({
            ...prev,
            [project_id]: priceData
          }));
        }
      }
    }
  );
  
  useEffect(() => {
    if (isConnected) {
      subscribe('prices');
    }
  }, [isConnected, subscribe]);
  
  return { prices, isConnected };
};