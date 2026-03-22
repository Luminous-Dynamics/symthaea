import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, TrendingUp, TrendingDown, Bell, Users, Zap, DollarSign, AlertCircle } from 'lucide-react';
import { usePriceWebSocket, useWebSocket, ConnectionState } from '../hooks/useWebSocket';

interface Update {
  id: string;
  type: 'price' | 'status' | 'investment' | 'milestone';
  title: string;
  description: string;
  timestamp: string;
  value?: number;
  change?: number;
  icon?: React.ElementType;
}

export const RealTimeUpdates: React.FC = () => {
  const [updates, setUpdates] = useState<Update[]>([]);
  const { prices, isConnected: priceConnected } = usePriceWebSocket();
  
  // General updates WebSocket
  const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8002/ws';
  const clientId = `updates-${Date.now()}`;
  
  const { 
    isConnected, 
    connectionState, 
    subscribe 
  } = useWebSocket(
    `${wsUrl}/${clientId}`,
    {
      onMessage: (message) => {
        if (message.type === 'status_change' && message.data) {
          const update: Update = {
            id: `status-${Date.now()}`,
            type: 'status',
            title: `Project Status Update`,
            description: `${message.data.project_id} changed from ${message.data.old_status} to ${message.data.new_status}`,
            timestamp: message.data.timestamp,
            icon: Activity
          };
          setUpdates(prev => [update, ...prev].slice(0, 10)); // Keep last 10 updates
        }
      }
    }
  );
  
  // Subscribe to updates channel
  useEffect(() => {
    if (isConnected) {
      subscribe('updates');
    }
  }, [isConnected, subscribe]);
  
  // Process price updates
  useEffect(() => {
    Object.entries(prices).forEach(([projectId, priceData]: [string, any]) => {
      const update: Update = {
        id: `price-${projectId}-${Date.now()}`,
        type: 'price',
        title: `${priceData.project_type?.toUpperCase() || 'Project'} Price Update`,
        description: projectId,
        timestamp: priceData.timestamp,
        value: priceData.current_price,
        change: priceData.change_24h,
        icon: priceData.change_24h > 0 ? TrendingUp : TrendingDown
      };
      setUpdates(prev => [update, ...prev].slice(0, 10));
    });
  }, [prices]);
  
  // Connection indicator
  const getConnectionColor = () => {
    switch (connectionState) {
      case ConnectionState.Connected:
        return 'bg-green-500';
      case ConnectionState.Connecting:
      case ConnectionState.Reconnecting:
        return 'bg-yellow-500';
      case ConnectionState.Error:
        return 'bg-red-500';
      default:
        return 'bg-gray-500';
    }
  };
  
  return (
    <div className="bg-white/5 backdrop-blur-xl rounded-2xl p-6 border border-white/10">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <Activity className="w-6 h-6 text-blue-400" />
          <h3 className="text-xl font-bold text-white">Live Updates</h3>
        </div>
        
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${getConnectionColor()} animate-pulse`} />
          <span className="text-sm text-gray-400">
            {connectionState === ConnectionState.Connected ? 'Live' : connectionState}
          </span>
        </div>
      </div>
      
      {/* Updates List */}
      <div className="space-y-3 max-h-96 overflow-y-auto">
        <AnimatePresence>
          {updates.length === 0 ? (
            <div className="text-center py-8 text-gray-400">
              <Activity className="w-12 h-12 mx-auto mb-3 opacity-30" />
              <p>Waiting for live updates...</p>
            </div>
          ) : (
            updates.map((update) => {
              const Icon = update.icon || Bell;
              return (
                <motion.div
                  key={update.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  className="flex items-start gap-3 p-3 bg-white/5 rounded-lg hover:bg-white/10 transition-colors"
                >
                  <div className={`p-2 rounded-lg ${
                    update.type === 'price' 
                      ? update.change && update.change > 0 
                        ? 'bg-green-500/20' 
                        : 'bg-red-500/20'
                      : 'bg-blue-500/20'
                  }`}>
                    <Icon className={`w-4 h-4 ${
                      update.type === 'price' 
                        ? update.change && update.change > 0 
                          ? 'text-green-400' 
                          : 'text-red-400'
                        : 'text-blue-400'
                    }`} />
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-white truncate">
                      {update.title}
                    </p>
                    <p className="text-xs text-gray-400 truncate">
                      {update.description}
                    </p>
                    {update.value !== undefined && (
                      <div className="flex items-center gap-2 mt-1">
                        <span className="text-sm font-bold text-white">
                          ${update.value.toFixed(2)}
                        </span>
                        {update.change !== undefined && (
                          <span className={`text-xs ${
                            update.change > 0 ? 'text-green-400' : 'text-red-400'
                          }`}>
                            {update.change > 0 ? '+' : ''}{update.change.toFixed(2)}%
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                  
                  <div className="text-xs text-gray-500">
                    {new Date(update.timestamp).toLocaleTimeString()}
                  </div>
                </motion.div>
              );
            })
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

// Live Activity Indicator Component
export const LiveActivityIndicator: React.FC = () => {
  const [activeUsers, setActiveUsers] = useState(0);
  const [viewing, setViewing] = useState(0);
  const [investing, setInvesting] = useState(0);
  
  const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8002/ws';
  const clientId = `activity-${Date.now()}`;
  
  const { isConnected } = useWebSocket(
    `${wsUrl}/${clientId}`,
    {
      onMessage: (message) => {
        if (message.type === 'user_activity' && message.activity) {
          const { action } = message.activity;
          
          // Update counters based on activity
          if (action === 'viewing') {
            setViewing(prev => prev + 1);
          } else if (action === 'investing') {
            setInvesting(prev => prev + 1);
          }
          
          setActiveUsers(prev => prev + 1);
          
          // Decay counters over time
          setTimeout(() => {
            if (action === 'viewing') {
              setViewing(prev => Math.max(0, prev - 1));
            } else if (action === 'investing') {
              setInvesting(prev => Math.max(0, prev - 1));
            }
            setActiveUsers(prev => Math.max(0, prev - 1));
          }, 30000); // 30 second decay
        }
      }
    }
  );
  
  useEffect(() => {
    // Simulate some initial activity for demo
    const interval = setInterval(() => {
      setActiveUsers(prev => Math.max(50, prev + Math.floor(Math.random() * 10 - 3)));
      setViewing(prev => Math.max(20, prev + Math.floor(Math.random() * 5 - 2)));
      setInvesting(prev => Math.max(5, prev + Math.floor(Math.random() * 3 - 1)));
    }, 5000);
    
    return () => clearInterval(interval);
  }, []);
  
  return (
    <div className="flex items-center gap-6 text-sm">
      <div className="flex items-center gap-2">
        <div className="relative">
          <Users className="w-4 h-4 text-green-400" />
          <div className="absolute -top-1 -right-1 w-2 h-2 bg-green-400 rounded-full animate-pulse" />
        </div>
        <span className="text-gray-300">
          <span className="font-bold text-white">{activeUsers}</span> Active Now
        </span>
      </div>
      
      <div className="flex items-center gap-2">
        <Activity className="w-4 h-4 text-blue-400" />
        <span className="text-gray-400">
          <span className="font-bold text-white">{viewing}</span> Viewing
        </span>
      </div>
      
      <div className="flex items-center gap-2">
        <DollarSign className="w-4 h-4 text-purple-400" />
        <span className="text-gray-400">
          <span className="font-bold text-white">{investing}</span> Investing
        </span>
      </div>
    </div>
  );
};

// Price ticker component
export const PriceTicker: React.FC = () => {
  const { prices, isConnected } = usePriceWebSocket();
  const [tickerItems, setTickerItems] = useState<any[]>([]);
  
  useEffect(() => {
    const items = Object.entries(prices).map(([id, data]: [string, any]) => ({
      id,
      type: data.project_type,
      price: data.current_price,
      change: data.change_24h,
      volume: data.volume_24h
    })).slice(0, 10); // Show top 10
    
    setTickerItems(items);
  }, [prices]);
  
  if (!isConnected || tickerItems.length === 0) {
    return null;
  }
  
  return (
    <div className="bg-black/50 backdrop-blur-xl border-t border-b border-white/10 py-2">
      <div className="flex items-center gap-8 animate-scroll-x">
        {tickerItems.map((item) => (
          <div key={item.id} className="flex items-center gap-3 whitespace-nowrap">
            <span className="text-sm font-medium text-gray-400">
              {item.type?.toUpperCase()}
            </span>
            <span className="text-sm font-bold text-white">
              ${item.price?.toFixed(2)}
            </span>
            <span className={`text-xs font-medium ${
              item.change > 0 ? 'text-green-400' : 'text-red-400'
            }`}>
              {item.change > 0 ? '+' : ''}{item.change?.toFixed(2)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};