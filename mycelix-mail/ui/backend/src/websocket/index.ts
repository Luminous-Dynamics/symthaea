import { Server as HttpServer } from 'http';
import { WebSocketServer, WebSocket } from 'ws';
import { verifyToken } from '../utils/jwt';

interface AuthenticatedWebSocket extends WebSocket {
  userId?: string;
}

const clients = new Map<string, Set<AuthenticatedWebSocket>>();

export const setupWebSocket = (server: HttpServer) => {
  const wss = new WebSocketServer({ server, path: '/ws' });

  wss.on('connection', (ws: AuthenticatedWebSocket, req) => {
    console.log('WebSocket connection attempt');

    // Extract token from query string
    const url = new URL(req.url!, `http://${req.headers.host}`);
    const token = url.searchParams.get('token');

    if (!token) {
      ws.close(1008, 'No token provided');
      return;
    }

    try {
      const { userId } = verifyToken(token);
      ws.userId = userId;

      // Add client to the map
      if (!clients.has(userId)) {
        clients.set(userId, new Set());
      }
      clients.get(userId)!.add(ws);

      console.log(`User ${userId} connected via WebSocket`);

      ws.on('close', () => {
        const userClients = clients.get(userId);
        if (userClients) {
          userClients.delete(ws);
          if (userClients.size === 0) {
            clients.delete(userId);
          }
        }
        console.log(`User ${userId} disconnected from WebSocket`);
      });

      ws.on('error', (error) => {
        console.error('WebSocket error:', error);
      });

      // Send welcome message
      ws.send(JSON.stringify({
        type: 'connected',
        message: 'WebSocket connection established',
      }));
    } catch (error) {
      console.error('WebSocket authentication error:', error);
      ws.close(1008, 'Invalid token');
    }
  });

  console.log('✅ WebSocket server initialized');
};

export const notifyUser = (userId: string, event: string, data: any) => {
  const userClients = clients.get(userId);
  if (!userClients) return;

  const message = JSON.stringify({ type: event, data });

  userClients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(message);
    }
  });
};

export const notifyNewEmail = (userId: string, email: any) => {
  notifyUser(userId, 'new_email', { email });
};

export const notifyEmailRead = (userId: string, emailId: string) => {
  notifyUser(userId, 'email_read', { emailId });
};
