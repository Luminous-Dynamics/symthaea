export const swaggerDocument = {
  openapi: '3.0.0',
  info: {
    title: 'Mycelix-Mail API',
    version: '1.0.0',
    description: 'A modern email client API with IMAP/SMTP support',
    contact: {
      name: 'Luminous Dynamics',
      url: 'https://github.com/Luminous-Dynamics/Mycelix-Mail',
    },
    license: {
      name: 'MIT',
      url: 'https://opensource.org/licenses/MIT',
    },
  },
  servers: [
    {
      url: 'http://localhost:3000',
      description: 'Development server',
    },
  ],
  components: {
    securitySchemes: {
      bearerAuth: {
        type: 'http',
        scheme: 'bearer',
        bearerFormat: 'JWT',
      },
    },
    schemas: {
      User: {
        type: 'object',
        properties: {
          id: { type: 'string', format: 'uuid' },
          email: { type: 'string', format: 'email' },
          firstName: { type: 'string' },
          lastName: { type: 'string' },
          createdAt: { type: 'string', format: 'date-time' },
          updatedAt: { type: 'string', format: 'date-time' },
        },
      },
      EmailAccount: {
        type: 'object',
        properties: {
          id: { type: 'string', format: 'uuid' },
          email: { type: 'string', format: 'email' },
          provider: { type: 'string' },
          imapHost: { type: 'string' },
          imapPort: { type: 'number' },
          imapSecure: { type: 'boolean' },
          smtpHost: { type: 'string' },
          smtpPort: { type: 'number' },
          smtpSecure: { type: 'boolean' },
          isDefault: { type: 'boolean' },
        },
      },
      Email: {
        type: 'object',
        properties: {
          id: { type: 'string', format: 'uuid' },
          messageId: { type: 'string' },
          subject: { type: 'string' },
          from: { type: 'object' },
          to: { type: 'array', items: { type: 'object' } },
          bodyText: { type: 'string' },
          bodyHtml: { type: 'string' },
          isRead: { type: 'boolean' },
          isStarred: { type: 'boolean' },
          date: { type: 'string', format: 'date-time' },
        },
      },
      Error: {
        type: 'object',
        properties: {
          status: { type: 'string', example: 'error' },
          message: { type: 'string' },
        },
      },
    },
  },
  paths: {
    '/api/auth/register': {
      post: {
        tags: ['Authentication'],
        summary: 'Register a new user',
        requestBody: {
          required: true,
          content: {
            'application/json': {
              schema: {
                type: 'object',
                required: ['email', 'password'],
                properties: {
                  email: { type: 'string', format: 'email' },
                  password: { type: 'string', minLength: 8 },
                  firstName: { type: 'string' },
                  lastName: { type: 'string' },
                },
              },
            },
          },
        },
        responses: {
          201: {
            description: 'User registered successfully',
            content: {
              'application/json': {
                schema: {
                  type: 'object',
                  properties: {
                    status: { type: 'string', example: 'success' },
                    data: {
                      type: 'object',
                      properties: {
                        user: { $ref: '#/components/schemas/User' },
                        token: { type: 'string' },
                      },
                    },
                  },
                },
              },
            },
          },
          400: { $ref: '#/components/responses/BadRequest' },
        },
      },
    },
    '/api/auth/login': {
      post: {
        tags: ['Authentication'],
        summary: 'Login user',
        requestBody: {
          required: true,
          content: {
            'application/json': {
              schema: {
                type: 'object',
                required: ['email', 'password'],
                properties: {
                  email: { type: 'string', format: 'email' },
                  password: { type: 'string' },
                },
              },
            },
          },
        },
        responses: {
          200: {
            description: 'Login successful',
          },
          401: {
            description: 'Invalid credentials',
          },
        },
      },
    },
    '/api/auth/me': {
      get: {
        tags: ['Authentication'],
        summary: 'Get current user profile',
        security: [{ bearerAuth: [] }],
        responses: {
          200: {
            description: 'User profile retrieved',
            content: {
              'application/json': {
                schema: {
                  type: 'object',
                  properties: {
                    status: { type: 'string', example: 'success' },
                    data: {
                      type: 'object',
                      properties: {
                        user: { $ref: '#/components/schemas/User' },
                      },
                    },
                  },
                },
              },
            },
          },
          401: { description: 'Unauthorized' },
        },
      },
    },
    '/api/accounts': {
      get: {
        tags: ['Email Accounts'],
        summary: 'Get all email accounts',
        security: [{ bearerAuth: [] }],
        responses: {
          200: {
            description: 'List of email accounts',
          },
        },
      },
      post: {
        tags: ['Email Accounts'],
        summary: 'Add new email account',
        security: [{ bearerAuth: [] }],
        requestBody: {
          required: true,
          content: {
            'application/json': {
              schema: {
                type: 'object',
                required: [
                  'email',
                  'provider',
                  'imapHost',
                  'imapPort',
                  'imapUser',
                  'imapPassword',
                  'smtpHost',
                  'smtpPort',
                  'smtpUser',
                  'smtpPassword',
                ],
                properties: {
                  email: { type: 'string', format: 'email' },
                  provider: { type: 'string' },
                  imapHost: { type: 'string' },
                  imapPort: { type: 'number' },
                  imapSecure: { type: 'boolean', default: true },
                  imapUser: { type: 'string' },
                  imapPassword: { type: 'string' },
                  smtpHost: { type: 'string' },
                  smtpPort: { type: 'number' },
                  smtpSecure: { type: 'boolean', default: true },
                  smtpUser: { type: 'string' },
                  smtpPassword: { type: 'string' },
                },
              },
            },
          },
        },
        responses: {
          201: { description: 'Account created successfully' },
          400: { description: 'Invalid request' },
        },
      },
    },
    '/api/emails': {
      get: {
        tags: ['Emails'],
        summary: 'Get emails with pagination and filters',
        security: [{ bearerAuth: [] }],
        parameters: [
          {
            name: 'accountId',
            in: 'query',
            schema: { type: 'string' },
          },
          {
            name: 'folderId',
            in: 'query',
            schema: { type: 'string' },
          },
          {
            name: 'page',
            in: 'query',
            schema: { type: 'number', default: 1 },
          },
          {
            name: 'limit',
            in: 'query',
            schema: { type: 'number', default: 50 },
          },
          {
            name: 'search',
            in: 'query',
            schema: { type: 'string' },
          },
        ],
        responses: {
          200: { description: 'List of emails' },
        },
      },
      post: {
        tags: ['Emails'],
        summary: 'Send email',
        security: [{ bearerAuth: [] }],
        requestBody: {
          required: true,
          content: {
            'application/json': {
              schema: {
                type: 'object',
                required: ['accountId', 'to', 'subject', 'body'],
                properties: {
                  accountId: { type: 'string' },
                  to: { type: 'array', items: { type: 'string' } },
                  subject: { type: 'string' },
                  body: { type: 'string' },
                  cc: { type: 'array', items: { type: 'string' } },
                  bcc: { type: 'array', items: { type: 'string' } },
                },
              },
            },
          },
        },
        responses: {
          201: { description: 'Email sent successfully' },
        },
      },
    },
  },
  tags: [
    { name: 'Authentication', description: 'User authentication endpoints' },
    { name: 'Email Accounts', description: 'Email account management' },
    { name: 'Emails', description: 'Email operations' },
    { name: 'Folders', description: 'Folder management' },
  ],
};
