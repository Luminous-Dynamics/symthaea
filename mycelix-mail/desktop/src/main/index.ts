/**
 * Mycelix Mail Desktop - Main Process
 *
 * Electron main process with native integrations
 */

import { app, BrowserWindow, ipcMain, shell, Tray, Menu, nativeTheme, Notification } from 'electron';
import { join } from 'path';
import { electronApp, optimizer, is } from '@electron-toolkit/utils';
import Store from 'electron-store';
import { autoUpdater } from 'electron-updater';

// Initialize store for settings
const store = new Store({
  defaults: {
    windowBounds: { width: 1200, height: 800 },
    startMinimized: false,
    closeToTray: true,
    checkUpdatesAutomatically: true,
    theme: 'system',
  },
});

let mainWindow: BrowserWindow | null = null;
let tray: Tray | null = null;

function createWindow(): void {
  const bounds = store.get('windowBounds') as { width: number; height: number };

  mainWindow = new BrowserWindow({
    width: bounds.width,
    height: bounds.height,
    minWidth: 800,
    minHeight: 600,
    show: false,
    autoHideMenuBar: true,
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    trafficLightPosition: { x: 16, y: 16 },
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: true,
      contextIsolation: true,
      nodeIntegration: false,
    },
    icon: join(__dirname, '../../resources/icon.png'),
  });

  mainWindow.on('ready-to-show', () => {
    if (!store.get('startMinimized')) {
      mainWindow?.show();
    }
  });

  // Save window bounds on close
  mainWindow.on('close', (event) => {
    if (store.get('closeToTray') && !app.isQuitting) {
      event.preventDefault();
      mainWindow?.hide();
      return;
    }

    const bounds = mainWindow?.getBounds();
    if (bounds) {
      store.set('windowBounds', { width: bounds.width, height: bounds.height });
    }
  });

  mainWindow.webContents.setWindowOpenHandler((details) => {
    shell.openExternal(details.url);
    return { action: 'deny' };
  });

  // Load the app
  if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL']);
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'));
  }
}

function createTray(): void {
  const iconPath = join(__dirname, '../../resources/tray-icon.png');
  tray = new Tray(iconPath);

  const contextMenu = Menu.buildFromTemplate([
    {
      label: 'Open Mycelix Mail',
      click: () => {
        mainWindow?.show();
      },
    },
    { type: 'separator' },
    {
      label: 'Compose New Email',
      click: () => {
        mainWindow?.show();
        mainWindow?.webContents.send('navigate', '/compose');
      },
    },
    {
      label: 'Check for New Emails',
      click: () => {
        mainWindow?.webContents.send('check-emails');
      },
    },
    { type: 'separator' },
    {
      label: 'Settings',
      click: () => {
        mainWindow?.show();
        mainWindow?.webContents.send('navigate', '/settings');
      },
    },
    { type: 'separator' },
    {
      label: 'Quit',
      click: () => {
        app.isQuitting = true;
        app.quit();
      },
    },
  ]);

  tray.setToolTip('Mycelix Mail');
  tray.setContextMenu(contextMenu);

  tray.on('click', () => {
    if (mainWindow?.isVisible()) {
      mainWindow.hide();
    } else {
      mainWindow?.show();
    }
  });
}

// IPC Handlers
function setupIpcHandlers(): void {
  // Get/set settings
  ipcMain.handle('store:get', (_, key) => store.get(key));
  ipcMain.handle('store:set', (_, key, value) => store.set(key, value));

  // Theme
  ipcMain.handle('theme:get', () => nativeTheme.themeSource);
  ipcMain.handle('theme:set', (_, theme: 'system' | 'light' | 'dark') => {
    nativeTheme.themeSource = theme;
    store.set('theme', theme);
  });

  // Window controls
  ipcMain.on('window:minimize', () => mainWindow?.minimize());
  ipcMain.on('window:maximize', () => {
    if (mainWindow?.isMaximized()) {
      mainWindow.unmaximize();
    } else {
      mainWindow?.maximize();
    }
  });
  ipcMain.on('window:close', () => mainWindow?.close());

  // Notifications
  ipcMain.on('notification:show', (_, options: { title: string; body: string; data?: any }) => {
    const notification = new Notification({
      title: options.title,
      body: options.body,
      icon: join(__dirname, '../../resources/icon.png'),
    });

    notification.on('click', () => {
      mainWindow?.show();
      if (options.data?.emailId) {
        mainWindow?.webContents.send('navigate', `/email/${options.data.emailId}`);
      }
    });

    notification.show();
  });

  // Badge count (macOS)
  ipcMain.on('badge:set', (_, count: number) => {
    if (process.platform === 'darwin') {
      app.dock.setBadge(count > 0 ? count.toString() : '');
    }
  });

  // Open external links
  ipcMain.on('shell:openExternal', (_, url: string) => {
    shell.openExternal(url);
  });

  // Check for updates
  ipcMain.handle('updater:check', async () => {
    try {
      const result = await autoUpdater.checkForUpdates();
      return result?.updateInfo;
    } catch (error) {
      return null;
    }
  });

  ipcMain.on('updater:install', () => {
    autoUpdater.quitAndInstall();
  });
}

// Auto-updater events
function setupAutoUpdater(): void {
  autoUpdater.autoDownload = false;
  autoUpdater.autoInstallOnAppQuit = true;

  autoUpdater.on('update-available', (info) => {
    mainWindow?.webContents.send('update-available', info);
  });

  autoUpdater.on('update-downloaded', (info) => {
    mainWindow?.webContents.send('update-downloaded', info);
  });

  autoUpdater.on('error', (error) => {
    mainWindow?.webContents.send('update-error', error.message);
  });

  if (store.get('checkUpdatesAutomatically')) {
    autoUpdater.checkForUpdates();
  }
}

// App lifecycle
app.whenReady().then(() => {
  // Set app user model id for windows
  electronApp.setAppUserModelId('com.mycelixmail.desktop');

  // Default open or close DevTools by F12 in development
  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window);
  });

  setupIpcHandlers();
  createWindow();
  createTray();

  if (!is.dev) {
    setupAutoUpdater();
  }

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    } else {
      mainWindow?.show();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  app.isQuitting = true;
});

// Deep linking
app.setAsDefaultProtocolClient('mycelixmail');

app.on('open-url', (event, url) => {
  event.preventDefault();
  mainWindow?.show();

  // Handle deep link
  if (url.startsWith('mycelixmail://compose')) {
    mainWindow?.webContents.send('navigate', '/compose');
  } else if (url.startsWith('mycelixmail://email/')) {
    const emailId = url.replace('mycelixmail://email/', '');
    mainWindow?.webContents.send('navigate', `/email/${emailId}`);
  }
});

// Type augmentation
declare module 'electron' {
  interface App {
    isQuitting: boolean;
  }
}
