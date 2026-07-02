// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Capacitor Configuration
 *
 * Native mobile app wrapper with:
 * - iOS and Android support
 * - Native audio engine
 * - Push notifications
 * - Offline storage
 * - Background audio
 */

import { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'music.mycelix.app',
  appName: 'Mycelix Music',
  webDir: 'out',
  bundledWebRuntime: false,

  // Server configuration
  server: {
    androidScheme: 'https',
    iosScheme: 'capacitor',
    hostname: 'app.mycelix.music',
  },

  // iOS specific configuration
  ios: {
    contentInset: 'automatic',
    backgroundColor: '#0F0F0F',
    preferredContentMode: 'mobile',
    allowsLinkPreview: true,
    scrollEnabled: true,
    limitsNavigationsToAppBoundDomains: true,
  },

  // Android specific configuration
  android: {
    backgroundColor: '#0F0F0F',
    allowMixedContent: false,
    captureInput: true,
    webContentsDebuggingEnabled: process.env.NODE_ENV === 'development',
  },

  // Plugin configuration
  plugins: {
    // Push notifications
    PushNotifications: {
      presentationOptions: ['badge', 'sound', 'alert'],
    },

    // Local notifications
    LocalNotifications: {
      smallIcon: 'ic_stat_notification',
      iconColor: '#8B5CF6',
      sound: 'notification.wav',
    },

    // Splash screen
    SplashScreen: {
      launchShowDuration: 2000,
      launchAutoHide: true,
      backgroundColor: '#0F0F0F',
      androidSplashResourceName: 'splash',
      androidScaleType: 'CENTER_CROP',
      showSpinner: true,
      spinnerColor: '#8B5CF6',
      splashFullScreen: true,
      splashImmersive: true,
    },

    // Status bar
    StatusBar: {
      style: 'dark',
      backgroundColor: '#0F0F0F',
    },

    // Keyboard
    Keyboard: {
      resize: 'body',
      style: 'dark',
      resizeOnFullScreen: true,
    },

    // Haptics
    Haptics: {
      selectionStartDuration: 10,
      impactMediumDuration: 15,
      notificationSuccessDuration: 20,
    },

    // App preferences
    Preferences: {
      group: 'MycelixMusic',
    },

    // HTTP
    CapacitorHttp: {
      enabled: true,
    },

    // Cookies
    CapacitorCookies: {
      enabled: true,
    },
  },

  // Background mode for audio playback
  cordova: {
    preferences: {
      ScrollEnabled: 'false',
      BackgroundColor: '0xff0F0F0F',
      AndroidPersistentFileLocation: 'Internal',
      'android-targetSdkVersion': '33',
      'android-minSdkVersion': '24',
    },
  },
};

export default config;
