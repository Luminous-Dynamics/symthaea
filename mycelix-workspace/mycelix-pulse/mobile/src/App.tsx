// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Pulse Mobile App
 *
 * React Native app with Expo
 */

import React, { useEffect, useState } from 'react';
import { StatusBar } from 'expo-status-bar';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import * as SplashScreen from 'expo-splash-screen';
import * as Notifications from 'expo-notifications';
import { Ionicons } from '@expo/vector-icons';

// Screens
import { InboxScreen } from './screens/InboxScreen';
import { EmailDetailScreen } from './screens/EmailDetailScreen';
import { ComposeScreen } from './screens/ComposeScreen';
import { ContactsScreen } from './screens/ContactsScreen';
import { TrustNetworkScreen } from './screens/TrustNetworkScreen';
import { SettingsScreen } from './screens/SettingsScreen';
import { LoginScreen } from './screens/LoginScreen';
import { BiometricLockScreen } from './screens/BiometricLockScreen';

// Providers
import { AuthProvider, useAuth } from './providers/AuthProvider';
import { ThemeProvider, useTheme } from './providers/ThemeProvider';

// Types
import { RootStackParamList, MainTabParamList } from './types/navigation';

// Keep splash screen visible while loading
SplashScreen.preventAutoHideAsync();

// Configure notifications
Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: true,
    shouldSetBadge: true,
  }),
});

const Stack = createNativeStackNavigator<RootStackParamList>();
const Tab = createBottomTabNavigator<MainTabParamList>();

// Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      retry: 2,
    },
  },
});

// Main tab navigator
function MainTabs() {
  const { colors } = useTheme();

  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        tabBarIcon: ({ focused, color, size }) => {
          let iconName: keyof typeof Ionicons.glyphMap;

          switch (route.name) {
            case 'Inbox':
              iconName = focused ? 'mail' : 'mail-outline';
              break;
            case 'Contacts':
              iconName = focused ? 'people' : 'people-outline';
              break;
            case 'Trust':
              iconName = focused ? 'shield-checkmark' : 'shield-checkmark-outline';
              break;
            case 'Settings':
              iconName = focused ? 'settings' : 'settings-outline';
              break;
            default:
              iconName = 'help-outline';
          }

          return <Ionicons name={iconName} size={size} color={color} />;
        },
        tabBarActiveTintColor: colors.primary,
        tabBarInactiveTintColor: colors.textSecondary,
        tabBarStyle: {
          backgroundColor: colors.surface,
          borderTopColor: colors.border,
        },
        headerStyle: {
          backgroundColor: colors.surface,
        },
        headerTintColor: colors.text,
      })}
    >
      <Tab.Screen
        name="Inbox"
        component={InboxScreen}
        options={{ title: 'Inbox' }}
      />
      <Tab.Screen
        name="Contacts"
        component={ContactsScreen}
        options={{ title: 'Contacts' }}
      />
      <Tab.Screen
        name="Trust"
        component={TrustNetworkScreen}
        options={{ title: 'Trust Network' }}
      />
      <Tab.Screen
        name="Settings"
        component={SettingsScreen}
        options={{ title: 'Settings' }}
      />
    </Tab.Navigator>
  );
}

// Root navigator
function RootNavigator() {
  const { isAuthenticated, isLoading, requiresBiometric } = useAuth();
  const { colors } = useTheme();
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    async function prepare() {
      // Load fonts, check auth, etc.
      await new Promise((resolve) => setTimeout(resolve, 1000));
      setIsReady(true);
      await SplashScreen.hideAsync();
    }

    prepare();
  }, []);

  if (!isReady || isLoading) {
    return null;
  }

  return (
    <Stack.Navigator
      screenOptions={{
        headerStyle: { backgroundColor: colors.surface },
        headerTintColor: colors.text,
        contentStyle: { backgroundColor: colors.background },
      }}
    >
      {!isAuthenticated ? (
        <Stack.Screen
          name="Login"
          component={LoginScreen}
          options={{ headerShown: false }}
        />
      ) : requiresBiometric ? (
        <Stack.Screen
          name="BiometricLock"
          component={BiometricLockScreen}
          options={{ headerShown: false }}
        />
      ) : (
        <>
          <Stack.Screen
            name="Main"
            component={MainTabs}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="EmailDetail"
            component={EmailDetailScreen}
            options={({ route }) => ({
              title: route.params?.subject || 'Email',
            })}
          />
          <Stack.Screen
            name="Compose"
            component={ComposeScreen}
            options={{
              title: 'New Email',
              presentation: 'modal',
            }}
          />
        </>
      )}
    </Stack.Navigator>
  );
}

// Main app component
export default function App() {
  useEffect(() => {
    // Request notification permissions
    Notifications.requestPermissionsAsync();

    // Handle notification taps
    const subscription = Notifications.addNotificationResponseReceivedListener(
      (response) => {
        const data = response.notification.request.content.data;
        if (data.emailId) {
          // Navigate to email
          // navigationRef.navigate('EmailDetail', { id: data.emailId });
        }
      }
    );

    return () => subscription.remove();
  }, []);

  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <SafeAreaProvider>
        <QueryClientProvider client={queryClient}>
          <ThemeProvider>
            <AuthProvider>
              <NavigationContainer>
                <StatusBar style="auto" />
                <RootNavigator />
              </NavigationContainer>
            </AuthProvider>
          </ThemeProvider>
        </QueryClientProvider>
      </SafeAreaProvider>
    </GestureHandlerRootView>
  );
}
