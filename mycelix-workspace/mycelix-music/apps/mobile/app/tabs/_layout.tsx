// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Tab Navigation Layout
 *
 * Bottom tab navigation for main app screens.
 */

import { Tabs } from 'expo-router';
import { View, StyleSheet } from 'react-native';
import { BlurView } from 'expo-blur';
import Svg, { Path, Circle } from 'react-native-svg';
import { useTheme } from '../../src/theme/ThemeContext';
import { MiniPlayer } from '../../src/components/Player/MiniPlayer';

// Custom icons for tabs
function HomeIcon({ color, focused }: { color: string; focused: boolean }) {
  return (
    <Svg width={24} height={24} viewBox="0 0 24 24" fill="none">
      <Path
        d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"
        stroke={color}
        strokeWidth={focused ? 2.5 : 2}
        fill={focused ? color + '20' : 'none'}
      />
      <Path d="M9 22V12h6v10" stroke={color} strokeWidth={focused ? 2.5 : 2} />
    </Svg>
  );
}

function LibraryIcon({ color, focused }: { color: string; focused: boolean }) {
  return (
    <Svg width={24} height={24} viewBox="0 0 24 24" fill="none">
      <Path
        d="M4 19V5a2 2 0 012-2h8a2 2 0 012 2v14a2 2 0 01-2 2H6a2 2 0 01-2-2z"
        stroke={color}
        strokeWidth={focused ? 2.5 : 2}
        fill={focused ? color + '20' : 'none'}
      />
      <Path d="M8 7h4M8 11h4M8 15h2" stroke={color} strokeWidth={focused ? 2.5 : 2} strokeLinecap="round" />
      <Path d="M18 3v18" stroke={color} strokeWidth={focused ? 2.5 : 2} strokeLinecap="round" />
      <Path d="M21 6v12" stroke={color} strokeWidth={focused ? 2.5 : 2} strokeLinecap="round" />
    </Svg>
  );
}

function SearchIcon({ color, focused }: { color: string; focused: boolean }) {
  return (
    <Svg width={24} height={24} viewBox="0 0 24 24" fill="none">
      <Circle
        cx={11}
        cy={11}
        r={8}
        stroke={color}
        strokeWidth={focused ? 2.5 : 2}
        fill={focused ? color + '20' : 'none'}
      />
      <Path d="M21 21l-4.35-4.35" stroke={color} strokeWidth={focused ? 2.5 : 2} strokeLinecap="round" />
    </Svg>
  );
}

function ProfileIcon({ color, focused }: { color: string; focused: boolean }) {
  return (
    <Svg width={24} height={24} viewBox="0 0 24 24" fill="none">
      <Circle
        cx={12}
        cy={8}
        r={4}
        stroke={color}
        strokeWidth={focused ? 2.5 : 2}
        fill={focused ? color + '20' : 'none'}
      />
      <Path
        d="M4 21v-2a4 4 0 014-4h8a4 4 0 014 4v2"
        stroke={color}
        strokeWidth={focused ? 2.5 : 2}
        strokeLinecap="round"
      />
    </Svg>
  );
}

export default function TabLayout() {
  const { colors } = useTheme();

  return (
    <View style={styles.container}>
      <Tabs
        screenOptions={{
          headerShown: false,
          tabBarStyle: styles.tabBar,
          tabBarBackground: () => (
            <BlurView intensity={80} tint="dark" style={StyleSheet.absoluteFill} />
          ),
          tabBarActiveTintColor: colors.primary,
          tabBarInactiveTintColor: colors.textMuted,
          tabBarLabelStyle: styles.tabLabel,
        }}
      >
        <Tabs.Screen
          name="index"
          options={{
            title: 'Home',
            tabBarIcon: ({ color, focused }) => <HomeIcon color={color} focused={focused} />,
          }}
        />
        <Tabs.Screen
          name="library"
          options={{
            title: 'Library',
            tabBarIcon: ({ color, focused }) => <LibraryIcon color={color} focused={focused} />,
          }}
        />
        <Tabs.Screen
          name="search"
          options={{
            title: 'Search',
            tabBarIcon: ({ color, focused }) => <SearchIcon color={color} focused={focused} />,
          }}
        />
        <Tabs.Screen
          name="profile"
          options={{
            title: 'Soul',
            tabBarIcon: ({ color, focused }) => <ProfileIcon color={color} focused={focused} />,
          }}
        />
      </Tabs>
      <MiniPlayer />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0a0a',
  },
  tabBar: {
    position: 'absolute',
    backgroundColor: 'transparent',
    borderTopWidth: 0,
    elevation: 0,
    height: 85,
    paddingBottom: 25,
  },
  tabLabel: {
    fontSize: 11,
    fontFamily: 'SpaceGrotesk-Medium',
    marginTop: 4,
  },
});
