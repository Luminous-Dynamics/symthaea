// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Trust Network Screen
 *
 * Web-of-trust visualization and attestation management
 */

import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
  RefreshControl,
  Dimensions,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import * as Haptics from 'expo-haptics';
import Svg, { Circle, Line, Text as SvgText, G } from 'react-native-svg';

import { useTheme } from '../providers/ThemeProvider';
import { api } from '../services/api';
import { Avatar } from '../components/Avatar';
import { TrustBadge } from '../components/TrustBadge';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

interface TrustNode {
  id: string;
  email: string;
  name?: string;
  trustScore: number;
  attestationsGiven: number;
  attestationsReceived: number;
  distance: number; // 0 = self, 1 = direct, 2+ = indirect
}

interface Attestation {
  id: string;
  from: string;
  fromName?: string;
  to: string;
  toName?: string;
  level: number;
  context: string;
  createdAt: string;
  expiresAt?: string;
}

type TabType = 'network' | 'received' | 'given';

export function TrustNetworkScreen() {
  const { colors } = useTheme();
  const queryClient = useQueryClient();

  const [activeTab, setActiveTab] = useState<TabType>('network');
  const [refreshing, setRefreshing] = useState(false);
  const [selectedNode, setSelectedNode] = useState<TrustNode | null>(null);

  // Fetch trust network
  const { data: network } = useQuery({
    queryKey: ['trustNetwork'],
    queryFn: () => api.getTrustNetwork(),
  });

  // Fetch attestations
  const { data: attestations } = useQuery({
    queryKey: ['attestations'],
    queryFn: () => api.getAttestations(),
  });

  // Revoke attestation mutation
  const revokeMutation = useMutation({
    mutationFn: (id: string) => api.revokeAttestation(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['attestations'] });
      queryClient.invalidateQueries({ queryKey: ['trustNetwork'] });
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    },
  });

  const handleRefresh = useCallback(async () => {
    setRefreshing(true);
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ['trustNetwork'] }),
      queryClient.invalidateQueries({ queryKey: ['attestations'] }),
    ]);
    setRefreshing(false);
  }, [queryClient]);

  const handleCreateAttestation = useCallback(() => {
    Alert.alert(
      'Create Attestation',
      'Attestation creation is available in the full app',
      [{ text: 'OK' }]
    );
  }, []);

  const handleRevokeAttestation = useCallback((attestation: Attestation) => {
    Alert.alert(
      'Revoke Attestation',
      `Are you sure you want to revoke your trust attestation for ${attestation.toName || attestation.to}?`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Revoke',
          style: 'destructive',
          onPress: () => revokeMutation.mutate(attestation.id),
        },
      ]
    );
  }, [revokeMutation]);

  const getTrustColor = useCallback((score: number) => {
    if (score >= 0.8) return '#22c55e';
    if (score >= 0.5) return '#eab308';
    if (score >= 0.2) return '#f97316';
    return '#ef4444';
  }, []);

  const renderNetworkGraph = useCallback(() => {
    if (!network?.nodes?.length) {
      return (
        <View style={styles.emptyContainer}>
          <Ionicons name="git-network-outline" size={64} color={colors.textSecondary} />
          <Text style={[styles.emptyText, { color: colors.textSecondary }]}>
            Your trust network will appear here
          </Text>
        </View>
      );
    }

    const graphSize = SCREEN_WIDTH - 32;
    const center = graphSize / 2;
    const nodes = network.nodes.slice(0, 20); // Limit for performance

    // Calculate positions in concentric circles
    const nodePositions = nodes.map((node: TrustNode, index: number) => {
      if (node.distance === 0) {
        return { x: center, y: center, node };
      }
      const ring = node.distance;
      const nodesInRing = nodes.filter((n: TrustNode) => n.distance === ring).length;
      const indexInRing = nodes.filter((n: TrustNode) => n.distance === ring).indexOf(node);
      const angle = (indexInRing / nodesInRing) * 2 * Math.PI - Math.PI / 2;
      const radius = ring * 70;
      return {
        x: center + radius * Math.cos(angle),
        y: center + radius * Math.sin(angle),
        node,
      };
    });

    return (
      <View style={styles.graphContainer}>
        <Svg width={graphSize} height={graphSize}>
          {/* Draw connections */}
          {network.edges?.map((edge: { from: string; to: string }, i: number) => {
            const fromPos = nodePositions.find((p: { node: TrustNode }) => p.node.id === edge.from);
            const toPos = nodePositions.find((p: { node: TrustNode }) => p.node.id === edge.to);
            if (fromPos && toPos) {
              return (
                <Line
                  key={`edge-${i}`}
                  x1={fromPos.x}
                  y1={fromPos.y}
                  x2={toPos.x}
                  y2={toPos.y}
                  stroke={colors.border}
                  strokeWidth={1}
                  opacity={0.5}
                />
              );
            }
            return null;
          })}

          {/* Draw nodes */}
          {nodePositions.map(({ x, y, node }: { x: number; y: number; node: TrustNode }, i: number) => (
            <G key={node.id}>
              <Circle
                cx={x}
                cy={y}
                r={node.distance === 0 ? 25 : 18}
                fill={getTrustColor(node.trustScore)}
                opacity={0.9}
                onPress={() => setSelectedNode(node)}
              />
              <SvgText
                x={x}
                y={y + 4}
                textAnchor="middle"
                fontSize={10}
                fill="#fff"
                fontWeight="bold"
              >
                {(node.name || node.email)[0].toUpperCase()}
              </SvgText>
            </G>
          ))}
        </Svg>

        {/* Legend */}
        <View style={styles.legend}>
          <View style={styles.legendItem}>
            <View style={[styles.legendDot, { backgroundColor: '#22c55e' }]} />
            <Text style={[styles.legendText, { color: colors.textSecondary }]}>High Trust</Text>
          </View>
          <View style={styles.legendItem}>
            <View style={[styles.legendDot, { backgroundColor: '#eab308' }]} />
            <Text style={[styles.legendText, { color: colors.textSecondary }]}>Medium</Text>
          </View>
          <View style={styles.legendItem}>
            <View style={[styles.legendDot, { backgroundColor: '#f97316' }]} />
            <Text style={[styles.legendText, { color: colors.textSecondary }]}>Low</Text>
          </View>
        </View>
      </View>
    );
  }, [network, colors, getTrustColor]);

  const renderAttestation = useCallback(
    (attestation: Attestation, type: 'given' | 'received') => (
      <View
        key={attestation.id}
        style={[styles.attestationCard, { backgroundColor: colors.surface }]}
      >
        <View style={styles.attestationHeader}>
          <Avatar
            email={type === 'given' ? attestation.to : attestation.from}
            name={type === 'given' ? attestation.toName : attestation.fromName}
            size={40}
          />
          <View style={styles.attestationInfo}>
            <Text style={[styles.attestationName, { color: colors.text }]}>
              {type === 'given'
                ? attestation.toName || attestation.to
                : attestation.fromName || attestation.from}
            </Text>
            <Text style={[styles.attestationEmail, { color: colors.textSecondary }]}>
              {type === 'given' ? attestation.to : attestation.from}
            </Text>
          </View>
          <TrustBadge score={attestation.level / 5} />
        </View>

        <View style={styles.attestationDetails}>
          <View style={styles.attestationRow}>
            <Ionicons name="pricetag-outline" size={16} color={colors.textSecondary} />
            <Text style={[styles.attestationContext, { color: colors.text }]}>
              {attestation.context}
            </Text>
          </View>
          <View style={styles.attestationRow}>
            <Ionicons name="calendar-outline" size={16} color={colors.textSecondary} />
            <Text style={[styles.attestationDate, { color: colors.textSecondary }]}>
              {new Date(attestation.createdAt).toLocaleDateString()}
            </Text>
          </View>
        </View>

        {type === 'given' && (
          <TouchableOpacity
            style={[styles.revokeButton, { borderColor: colors.error }]}
            onPress={() => handleRevokeAttestation(attestation)}
          >
            <Text style={[styles.revokeText, { color: colors.error }]}>Revoke</Text>
          </TouchableOpacity>
        )}
      </View>
    ),
    [colors, handleRevokeAttestation]
  );

  const styles = StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: colors.background,
    },
    tabs: {
      flexDirection: 'row',
      backgroundColor: colors.surface,
      borderBottomWidth: 1,
      borderBottomColor: colors.border,
    },
    tab: {
      flex: 1,
      paddingVertical: 14,
      alignItems: 'center',
    },
    activeTab: {
      borderBottomWidth: 2,
      borderBottomColor: colors.primary,
    },
    tabText: {
      fontSize: 14,
      fontWeight: '500',
      color: colors.textSecondary,
    },
    activeTabText: {
      color: colors.primary,
    },
    scrollView: {
      flex: 1,
    },
    content: {
      padding: 16,
    },
    statsRow: {
      flexDirection: 'row',
      marginBottom: 20,
    },
    statCard: {
      flex: 1,
      padding: 16,
      borderRadius: 12,
      marginHorizontal: 4,
      alignItems: 'center',
    },
    statValue: {
      fontSize: 24,
      fontWeight: '700',
      color: colors.text,
    },
    statLabel: {
      fontSize: 12,
      color: colors.textSecondary,
      marginTop: 4,
    },
    graphContainer: {
      alignItems: 'center',
      marginBottom: 20,
    },
    legend: {
      flexDirection: 'row',
      justifyContent: 'center',
      marginTop: 16,
    },
    legendItem: {
      flexDirection: 'row',
      alignItems: 'center',
      marginHorizontal: 12,
    },
    legendDot: {
      width: 12,
      height: 12,
      borderRadius: 6,
      marginRight: 6,
    },
    legendText: {
      fontSize: 12,
    },
    emptyContainer: {
      alignItems: 'center',
      padding: 40,
    },
    emptyText: {
      fontSize: 16,
      textAlign: 'center',
      marginTop: 16,
    },
    sectionTitle: {
      fontSize: 18,
      fontWeight: '600',
      color: colors.text,
      marginBottom: 12,
      marginTop: 8,
    },
    attestationCard: {
      padding: 16,
      borderRadius: 12,
      marginBottom: 12,
    },
    attestationHeader: {
      flexDirection: 'row',
      alignItems: 'center',
    },
    attestationInfo: {
      flex: 1,
      marginLeft: 12,
    },
    attestationName: {
      fontSize: 16,
      fontWeight: '500',
    },
    attestationEmail: {
      fontSize: 13,
      marginTop: 2,
    },
    attestationDetails: {
      marginTop: 12,
      paddingTop: 12,
      borderTopWidth: 1,
      borderTopColor: colors.border,
    },
    attestationRow: {
      flexDirection: 'row',
      alignItems: 'center',
      marginBottom: 6,
    },
    attestationContext: {
      fontSize: 14,
      marginLeft: 8,
    },
    attestationDate: {
      fontSize: 13,
      marginLeft: 8,
    },
    revokeButton: {
      marginTop: 12,
      paddingVertical: 8,
      borderRadius: 8,
      borderWidth: 1,
      alignItems: 'center',
    },
    revokeText: {
      fontSize: 14,
      fontWeight: '500',
    },
    fab: {
      position: 'absolute',
      bottom: 24,
      right: 24,
      width: 56,
      height: 56,
      borderRadius: 28,
      backgroundColor: colors.primary,
      alignItems: 'center',
      justifyContent: 'center',
      elevation: 4,
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.25,
      shadowRadius: 4,
    },
    selectedNodeCard: {
      backgroundColor: colors.surface,
      padding: 16,
      borderRadius: 12,
      marginBottom: 16,
    },
    selectedNodeHeader: {
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'space-between',
    },
    selectedNodeName: {
      fontSize: 18,
      fontWeight: '600',
      color: colors.text,
    },
    selectedNodeEmail: {
      fontSize: 14,
      color: colors.textSecondary,
    },
    selectedNodeStats: {
      flexDirection: 'row',
      marginTop: 12,
    },
    selectedNodeStat: {
      flex: 1,
      alignItems: 'center',
    },
    selectedNodeStatValue: {
      fontSize: 20,
      fontWeight: '600',
      color: colors.text,
    },
    selectedNodeStatLabel: {
      fontSize: 11,
      color: colors.textSecondary,
    },
  });

  const receivedAttestations = attestations?.filter((a: Attestation) => a.to === network?.self?.email) || [];
  const givenAttestations = attestations?.filter((a: Attestation) => a.from === network?.self?.email) || [];

  return (
    <SafeAreaView style={styles.container} edges={['bottom']}>
      {/* Tabs */}
      <View style={styles.tabs}>
        {(['network', 'received', 'given'] as TabType[]).map(tab => (
          <TouchableOpacity
            key={tab}
            style={[styles.tab, activeTab === tab && styles.activeTab]}
            onPress={() => setActiveTab(tab)}
          >
            <Text style={[styles.tabText, activeTab === tab && styles.activeTabText]}>
              {tab === 'network' ? 'Network' : tab === 'received' ? 'Received' : 'Given'}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      <ScrollView
        style={styles.scrollView}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={handleRefresh}
            tintColor={colors.primary}
          />
        }
      >
        <View style={styles.content}>
          {/* Stats */}
          <View style={styles.statsRow}>
            <View style={[styles.statCard, { backgroundColor: colors.surface }]}>
              <Text style={styles.statValue}>{network?.nodes?.length || 0}</Text>
              <Text style={styles.statLabel}>Network Size</Text>
            </View>
            <View style={[styles.statCard, { backgroundColor: colors.surface }]}>
              <Text style={styles.statValue}>{givenAttestations.length}</Text>
              <Text style={styles.statLabel}>Given</Text>
            </View>
            <View style={[styles.statCard, { backgroundColor: colors.surface }]}>
              <Text style={styles.statValue}>{receivedAttestations.length}</Text>
              <Text style={styles.statLabel}>Received</Text>
            </View>
          </View>

          {/* Selected Node Info */}
          {selectedNode && (
            <View style={styles.selectedNodeCard}>
              <View style={styles.selectedNodeHeader}>
                <View>
                  <Text style={styles.selectedNodeName}>
                    {selectedNode.name || selectedNode.email}
                  </Text>
                  <Text style={styles.selectedNodeEmail}>{selectedNode.email}</Text>
                </View>
                <TouchableOpacity onPress={() => setSelectedNode(null)}>
                  <Ionicons name="close" size={24} color={colors.textSecondary} />
                </TouchableOpacity>
              </View>
              <View style={styles.selectedNodeStats}>
                <View style={styles.selectedNodeStat}>
                  <Text style={styles.selectedNodeStatValue}>
                    {Math.round(selectedNode.trustScore * 100)}%
                  </Text>
                  <Text style={styles.selectedNodeStatLabel}>Trust Score</Text>
                </View>
                <View style={styles.selectedNodeStat}>
                  <Text style={styles.selectedNodeStatValue}>
                    {selectedNode.distance}
                  </Text>
                  <Text style={styles.selectedNodeStatLabel}>Degrees</Text>
                </View>
                <View style={styles.selectedNodeStat}>
                  <Text style={styles.selectedNodeStatValue}>
                    {selectedNode.attestationsReceived}
                  </Text>
                  <Text style={styles.selectedNodeStatLabel}>Attestations</Text>
                </View>
              </View>
            </View>
          )}

          {/* Tab Content */}
          {activeTab === 'network' && renderNetworkGraph()}

          {activeTab === 'received' && (
            <>
              <Text style={styles.sectionTitle}>Trust Received</Text>
              {receivedAttestations.length > 0 ? (
                receivedAttestations.map((a: Attestation) => renderAttestation(a, 'received'))
              ) : (
                <View style={styles.emptyContainer}>
                  <Ionicons name="ribbon-outline" size={48} color={colors.textSecondary} />
                  <Text style={[styles.emptyText, { color: colors.textSecondary }]}>
                    No attestations received yet
                  </Text>
                </View>
              )}
            </>
          )}

          {activeTab === 'given' && (
            <>
              <Text style={styles.sectionTitle}>Trust Given</Text>
              {givenAttestations.length > 0 ? (
                givenAttestations.map((a: Attestation) => renderAttestation(a, 'given'))
              ) : (
                <View style={styles.emptyContainer}>
                  <Ionicons name="ribbon-outline" size={48} color={colors.textSecondary} />
                  <Text style={[styles.emptyText, { color: colors.textSecondary }]}>
                    No attestations given yet
                  </Text>
                </View>
              )}
            </>
          )}
        </View>
      </ScrollView>

      {/* Create Attestation FAB */}
      <TouchableOpacity style={styles.fab} onPress={handleCreateAttestation}>
        <Ionicons name="add" size={28} color="#fff" />
      </TouchableOpacity>
    </SafeAreaView>
  );
}
