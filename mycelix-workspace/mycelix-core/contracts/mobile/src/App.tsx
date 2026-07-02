// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Mobile App
 * React Native client for Mycelix smart contracts
 */

import React, { useState, useEffect } from 'react';
import {
  SafeAreaView,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  View,
  TextInput,
  TouchableOpacity,
  Alert,
} from 'react-native';
import { ethers } from 'ethers';

// Contract configuration
const CONTRACTS = {
  registry: '0x556b810371e3d8D9E5753117514F03cC6C93b835',
  reputation: '0xf3B343888a9b82274cEfaa15921252DB6c5f48C9',
  payment: '0x94417A3645824CeBDC0872c358cf810e4Ce4D1cB',
};

const RPC_URL = 'https://ethereum-sepolia-rpc.publicnode.com';

const REGISTRY_ABI = [
  'function totalDids() external view returns (uint256)',
  'function registrationFee() external view returns (uint256)',
  'function didRecords(bytes32 did) external view returns (address owner, uint256 registeredAt, uint256 updatedAt, bytes32 metadataHash, bool revoked)',
];

interface NetworkStats {
  totalDids: string;
  registrationFee: string;
  blockNumber: number;
}

function App(): React.JSX.Element {
  const [stats, setStats] = useState<NetworkStats | null>(null);
  const [lookupDid, setLookupDid] = useState('');
  const [didInfo, setDidInfo] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadStats();
  }, []);

  async function loadStats() {
    try {
      const provider = new ethers.JsonRpcProvider(RPC_URL);
      const registry = new ethers.Contract(CONTRACTS.registry, REGISTRY_ABI, provider);

      const [totalDids, registrationFee, blockNumber] = await Promise.all([
        registry.totalDids(),
        registry.registrationFee(),
        provider.getBlockNumber(),
      ]);

      setStats({
        totalDids: totalDids.toString(),
        registrationFee: ethers.formatEther(registrationFee),
        blockNumber,
      });
    } catch (err) {
      console.error('Failed to load stats:', err);
    }
  }

  async function handleLookup() {
    if (!lookupDid.trim()) {
      Alert.alert('Error', 'Please enter a DID to lookup');
      return;
    }

    setLoading(true);
    try {
      const provider = new ethers.JsonRpcProvider(RPC_URL);
      const registry = new ethers.Contract(CONTRACTS.registry, REGISTRY_ABI, provider);
      const didHash = ethers.keccak256(ethers.toUtf8Bytes(lookupDid));
      const [owner, registeredAt, , metadataHash, revoked] = await registry.didRecords(didHash);

      if (owner === ethers.ZeroAddress) {
        setDidInfo({ notFound: true });
      } else {
        setDidInfo({
          owner,
          registeredAt: new Date(Number(registeredAt) * 1000).toLocaleString(),
          metadataHash,
          revoked,
        });
      }
    } catch (err: any) {
      Alert.alert('Error', err.message || 'Failed to lookup DID');
    } finally {
      setLoading(false);
    }
  }

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#0F0F23" />
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <Text style={styles.title}>🍄 Mycelix</Text>
          <Text style={styles.subtitle}>Decentralized Identity on Sepolia</Text>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Network Stats</Text>
          <View style={styles.statsGrid}>
            <View style={styles.statCard}>
              <Text style={styles.statValue}>{stats?.totalDids || '...'}</Text>
              <Text style={styles.statLabel}>Total DIDs</Text>
            </View>
            <View style={styles.statCard}>
              <Text style={styles.statValue}>{stats?.blockNumber?.toLocaleString() || '...'}</Text>
              <Text style={styles.statLabel}>Block Height</Text>
            </View>
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Lookup DID</Text>
          <TextInput
            style={styles.input}
            placeholder="did:mycelix:identifier"
            placeholderTextColor="#666"
            value={lookupDid}
            onChangeText={setLookupDid}
            autoCapitalize="none"
            autoCorrect={false}
          />
          <TouchableOpacity
            style={[styles.button, loading && styles.buttonDisabled]}
            onPress={handleLookup}
            disabled={loading}
          >
            <Text style={styles.buttonText}>{loading ? 'Looking up...' : 'Lookup'}</Text>
          </TouchableOpacity>

          {didInfo && (
            <View style={styles.resultCard}>
              {didInfo.notFound ? (
                <Text style={styles.resultText}>DID not found</Text>
              ) : (
                <>
                  <Text style={styles.resultLabel}>Owner</Text>
                  <Text style={styles.resultValue}>{didInfo.owner}</Text>
                  <Text style={styles.resultLabel}>Registered</Text>
                  <Text style={styles.resultValue}>{didInfo.registeredAt}</Text>
                  <Text style={styles.resultLabel}>Revoked</Text>
                  <Text style={styles.resultValue}>{didInfo.revoked ? 'Yes' : 'No'}</Text>
                </>
              )}
            </View>
          )}
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Contracts</Text>
          <Text style={styles.contractText}>Registry: {CONTRACTS.registry.slice(0, 10)}...</Text>
          <Text style={styles.contractText}>Reputation: {CONTRACTS.reputation.slice(0, 10)}...</Text>
          <Text style={styles.contractText}>Payment: {CONTRACTS.payment.slice(0, 10)}...</Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0F0F23',
  },
  scrollContent: {
    padding: 20,
  },
  header: {
    alignItems: 'center',
    marginBottom: 30,
  },
  title: {
    fontSize: 36,
    fontWeight: 'bold',
    color: '#8B5CF6',
  },
  subtitle: {
    fontSize: 16,
    color: '#94A3B8',
    marginTop: 5,
  },
  section: {
    backgroundColor: '#1A1A2E',
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#E2E8F0',
    marginBottom: 15,
  },
  statsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  statCard: {
    flex: 1,
    backgroundColor: '#252542',
    borderRadius: 12,
    padding: 15,
    marginHorizontal: 5,
    alignItems: 'center',
  },
  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#8B5CF6',
  },
  statLabel: {
    fontSize: 12,
    color: '#94A3B8',
    marginTop: 5,
  },
  input: {
    backgroundColor: '#252542',
    borderRadius: 8,
    padding: 15,
    fontSize: 16,
    color: '#E2E8F0',
    marginBottom: 15,
  },
  button: {
    backgroundColor: '#8B5CF6',
    borderRadius: 8,
    padding: 15,
    alignItems: 'center',
  },
  buttonDisabled: {
    opacity: 0.5,
  },
  buttonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
  resultCard: {
    backgroundColor: '#252542',
    borderRadius: 8,
    padding: 15,
    marginTop: 15,
  },
  resultText: {
    color: '#94A3B8',
    fontSize: 14,
  },
  resultLabel: {
    color: '#94A3B8',
    fontSize: 12,
    marginTop: 10,
  },
  resultValue: {
    color: '#E2E8F0',
    fontSize: 14,
    marginTop: 2,
  },
  contractText: {
    color: '#8B5CF6',
    fontSize: 12,
    fontFamily: 'monospace',
    marginBottom: 5,
  },
});

export default App;
