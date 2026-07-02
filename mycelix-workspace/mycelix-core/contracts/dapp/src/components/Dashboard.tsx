// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { useState, useEffect } from 'react'
import { ethers } from 'ethers'

interface DashboardProps {
  provider: ethers.JsonRpcProvider
  registryAddress: string
}

interface NetworkStats {
  totalDids: string
  registrationFee: string
  blockNumber: number
  gasPrice: string
}

interface RecentDID {
  did: string
  owner: string
  blockNumber: number
  txHash: string
}

const REGISTRY_ABI = [
  "function totalDids() external view returns (uint256)",
  "function registrationFee() external view returns (uint256)",
  "event DIDRegistered(bytes32 indexed did, address indexed owner)",
]

export function Dashboard({ provider, registryAddress }: DashboardProps) {
  const [stats, setStats] = useState<NetworkStats | null>(null)
  const [recentDids, setRecentDids] = useState<RecentDID[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadStats()
    loadRecentDids()
    const interval = setInterval(loadStats, 30000) // Refresh every 30s
    return () => clearInterval(interval)
  }, [])

  async function loadStats() {
    try {
      const registry = new ethers.Contract(registryAddress, REGISTRY_ABI, provider)
      const [totalDids, registrationFee, blockNumber, feeData] = await Promise.all([
        registry.totalDids(),
        registry.registrationFee(),
        provider.getBlockNumber(),
        provider.getFeeData(),
      ])

      setStats({
        totalDids: totalDids.toString(),
        registrationFee: ethers.formatEther(registrationFee),
        blockNumber,
        gasPrice: feeData.gasPrice ? ethers.formatUnits(feeData.gasPrice, 'gwei') : '0',
      })
    } catch (err) {
      console.error('Failed to load stats:', err)
    } finally {
      setLoading(false)
    }
  }

  async function loadRecentDids() {
    try {
      const registry = new ethers.Contract(registryAddress, REGISTRY_ABI, provider)
      const currentBlock = await provider.getBlockNumber()
      const fromBlock = Math.max(0, currentBlock - 10000) // Last ~10k blocks

      const filter = registry.filters.DIDRegistered()
      const events = await registry.queryFilter(filter, fromBlock, 'latest')

      const recent = events.slice(-5).reverse().map((event: any) => ({
        did: event.args[0],
        owner: event.args[1],
        blockNumber: event.blockNumber,
        txHash: event.transactionHash,
      }))

      setRecentDids(recent)
    } catch (err) {
      console.error('Failed to load recent DIDs:', err)
    }
  }

  if (loading) {
    return <div className="dashboard loading">Loading network stats...</div>
  }

  return (
    <div className="dashboard">
      <h2>Network Dashboard</h2>

      <div className="stats-grid">
        <div className="stat-card">
          <h3>{stats?.totalDids || '0'}</h3>
          <p>Total DIDs</p>
        </div>
        <div className="stat-card">
          <h3>{stats?.registrationFee || '0'}</h3>
          <p>Reg. Fee (ETH)</p>
        </div>
        <div className="stat-card">
          <h3>{stats?.blockNumber?.toLocaleString() || '0'}</h3>
          <p>Block Height</p>
        </div>
        <div className="stat-card">
          <h3>{parseFloat(stats?.gasPrice || '0').toFixed(2)}</h3>
          <p>Gas (Gwei)</p>
        </div>
      </div>

      {recentDids.length > 0 && (
        <div className="recent-activity">
          <h3>Recent Registrations</h3>
          <div className="activity-list">
            {recentDids.map((did, i) => (
              <div key={i} className="activity-item">
                <div className="activity-info">
                  <span className="activity-did" title={did.did}>
                    {did.did.slice(0, 10)}...{did.did.slice(-6)}
                  </span>
                  <span className="activity-owner">
                    by {did.owner.slice(0, 6)}...{did.owner.slice(-4)}
                  </span>
                </div>
                <a
                  href={`https://sepolia.etherscan.io/tx/${did.txHash}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="activity-link"
                >
                  Block {did.blockNumber}
                </a>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default Dashboard
