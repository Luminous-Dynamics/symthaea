// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { useState, useEffect } from 'react'
import { ethers } from 'ethers'
import './App.css'

// Contract addresses on Sepolia
const CONTRACTS = {
  registry: "0x556b810371e3d8D9E5753117514F03cC6C93b835",
  reputation: "0xf3B343888a9b82274cEfaa15921252DB6c5f48C9",
  payment: "0x94417A3645824CeBDC0872c358cf810e4Ce4D1cB",
}

// Simplified ABIs
const REGISTRY_ABI = [
  "function registerDID(bytes32 did, bytes32 metadataHash) external payable",
  "function didRecords(bytes32 did) external view returns (address owner, uint256 registeredAt, uint256 updatedAt, bytes32 metadataHash, bool revoked)",
  "function totalDids() external view returns (uint256)",
  "function registrationFee() external view returns (uint256)",
]

const REPUTATION_ABI = [
  "function getReputation(address subject) external view returns (int256 score, uint256 submissions, uint256 lastUpdate)",
]

declare global {
  interface Window {
    ethereum?: any;
  }
}

function App() {
  const [account, setAccount] = useState<string | null>(null)
  const [balance, setBalance] = useState<string>('0')
  const [chainId, setChainId] = useState<number | null>(null)
  const [totalDids, setTotalDids] = useState<string>('0')
  const [didInput, setDidInput] = useState('')
  const [metadataInput, setMetadataInput] = useState('')
  const [txHash, setTxHash] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [lookupDid, setLookupDid] = useState('')
  const [didInfo, setDidInfo] = useState<any>(null)
  const [reputation, setReputation] = useState<any>(null)

  const isSepoliaNetwork = chainId === 11155111

  useEffect(() => {
    checkConnection()
    loadContractData()
  }, [])

  useEffect(() => {
    if (account) {
      loadContractData()
      loadReputation(account)
    }
  }, [account])

  async function checkConnection() {
    if (window.ethereum) {
      try {
        const accounts = await window.ethereum.request({ method: 'eth_accounts' })
        if (accounts.length > 0) {
          setAccount(accounts[0])
          const balance = await window.ethereum.request({
            method: 'eth_getBalance',
            params: [accounts[0], 'latest']
          })
          setBalance(ethers.formatEther(balance))
          const chainId = await window.ethereum.request({ method: 'eth_chainId' })
          setChainId(parseInt(chainId, 16))
        }
      } catch (err) {
        console.error(err)
      }
    }
  }

  async function connectWallet() {
    if (!window.ethereum) {
      alert('Please install MetaMask!')
      return
    }
    try {
      const accounts = await window.ethereum.request({ method: 'eth_requestAccounts' })
      setAccount(accounts[0])
      const balance = await window.ethereum.request({
        method: 'eth_getBalance',
        params: [accounts[0], 'latest']
      })
      setBalance(ethers.formatEther(balance))
      const chainId = await window.ethereum.request({ method: 'eth_chainId' })
      setChainId(parseInt(chainId, 16))
    } catch (err) {
      console.error(err)
    }
  }

  async function switchToSepolia() {
    try {
      await window.ethereum.request({
        method: 'wallet_switchEthereumChain',
        params: [{ chainId: '0xaa36a7' }],
      })
      setChainId(11155111)
    } catch (err: any) {
      if (err.code === 4902) {
        await window.ethereum.request({
          method: 'wallet_addEthereumChain',
          params: [{
            chainId: '0xaa36a7',
            chainName: 'Sepolia',
            rpcUrls: ['https://ethereum-sepolia-rpc.publicnode.com'],
            nativeCurrency: { name: 'ETH', symbol: 'ETH', decimals: 18 },
            blockExplorerUrls: ['https://sepolia.etherscan.io']
          }]
        })
      }
    }
  }

  async function loadContractData() {
    try {
      const provider = new ethers.JsonRpcProvider('https://ethereum-sepolia-rpc.publicnode.com')
      const registry = new ethers.Contract(CONTRACTS.registry, REGISTRY_ABI, provider)
      const total = await registry.totalDids()
      setTotalDids(total.toString())
    } catch (err) {
      console.error('Failed to load contract data:', err)
    }
  }

  async function loadReputation(address: string) {
    try {
      const provider = new ethers.JsonRpcProvider('https://ethereum-sepolia-rpc.publicnode.com')
      const reputation = new ethers.Contract(CONTRACTS.reputation, REPUTATION_ABI, provider)
      const [score, submissions, lastUpdate] = await reputation.getReputation(address)
      setReputation({ score: score.toString(), submissions: submissions.toString(), lastUpdate: lastUpdate.toString() })
    } catch (err) {
      console.error('Failed to load reputation:', err)
    }
  }

  async function registerDID() {
    if (!account || !didInput || !metadataInput) return
    if (!isSepoliaNetwork) {
      alert('Please switch to Sepolia network')
      return
    }

    setLoading(true)
    setTxHash(null)

    try {
      const provider = new ethers.BrowserProvider(window.ethereum)
      const signer = await provider.getSigner()
      const registry = new ethers.Contract(CONTRACTS.registry, REGISTRY_ABI, signer)

      const didHash = ethers.keccak256(ethers.toUtf8Bytes(didInput))
      const metadataHash = ethers.keccak256(ethers.toUtf8Bytes(metadataInput))
      const fee = await registry.registrationFee()

      const tx = await registry.registerDID(didHash, metadataHash, { value: fee })
      setTxHash(tx.hash)
      await tx.wait()
      loadContractData()
      alert('DID registered successfully!')
    } catch (err: any) {
      console.error(err)
      alert('Error: ' + (err.message || err))
    } finally {
      setLoading(false)
    }
  }

  async function lookupDID() {
    if (!lookupDid) return
    try {
      const provider = new ethers.JsonRpcProvider('https://ethereum-sepolia-rpc.publicnode.com')
      const registry = new ethers.Contract(CONTRACTS.registry, REGISTRY_ABI, provider)
      const didHash = ethers.keccak256(ethers.toUtf8Bytes(lookupDid))
      const [owner, registeredAt, , metadataHash, revoked] = await registry.didRecords(didHash)

      if (owner === ethers.ZeroAddress) {
        setDidInfo({ notFound: true })
      } else {
        setDidInfo({
          owner,
          registeredAt: new Date(Number(registeredAt) * 1000).toLocaleString(),
          metadataHash,
          revoked
        })
      }
    } catch (err) {
      console.error(err)
    }
  }

  return (
    <div className="app">
      <header>
        <h1>🍄 Mycelix</h1>
        <p>Decentralized Identity & Reputation on Sepolia</p>
      </header>

      <section className="wallet-section">
        {!account ? (
          <button className="connect-btn" onClick={connectWallet}>
            Connect Wallet
          </button>
        ) : (
          <div className="wallet-info">
            <p><strong>Connected:</strong> {account.slice(0, 6)}...{account.slice(-4)}</p>
            <p><strong>Balance:</strong> {parseFloat(balance).toFixed(4)} ETH</p>
            <p><strong>Network:</strong> {isSepoliaNetwork ? '✅ Sepolia' : '❌ Wrong Network'}</p>
            {!isSepoliaNetwork && (
              <button onClick={switchToSepolia}>Switch to Sepolia</button>
            )}
          </div>
        )}
      </section>

      <section className="stats-section">
        <h2>Network Stats</h2>
        <div className="stats-grid">
          <div className="stat-card">
            <h3>{totalDids}</h3>
            <p>Total DIDs</p>
          </div>
          <div className="stat-card">
            <h3>{reputation?.submissions || '0'}</h3>
            <p>Your Reputation Submissions</p>
          </div>
        </div>
      </section>

      <section className="register-section">
        <h2>Register DID</h2>
        <div className="form">
          <input
            type="text"
            placeholder="did:mycelix:your-identifier"
            value={didInput}
            onChange={(e) => setDidInput(e.target.value)}
          />
          <input
            type="text"
            placeholder="ipfs://your-metadata-uri"
            value={metadataInput}
            onChange={(e) => setMetadataInput(e.target.value)}
          />
          <button
            onClick={registerDID}
            disabled={loading || !account || !isSepoliaNetwork}
          >
            {loading ? 'Registering...' : 'Register DID'}
          </button>
          {txHash && (
            <p className="tx-link">
              TX: <a href={`https://sepolia.etherscan.io/tx/${txHash}`} target="_blank" rel="noopener noreferrer">
                {txHash.slice(0, 10)}...
              </a>
            </p>
          )}
        </div>
      </section>

      <section className="lookup-section">
        <h2>Lookup DID</h2>
        <div className="form">
          <input
            type="text"
            placeholder="did:mycelix:identifier-to-lookup"
            value={lookupDid}
            onChange={(e) => setLookupDid(e.target.value)}
          />
          <button onClick={lookupDID}>Lookup</button>
        </div>
        {didInfo && (
          <div className="did-result">
            {didInfo.notFound ? (
              <p>DID not found</p>
            ) : (
              <>
                <p><strong>Owner:</strong> {didInfo.owner}</p>
                <p><strong>Registered:</strong> {didInfo.registeredAt}</p>
                <p><strong>Revoked:</strong> {didInfo.revoked ? 'Yes' : 'No'}</p>
              </>
            )}
          </div>
        )}
      </section>

      <section className="contracts-section">
        <h2>Contract Addresses</h2>
        <div className="contracts-list">
          <p>
            <strong>Registry:</strong>{' '}
            <a href={`https://sepolia.etherscan.io/address/${CONTRACTS.registry}`} target="_blank">
              {CONTRACTS.registry.slice(0, 10)}...
            </a>
          </p>
          <p>
            <strong>Reputation:</strong>{' '}
            <a href={`https://sepolia.etherscan.io/address/${CONTRACTS.reputation}`} target="_blank">
              {CONTRACTS.reputation.slice(0, 10)}...
            </a>
          </p>
          <p>
            <strong>Payment:</strong>{' '}
            <a href={`https://sepolia.etherscan.io/address/${CONTRACTS.payment}`} target="_blank">
              {CONTRACTS.payment.slice(0, 10)}...
            </a>
          </p>
        </div>
      </section>

      <footer>
        <p>Mycelix Network - Decentralized ML with Byzantine Resistance</p>
        <p>
          <a href="https://github.com/Luminous-Dynamics/Mycelix-Core" target="_blank">GitHub</a>
          {' | '}
          <a href="https://sourcify.dev/#/lookup/0x556b810371e3d8D9E5753117514F03cC6C93b835" target="_blank">Verified Source</a>
        </p>
      </footer>
    </div>
  )
}

export default App
