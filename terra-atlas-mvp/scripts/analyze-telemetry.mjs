#!/usr/bin/env node
/**
 * Quick analysis for telemetry events captured in data/telemetry-events.log.
 *
 * Usage:
 *   npm run telemetry:report           # defaults to data/telemetry-events.log
 *   npm run telemetry:report -- custom-log-path.log
 */

import { createReadStream, existsSync } from 'fs'
import { resolve } from 'path'
import readline from 'readline'

const defaultLogPath = resolve(process.cwd(), 'data', 'telemetry-events.log')
const providedPath = process.argv[2] && process.argv[2] !== '--' ? process.argv[2] : undefined
const logFile = resolve(process.cwd(), providedPath ?? defaultLogPath)

if (!existsSync(logFile)) {
  console.log(`No telemetry log found at ${logFile}. Record some sessions first.`)
  process.exit(0)
}

function createInterface(file) {
  const stream = createReadStream(file, { encoding: 'utf8' })
  return readline.createInterface({
    input: stream,
    crlfDelay: Infinity,
  })
}

async function analyze() {
  const rl = createInterface(logFile)
  const records = []

  for await (const line of rl) {
    if (!line.trim()) continue
    try {
      const parsed = JSON.parse(line)
      if (parsed?.type === 'globe_performance') {
        records.push(parsed)
      }
    } catch (error) {
      console.warn('⚠️  Skipping malformed line:', line.slice(0, 80))
    }
  }

  if (records.length === 0) {
    console.log(`No globe performance entries found in ${logFile}`)
    return
  }

  const totals = {
    mount: 0,
    init: 0,
    markers: 0,
    total: 0,
    fps: 0,
  }

  let minTotal = Infinity
  let maxTotal = -Infinity

  records.forEach(record => {
    const metrics = record.metrics ?? {}
    totals.mount += metrics.mountDurationMs ?? 0
    totals.init += metrics.threeJsInitDurationMs ?? 0
    totals.markers += metrics.markersRenderDurationMs ?? 0
    totals.total += metrics.totalLoadTimeMs ?? 0
    totals.fps += metrics.averageFps ?? 0

    if (typeof metrics.totalLoadTimeMs === 'number') {
      minTotal = Math.min(minTotal, metrics.totalLoadTimeMs)
      maxTotal = Math.max(maxTotal, metrics.totalLoadTimeMs)
    }
  })

  const count = records.length
  const avg = value => (value / count).toFixed(2)

  console.log('📊 Telemetry Summary')
  console.log('────────────────────')
  console.log(`File: ${logFile}`)
  console.log(`Entries analyzed: ${count}`)
  console.log('')
  console.log('Averages (ms):')
  console.log(`  Mount:        ${avg(totals.mount)}`)
  console.log(`  Three.js init:${avg(totals.init)}`)
  console.log(`  Markers:      ${avg(totals.markers)}`)
  console.log(`  Total load:   ${avg(totals.total)} (min ${isFinite(minTotal) ? minTotal.toFixed(2) : 'n/a'} / max ${isFinite(maxTotal) ? maxTotal.toFixed(2) : 'n/a'})`)
  console.log('')
  console.log(`Average FPS:    ${(totals.fps / count).toFixed(1)}`)
  console.log('')

  const recent = records.slice(-3)
  console.log('Last 3 events:')
  recent.forEach(record => {
    const time = record.metrics?.timestamp ?? record.receivedAt ?? 'unknown'
    const load = record.metrics?.totalLoadTimeMs?.toFixed(1) ?? 'n/a'
    const fps = record.metrics?.averageFps?.toFixed(1) ?? 'n/a'
    const route = record.meta?.route ?? 'unknown'
    console.log(`  • ${time} | total ${load} ms | fps ${fps} | route ${route}`)
  })
}

analyze().catch(error => {
  console.error('Failed to analyze telemetry:', error)
  process.exit(1)
})
