import { NextResponse } from 'next/server'
import { promises as fs } from 'fs'
import path from 'path'
import logger from '@/lib/logger'

const TELEMETRY_FILE = path.join(process.cwd(), 'data', 'telemetry-events.log')

type TelemetryPayload = {
  type: string
  metrics?: Record<string, unknown>
  meta?: Record<string, unknown>
}

const MAX_PAYLOAD_SIZE = 10 * 1024 // 10 KB

async function persist(payload: any) {
  await fs.mkdir(path.dirname(TELEMETRY_FILE), { recursive: true })
  await fs.appendFile(TELEMETRY_FILE, `${JSON.stringify(payload)}\n`, { encoding: 'utf8' })
}

export async function POST(request: Request) {
  try {
    const contentLength = request.headers.get('content-length')
    if (contentLength && Number(contentLength) > MAX_PAYLOAD_SIZE) {
      return NextResponse.json({ error: 'Payload too large' }, { status: 413 })
    }

    const body = (await request.json()) as TelemetryPayload

    if (!body?.type || typeof body.type !== 'string') {
      return NextResponse.json({ error: 'Missing telemetry type' }, { status: 400 })
    }

    const record = {
      type: body.type,
      metrics: body.metrics ?? {},
      meta: body.meta ?? {},
      receivedAt: new Date().toISOString(),
      userAgent: request.headers.get('user-agent'),
      referer: request.headers.get('referer'),
      ip: request.headers.get('x-forwarded-for') || request.headers.get('x-real-ip') || 'unknown',
    }

    await persist(record)

    return NextResponse.json({ ok: true })
  } catch (error) {
    logger.error('Telemetry error', error)
    return NextResponse.json({ error: 'Failed to record telemetry' }, { status: 500 })
  }
}

export async function GET() {
  return NextResponse.json({ message: 'Telemetry endpoint. POST metrics here.' })
}
