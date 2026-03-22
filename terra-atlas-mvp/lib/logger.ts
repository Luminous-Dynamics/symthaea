/* eslint-disable no-console */

type LogPayload = {
  message: string
  context?: Record<string, unknown>
}

function format({ message, context }: LogPayload) {
  return context ? `${message} | ${JSON.stringify(context)}` : message
}

export const logger = {
  info(payload: LogPayload | string) {
    const formatted = typeof payload === 'string' ? payload : format(payload)
    console.log(formatted)
  },
  warn(payload: LogPayload | string) {
    const formatted = typeof payload === 'string' ? payload : format(payload)
    console.warn(formatted)
  },
  error(payload: LogPayload | string, error?: unknown) {
    const base = typeof payload === 'string' ? payload : format(payload)
    if (error) {
      console.error(base, error)
    } else {
      console.error(base)
    }
  },
}

export default logger
