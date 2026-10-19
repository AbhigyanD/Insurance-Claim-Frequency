import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { DATASETS } from './data'

const dataDir = join(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'data')
const load = (name) => JSON.parse(readFileSync(join(dataDir, `${name}.json`), 'utf8'))

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn(async (url) => {
    const name = String(url).split('/').pop().replace('.json', '')
    return { ok: DATASETS.includes(name), status: DATASETS.includes(name) ? 200 : 404, json: async () => load(name) }
  }))
})
afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('exported data files', () => {
  it('contain only finite numbers and validation-split aggregates', () => {
    for (const name of DATASETS) {
      const text = JSON.stringify(load(name))
      expect(text).not.toMatch(/NaN|Infinity/)
    }
    expect(load('meta').split).toBe('validation')
    // no policy-level identifiers are exported
    expect(JSON.stringify(load('risk'))).not.toContain('IDpol')
  })
})

describe('App', () => {
  it('shows the overview with real exported figures', async () => {
    render(<App />)
    expect(screen.getByRole('status')).toHaveTextContent(/loading/i)
    await waitFor(() => expect(screen.getByText('Policies evaluated')).toBeInTheDocument())
    expect(screen.getByText('135,598')).toBeInTheDocument()
    expect(screen.getByText(/validation split/i, { selector: 'strong' })).toBeInTheDocument()
    expect(screen.getByText('0.925')).toBeInTheDocument()
    // the cost gap is presented with its interval and not attributed to a cause the data cannot confirm
    expect(screen.getByText(/95% bootstrap interval for actual/i)).toHaveTextContent(/within sampling noise/)
    expect(screen.queryByText(/the gap comes from severity/i)).not.toBeInTheDocument()
  })

  it('switches tabs and shows association-not-causation warning on drivers', async () => {
    render(<App />)
    await waitFor(() => screen.getByText('Policies evaluated'))
    fireEvent.click(screen.getByRole('tab', { name: 'Risk drivers' }))
    expect(screen.getByText(/not causal effects/i)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('tab', { name: 'Monitoring' }))
    expect(screen.getByText(/baseline, not evidence about production drift/i)).toBeInTheDocument()
  })

  it('shows an actionable error when data cannot be loaded', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: false, status: 404, json: async () => ({}) })))
    render(<App />)
    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument())
    expect(screen.getByRole('alert')).toHaveTextContent(/export_frontend_data/)
  })
})
