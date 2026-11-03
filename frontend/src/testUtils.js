import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { vi } from 'vitest'
import { DATASETS } from './data'

const dataDir = join(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'data')

// The real exported JSON, so tests exercise the same numbers the dashboard shows.
export const loadData = (name) => JSON.parse(readFileSync(join(dataDir, `${name}.json`), 'utf8'))

// Replace fetch with a stub that serves the exported files (404 for anything else).
export function stubFetch() {
  vi.stubGlobal('fetch', vi.fn(async (url) => {
    const name = String(url).split('/').pop().replace('.json', '')
    const ok = DATASETS.includes(name)
    return { ok, status: ok ? 200 : 404, json: async () => loadData(name) }
  }))
}
