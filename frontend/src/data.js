import { useEffect, useState } from 'react'

export const DATASETS = [
  'meta', 'overview', 'segments', 'risk', 'distribution', 'models', 'calibration', 'explain', 'monitoring',
]

// Loads every exported JSON file once. Static files only: no server, no credentials.
export function useDashboardData(base = import.meta.env.BASE_URL) {
  const [state, setState] = useState({ status: 'loading', data: null, error: null })
  useEffect(() => {
    let cancelled = false
    Promise.all(
      DATASETS.map((name) =>
        fetch(`${base}data/${name}.json`).then((r) => {
          if (!r.ok) throw new Error(`Could not load ${name}.json (HTTP ${r.status})`)
          return r.json()
        }),
      ),
    )
      .then((parts) => {
        if (!cancelled) setState({ status: 'ready', data: Object.fromEntries(DATASETS.map((n, i) => [n, parts[i]])), error: null })
      })
      .catch((e) => {
        if (!cancelled) setState({ status: 'error', data: null, error: e.message })
      })
    return () => {
      cancelled = true
    }
  }, [base])
  return state
}
