import React, { useState } from 'react'
import { useDashboardData } from './data'
import Drivers from './pages/Drivers'
import Models from './pages/Models'
import Monitoring from './pages/Monitoring'
import Overview from './pages/Overview'
import Risk from './pages/Risk'

const TABS = [
  { id: 'overview', label: 'Executive overview', Page: Overview },
  { id: 'risk', label: 'Risk analysis', Page: Risk },
  { id: 'models', label: 'Model performance', Page: Models },
  { id: 'drivers', label: 'Risk drivers', Page: Drivers },
  { id: 'monitoring', label: 'Monitoring', Page: Monitoring },
]

export default function App() {
  const { status, data, error } = useDashboardData()
  const [tab, setTab] = useState('overview')
  const Active = TABS.find((t) => t.id === tab).Page

  return (
    <div className="app">
      <header className="top">
        <div>
          <h1>Claims Pricing Dashboard</h1>
          <p className="muted" style={{ margin: 0 }}>Motor third-party liability: claim frequency, severity and expected claims cost</p>
        </div>
        {data && <span className="pill">Validation split · historical outcomes vs model predictions</span>}
      </header>

      <nav className="tabs" role="tablist" aria-label="Dashboard sections">
        {TABS.map((t) => (
          <button key={t.id} role="tab" aria-selected={tab === t.id} onClick={() => setTab(t.id)}>{t.label}</button>
        ))}
      </nav>

      <main>
        {status === 'loading' && <div className="loading" role="status">Loading model results…</div>}
        {status === 'error' && <div className="error" role="alert">{error}. Run <code>python -m scripts.export_frontend_data</code> to generate the data files.</div>}
        {status === 'ready' && <Active data={data} />}
      </main>

      {data && (
        <footer>
          <div>Source: {data.meta.source}</div>
          <div>{data.meta.note}</div>
          <div>Generated {data.meta.generated_at}</div>
        </footer>
      )}
    </div>
  )
}
