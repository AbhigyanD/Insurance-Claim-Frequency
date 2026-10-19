import React from 'react'
import { ThresholdBars } from '../components/charts'
import { Badge, Callout, Card, Table } from '../components/ui'
import { fmtNum } from '../format'

const kind = (s) => (s === 'stable' ? 'good' : s === 'moderate' ? 'warn' : 'bad')

export default function Monitoring({ data }) {
  const { psi, note } = data.monitoring
  const features = psi.filter((r) => r.kind === 'feature')
  const preds = psi.filter((r) => r.kind === 'prediction')

  return (
    <>
      <Callout kind="warn">{note}</Callout>
      <div className="grid two">
        <Card title="Input feature drift (PSI)" subtitle="Training vs validation. Below 0.1 is stable, 0.1–0.25 moderate, above 0.25 major.">
          <ThresholdBars data={features} xKey="feature" yKey="psi" yFormat={(v) => fmtNum(v, 3)} thresholds={[{ value: 0.1, label: '0.1' }, { value: 0.25, label: '0.25' }]} height={320} />
        </Card>
        <Card title="Prediction drift (PSI)" subtitle="Distribution of model outputs, training vs validation">
          <ThresholdBars data={preds} xKey="feature" yKey="psi" yFormat={(v) => fmtNum(v, 3)} thresholds={[{ value: 0.1, label: '0.1' }, { value: 0.25, label: '0.25' }]} height={320} />
        </Card>
      </div>
      <Card title="All PSI values">
        <Table
          columns={[
            { key: 'feature', label: 'Variable', align: 'left' },
            { key: 'kind', label: 'Type', align: 'left' },
            { key: 'psi', label: 'PSI', render: (v) => v.toFixed(5) },
            { key: 'status', label: 'Status', render: (v) => <Badge kind={kind(v)}>{v}</Badge> },
          ]}
          rows={psi}
        />
      </Card>
      <Card title="Not yet available">
        <p className="muted">
          Performance over time, and data-quality alerts on incoming policies, need dated outcomes or a live feed. This dataset has no date column,
          so those views are intentionally absent rather than simulated.
        </p>
      </Card>
    </>
  )
}
